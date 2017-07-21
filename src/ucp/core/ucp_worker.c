/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_worker.h"
#include "ucp_mm.h"
#include "ucp_request.inl"

#include <ucp/wireup/address.h>
#include <ucp/wireup/stub_ep.h>
#include <ucp/tag/eager.h>
#include <ucp/tag/offload.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/queue.h>
#include <ucs/type/cpu_set.h>
#include <ucs/sys/string.h>
#include <sys/poll.h>


#if ENABLE_STATS
static ucs_stats_class_t ucp_worker_stats_class = {
    .name           = "ucp_worker",
    .num_counters   = UCP_WORKER_STAT_LAST,
    .counter_names  = {
        [UCP_WORKER_STAT_TAG_RX_EAGER_MSG]         = "rx_eager_msg",
        [UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG]    = "rx_sync_msg",
        [UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_EXP]   = "rx_eager_chunk_exp",
        [UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP] = "rx_eager_chunk_unexp",
        [UCP_WORKER_STAT_TAG_RX_RNDV_EXP]          = "rx_rndv_rts_exp",
        [UCP_WORKER_STAT_TAG_RX_RNDV_UNEXP]        = "rx_rndv_rts_unexp"
    }
};
#endif


ucs_mpool_ops_t ucp_am_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = ucs_empty_function,
    .obj_cleanup   = ucs_empty_function
};


ucs_mpool_ops_t ucp_reg_mpool_ops = {
    .chunk_alloc   = ucp_mpool_malloc,
    .chunk_release = ucp_mpool_free,
    .obj_init      = ucp_mpool_obj_init,
    .obj_cleanup   = ucs_empty_function
};


static void ucp_worker_close_ifaces(ucp_worker_h worker)
{
    ucp_rsc_index_t rsc_index;

    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index].iface == NULL) {
            continue;
        }

        if (ucp_worker_is_tl_tag_offload(worker, rsc_index)) {
            ucs_queue_remove(&worker->context->tm.offload.ifaces,
                             &worker->ifaces[rsc_index].queue);
            ucp_context_tag_offload_enable(worker->context);
        }

        uct_iface_close(worker->ifaces[rsc_index].iface);
    }
}

static ucs_status_t ucp_worker_set_am_handlers(ucp_worker_h worker,
                                               uct_iface_h iface,
                                               const uct_iface_attr_t *iface_attr)
{
    ucp_context_h context = worker->context;
    ucs_status_t status;
    unsigned am_id;

    for (am_id = 0; am_id < UCP_AM_ID_LAST; ++am_id) {
        if (!(context->config.features & ucp_am_handlers[am_id].features)) {
            continue;
        }

        if ((ucp_am_handlers[am_id].flags & UCT_AM_CB_FLAG_SYNC) &&
            !(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_CB_SYNC))
        {
            /* Do not register a sync callback on interface which does not
             * support it. The transport selection logic should not use async
             * transports for protocols with sync active message handlers.
             */
            continue;
        }

        status = uct_iface_set_am_handler(iface, am_id, ucp_am_handlers[am_id].cb,
                                          worker,
                                          ucp_am_handlers[am_id].flags);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static ucs_status_t ucp_stub_am_handler(void *arg, void *data, size_t length,
                                        unsigned flags)
{
    ucp_worker_h worker = arg;
    ucs_trace("worker %p: drop message", worker);
    return UCS_OK;
}

static void ucp_worker_remove_am_handlers(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t tl_id;
    unsigned am_id;

    ucs_debug("worker %p: remove active message handlers", worker);
    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        for (am_id = 0; am_id < UCP_AM_ID_LAST; ++am_id) {
            if (context->config.features & ucp_am_handlers[am_id].features) {
                (void)uct_iface_set_am_handler(worker->ifaces[tl_id].iface,
                                               am_id, ucp_stub_am_handler,
                                               worker, UCT_AM_CB_FLAG_ASYNC);
            }
        }
    }
}

static void ucp_worker_am_tracer(void *arg, uct_am_trace_type_t type,
                                 uint8_t id, const void *data, size_t length,
                                 char *buffer, size_t max)
{
    ucp_worker_h worker = arg;
    ucp_am_tracer_t tracer;

    tracer = ucp_am_handlers[id].tracer;
    if (tracer != NULL) {
        tracer(worker, type, id, data, length, buffer, max);
    }
}

static ucs_status_t ucp_worker_wakeup_add_fd(ucp_worker_h worker, int event_fd)
{
    struct epoll_event event = {0};
    int ret;

    if (!(worker->context->config.features & UCP_FEATURE_WAKEUP)) {
        return UCS_OK;
    }

    event.data.fd = event_fd;
    event.events  = EPOLLIN;

    ret = epoll_ctl(worker->epfd, EPOLL_CTL_ADD, event_fd, &event);
    if (ret == -1) {
        ucs_error("epoll_ctl(epfd=%d, ADD, fd=%d) failed: %m", worker->epfd,
                  event_fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t ucp_worker_wakeup_init(ucp_worker_h worker,
                                           ucp_wakeup_event_t events)
{
    ucp_context_h context = worker->context;
    ucs_status_t status;
    int ret, i;

    if (!(context->config.features & UCP_FEATURE_WAKEUP)) {
        worker->epfd           = -1;
        worker->wakeup_pipe[0] = -1;
        worker->wakeup_pipe[1] = -1;
        worker->uct_events     = 0;
        status = UCS_OK;
        goto out;
    }

    worker->epfd = epoll_create(context->num_tls);
    if (worker->epfd == -1) {
        ucs_error("Failed to create epoll file descriptor: %m");
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    ret = pipe(worker->wakeup_pipe);
    if (ret != 0) {
        ucs_error("Failed to create pipe: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_close_epfd;
    }

    for (i = 0; i < 2; ++i) {
        status = ucs_sys_fcntl_modfl(worker->wakeup_pipe[i], O_NONBLOCK, 0);
        if (status != UCS_OK) {
            goto err_pipe_cleanup;
        }
    }

    ucp_worker_wakeup_add_fd(worker, worker->wakeup_pipe[0]);

    worker->uct_events = 0;

    /* FIXME: any TAG flag initializes all types of completion because of
     *        possible issues in RNDV protocol. The optimization may be
     *        implemented with using of separated UCP descriptors or manual
     *        signaling in RNDV and similar cases, see conversation in PR #1277
     */
    if (events & (UCP_WAKEUP_TAG_SEND | UCP_WAKEUP_TAG_RECV)) {
        worker->uct_events |= UCT_EVENT_SEND_COMP | UCT_EVENT_RECV_AM;
    }

    if (events & (UCP_WAKEUP_RMA | UCP_WAKEUP_AMO)) {
        worker->uct_events |= UCT_EVENT_SEND_COMP;
    }

    return UCS_OK;

err_pipe_cleanup:
    close(worker->wakeup_pipe[0]);
    close(worker->wakeup_pipe[1]);
err_close_epfd:
    close(worker->epfd);
out:
    return status;
}

static void ucp_worker_wakeup_cleanup(ucp_worker_h worker)
{
    if (worker->epfd != -1) {
        close(worker->epfd);
    }
    if (worker->wakeup_pipe[0] != -1) {
        close(worker->wakeup_pipe[0]);
    }
    if (worker->wakeup_pipe[1] != -1) {
        close(worker->wakeup_pipe[1]);
    }
}

static void
ucp_worker_iface_error_handler(void *arg, uct_ep_h uct_ep, ucs_status_t status)
{
    ucp_worker_h       worker           = (ucp_worker_h)arg;
    ucp_ep_h           ucp_ep           = NULL;
    uct_ep_h           aux_ep           = NULL;
    uint64_t           dest_uuid UCS_V_UNUSED;
    ucp_ep_h           ucp_ep_iter;
    khiter_t           ucp_ep_errh_iter;
    ucp_err_handler_t  err_handler;
    ucp_lane_index_t   lane, n_lanes, failed_lane;

    /* TODO: need to optimize uct_ep -> ucp_ep lookup */
    kh_foreach(&worker->ep_hash, dest_uuid, ucp_ep_iter, {
        for (lane = 0; lane < ucp_ep_num_lanes(ucp_ep_iter); ++lane) {
            if ((uct_ep == ucp_ep_iter->uct_eps[lane]) ||
                ucp_stub_ep_test_aux(ucp_ep_iter->uct_eps[lane], uct_ep)) {
                ucp_ep = ucp_ep_iter;
                failed_lane = lane;
                goto found_ucp_ep;
            }
        }
    });

    ucs_fatal("No uct_ep_h %p associated with ucp_ep_h on ucp_worker_h %p",
              uct_ep, worker);
    return;

found_ucp_ep:

    /* Purge outstanding */
    uct_ep_pending_purge(ucp_ep_iter->uct_eps[lane], ucp_ep_err_pending_purge,
                         UCS_STATUS_PTR(status));

    /* Destroy all lanes excepting failed one since ucp_ep becomes unusable as well */
    for (lane = 0, n_lanes = ucp_ep_num_lanes(ucp_ep); lane < n_lanes; ++lane) {
        if ((lane != failed_lane) && ucp_ep->uct_eps[lane]) {
            /* TODO flush and purge outstanding operations */
            uct_ep_destroy(ucp_ep->uct_eps[lane]);
            ucp_ep->uct_eps[lane] = NULL;
        }
    }

    /* Move failed lane at 0 index */
    if (failed_lane != 0) {
        ucp_ep->uct_eps[0] = ucp_ep->uct_eps[failed_lane];
        ucp_ep->uct_eps[failed_lane] = NULL;
    }

    /* NOTE: if failed ep is stub auxiliary then we need to replace the lane
     *       with failed ep and destroy stub ep
     */
    if (ucp_stub_ep_test_aux(ucp_ep_iter->uct_eps[0], uct_ep)) {
        aux_ep = ucp_stub_ep_extract_aux(ucp_ep_iter->uct_eps[0]);
        uct_ep_destroy(ucp_ep_iter->uct_eps[0]);
        ucp_ep_iter->uct_eps[0] = aux_ep;
    }

    /* Redirect all lanes to failed one */
    ucp_ep_config_key_t key = ucp_ep_config(ucp_ep)->key;
    key.am_lane            = 0;
    key.wireup_lane        = 0;
    key.rndv_lane          = 0;
    key.amo_lanes[0]       = 0;
    key.rma_lanes[0]       = 0;
    key.lanes[0].rsc_index = UCP_NULL_RESOURCE;
    key.num_lanes          = 1;
    key.status             = status;

    ucp_ep->cfg_index = ucp_worker_get_ep_config(worker, &key);
    ucp_ep->flags    |= UCP_EP_FLAG_FAILED;
    ucp_ep->am_lane   = 0;

    ucp_ep_errh_iter = kh_get(ucp_ep_errh_hash, &worker->ep_errh_hash,
                              (uintptr_t)ucp_ep);
    if (ucp_ep_errh_iter != kh_end(&worker->ep_errh_hash)) {
        err_handler = kh_val(&worker->ep_errh_hash, ucp_ep_errh_iter);
        err_handler.cb(err_handler.arg, ucp_ep, status);
    } else {
        ucs_error("Error %s was not handled for ep %p",
                  ucs_status_string(status), ucp_ep);
    }
}

static ucs_status_t
ucp_worker_add_iface(ucp_worker_h worker, ucp_rsc_index_t tl_id,
                     size_t rx_headroom, ucs_cpu_set_t const * cpu_mask_param)
{
    ucp_context_h context = worker->context;
    ucp_tl_resource_desc_t *resource = &context->tl_rscs[tl_id];
    uct_iface_config_t *iface_config;
    ucs_status_t status;
    uct_iface_h iface;
    uct_iface_attr_t *attr;
    uct_iface_params_t iface_params;
    int tl_event_fd;

    /* Read configuration
     * TODO pass env_prefix from context */
    status = uct_iface_config_read(resource->tl_rsc.tl_name, NULL, NULL,
                                   &iface_config);
    if (status != UCS_OK) {
        goto out;
    }

    memset(&iface_params, 0, sizeof(iface_params));
    iface_params.tl_name         = resource->tl_rsc.tl_name;
    iface_params.dev_name        = resource->tl_rsc.dev_name;
    iface_params.stats_root      = UCS_STATS_RVAL(worker->stats);
    iface_params.rx_headroom     = rx_headroom;
    iface_params.cpu_mask        = *cpu_mask_param;
    iface_params.err_handler_arg = worker;
    iface_params.err_handler     = ucp_worker_iface_error_handler;
    iface_params.eager_arg       = iface_params.rndv_arg = worker;
    iface_params.eager_cb        = ucp_tag_offload_unexp_eager;
    iface_params.rndv_cb         = ucp_tag_offload_unexp_rndv;

    /* Open UCT interface */
    status = uct_iface_open(context->tl_mds[resource->md_index].md, worker->uct,
                            &iface_params, iface_config, &iface);
    uct_config_release(iface_config);

    if (status != UCS_OK) {
        goto out;
    }

    worker->ifaces[tl_id].worker         = worker;
    worker->ifaces[tl_id].proxy_am_count = 0;

    VALGRIND_MAKE_MEM_UNDEFINED(&worker->ifaces[tl_id].attr,
                                sizeof(worker->ifaces[tl_id].attr));
    status = uct_iface_query(iface, &worker->ifaces[tl_id].attr);
    if (status != UCS_OK) {
        goto out;
    }

    attr = &worker->ifaces[tl_id].attr;

    /* Set active message handlers for tag matching */
    if ((attr->cap.flags & (UCT_IFACE_FLAG_AM_SHORT|UCT_IFACE_FLAG_AM_BCOPY|UCT_IFACE_FLAG_AM_ZCOPY))) {
        status = ucp_worker_set_am_handlers(worker, iface, attr);
        if (status != UCS_OK) {
            goto out_close_iface;
        }

        status = uct_iface_set_am_tracer(iface, ucp_worker_am_tracer, worker);
        if (status != UCS_OK) {
            goto out_close_iface;
        }
    }

    /* Set wake-up handlers */
    if (attr->cap.flags & UCP_WORKER_UCT_EVENT_CAP_FLAGS) {
        status = uct_iface_event_fd_get(iface, &tl_event_fd);
        if (status != UCS_OK) {
            goto out_close_iface;
        }

        status = ucp_worker_wakeup_add_fd(worker, tl_event_fd);
        if (status != UCS_OK) {
            goto out_close_iface;
        }
    }

    if (ucp_worker_is_tl_tag_offload(worker, tl_id)) {
        worker->ifaces[tl_id].rsc_index = tl_id;
        ucs_queue_push(&context->tm.offload.ifaces, &worker->ifaces[tl_id].queue);
        ucp_context_tag_offload_enable(context);
    }

    ucs_debug("created interface[%d] using "UCT_TL_RESOURCE_DESC_FMT" on worker %p",
              tl_id, UCT_TL_RESOURCE_DESC_ARG(&resource->tl_rsc), worker);

    worker->ifaces[tl_id].iface = iface;
    return UCS_OK;

out_close_iface:
    uct_iface_close(iface);
out:
    /* coverity[leaked_storage] */
    return status;
}

static void ucp_worker_enable_atomic_tl(ucp_worker_h worker, const char *mode,
                                        ucp_rsc_index_t rsc_index)
{
    ucs_assert(rsc_index != UCP_NULL_RESOURCE);
    ucs_trace("worker %p: using %s atomics on iface[%d]=" UCT_TL_RESOURCE_DESC_FMT,
              worker, mode, rsc_index,
              UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[rsc_index].tl_rsc));
    worker->atomic_tls |= UCS_BIT(rsc_index);
}

static void ucp_worker_init_cpu_atomics(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t rsc_index;

    /* Enable all interfaces which have host-based atomics */
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index].attr.cap.flags & UCT_IFACE_FLAG_ATOMIC_CPU) {
            ucp_worker_enable_atomic_tl(worker, "cpu", rsc_index);
        }
    }
}

static void ucp_worker_init_device_atomics(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_address_iface_attr_t dummy_iface_attr;
    ucp_tl_resource_desc_t *rsc, *best_rsc;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t rsc_index;
    uint64_t iface_cap_flags;
    double score, best_score;
    ucp_rsc_index_t md_index;
    uct_md_attr_t *md_attr;
    uint64_t supp_tls;
    uint8_t priority, best_priority;

    iface_cap_flags = ucp_context_uct_atomic_iface_flags(context) |
                      UCT_IFACE_FLAG_ATOMIC_DEVICE;

    dummy_iface_attr.bandwidth  = 1e12;
    dummy_iface_attr.cap_flags  = -1;
    dummy_iface_attr.overhead   = 0;
    dummy_iface_attr.priority   = 0;
    dummy_iface_attr.lat_ovh    = 0;

    supp_tls                    = 0;
    best_score                  = -1;
    best_rsc                    = NULL;
    best_priority               = 0;

    /* Select best interface for atomics device */
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        rsc        = &context->tl_rscs[rsc_index];
        md_index   = rsc->md_index;
        md_attr    = &context->tl_mds[md_index].attr;
        iface_attr = &worker->ifaces[rsc_index].attr;

        if (!(md_attr->cap.flags & UCT_MD_FLAG_REG) ||
            !ucs_test_all_flags(iface_attr->cap.flags, iface_cap_flags))
        {
            continue;
        }

        supp_tls |= UCS_BIT(rsc_index);
        priority  = iface_attr->priority;

        score = ucp_wireup_amo_score_func(context, md_attr, iface_attr,
                                          &dummy_iface_attr);
        if ((score > best_score) ||
            ((score == best_score) && (priority > best_priority)))
        {
            best_rsc      = rsc;
            best_score    = score;
            best_priority = priority;
        }
    }

    if (best_rsc == NULL) {
        ucs_debug("worker %p: no support for atomics", worker);
        return;
    }

    /* Enable atomics on all resources using same device as the "best" resource */
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        rsc = &context->tl_rscs[rsc_index];
        if ((supp_tls & UCS_BIT(rsc_index)) &&
            (rsc->md_index == best_rsc->md_index) &&
            !strncmp(rsc->tl_rsc.dev_name, best_rsc->tl_rsc.dev_name,
                     UCT_DEVICE_NAME_MAX))
        {
            ucp_worker_enable_atomic_tl(worker, "device", rsc_index);
        }
    }
}

static void ucp_worker_init_guess_atomics(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t rsc_index;
    uint64_t accumulated_flags = 0;

    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        accumulated_flags |= worker->ifaces[rsc_index].attr.cap.flags;
    }

    if (accumulated_flags & UCT_IFACE_FLAG_ATOMIC_DEVICE) {
	ucp_worker_init_device_atomics(worker);
    } else {
	ucp_worker_init_cpu_atomics(worker);
    }
}

static void ucp_worker_init_atomic_tls(ucp_worker_h worker)
{
    ucp_context_h context = worker->context;

    worker->atomic_tls = 0;

    if (context->config.features & (UCP_FEATURE_AMO32|UCP_FEATURE_AMO64)) {
        switch(context->config.ext.atomic_mode) {
        case UCP_ATOMIC_MODE_CPU:
            ucp_worker_init_cpu_atomics(worker);
            break;
        case UCP_ATOMIC_MODE_DEVICE:
            ucp_worker_init_device_atomics(worker);
            break;
        case UCP_ATOMIC_MODE_GUESS:
            ucp_worker_init_guess_atomics(worker);
            break;
        default:
            ucs_fatal("unsupported atomic mode: %d",
                      context->config.ext.atomic_mode);
        }
    }
}

static ucs_status_t ucp_worker_init_mpools(ucp_worker_h worker,
                                           size_t rx_headroom)
{
    size_t           max_mp_entry_size = 0;
    ucp_context_t    *context          = worker->context;
    uct_iface_attr_t *if_attr;
    size_t           tl_id;
    ucs_status_t     status;

    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        if_attr = &worker->ifaces[tl_id].attr;
        max_mp_entry_size = ucs_max(max_mp_entry_size,
                                    if_attr->cap.am.max_short);
        max_mp_entry_size = ucs_max(max_mp_entry_size,
                                    if_attr->cap.am.max_bcopy);
        max_mp_entry_size = ucs_max(max_mp_entry_size,
                                    if_attr->cap.am.max_zcopy);
    }

    status = ucs_mpool_init(&worker->am_mp, 0,
                            max_mp_entry_size + rx_headroom,
                            0, UCS_SYS_CACHE_LINE_SIZE, 128, UINT_MAX,
                            &ucp_am_mpool_ops, "ucp_am_bufs");
    if (status != UCS_OK) {
        goto out;
    }

    status = ucs_mpool_init(&worker->reg_mp, 0,
                            context->config.ext.seg_size + sizeof(ucp_mem_desc_t),
                            sizeof(ucp_mem_desc_t), UCS_SYS_CACHE_LINE_SIZE,
                            128, UINT_MAX, &ucp_reg_mpool_ops, "ucp_reg_bufs");
    if (status != UCS_OK) {
        goto err_release_am_mpool;
    }

    return UCS_OK;

err_release_am_mpool:
    ucs_mpool_cleanup(&worker->am_mp, 0);
out:
    return status;
}

/* All the ucp endpoints will share the configurations. No need for every ep to
 * have it's own configuration (to save memory footprint). Same config can be used
 * by different eps.
 * A 'key' identifies an entry in the ep_config array. An entry holds the key and
 * additional configuration parameters and thresholds.
 */
unsigned ucp_worker_get_ep_config(ucp_worker_h worker,
                                  const ucp_ep_config_key_t *key)
{
    ucp_ep_config_t *config;
    unsigned config_idx;

    /* Search for the given key in the ep_config array */
    for (config_idx = 0; config_idx < worker->ep_config_count; ++config_idx) {
        if (ucp_ep_config_is_equal(&worker->ep_config[config_idx].key, key)) {
            goto out;
        }
    }

    if (worker->ep_config_count >= worker->ep_config_max) {
        /* TODO support larger number of configurations */
        ucs_fatal("too many ep configurations: %d", worker->ep_config_count);
    }

    /* Create new configuration */
    config_idx = worker->ep_config_count++;
    config     = &worker->ep_config[config_idx];

    memset(config, 0, sizeof(*config));
    config->key = *key;
    ucp_ep_config_init(worker, config);

out:
    return config_idx;
}

ucs_status_t ucp_worker_create(ucp_context_h context,
                               const ucp_worker_params_t *params,
                               ucp_worker_h *worker_p)
{
    ucp_rsc_index_t tl_id;
    ucp_worker_h worker;
    ucs_status_t status;
    unsigned config_count;
    unsigned name_length;
    ucs_cpu_set_t empty_cpu_mask;
    ucs_thread_mode_t thread_mode;
    ucp_wakeup_event_t events;
    const ucs_cpu_set_t *cpu_mask;

    /* Space for eager header is needed for unexpected tag offload messages */
    const size_t rx_headroom = sizeof(ucp_recv_desc_t) + sizeof(ucp_eager_sync_hdr_t);

    config_count = ucs_min((context->num_tls + 1) * (context->num_tls + 1) * context->num_tls,
                           UINT8_MAX);

    worker = ucs_calloc(1, sizeof(*worker) +
                           sizeof(*worker->ep_config) * config_count,
                        "ucp worker");
    if (worker == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_THREAD_MODE) {
        thread_mode = params->thread_mode;
    } else {
        thread_mode = UCS_THREAD_MODE_SINGLE;
    }

    if (thread_mode != UCS_THREAD_MODE_MULTI) {
        worker->mt_lock.mt_type = UCP_MT_TYPE_NONE;
    } else if (context->config.ext.use_mt_mutex) {
        worker->mt_lock.mt_type = UCP_MT_TYPE_MUTEX;
    } else {
        worker->mt_lock.mt_type = UCP_MT_TYPE_SPINLOCK;
    }

    UCP_THREAD_LOCK_INIT(&worker->mt_lock);

    worker->context         = context;
    worker->uuid            = ucs_generate_uuid((uintptr_t)worker);
    worker->stub_pend_count = 0;
    worker->inprogress      = 0;
    worker->ep_config_max   = config_count;
    worker->ep_config_count = 0;

    name_length = ucs_min(UCP_WORKER_NAME_MAX,
                          context->config.ext.max_worker_name + 1);
    ucs_snprintf_zero(worker->name, name_length, "%s:%d", ucs_get_host_name(),
                      getpid());

    kh_init_inplace(ucp_worker_ep_hash, &worker->ep_hash);
    kh_init_inplace(ucp_ep_errh_hash,   &worker->ep_errh_hash);

    worker->ifaces = ucs_calloc(context->num_tls, sizeof(ucp_worker_iface_t),
                                "ucp iface");
    if (worker->ifaces == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free;
    }

    /* Create statistics */
    status = UCS_STATS_NODE_ALLOC(&worker->stats, &ucp_worker_stats_class,
                                  ucs_stats_get_root(), "-%p", worker);
    if (status != UCS_OK) {
        goto err_free_ifaces;
    }

    status = ucs_async_context_init(&worker->async, UCS_ASYNC_MODE_THREAD);
    if (status != UCS_OK) {
        goto err_free_stats;
    }

    /* Create the underlying UCT worker */
    status = uct_worker_create(&worker->async, thread_mode, &worker->uct);
    if (status != UCS_OK) {
        goto err_destroy_async;
    }

    /* Create memory pool for requests */
    status = ucs_mpool_init(&worker->req_mp, 0,
                            sizeof(ucp_request_t) + context->config.request.size,
                            0, UCS_SYS_CACHE_LINE_SIZE, 128, UINT_MAX,
                            &ucp_request_mpool_ops, "ucp_requests");
    if (status != UCS_OK) {
        goto err_destroy_uct_worker;
    }

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_EVENTS) {
        events = params->events;
    } else {
        events = UCP_WAKEUP_RMA | UCP_WAKEUP_AMO | UCP_WAKEUP_TAG_SEND |
                 UCP_WAKEUP_TAG_RECV;
    }

    /* Create epoll set which combines events from all transports */
    status = ucp_worker_wakeup_init(worker, events);
    if (status != UCS_OK) {
        goto err_req_mp_cleanup;
    }

    if (params->field_mask & UCP_WORKER_PARAM_FIELD_CPU_MASK) {
        cpu_mask = &params->cpu_mask;
    } else {
        UCS_CPU_ZERO(&empty_cpu_mask);
        cpu_mask = &empty_cpu_mask;
    }

    /* Open all resources as interfaces on this worker */
    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        status = ucp_worker_add_iface(worker, tl_id, rx_headroom, cpu_mask);
        if (status != UCS_OK) {
            goto err_close_ifaces;
        }
    }

    /* Init AM and registered memory pools */
    status = ucp_worker_init_mpools(worker, rx_headroom);
    if (status != UCS_OK) {
        goto err_close_ifaces;
    }

    /* Select atomic resources */
    ucp_worker_init_atomic_tls(worker);

    *worker_p = worker;
    return UCS_OK;

err_close_ifaces:
    ucp_worker_close_ifaces(worker);
    ucp_worker_wakeup_cleanup(worker);
err_req_mp_cleanup:
    ucs_mpool_cleanup(&worker->req_mp, 1);
err_destroy_uct_worker:
    uct_worker_destroy(worker->uct);
err_destroy_async:
    ucs_async_context_cleanup(&worker->async);
err_free_stats:
    UCS_STATS_NODE_FREE(worker->stats);
err_free_ifaces:
    ucs_free(worker->ifaces);
err_free:
    UCP_THREAD_LOCK_FINALIZE(&worker->mt_lock);
    ucs_free(worker);
    return status;
}

static void ucp_worker_destroy_eps(ucp_worker_h worker)
{
    ucp_ep_h ep;

    ucs_debug("worker %p: destroy all endpoints", worker);
    kh_foreach_value(&worker->ep_hash, ep,
                     ucp_ep_destroy_internal(ep, " from worker destroy"));
}

void ucp_worker_destroy(ucp_worker_h worker)
{
    ucs_trace_func("worker=%p", worker);
    ucp_worker_remove_am_handlers(worker);
    ucp_worker_destroy_eps(worker);
    ucs_mpool_cleanup(&worker->am_mp, 1);
    ucs_mpool_cleanup(&worker->reg_mp, 1);
    ucp_worker_close_ifaces(worker);
    ucp_worker_wakeup_cleanup(worker);
    ucs_mpool_cleanup(&worker->req_mp, 1);
    uct_worker_destroy(worker->uct);
    ucs_async_context_cleanup(&worker->async);
    ucs_free(worker->ifaces);
    kh_destroy_inplace(ucp_worker_ep_hash, &worker->ep_hash);
    kh_destroy_inplace(ucp_ep_errh_hash, &worker->ep_errh_hash);
    UCP_THREAD_LOCK_FINALIZE(&worker->mt_lock);
    UCS_STATS_NODE_FREE(worker->stats);
    ucs_free(worker);
}

ucs_status_t ucp_worker_query(ucp_worker_h worker,
                              ucp_worker_attr_t *attr)
{
    if (attr->field_mask & UCP_WORKER_ATTR_FIELD_THREAD_MODE) {
        if (UCP_THREAD_IS_REQUIRED(&worker->mt_lock)) {
            attr->thread_mode = UCS_THREAD_MODE_MULTI;
        } else {
            attr->thread_mode = UCS_THREAD_MODE_SINGLE;
        }
    }

    return UCS_OK;
}

void ucp_worker_progress(ucp_worker_h worker)
{
    /* worker->inprogress is used only for assertion check.
     * coverity[assert_side_effect]
     */
    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    ucs_assert(worker->inprogress++ == 0);
    uct_worker_progress(worker->uct);
    ucs_async_check_miss(&worker->async);

    /* coverity[assert_side_effect] */
    ucs_assert(--worker->inprogress == 0);

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
}

ucs_status_t ucp_worker_get_efd(ucp_worker_h worker, int *fd)
{
    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
    *fd = worker->epfd;
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return UCS_OK;
}

ucs_status_t ucp_worker_arm(ucp_worker_h worker)
{
    int res;
    char buf;
    ucs_status_t status;
    ucp_rsc_index_t tl_id;
    ucp_context_h context = worker->context;

    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        if (worker->ifaces[tl_id].attr.cap.flags & UCP_WORKER_UCT_EVENT_CAP_FLAGS) {
            status = uct_iface_event_arm(worker->ifaces[tl_id].iface, worker->uct_events);
            if (status != UCS_OK) {
                return status;
            }
        }
    }

    do {
        res = read(worker->wakeup_pipe[0], &buf, 1);
    } while (res != -1);
    if (errno != EAGAIN && errno != EINTR) {
        ucs_error("Read from internal pipe failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

void ucp_worker_wait_mem(ucp_worker_h worker, void *address)
{
   ucs_arch_wait_mem(address);
}

ucs_status_t ucp_worker_wait(ucp_worker_h worker)
{
    int res;
    int epoll_fd = -1;
    ucs_status_t status;
    struct epoll_event *events;
    ucp_context_h context = worker->context;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    status = ucp_worker_get_efd(worker, &epoll_fd);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucp_worker_arm(worker);
    if (UCS_ERR_BUSY == status) { /* if UCS_ERR_BUSY returned - no poll() must called */
        status = UCS_OK;
        goto out;
    } else if (status != UCS_OK) {
        goto out;
    }

    events = ucs_malloc(context->num_tls * sizeof(*events), "wakeup events");
    if (events == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    do {
        ucs_debug("epoll_wait loop with epfd %d maxevents %d timeout %d",
                   epoll_fd, context->num_tls, -1);
        res = epoll_wait(epoll_fd, events, context->num_tls, -1);
    } while ((res == -1) && (errno == EINTR));

    free(events);

    if (res == -1) {
        ucs_error("Polling internally for events failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    status = UCS_OK;
out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return status;
}

ucs_status_t ucp_worker_signal(ucp_worker_h worker)
{
    char buf = 0;
    int res;
    ucs_status_t status = UCS_OK;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    res = write(worker->wakeup_pipe[1], &buf, 1);
    if ((res != 1)  && (errno != EAGAIN)) {
        ucs_error("Signaling wakeup failed: %m");
        status = UCS_ERR_IO_ERROR;
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    return status;
}

ucs_status_t ucp_worker_get_address(ucp_worker_h worker, ucp_address_t **address_p,
                                    size_t *address_length_p)
{
    ucs_status_t status;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    status = ucp_address_pack(worker, NULL, -1, NULL, address_length_p,
                              (void**)address_p);

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);

    return status;
}

void ucp_worker_release_address(ucp_worker_h worker, ucp_address_t *address)
{
    ucs_free(address);
}

ucp_ep_h ucp_worker_get_reply_ep(ucp_worker_h worker, uint64_t dest_uuid)
{
    ucs_status_t status;
    ucp_ep_h ep;

    UCS_ASYNC_BLOCK(&worker->async);

    ep = ucp_worker_ep_find(worker, dest_uuid);
    if (ep == NULL) {
        status = ucp_ep_create_stub(worker, dest_uuid, "for-sending-reply", &ep);
        if (status != UCS_OK) {
            goto err;
        }
    } else {
        ucs_debug("found ep %p", ep);
    }

    UCS_ASYNC_UNBLOCK(&worker->async);
    return ep;

err:
    UCS_ASYNC_UNBLOCK(&worker->async);
    ucs_fatal("failed to create reply endpoint: %s", ucs_status_string(status));
}

ucp_request_t *ucp_worker_allocate_reply(ucp_worker_h worker, uint64_t dest_uuid)
{
    ucp_request_t *req;

    req = ucp_request_get(worker);
    if (req == NULL) {
        ucs_fatal("could not allocate request");
    }

    req->flags   = 0;
    req->send.ep = ucp_worker_get_reply_ep(worker, dest_uuid);
    return req;
}

void ucp_worker_print_info(ucp_worker_h worker, FILE *stream)
{
    ucp_context_h context = worker->context;
    ucp_address_t *address;
    size_t address_length;
    ucs_status_t status;
    ucp_rsc_index_t rsc_index;
    int first;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    fprintf(stream, "#\n");
    fprintf(stream, "# UCP worker '%s'\n", ucp_worker_get_name(worker));
    fprintf(stream, "#\n");

    status = ucp_worker_get_address(worker, &address, &address_length);
    if (status == UCS_OK) {
        ucp_worker_release_address(worker, address);
        fprintf(stream, "#                 address: %zu bytes\n", address_length);
    } else {
        fprintf(stream, "# <failed to get address>\n");
    }

    if (context->config.features & (UCP_FEATURE_AMO32|UCP_FEATURE_AMO64)) {
        fprintf(stream, "#                 atomics: ");
        first = 1;
        for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
            if (worker->atomic_tls & UCS_BIT(rsc_index)) {
                if (!first) {
                    fprintf(stream, ", ");
                }
                fprintf(stream, "%d:"UCT_TL_RESOURCE_DESC_FMT, rsc_index,
                        UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[rsc_index].tl_rsc));
                first = 0;
            }
        }
        fprintf(stream, "\n");
    }

    fprintf(stream, "#\n");

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
}
