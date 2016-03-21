/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_worker.h"

#include <ucp/wireup/address.h>
#include <ucp/tag/eager.h>
#include <ucs/datastruct/mpool.inl>


static void ucp_worker_close_ifaces(ucp_worker_h worker)
{
    ucp_rsc_index_t rsc_index;

    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index] == NULL) {
            continue;
        }

        uct_iface_close(worker->ifaces[rsc_index]);
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

static ucs_status_t ucp_stub_am_handler(void *arg, void *data, size_t length, void *desc)
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
                (void)uct_iface_set_am_handler(worker->ifaces[tl_id], am_id,
                                               ucp_stub_am_handler, worker,
                                               UCT_AM_CB_FLAG_ASYNC);
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

static ucs_status_t ucp_worker_add_iface(ucp_worker_h worker,
                                         ucp_rsc_index_t tl_id)
{
    ucp_context_h context = worker->context;
    ucp_tl_resource_desc_t *resource = &context->tl_rscs[tl_id];
    uct_iface_config_t *iface_config;
    ucs_status_t status;
    uct_iface_h iface;
    uct_iface_attr_t *attr;

    /* Read configuration
     * TODO pass env_prefix from context */
    status = uct_iface_config_read(resource->tl_rsc.tl_name, NULL, NULL,
                                   &iface_config);
    if (status != UCS_OK) {
        goto out;
    }

    /* Open UCT interface */
    status = uct_iface_open(context->pds[resource->pd_index], worker->uct,
                            resource->tl_rsc.tl_name, resource->tl_rsc.dev_name,
                            sizeof(ucp_recv_desc_t), iface_config, &iface);
    uct_config_release(iface_config);

    if (status != UCS_OK) {
        goto out;
    }

    status = uct_iface_query(iface, &worker->iface_attrs[tl_id]);
    if (status != UCS_OK) {
        goto out;
    }

    attr = &worker->iface_attrs[tl_id];

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

    ucs_debug("created interface[%d] using "UCT_TL_RESOURCE_DESC_FMT" on worker %p",
              tl_id, UCT_TL_RESOURCE_DESC_ARG(&resource->tl_rsc), worker);

    worker->ifaces[tl_id] = iface;
    return UCS_OK;

out_close_iface:
    uct_iface_close(iface);
out:
    return status;
}

static void ucp_worker_set_config(ucp_worker_h worker, ucp_rsc_index_t tl_id)
{
    ucp_context_h context        = worker->context;
    uct_iface_attr_t *iface_attr = &worker->iface_attrs[tl_id];
    ucp_ep_config_t *config      = &worker->ep_config[tl_id];
    uct_pd_attr_t *pd_attr       = &context->pd_attrs[context->tl_rscs[tl_id].pd_index];
    double zcopy_thresh;

    memset(config, 0, sizeof(*config));

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
        config->max_eager_short  = iface_attr->cap.am.max_short - sizeof(ucp_eager_hdr_t);
        config->max_am_short     = iface_attr->cap.am.max_short - sizeof(uint64_t);
    }

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_SHORT) {
        config->max_put_short    = iface_attr->cap.put.max_short;
    }

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) {
        config->max_am_bcopy     = iface_attr->cap.am.max_bcopy;
    }

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) {
        config->max_put_bcopy    = iface_attr->cap.put.max_bcopy;
    }

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_BCOPY) {
        config->max_get_bcopy    = iface_attr->cap.get.max_bcopy;
    }

    if ((iface_attr->cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) &&
        (pd_attr->cap.flags & UCT_PD_FLAG_REG))
    {
        config->max_am_zcopy  = iface_attr->cap.am.max_zcopy;
        config->max_put_zcopy = iface_attr->cap.put.max_zcopy;
        config->max_get_zcopy = iface_attr->cap.get.max_zcopy;

        if (context->config.ext.zcopy_thresh == UCS_CONFIG_MEMUNITS_AUTO) {
            /* auto */
            zcopy_thresh = pd_attr->reg_cost.overhead / (
                                    (1.0 / context->config.ext.bcopy_bw) -
                                    (1.0 / iface_attr->bandwidth) -
                                    pd_attr->reg_cost.growth);
            if (zcopy_thresh < 0) {
                config->zcopy_thresh      = SIZE_MAX;
                config->sync_zcopy_thresh = -1;
            } else {
                config->zcopy_thresh = zcopy_thresh;
                config->sync_zcopy_thresh = zcopy_thresh;
            }
        } else {
            config->zcopy_thresh = context->config.ext.zcopy_thresh;
            config->sync_zcopy_thresh = context->config.ext.zcopy_thresh;
        }
    } else {
        config->zcopy_thresh      = SIZE_MAX;
        config->sync_zcopy_thresh = -1;
    }

    config->bcopy_thresh     = context->config.ext.bcopy_thresh;
    config->rndv_thresh      = SIZE_MAX;
    config->sync_rndv_thresh = SIZE_MAX;
}

static void ucp_worker_set_stub_config(ucp_worker_h worker)
{
    ucp_context_h context        = worker->context;
    ucp_ep_config_t *config      = &worker->ep_config[context->num_tls];

    memset(config, 0, sizeof(*config));

    config->max_am_bcopy      = 256;
    config->zcopy_thresh      = SIZE_MAX;
    config->sync_zcopy_thresh = SIZE_MAX;
    config->rndv_thresh       = SIZE_MAX;
    config->sync_rndv_thresh  = SIZE_MAX;
}

ucs_status_t ucp_worker_create(ucp_context_h context, ucs_thread_mode_t thread_mode,
                               ucp_worker_h *worker_p)
{
    ucp_rsc_index_t tl_id;
    ucp_worker_h worker;
    ucs_status_t status;
#if ENABLE_DEBUG_DATA
    unsigned name_length;
#endif

    worker = ucs_calloc(1, sizeof(*worker) +
                           sizeof(*worker->ep_config) * (context->num_tls + 1),
                        "ucp worker");
    if (worker == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    worker->context       = context;
    worker->uuid          = ucs_generate_uuid((uintptr_t)worker);
    worker->stub_pend_count = 0;
#if ENABLE_ASSERT
    worker->inprogress    = 0;
#endif
#if ENABLE_DEBUG_DATA
    name_length = ucs_min(UCP_WORKER_NAME_MAX,
                          context->config.ext.max_worker_name + 1);
    ucs_snprintf_zero(worker->name, name_length, "%s:%d", ucs_get_host_name(),
                      getpid());
#endif

    worker->ep_hash = ucs_malloc(sizeof(*worker->ep_hash) * UCP_WORKER_EP_HASH_SIZE,
                                 "ucp_ep_hash");
    if (worker->ep_hash == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free;
    }

    sglib_hashed_ucp_ep_t_init(worker->ep_hash);

    worker->ifaces = ucs_calloc(context->num_tls, sizeof(*worker->ifaces),
                                "ucp iface");
    if (worker->ifaces == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_ep_hash;
    }

    worker->iface_attrs = ucs_calloc(context->num_tls,
                                     sizeof(*worker->iface_attrs),
                                     "ucp iface_attr");
    if (worker->iface_attrs == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_ifaces;
    }

    status = ucs_async_context_init(&worker->async, UCS_ASYNC_MODE_THREAD);
    if (status != UCS_OK) {
        goto err_free_attrs;
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

    /* Open all resources as interfaces on this worker */
    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        status = ucp_worker_add_iface(worker, tl_id);
        if (status != UCS_OK) {
            goto err_close_ifaces;
        }

        ucp_worker_set_config(worker, tl_id);
    }

    ucp_worker_set_stub_config(worker);

    *worker_p = worker;
    return UCS_OK;

err_close_ifaces:
    ucp_worker_close_ifaces(worker);
    ucs_mpool_cleanup(&worker->req_mp, 1);
err_destroy_uct_worker:
    uct_worker_destroy(worker->uct);
err_destroy_async:
    ucs_async_context_cleanup(&worker->async);
err_free_attrs:
    ucs_free(worker->iface_attrs);
err_free_ifaces:
    ucs_free(worker->ifaces);
err_free_ep_hash:
    ucs_free(worker->ep_hash);
err_free:
    ucs_free(worker);
err:
    return status;
}

static void ucp_worker_destroy_eps(ucp_worker_h worker)
{
    struct sglib_hashed_ucp_ep_t_iterator iter;
    ucp_ep_h ep;

    ucs_debug("worker %p: destroy all endpoints", worker);
    for (ep = sglib_hashed_ucp_ep_t_it_init(&iter, worker->ep_hash); ep != NULL;
         ep = sglib_hashed_ucp_ep_t_it_next(&iter))
    {
        ucp_ep_destroy(ep);
    }
}

void ucp_worker_destroy(ucp_worker_h worker)
{
    ucs_trace_func("worker=%p", worker);
    ucp_worker_remove_am_handlers(worker);
    ucp_worker_destroy_eps(worker);
    ucp_worker_close_ifaces(worker);
    ucs_mpool_cleanup(&worker->req_mp, 1);
    uct_worker_destroy(worker->uct);
    ucs_async_context_cleanup(&worker->async);
    ucs_free(worker->iface_attrs);
    ucs_free(worker->ifaces);
    ucs_free(worker->ep_hash);
    ucs_free(worker);
}

void ucp_worker_progress(ucp_worker_h worker)
{
    /* worker->inprogress is used only for assertion check.
     * coverity[assert_side_effect]
     */
    ucs_assert(worker->inprogress++ == 0);
    uct_worker_progress(worker->uct);
    ucs_async_check_miss(&worker->async);

    /* coverity[assert_side_effect] */
    ucs_assert(--worker->inprogress == 0);
}

ucs_status_t ucp_worker_get_address(ucp_worker_h worker, ucp_address_t **address_p,
                                    size_t *address_length_p)
{
    return ucp_address_pack(worker, NULL, -1, NULL, address_length_p,
                            (void**)address_p);
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
        status = ucp_ep_new(worker, dest_uuid, "??", "for-sending-reply", &ep);
        if (status != UCS_OK) {
            goto err;
        }

        status = ucp_wireup_create_stub_ep(ep);
        if (status != UCS_OK) {
            ucp_ep_delete(ep);
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
    ucp_ep_h ep;

    req = ucs_mpool_get_inline(&worker->req_mp);
    if (req == NULL) {
        ucs_fatal("could not allocate request");
    }

    ep = ucp_worker_get_reply_ep(worker, dest_uuid);
    ucp_send_req_init(req, ep);
    return req;
}

SGLIB_DEFINE_LIST_FUNCTIONS(ucp_ep_t, ucp_worker_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(ucp_ep_t, UCP_WORKER_EP_HASH_SIZE,
                                        ucp_worker_ep_hash);

static void ucp_worker_print_config(FILE *stream, const char * const *names,
                                    const size_t *values, unsigned count,
                                    const char *rel)
{
    char buf[256];
    unsigned i;

    fprintf(stream, "#   ");
    for (i = 0; i < count; ++i) {
        if (values[i] == SIZE_MAX) {
            strcpy(buf, "(inf)");
        } else {
            snprintf(buf, sizeof(buf), "%zu", values[i]);
        }
        fprintf(stream, " %10s %s %-10s", names[i], rel, buf);
    }
    fprintf(stream, "\n");
}

void ucp_worker_proto_print(ucp_worker_h worker, FILE *stream, const char *title,
                            ucs_config_print_flags_t print_flags)
{
    ucp_context_h context = worker->context;
    ucp_ep_config_t *config;
    ucp_rsc_index_t tl_id;
    char rsc_name[UCT_TL_NAME_MAX + UCT_DEVICE_NAME_MAX + 2];
    ucp_address_t *address;
    size_t address_length;
    ucs_status_t status;

    if (print_flags & UCS_CONFIG_PRINT_HEADER) {
        fprintf(stream, "#\n");
        fprintf(stream, "# %s\n", title);
        fprintf(stream, "#\n");
    }

    fprintf(stream, "# Name:           `%s'\n", ucp_worker_get_name(worker));

    status = ucp_worker_get_address(worker, &address, &address_length);
    if (status == UCS_OK) {
        ucp_worker_release_address(worker, address);
        fprintf(stream, "# Address length: %zu bytes\n", address_length);
    } else {
        fprintf(stream, "# <failed to get address>\n");
    }

    fprintf(stream, "#\n");

    fprintf(stream, "# Transports: \n");
    fprintf(stream, "#\n");

    for (tl_id = 0; tl_id < worker->context->num_tls; ++tl_id) {

        snprintf(rsc_name, sizeof(rsc_name), UCT_TL_RESOURCE_DESC_FMT,
                 UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[tl_id].tl_rsc));

        fprintf(stream, "# %3d %-18s\n", tl_id, rsc_name);
        fprintf(stream, "#\n");

        config = &worker->ep_config[tl_id];
        {
            const char *names[] = {"egr_short", "put_short", "am_short"};
            size_t     values[] = {config->max_eager_short, config->max_put_short,
                                   config->max_am_short};
            ucp_worker_print_config(stream, names, values, 3, "<=");
        }

        {
            const char *names[] = {"am_bcopy", "put_bcopy", "get_bcopy"};
            size_t     values[] = {config->max_am_bcopy, config->max_put_bcopy,
                                   config->max_get_bcopy};
            ucp_worker_print_config(stream, names, values, 3, "<=");
        }

        {
            const char *names[] = {"am_zcopy", "put_zcopy", "get_zcopy"};
            size_t     values[] = {config->max_am_zcopy, config->max_put_zcopy,
                                   config->max_get_zcopy};
            ucp_worker_print_config(stream, names, values, 3, "<=");
        }

        {
            const char *names[] = {"bcopy", "rndv", "zcopy"};
            size_t     values[] = {config->bcopy_thresh, config->rndv_thresh,
                                   config->zcopy_thresh};
            ucp_worker_print_config(stream, names, values, 3, ">=");
        }

        fprintf(stream, "#\n");
        fprintf(stream, "#\n");
    }
}
