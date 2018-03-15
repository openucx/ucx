/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ugni_udt_iface.h"
#include "ugni_udt_ep.h"
#include <uct/ugni/base/ugni_device.h>
#include <uct/ugni/base/ugni_md.h>
#include <poll.h>

#define UCT_UGNI_UDT_TL_NAME "ugni_udt"

static ucs_config_field_t uct_ugni_udt_iface_config_table[] = {
    {"", "ALLOC=huge,thp,mmap,heap", NULL,
    ucs_offsetof(uct_ugni_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    UCT_IFACE_MPOOL_CONFIG_FIELDS("UDT", -1, 0, "udt",
                                  ucs_offsetof(uct_ugni_iface_config_t, mpool),
                                  "\nAttention: Setting this param with value != -1 is a dangerous thing\n"
                                  "and could cause deadlock or performance degradation."),

    {NULL}
};

static ucs_status_t processs_datagram(uct_ugni_udt_iface_t *iface, uct_ugni_udt_desc_t *desc)
{
    ucs_status_t status;
    uct_ugni_udt_header_t *header;
    void *payload;

    header = uct_ugni_udt_get_rheader(desc, iface);
    payload = uct_ugni_udt_get_rpayload(desc, iface);
    uct_iface_trace_am(&iface->super.super, UCT_AM_TRACE_TYPE_RECV,
                       header->am_id, payload, header->length, "RX: AM");
    status = uct_iface_invoke_am(&iface->super.super, header->am_id, payload,
                                 header->length, UCT_CB_PARAM_FLAG_DESC);
    return status;
}

static ucs_status_t recieve_datagram(uct_ugni_udt_iface_t *iface, uint64_t id, uct_ugni_udt_ep_t **ep_out)
{
    uint32_t rem_addr, rem_id;
    gni_post_state_t post_state;
    gni_return_t ugni_rc;
    uct_ugni_udt_ep_t *ep;
    gni_ep_handle_t gni_ep;
    uct_ugni_udt_desc_t *desc;
    uct_ugni_udt_header_t *header;

    ucs_trace_func("iface=%p, id=%lx", iface, id);

    if (UCT_UGNI_UDT_ANY == id) {
        ep = NULL;
        gni_ep = iface->ep_any;
        desc = iface->desc_any;
    } else {
        ep = ucs_derived_of(uct_ugni_iface_lookup_ep(&iface->super, id),
                            uct_ugni_udt_ep_t);
        gni_ep = ep->super.ep;
        desc = ep->posted_desc;
    }

    *ep_out = ep;
    uct_ugni_cdm_lock(&iface->super.cdm);
    ugni_rc = GNI_EpPostDataWaitById(gni_ep, id, -1, &post_state, &rem_addr, &rem_id);
    uct_ugni_cdm_unlock(&iface->super.cdm);
    if (ucs_unlikely(GNI_RC_SUCCESS != ugni_rc)) {
        ucs_error("GNI_EpPostDataWaitById, id=%lu Error status: %s %d",
                  id, gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_IO_ERROR;
    }
    if (GNI_POST_TERMINATED == post_state) {
        return UCS_ERR_CANCELED;
    }

    if (GNI_POST_COMPLETED != post_state) {
        ucs_error("GNI_EpPostDataWaitById gave unexpected response: %u", post_state);
        return UCS_ERR_IO_ERROR;
    }

    if (UCT_UGNI_UDT_ANY != id) {
        --iface->super.outstanding;
    }

    header = uct_ugni_udt_get_rheader(desc, iface);

    ucs_trace("Got datagram id: %lu type: %i len: %i am_id: %i", id, header->type, header->length, header->am_id);

    if (UCT_UGNI_UDT_PAYLOAD != header->type) {
        /* ack message, no data */
        ucs_assert_always(NULL != ep);
        ucs_mpool_put(ep->posted_desc);
        uct_ugni_check_flush(ep->desc_flush_group);
        ep->posted_desc = NULL;
        return UCS_OK;
    }

    return UCS_INPROGRESS;
}

static void *uct_ugni_udt_device_thread(void *arg)
{
    uct_ugni_udt_iface_t *iface = (uct_ugni_udt_iface_t *)arg;
    gni_return_t ugni_rc;
    uint64_t id;

    while (1) {
        pthread_mutex_lock(&iface->device_lock);
        while (iface->events_ready) {
            pthread_cond_wait(&iface->device_condition, &iface->device_lock);
        }
        pthread_mutex_unlock(&iface->device_lock);
        ugni_rc = GNI_PostdataProbeWaitById(uct_ugni_udt_iface_nic_handle(iface),-1,&id);
        if (ucs_unlikely(GNI_RC_SUCCESS != ugni_rc)) {
            ucs_error("GNI_PostDataProbeWaitById, Error status: %s %d\n",
                      gni_err_str[ugni_rc], ugni_rc);
            continue;
        }
        if (ucs_unlikely(UCT_UGNI_UDT_CANCEL == id)) {
            /* When the iface is torn down, it will post and cancel a datagram with a
             * magic cookie as it's id that tells us to shut down.
             */
            break;
        }
        iface->events_ready = 1;
        ucs_trace("Recieved a new datagram");
        ucs_async_pipe_push(&iface->event_pipe);
    }

    return NULL;
}

unsigned uct_ugni_udt_progress(void *arg)
{
    uct_ugni_udt_iface_t * iface = (uct_ugni_udt_iface_t *)arg;

    uct_ugni_enter_async(&iface->super);
    ucs_arbiter_dispatch(&iface->super.arbiter, 1, uct_ugni_udt_ep_process_pending, NULL);
    uct_ugni_leave_async(&iface->super);
    return 0;
}

static void uct_ugni_udt_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_ugni_udt_desc_t *ugni_desc;
    uct_ugni_udt_iface_t *iface = ucs_container_of(self, uct_ugni_udt_iface_t,
                                                   release_desc);

    ugni_desc = (uct_ugni_udt_desc_t *)((uct_recv_desc_t *)desc - 1);
    ucs_assert_always(NULL != ugni_desc);
    uct_ugni_udt_reset_desc(ugni_desc, iface);
    ucs_mpool_put(ugni_desc);
}

static ucs_status_t uct_ugni_udt_query_tl_resources(uct_md_h md,
                                                    uct_tl_resource_desc_t **resource_p,
                                                    unsigned *num_resources_p)
{
    return uct_ugni_query_tl_resources(md, UCT_UGNI_UDT_TL_NAME,
                                       resource_p, num_resources_p);
}

static ucs_status_t uct_ugni_udt_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ugni_udt_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_udt_iface_t);

    memset(iface_attr, 0, sizeof(uct_iface_attr_t));
    iface_attr->cap.am.max_short       = iface->config.udt_seg_size -
                                         sizeof(uct_ugni_udt_header_t);
    iface_attr->cap.am.max_bcopy       = iface->config.udt_seg_size -
                                         sizeof(uct_ugni_udt_header_t);
    iface_attr->cap.am.opt_zcopy_align = 1;
    iface_attr->cap.am.align_mtu       = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->device_addr_len        = sizeof(uct_devaddr_ugni_t);
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_ugni_t);
    iface_attr->ep_addr_len            = 0;
    iface_attr->max_conn_priv          = 0;
    iface_attr->cap.flags              = UCT_IFACE_FLAG_AM_SHORT |
                                         UCT_IFACE_FLAG_AM_BCOPY |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                         UCT_IFACE_FLAG_PENDING |
                                         UCT_IFACE_FLAG_CB_ASYNC;

    iface_attr->overhead               = 1e-6;  /* 1 usec */
    iface_attr->latency.overhead       = 40e-6; /* 40 usec */
    iface_attr->latency.growth         = 0;
    iface_attr->bandwidth              = pow(1024, 2); /* bytes */
    iface_attr->priority               = 0;
    return UCS_OK;
}

void uct_ugni_proccess_datagram_pipe(int event_id, void *arg) {
    uct_ugni_udt_iface_t *iface = (uct_ugni_udt_iface_t *)arg;
    uct_ugni_udt_ep_t *ep;
    uct_ugni_udt_desc_t *datagram;
    ucs_status_t status;
    void *user_desc;
    gni_return_t ugni_rc;
    uint64_t id;

    ucs_trace_func("");

    uct_ugni_cdm_lock(&iface->super.cdm);
    ugni_rc = GNI_PostDataProbeById(uct_ugni_udt_iface_nic_handle(iface), &id);
    uct_ugni_cdm_unlock(&iface->super.cdm);
    while (GNI_RC_SUCCESS == ugni_rc) {
        status = recieve_datagram(iface, id, &ep);
        if (UCS_INPROGRESS == status) {
            if (ep != NULL){
                ucs_trace_data("Processing reply");
                datagram = ep->posted_desc;
                status = processs_datagram(iface, datagram);
                if (UCS_OK != status) {
                    user_desc = uct_ugni_udt_get_user_desc(datagram, iface);
                    uct_recv_desc(user_desc) = &iface->release_desc;
                } else {
                    ucs_mpool_put(datagram);
                }
                ep->posted_desc = NULL;
                uct_ugni_check_flush(ep->desc_flush_group);
            } else {
                ucs_trace_data("Processing wildcard");
                datagram = iface->desc_any;
                status = processs_datagram(iface, datagram);
                if (UCS_OK != status) {
                    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc,
                                             iface->desc_any, iface->desc_any=NULL);
                    user_desc = uct_ugni_udt_get_user_desc(datagram, iface);
                    uct_recv_desc(user_desc) = &iface->release_desc;
                }
                status = uct_ugni_udt_ep_any_post(iface);
                if (UCS_OK != status) {
                    /* We can't continue if we can't post the first receive */
                    ucs_error("Failed to post wildcard request");
                    return;
                }
            }
        }
        uct_ugni_cdm_lock(&iface->super.cdm);
        ugni_rc = GNI_PostDataProbeById(uct_ugni_udt_iface_nic_handle(iface), &id);
        uct_ugni_cdm_unlock(&iface->super.cdm);
    }

    ucs_async_pipe_drain(&iface->event_pipe);
    pthread_mutex_lock(&iface->device_lock);
    iface->events_ready = 0;
    pthread_mutex_unlock(&iface->device_lock);
    ucs_trace("Signaling device thread to resume monitoring");
    pthread_cond_signal(&iface->device_condition);

}

static void uct_ugni_udt_clean_wildcard(uct_ugni_udt_iface_t *iface)
{
    gni_return_t ugni_rc;
    uint32_t rem_addr, rem_id;
    gni_post_state_t post_state;
    uct_ugni_cdm_lock(&iface->super.cdm);
    ugni_rc = GNI_EpPostDataCancelById(iface->ep_any, UCT_UGNI_UDT_ANY);
    if (GNI_RC_SUCCESS != ugni_rc) {
        uct_ugni_cdm_unlock(&iface->super.cdm);
        ucs_error("GNI_EpPostDataCancel failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return;
    }
    ugni_rc = GNI_EpPostDataTestById(iface->ep_any, UCT_UGNI_UDT_ANY, &post_state, &rem_addr, &rem_id);
    if (GNI_RC_SUCCESS != ugni_rc) {
        if (GNI_RC_NO_MATCH != ugni_rc) {
            uct_ugni_cdm_unlock(&iface->super.cdm);
            ucs_error("GNI_EpPostDataTestById failed, Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return;
        }
    } else {
        if (GNI_POST_PENDING == post_state) {
            ugni_rc = GNI_EpPostDataWaitById(iface->ep_any, UCT_UGNI_UDT_ANY, -1, &post_state, &rem_addr, &rem_id);
        }
    }
    ugni_rc = GNI_EpDestroy(iface->ep_any);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_EpDestroy failed, Error status: %s %d\n",
                  gni_err_str[ugni_rc], ugni_rc);
    }
    uct_ugni_cdm_unlock(&iface->super.cdm);
}

/* Before this function is called, you MUST
 * A) Deregister the datagram processing function from the async thread.
 * B) Cancel the wildcard datagram.
 * C) Drain all other messages from the queue.
 */
static inline void uct_ugni_udt_terminate_thread(uct_ugni_udt_iface_t *iface)
{
    gni_return_t ugni_rc;
    gni_ep_handle_t   ep;

    uct_ugni_cdm_lock(&iface->super.cdm);
    ugni_rc = GNI_EpCreate(uct_ugni_udt_iface_nic_handle(iface), iface->super.local_cq, &ep);
    if (GNI_RC_SUCCESS != ugni_rc) {
        uct_ugni_cdm_unlock(&iface->super.cdm);
        ucs_error("GNI_EpCreate, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return;
    }
    ugni_rc = GNI_EpBind(ep, iface->super.cdm.dev->address, iface->super.cdm.domain_id);
    if (GNI_RC_SUCCESS != ugni_rc) {
        GNI_EpDestroy(ep);
        uct_ugni_cdm_unlock(&iface->super.cdm);
        ucs_error("GNI_EpBind failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return;
    }
    ugni_rc = GNI_EpPostDataWId(ep,
                                NULL, 0,
                                NULL, 0,
                                UCT_UGNI_UDT_CANCEL);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("Couldn't send cancel message to UGNI interface! %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
    }
    /* When the gni_ep is destroyed the above post will be canceled */
    ugni_rc = GNI_EpDestroy(ep);
    uct_ugni_cdm_unlock(&iface->super.cdm);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_EpDestroy failed, Error status: %s %d\n",
                  gni_err_str[ugni_rc], ugni_rc);
    }
}

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_udt_iface_t)
{
    void *dummy;
    uct_ugni_enter_async(&self->super);
    uct_ugni_udt_clean_wildcard(self);
    ucs_async_remove_handler(ucs_async_pipe_rfd(&self->event_pipe),1);
    if (self->events_ready) {
        uct_ugni_proccess_datagram_pipe(ucs_async_pipe_rfd(&self->event_pipe),self);
    }
    uct_ugni_udt_terminate_thread(self);
    pthread_join(self->event_thread, &dummy);
    ucs_async_pipe_destroy(&self->event_pipe);
    ucs_mpool_put(self->desc_any);
    ucs_mpool_cleanup(&self->free_desc, 1);
    pthread_mutex_destroy(&self->device_lock);
    uct_ugni_leave_async(&self->super);
}

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_udt_iface_t, uct_iface_t);

static uct_iface_ops_t uct_ugni_udt_iface_ops = {
    .ep_am_short              = uct_ugni_udt_ep_am_short,
    .ep_am_bcopy              = uct_ugni_udt_ep_am_bcopy,
    .ep_pending_add           = uct_ugni_udt_ep_pending_add,
    .ep_pending_purge         = uct_ugni_udt_ep_pending_purge,
    .ep_flush                 = uct_ugni_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_udt_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_udt_ep_t),
    .iface_flush              = uct_ugni_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress        = (void*)uct_ugni_udt_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_udt_iface_t),
    .iface_query              = uct_ugni_udt_iface_query,
    .iface_get_address        = uct_ugni_iface_get_address,
    .iface_get_device_address = uct_ugni_iface_get_dev_address,
    .iface_is_reachable       = uct_ugni_iface_is_reachable
};

static ucs_mpool_ops_t uct_ugni_udt_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

static UCS_CLASS_INIT_FUNC(uct_ugni_udt_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_ugni_iface_config_t *config = ucs_derived_of(tl_config, uct_ugni_iface_config_t);
    ucs_status_t status;
    uct_ugni_udt_desc_t *desc;
    gni_return_t ugni_rc;
    int rc;

    UCS_CLASS_CALL_SUPER_INIT(uct_ugni_iface_t, md, worker, params,
                              &uct_ugni_udt_iface_ops,
                              &config->super UCS_STATS_ARG(NULL));

    /* Setting initial configuration */
    self->config.udt_seg_size = GNI_DATAGRAM_MAXSIZE;
    self->config.rx_headroom  = params->rx_headroom;
    self->release_desc.cb     = uct_ugni_udt_iface_release_desc;

    status = ucs_async_pipe_create(&self->event_pipe);
    if (UCS_OK != status) {
        ucs_error("Pipe creation failed");
        goto exit;
    }

    status = ucs_mpool_init(&self->free_desc,
                            0,
                            uct_ugni_udt_get_diff(self) + self->config.udt_seg_size * 2,
                            uct_ugni_udt_get_diff(self),
                            UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                            128,                          /* grow */
                            config->mpool.max_bufs,       /* max buffers */
                            &uct_ugni_udt_desc_mpool_ops,
                            "UGNI-UDT-DESC");

    if (UCS_OK != status) {
        ucs_error("Mpool creation failed");
        goto clean_pipe;
    }

    ugni_rc = GNI_EpCreate(uct_ugni_udt_iface_nic_handle(self), NULL, &self->ep_any);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_EpCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_NO_DEVICE;
        goto clean_free_desc;
    }

    UCT_TL_IFACE_GET_TX_DESC(&self->super.super, &self->free_desc,
                             desc, goto clean_ep);
    ucs_debug("First wildcard desc is %p", desc);

    /* Init any desc */
    self->desc_any = desc;
    status = uct_ugni_udt_ep_any_post(self);
    if (UCS_OK != status) {
        /* We can't continue if we can't post the first receive */
        ucs_error("Failed to post wildcard request");
        goto clean_any_desc;
    }

    status = ucs_async_set_event_handler(self->super.super.worker->async->mode,
                                         ucs_async_pipe_rfd(&self->event_pipe),
                                         POLLIN,
                                         uct_ugni_proccess_datagram_pipe,
                                         self, self->super.super.worker->async);
                                 
    if (UCS_OK != status) {
        goto clean_cancel_desc;
    }

    pthread_mutex_init(&self->device_lock, NULL);
    pthread_cond_init(&self->device_condition, NULL);
    self->events_ready = 0;

    rc = pthread_create(&self->event_thread, NULL, uct_ugni_udt_device_thread, self);
    if(0 != rc) {
        goto clean_remove_event;
    }

    return UCS_OK;

 clean_remove_event:
    ucs_async_pipe_destroy(&self->event_pipe);
 clean_cancel_desc:
    uct_ugni_udt_clean_wildcard(self);
 clean_any_desc:
    ucs_mpool_put(self->desc_any);
 clean_ep:
    ugni_rc = GNI_EpDestroy(self->ep_any);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_EpDestroy failed, Error status: %s %d",
                 gni_err_str[ugni_rc], ugni_rc);
    }
 clean_free_desc:
    ucs_mpool_cleanup(&self->free_desc, 1);
 clean_pipe:
    ucs_async_pipe_destroy(&self->event_pipe);
 exit:
    uct_ugni_cleanup_base_iface(&self->super);
    ucs_error("Failed to activate interface");
    return status;
}

UCS_CLASS_DEFINE(uct_ugni_udt_iface_t, uct_ugni_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_udt_iface_t, uct_iface_t, uct_md_h,
                          uct_worker_h, const uct_iface_params_t*,
                          const uct_iface_config_t*);

UCT_TL_COMPONENT_DEFINE(uct_ugni_udt_tl_component,
                        uct_ugni_udt_query_tl_resources,
                        uct_ugni_udt_iface_t,
                        UCT_UGNI_UDT_TL_NAME,
                        "UGNI_UDT",
                        uct_ugni_udt_iface_config_table,
                        uct_ugni_iface_config_t);

UCT_MD_REGISTER_TL(&uct_ugni_md_component, &uct_ugni_udt_tl_component);
