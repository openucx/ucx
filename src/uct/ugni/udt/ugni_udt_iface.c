/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <pmi.h>
#include "ucs/type/class.h"
#include "uct/base/uct_pd.h"

#include <ucs/arch/cpu.h>
#include <uct/ugni/base/ugni_iface.h>
#include "ugni_udt_iface.h"
#include "ugni_udt_ep.h"

#define UCT_UGNI_UDT_TL_NAME "ugni_udt"

static ucs_config_field_t uct_ugni_udt_iface_config_table[] = {
    {"", "ALLOC=huge,mmap,heap", NULL,
    ucs_offsetof(uct_ugni_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    UCT_IFACE_MPOOL_CONFIG_FIELDS("UDT", -1, 0, "udt",
                                  ucs_offsetof(uct_ugni_iface_config_t, mpool),
                                  "\nAttention: Setting this param with value != -1 is a dangerous thing\n"
                                  "and could cause deadlock or performance degradation."),

    {NULL}
};

static void uct_ugni_udt_progress(void *arg)
{
    uint32_t rem_addr,
             rem_id;
    uint64_t id;
    void *payload;
    void *user_desc;

    ucs_status_t status;

    uct_ugni_udt_desc_t *desc;
    uct_ugni_udt_header_t *header;
    uct_ugni_udt_iface_t * iface = (uct_ugni_udt_iface_t *)arg;
    uct_ugni_udt_ep_t *ep;

    gni_ep_handle_t ugni_ep;
    gni_post_state_t post_state;
    gni_return_t ugni_rc;

    pthread_mutex_lock(&uct_ugni_global_lock);
    ugni_rc = GNI_PostDataProbeById(iface->super.nic_handle, &id);
    if (ucs_unlikely(GNI_RC_SUCCESS != ugni_rc)) {
        if (GNI_RC_NO_MATCH != ugni_rc) {
            ucs_error("GNI_PostDataProbeById , Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
        }
        goto exit;
    }

    if (UCT_UGNI_UDT_ANY == id) {
        /* New incomming message */
        ep = NULL;
        ugni_ep = iface->ep_any;
        desc = iface->desc_any;
    } else {
        /* Ack message */
        ep = ucs_derived_of(uct_ugni_iface_lookup_ep(&iface->super, id),
                            uct_ugni_udt_ep_t);
        if (ucs_unlikely(NULL == ep)) {
            ucs_error("Can not lookup ep with id %"PRIx64,id);
            goto exit;
        }
        ugni_ep = ep->super.ep;
        desc = ep->posted_desc;
    }

    ugni_rc = GNI_EpPostDataWaitById(ugni_ep, id, -1, &post_state, &rem_addr, &rem_id);
    if (ucs_unlikely(GNI_RC_SUCCESS != ugni_rc)) {
        ucs_error("GNI_EpPostDataWaitById, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        goto exit;
    }

    header = uct_ugni_udt_get_rheader(desc, iface);
    payload = uct_ugni_udt_get_rpayload(desc, iface);
    user_desc = uct_ugni_udt_get_user_desc(desc, iface);

    if (UCT_UGNI_UDT_ANY == id) {
        /* New incomming message */
        ucs_assert_always(header->type == UCT_UGNI_UDT_PAYLOAD);
        uct_iface_trace_am(&iface->super.super, UCT_AM_TRACE_TYPE_RECV,
                           header->am_id, payload, header->length, "RX: AM");
        status = uct_iface_invoke_am(&iface->super.super, header->am_id, payload,
                                     header->length, user_desc);
        if (UCS_OK != status) {
            uct_ugni_udt_desc_t *new_desc;
            /* set iface for a later release call */
            uct_recv_desc_iface(user_desc) = &iface->super.super.super;
            /* Allocate a new element */
            UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc,
                                     new_desc, goto exit);
            /* set the new desc */
            iface->desc_any = new_desc;
        }
        status = uct_ugni_udt_ep_any_post(iface);
        if (ucs_unlikely(UCS_OK != status)) {
            ucs_error("Failed to post uct_ugni_udt_ep_any_post");
            goto exit;
        }
    } else {
        /* Ack message */
        ucs_assert_always(NULL != ep);

        if (header->type == UCT_UGNI_UDT_PAYLOAD) {
            /* data message was received */
            uct_iface_trace_am(&iface->super.super, UCT_AM_TRACE_TYPE_RECV,
                               header->am_id, payload, header->length, "RX: AM");
            status = uct_iface_invoke_am(&iface->super.super, header->am_id, payload,
                                         header->length, user_desc);
            if (UCS_OK == status) {
                uct_ugni_udt_reset_desc(desc, iface);
                ucs_mpool_put(desc);
            } else {
                /* set iface for a later release call */
                uct_recv_desc_iface(user_desc) = &iface->super.super.super;
            }
        }
        /* no data, just an ack */
        --iface->super.outstanding;
        --ep->super.outstanding;
        ep->posted_desc = NULL;
    }

    /* have a go a processing the pending queue */

exit:
    pthread_mutex_unlock(&uct_ugni_global_lock);
    ucs_arbiter_dispatch(&iface->super.arbiter, 1, uct_ugni_ep_process_pending, NULL);
}

static void uct_ugni_udt_iface_release_am_desc(uct_iface_t *tl_iface, void *desc)
{
    uct_ugni_udt_desc_t *ugni_desc;
    uct_ugni_udt_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_udt_iface_t);

    ucs_debug("Called uct_ugni_udt_iface_release_am_desc");
    ugni_desc = (uct_ugni_udt_desc_t *)((uct_am_recv_desc_t *)desc - 1);
    ucs_assert_always(NULL != ugni_desc);
    uct_ugni_udt_reset_desc(ugni_desc, iface);
    ucs_mpool_put(ugni_desc);
}

static ucs_status_t uct_ugni_udt_query_tl_resources(uct_pd_h pd,
                                                    uct_tl_resource_desc_t **resource_p,
                                                    unsigned *num_resources_p)
{
    return uct_ugni_query_tl_resources(pd, UCT_UGNI_UDT_TL_NAME,
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
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_ugni_t);
    iface_attr->ep_addr_len            = 0;
    iface_attr->cap.flags              = UCT_IFACE_FLAG_AM_SHORT |
                                         UCT_IFACE_FLAG_AM_BCOPY |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                         UCT_IFACE_FLAG_PENDING |
                                         UCT_IFACE_FLAG_AM_THREAD_SINGLE;

    iface_attr->overhead               = 1e-6;  /* 1 usec */
    iface_attr->latency                = 40e-6; /* 40 usec */
    iface_attr->bandwidth              = pow(1024, 2); /* bytes */
    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_ugni_udt_iface_t)
{
    gni_return_t ugni_rc;

    uct_worker_progress_unregister(self->super.super.worker,
                                   uct_ugni_udt_progress, self);
    if (!self->super.activated) {
        /* We done with release */
        return;
    }

    ugni_rc = GNI_EpPostDataCancel(self->ep_any);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_debug("GNI_EpPostDataCancel failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return;
    }
    ucs_mpool_put(self->desc_any);
    ucs_mpool_cleanup(&self->free_desc, 1);
}

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_udt_iface_t, uct_iface_t);

uct_iface_ops_t uct_ugni_udt_iface_ops = {
    .iface_query           = uct_ugni_udt_iface_query,
    .iface_flush           = uct_ugni_iface_flush,
    .iface_close           = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_udt_iface_t),
    .iface_get_address     = uct_ugni_iface_get_address,
    .iface_is_reachable    = uct_ugni_iface_is_reachable,
    .iface_release_am_desc = uct_ugni_udt_iface_release_am_desc,
    .ep_create_connected   = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_udt_ep_t),
    .ep_destroy            = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_udt_ep_t),
    .ep_pending_add        = uct_ugni_ep_pending_add,
    .ep_pending_purge      = uct_ugni_ep_pending_purge,
    .ep_am_short           = uct_ugni_udt_ep_am_short,
    .ep_am_bcopy           = uct_ugni_udt_ep_am_bcopy,
    .ep_flush              = uct_ugni_ep_flush,
};

static ucs_mpool_ops_t uct_ugni_udt_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

static UCS_CLASS_INIT_FUNC(uct_ugni_udt_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_ugni_iface_config_t *config = ucs_derived_of(tl_config, uct_ugni_iface_config_t);
    ucs_status_t status;
    uct_ugni_udt_desc_t *desc;
    gni_return_t ugni_rc;

    pthread_mutex_lock(&uct_ugni_global_lock);

    UCS_CLASS_CALL_SUPER_INIT(uct_ugni_iface_t, pd, worker, dev_name, &uct_ugni_udt_iface_ops,
                              &config->super UCS_STATS_ARG(NULL));

    /* Setting initial configuration */
    self->config.udt_seg_size = GNI_DATAGRAM_MAXSIZE;
    self->config.rx_headroom  = rx_headroom;

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
        goto exit;
    }

    status = ugni_activate_iface(&self->super);
    if (UCS_OK != status) {
        ucs_error("Failed to activate the interface");
        goto clean_desc;
    }

    ugni_rc = GNI_EpCreate(self->super.nic_handle, NULL, &self->ep_any);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_NO_DEVICE;
        goto clean_iface;
    }

    UCT_TL_IFACE_GET_TX_DESC(&self->super.super, &self->free_desc,
                             desc, goto clean_ep);
    /* Init any desc */
    self->desc_any = desc;
    status = uct_ugni_udt_ep_any_post(self);
    if (UCS_OK != status) {
        /* We can't continue if we can't post the first receive */
        ucs_error("Failed to post wildcard request");
        goto clean_ep;
    }

    /* TBD: eventually the uct_ugni_progress has to be moved to
     * udt layer so each ugni layer will have own progress */
    uct_worker_progress_register(worker, uct_ugni_udt_progress, self);
    pthread_mutex_unlock(&uct_ugni_global_lock);
    return UCS_OK;

clean_ep:
    ugni_rc = GNI_EpDestroy(self->ep_any);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_EpDestroy failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
    }
clean_iface:
    ugni_deactivate_iface(&self->super);
clean_desc:
    ucs_mpool_cleanup(&self->free_desc, 1);
exit:
    ucs_error("Failed to activate interface");
    pthread_mutex_unlock(&uct_ugni_global_lock);
    return status;
}

UCS_CLASS_DEFINE(uct_ugni_udt_iface_t, uct_ugni_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_udt_iface_t, uct_iface_t,
                          uct_pd_h, uct_worker_h,
                          const char*, size_t, const uct_iface_config_t *);

UCT_TL_COMPONENT_DEFINE(uct_ugni_udt_tl_component,
                        uct_ugni_udt_query_tl_resources,
                        uct_ugni_udt_iface_t,
                        UCT_UGNI_UDT_TL_NAME,
                        "UGNI_UDT",
                        uct_ugni_udt_iface_config_table,
                        uct_ugni_iface_config_t);

UCT_PD_REGISTER_TL(&uct_ugni_pd_component, &uct_ugni_udt_tl_component);
