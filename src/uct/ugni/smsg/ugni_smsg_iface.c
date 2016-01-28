/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <pmi.h>
#include "ucs/type/class.h"

#include <ucs/arch/cpu.h>
#include <uct/ugni/base/ugni_iface.h>
#include "ugni_smsg_iface.h"
#include "ugni_smsg_ep.h"

#define UCT_UGNI_SMSG_TL_NAME "ugni_smsg"

static ucs_config_field_t uct_ugni_smsg_iface_config_table[] = {
    {"", "ALLOC=huge,mmap,heap", NULL,
     ucs_offsetof(uct_ugni_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    UCT_IFACE_MPOOL_CONFIG_FIELDS("SMSG", -1, 0, "smsg",
                                  ucs_offsetof(uct_ugni_iface_config_t, mpool),
                                  "\nAttention: Setting this param with value != -1 is a dangerous thing\n"
                                  "and could cause deadlock or performance degradation."),

    {NULL}
};

static void progress_local_cq(uct_ugni_smsg_iface_t *iface){
    gni_return_t ugni_rc;
    gni_cq_entry_t event_data;
    uct_ugni_smsg_desc_t message_data;
    uct_ugni_smsg_desc_t *message_pointer;

    if(0 == iface->super.outstanding){
        return;
    }

    ugni_rc = GNI_CqGetEvent(iface->super.local_cq, &event_data);
    if(GNI_RC_NOT_DONE == ugni_rc){
        return;
    }

    if((GNI_RC_SUCCESS != ugni_rc && !event_data) || GNI_CQ_OVERRUN(event_data)){
        /* TODO: handle overruns */
        ucs_error("Error posting data. CQ overrun = %d", (int)GNI_CQ_OVERRUN(event_data));
        return;
    }

    message_data.msg_id = GNI_CQ_GET_MSG_ID(event_data);
    message_pointer = sglib_hashed_uct_ugni_smsg_desc_t_find_member(iface->smsg_list,&message_data);
    message_pointer->ep->outstanding--;
    iface->super.outstanding--;
    sglib_hashed_uct_ugni_smsg_desc_t_delete(iface->smsg_list,message_pointer);
    ucs_mpool_put(message_pointer);
}

static void process_mbox(uct_ugni_smsg_iface_t *iface, uct_ugni_smsg_ep_t *ep){
    ucs_status_t status;
    uint8_t tag;
    void *data_ptr;
    gni_return_t ugni_rc;
    uct_ugni_smsg_header_t *header;
    void *user_data;

    pthread_mutex_lock(&uct_ugni_global_lock);

    while(1){
        tag = GNI_SMSG_ANY_TAG;
        ugni_rc = GNI_SmsgGetNextWTag(ep->super.ep, (void **)&data_ptr, &tag);

        /* Yes, GNI_RC_NOT_DONE means that you're done with the smsg mailbox */
        if(GNI_RC_NOT_DONE == ugni_rc){
            pthread_mutex_unlock(&uct_ugni_global_lock);
            return;
        }

        if(GNI_RC_SUCCESS != ugni_rc){
            ucs_error("Unhandled smsg error: %s %d", gni_err_str[ugni_rc], ugni_rc);
            pthread_mutex_unlock(&uct_ugni_global_lock);
            return;
        }

        if(NULL == data_ptr){
            ucs_error("Empty data pointer in smsg.");
            pthread_mutex_unlock(&uct_ugni_global_lock);
            return;
        }

        header = (uct_ugni_smsg_header_t *)data_ptr;
        user_data = (void *)(header + 1);
        void *user_desc = iface->user_desc+1;

        uct_iface_trace_am(&iface->super.super, UCT_AM_TRACE_TYPE_RECV,
                           tag, user_data, header->length, "RX: AM");

        status = uct_iface_invoke_am(&iface->super.super, tag, user_data,
                                     header->length, user_desc);

        if(status != UCS_OK){
            uct_recv_desc_iface(user_desc) = &iface->super.super.super;
            UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc,
                                     iface->user_desc, iface->user_desc = NULL);
        }

        ugni_rc = GNI_SmsgRelease(ep->super.ep);
        if(GNI_RC_SUCCESS != ugni_rc){
            ucs_error("Unhandled smsg error in GNI_SmsgRelease: %s %d", gni_err_str[ugni_rc], ugni_rc);
            pthread_mutex_unlock(&uct_ugni_global_lock);
            return;
        }
    }
}

static void uct_ugni_smsg_handle_remote_overflow(uct_ugni_smsg_iface_t *iface){
    gni_return_t ugni_rc;
    gni_cq_entry_t event_data;
    struct sglib_hashed_uct_ugni_ep_t_iterator ep_iterator;
    uct_ugni_ep_t *current_ep;
    uct_ugni_smsg_ep_t *ep;

    /* We don't know which EP dropped a completion entry, so flush everything */
    do{
        ugni_rc = GNI_CqGetEvent(iface->remote_cq, &event_data);
    } while(GNI_RC_NOT_DONE != ugni_rc);

    current_ep = sglib_hashed_uct_ugni_ep_t_it_init(&ep_iterator, iface->super.eps);
    while(NULL != current_ep){
        ep = ucs_derived_of(current_ep, uct_ugni_smsg_ep_t);
        process_mbox(iface, ep);
        current_ep = sglib_hashed_uct_ugni_ep_t_it_next(&ep_iterator);
    }
}

static void progress_remote_cq(uct_ugni_smsg_iface_t *iface){
    gni_return_t ugni_rc;
    gni_cq_entry_t event_data;
    uct_ugni_ep_t tl_ep;
    uct_ugni_ep_t *ugni_ep;
    uct_ugni_smsg_ep_t *ep;

    ugni_rc = GNI_CqGetEvent(iface->remote_cq, &event_data);

    if(GNI_RC_NOT_DONE == ugni_rc){
        return;
    }

    if (GNI_RC_SUCCESS != ugni_rc || !GNI_CQ_STATUS_OK(event_data) || GNI_CQ_OVERRUN(event_data)) {
        if(GNI_RC_ERROR_RESOURCE == ugni_rc || (GNI_RC_SUCCESS == ugni_rc && GNI_CQ_OVERRUN(event_data))){
            uct_ugni_smsg_handle_remote_overflow(iface);
            return;
        }
        ucs_error("GNI_CqGetEvent falied with unhandled error. Error status %s %d ",
                  gni_err_str[ugni_rc], ugni_rc);
        return;
    }

    tl_ep.hash_key = GNI_CQ_GET_INST_ID(event_data);
    ugni_ep = sglib_hashed_uct_ugni_ep_t_find_member(iface->super.eps, &tl_ep);
    ep = ucs_derived_of(ugni_ep, uct_ugni_smsg_ep_t);

    process_mbox(iface, ep);
}

UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_smsg_iface_t, uct_iface_t);

static void uct_ugni_smsg_progress(void *arg)
{
    uct_ugni_smsg_iface_t *iface = (uct_ugni_smsg_iface_t *)arg;

    progress_local_cq(iface);
    progress_remote_cq(iface);

    /* have a go a processing the pending queue */
    ucs_arbiter_dispatch(&iface->super.arbiter, 1, uct_ugni_ep_process_pending, NULL);
}

static void uct_ugni_smsg_iface_release_am_desc(uct_iface_t *tl_iface, void *desc)
{
    uct_ugni_smsg_desc_t *ugni_desc = ((uct_ugni_smsg_desc_t *)desc)-1;
    ucs_mpool_put(ugni_desc);
}

static ucs_status_t uct_ugni_smsg_query_tl_resources(uct_pd_h pd,
                                                     uct_tl_resource_desc_t **resource_p,
                                                     unsigned *num_resources_p)
{
    return uct_ugni_query_tl_resources(pd, UCT_UGNI_SMSG_TL_NAME,
                                       40000, /* nano sec */
                                       pow(1024,2), /* bytes */
                                       resource_p, num_resources_p);
}

static ucs_status_t uct_ugni_smsg_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ugni_smsg_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_smsg_iface_t);

    memset(iface_attr, 0, sizeof(uct_iface_attr_t));
    iface_attr->cap.am.max_short       = iface->config.smsg_seg_size-sizeof(uint64_t);
    iface_attr->cap.am.max_bcopy       = iface->config.smsg_seg_size;
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_ugni_t);
    iface_attr->ep_addr_len            = sizeof(uct_sockaddr_smsg_ugni_t);
    iface_attr->cap.flags              = UCT_IFACE_FLAG_AM_SHORT |
                                         UCT_IFACE_FLAG_AM_BCOPY |
                                         UCT_IFACE_FLAG_CONNECT_TO_EP |
                                         UCT_IFACE_FLAG_AM_THREAD_SINGLE;
    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_ugni_smsg_iface_t)
{
    uct_worker_progress_unregister(self->super.super.worker,
                                   uct_ugni_smsg_progress, self);
    if (!self->super.activated) {
        return;
    }
    ucs_mpool_put(self->user_desc);

    ucs_mpool_cleanup(&self->free_desc, 1);
    ucs_mpool_cleanup(&self->free_mbox, 1);
}

static ucs_status_t uct_ugni_smsg_iface_flush(uct_iface_h tl_iface)
{
    uct_ugni_smsg_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_smsg_iface_t);

    if (0 == iface->super.outstanding) {
        UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
        return UCS_OK;
    }

    progress_local_cq(iface);

    UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_INPROGRESS;
}

static ucs_status_t uct_ugni_smsg_ep_flush(uct_ep_h tl_ep){
    uct_ugni_smsg_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_smsg_ep_t);
    uct_ugni_smsg_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_smsg_iface_t);

    if (0 == ep->super.outstanding) {
        UCT_TL_EP_STAT_FLUSH(ucs_derived_of(tl_ep, uct_base_ep_t));
        return UCS_OK;
    }

    progress_local_cq(iface);

    UCT_TL_EP_STAT_FLUSH_WAIT(ucs_derived_of(tl_ep, uct_base_ep_t));
    return UCS_INPROGRESS;
}

uct_iface_ops_t uct_ugni_smsg_iface_ops = {
    .iface_query           = uct_ugni_smsg_iface_query,
    .iface_flush           = uct_ugni_smsg_iface_flush,
    .iface_close           = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_smsg_iface_t),
    .iface_get_address     = uct_ugni_iface_get_address,
    .iface_is_reachable    = uct_ugni_iface_is_reachable,
    .iface_release_am_desc = uct_ugni_smsg_iface_release_am_desc,
    .ep_create             = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_smsg_ep_t),
    .ep_get_address        = uct_ugni_smsg_ep_get_address,
    .ep_connect_to_ep      = uct_ugni_smsg_ep_connect_to_ep,
    .ep_destroy            = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_smsg_ep_t),
    .ep_pending_add        = uct_ugni_ep_pending_add,
    .ep_pending_purge      = uct_ugni_ep_pending_purge,
    .ep_am_short           = uct_ugni_smsg_ep_am_short,
    .ep_am_bcopy           = uct_ugni_smsg_ep_am_bcopy,
    .ep_flush              = uct_ugni_smsg_ep_flush,
};

#define UCT_UGNI_LOCAL_CQ (8192)

static ucs_status_t ugni_smsg_activate_iface(uct_ugni_smsg_iface_t *iface)
{
    ucs_status_t status;
    gni_return_t ugni_rc;

    if(iface->super.activated) {
        return UCS_OK;
    }
    /*pull out these chunks into common routines */
    status = uct_ugni_init_nic(0, &iface->super.domain_id,
                               &iface->super.cdm_handle, &iface->super.nic_handle,
                               &iface->super.pe_address);
    if (UCS_OK != status) {
        ucs_error("Failed to UGNI NIC, Error status: %d", status);
        return status;
    }

    ugni_rc = GNI_CqCreate(iface->super.nic_handle, UCT_UGNI_LOCAL_CQ, 0,
                           GNI_CQ_NOBLOCK,
                           NULL, NULL, &iface->super.local_cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CqCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    ugni_rc = GNI_CqCreate(iface->super.nic_handle, 40000, 0,
                           GNI_CQ_NOBLOCK,
                           NULL, NULL, &iface->remote_cq);

    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CqCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    iface->super.activated = true;

    /* iface is activated */
    return UCS_OK;
}

static ucs_status_t ugni_smsg_deactivate_iface(uct_ugni_smsg_iface_t *iface)
{
    gni_return_t ugni_rc;

    if(!iface->super.activated) {
        return UCS_OK;
    }

    ugni_rc = GNI_CqDestroy(iface->super.local_cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_CqDestroy failed, Error status: %s %d",
                 gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_IO_ERROR;
    }

    ugni_rc = GNI_CqDestroy(iface->remote_cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_CqDestroy failed, Error status: %s %d",
                 gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_IO_ERROR;
    }

    ugni_rc = GNI_CdmDestroy(iface->super.cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_CdmDestroy error status: %s (%d)",
                 gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_IO_ERROR;
    }

    iface->super.activated = false ;
    return UCS_OK;
}

static ucs_mpool_ops_t uct_ugni_smsg_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = uct_ugni_base_desc_init,
    .obj_cleanup   = NULL
};

static ucs_mpool_ops_t uct_ugni_smsg_mbox_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_mmap,
    .chunk_release = ucs_mpool_chunk_munmap,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

static UCS_CLASS_INIT_FUNC(uct_ugni_smsg_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_ugni_iface_config_t *config = ucs_derived_of(tl_config, uct_ugni_iface_config_t);
    ucs_status_t status;
    gni_return_t ugni_rc;
    unsigned int bytes_per_mbox;
    gni_smsg_attr_t smsg_attr;

    pthread_mutex_lock(&uct_ugni_global_lock);

    UCS_CLASS_CALL_SUPER_INIT(uct_ugni_iface_t, pd, worker, dev_name, &uct_ugni_smsg_iface_ops,
                              &config->super UCS_STATS_ARG(NULL));

    /* Setting initial configuration */
    self->config.smsg_seg_size = 2048;
    self->config.rx_headroom  = rx_headroom;
    self->config.smsg_max_retransmit = 16;
    self->config.smsg_max_credit = 8;
    self->smsg_id = 0;

    smsg_attr.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
    smsg_attr.mbox_maxcredit = self->config.smsg_max_credit;
    smsg_attr.msg_maxsize = self->config.smsg_seg_size;

    ugni_rc = GNI_SmsgBufferSizeNeeded(&(smsg_attr), &bytes_per_mbox);
    self->bytes_per_mbox = ucs_align_up_pow2(bytes_per_mbox, ucs_get_page_size());

    if (ugni_rc != GNI_RC_SUCCESS) {
        ucs_error("Smsg buffer size calculation failed");
        status = UCS_ERR_INVALID_PARAM;
        goto exit;
    }

    status = ucs_mpool_init(&self->free_desc,
                            0,
                            self->config.smsg_seg_size + sizeof(uct_ugni_smsg_desc_t),
                            0,
                            UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                            128           ,               /* grow */
                            config->mpool.max_bufs,       /* max buffers */
                            &uct_ugni_smsg_desc_mpool_ops,
                            "UGNI-SMSG-DESC");

    if (UCS_OK != status) {
        ucs_error("Desc Mpool creation failed");
        goto exit;
    }

    status = ucs_mpool_init(&self->free_mbox,
                            0,
                            self->bytes_per_mbox + sizeof(uct_ugni_smsg_mbox_t),
                            sizeof(uct_ugni_smsg_mbox_t),
                            UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                            128,                          /* grow */
                            config->mpool.max_bufs,       /* max buffers */
                            &uct_ugni_smsg_mbox_mpool_ops,
                            "UGNI-SMSG-MBOX");

    if (UCS_OK != status) {
        ucs_error("Mbox Mpool creation failed");
        goto clean_desc;
    }

    UCT_TL_IFACE_GET_TX_DESC(&self->super.super, &self->free_desc,
                             self->user_desc, self->user_desc = NULL);

    status = ugni_smsg_activate_iface(self);
    if (UCS_OK != status) {
        ucs_error("Failed to activate the interface");
        goto clean_mbox;
    }

    ugni_rc = GNI_SmsgSetMaxRetrans(self->super.nic_handle, self->config.smsg_max_retransmit);

    if (ugni_rc != GNI_RC_SUCCESS) {
        ucs_error("Smsg setting max retransmit count failed.");
        status = UCS_ERR_INVALID_PARAM;
        goto clean_iface;
    }

    /* TBD: eventually the uct_ugni_progress has to be moved to
     * udt layer so each ugni layer will have own progress */
    uct_worker_progress_register(worker, uct_ugni_smsg_progress, self);
    pthread_mutex_unlock(&uct_ugni_global_lock);
    return UCS_OK;

 clean_iface:
    ugni_smsg_deactivate_iface(self);
 clean_desc:
    ucs_mpool_put(self->user_desc);
    ucs_mpool_cleanup(&self->free_desc, 1);
 clean_mbox:
    ucs_mpool_cleanup(&self->free_mbox, 1);
 exit:
    ucs_error("Failed to activate interface");
    pthread_mutex_unlock(&uct_ugni_global_lock);
    return status;
}

UCS_CLASS_DEFINE(uct_ugni_smsg_iface_t, uct_ugni_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_smsg_iface_t, uct_iface_t,
                          uct_pd_h, uct_worker_h,
                          const char*, size_t, const uct_iface_config_t *);

UCT_TL_COMPONENT_DEFINE(uct_ugni_smsg_tl_component,
                        uct_ugni_smsg_query_tl_resources,
                        uct_ugni_smsg_iface_t,
                        UCT_UGNI_SMSG_TL_NAME,
                        "UGNI_SMSG",
                        uct_ugni_smsg_iface_config_table,
                        uct_ugni_iface_config_t);

UCT_PD_REGISTER_TL(&uct_ugni_pd_component, &uct_ugni_smsg_tl_component);
