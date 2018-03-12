/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ugni_smsg_iface.h"
#include "ugni_smsg_ep.h"
#include <uct/ugni/base/ugni_def.h>
#include <uct/ugni/base/ugni_md.h>
#include <uct/ugni/base/ugni_device.h>
#include <ucs/arch/cpu.h>

#define UCT_UGNI_SMSG_TL_NAME "ugni_smsg"

static ucs_config_field_t uct_ugni_smsg_iface_config_table[] = {
    {"", "ALLOC=huge,thp,mmap,heap", NULL,
     ucs_offsetof(uct_ugni_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    UCT_IFACE_MPOOL_CONFIG_FIELDS("SMSG", -1, 0, "smsg",
                                  ucs_offsetof(uct_ugni_iface_config_t, mpool),
                                  "\nAttention: Setting this param with value != -1 is a dangerous thing\n"
                                  "and could cause deadlock or performance degradation."),

    {NULL}
};

static ucs_status_t progress_local_cq(uct_ugni_smsg_iface_t *iface){
    gni_return_t ugni_rc;
    gni_cq_entry_t event_data;
    uct_ugni_smsg_desc_t message_data;
    uct_ugni_smsg_desc_t *message_pointer;

    uct_ugni_cdm_lock(&iface->super.cdm);
    ugni_rc = GNI_CqGetEvent(iface->super.local_cq, &event_data);
    uct_ugni_cdm_unlock(&iface->super.cdm);
    if(GNI_RC_NOT_DONE == ugni_rc){
        return UCS_OK;
    }

    if((GNI_RC_SUCCESS != ugni_rc && !event_data) || GNI_CQ_OVERRUN(event_data)){
        /* TODO: handle overruns */
        ucs_error("Error posting data. CQ overrun = %d", (int)GNI_CQ_OVERRUN(event_data));
        return UCS_ERR_NO_RESOURCE;
    }

    message_data.msg_id = GNI_CQ_GET_MSG_ID(event_data);
    message_pointer = sglib_hashed_uct_ugni_smsg_desc_t_find_member(iface->smsg_list,&message_data);
    ucs_assert(NULL != message_pointer);
    uct_ugni_check_flush(message_pointer->flush_group);
    iface->super.outstanding--;
    sglib_hashed_uct_ugni_smsg_desc_t_delete(iface->smsg_list,message_pointer);
    ucs_mpool_put(message_pointer);
    return UCS_INPROGRESS;
}

static void process_mbox(uct_ugni_smsg_iface_t *iface, uct_ugni_smsg_ep_t *ep){
    uint8_t tag;
    void *data_ptr;
    gni_return_t ugni_rc;
    uct_ugni_smsg_header_t *header;
    void *user_data;

    /* Only one thread at a time can process mboxes for the iface. After it's done
       then everyone's messages have been drained. */
    if (!ucs_spin_trylock(&iface->mbox_lock)) {
        return;
    }
    while(1){
        tag = GNI_SMSG_ANY_TAG;
        uct_ugni_cdm_lock(&iface->super.cdm);
        ugni_rc = GNI_SmsgGetNextWTag(ep->super.ep, (void **)&data_ptr, &tag);
        uct_ugni_cdm_unlock(&iface->super.cdm);
        /* Yes, GNI_RC_NOT_DONE means that you're done with the smsg mailbox */
        if(GNI_RC_NOT_DONE == ugni_rc){
            break;
        }
        if(GNI_RC_SUCCESS != ugni_rc){
            ucs_error("Unhandled smsg error: %s %d", gni_err_str[ugni_rc], ugni_rc);
            break;
        }
        if(NULL == data_ptr){
            ucs_error("Empty data pointer in smsg.");
            break;
        }
        header = (uct_ugni_smsg_header_t *)data_ptr;
        user_data = (void *)(header + 1);

        uct_iface_trace_am(&iface->super.super, UCT_AM_TRACE_TYPE_RECV,
                           tag, user_data, header->length, "RX: AM");

        uct_iface_invoke_am(&iface->super.super, tag, user_data,
                            header->length, 0);
        uct_ugni_cdm_lock(&iface->super.cdm);
        ugni_rc = GNI_SmsgRelease(ep->super.ep);
        uct_ugni_cdm_unlock(&iface->super.cdm);
        if(GNI_RC_SUCCESS != ugni_rc){
            ucs_error("Unhandled smsg error in GNI_SmsgRelease: %s %d", gni_err_str[ugni_rc], ugni_rc);
            break;
        }
    }
    ucs_spin_unlock(&iface->mbox_lock);
}

static void uct_ugni_smsg_handle_remote_overflow(uct_ugni_smsg_iface_t *iface){
    gni_return_t ugni_rc;
    gni_cq_entry_t event_data;
    struct sglib_hashed_uct_ugni_ep_t_iterator ep_iterator;
    uct_ugni_ep_t *current_ep;
    uct_ugni_smsg_ep_t *ep;

    /* We don't know which EP dropped a completion entry, so flush everything */
    uct_ugni_cdm_lock(&iface->super.cdm);
    do{
        ugni_rc = GNI_CqGetEvent(iface->remote_cq, &event_data);
    } while(GNI_RC_NOT_DONE != ugni_rc);
    uct_ugni_cdm_unlock(&iface->super.cdm);
    current_ep = sglib_hashed_uct_ugni_ep_t_it_init(&ep_iterator, iface->super.eps);

    while(NULL != current_ep){
        ep = ucs_derived_of(current_ep, uct_ugni_smsg_ep_t);
        process_mbox(iface, ep);
        current_ep = sglib_hashed_uct_ugni_ep_t_it_next(&ep_iterator);
    }
}

ucs_status_t progress_remote_cq(uct_ugni_smsg_iface_t *iface)
{
    gni_return_t ugni_rc;
    gni_cq_entry_t event_data;
    uct_ugni_ep_t tl_ep;
    uct_ugni_ep_t *ugni_ep;
    uct_ugni_smsg_ep_t *ep;

    uct_ugni_cdm_lock(&iface->super.cdm);
    ugni_rc = GNI_CqGetEvent(iface->remote_cq, &event_data);
    uct_ugni_cdm_unlock(&iface->super.cdm);
    if(GNI_RC_NOT_DONE == ugni_rc){
        return UCS_OK;
    }

    if (GNI_RC_SUCCESS != ugni_rc || !GNI_CQ_STATUS_OK(event_data) || GNI_CQ_OVERRUN(event_data)) {
        if(GNI_RC_ERROR_RESOURCE == ugni_rc || (GNI_RC_SUCCESS == ugni_rc && GNI_CQ_OVERRUN(event_data))){
            ucs_debug("Detected remote CQ overrun. ungi_rc = %d [%s]", ugni_rc, gni_err_str[ugni_rc]);
            uct_ugni_smsg_handle_remote_overflow(iface);
            return UCS_OK;
        }
        ucs_error("GNI_CqGetEvent falied with unhandled error. Error status %s %d ",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_IO_ERROR;
    }

    tl_ep.hash_key = GNI_CQ_GET_INST_ID(event_data);
    ugni_ep = sglib_hashed_uct_ugni_ep_t_find_member(iface->super.eps, &tl_ep);
    ep = ucs_derived_of(ugni_ep, uct_ugni_smsg_ep_t);

    process_mbox(iface, ep);
    return UCS_INPROGRESS;
}

UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_smsg_iface_t, uct_iface_t);

static unsigned uct_ugni_smsg_progress(void *arg)
{
    uct_ugni_smsg_iface_t *iface = (uct_ugni_smsg_iface_t *)arg;
    ucs_status_t status;
    unsigned count = 0;

    do {
        ++count;
        status = progress_local_cq(iface);
    } while(status == UCS_INPROGRESS);
    do {
        ++count;
        status = progress_remote_cq(iface);
    } while(status == UCS_INPROGRESS);

    /* have a go a processing the pending queue */

    ucs_arbiter_dispatch(&iface->super.arbiter, iface->config.smsg_max_credit,
                         uct_ugni_ep_process_pending, NULL);
    return count - 2;
}

static ucs_status_t uct_ugni_smsg_query_tl_resources(uct_md_h md,
                                                     uct_tl_resource_desc_t **resource_p,
                                                     unsigned *num_resources_p)
{
    return uct_ugni_query_tl_resources(md, UCT_UGNI_SMSG_TL_NAME,
                                       resource_p, num_resources_p);
}

static ucs_status_t uct_ugni_smsg_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ugni_smsg_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_smsg_iface_t);

    memset(iface_attr, 0, sizeof(uct_iface_attr_t));
    iface_attr->cap.am.max_short       = iface->config.smsg_seg_size-sizeof(uint64_t);
    iface_attr->cap.am.max_bcopy       = iface->config.smsg_seg_size;
    iface_attr->cap.am.opt_zcopy_align = 1;
    iface_attr->cap.am.align_mtu       = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->device_addr_len        = sizeof(uct_devaddr_ugni_t);
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_ugni_t);
    iface_attr->ep_addr_len            = sizeof(uct_sockaddr_smsg_ugni_t);
    iface_attr->max_conn_priv          = 0;
    iface_attr->cap.flags              = UCT_IFACE_FLAG_AM_SHORT |
                                         UCT_IFACE_FLAG_AM_BCOPY |
                                         UCT_IFACE_FLAG_CONNECT_TO_EP |
                                         UCT_IFACE_FLAG_CB_SYNC  |
                                         UCT_IFACE_FLAG_PENDING;

    iface_attr->overhead               = 1e-6;  /* 1 usec */
    iface_attr->latency.overhead       = 40e-6; /* 40 usec */
    iface_attr->latency.growth         = 0;
    iface_attr->bandwidth              = pow(1024, 2); /* bytes */
    iface_attr->priority               = 0;
    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_ugni_smsg_iface_t)
{
    uct_worker_progress_remove(self->super.super.worker, &self->super.super.prog);
    ucs_mpool_cleanup(&self->free_desc, 1);
    ucs_mpool_cleanup(&self->free_mbox, 1);
    uct_ugni_destroy_cq(self->remote_cq, &self->super.cdm);
    ucs_spinlock_destroy(&self->mbox_lock);
}

static uct_iface_ops_t uct_ugni_smsg_iface_ops = {
    .ep_am_short              = uct_ugni_smsg_ep_am_short,
    .ep_am_bcopy              = uct_ugni_smsg_ep_am_bcopy,
    .ep_pending_add           = uct_ugni_ep_pending_add,
    .ep_pending_purge         = uct_ugni_ep_pending_purge,
    .ep_flush                 = uct_ugni_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_smsg_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_smsg_ep_t),
    .ep_get_address           = uct_ugni_smsg_ep_get_address,
    .ep_connect_to_ep         = uct_ugni_smsg_ep_connect_to_ep,
    .iface_flush              = uct_ugni_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = (void*)uct_ugni_smsg_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_smsg_iface_t),
    .iface_query              = uct_ugni_smsg_iface_query,
    .iface_get_device_address = uct_ugni_iface_get_dev_address,
    .iface_get_address        = uct_ugni_iface_get_address,
    .iface_is_reachable       = uct_ugni_iface_is_reachable
};

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

static UCS_CLASS_INIT_FUNC(uct_ugni_smsg_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_ugni_iface_config_t *config = ucs_derived_of(tl_config, uct_ugni_iface_config_t);
    ucs_status_t status;
    gni_return_t ugni_rc;
    unsigned int bytes_per_mbox;
    gni_smsg_attr_t smsg_attr;

    UCS_CLASS_CALL_SUPER_INIT(uct_ugni_iface_t, md, worker, params,
                              &uct_ugni_smsg_iface_ops,
                              &config->super UCS_STATS_ARG(NULL));

    /* Setting initial configuration */
    self->config.smsg_seg_size = 2048;
    self->config.rx_headroom  = params->rx_headroom;
    self->config.smsg_max_retransmit = 16;
    self->config.smsg_max_credit = 8;
    self->smsg_id = 0;

    smsg_attr.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
    smsg_attr.mbox_maxcredit = self->config.smsg_max_credit;
    smsg_attr.msg_maxsize = self->config.smsg_seg_size;
    status = ucs_spinlock_init(&self->mbox_lock);
    if (UCS_OK != status) {
            goto exit;
    }

    status = uct_ugni_create_cq(&self->remote_cq, 40000, &self->super.cdm);
    if (UCS_OK != status) {
        goto clean_lock;
    }
    ugni_rc = GNI_SmsgBufferSizeNeeded(&(smsg_attr), &bytes_per_mbox);
    self->bytes_per_mbox = ucs_align_up_pow2(bytes_per_mbox, ucs_get_page_size());

    if (ugni_rc != GNI_RC_SUCCESS) {
        ucs_error("Smsg buffer size calculation failed");
        status = UCS_ERR_INVALID_PARAM;
        goto clean_cq;
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
        goto clean_cq;
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
        goto clean_mbox;
    }

    ugni_rc = GNI_SmsgSetMaxRetrans(uct_ugni_iface_nic_handle(&self->super), self->config.smsg_max_retransmit);

    if (ugni_rc != GNI_RC_SUCCESS) {
        ucs_error("Smsg setting max retransmit count failed.");
        status = UCS_ERR_INVALID_PARAM;
        goto clean_desc;
    }

    /* TBD: eventually the uct_ugni_progress has to be moved to
     * udt layer so each ugni layer will have own progress */
    uct_worker_progress_add_safe(self->super.super.worker, uct_ugni_smsg_progress,
                                 self, &self->super.super.prog);

    return UCS_OK;

 clean_desc:
    ucs_mpool_cleanup(&self->free_desc, 1);
 clean_mbox:
    ucs_mpool_cleanup(&self->free_mbox, 1);
 clean_cq:
    uct_ugni_destroy_cq(self->remote_cq, &self->super.cdm);
 clean_lock:
    ucs_spinlock_destroy(&self->mbox_lock);
 exit:
    uct_ugni_cleanup_base_iface(&self->super);
    ucs_error("Failed to activate interface");
    return status;
}

UCS_CLASS_DEFINE(uct_ugni_smsg_iface_t, uct_ugni_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_smsg_iface_t, uct_iface_t, uct_md_h,
                          uct_worker_h, const uct_iface_params_t*,
                          const uct_iface_config_t *);

UCT_TL_COMPONENT_DEFINE(uct_ugni_smsg_tl_component,
                        uct_ugni_smsg_query_tl_resources,
                        uct_ugni_smsg_iface_t,
                        UCT_UGNI_SMSG_TL_NAME,
                        "UGNI_SMSG",
                        uct_ugni_smsg_iface_config_table,
                        uct_ugni_iface_config_t);

UCT_MD_REGISTER_TL(&uct_ugni_md_component, &uct_ugni_smsg_tl_component);
