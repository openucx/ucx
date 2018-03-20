/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ugni_rdma_ep.h"
#include "ugni_rdma_iface.h"
#include <uct/ugni/base/ugni_def.h>
#include <uct/ugni/base/ugni_md.h>
#include <uct/ugni/base/ugni_device.h>

static ucs_config_field_t uct_ugni_rdma_iface_config_table[] = {
    /* This tuning controls the allocation priorities for bouncing buffers */
    { "", "MAX_SHORT=2048;MAX_BCOPY=2048;ALLOC=huge,mmap,heap", NULL,
    ucs_offsetof(uct_ugni_rdma_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    UCT_IFACE_MPOOL_CONFIG_FIELDS("RDMA", -1, 0, "rdma",
                                  ucs_offsetof(uct_ugni_rdma_iface_config_t, mpool),
                                  "\nAttention: Setting this param with value != -1 is a dangerous thing\n"
                                  "and could cause deadlock or performance degradation."),

    {NULL}
};

static ucs_status_t uct_ugni_rdma_query_tl_resources(uct_md_h md,
                                                     uct_tl_resource_desc_t **resource_p,
                                                     unsigned *num_resources_p)
{
    return uct_ugni_query_tl_resources(md, UCT_UGNI_RDMA_TL_NAME,
                                       resource_p, num_resources_p);
}

static ucs_status_t uct_ugni_rdma_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_rdma_iface_t);

    memset(iface_attr, 0, sizeof(uct_iface_attr_t));
    iface_attr->cap.put.max_short       = iface->config.fma_seg_size;
    iface_attr->cap.put.max_bcopy       = iface->config.fma_seg_size;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = iface->config.rdma_max_size;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_bcopy       = iface->config.fma_seg_size - 8; /* alignment offset 4 (addr)+ 4 (len)*/
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = iface->config.rdma_max_size;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->cap.am.max_iov         = 1;
    iface_attr->cap.am.opt_zcopy_align = 1;
    iface_attr->cap.am.align_mtu       = iface_attr->cap.am.opt_zcopy_align;

    iface_attr->device_addr_len        = sizeof(uct_devaddr_ugni_t);
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_ugni_t);
    iface_attr->ep_addr_len            = 0;
    iface_attr->max_conn_priv          = 0;
    iface_attr->cap.flags              = UCT_IFACE_FLAG_PUT_SHORT |
                                         UCT_IFACE_FLAG_PUT_BCOPY |
                                         UCT_IFACE_FLAG_PUT_ZCOPY |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP64 |
                                         UCT_IFACE_FLAG_ATOMIC_FADD64  |
                                         UCT_IFACE_FLAG_ATOMIC_ADD64   |
                                         UCT_IFACE_FLAG_ATOMIC_DEVICE  |
                                         UCT_IFACE_FLAG_GET_BCOPY      |
                                         UCT_IFACE_FLAG_GET_ZCOPY      |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                         UCT_IFACE_FLAG_PENDING;

    if (uct_ugni_check_device_type(&iface->super, GNI_DEVICE_ARIES)) {
        iface_attr->cap.flags         |= UCT_IFACE_FLAG_PUT_SHORT |
                                         UCT_IFACE_FLAG_ATOMIC_SWAP64 |
                                         UCT_IFACE_FLAG_ATOMIC_SWAP32 |
                                         UCT_IFACE_FLAG_ATOMIC_FADD32 |
                                         UCT_IFACE_FLAG_ATOMIC_ADD32 |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP32;
    }
    iface_attr->overhead               = 80e-9; /* 80 ns */
    iface_attr->latency.overhead       = 900e-9; /* 900 ns */
    iface_attr->latency.growth         = 0;
    iface_attr->bandwidth              = 6911 * pow(1024,2); /* bytes */
    iface_attr->priority               = 0;
    return UCS_OK;
}

void uct_ugni_base_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
  uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *) obj;
  /* zero base descriptor */
  memset(base, 0 , sizeof(*base));
  base->free_cb = ucs_mpool_put;
}

void uct_ugni_base_desc_key_init(uct_iface_h iface, void *obj, uct_mem_h memh)
{
  uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *)obj;
  /* call base initialization */
  uct_ugni_base_desc_init(NULL, obj, NULL);
  /* set local keys */
  base->desc.local_mem_hndl = *(gni_mem_handle_t *)memh;
}

unsigned uct_ugni_progress(void *arg)
{
    gni_cq_entry_t  event_data = 0;
    gni_post_descriptor_t *event_post_desc_ptr;
    uct_ugni_base_desc_t *desc;
    uct_ugni_iface_t * iface = (uct_ugni_iface_t *)arg;
    gni_return_t ugni_rc;
    unsigned count = 0;

    while (1) {
        uct_ugni_cdm_lock(&iface->cdm);
        ugni_rc = GNI_CqGetEvent(iface->local_cq, &event_data);
        if (GNI_RC_NOT_DONE == ugni_rc) {
            uct_ugni_cdm_unlock(&iface->cdm);
            break;
        }
        if ((GNI_RC_SUCCESS != ugni_rc && !event_data) || GNI_CQ_OVERRUN(event_data)) {
            uct_ugni_cdm_unlock(&iface->cdm);
            ucs_error("GNI_CqGetEvent falied. Error status %s %d ",
                      gni_err_str[ugni_rc], ugni_rc);
            return count;
        }

        ugni_rc = GNI_GetCompleted(iface->local_cq, event_data, &event_post_desc_ptr);
        uct_ugni_cdm_unlock(&iface->cdm);
        if (GNI_RC_SUCCESS != ugni_rc && GNI_RC_TRANSACTION_ERROR != ugni_rc) {
            ucs_error("GNI_GetCompleted falied. Error status %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return count;
        }

        desc = (uct_ugni_base_desc_t *)event_post_desc_ptr;
        ucs_trace_async("Completion received on %p", desc);
        if (NULL != desc->comp_cb) {
            uct_invoke_completion(desc->comp_cb, UCS_OK);
        }
        desc->free_cb(desc);
        iface->outstanding--;
        uct_ugni_check_flush(desc->flush_group);
        ++count;
    }
    /* have a go a processing the pending queue */
    ucs_arbiter_dispatch(&iface->arbiter, 1, uct_ugni_ep_process_pending, NULL);
    return count;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_rdma_iface_t)
{
    uct_worker_progress_remove(self->super.super.worker, &self->super.super.prog);
    ucs_mpool_cleanup(&self->free_desc_get_buffer, 1);
    ucs_mpool_cleanup(&self->free_desc_get, 1);
    ucs_mpool_cleanup(&self->free_desc_famo, 1);
    ucs_mpool_cleanup(&self->free_desc_buffer, 1);
    ucs_mpool_cleanup(&self->free_desc, 1);
}

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_rdma_iface_t, uct_iface_t);

static uct_iface_ops_t uct_ugni_aries_rdma_iface_ops = {
    .ep_put_short             = uct_ugni_ep_put_short,
    .ep_put_bcopy             = uct_ugni_ep_put_bcopy,
    .ep_put_zcopy             = uct_ugni_ep_put_zcopy,
    .ep_get_bcopy             = uct_ugni_ep_get_bcopy,
    .ep_get_zcopy             = uct_ugni_ep_get_zcopy,
    .ep_am_short              = uct_ugni_ep_am_short,
    .ep_atomic_add64          = uct_ugni_ep_atomic_add64,
    .ep_atomic_fadd64         = uct_ugni_ep_atomic_fadd64,
    .ep_atomic_cswap64        = uct_ugni_ep_atomic_cswap64,
    .ep_atomic_swap64         = uct_ugni_ep_atomic_swap64,
    .ep_atomic_add32          = uct_ugni_ep_atomic_add32,
    .ep_atomic_fadd32         = uct_ugni_ep_atomic_fadd32,
    .ep_atomic_cswap32        = uct_ugni_ep_atomic_cswap32,
    .ep_atomic_swap32         = uct_ugni_ep_atomic_swap32,
    .ep_pending_add           = uct_ugni_ep_pending_add,
    .ep_pending_purge         = uct_ugni_ep_pending_purge,
    .ep_flush                 = uct_ugni_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_ep_t),
    .iface_flush              = uct_ugni_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = (void*)uct_ugni_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_rdma_iface_t),
    .iface_query              = uct_ugni_rdma_iface_query,
    .iface_get_device_address = uct_ugni_iface_get_dev_address,
    .iface_get_address        = uct_ugni_iface_get_address,
    .iface_is_reachable       = uct_ugni_iface_is_reachable
};

static uct_iface_ops_t uct_ugni_gemini_rdma_iface_ops = {
    .ep_put_short             = uct_ugni_ep_put_short,
    .ep_put_bcopy             = uct_ugni_ep_put_bcopy,
    .ep_put_zcopy             = uct_ugni_ep_put_zcopy,
    .ep_get_bcopy             = uct_ugni_ep_get_bcopy,
    .ep_get_zcopy             = uct_ugni_ep_get_zcopy,
    .ep_am_short              = uct_ugni_ep_am_short,
    .ep_atomic_add64          = uct_ugni_ep_atomic_add64,
    .ep_atomic_fadd64         = uct_ugni_ep_atomic_fadd64,
    .ep_atomic_cswap64        = uct_ugni_ep_atomic_cswap64,
    .ep_pending_add           = uct_ugni_ep_pending_add,
    .ep_pending_purge         = uct_ugni_ep_pending_purge,
    .ep_flush                 = uct_ugni_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create_connected      = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_ep_t),
    .iface_flush              = uct_ugni_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = ucs_empty_function,
    .iface_progress_disable   = ucs_empty_function,
    .iface_progress           = (void*)uct_ugni_progress,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_rdma_iface_t),
    .iface_query              = uct_ugni_rdma_iface_query,
    .iface_get_device_address = uct_ugni_iface_get_dev_address,
    .iface_get_address        = uct_ugni_iface_get_address,
    .iface_is_reachable       = uct_ugni_iface_is_reachable
};

static ucs_mpool_ops_t uct_ugni_rdma_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = uct_ugni_base_desc_init,
    .obj_cleanup   = NULL
};

static uct_iface_ops_t *uct_ugni_rdma_choose_ops_by_device(uct_ugni_device_t *dev)
{
    switch(dev->type) {
    case GNI_DEVICE_GEMINI:
        return &uct_ugni_gemini_rdma_iface_ops;
    case GNI_DEVICE_ARIES:
        return &uct_ugni_aries_rdma_iface_ops;
    default:
        ucs_error("Unexpected device found in uct_ugni_rdma_choose_ops_by_device."
                  "unexpected device type %s", dev->type_name);
        return NULL;
    }
}

static UCS_CLASS_INIT_FUNC(uct_ugni_rdma_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_ugni_rdma_iface_config_t *config = ucs_derived_of(tl_config, uct_ugni_rdma_iface_config_t);
    ucs_status_t status;
    uct_ugni_device_t *dev = uct_ugni_device_by_name(params->mode.device.dev_name);
    uct_iface_ops_t *ops;

    ops = uct_ugni_rdma_choose_ops_by_device(dev);
    if (NULL == ops) {
        status = UCS_ERR_NO_DEVICE;
        goto exit;
    }
    UCS_CLASS_CALL_SUPER_INIT(uct_ugni_iface_t, md, worker, params, ops,
                              &config->super UCS_STATS_ARG(NULL));
    /* Setting initial configuration */
    self->config.fma_seg_size  = UCT_UGNI_MAX_FMA;
    self->config.rdma_max_size = UCT_UGNI_MAX_RDMA;

    status = ucs_mpool_init(&self->free_desc,
                            0,
                            sizeof(uct_ugni_base_desc_t),
                            0,                            /* alignment offset */
                            UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                            128,                          /* grow */
                            config->mpool.max_bufs,       /* max buffers */
                            &uct_ugni_rdma_desc_mpool_ops,
                            "UGNI-DESC-ONLY");
    if (UCS_OK != status) {
        ucs_error("Mpool creation failed");
        goto exit;
    }

    status = ucs_mpool_init(&self->free_desc_get,
                            0,
                            sizeof(uct_ugni_rdma_fetch_desc_t),
                            0,                            /* alignment offset */
                            UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                            128 ,                         /* grow */
                            config->mpool.max_bufs,       /* max buffers */
                            &uct_ugni_rdma_desc_mpool_ops,
                            "UGNI-GET-DESC-ONLY");
    if (UCS_OK != status) {
        ucs_error("Mpool creation failed");
        goto clean_desc;
    }

    status = ucs_mpool_init(&self->free_desc_buffer,
                            0,
                            sizeof(uct_ugni_base_desc_t) + self->config.fma_seg_size,
                            sizeof(uct_ugni_base_desc_t), /* alignment offset */
                            UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                            128 ,                         /* grow */
                            config->mpool.max_bufs,       /* max buffers */
                            &uct_ugni_rdma_desc_mpool_ops,
                            "UGNI-DESC-BUFFER");
    if (UCS_OK != status) {
        ucs_error("Mpool creation failed");
        goto clean_desc_get;
    }

    status = uct_iface_mpool_init(&self->super.super,
                                  &self->free_desc_famo,
                                  sizeof(uct_ugni_rdma_fetch_desc_t) + 8,
                                  sizeof(uct_ugni_rdma_fetch_desc_t),/* alignment offset */
                                  UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                                  &config->mpool,               /* mpool config */
                                  128 ,                         /* grow */
                                  uct_ugni_base_desc_key_init,  /* memory/key init */
                                  "UGNI-DESC-FAMO");
    if (UCS_OK != status) {
        ucs_error("Mpool creation failed");
        goto clean_buffer;
    }

    status = uct_iface_mpool_init(&self->super.super,
                                  &self->free_desc_get_buffer,
                                  sizeof(uct_ugni_rdma_fetch_desc_t) +
                                  self->config.fma_seg_size,
                                  sizeof(uct_ugni_rdma_fetch_desc_t), /* alignment offset */
                                  UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                                  &config->mpool,               /* mpool config */
                                  128 ,                         /* grow */
                                  uct_ugni_base_desc_key_init,  /* memory/key init */
                                  "UGNI-DESC-GET");
    if (UCS_OK != status) {
        ucs_error("Mpool creation failed");
        goto clean_famo;
    }

    /* TBD: eventually the uct_ugni_progress has to be moved to
     * rdma layer so each ugni layer will have own progress */
    uct_worker_progress_add_safe(self->super.super.worker, uct_ugni_progress, self,
                                 &self->super.super.prog);
    return UCS_OK;

clean_famo:
    ucs_mpool_cleanup(&self->free_desc_famo, 1);
clean_buffer:
    ucs_mpool_cleanup(&self->free_desc_buffer, 1);
clean_desc_get:
    ucs_mpool_cleanup(&self->free_desc_get, 1);
clean_desc:
    ucs_mpool_cleanup(&self->free_desc, 1);
exit:
    uct_ugni_cleanup_base_iface(&self->super);
    ucs_error("Failed to activate interface");
    return status;
}

UCS_CLASS_DEFINE(uct_ugni_rdma_iface_t, uct_ugni_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_rdma_iface_t, uct_iface_t, uct_md_h,
                          uct_worker_h, const uct_iface_params_t*,
                          const uct_iface_config_t*);

UCT_TL_COMPONENT_DEFINE(uct_ugni_rdma_tl_component,
                        uct_ugni_rdma_query_tl_resources,
                        uct_ugni_rdma_iface_t,
                        UCT_UGNI_RDMA_TL_NAME,
                        "UGNI_RDMA",
                        uct_ugni_rdma_iface_config_table,
                        uct_ugni_rdma_iface_config_t);
UCT_MD_REGISTER_TL(&uct_ugni_md_component, &uct_ugni_rdma_tl_component);
