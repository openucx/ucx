/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "ucs/type/class.h"
#include "uct/tl/context.h"

#include "ugni_iface.h"
#include "ugni_ep.h"

unsigned ugni_domain_global_counter = 0;

static void uct_ugni_progress(void *arg)
{
    gni_cq_entry_t  event_data = 0;
    gni_post_descriptor_t *event_post_desc_ptr;
    uct_ugni_base_desc_t *desc;
    uct_ugni_iface_t * iface = (uct_ugni_iface_t *)arg;
    gni_return_t ugni_rc;

    ugni_rc = GNI_CqGetEvent(iface->local_cq, &event_data);
    if (GNI_RC_NOT_DONE == ugni_rc) {
        return;
    }

    if ((GNI_RC_SUCCESS != ugni_rc && !event_data) || GNI_CQ_OVERRUN(event_data)) {
        ucs_error("GNI_CqGetEvent falied. Error status %s %d ",
                  gni_err_str[ugni_rc], ugni_rc);
        return;
    }

    ugni_rc = GNI_GetCompleted(iface->local_cq, event_data, &event_post_desc_ptr);
    if (GNI_RC_SUCCESS != ugni_rc && GNI_RC_TRANSACTION_ERROR != ugni_rc) {
        ucs_error("GNI_GetCompleted falied. Error status %s %d %d",
                  gni_err_str[ugni_rc], ugni_rc, GNI_RC_TRANSACTION_ERROR);
        return;
    }

    desc = (uct_ugni_base_desc_t *)event_post_desc_ptr;
    ucs_trace_async("Completion received on %p", desc);

    if (NULL != desc->comp_cb) {
        uct_invoke_completion(desc->comp_cb, desc + 1);
    }
    --iface->outstanding;
    --desc->ep->outstanding;
    ucs_mpool_put(desc);
    return;
}

static ucs_status_t uct_ugni_iface_flush(uct_iface_h tl_iface)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    if (0 == iface->outstanding) {
        return UCS_OK;
    }
    uct_ugni_progress(iface);
    return UCS_ERR_NO_RESOURCE;
}

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_iface_t)(uct_iface_t*);

ucs_status_t uct_ugni_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);

    *(uct_ugni_iface_addr_t*)iface_addr = iface->address;
    return UCS_OK;
}

ucs_status_t uct_ugni_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);

    memset(iface_attr, 0, sizeof(uct_iface_attr_t));
    iface_attr->cap.put.max_short      = iface->config.fma_seg_size;
    iface_attr->cap.put.max_bcopy      = iface->config.fma_seg_size;
    iface_attr->cap.put.max_zcopy      = iface->config.rdma_max_size;
    iface_attr->cap.get.max_bcopy      = iface->config.fma_seg_size - 8; /* alignment offset 4 (addr)+ 4 (len)*/
    iface_attr->cap.get.max_zcopy      = iface->config.rdma_max_size;
    iface_attr->iface_addr_len         = sizeof(uct_ugni_iface_addr_t);
    iface_attr->ep_addr_len            = sizeof(uct_ugni_ep_addr_t);
    iface_attr->cap.flags              = UCT_IFACE_FLAG_PUT_SHORT |
                                         UCT_IFACE_FLAG_PUT_BCOPY |
                                         UCT_IFACE_FLAG_PUT_ZCOPY |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP64 |
                                         UCT_IFACE_FLAG_ATOMIC_FADD64  |
                                         UCT_IFACE_FLAG_ATOMIC_ADD64   |
                                         UCT_IFACE_FLAG_GET_BCOPY      |
                                         UCT_IFACE_FLAG_GET_ZCOPY;

    iface_attr->completion_priv_len    = 0; /* TBD */

    return UCS_OK;
}

#define UCT_UGNI_RKEY_MAGIC  0xdeadbeefLL

static ucs_status_t uct_ugni_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    uct_ugni_iface_t *iface = ucs_container_of(pd, uct_ugni_iface_t, pd);

    ucs_snprintf_zero(pd_attr->name, sizeof(pd_attr->name), "%s",
                      iface->dev->fname);
    pd_attr->rkey_packed_size  = 3 * sizeof(uint64_t);
    pd_attr->cap.flags         = UCT_PD_FLAG_REG;
    pd_attr->cap.max_alloc     = 0;
    pd_attr->cap.max_reg       = ULONG_MAX;

    /* TODO make it configurable */
    pd_attr->alloc_methods.count = 3;
    pd_attr->alloc_methods.methods[0] = UCT_ALLOC_METHOD_HUGE;
    pd_attr->alloc_methods.methods[1] = UCT_ALLOC_METHOD_MMAP;
    pd_attr->alloc_methods.methods[2] = UCT_ALLOC_METHOD_HEAP;
    return UCS_OK;
}

static ucs_status_t uct_ugni_mem_reg(uct_pd_h pd, void *address, size_t length,
                                     uct_mem_h *memh_p)
{
    ucs_status_t rc;
    gni_return_t ugni_rc;
    uct_ugni_iface_t *iface = ucs_container_of(pd, uct_ugni_iface_t, pd);
    gni_mem_handle_t * mem_hndl = NULL;

    if (0 == length) {
        ucs_error("Unexpected length %zu", length);
        return UCS_ERR_INVALID_PARAM;
    }

    mem_hndl = ucs_malloc(sizeof(gni_mem_handle_t), "gni_mem_handle_t");
    if (NULL == mem_hndl) {
        ucs_error("Failed to allocate memory for gni_mem_handle_t");
        rc = UCS_ERR_NO_MEMORY;
        goto mem_err;
    }

    ugni_rc = GNI_MemRegister(iface->nic_handle, (uint64_t)address,
                              length, NULL,
                              GNI_MEM_READWRITE | GNI_MEM_RELAXED_PI_ORDERING,
                              -1, mem_hndl);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_MemRegister failed (addr %p, size %zu), Error status: %s %d",
                 address, length, gni_err_str[ugni_rc], ugni_rc);
        rc = UCS_ERR_IO_ERROR;
        goto mem_err;
    }

    ucs_debug("Memory registration address %p, len %lu, keys [%"PRIx64" %"PRIx64"]",
              address, length, mem_hndl->qword1, mem_hndl->qword2);
    *memh_p = mem_hndl;
    return UCS_OK;

mem_err:
    free(mem_hndl);
    return rc;
}

static ucs_status_t uct_ugni_mem_dereg(uct_pd_h pd, uct_mem_h memh)
{
    uct_ugni_iface_t *iface = ucs_container_of(pd, uct_ugni_iface_t, pd);
    gni_mem_handle_t *mem_hndl = (gni_mem_handle_t *) memh;
    gni_return_t ugni_rc;
    ucs_status_t rc = UCS_OK;

    ugni_rc = GNI_MemDeregister(iface->nic_handle, mem_hndl);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_MemDeregister failed, Error status: %s %d",
                 gni_err_str[ugni_rc], ugni_rc);
        rc = UCS_ERR_IO_ERROR;
    }
    ucs_free(mem_hndl);
    return rc;
}

static ucs_status_t uct_ugni_rkey_pack(uct_pd_h pd, uct_mem_h memh,
                                       void *rkey_buffer)
{
    gni_mem_handle_t *mem_hndl = (gni_mem_handle_t *) memh;
    uint64_t *ptr = rkey_buffer;

    ptr[0] = UCT_UGNI_RKEY_MAGIC;
    ptr[1] = mem_hndl->qword1;
    ptr[2] = mem_hndl->qword2;
    ucs_debug("Packed [ %"PRIx64" %"PRIx64" %"PRIx64"]", ptr[0], ptr[1], ptr[2]);
    return UCS_OK;
}

static void uct_ugni_rkey_release(uct_pd_h pd, uct_rkey_bundle_t *rkey_ob)
{
    ucs_free((void *)rkey_ob->rkey);
}

ucs_status_t uct_ugni_rkey_unpack(uct_pd_h pd, void *rkey_buffer,
                                  uct_rkey_bundle_t *rkey_ob)
{
    uint64_t *ptr = rkey_buffer;
    gni_mem_handle_t *mem_hndl = NULL;
    uint64_t magic = 0;

    ucs_debug("Unpacking [ %"PRIx64" %"PRIx64" %"PRIx64"]", ptr[0], ptr[1], ptr[2]);
    magic = ptr[0];
    if (magic != UCT_UGNI_RKEY_MAGIC) {
        ucs_error("Failed to identify key. Expected %llx but received %"PRIx64"",
                  UCT_UGNI_RKEY_MAGIC, magic);
        return UCS_ERR_UNSUPPORTED;
    }

    mem_hndl = ucs_malloc(sizeof(gni_mem_handle_t), "gni_mem_handle_t");
    if (NULL == mem_hndl) {
        ucs_error("Failed to allocate memory for gni_mem_handle_t");
        return UCS_ERR_NO_MEMORY;
    }

    mem_hndl->qword1 = ptr[1];
    mem_hndl->qword2 = ptr[2];
    rkey_ob->type = (void*)uct_ugni_rkey_release;
    rkey_ob->rkey = (uintptr_t)mem_hndl;
    return UCS_OK;

    /* need to add rkey release */
}

uct_iface_ops_t uct_ugni_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_iface_t),
    .iface_get_address   = uct_ugni_iface_get_address,
    .iface_flush         = uct_ugni_iface_flush,
    .ep_get_address      = uct_ugni_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_ugni_ep_connect_to_ep,
    .iface_query         = uct_ugni_iface_query,
    .ep_put_short        = uct_ugni_ep_put_short,
    .ep_put_bcopy        = uct_ugni_ep_put_bcopy,
    .ep_put_zcopy        = uct_ugni_ep_put_zcopy,
    .ep_am_short         = uct_ugni_ep_am_short,
    .ep_atomic_add64     = uct_ugni_ep_atomic_add64,
    .ep_atomic_fadd64    = uct_ugni_ep_atomic_fadd64,
    .ep_atomic_cswap64   = uct_ugni_ep_atomic_cswap64,
    .ep_get_bcopy        = uct_ugni_ep_get_bcopy,
    .ep_get_zcopy        = uct_ugni_ep_get_zcopy,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_ugni_ep_t),
};

uct_pd_ops_t uct_ugni_pd_ops = {
    .query        = uct_ugni_pd_query,
    .mem_reg      = uct_ugni_mem_reg,
    .mem_dereg    = uct_ugni_mem_dereg,
    .rkey_pack    = uct_ugni_rkey_pack,
    .rkey_unpack  = uct_ugni_rkey_unpack,
    .rkey_release = uct_ugni_rkey_release
};

static void uct_ugni_base_desc_init(void *mp_context, void *obj, void *chunk, void *arg)
{
    uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *) obj;
    /* zero ugni descriptor */
    memset(&base->desc, 0 , sizeof(base->desc));
}

static void uct_ugni_base_desc_key_init(uct_iface_h iface, void *obj, uct_mem_h memh)
{
    uct_ugni_base_desc_t *base = (uct_ugni_base_desc_t *)obj;
    /* call base initialization */
    uct_ugni_base_desc_init(iface, obj, NULL, NULL);
    /* set local keys */
    base->desc.local_mem_hndl = *(gni_mem_handle_t *)memh;
}
  

static UCS_CLASS_INIT_FUNC(uct_ugni_iface_t, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           uct_iface_config_t *tl_config)
{
    uct_ugni_iface_config_t *config = ucs_derived_of(tl_config, uct_ugni_iface_config_t);
    uct_ugni_context_t *ugni_ctx = ucs_component_get(worker->context,
                                                     ugni, uct_ugni_context_t);
    uct_ugni_device_t *dev;
    ucs_status_t rc;

    dev = uct_ugni_device_by_name(ugni_ctx, dev_name);
    if (NULL == dev) {
        ucs_error("No device was found: %s", dev_name);
        rc = UCS_ERR_NO_DEVICE;
        goto error;
    }

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_ugni_iface_ops, worker,
                              &self->pd, &config->super UCS_STATS_ARG(NULL));

    self->pd.ops           = &uct_ugni_pd_ops;
    self->dev              = dev;
    self->address.nic_addr = dev->address;

    /* Setting initial configuration */
    self->config.fma_seg_size  = UCT_UGNI_MAX_FMA;
    self->config.rdma_max_size = UCT_UGNI_MAX_RDMA;

    rc = ucs_mpool_create("UGNI-DESC-ONLY", sizeof(uct_ugni_base_desc_t),
                          0,                            /* alignment offset */
                          UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                          128 ,                         /* grow */
                          config->mpool.max_bufs,       /* max buffers */
                          &self->super.super,           /* iface */
                          ucs_mpool_hugetlb_malloc,     /* allocation hooks */
                          ucs_mpool_hugetlb_free,       /* free hook */
                          uct_ugni_base_desc_init,      /* init func */
                          NULL , &self->free_desc);
    if (UCS_OK != rc) {
        ucs_error("Mpool creation failed");
        goto error;
    }

    rc = ucs_mpool_create("UGNI-DESC-BUFFER", 
                          sizeof(uct_ugni_base_desc_t) +
                          self->config.fma_seg_size,
                          sizeof(uct_ugni_base_desc_t), /* alignment offset */
                          UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                          128 ,                         /* grow */
                          config->mpool.max_bufs,       /* max buffers */
                          &self->super.super,           /* iface */
                          ucs_mpool_hugetlb_malloc,     /* allocation hooks */
                          ucs_mpool_hugetlb_free,       /* free hook */
                          uct_ugni_base_desc_init,      /* init func */
                          NULL , &self->free_desc_buffer);
    if (UCS_OK != rc) {
        ucs_error("Mpool creation failed");
        goto clean_desc;
    }

    rc = uct_iface_mpool_create(&self->super.super, 
                                sizeof(uct_ugni_base_desc_t) + 8,
                                sizeof(uct_ugni_base_desc_t), /* alignment offset */
                                UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                                &config->mpool,               /* mpool config */ 
                                128 ,                         /* grow */
                                uct_ugni_base_desc_key_init,  /* memory/key init */
                                "UGNI-DESC-FAMO",             /* name */
                                &self->free_desc_famo);       /* mpool */
    if (UCS_OK != rc) {
        ucs_error("Mpool creation failed");
        goto clean_buffer;
    }

    rc = uct_iface_mpool_create(&self->super.super, 
                                sizeof(uct_ugni_get_desc_t) +
                                self->config.fma_seg_size,
                                sizeof(uct_ugni_get_desc_t), /* alignment offset */
                                UCS_SYS_CACHE_LINE_SIZE,      /* alignment */
                                &config->mpool,               /* mpool config */ 
                                128 ,                         /* grow */
                                uct_ugni_base_desc_key_init,  /* memory/key init */
                                "UGNI-DESC-GET",              /* name */
                                &self->free_desc_fget);       /* mpool */
    if (UCS_OK != rc) {
        ucs_error("Mpool creation failed");
        goto clean_famo;
    }

    ucs_notifier_chain_add(&worker->progress_chain, uct_ugni_progress, self);

    self->activated = false;
    self->outstanding = 0;
    /* TBD: atomic increment */
    ++ugni_ctx->num_ifaces;
    rc = ugni_activate_iface(self, ugni_ctx);
    if (UCS_OK == rc) {
        return rc;
    }

    ucs_error("Failed to activate interface");

clean_famo:
    ucs_mpool_destroy(self->free_desc_famo);
clean_buffer:
    ucs_mpool_destroy(self->free_desc_buffer);
clean_desc:
    ucs_mpool_destroy(self->free_desc);
error:
    return rc;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_iface_t)
{
    gni_return_t ugni_rc;

    ucs_notifier_chain_remove(&self->super.worker->progress_chain,
                              uct_ugni_progress, self);

    if (!self->activated) {
        /* We done with release */
        return;
    }

    ucs_mpool_destroy(self->free_desc_fget);
    ucs_mpool_destroy(self->free_desc_famo);
    ucs_mpool_destroy(self->free_desc_buffer);
    ucs_mpool_destroy(self->free_desc);

    /* TBD: Clean endpoints first (unbind and destroy) ?*/
    ucs_atomic_add32(&ugni_domain_global_counter, -1);
    ugni_rc = GNI_CqDestroy(self->local_cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_CqDestroy failed, Error status: %s %d",
                 gni_err_str[ugni_rc], ugni_rc);
    }
    ugni_rc = GNI_CdmDestroy(self->cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_CdmDestroy error status: %s (%d)",
                 gni_err_str[ugni_rc], ugni_rc);
    }
    self->activated = false;
}

UCS_CLASS_DEFINE(uct_ugni_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_iface_t, uct_iface_t, uct_worker_h,
                                 const char*, size_t, uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_iface_t, uct_iface_t);

uct_tl_ops_t uct_ugni_tl_ops = {
    .query_resources     = uct_ugni_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_ugni_iface_t),
};

#define UCT_UGNI_LOCAL_CQ (8192)
ucs_status_t ugni_activate_iface(uct_ugni_iface_t *iface, uct_ugni_context_t
                                 *ugni_ctx)
{
    int modes,
        rc,
        d_id;
    gni_return_t ugni_rc;

    if(iface->activated) {
        return UCS_OK;
    }
    /* Make sure that context is activated */
    rc = ugni_activate_domain(ugni_ctx);
    if (UCS_OK != rc) {
        ucs_error("Failed to activate context, Error status: %d", rc);
        return rc;
    }

    d_id = ucs_atomic_fadd32(&ugni_domain_global_counter, 1);

    iface->domain_id = ugni_ctx->pmi_rank_id + ugni_ctx->pmi_num_of_ranks *
                       d_id;
    modes = GNI_CDM_MODE_FORK_FULLCOPY | GNI_CDM_MODE_CACHED_AMO_ENABLED |
            GNI_CDM_MODE_ERR_NO_KILL | GNI_CDM_MODE_FAST_DATAGRAM_POLL;
    ucs_debug("Creating new domain with id %d (%d + %d * %d)",
              iface->domain_id, ugni_ctx->pmi_rank_id,
              ugni_ctx->pmi_num_of_ranks, d_id);
    ugni_rc = GNI_CdmCreate(iface->domain_id, ugni_ctx->ptag, ugni_ctx->cookie,
                            modes, &iface->cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    ugni_rc = GNI_CdmAttach(iface->cdm_handle, iface->dev->device_id,
                            &iface->pe_address, &iface->nic_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmAttach failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    ugni_rc = GNI_CqCreate(iface->nic_handle, UCT_UGNI_LOCAL_CQ, 0,
                           GNI_CQ_NOBLOCK,
                           NULL, NULL, &iface->local_cq);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CqCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }
    iface->activated = true;

    /* iface is activated */
    return UCS_OK;
}
