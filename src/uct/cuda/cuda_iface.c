/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "ucs/type/class.h"
#include "uct/tl/context.h"

#include "cuda_iface.h"
#include "cuda_ep.h"
#include "cuda_context.h"

static ucs_status_t uct_cuda_iface_flush(uct_iface_h tl_iface)
{
    return UCS_OK;
}

/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_iface_t)(uct_iface_t*);

ucs_status_t uct_cuda_iface_get_address(uct_iface_h tl_iface, 
                                        uct_iface_addr_t *iface_addr)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_iface_t);

    *(uct_cuda_iface_addr_t*)iface_addr = iface->addr;
    return UCS_OK;
}

#define UCT_CUDA_MAX_SHORT_LENGTH 2048 /* FIXME temp value for now */

ucs_status_t uct_cuda_iface_query(uct_iface_h iface, uct_iface_attr_t *iface_attr)
{
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    /* FIXME all of these values */
    iface_attr->iface_addr_len         = sizeof(uct_cuda_iface_addr_t);
    iface_attr->ep_addr_len            = sizeof(uct_cuda_ep_addr_t);
    iface_attr->cap.flags              = 0;

    iface_attr->cap.put.max_short      = 0;
    iface_attr->cap.put.max_bcopy      = 0;
    iface_attr->cap.put.max_zcopy      = 0;

    iface_attr->cap.get.max_bcopy      = 0;
    iface_attr->cap.get.max_zcopy      = 0;

    iface_attr->cap.am.max_short       = 0;
    iface_attr->cap.am.max_bcopy       = 0;
    iface_attr->cap.am.max_zcopy       = 0;
    iface_attr->cap.am.max_hdr         = 0;

    iface_attr->completion_priv_len    = 0; /* TBD */
    return UCS_OK;
}

static ucs_status_t uct_cuda_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
  ucs_snprintf_zero(pd_attr->name, sizeof(pd_attr->name), "%s",
                    "cuda");
  pd_attr->rkey_packed_size  = 0; /* TBD */
  pd_attr->cap.flags         = UCT_PD_FLAG_REG;
  pd_attr->cap.max_alloc     = 0;
  pd_attr->cap.max_reg       = ULONG_MAX;

  /* TODO make it configurable */
  pd_attr->alloc_methods.count = 1;
  pd_attr->alloc_methods.methods[0] = UCT_ALLOC_METHOD_HEAP;

  return UCS_OK;
}

static ucs_status_t uct_cuda_rkey_pack(uct_pd_h pd, uct_mem_h memh,
                                      void *rkey_buffer)
{
    return UCS_OK;
}

static void uct_cuda_rkey_release(uct_pd_h pd, const uct_rkey_bundle_t *rkey_ob)
{
  return;
}

ucs_status_t uct_cuda_rkey_unpack(uct_pd_h pd, const void *rkey_buffer,
                                  uct_rkey_bundle_t *rkey_ob)
{
    return UCS_OK;
}

uct_iface_ops_t uct_cuda_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_iface_t),
    .iface_get_address   = uct_cuda_iface_get_address,
    .iface_flush         = uct_cuda_iface_flush,
    .ep_get_address      = uct_cuda_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_cuda_ep_connect_to_ep,
    .iface_query         = uct_cuda_iface_query,
    .ep_put_short        = uct_cuda_ep_put_short,
    .ep_am_short         = uct_cuda_ep_am_short,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_cuda_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_ep_t),
};

static ucs_status_t uct_cuda_mem_reg(uct_pd_h pd, void *address, size_t length,
                                     uct_mem_h *memh_p)
{
    ucs_status_t rc;
    uct_mem_h * mem_hndl = NULL;
    mem_hndl = ucs_malloc(sizeof(void *), "cuda handle for test passing");
    if (NULL == mem_hndl) {
      ucs_error("Failed to allocate memory for gni_mem_handle_t");
      rc = UCS_ERR_NO_MEMORY;
      goto mem_err;
    }
    *memh_p = mem_hndl;
    return UCS_OK;
 mem_err:
    return rc;
}

static ucs_status_t uct_cuda_mem_dereg(uct_pd_h pd, uct_mem_h memh)
{
    ucs_free(memh);
    return UCS_OK;
}

uct_pd_ops_t uct_cuda_pd_ops = {
    .query        = uct_cuda_pd_query,
    .rkey_pack    = uct_cuda_rkey_pack,
    .rkey_unpack  = uct_cuda_rkey_unpack,
    .rkey_release = uct_cuda_rkey_release,
    .mem_reg      = uct_cuda_mem_reg,
    .mem_dereg    = uct_cuda_mem_dereg
};

static uct_pd_t uct_cuda_pd = {
    .ops = &uct_cuda_pd_ops
};

static UCS_CLASS_INIT_FUNC(uct_cuda_iface_t , uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_cuda_iface_ops, worker,
                              &uct_cuda_pd, tl_config UCS_STATS_ARG(NULL));

    if(strcmp(dev_name, UCT_CUDA_TL_NAME) != 0) {
        ucs_error("No device was found: %s", dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->pd.super.ops = &uct_cuda_pd_ops;
    self->super.super.pd   = &self->pd.super;

    self->config.max_put   = UCT_CUDA_MAX_SHORT_LENGTH;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_iface_t)
{
    /* tasks to tear down the domain */
}

UCS_CLASS_DEFINE(uct_cuda_iface_t, uct_base_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_iface_t, uct_iface_t, uct_worker_h,
                                 const char*, size_t, const uct_iface_config_t *);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_iface_t, uct_iface_t);

uct_tl_ops_t uct_cuda_tl_ops = {
    .query_resources     = uct_cuda_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_cuda_iface_t),
};
