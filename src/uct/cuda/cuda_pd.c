/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "cuda_pd.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>


static ucs_status_t uct_cuda_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    pd_attr->cap.flags         = UCT_PD_FLAG_REG;
    pd_attr->cap.max_alloc     = 0;
    pd_attr->cap.max_reg       = ULONG_MAX;
    memset(&pd_attr->local_cpus, 0xff, sizeof(pd_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_cuda_mkey_pack(uct_pd_h pd, uct_mem_h memh,
                                      void *rkey_buffer)
{
    return UCS_OK;
}

static ucs_status_t uct_cuda_rkey_unpack(uct_pd_component_t *pdc,
                                         const void *rkey_buffer, uct_rkey_t *rkey_p,
                                         void **handle_p)
{
    *rkey_p   = 0xdeadbeef;
    *handle_p = NULL;
    return UCS_OK;
}

static ucs_status_t uct_cuda_rkey_release(uct_pd_component_t *pdc, uct_rkey_t rkey,
                                          void *handle)
{
    return UCS_OK;
}

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

static ucs_status_t uct_cuda_query_pd_resources(uct_pd_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    return uct_single_pd_resource(&uct_cuda_pd, resources_p, num_resources_p);
}

static ucs_status_t uct_cuda_pd_open(const char *pd_name, uct_pd_h *pd_p)
{
    static uct_pd_ops_t pd_ops = {
        .close        = (void*)ucs_empty_function,
        .query        = uct_cuda_pd_query,
        .mkey_pack    = uct_cuda_mkey_pack,
        .mem_reg      = uct_cuda_mem_reg,
        .mem_dereg    = uct_cuda_mem_dereg
    };
    static uct_pd_t pd = {
        .ops          = &pd_ops,
        .component    = &uct_cuda_pd
    };

    *pd_p = &pd;
    return UCS_OK;
}

UCT_PD_COMPONENT_DEFINE(uct_cuda_pd, UCT_CUDA_PD_NAME,
                        uct_cuda_query_pd_resources, uct_cuda_pd_open, NULL,
                        0, uct_cuda_rkey_unpack, uct_cuda_rkey_release);

