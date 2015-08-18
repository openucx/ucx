/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#define _GNU_SOURCE
#include <sys/uio.h>
#include "cma_pd.h"

uct_pd_component_t uct_cma_pd_component;

static ucs_status_t uct_cma_query_pd_resources(uct_pd_resource_desc_t **resources_p,
                                               unsigned *num_resources_p)
{
    ssize_t delivered;
    uint64_t test_dst = 0;
    uint64_t test_src = 0;
    pid_t dst = getpid();
    struct iovec local_iov  = {.iov_base = &test_src,
                               .iov_len = sizeof(test_src)};
    struct iovec remote_iov = {.iov_base = &test_dst,
                               .iov_len = sizeof(test_dst)};

    delivered = process_vm_writev(dst, &local_iov, 1, &remote_iov, 1, 0);
    if (ucs_unlikely(delivered != sizeof(test_dst))) {
        ucs_debug("CMA is disabled:"
                  "process_vm_writev delivered %zu instead of %zu",
                   delivered, sizeof(test_dst));
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    return uct_single_pd_resource(&uct_cma_pd_component,
                                  resources_p,
                                  num_resources_p);
}

static ucs_status_t uct_cma_mem_reg(uct_pd_h pd, void *address, size_t length,
                        uct_mem_h *memh_p)
{
    /* For testing we have to make sure that
     * memh_h != UCT_INVALID_MEM_HANDLE
     * otherwise gtest is not happy */
    *memh_p = (void *) 0xdeadbeef;
    return UCS_OK;
}

static ucs_status_t uct_cma_pd_open(const char *pd_name, uct_pd_h *pd_p)
{
    static uct_pd_ops_t pd_ops = {
        .close        = (void*)ucs_empty_function,
        .query        = uct_cma_pd_query,
        .mem_alloc    = (void*)ucs_empty_function_return_success,
        .mem_free     = (void*)ucs_empty_function_return_success,
        .mkey_pack    = (void*)ucs_empty_function_return_success,
        .mem_reg      = uct_cma_mem_reg,
        .mem_dereg    = (void*)ucs_empty_function_return_success
    };
    static uct_pd_t pd = {
        .ops          = &pd_ops,
        .component    = &uct_cma_pd_component
    };

    *pd_p = &pd;
    return UCS_OK;
}

UCT_PD_COMPONENT_DEFINE(uct_cma_pd_component, "cma",
        uct_cma_query_pd_resources, uct_cma_pd_open, NULL,
        0, ucs_empty_function_return_success,
        ucs_empty_function_return_success)

ucs_status_t uct_cma_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    pd_attr->rkey_packed_size  = 0;
    pd_attr->cap.flags         = UCT_PD_FLAG_REG;
    pd_attr->cap.max_alloc     = 0;
    pd_attr->cap.max_reg       = ULONG_MAX;

    memset(&pd_attr->local_cpus, 0xff, sizeof(pd_attr->local_cpus));
    return UCS_OK;
}
