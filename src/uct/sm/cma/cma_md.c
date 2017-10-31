/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#define _GNU_SOURCE
#include <sys/uio.h>
#include "cma_md.h"

uct_md_component_t uct_cma_md_component;

static ucs_status_t uct_cma_query_md_resources(uct_md_resource_desc_t **resources_p,
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

    return uct_single_md_resource(&uct_cma_md_component,
                                  resources_p,
                                  num_resources_p);
}

static ucs_status_t uct_cma_mem_reg(uct_md_h md, void *address, size_t length,
                                    unsigned flags, uct_mem_h *memh_p)
{
    /* For testing we have to make sure that
     * memh_h != UCT_MEM_HANDLE_NULL
     * otherwise gtest is not happy */
    UCS_STATIC_ASSERT((uint64_t)0xdeadbeef != (uint64_t)UCT_MEM_HANDLE_NULL);
    *memh_p = (void *) 0xdeadbeef;
    return UCS_OK;
}

static ucs_status_t uct_cma_md_open(const char *md_name, const uct_md_config_t *md_config,
                                    uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close        = (void*)ucs_empty_function,
        .query        = uct_cma_md_query,
        .mem_alloc    = (void*)ucs_empty_function_return_success,
        .mem_free     = (void*)ucs_empty_function_return_success,
        .mkey_pack    = (void*)ucs_empty_function_return_success,
        .mem_reg      = uct_cma_mem_reg,
        .mem_dereg    = (void*)ucs_empty_function_return_success,
        .is_mem_type_owned = (void *)ucs_empty_function_return_zero,
    };
    static uct_md_t md = {
        .ops          = &md_ops,
        .component    = &uct_cma_md_component
    };

    *md_p = &md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_cma_md_component, "cma",
                        uct_cma_query_md_resources, uct_cma_md_open, NULL,
                        uct_md_stub_rkey_unpack,
                        ucs_empty_function_return_success, "CMA_",
                        uct_md_config_table, uct_md_config_t)

ucs_status_t uct_cma_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->rkey_packed_size  = 0;
    md_attr->cap.flags         = UCT_MD_FLAG_REG;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_HOST);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_HOST;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;
    md_attr->reg_cost.overhead = 9e-9;
    md_attr->reg_cost.growth   = 0;

    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}
