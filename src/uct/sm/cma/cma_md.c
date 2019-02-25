/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#define _GNU_SOURCE
#include "cma_md.h"

#include <ucs/sys/string.h>
#include <sys/prctl.h>
#include <sys/uio.h>
#include <string.h>

#if HAVE_SYS_CAPABILITY_H
#  include <sys/capability.h>
#endif


uct_md_component_t uct_cma_md_component;

static int uct_cma_test_ptrace_scope()
{
    static const char *ptrace_scope_file = "/proc/sys/kernel/yama/ptrace_scope";
    const char *extra_info_str;
    int cma_supported;
    char buffer[32];
    ssize_t nread;
    char *value;

    /* Check if ptrace_scope allows using CMA.
     * See https://www.kernel.org/doc/Documentation/security/Yama.txt
     */
    nread = ucs_read_file(buffer, sizeof(buffer) - 1, 1, "%s", ptrace_scope_file);
    if (nread < 0) {
        /* Cannot read file - assume that Yama security module is not enabled */
        ucs_debug("could not read '%s' - assuming Yama security is not enforced",
                  ptrace_scope_file);
        return 1;
    }

    ucs_assert(nread < sizeof(buffer));
    extra_info_str = "";
    cma_supported  = 0;
    buffer[nread]  = '\0';
    value          = ucs_strtrim(buffer);
    if(!strcmp(value, "0")) {
        /* ptrace scope 0 allow attaching within same UID */
        cma_supported = 1;
    } else if (!strcmp(value, "1")) {
        /* ptrace scope 1 allows attaching with explicit permission by prctl() */
#if HAVE_DECL_PR_SET_PTRACER
        int ret = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
        if (!ret) {
            extra_info_str = ", enabled PR_SET_PTRACER_ANY";
            cma_supported  = 1;
        } else {
            extra_info_str = " and prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY) failed";
        }
#else
        extra_info_str = " but no PR_SET_PTRACER";
#endif
    } else if (!strcmp(value, "2")) {
        /* ptrace scope 2 means only a process with CAP_SYS_PTRACE can attach */
#if HAVE_SYS_CAPABILITY_H
        ucs_status_t status;
        uint32_t ecap;

        status = ucs_sys_get_proc_cap(&ecap);
        UCS_STATIC_ASSERT(CAP_SYS_PTRACE < 32);
        if ((status == UCS_OK) && (ecap & CAP_SYS_PTRACE)) {
            extra_info_str = ", process has CAP_SYS_PTRACE";
            cma_supported = 1;
        } else
#endif
            extra_info_str = " but no CAP_SYS_PTRACE";
    } else {
        /* ptrace scope 3 means attach is completely disabled on the system */
    }

    /* coverity[result_independent_of_operands] */
    ucs_log(cma_supported ? UCS_LOG_LEVEL_TRACE : UCS_LOG_LEVEL_DEBUG,
            "ptrace_scope is %s%s, CMA is %ssupported",
            value, extra_info_str, cma_supported ? "" : "un");
    return cma_supported;
}

static int uct_cma_test_writev()
{
    uint64_t test_dst       = 0;
    uint64_t test_src       = 0;
    struct iovec local_iov  = {.iov_base = &test_src,
                               .iov_len = sizeof(test_src)};
    struct iovec remote_iov = {.iov_base = &test_dst,
                               .iov_len = sizeof(test_dst)};
    ssize_t delivered;

    delivered = process_vm_writev(getpid(), &local_iov, 1, &remote_iov, 1, 0);
    if (delivered != sizeof(test_dst)) {
        ucs_debug("CMA is disabled:"
                  "process_vm_writev delivered %zu instead of %zu",
                   delivered, sizeof(test_dst));
        return 0;
    }

    return 1;
}

static ucs_status_t uct_cma_query_md_resources(uct_md_resource_desc_t **resources_p,
                                               unsigned *num_resources_p)
{
    if (uct_cma_test_writev() && uct_cma_test_ptrace_scope()) {
        return uct_single_md_resource(&uct_cma_md_component,
                                      resources_p,
                                      num_resources_p);
    } else {
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }
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
                        uct_md_config_table, uct_md_config_t,
                        ucs_empty_function_return_unsupported)

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
