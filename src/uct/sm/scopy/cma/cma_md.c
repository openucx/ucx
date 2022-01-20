/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#include "cma_md.h"

#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <sys/prctl.h>
#include <sys/uio.h>
#include <string.h>

#if HAVE_SYS_CAPABILITY_H
#  include <sys/capability.h>
#endif


typedef struct uct_cma_md {
    struct uct_md       super;
    uint64_t            extra_caps;
} uct_cma_md_t;

typedef struct uct_cma_md_config {
    uct_md_config_t     super;
    int                 mem_invalidate;
} uct_cma_md_config_t;

static ucs_config_field_t uct_cma_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_cma_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"MEMORY_INVALIDATE", "n", "Expose memory invalidate support capability.\n"
     "Note: this capability is not really supported yet. This variable will\n"
     "be deprecated, when memory invalidation support is implemented.",
     ucs_offsetof(uct_cma_md_config_t, mem_invalidate), UCS_CONFIG_TYPE_BOOL},

    {NULL}
};

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
                  "process_vm_writev delivered %zd instead of %zu: %m",
                  delivered, sizeof(test_dst));
        return 0;
    }

    return 1;
}

static ucs_status_t
uct_cma_query_md_resources(uct_component_t *component,
                           uct_md_resource_desc_t **resources_p,
                           unsigned *num_resources_p)
{
    if (uct_cma_test_writev() && uct_cma_test_ptrace_scope()) {
        return uct_md_query_single_md_resource(component, resources_p,
                                               num_resources_p);
    } else {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
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

static ucs_status_t uct_cma_mem_dereg(uct_md_h uct_md,
                                      const uct_md_mem_dereg_params_t *params)
{
    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    ucs_assert(params->memh == (void*)0xdeadbeef);

    return UCS_OK;
}

static void uct_cma_md_close(uct_md_h md)
{
    ucs_free(md);
}

static ucs_status_t
uct_cma_md_open(uct_component_t *component, const char *md_name,
                const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    const uct_cma_md_config_t *md_config = ucs_derived_of(uct_md_config,
                                                          uct_cma_md_config_t);
    static uct_md_ops_t md_ops = {
        .close                  = uct_cma_md_close,
        .query                  = uct_cma_md_query,
        .mem_alloc              = (uct_md_mem_alloc_func_t)ucs_empty_function_return_success,
        .mem_free               = (uct_md_mem_free_func_t)ucs_empty_function_return_success,
        .mkey_pack              = (uct_md_mkey_pack_func_t)ucs_empty_function_return_success,
        .mem_reg                = uct_cma_mem_reg,
        .mem_dereg              = uct_cma_mem_dereg,
        .is_sockaddr_accessible = ucs_empty_function_return_zero_int,
        .detect_memory_type     = ucs_empty_function_return_unsupported,
    };
    uct_cma_md_t *cma_md;

    cma_md = ucs_malloc(sizeof(uct_cma_md_t), "uct_cma_md_t");
    if (cma_md == NULL) {
        ucs_error("Failed to allocate memory for uct_cma_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    cma_md->super.ops       = &md_ops;
    cma_md->super.component = &uct_cma_component;
    cma_md->extra_caps      = (md_config->mem_invalidate == UCS_YES) ?
                              UCT_MD_FLAG_INVALIDATE : 0ul;
    *md_p                   = &cma_md->super;

    return UCS_OK;
}

ucs_status_t uct_cma_md_query(uct_md_h uct_md, uct_md_attr_t *md_attr)
{
    uct_cma_md_t *md = ucs_derived_of(uct_md, uct_cma_md_t);

    md_attr->rkey_packed_size     = 0;
    md_attr->cap.flags            = UCT_MD_FLAG_REG | md->extra_caps;
    md_attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.alloc_mem_types  = 0;
    md_attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.detect_mem_types = 0;
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = ULONG_MAX;
    md_attr->reg_cost             = ucs_linear_func_make(9e-9, 0);

    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

uct_component_t uct_cma_component = {
    .query_md_resources = uct_cma_query_md_resources,
    .md_open            = uct_cma_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_md_stub_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_success,
    .name               = "cma",
    .md_config          = {
        .name           = "CMA memory domain",
        .prefix         = "CMA_",
        .table          = uct_cma_md_config_table,
        .size           = sizeof(uct_cma_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_cma_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
