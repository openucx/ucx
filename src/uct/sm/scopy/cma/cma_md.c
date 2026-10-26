/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
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
#include <sys/uio.h>
#include <string.h>


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
    if (uct_cma_test_writev() && ucs_sys_ptrace_enable_peers()) {
        return uct_md_query_single_md_resource(component, resources_p,
                                               num_resources_p);
    } else {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }
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
        .close              = uct_cma_md_close,
        .query              = uct_cma_md_query,
        .mem_alloc          = (uct_md_mem_alloc_func_t)ucs_empty_function_return_unsupported,
        .mem_free           = (uct_md_mem_free_func_t)ucs_empty_function_return_unsupported,
        .mem_advise         = (uct_md_mem_advise_func_t)ucs_empty_function_return_unsupported,
        .mem_reg            = (uct_md_mem_reg_func_t)ucs_empty_function_return_unsupported,
        .mem_dereg          = (uct_md_mem_dereg_func_t)ucs_empty_function_return_unsupported,
        .mem_query          = (uct_md_mem_query_func_t)ucs_empty_function_return_unsupported,
        .mkey_pack          = (uct_md_mkey_pack_func_t)ucs_empty_function_return_unsupported,
        .mem_attach         = (uct_md_mem_attach_func_t)ucs_empty_function_return_unsupported,
        .detect_memory_type = (uct_md_detect_memory_type_func_t)ucs_empty_function_return_unsupported,
        .mem_elem_pack      = (uct_md_mem_elem_pack_func_t)ucs_empty_function_return_unsupported,
        .mem_elem_release   = (uct_md_mem_elem_release_func_t)ucs_empty_function
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
                              (UCT_MD_FLAG_INVALIDATE     |
                               UCT_MD_FLAG_INVALIDATE_RMA |
                               UCT_MD_FLAG_INVALIDATE_AMO) : 0ul;
    *md_p                   = &cma_md->super;

    return UCS_OK;
}

ucs_status_t uct_cma_md_query(uct_md_h uct_md, uct_md_attr_v2_t *md_attr)
{
    uct_cma_md_t *md = ucs_derived_of(uct_md, uct_cma_md_t);

    uct_md_base_md_query(md_attr);
    md_attr->flags            = md->extra_caps;
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    return UCS_OK;
}

uct_component_t uct_cma_component = {
    .query_md_resources = uct_cma_query_md_resources,
    .md_open            = uct_cma_md_open,
    .cm_open            = (uct_component_cm_open_func_t)ucs_empty_function_return_unsupported,
    .rkey_unpack        = (uct_component_rkey_unpack_func_t)ucs_empty_function_return_unsupported,
    .rkey_ptr           = (uct_component_rkey_ptr_func_t)ucs_empty_function_return_unsupported,
    .rkey_release       = (uct_component_rkey_release_func_t)ucs_empty_function_return_success,
    .rkey_compare       = (uct_component_rkey_compare_func_t)ucs_empty_function_return_unsupported,
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
