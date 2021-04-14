/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tcp.h"
#include "tcp_sockcm.h"
#include <uct/base/uct_md.h>


static ucs_status_t uct_tcp_md_query(uct_md_h md, uct_md_attr_t *attr)
{
    /* Dummy memory registration provided. No real memory handling exists */
    attr->cap.flags               = UCT_MD_FLAG_REG |
                                    UCT_MD_FLAG_NEED_RKEY; /* TODO ignore rkey in rma/amo ops */
    attr->cap.max_alloc           = 0;
    attr->cap.reg_mem_types       = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    attr->cap.alloc_mem_types     = 0;
    attr->cap.access_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    attr->cap.detect_mem_types    = 0;
    attr->cap.max_reg             = ULONG_MAX;
    attr->rkey_packed_size        = 0;
    attr->reg_cost                = ucs_linear_func_make(0, 0);
    memset(&attr->local_cpus, 0xff, sizeof(attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_tcp_md_mem_reg(uct_md_h md, void *address, size_t length,
                                       unsigned flags, uct_mem_h *memh_p)
{
    /* We have to emulate memory registration. Return dummy pointer */
    *memh_p = (void*)0xdeadbeef;
    return UCS_OK;
}

static ucs_status_t
uct_tcp_md_open(uct_component_t *component, const char *md_name,
                const uct_md_config_t *md_config, uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close              = ucs_empty_function,
        .query              = uct_tcp_md_query,
        .mkey_pack          = ucs_empty_function_return_success,
        .mem_reg            = uct_tcp_md_mem_reg,
        .mem_dereg          = ucs_empty_function_return_success,
        .detect_memory_type = ucs_empty_function_return_unsupported
    };
    static uct_md_t md = {
        .ops          = &md_ops,
        .component    = &uct_tcp_component
    };

    *md_p = &md;
    return UCS_OK;
}

static ucs_status_t uct_tcp_md_rkey_unpack(uct_component_t *component,
                                           const void *rkey_buffer,
                                           uct_rkey_t *rkey_p, void **handle_p)
{
    /**
     * Pseudo stub function for the key unpacking
     * Need rkey == 0 due to work with same process to reuse uct_base_[put|get|atomic]*
     */
    *rkey_p   = 0;
    *handle_p = NULL;
    return UCS_OK;
}

uct_component_t uct_tcp_component = {
    .query_md_resources = uct_md_query_single_md_resource,
    .md_open            = uct_tcp_md_open,
    .cm_open            = UCS_CLASS_NEW_FUNC_NAME(uct_tcp_sockcm_t),
    .rkey_unpack        = uct_tcp_md_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_success,
    .name               = UCT_TCP_NAME,
    .md_config          = UCT_MD_DEFAULT_CONFIG_INITIALIZER,
    .cm_config          = {
        .name           = "TCP-SOCKCM connection manager",
        .prefix         = "TCP_CM_",
        .table          = uct_tcp_sockcm_config_table,
        .size           = sizeof(uct_tcp_sockcm_config_t),
     },
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_tcp_component),
    .flags              = UCT_COMPONENT_FLAG_CM
};
UCT_COMPONENT_REGISTER(&uct_tcp_component)
