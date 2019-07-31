/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tcp.h"

#include <uct/base/uct_md.h>


static ucs_status_t uct_tcp_md_query(uct_md_h md, uct_md_attr_t *attr)
{
    attr->cap.flags               = 0;
    attr->cap.max_alloc           = 0;
    attr->cap.reg_mem_types       = 0;
    attr->cap.access_mem_type     = UCS_MEMORY_TYPE_HOST;
    attr->cap.detect_mem_types    = 0;
    attr->cap.max_reg             = 0;
    attr->rkey_packed_size        = 0;
    attr->reg_cost.overhead       = 0;
    attr->reg_cost.growth         = 0;
    memset(&attr->local_cpus, 0xff, sizeof(attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t
uct_tcp_md_open(uct_component_t *component, const char *md_name,
                const uct_md_config_t *md_config, uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close              = ucs_empty_function,
        .query              = uct_tcp_md_query,
        .mkey_pack          = ucs_empty_function_return_unsupported,
        .mem_reg            = ucs_empty_function_return_unsupported,
        .mem_dereg          = ucs_empty_function_return_unsupported,
        .detect_memory_type = ucs_empty_function_return_unsupported,
    };
    static uct_md_t md = {
        .ops          = &md_ops,
        .component    = &uct_tcp_component
    };

    *md_p = &md;
    return UCS_OK;
}

uct_component_t uct_tcp_component = {
    .query_md_resources = uct_md_query_single_md_resource,
    .md_open            = uct_tcp_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = ucs_empty_function_return_unsupported,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_unsupported,
    .name               = UCT_TCP_NAME,
    .md_config          = UCT_MD_DEFAULT_CONFIG_INITIALIZER,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_tcp_component),
    .cap_flags          = 0
};
UCT_COMPONENT_REGISTER(&uct_tcp_component)
