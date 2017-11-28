/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "tcp.h"


static ucs_status_t uct_tcp_md_query(uct_md_h md, uct_md_attr_t *attr)
{
    attr->cap.flags         = 0;
    attr->cap.max_alloc     = 0;
    attr->cap.reg_mem_types = 0;
    attr->cap.mem_type      = 0;
    attr->cap.max_reg       = 0;
    attr->rkey_packed_size  = 0;
    attr->reg_cost.overhead = 0;
    attr->reg_cost.growth   = 0;
    memset(&attr->local_cpus, 0xff, sizeof(attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_tcp_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    return uct_single_md_resource(&uct_tcp_md, resources_p, num_resources_p);
}

static ucs_status_t uct_tcp_md_open(const char *md_name, const uct_md_config_t *md_config,
                                    uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close        = ucs_empty_function,
        .query        = uct_tcp_md_query,
        .mkey_pack    = ucs_empty_function_return_unsupported,
        .mem_reg      = ucs_empty_function_return_unsupported,
        .mem_dereg    = ucs_empty_function_return_unsupported,
        .is_mem_type_owned = (void *)ucs_empty_function_return_zero,
    };
    static uct_md_t md = {
        .ops          = &md_ops,
        .component    = &uct_tcp_md
    };

    *md_p = &md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_tcp_md, UCT_TCP_NAME,
                        uct_tcp_query_md_resources, uct_tcp_md_open, NULL,
                        ucs_empty_function_return_unsupported,
                        ucs_empty_function_return_success, "TCP_",
                        uct_md_config_table, uct_md_config_t);
