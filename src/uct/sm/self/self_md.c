/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "self_md.h"


static ucs_status_t uct_self_md_query(uct_md_h md, uct_md_attr_t *attr)
{
    /* Dummy memory registration provided. No real memory handling exists */
    attr->cap.flags         = UCT_MD_FLAG_REG |
                              UCT_MD_FLAG_NEED_RKEY; /* TODO ignore rkey in rma/amo ops */
    attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_HOST);
    attr->cap.mem_type      = UCT_MD_MEM_TYPE_HOST;
    attr->cap.max_alloc     = 0;
    attr->cap.max_reg       = ULONG_MAX;
    attr->rkey_packed_size  = 0; /* uct_md_query adds UCT_MD_COMPONENT_NAME_MAX to this */
    attr->reg_cost.overhead = 0;
    attr->reg_cost.growth   = 0;
    memset(&attr->local_cpus, 0xff, sizeof(attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_self_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    return uct_single_md_resource(&uct_self_md, resources_p, num_resources_p);
}

static ucs_status_t uct_self_mem_reg(uct_md_h md, void *address, size_t length,
                                     unsigned flags, uct_mem_h *memh_p)
{
    /* We have to emulate memory registration. Return dummy pointer */
    *memh_p = (void *) 0xdeadbeef;
    return UCS_OK;
}

static ucs_status_t uct_self_md_open(const char *md_name, const uct_md_config_t *md_config,
                                     uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close        = (void*)ucs_empty_function,
        .query        = uct_self_md_query,
        .mkey_pack    = ucs_empty_function_return_success,
        .mem_reg      = uct_self_mem_reg,
        .mem_dereg    = ucs_empty_function_return_success,
        .is_mem_type_owned = (void *)ucs_empty_function_return_zero,
    };
    static uct_md_t md = {
        .ops          = &md_ops,
        .component    = &uct_self_md
    };

    *md_p = &md;
    return UCS_OK;
}
/**
 * Pseudo stub function for the key unpacking
 * Need rkey == 0 due to work with same process to reuse uct_base_[put|get|atomic]*
 */
static ucs_status_t uct_self_md_rkey_unpack(uct_md_component_t *mdc,
                                            const void *rkey_buffer, uct_rkey_t *rkey_p,
                                            void **handle_p)
{
    *rkey_p   = 0;
    *handle_p = NULL;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_self_md, UCT_SELF_NAME,
                        uct_self_query_md_resources, uct_self_md_open, NULL,
                        uct_self_md_rkey_unpack,
                        ucs_empty_function_return_success, "SELF_",
                        uct_md_config_table, uct_md_config_t);

