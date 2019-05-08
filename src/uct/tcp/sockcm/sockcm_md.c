/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "sockcm_md.h"

#define UCT_SOCKCM_MD_PREFIX              "sockcm"

static ucs_config_field_t uct_sockcm_md_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_sockcm_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},
  {NULL}
};

static void uct_sockcm_md_close(uct_md_h md);

static uct_md_ops_t uct_sockcm_md_ops = {
    .close                  = uct_sockcm_md_close,
    .query                  = uct_sockcm_md_query,
    .is_sockaddr_accessible = uct_sockcm_is_sockaddr_accessible,
    .is_mem_type_owned      = (void *)ucs_empty_function_return_zero,
};

static void uct_sockcm_md_close(uct_md_h md)
{
    uct_sockcm_md_t *sockcm_md = ucs_derived_of(md, uct_sockcm_md_t);
    ucs_free(sockcm_md);
}

ucs_status_t uct_sockcm_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_SOCKADDR;
    md_attr->cap.reg_mem_types = 0;
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_HOST;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = 0;
    md_attr->rkey_packed_size  = 0;
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

int uct_sockcm_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                      uct_sockaddr_accessibility_t mode)
{
    return 0;
}

static ucs_status_t uct_sockcm_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                  unsigned *num_resources_p)
{
    return uct_single_md_resource(&uct_sockcm_mdc, resources_p, num_resources_p);
}

static ucs_status_t
uct_sockcm_md_open(const char *md_name, const uct_md_config_t *uct_md_config,
                   uct_md_h *md_p)
{
    uct_sockcm_md_t *md;

    md = ucs_malloc(sizeof(*md), "sockcm_md");
    if (md == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops            = &uct_sockcm_md_ops;
    md->super.component      = &uct_sockcm_mdc;

    *md_p = &md->super;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_sockcm_mdc, UCT_SOCKCM_MD_PREFIX,
                        uct_sockcm_query_md_resources, uct_sockcm_md_open, NULL,
                        ucs_empty_function_return_unsupported,
                        (void*)ucs_empty_function_return_success,
                        "SOCKCM_", uct_sockcm_md_config_table, uct_sockcm_md_config_t);
