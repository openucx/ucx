/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "sockcm_md.h"

#define UCT_SOCKCM_NAME              "sockcm"

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
    .detect_memory_type     = ucs_empty_function_return_unsupported,
};

static void uct_sockcm_md_close(uct_md_h md)
{
    uct_sockcm_md_t *sockcm_md = ucs_derived_of(md, uct_sockcm_md_t);
    ucs_free(sockcm_md);
}

ucs_status_t uct_sockcm_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags            = UCT_MD_FLAG_SOCKADDR;
    md_attr->cap.reg_mem_types    = 0;
    md_attr->cap.access_mem_type  = UCS_MEMORY_TYPE_HOST;
    md_attr->cap.detect_mem_types = 0;
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = 0;
    md_attr->rkey_packed_size     = 0;
    md_attr->reg_cost.overhead    = 0;
    md_attr->reg_cost.growth      = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

int uct_sockcm_is_sockaddr_accessible(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                      uct_sockaddr_accessibility_t mode)
{
    return 0;
}

static ucs_status_t
uct_sockcm_md_open(uct_component_t *component, const char *md_name,
                   const uct_md_config_t *config, uct_md_h *md_p)
{
    uct_sockcm_md_t *md;

    md = ucs_malloc(sizeof(*md), "sockcm_md");
    if (md == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops            = &uct_sockcm_md_ops;
    md->super.component      = &uct_sockcm_component;

    /* cppcheck-suppress autoVariables */
    *md_p = &md->super;
    return UCS_OK;
}

uct_component_t uct_sockcm_component = {
    .query_md_resources = uct_md_query_single_md_resource,
    .md_open            = uct_sockcm_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = ucs_empty_function_return_unsupported,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_unsupported,
    .name               = UCT_SOCKCM_NAME,
    .md_config          = {
        .name           = "Sock-CM memory domain",
        .prefix         =  "SOCKCM_",
        .table          = uct_sockcm_md_config_table,
        .size           = sizeof(uct_sockcm_md_config_t),
    },
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_sockcm_component)
};
UCT_COMPONENT_REGISTER(&uct_sockcm_component)
