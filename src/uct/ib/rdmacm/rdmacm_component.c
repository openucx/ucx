/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2021.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rdmacm_cm.h"


static ucs_config_field_t uct_rdmacm_cm_config_table[] = {
    {"CM_", "", NULL,
     ucs_offsetof(uct_rdmacm_cm_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_cm_config_table)},

    {"SOURCE_ADDRESS", "",
     "If non-empty, specify the local source address (IPv4 or IPv6) to use \n"
     "when creating a client connection",
     ucs_offsetof(uct_rdmacm_cm_config_t, src_addr), UCS_CONFIG_TYPE_STRING},

    {"TIMEOUT", "10s",
     "Timeout for RDMA address and route resolve operations",
     ucs_offsetof(uct_rdmacm_cm_config_t, timeout), UCS_CONFIG_TYPE_TIME},

    {"RESERVED_QPN", "try",
     "Reserved qpn enable mode:\n"
     "  yes  - Always use reserved qpn, app fail if it's not supported\n"
     "  try  - Use reserved qpn if it's supported, otherwise use dummy qp\n"
     "  no   - Always use dummy qp",
     ucs_offsetof(uct_rdmacm_cm_config_t, reserved_qpn),
                  UCS_CONFIG_TYPE_TERNARY},

    {NULL}
};

static ucs_status_t
uct_rdmacm_query_md_resources(uct_component_t *component,
                              uct_md_resource_desc_t **resources_p,
                              unsigned *num_resources_p)
{
    *resources_p     = NULL;
    *num_resources_p = 0;
    return UCS_OK;
}

uct_component_t uct_rdmacm_component = {
    .query_md_resources = uct_rdmacm_query_md_resources,
    .md_open            = ucs_empty_function_return_unsupported,
    .cm_open            = UCS_CLASS_NEW_FUNC_NAME(uct_rdmacm_cm_t),
    .rkey_unpack        = ucs_empty_function_return_unsupported,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_success,
    .name               = "rdmacm",
    .md_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .cm_config          = {
        .name           = "RDMA-CM connection manager",
        .prefix         = "RDMA_CM_",
        .table          = uct_rdmacm_cm_config_table,
        .size           = sizeof(uct_rdmacm_cm_config_t),
    },
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_rdmacm_component),
    .flags              = UCT_COMPONENT_FLAG_CM,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};

UCS_F_CTOR void uct_rdmacm_init(void)
{
    uct_component_register(&uct_rdmacm_component);
}

UCS_F_DTOR void uct_rdmacm_cleanup(void)
{
    uct_component_unregister(&uct_rdmacm_component);
}
