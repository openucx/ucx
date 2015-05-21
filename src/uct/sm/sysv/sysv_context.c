/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "sysv_context.h"

ucs_status_t uct_sysv_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p);

/* FIXME we need to figure out how to build this table in a hierarchical fashion
 * so that all shared memory transports can share the common parts with the base
 * class
 */
ucs_config_field_t uct_sysv_iface_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_sysv_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    {NULL}
};

ucs_status_t uct_sysv_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resource_p,
                                      unsigned *num_resources_p)
{
    ucs_status_t status;

    uct_resource_desc_t *resource = NULL;

    status = uct_sm_query_resources(&resource, UCT_SYSV_TL_NAME);
    if (UCS_OK != status) return status;

    /* can override resource->latency/bandwidth here if desired */

    *num_resources_p = 1;
    *resource_p     = resource;

    return UCS_OK;
}

ucs_status_t uct_sysv_init(uct_context_h context)
{
    ucs_status_t status;

    status = uct_register_tl(context, "sysv", uct_sysv_iface_config_table,
                             sizeof(uct_sysv_iface_config_t), "sysv_",
                             &uct_sysv_tl_ops);
    if (UCS_OK != status) {
        ucs_error("Failed to register context (%s)",
                  ucs_status_string(status));
        return status;
    }

    ucs_debug("Initialized sysv component");

    return UCS_OK;
}

void uct_sysv_cleanup(uct_context_t *context)
{
    /* no-op */
}

UCS_COMPONENT_DEFINE(uct_context_t, sysv, uct_sysv_init, uct_sysv_cleanup, 0)
