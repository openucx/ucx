/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "ucs/debug/memtrack.h"
#include "ucs/type/class.h"

#include "uct/tl/context.h"
#include "sysv_iface.h"
#include "sysv_context.h"

ucs_status_t uct_sysv_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p);

ucs_config_field_t uct_sysv_iface_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_sysv_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    {NULL}
};

ucs_status_t sysv_activate_domain(uct_sysv_context_t *sysv_ctx)
{
    return UCS_OK; /* No op */
}

ucs_status_t uct_sysv_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p)
{
    uct_sysv_context_t *sysv_ctx = ucs_component_get(context, sysv,
                                                   uct_sysv_context_t);

    uct_resource_desc_t *resources;

    if (sysv_ctx->num_devices == 0) {
        return UCS_ERR_NO_DEVICE;
    }

    /* sysv tl currently supports only a single device */

    /* Allocate resources array */
    resources = ucs_calloc(sysv_ctx->num_devices, sizeof(uct_resource_desc_t),
                           "resource desc");
    if (NULL == resources) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    uct_device_get_resource(&sysv_ctx->device,
                            &resources[0]);

    *num_resources_p = sysv_ctx->num_devices;
    *resources_p     = resources;

    return UCS_OK;
}

ucs_status_t uct_sysv_init(uct_context_h context)
{
    uct_sysv_context_t *sysv_ctx = ucs_component_get(context, sysv, 
                                                   uct_sysv_context_t);

    ucs_status_t status;

    sysv_ctx->activated = false;
    sysv_ctx->num_ifaces = 0;
    sysv_ctx->num_devices= 0;

    /* define one "device" for now. 
     * more complex logic for device(s) could be inserted here later on */

    /* create the single dummy device */

    status = uct_sysv_device_create(context, 1, &sysv_ctx->device);
    if (status != UCS_OK) {
        ucs_warn("Failed to initialize sysv device 0 (%s), ignoring it",
                  ucs_status_string(status));
        /* FIXME should I / howto bail? */
    } else {
        ++sysv_ctx->num_devices;
    }

    status = uct_register_tl(context, "sysv", uct_sysv_iface_config_table,
                             sizeof(uct_sysv_iface_config_t), "sysv_",
                             &uct_sysv_tl_ops);
    if (UCS_OK != status) {
        ucs_error("Failed to register context (%s), ignoring it",
                  ucs_status_string(status));
        /* FIXME should I / howto bail? */
    }

    sysv_ctx->activated = true;

    ucs_debug("Initialized sysv component with %d devices", 
               sysv_ctx->num_devices);
    ucs_debug("sysv context %p was activated", sysv_ctx);

    return UCS_OK;

}

void uct_sysv_cleanup(uct_context_t *context)
{
    /* no-op */
}
UCS_COMPONENT_DEFINE(uct_context_t, sysv, uct_sysv_init, uct_sysv_cleanup, 
                     sizeof(uct_sysv_context_t))

