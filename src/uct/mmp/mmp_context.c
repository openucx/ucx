/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "ucs/debug/memtrack.h"
#include "ucs/type/class.h"

#include "uct/tl/context.h"
#include "mmp_iface.h"
#include "mmp_context.h"

ucs_status_t uct_mmp_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p);

ucs_config_field_t uct_mmp_iface_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_mmp_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    {NULL}
};

ucs_status_t uct_mmp_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p)
{
    uct_mmp_context_t *mmp_ctx = ucs_component_get(context, mmp,
                                                   uct_mmp_context_t);

    uct_resource_desc_t *resources;
    unsigned dev_index;

    if (mmp_ctx->num_devices == 0) {
        return UCS_ERR_NO_DEVICE;
    }

    /* mmp currently supports only a single device.  
     * leaving it general here for multiple devices. */

    /* Allocate resources array */
    resources = ucs_calloc(mmp_ctx->num_devices, sizeof(uct_resource_desc_t),
                           "resource desc");
    if (NULL == resources) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    for (dev_index = 0; dev_index < mmp_ctx->num_devices; ++dev_index) {
        uct_device_get_resource(&mmp_ctx->devices[dev_index],
                                &resources[dev_index]);
    }

    *num_resources_p = mmp_ctx->num_devices;
    *resources_p     = resources;

    return UCS_OK;
}

ucs_status_t uct_mmp_init(uct_context_h context)
{
    uct_mmp_context_t *mmp_ctx = ucs_component_get(context, mmp, 
                                                     uct_mmp_context_t);

    ucs_status_t status;

    mmp_ctx->activated = false;
    mmp_ctx->num_ifaces = 0;

    /* define one "device" for now. 
     * more complex logic for device(s) could be inserted here later on */

    /* create the single dummy device */

    status = uct_ugni_device_create(context, 1, &ugni_ctx->devices[i]);
    if (status != UCS_OK) {
        ucs_warn("Failed to initialize ugni device %d (%s), ignoring it",
                 i, ucs_status_string(status));
        /* FIXME howto bail? */
    } else {
        ++mmp_ctx->num_devices;
    }

    status = uct_register_tl(context, "mmp", uct_mmp_iface_config_table,
                             sizeof(uct_mmp_iface_config_t), "mmp_",
                             &uct_mmp_tl_ops);
    if (UCS_OK != status) {
        ucs_error("Failed to register context (%s), ignoring it",
                  ucs_status_string(status));
        /* FIXME howto bail? */
    }

    mmp_ctx->activated = true;

    ucs_debug("Initialized MMP component with %d devices", 
               mmp_ctx->num_devices);
    ucs_debug("MMP context %p was activated", mmp_ctx);

    return UCS_OK;

}

void uct_mmp_cleanup(uct_context_t *context)
{
    /* no-op */
}
UCS_COMPONENT_DEFINE(uct_context_t, mmp, uct_mmp_init, uct_mmp_cleanup, 
                     sizeof(uct_mmp_context_t))

