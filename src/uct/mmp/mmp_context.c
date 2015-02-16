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

/* FIXME here early on, it seem like a lot of this is the same as the ugni tl,
 * so should there be more of an abstraction for context handling? */

/* FIXME what are the relevant fields for this table?
ucs_config_field_t uct_mmp_iface_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_mmp_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    UCT_IFACE_MPOOL_CONFIG_FIELDS("FMA", -1, "fma",
                                  ucs_offsetof(uct_mmp_iface_config_t, mpool),
                                  "\nAttention: Setting this param with value
                                  != -1 is a dangerous thing\n" "and could
                                  cause deadlock or performance degradation."),

    {NULL}
};
*/

/* called by uct_activate_domain */
static ucs_status_t get_cookie(uint32_t *cookie)
{
    /* FIXME what is the mmp cookie? */
}

/* called by uct_activate_domain */
static ucs_status_t get_ptag(uint8_t *ptag)
{
    /* FIXME what is the mmp ptag? */
}

/* not called from the UCT library */
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

    /* FIXME does mmp_ctx need to be freed/cleared/destroyed? We ask for another
     * one in uct_mmp_init() below */

    return UCS_OK;
}

/* gets called by mmp_activate_iface() first thing */
ucs_status_t mmp_activate_domain(uct_mmp_context_t *mmp_ctx)
{
    int rc;
    
    if(mmp_ctx->activated) {
        return UCS_OK;
    }

    /* FIXME get number of ranks */
    /* FIXME get my rank id */

    rc = get_ptag(&mmp_ctx->ptag);
    if (UCS_OK != rc) {
        ucs_error("get_ptag failed, Error status: %d", rc);
        return rc;
    }
    ucs_debug("MMP ptag %d", mmp_ctx->ptag);

    rc = get_cookie(&mmp_ctx->cookie);
    if (UCS_OK != rc) {
        ucs_error("get_cookie failed, Error status: %d", rc);
        return rc;
    }
    ucs_debug("MMP cookie %d", mmp_ctx->cookie);

    /* Context and domain is activated */
    mmp_ctx->activated = true;
    ucs_debug("MMP context %p was activated", mmp_ctx);
    return UCS_OK;
}

/* not called from the UCT library */
ucs_status_t uct_mmp_init(uct_context_h context)
{
    uct_mmp_context_t *mmp_ctx = ucs_component_get(context, mmp, 
                                                     uct_mmp_context_t);

    ucs_status_t status;
    int i, num_devices;
    int *dev_ids = NULL;

    mmp_ctx->activated = false;
    mmp_ctx->num_ifaces = 0;

    /* FIXME decide how many devices are available */

    if (0 == mmp_ctx->num_devices) {
        ucs_debug("MMP No device found");
        mmp_ctx->devices = NULL;
        mmp_ctx->activated = 0;
        return UCS_OK;
    }

    /* Allocate array for devices */
    mmp_ctx->devices = ucs_calloc(mmp_ctx->num_devices,
                                   sizeof(uct_mmp_device_t), "mmp device");
    if (NULL == mmp_ctx->devices) {
        ucs_error("Failed to allocate memory");
        status = UCS_ERR_NO_MEMORY;
        goto err_zero;
    }

    dev_ids = ucs_calloc(mmp_ctx->num_devices, sizeof(int), "mmp device ids");
    if (NULL == dev_ids) {
        ucs_error("Failed to allocate memory");
        status = UCS_ERR_NO_MEMORY;
        goto err_ids;
    }

    /* FIXME get local device ids */

    num_devices = 0;
    for (i = 0; i < mmp_ctx->num_devices; i++) {
        status = uct_mmp_device_create(context, dev_ids[i], 
                                        &mmp_ctx->devices[i]);
        if (status != UCS_OK) {
            ucs_warn("Failed to initialize mmp device %d (%s), ignoring it",
                     i, ucs_status_string(status));
        } else {
            ++num_devices;
        }
    }

    if (num_devices != mmp_ctx->num_devices) {
        ucs_warn("Error in detection devices");
        status = UCS_ERR_NO_DEVICE;
        goto err_dev;
    }

    ucs_debug("Initialized MMP component with %d devices", 
               mmp_ctx->num_devices);

    status = uct_register_tl(context, "mmp", uct_mmp_iface_config_table,
                             sizeof(uct_mmp_iface_config_t), "mmp_",
                             &uct_mmp_tl_ops);
    if (UCS_OK != status) {
        ucs_error("Failed to register context (%s), ignoring it",
                  ucs_status_string(status));
        goto err_dev;
    }

    ucs_free(dev_ids);

    return UCS_OK;

err_dev:
    ucs_free(dev_ids);
err_ids:
    ucs_free(mmp_ctx->devices);
err_zero:
    return status;
}

void uct_mmp_cleanup(uct_context_t *context)
{
    uct_mmp_context_t *mmp_ctx = ucs_component_get(context, mmp, 
                                                     uct_mmp_context_t);
    int i;

    for (i = 0; i < mmp_ctx->num_devices; ++i) {
        uct_mmp_device_destroy(&mmp_ctx->devices[i]);
    }
    ucs_free(mmp_ctx->devices);
}
UCS_COMPONENT_DEFINE(uct_context_t, mmp, uct_mmp_init, uct_mmp_cleanup, 
                     sizeof(uct_mmp_context_t))

uct_mmp_device_t * uct_mmp_device_by_name(uct_mmp_context_t *mmp_ctx,
                                            const char *dev_name)
{
    uct_mmp_device_t *dev;
    unsigned dev_index;

    if ((NULL == dev_name) || (NULL == mmp_ctx)) {
        ucs_error("Bad parameter. Device name and/or context are set to NULL");
        return NULL;
    }

    for (dev_index = 0; dev_index < mmp_ctx->num_devices; ++dev_index) {
        dev = &mmp_ctx->devices[dev_index];
        if (strlen(dev_name) == strlen(dev->fname) &&
                0 == strncmp(dev_name, dev->fname, strlen(dev->fname))) {
            ucs_info("Device found: %s", dev_name);
            return dev;
        }
    }

    /* Device not found */
    ucs_error("Cannot find: %s", dev_name);
    return NULL;
}
