/**
 * Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * $COPYRIGHT$
 * $HEADER$
 */

#include <pmi.h>

#include "ucs/debug/memtrack.h"
#include "ucs/type/class.h"

#include "uct/tl/context.h"
#include "ugni_iface.h"
#include "ugni_context.h"

ucs_status_t uct_ugni_query_resources(uct_context_h context,
        uct_resource_desc_t **resources_p,
        unsigned *num_resources_p);

ucs_config_field_t uct_ugni_iface_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_ugni_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    {NULL}
};

static ucs_status_t get_cookie(uint32_t *cookie)
{
    char           *cookie_str;
    char           *cookie_token;

    cookie_str = getenv("PMI_GNI_COOKIE");
    if (NULL == cookie_str) {
        ucs_error("getenv PMI_GNI_COOKIE failed");
        return UCS_ERR_IO_ERROR;
    }

    cookie_token = strtok(cookie_str, ":");
    if (NULL == cookie_token) {
        ucs_error("Failed to read PMI_GNI_COOKIE token");
        return UCS_ERR_IO_ERROR;
    }

    *cookie = (uint32_t) atoi(cookie_token);
    return UCS_OK;
}

static ucs_status_t get_ptag(uint8_t *ptag)
{
    char           *ptag_str;
    char           *ptag_token;

    ptag_str = getenv("PMI_GNI_PTAG");
    if (NULL == ptag_str) {
        ucs_error("getenv PMI_GNI_PTAG failed");
        return UCS_ERR_IO_ERROR;
    }

    ptag_token = strtok(ptag_str, ":");
    if (NULL == ptag_token) {
        ucs_error("Failed to read PMI_GNI_PTAG token");
        return UCS_ERR_IO_ERROR;
    }

    *ptag = (uint8_t) atoi(ptag_token);
    return UCS_OK;
}

ucs_status_t uct_ugni_query_resources(uct_context_h context,
        uct_resource_desc_t **resources_p,
        unsigned *num_resources_p)
{
    uct_ugni_context_t *ugni_ctx = ucs_component_get(context, ugni, uct_ugni_context_t);
    uct_resource_desc_t *resources;
    unsigned dev_index;

    /* Allocate resources array */
    resources = ucs_calloc(ugni_ctx->num_devices, sizeof(uct_resource_desc_t),
                           "resource desc");
    if (NULL == resources) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    for (dev_index = 0; dev_index < ugni_ctx->num_devices; ++dev_index) {
        uct_device_get_resource(&ugni_ctx->devices[dev_index],
                                &resources[dev_index]);
    }

    *num_resources_p = ugni_ctx->num_devices;
    *resources_p     = resources;

    return UCS_OK;
}

ucs_status_t ugni_activate_domain(uct_context_h context)
{
    uct_ugni_context_t *ugni_ctx = ucs_component_get(context, ugni, uct_ugni_context_t);
    int modes,
        spawned,
        rc;
    gni_return_t ugni_rc = GNI_RC_SUCCESS;

    /* Fetch information from Cray's PMI */
    rc = PMI_Init(&spawned);
    if (PMI_SUCCESS != rc) {
        ucs_error("PMI_Init failed, Error status: %d", rc);
        return UCS_ERR_IO_ERROR;
    }

    rc = PMI_Get_size(&ugni_ctx->pmi_num_of_ranks);
    if (PMI_SUCCESS != rc) {
        ucs_error("PMI_Get_size failed, Error status: %d", rc);
        return UCS_ERR_IO_ERROR;
    }

    rc = PMI_Get_rank(&ugni_ctx->pmi_rank_id);
    if (PMI_SUCCESS != rc) {
        ucs_error("PMI_Get_rank failed, Error status: %d", rc);
        return UCS_ERR_IO_ERROR;
    }

    rc = get_ptag(&ugni_ctx->ptag);
    if (UCS_OK != rc) {
        ucs_error("get_ptag failed, Error status: %d", rc);
        return rc;
    }

    rc = get_cookie(&ugni_ctx->cookie);
    if (UCS_OK != rc) {
        ucs_error("get_cookie failed, Error status: %d", rc);
        return rc;
    }

    ugni_ctx->id = getpid(); /* TBD */
    modes = GNI_CDM_MODE_FORK_FULLCOPY | GNI_CDM_MODE_CACHED_AMO_ENABLED |
            GNI_CDM_MODE_ERR_NO_KILL | GNI_CDM_MODE_FAST_DATAGRAM_POLL;
    ugni_rc = GNI_CdmCreate(ugni_ctx->id, ugni_ctx->ptag, ugni_ctx->cookie,
                            modes, &ugni_ctx->cdm_handle);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    /* Context and domain is activated */
    ugni_ctx->activated = true;
    return UCS_OK;
}

ucs_status_t uct_ugni_init(uct_context_h context)
{
    uct_ugni_context_t *ugni_ctx = ucs_component_get(context, ugni, uct_ugni_context_t);
    ucs_status_t status;
    int i, num_devices;
    int *dev_ids = NULL;
    gni_return_t ugni_rc = GNI_RC_SUCCESS;

    /* The code is designed to support
     * more than single device */
    ugni_rc = GNI_GetNumLocalDevices(&ugni_ctx->num_devices);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_GetNumLocalDevices failed, Error status: %s %d",
                gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_NO_DEVICE;
        goto err_zero;
    }

    if (0 == ugni_ctx->num_devices) {
        ucs_warn("UGNI No device found");
        status = UCS_ERR_NO_DEVICE;
        goto err_zero;
    }

    /* Allocate array for devices */
    ugni_ctx->devices = ucs_calloc(ugni_ctx->num_devices,
                                   sizeof(uct_ugni_device_t), "ugni device");
    if (NULL == ugni_ctx->devices) {
        ucs_error("Failed to allocate memory");
        status = UCS_ERR_NO_MEMORY;
        goto err_zero;
    }

    dev_ids = ucs_calloc(ugni_ctx->num_devices, sizeof(int), "ugni device ids");
    if (NULL == dev_ids) {
        ucs_error("Failed to allocate memory");
        status = UCS_ERR_NO_MEMORY;
        goto err_ids;
    }

    ugni_rc = GNI_GetLocalDeviceIds(ugni_ctx->num_devices, dev_ids);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_GetLocalDeviceIds failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_NO_DEVICE;
        goto err_dev;
    }

    num_devices = 0;
    for (i = 0; i < ugni_ctx->num_devices; i++) {
        status = uct_ugni_device_create(dev_ids[i], &ugni_ctx->devices[i]);
        if (status != UCS_OK) {
            ucs_warn("Failed to initialize ugni device %d (%s), ignoring it",
                     i, ucs_status_string(status));
        } else {
            ++num_devices;
        }
    }

    if (num_devices != ugni_ctx->num_devices) {
        ucs_warn("Error in detection devices");
        status = UCS_ERR_NO_DEVICE;
        goto err_dev;
    }

    ucs_debug("Initialized UGNI component with %d devices", ugni_ctx->num_devices);

    status = uct_register_tl(context, "ugni", uct_ugni_iface_config_table,
                             sizeof(uct_ugni_iface_config_t), "UGNI_",
                             &uct_ugni_tl_ops);
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
    ucs_free(ugni_ctx->devices);
err_zero:
    return status;
}

void uct_ugni_cleanup(uct_context_t *context)
{
    uct_ugni_context_t *ugni_ctx = ucs_component_get(context, ugni, uct_ugni_context_t);
    gni_return_t ugni_rc;
    int i;

    for (i = 0; i < ugni_ctx->num_devices; ++i) {
        uct_ugni_device_destroy(&ugni_ctx->devices[i]);
    }
    ucs_free(ugni_ctx->devices);

    if (ugni_ctx->activated) {
        ugni_rc = GNI_CdmDestroy(ugni_ctx->cdm_handle);
        if (GNI_RC_SUCCESS != ugni_rc) {
            ucs_warn("GNI_CdmDestroy error status: %s (%d)",
                     gni_err_str[ugni_rc], ugni_rc);
        }
    }
}
UCS_COMPONENT_DEFINE(uct_context_t, ugni, uct_ugni_init, uct_ugni_cleanup, sizeof(uct_ugni_context_t))

uct_ugni_device_t * uct_ugni_device_by_name(uct_ugni_context_t *ugni_ctx,
                                            const char *dev_name)
{
    uct_ugni_device_t *dev;
    unsigned dev_index;

    if ((NULL == dev_name) || (NULL == ugni_ctx)) {
        ucs_error("Bad parameter. Device name and/or context are set to NULL");
        return NULL;
    }

    for (dev_index = 0; dev_index < ugni_ctx->num_devices; ++dev_index) {
        dev = &ugni_ctx->devices[dev_index];
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
