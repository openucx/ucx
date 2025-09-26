/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ze_base.h"

#include <ucs/sys/module.h>

#include <pthread.h>

#define UCT_ZE_MAX_DEVICES 32

static struct {
    ze_driver_handle_t driver;
    ze_device_handle_t devices[UCT_ZE_MAX_DEVICES];
    int                num_devices;
} uct_ze_base_info;

ze_result_t uct_ze_base_init(void)
{
    static ucs_init_once_t init = UCS_INIT_ONCE_INITIALIZER;
    ze_result_t ret = ZE_RESULT_SUCCESS;
    unsigned count;

    UCS_INIT_ONCE(&init) {
        ret = zeInit(ZE_INIT_FLAG_GPU_ONLY);
        if (ret != ZE_RESULT_SUCCESS) {
            ucs_debug("failure to initialize ze library: 0x%x", ret);
            continue;
        }

        count = 1;
        ret   = zeDriverGet(&count, &uct_ze_base_info.driver);
        if (ret != ZE_RESULT_SUCCESS) {
            ucs_debug("failure to get ze driver: 0x%x", ret);
            continue;
        }

        count = UCT_ZE_MAX_DEVICES;
        ret   = zeDeviceGet(uct_ze_base_info.driver, &count,
                            uct_ze_base_info.devices);
        if (ret != ZE_RESULT_SUCCESS) {
            ucs_debug("failure to get ze driver: 0x%x", ret);
            continue;
        }

        uct_ze_base_info.num_devices = count;
    }

    return ret;
}

ze_driver_handle_t uct_ze_base_get_driver(void)
{
    if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
        return NULL;
    }

    return uct_ze_base_info.driver;
}

ze_device_handle_t uct_ze_base_get_device(int dev_num)
{
    if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
        return NULL;
    }

    if (dev_num < 0 || dev_num >= uct_ze_base_info.num_devices) {
        return NULL;
    }

    return uct_ze_base_info.devices[dev_num];
}

ucs_status_t
uct_ze_base_query_md_resources(uct_component_h component,
                               uct_md_resource_desc_t **resources_p,
                               unsigned *num_resources_p)
{
    if (uct_ze_base_init() != ZE_RESULT_SUCCESS) {
        ucs_debug("could not initialize ZE support");
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

ucs_status_t uct_ze_base_query_devices(uct_md_h md,
                                       uct_tl_device_resource_t **tl_devices_p,
                                       unsigned *num_tl_devices_p)
{
    return uct_single_device_resource(md, md->component->name,
                                      UCT_DEVICE_TYPE_ACC,
                                      UCS_SYS_DEVICE_ID_UNKNOWN, tl_devices_p,
                                      num_tl_devices_p);
}

UCS_MODULE_INIT()
{
    return UCS_OK;
}
