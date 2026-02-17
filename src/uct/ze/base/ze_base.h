/*
 * Copyright (C) Intel Corporation, 2023-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef ZE_BASE_H
#define ZE_BASE_H

#include <ucs/type/status.h>
#include <ucs/sys/topo/base/topo.h>
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>

#include <level_zero/ze_api.h>


#define UCT_ZE_FUNC(_func, _log_level) \
    ({ \
        ucs_status_t _status = UCS_OK; \
        do { \
            ze_result_t _ret = (_func); \
            if (_ret == ZE_RESULT_NOT_READY) { \
                _status = UCS_INPROGRESS; \
            } else if (_ret != ZE_RESULT_SUCCESS) { \
                ucs_log((_log_level), "%s failed: 0x%x", \
                        UCS_PP_MAKE_STRING(_func), _ret); \
                _status = UCS_ERR_IO_ERROR; \
            } \
        } while (0); \
        _status; \
    })

#define UCT_ZE_FUNC_LOG_ERR(_func)   UCT_ZE_FUNC(_func, UCS_LOG_LEVEL_ERROR)
#define UCT_ZE_FUNC_LOG_DEBUG(_func) UCT_ZE_FUNC(_func, UCS_LOG_LEVEL_DEBUG)

#define UCT_ZE_MAX_SUBDEVICES 8

/* Level Zero device (root device) with sub-devices */
typedef struct {
    ze_device_handle_t     root_device;
    ze_device_properties_t device_props;
    ucs_sys_device_t       sys_dev;
    int                    num_subdevices;
    ze_device_handle_t     subdevices[UCT_ZE_MAX_SUBDEVICES];
    int                    device_index;
} uct_ze_device_t;


/* Level Zero sub-device descriptor */
typedef struct {
    const uct_ze_device_t *device;
    int                   subdevice_idx;
    int                   global_id;
} uct_ze_subdevice_t;


ze_result_t uct_ze_base_init(void);

ze_driver_handle_t uct_ze_base_get_driver(void);

const uct_ze_subdevice_t *uct_ze_base_get_subdevice_by_global_id(int global_id);

ze_device_handle_t uct_ze_base_get_device_handle_from_subdevice(
        const uct_ze_subdevice_t *subdevice);

ucs_status_t
uct_ze_base_query_md_resources(uct_component_h component,
                               uct_md_resource_desc_t **resources_p,
                               unsigned *num_resources_p);

ucs_status_t uct_ze_base_query_devices(uct_md_h md,
                                       uct_tl_device_resource_t **tl_devices_p,
                                       unsigned *num_tl_devices_p);

#endif
