/*
 * Copyright (C) Intel Corporation, 2023-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ZE_COPY_MD_H
#define UCT_ZE_COPY_MD_H

#include <level_zero/ze_api.h>
#include <uct/base/uct_md.h>
#include <ucs/config/types.h>


extern uct_component_t uct_ze_copy_component;


/*
 * @brief ze_copy MD descriptor
 */
typedef struct uct_ze_copy_md {
    uct_md_t            super; /**< Domain info */
    ze_context_handle_t ze_context;
    ze_device_handle_t  ze_device;
    int                 host_import; /**< Import registered host memory */
} uct_ze_copy_md_t;


/**
 * ze copy domain configuration.
 */
typedef struct uct_ze_copy_md_config {
    uct_md_config_t super;
    int             device_ordinal;
    int             host_import;
} uct_ze_copy_md_config_t;


/* Host pointer import extension of the Level Zero driver */
typedef ze_result_t (*uct_ze_copy_import_ptr_func_t)(ze_driver_handle_t driver,
                                                     void *ptr, size_t size);

typedef ze_result_t (*uct_ze_copy_release_ptr_func_t)(ze_driver_handle_t driver,
                                                      void *ptr);

typedef ze_result_t (*uct_ze_copy_get_base_func_t)(ze_driver_handle_t driver,
                                                   void *ptr, void **base_p);


/**
 * Host pointer import entry points. The extension bypasses the loader, so
 * @a driver is the driver's own handle rather than the loader's.
 */
typedef struct uct_ze_copy_host_import {
    ze_driver_handle_t             driver;
    uct_ze_copy_import_ptr_func_t  import;
    uct_ze_copy_release_ptr_func_t release;
    uct_ze_copy_get_base_func_t    get_base;
} uct_ze_copy_host_import_t;


/**
 * @return Host pointer import entry points, or NULL when the loader or the
 *         driver does not provide them.
 */
const uct_ze_copy_host_import_t *uct_ze_copy_md_get_host_import(void);


#endif
