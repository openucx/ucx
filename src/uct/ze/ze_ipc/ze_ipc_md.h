/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ZE_IPC_MD_H
#define UCT_ZE_IPC_MD_H

#include <uct/base/uct_md.h>
#include <level_zero/ze_api.h>


extern uct_component_t uct_ze_ipc_component;


/**
 * @brief ze ipc MD descriptor
 */
typedef struct uct_ze_ipc_md {
    uct_md_t            super;      /**< Domain info */
    ze_context_handle_t ze_context; /**< Level Zero context */
    ze_device_handle_t  ze_device;  /**< Level Zero device */
} uct_ze_ipc_md_t;


/**
 * @brief ze ipc domain configuration.
 */
typedef struct uct_ze_ipc_md_config {
    uct_md_config_t super;
    int             device_ordinal;
} uct_ze_ipc_md_config_t;


/**
 * @brief ze ipc remote key for put/get
 */
typedef struct uct_ze_ipc_key {
    ze_ipc_mem_handle_t ipc_handle; /**< IPC memory handle */
    pid_t               pid;        /**< Remote process ID (for cache) */
    uintptr_t           address;    /**< Base address of the allocation */
    size_t              length;     /**< Size of the allocation */
    int                 dev_num;    /**< GPU device number */
} uct_ze_ipc_key_t;


#endif
