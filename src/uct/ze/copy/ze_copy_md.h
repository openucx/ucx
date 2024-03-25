/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
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
} uct_ze_copy_md_t;


/**
 * ze copy domain configuration.
 */
typedef struct uct_ze_copy_md_config {
    uct_md_config_t super;
    int             device_ordinal;
} uct_ze_copy_md_config_t;


#endif
