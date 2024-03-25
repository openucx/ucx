/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ZE_COPY_IFACE_H
#define UCT_ZE_COPY_IFACE_H

#include <uct/base/uct_iface.h>
#include <level_zero/ze_api.h>


#define UCT_ZE_COPY_TL_NAME "ze_cpy"


typedef uint64_t uct_ze_copy_iface_addr_t;


typedef struct uct_ze_copy_iface {
    uct_base_iface_t          super;
    uct_ze_copy_iface_addr_t  id;
    ze_command_queue_handle_t ze_cmdq;
    ze_command_list_handle_t  ze_cmdl;
} uct_ze_copy_iface_t;

#endif
