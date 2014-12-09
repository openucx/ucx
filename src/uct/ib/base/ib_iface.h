/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_IFACE_H
#define UCT_IB_IFACE_H

#include "ib_device.h"

#include <uct/api/uct.h>
#include <uct/tl/tl_base.h>
#include <ucs/sys/compiler.h>
#include <ucs/config/parser.h>


typedef struct uct_ib_iface_addr {
    uct_iface_addr_t    super;
    uint16_t            lid; /* TODO support RoCE/GRH */
} uct_ib_iface_addr_t;


typedef struct uct_ib_iface {
    uct_base_iface_t        super;
    uint8_t                 port_num;
    /* TODO
     * lmc
     * sl
     * gid_index
     * comp_channel;
     */
    uct_ib_iface_addr_t     addr;
    struct ibv_cq           *send_cq;
    struct ibv_cq           *recv_cq;
} uct_ib_iface_t;

typedef struct uct_ib_iface_config {
    uct_iface_config_t      super;
} uct_ib_iface_config_t;


extern ucs_config_field_t uct_ib_iface_config_table[];

static inline uct_ib_device_t * uct_ib_iface_device(uct_ib_iface_t *iface)
{
    return ucs_derived_of(iface->super.super.pd, uct_ib_device_t);
}

#endif
