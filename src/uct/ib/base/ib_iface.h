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


typedef struct uct_ib_iface {
    uct_iface_t             super;
    uct_ib_device_t         *device;
    uint8_t                 port_num;

    /* TODO
     * lmc
     * port_addr;
     * sl
     * gid_index
     * port_num
     * comp_channel;
     */
} uct_ib_iface_t;


ucs_status_t ucs_ib_iface_init(uct_context_h context, uct_ib_iface_t *iface,
                               const char *hw_name);
void ucs_ib_iface_cleanup(uct_ib_iface_t *iface);


#endif
