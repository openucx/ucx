/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ib_iface.h"
#include "ib_context.h"

#include <ucs/type/component.h>
#include <string.h>
#include <stdlib.h>


static ucs_status_t uct_ib_iface_find_port(uct_ib_context_t *ibctx,
                                           uct_ib_iface_t *iface,
                                           const char *hw_name)
{
    uct_ib_device_t *dev;
    const char *dev_name;
    unsigned port_num;
    unsigned dev_index;
    size_t devname_len;
    char *p;

    p = strrchr(hw_name, ':');
    if (p == NULL) {
        return UCS_ERR_INVALID_PARAM; /* Wrong hw_name format */
    }
    devname_len = p - hw_name;

    for (dev_index = 0; dev_index < ibctx->num_devices; ++dev_index) {
        dev = ibctx->devices[dev_index];
        dev_name = uct_ib_device_name(dev);
        if ((strlen(dev_name) == devname_len) && !strncmp(dev_name, hw_name, devname_len)) {
            port_num = strtod(p + 1, &p);
            if (*p != '\0') {
                return UCS_ERR_INVALID_PARAM; /* Failed to parse port number */
            }
            if ((port_num < dev->first_port) || (port_num >= dev->first_port + dev->num_ports)) {
                return UCS_ERR_NO_DEVICE; /* Port number out of range */
            }

            iface->device   = dev;
            iface->port_num = port_num;
            return UCS_OK;
        }
    }

    /* Device not found */
    return UCS_ERR_NO_DEVICE;
}

ucs_status_t ucs_ib_iface_init(uct_context_h context, uct_ib_iface_t *iface,
                               const char *hw_name)
{
    uct_ib_context_t *ibctx = ucs_component_get(context, ib, uct_ib_context_t);
    ucs_status_t status;

    status = uct_ib_iface_find_port(ibctx, iface, hw_name);
    return status;
}

void ucs_ib_iface_cleanup(uct_ib_iface_t *iface)
{
}
