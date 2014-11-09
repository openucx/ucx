/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ib_iface.h"
#include "ib_context.h"

#include <ucs/type/component.h>
#include <ucs/debug/log.h>
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

            iface->super.pd = &dev->super;
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
    struct ibv_exp_port_attr *port_attr;
    uct_ib_device_t *dev;
    ucs_status_t status;

    status = uct_ib_iface_find_port(ibctx, iface, hw_name);
    if (status != UCS_OK) {
        goto err;
    }

    dev = uct_ib_iface_device(iface);

    /* TODO comp_channel */
    /* TODO cqe */
    iface->send_cq = ibv_create_cq(dev->ibv_context, 1024, NULL, NULL, 0);
    if (iface->send_cq == NULL) {
        ucs_error("Failed to create send cq: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    iface->recv_cq = ibv_create_cq(dev->ibv_context, 1024, NULL, NULL, 0);
    if (iface->recv_cq == NULL) {
        ucs_error("Failed to create recv cq: %m");
        goto err_destroy_send_cq;
    }

    port_attr = uct_ib_device_port_attr(dev, iface->port_num);
    switch (port_attr->link_layer) {
    case IBV_LINK_LAYER_UNSPECIFIED:
    case IBV_LINK_LAYER_INFINIBAND:
        iface->addr.lid = port_attr->lid;
        break;
    default:
        ucs_error("Unsupported link layer");
        goto err_destroy_recv_cq;
    }

    return UCS_OK;

err_destroy_recv_cq:
    ibv_destroy_cq(iface->recv_cq);
err_destroy_send_cq:
    ibv_destroy_cq(iface->send_cq);
err:
    return status;
}

void ucs_ib_iface_cleanup(uct_ib_iface_t *iface)
{
    int ret;

    ret = ibv_destroy_cq(iface->recv_cq);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(recv_cq) returned %d: %m", ret);
    }

    ret = ibv_destroy_cq(iface->send_cq);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(send_cq) returned %d: %m", ret);
    }
}
