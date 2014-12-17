/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ib_iface.h"
#include "ib_context.h"

#include <uct/tl/context.h>
#include <ucs/type/component.h>
#include <ucs/type/class.h>
#include <ucs/debug/log.h>
#include <string.h>
#include <stdlib.h>


static ucs_status_t uct_ib_iface_find_port(uct_ib_context_t *ibctx,
                                           uct_ib_iface_t *iface,
                                           const char *dev_name)
{
    uct_ib_device_t *dev;
    const char *ibdev_name;
    unsigned port_num;
    unsigned dev_index;
    size_t devname_len;
    char *p;

    p = strrchr(dev_name, ':');
    if (p == NULL) {
        return UCS_ERR_INVALID_PARAM; /* Wrong dev_name format */
    }
    devname_len = p - dev_name;

    for (dev_index = 0; dev_index < ibctx->num_devices; ++dev_index) {
        dev = ibctx->devices[dev_index];
        ibdev_name = uct_ib_device_name(dev);
        if ((strlen(ibdev_name) == devname_len) &&
            !strncmp(ibdev_name, dev_name, devname_len))
        {
            port_num = strtod(p + 1, &p);
            if (*p != '\0') {
                return UCS_ERR_INVALID_PARAM; /* Failed to parse port number */
            }
            if ((port_num < dev->first_port) || (port_num >= dev->first_port + dev->num_ports)) {
                return UCS_ERR_NO_DEVICE; /* Port number out of range */
            }

            iface->super.super.pd = &dev->super;
            iface->port_num       = port_num;
            return UCS_OK;
        }
    }

    /* Device not found */
    return UCS_ERR_NO_DEVICE;
}

static UCS_CLASS_INIT_FUNC(uct_ib_iface_t, uct_iface_ops_t *ops,
                           uct_context_h context, const char *dev_name)
{
    uct_ib_context_t *ibctx = ucs_component_get(context, ib, uct_ib_context_t);
    uct_ib_device_t *dev;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(ops);

    status = uct_ib_iface_find_port(ibctx, self, dev_name);
    if (status != UCS_OK) {
        goto err;
    }

    dev = uct_ib_iface_device(self);

    /* TODO comp_channel */
    /* TODO cqe */
    self->send_cq = ibv_create_cq(dev->ibv_context, 1024, NULL, NULL, 0);
    if (self->send_cq == NULL) {
        ucs_error("Failed to create send cq: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    self->recv_cq = ibv_create_cq(dev->ibv_context, 1024, NULL, NULL, 0);
    if (self->recv_cq == NULL) {
        ucs_error("Failed to create recv cq: %m");
        goto err_destroy_send_cq;
    }

    if (uct_ib_device_is_port_ib(dev, self->port_num)) {
        self->addr.lid = uct_ib_device_port_attr(dev, self->port_num)->lid;
    } else {
        ucs_error("Unsupported link layer");
        goto err_destroy_recv_cq;
    }

    return UCS_OK;

err_destroy_recv_cq:
    ibv_destroy_cq(self->recv_cq);
err_destroy_send_cq:
    ibv_destroy_cq(self->send_cq);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ib_iface_t)
{
    int ret;

    ret = ibv_destroy_cq(self->recv_cq);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(recv_cq) returned %d: %m", ret);
    }

    ret = ibv_destroy_cq(self->send_cq);
    if (ret != 0) {
        ucs_warn("ibv_destroy_cq(send_cq) returned %d: %m", ret);
    }
}

UCS_CLASS_DEFINE(uct_ib_iface_t, uct_base_iface_t);

ucs_config_field_t uct_ib_iface_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_ib_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

  {NULL}
};
