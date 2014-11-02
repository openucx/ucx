/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#define _GNU_SOURCE /* for CPU_ZERO/CPU_SET in sched.h */
#include "ib_device.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <sched.h>


static ucs_status_t
uct_ib_device_get_affinity(const char *dev_name, cpu_set_t *cpu_mask)
{
    char *p, buf[CPU_SETSIZE];
    ssize_t nread;
    uint32_t word;
    int base, k;

    nread = ucs_read_file(buf, sizeof(buf), 0,
                          "/sys/class/infiniband/%s/device/local_cpus",
                          dev_name);
    if (nread < 0) {
        return UCS_ERR_IO_ERROR;
    }

    base = 0;
    CPU_ZERO(cpu_mask);
    do {
        p = strrchr(buf, ',');
        if (p == NULL) {
            p = buf;
        } else if (*p == ',') {
            *(p++) = 0;
        }

        word = strtoul(p, 0, 16);
        for (k = 0; word; ++k, word >>= 1) {
            if (word & 1) {
                CPU_SET(base + k, cpu_mask);
            }
        }
        base += 32;
    } while ((base < CPU_SETSIZE) && (p != buf));

    return UCS_OK;
}

ucs_status_t uct_ib_device_create(struct ibv_device *ibv_device, uct_ib_device_t **dev_p)
{
    struct ibv_context *ibv_context;
    struct ibv_exp_device_attr dev_attr;
    uct_ib_device_t *dev;
    ucs_status_t status;
    uint8_t first_port, num_ports, i;
    int ret;

    /* Open verbs context */
    ibv_context = ibv_open_device(ibv_device);
    if (ibv_context == NULL) {
        ucs_error("Failed to open %s: %m", ibv_get_device_name(ibv_device));
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    /* Read device properties */
    IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(&dev_attr);
    ret = ibv_exp_query_device(ibv_context, &dev_attr);
    if (ret != 0) {
        ucs_error("ibv_query_device() returned %d: %m", ret);
        status = UCS_ERR_IO_ERROR;
        goto err_free_context;
    }

    /* Check device type*/
    switch (ibv_device->node_type) {
    case IBV_NODE_SWITCH:
        first_port = 0;
        num_ports  = 1;
        break;
    case IBV_NODE_CA:
    default:
        first_port = 1;
        num_ports  = dev_attr.phys_port_cnt;
        break;
    }

    /* Allocate device */
    dev = ucs_malloc(sizeof(*dev) + sizeof(*dev->port_attr) * num_ports,
                     "ib device");
    if (dev == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context;
    }

    /* Save device information */
    dev->ibv_context = ibv_context;
    dev->dev_attr    = dev_attr;
    dev->first_port  = first_port;
    dev->num_ports   = num_ports;

    /* Get device locality */
    status = uct_ib_device_get_affinity(ibv_get_device_name(ibv_device), &dev->local_cpus);
    if (status != UCS_OK) {
        goto err_free_device;
    }

    /* Query all ports */
    for (i = 0; i < dev->num_ports; ++i) {
        IBV_EXP_PORT_ATTR_SET_COMP_MASK(&dev->port_attr[i]);
        ret = ibv_exp_query_port(dev->ibv_context, i + dev->first_port,
                                 &dev->port_attr[i]);
        if (ret != 0) {
            ucs_error("ibv_query_port() returned %d: %m", ret);
            status = UCS_ERR_IO_ERROR;
            goto err_free_device;
        }
    }

    ucs_debug("created device '%s' (%s) with %d ports", uct_ib_device_name(dev),
              ibv_node_type_str(ibv_device->node_type),
              dev->num_ports);

    *dev_p = dev;
    return UCS_OK;

err_free_device:
    ucs_free(dev);
err_free_context:
    ibv_close_device(ibv_context);
err:
    return status;
}

void uct_ib_device_destroy(uct_ib_device_t *dev)
{
    ibv_close_device(dev->ibv_context);
    ucs_free(dev);
}

int uct_ib_device_port_check(uct_ib_device_t *dev, uint8_t port_num, unsigned flags)
{
    if (port_num < dev->first_port || port_num >= dev->first_port + dev->num_ports) {
        return 0;
    }

    if (uct_ib_device_port_attr(dev, port_num)->state != IBV_PORT_ACTIVE) {
        return 0;
    }

    /* TODO check flags, e.g DC support */

    return 1;
}

const char *uct_ib_device_name(uct_ib_device_t *dev)
{
    return ibv_get_device_name(dev->ibv_context->device);
}

ucs_status_t uct_ib_device_port_get_resource(uct_ib_device_t *dev, uint8_t port_num,
                                             uct_resource_desc_t *resource)
{
    static unsigned ib_port_widths[] = {
        [0] = 1,
        [1] = 4,
        [2] = 8,
        [3] = 12
    };
    struct sockaddr_in6 *in6_addr;
    double encoding, signal_rate;
    union ibv_gid gid;
    unsigned active_width;
    int ret;

    /* HCA:Port is the hardware resource name */
    ucs_snprintf_zero(resource->hw_name, sizeof(resource->hw_name), "%s:%d",
                      uct_ib_device_name(dev), port_num);

    /* Port network address */
    switch (uct_ib_device_port_attr(dev, port_num)->link_layer) {
    case IBV_LINK_LAYER_UNSPECIFIED:
    case IBV_LINK_LAYER_INFINIBAND:
        /*
         * For Infiniband, take the subnet prefix.
         */
        in6_addr = (struct sockaddr_in6 *)&(resource->subnet_addr);
        resource->addrlen      = sizeof(*in6_addr);
        in6_addr->sin6_family  = AF_INET6;
        ret = ibv_query_gid(dev->ibv_context, port_num, 0, &gid);
        if (ret != 0) {
            ucs_error("ibv_query_gid(%s:%d) failed: %m", uct_ib_device_name(dev),
                      port_num);
            return UCS_ERR_IO_ERROR;

        }

        gid.global.interface_id = 0; /* Zero-out GUID, keep only subnet prefix */
        UCS_STATIC_ASSERT(sizeof(in6_addr->sin6_addr) == sizeof(gid.raw));
        memcpy(&in6_addr->sin6_addr, &gid.raw, sizeof(gid.raw));
        break;
    case IBV_LINK_LAYER_ETHERNET:
        return UCS_ERR_UNSUPPORTED;
    default:
        ucs_error("Invalid link layer on %s:%d", uct_ib_device_name(dev), port_num);
        return UCS_ERR_IO_ERROR;
    }

    /* Copy local CPUs mask */
    resource->local_cpus = dev->local_cpus;

    /* Get active width */
    active_width = uct_ib_device_port_attr(dev, port_num)->active_width;
    if (!ucs_is_pow2(active_width) ||
        (active_width < 1) || (ucs_ilog2(active_width) > 3))
    {
        ucs_error("Invalid active_width on %s:%d: %d",
                  uct_ib_device_name(dev), port_num, active_width);
        return UCS_ERR_IO_ERROR;
    }

    /* Bandwidth calculation: Width * SignalRate * Encoding */
    switch (uct_ib_device_port_attr(dev, port_num)->active_speed) {
    case 1: /* SDR */
        resource->latency = 5000;
        signal_rate       = 2.5;
        encoding          = 8.0/10.0;
        break;
    case 2: /* DDR */
        resource->latency = 2500;
        signal_rate       = 5.0;
        encoding          = 8.0/10.0;
        break;
    case 4: /* QDR */
        resource->latency = 1300;
        signal_rate       = 10.0;
        encoding          = 8.0/10.0;
        break;
    case 8: /* FDR10 */
        resource->latency = 700;
        signal_rate       = 10.3125;
        encoding          = 64.0/66.0;
        break;
    case 16: /* FDR */
        resource->latency = 700;
        signal_rate       = 14.0625;
        encoding          = 64.0/66.0;
        break;
    case 32: /* EDR */
        resource->latency = 500;
        signal_rate       = 25.0;
        encoding          = 64.0/66.0;
        break;
    default:
        ucs_error("Invalid active_speed on %s:%d: %d",
                  uct_ib_device_name(dev), port_num,
                  uct_ib_device_port_attr(dev, port_num)->active_speed);
        return UCS_ERR_IO_ERROR;
    }

    resource->bandwidth = (long)((signal_rate * 1e9 * encoding *
                                  ib_port_widths[ucs_ilog2(active_width)]) / 8.0 + 0.5);
    return UCS_OK;
}
