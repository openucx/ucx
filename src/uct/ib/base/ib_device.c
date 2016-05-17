/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ib_device.h"
#include "ib_pd.h"

#include <ucs/arch/bitops.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/async/async.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <sys/poll.h>
#include <sched.h>


#if ENABLE_STATS
static ucs_stats_class_t uct_ib_device_stats_class = {
    .name           = "",
    .num_counters   = UCT_IB_DEVICE_STAT_LAST,
    .counter_names = {
        [UCT_IB_DEVICE_STAT_ASYNC_EVENT] = "async_event"
    }
};
#endif

static void uct_ib_device_get_affinity(const char *dev_name, cpu_set_t *cpu_mask)
{
    char *p, buf[CPU_SETSIZE];
    ssize_t nread;
    uint32_t word;
    int base, k;

    CPU_ZERO(cpu_mask);
    nread = ucs_read_file(buf, sizeof(buf), 1,
                          "/sys/class/infiniband/%s/device/local_cpus",
                          dev_name);
    if (nread >= 0) {
        base = 0;
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
    } else {
        /* If affinity file is not present, treat all CPUs as local */
        for (k = 0; k < CPU_SETSIZE; ++k) {
            CPU_SET(k, cpu_mask);
        }
    }
}

static void uct_ib_async_event_handler(void *arg)
{
    uct_ib_device_t *dev = arg;
    struct ibv_async_event event;
    char event_info[200];
    int ret;

    ret = ibv_get_async_event(dev->ibv_context, &event);
    if (ret != 0) {
        ucs_warn("ibv_get_async_event() failed: %m");
        return;
    }

    switch (event.event_type) {
    case IBV_EVENT_CQ_ERR:
        snprintf(event_info, sizeof(event_info), "%s on CQ %p",
                 ibv_event_type_str(event.event_type), event.element.cq);
        break;
    case IBV_EVENT_QP_FATAL:
    case IBV_EVENT_QP_REQ_ERR:
    case IBV_EVENT_QP_ACCESS_ERR:
    case IBV_EVENT_COMM_EST:
    case IBV_EVENT_SQ_DRAINED:
    case IBV_EVENT_PATH_MIG:
    case IBV_EVENT_PATH_MIG_ERR:
        snprintf(event_info, sizeof(event_info), "%s on QPN 0x%x",
                 ibv_event_type_str(event.event_type), event.element.qp->qp_num);
        break;
    case IBV_EVENT_QP_LAST_WQE_REACHED:
        snprintf(event_info, sizeof(event_info), "SRQ-attached QP 0x%x was flushed",
                 event.element.qp->qp_num);
        break;
    case IBV_EVENT_SRQ_ERR:
    case IBV_EVENT_SRQ_LIMIT_REACHED:
        snprintf(event_info, sizeof(event_info), "%s on SRQ %p",
                 ibv_event_type_str(event.event_type), event.element.srq);
        break;
    case IBV_EVENT_DEVICE_FATAL:
    case IBV_EVENT_PORT_ACTIVE:
    case IBV_EVENT_PORT_ERR:
#if HAVE_DECL_IBV_EVENT_GID_CHANGE
    case IBV_EVENT_GID_CHANGE:
#endif
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_CLIENT_REREGISTER:
        snprintf(event_info, sizeof(event_info), "%s on port %d",
                 ibv_event_type_str(event.event_type), event.element.port_num);
        break;
#if HAVE_STRUCT_IBV_ASYNC_EVENT_ELEMENT_DCT
    case IBV_EXP_EVENT_DCT_KEY_VIOLATION:
    case IBV_EXP_EVENT_DCT_ACCESS_ERR:
    case IBV_EXP_EVENT_DCT_REQ_ERR:
        snprintf(event_info, sizeof(event_info), "%s on DCTN 0x%x",
                 ibv_event_type_str(event.event_type), event.element.dct->dct_num);
        break;
#endif
    default:
        snprintf(event_info, sizeof(event_info), "%s",
                 ibv_event_type_str(event.event_type));
        break;
    };

    UCS_STATS_UPDATE_COUNTER(dev->stats, UCT_IB_DEVICE_STAT_ASYNC_EVENT, +1);
    ucs_warn("IB Async event on %s: %s", uct_ib_device_name(dev), event_info);
    ibv_ack_async_event(&event);
}

ucs_status_t uct_ib_device_init(uct_ib_device_t *dev, struct ibv_device *ibv_device
                                UCS_STATS_ARG(ucs_stats_node_t *stats_parent))
{
    ucs_status_t status;
    uint8_t i;
    int ret;

    setenv("MLX5_TOTAL_UUARS",       "64", 1);
    setenv("MLX5_NUM_LOW_LAT_UUARS", "60", 1);

    /* Open verbs context */
    dev->ibv_context = ibv_open_device(ibv_device);
    if (dev->ibv_context == NULL) {
        ucs_error("Failed to open %s: %m", ibv_get_device_name(ibv_device));
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    /* Read device properties */
    IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(&dev->dev_attr);
    ret = ibv_exp_query_device(dev->ibv_context, &dev->dev_attr);
    if (ret != 0) {
        ucs_error("ibv_query_device() returned %d: %m", ret);
        status = UCS_ERR_IO_ERROR;
        goto err_free_context;
    }

    /* Check device type*/
    switch (ibv_device->node_type) {
    case IBV_NODE_SWITCH:
        dev->first_port = 0;
        dev->num_ports  = 1;
        break;
    case IBV_NODE_CA:
    default:
        dev->first_port = 1;
        dev->num_ports  = dev->dev_attr.phys_port_cnt;
        break;
    }

    if (dev->num_ports > UCT_IB_DEV_MAX_PORTS) {
        ucs_error("%s has %d ports, but only up to %d are supported",
                  ibv_get_device_name(ibv_device), dev->num_ports,
                  UCT_IB_DEV_MAX_PORTS);
        status = UCS_ERR_UNSUPPORTED;
        goto err_free_context;
    }

    /* Get device locality */
    uct_ib_device_get_affinity(ibv_get_device_name(ibv_device), &dev->local_cpus);

    /* Query all ports */
    for (i = 0; i < dev->num_ports; ++i) {
        IBV_EXP_PORT_ATTR_SET_COMP_MASK(&dev->port_attr[i]);
        ret = ibv_exp_query_port(dev->ibv_context, i + dev->first_port,
                                 &dev->port_attr[i]);
        if (ret != 0) {
            ucs_error("ibv_query_port() returned %d: %m", ret);
            status = UCS_ERR_IO_ERROR;
            goto err_free_context;
        }
    }

    status = UCS_STATS_NODE_ALLOC(&dev->stats, &uct_ib_device_stats_class,
                                  stats_parent, "device");
    if (status != UCS_OK) {
        goto err_free_context;
    }

    status = ucs_sys_fcntl_modfl(dev->ibv_context->async_fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_release_stats;
    }

    /* Register to IB async events
     * TODO have option to set async mode as signal/thread.
     */
    status = ucs_async_set_event_handler(UCS_ASYNC_MODE_THREAD,
                                         dev->ibv_context->async_fd,
                                         POLLIN, uct_ib_async_event_handler, dev, NULL);
    if (status != UCS_OK) {
        goto err_release_stats;
    }

    ucs_debug("initialized device '%s' (%s) with %d ports", uct_ib_device_name(dev),
              ibv_node_type_str(ibv_device->node_type),
              dev->num_ports);
    return UCS_OK;

err_release_stats:
    UCS_STATS_NODE_FREE(dev->stats);
err_free_context:
    ibv_close_device(dev->ibv_context);
err:
    return status;
}

void uct_ib_device_cleanup(uct_ib_device_t *dev)
{
    ucs_debug("destroying ib device %s", uct_ib_device_name(dev));

    ucs_async_unset_event_handler(dev->ibv_context->async_fd);
    UCS_STATS_NODE_FREE(dev->stats);
    ibv_close_device(dev->ibv_context);
}

ucs_status_t uct_ib_device_port_check(uct_ib_device_t *dev, uint8_t port_num,
                                      unsigned flags)
{
    if (port_num < dev->first_port || port_num >= dev->first_port + dev->num_ports) {
        return UCS_ERR_NO_DEVICE;
    }

    if (uct_ib_device_port_attr(dev, port_num)->state != IBV_PORT_ACTIVE) {
        ucs_trace("%s:%d is not active (state: %d)", uct_ib_device_name(dev),
                  port_num, uct_ib_device_port_attr(dev, port_num)->state);
        return UCS_ERR_UNREACHABLE;
    }

    if (!uct_ib_device_is_port_ib(dev, port_num)) {
         ucs_trace("%s:%d is not IB link layer", uct_ib_device_name(dev),
                   port_num);
         return UCS_ERR_UNSUPPORTED;
    }

    if (flags & UCT_IB_DEVICE_FLAG_DC) {
        if (!IBV_DEVICE_HAS_DC(&dev->dev_attr)) {
            ucs_trace("%s:%d does not support DC", uct_ib_device_name(dev), port_num);
            return UCS_ERR_UNSUPPORTED;
        }
    }

    if (flags & UCT_IB_DEVICE_FLAG_MLX4_PRM) {
        ucs_trace("%s:%d does not support mlx4 PRM", uct_ib_device_name(dev), port_num);
        return UCS_ERR_UNSUPPORTED; /* Unsupported yet */
    }

    if (flags & UCT_IB_DEVICE_FLAG_MLX5_PRM) {
        /* TODO list all devices with their flags */
        if (dev->dev_attr.vendor_id != 0x02c9 ||
            (dev->dev_attr.vendor_part_id != 4113 && dev->dev_attr.vendor_part_id != 4115 &&
             dev->dev_attr.vendor_part_id != 4117))
        {
            ucs_trace("%s:%d does not support mlx5 PRM", uct_ib_device_name(dev), port_num);
            return UCS_ERR_UNSUPPORTED;
        }
    }

    return UCS_OK;
}

const char *uct_ib_device_name(uct_ib_device_t *dev)
{
    return ibv_get_device_name(dev->ibv_context->device);
}

int uct_ib_device_is_port_ib(uct_ib_device_t *dev, uint8_t port_num)
{
#if HAVE_DECL_IBV_LINK_LAYER_INFINIBAND
    switch (uct_ib_device_port_attr(dev, port_num)->link_layer) {
    case IBV_LINK_LAYER_UNSPECIFIED:
    case IBV_LINK_LAYER_INFINIBAND:
        return 1;
    case IBV_LINK_LAYER_ETHERNET:
        return 0;
    default:
        ucs_fatal("Invalid link layer on %s:%d", uct_ib_device_name(dev), port_num);
    }
#else
    return 1;
#endif
}

size_t uct_ib_mtu_value(enum ibv_mtu mtu)
{
    switch (mtu) {
    case IBV_MTU_256:
        return 256;
    case IBV_MTU_512:
        return 512;
    case IBV_MTU_1024:
        return 1024;
    case IBV_MTU_2048:
        return 2048;
    case IBV_MTU_4096:
        return 4096;
    }
    ucs_fatal("Invalid MTU value (%d)", mtu);
}

uint8_t uct_ib_to_fabric_time(double time)
{
    double to;
    long t;

    to = log(time / 4.096e-6) / log(2.0);
    if (to < 1) {
        return 1; /* Very small timeout */
    } else if (to > 30) {
        return 0; /* No timeout */
    } else {
        t = (long)(to + 0.5);
        ucs_assert(t >= 1 && t < 31);
        return t;
    }
}

size_t uct_ib_address_size(uct_ib_address_scope_t scope)
{
    switch (scope) {
    case UCT_IB_ADDRESS_SCOPE_LINK_LOCAL:
        return sizeof(uct_ib_address_t) +
               sizeof(uint16_t); /* lid */
    case UCT_IB_ADDRESS_SCOPE_SITE_LOCAL:
        return sizeof(uct_ib_address_t) +
               sizeof(uint16_t) + /* lid */
               sizeof(uint64_t) + /* if_id */
               sizeof(uint16_t);  /* subnet16 */
    case UCT_IB_ADDRESS_SCOPE_GLOBAL:
        return sizeof(uct_ib_address_t) +
               sizeof(uint16_t) + /* lid */
               sizeof(uint64_t) + /* if_id */
               sizeof(uint64_t);  /* subney64 */
    default:
        ucs_fatal("Invalid IB address scope: %d", scope);
    }
}

void uct_ib_address_pack(uct_ib_device_t *dev, uint8_t port_num,
                         uct_ib_address_scope_t scope, const union ibv_gid *gid,
                         uct_ib_address_t *ib_addr)
{
    void *ptr = ib_addr + 1;

    ib_addr->flags  = 0;
    ib_addr->dev_id = dev->dev_attr.vendor_part_id;

    /* LID */
    /* TODO check IB link layer of the port */
    ib_addr->flags |= UCT_IB_ADDRESS_FLAG_LID;
    *(uint16_t*)ptr = uct_ib_device_port_attr(dev, port_num)->lid;
    ptr += sizeof(uint16_t);

    if (scope >= UCT_IB_ADDRESS_SCOPE_SITE_LOCAL) {
        ib_addr->flags |= UCT_IB_ADDRESS_FLAG_IF_ID;
        *(uint64_t*)ptr = gid->global.interface_id;
        ptr += sizeof(uint64_t);

        if (scope >= UCT_IB_ADDRESS_SCOPE_GLOBAL) {
            /* Global */
            ib_addr->flags |= UCT_IB_ADDRESS_FLAG_SUBNET64;
            *(uint64_t*)ptr = gid->global.subnet_prefix;
        } else {
            /* Site-local */
            ib_addr->flags |= UCT_IB_ADDRESS_FLAG_SUBNET16;
            *(uint16_t*)ptr = gid->global.subnet_prefix >> 48;
        }
    }
}

void uct_ib_address_unpack(const uct_ib_address_t *ib_addr, uint16_t *lid,
                           uint8_t *is_global, union ibv_gid *gid)
{
    const void *ptr = ib_addr + 1;

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_LID) {
        *lid = *(uint16_t*)ptr;
        ptr += sizeof(uint16_t);
    } else {
        *lid = 0;
    }

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_IF_ID) {
        gid->global.interface_id = *(uint64_t*)ptr;
        ptr += sizeof(uint64_t);
    } else {
        gid->global.interface_id = 0;
    }

    gid->global.subnet_prefix = UCT_IB_LINK_LOCAL_PREFIX; /* Default prefix */
    *is_global                = 0;

    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_SUBNET16) {
        gid->global.subnet_prefix = UCT_IB_SITE_LOCAL_PREFIX |
                                    ((uint64_t)*(uint16_t*)ptr << 48);
        *is_global                = 1;
    }
    if (ib_addr->flags & UCT_IB_ADDRESS_FLAG_SUBNET64) {
        gid->global.subnet_prefix = *(uint64_t*)ptr;
        *is_global                = 1;
    }
}

const char *uct_ib_address_str(const uct_ib_address_t *ib_addr, char *buf,
                               size_t max)
{
    union ibv_gid gid;
    uint8_t is_global;
    uint16_t lid;
    char *p, *endp;

    uct_ib_address_unpack(ib_addr, &lid, &is_global, &gid);

    if (is_global) {
        p    = buf;
        endp = buf + max;
        if (lid != 0) {
            snprintf(p, endp - p, "lid %d ", lid);
            p += strlen(p);
        }
        inet_ntop(AF_INET6, &gid, p, endp - p);
    } else {
        snprintf(buf, max, "lid %d", lid);
    }

    return buf;
}


ucs_status_t uct_ib_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    uct_tl_resource_desc_t *resources, *rsc;
    unsigned num_resources;
    ucs_status_t status;
    uint8_t port_num;

    /* Allocate resources array
     * We may allocate more memory than really required, but it's not so bad. */
    resources = ucs_calloc(dev->num_ports, sizeof(uct_tl_resource_desc_t),
                           "ib resource");
    if (resources == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* Second pass: fill port information */
    num_resources = 0;
    for (port_num = dev->first_port; port_num < dev->first_port + dev->num_ports;
         ++port_num)
    {
        /* Check port capabilities */
        status = uct_ib_device_port_check(dev, port_num, flags);
        if (status != UCS_OK) {
           ucs_trace("%s:%d does not support flags 0x%x: %s",
                     uct_ib_device_name(dev), port_num, flags,
                     ucs_status_string(status));
           continue;
        }

        /* Get port information */
        rsc = &resources[num_resources];
        ucs_snprintf_zero(rsc->dev_name, sizeof(rsc->dev_name), "%s:%d",
                          uct_ib_device_name(dev), port_num);
        ucs_snprintf_zero(rsc->tl_name, UCT_TL_NAME_MAX, "%s", tl_name);
        rsc->dev_type = UCT_DEVICE_TYPE_NET;

        ucs_debug("found usable port for tl %s %s:%d", tl_name,
                  uct_ib_device_name(dev), port_num);
        ++num_resources;
    }

    if (num_resources == 0) {
        ucs_debug("no compatible IB ports found for flags 0x%x", flags);
        status = UCS_ERR_NO_DEVICE;
        goto err_free;
    }

    *num_resources_p = num_resources;
    *resources_p     = resources;
    return UCS_OK;

err_free:
    ucs_free(resources);
err:
    return status;
}

ucs_status_t uct_ib_device_find_port(uct_ib_device_t *dev,
                                     const char *resource_dev_name,
                                     uint8_t *p_port_num)
{
    const char *ibdev_name;
    unsigned port_num;
    size_t devname_len;
    char *p;

    p = strrchr(resource_dev_name, ':');
    if (p == NULL) {
        goto err; /* Wrong device name format */
    }
    devname_len = p - resource_dev_name;

    ibdev_name = uct_ib_device_name(dev);
    if ((strlen(ibdev_name) != devname_len) ||
        strncmp(ibdev_name, resource_dev_name, devname_len))
    {
        goto err; /* Device name is wrong */
    }

    port_num = strtod(p + 1, &p);
    if (*p != '\0') {
        goto err; /* Failed to parse port number */
    }
    if ((port_num < dev->first_port) || (port_num >= dev->first_port + dev->num_ports)) {
        goto err; /* Port number out of range */
    }

    *p_port_num = port_num;
    return UCS_OK;

err:
    ucs_error("%s: failed to find port", resource_dev_name);
    return UCS_ERR_NO_DEVICE;
}

ucs_status_t uct_ib_device_mtu(const char *dev_name, uct_pd_h pd, int *p_mtu)
{

    uct_ib_device_t *dev = &ucs_derived_of(pd, uct_ib_pd_t)->dev;
    uint8_t port_num;
    ucs_status_t status;

    status = uct_ib_device_find_port(dev, dev_name, &port_num);
    if (status != UCS_OK) {
        return status;
    }

    *p_mtu = uct_ib_mtu_value(uct_ib_device_port_attr(dev, port_num)->active_mtu);
    return UCS_OK;
}
