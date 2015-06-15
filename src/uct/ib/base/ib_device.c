/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#define _GNU_SOURCE /* for CPU_ZERO/CPU_SET in sched.h */
#include "ib_device.h"

#include <uct/tl/context.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/async/async.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <sys/poll.h>
#include <sched.h>


#define UCT_IB_PD_PREFIX         "ib"
#define UCT_IB_MEM_ACCESS_FLAGS  (IBV_ACCESS_LOCAL_WRITE | \
                                  IBV_ACCESS_REMOTE_WRITE | \
                                  IBV_ACCESS_REMOTE_READ | \
                                  IBV_ACCESS_REMOTE_ATOMIC)

#if ENABLE_STATS
static ucs_stats_class_t uct_ib_device_stats_class = {
    .name           = "",
    .num_counters   = UCT_IB_DEVICE_STAT_LAST,
    .counter_names = {
        [UCT_IB_DEVICE_STAT_MEM_ALLOC]   = "mem_alloc",
        [UCT_IB_DEVICE_STAT_MEM_REG]     = "mem_reg",
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

static void uct_ib_pd_close(uct_pd_h pd)
{
    uct_ib_device_t *dev = ucs_derived_of(pd, uct_ib_device_t);

    ucs_debug("closing ib device %s", uct_ib_device_name(dev));
    ucs_async_unset_event_handler(dev->ibv_context->async_fd);
    UCS_STATS_NODE_FREE(dev->stats);
    ibv_dealloc_pd(dev->pd);
    ibv_close_device(dev->ibv_context);
    ucs_free(dev);
}

static ucs_status_t uct_ib_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    uct_ib_device_t *dev = ucs_derived_of(pd, uct_ib_device_t);

    pd_attr->cap.max_alloc = ULONG_MAX; /* TODO query device */
    pd_attr->cap.max_reg   = ULONG_MAX; /* TODO query device */
    pd_attr->cap.flags     = UCT_PD_FLAG_REG;
    pd_attr->local_cpus    = dev->local_cpus;
    return UCS_OK;
}

static ucs_status_t uct_ib_dereg_mr(struct ibv_mr *mr)
{
    int ret;

    ret = ibv_dereg_mr(mr);
    if (ret != 0) {
        ucs_error("ibv_dereg_mr() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_ib_mem_alloc(uct_pd_h pd, size_t *length_p, void **address_p,
                                     uct_mem_h *memh_p UCS_MEMTRACK_ARG)
{
    uct_ib_device_t *dev = ucs_derived_of(pd, uct_ib_device_t);
    struct ibv_exp_reg_mr_in in = {
        dev->pd,
        NULL,
        ucs_memtrack_adjust_alloc_size(*length_p),
        UCT_IB_MEM_ACCESS_FLAGS | IBV_EXP_ACCESS_ALLOCATE_MR,
        0
    };
    struct ibv_mr *mr;

    mr = ibv_exp_reg_mr(&in);
    if (mr == NULL) {
        ucs_error("ibv_exp_reg_mr(in={NULL, length=%Zu, flags=0x%lx}) failed: %m",
                  ucs_memtrack_adjust_alloc_size(*length_p),
                  (unsigned long)(UCT_IB_MEM_ACCESS_FLAGS | IBV_EXP_ACCESS_ALLOCATE_MR));
        return UCS_ERR_IO_ERROR;
    }

    UCS_STATS_UPDATE_COUNTER(dev->stats, UCT_IB_DEVICE_STAT_MEM_ALLOC, +1);
    *address_p = mr->addr;
    *length_p  = mr->length;
    ucs_memtrack_allocated(address_p, length_p UCS_MEMTRACK_VAL);
    *memh_p = mr;
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_free(uct_pd_h pd, uct_mem_h memh)
{
    struct ibv_mr *mr = memh;
    void UCS_V_UNUSED *address = mr->addr;

    ucs_memtrack_releasing(&address);
    return uct_ib_dereg_mr(mr);
}

static ucs_status_t uct_ib_mem_reg(uct_pd_h pd, void *address, size_t length,
                                   uct_mem_h *memh_p)
{
    uct_ib_device_t *dev = ucs_derived_of(pd, uct_ib_device_t);
    struct ibv_mr *mr;

    mr = ibv_reg_mr(dev->pd, address, length, UCT_IB_MEM_ACCESS_FLAGS);
    if (mr == NULL) {
        ucs_error("ibv_reg_mr(address=%p, length=%zu, flags=0x%x) failed: %m",
                  address, length, UCT_IB_MEM_ACCESS_FLAGS);
        return UCS_ERR_IO_ERROR;
    }

    UCS_STATS_UPDATE_COUNTER(dev->stats, UCT_IB_DEVICE_STAT_MEM_REG, +1);
    *memh_p = mr;
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_dereg(uct_pd_h pd, uct_mem_h memh)
{
    struct ibv_mr *mr = memh;
    return uct_ib_dereg_mr(mr);
}

static ucs_status_t uct_ib_mkey_pack(uct_pd_h pd, uct_mem_h memh,
                                     void *rkey_buffer)
{
    struct ibv_mr *mr = memh;
    *(uint32_t*)rkey_buffer = mr->rkey;
    ucs_trace("packed rkey: 0x%x", mr->rkey);
    return UCS_OK;
}

static ucs_status_t uct_ib_rkey_unpack(const void *rkey_buffer, uct_rkey_t *rkey_p,
                                       void **handle_p)
{
    uint32_t ib_rkey = *(const uint32_t*)rkey_buffer;

    *rkey_p   = ib_rkey;
    *handle_p = NULL;
    ucs_trace("unpacked rkey: 0x%x", ib_rkey);
    return UCS_OK;
}

uct_pd_ops_t uct_ib_pd_ops = {
    .close        = uct_ib_pd_close,
    .query        = uct_ib_pd_query,
    .mem_alloc    = uct_ib_mem_alloc,
    .mem_free     = uct_ib_mem_free,
    .mem_reg      = uct_ib_mem_reg,
    .mem_dereg    = uct_ib_mem_dereg,
    .mkey_pack    = uct_ib_mkey_pack,
};

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

static ucs_status_t uct_ib_device_create(struct ibv_device *ibv_device,
                                         uct_ib_device_t **dev_p)
{
    struct ibv_context *ibv_context;
    struct ibv_exp_device_attr dev_attr;
    uct_ib_device_t *dev;
    ucs_status_t status;
    uint8_t first_port, num_ports, i;
    int ret;

    setenv("MLX5_TOTAL_UUARS",       "64", 1);
    setenv("MLX5_NUM_LOW_LAT_UUARS", "60", 1);

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
    dev->super.ops     = &uct_ib_pd_ops;
    dev->ibv_context   = ibv_context;
    dev->dev_attr      = dev_attr;
    dev->first_port    = first_port;
    dev->num_ports     = num_ports;

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
            goto err_free_device;
        }
    }

    /* Allocate protection domain */
    dev->pd = ibv_alloc_pd(dev->ibv_context);
    if (dev->pd == NULL) {
        ucs_error("ibv_alloc_pd() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free_device;
    }

    status = UCS_STATS_NODE_ALLOC(&dev->stats, &uct_ib_device_stats_class, NULL,
                                  "%s-%p", uct_ib_device_name(dev), dev);
    if (status != UCS_OK) {
        goto err_destroy_pd;
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

    ucs_debug("created device '%s' (%s) with %d ports", uct_ib_device_name(dev),
              ibv_node_type_str(ibv_device->node_type),
              dev->num_ports);

    *dev_p = dev;
    return UCS_OK;

err_release_stats:
    UCS_STATS_NODE_FREE(dev->stats);
err_destroy_pd:
    ibv_dealloc_pd(dev->pd);
err_free_device:
    ucs_free(dev);
err_free_context:
    ibv_close_device(ibv_context);
err:
    return status;
}

static ucs_status_t uct_ib_device_port_check(uct_ib_device_t *dev, uint8_t port_num,
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
        if (dev->dev_attr.vendor_id != 0x02c9 || dev->dev_attr.vendor_part_id != 4113) {
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

static ucs_status_t
uct_ib_device_port_get_resource(uct_ib_device_t *dev, uint8_t port_num,
                                size_t tl_hdr_len, uint64_t tl_overhead_ns,
                                uct_tl_resource_desc_t *resource)
{
    static unsigned ib_port_widths[] = {
        [0] = 1,
        [1] = 4,
        [2] = 8,
        [3] = 12
    };
    double encoding, signal_rate, wire_speed;
    unsigned active_width;
    size_t extra_pkt_len;
    size_t mtu;

    /* HCA:Port is the hardware resource name */
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s:%d",
                      uct_ib_device_name(dev), port_num);

    if (!uct_ib_device_is_port_ib(dev, port_num)) {
        return UCS_ERR_UNSUPPORTED;
    }

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

    mtu           = uct_ib_mtu_value(uct_ib_device_port_attr(dev, port_num)->active_mtu);
    wire_speed    = (signal_rate * 1e9 * encoding *
                     ib_port_widths[ucs_ilog2(active_width)]) / 8.0;
    extra_pkt_len = UCT_IB_LRH_LEN + UCT_IB_BTH_LEN + tl_hdr_len +
                    UCT_IB_ICRC_LEN + UCT_IB_VCRC_LEN + UCT_IB_DELIM_LEN;

    resource->latency   += tl_overhead_ns;
    resource->bandwidth = (long)((wire_speed * mtu) / (mtu + extra_pkt_len) + 0.5);
    return UCS_OK;
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

ucs_status_t uct_ib_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              size_t tl_hdr_len, uint64_t tl_overhead_ns,
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
        status = uct_ib_device_port_get_resource(dev, port_num, tl_hdr_len,
                                                 tl_overhead_ns, rsc);
        if (status != UCS_OK) {
            ucs_debug("failed to get port info for %s:%d: %s",
                      uct_ib_device_name(dev), port_num,
                      ucs_status_string(status));
            continue;
        }

        ucs_snprintf_zero(rsc->tl_name, UCT_TL_NAME_MAX, "%s", tl_name);
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

static void uct_ib_make_pd_name(char pd_name[UCT_PD_NAME_MAX], struct ibv_device *device)
{
    snprintf(pd_name, UCT_PD_NAME_MAX, "%s/%s", UCT_IB_PD_PREFIX, device->dev_name);
}

static ucs_status_t uct_ib_query_pd_resources(uct_pd_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    uct_pd_resource_desc_t *resources;
    struct ibv_device **device_list;
    ucs_status_t status;
    int i, num_devices;

    /* Get device list from driver */
    device_list = ibv_get_device_list(&num_devices);
    if (device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    resources = ucs_calloc(num_devices, sizeof(*resources), "ib resources");
    if (resources == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_device_list;
    }

    for (i = 0; i < num_devices; ++i) {
        uct_ib_make_pd_name(resources[i].pd_name, device_list[i]);
    }

    *resources_p     = resources;
    *num_resources_p = num_devices;
    status = UCS_OK;

out_free_device_list:
    ibv_free_device_list(device_list);
out:
    return status;
}

static ucs_status_t uct_ib_pd_open(const char *pd_name, uct_pd_h *pd_p)
{
    char tmp_pd_name[UCT_PD_NAME_MAX];
    struct ibv_device **device_list;
    uct_ib_device_t *dev = NULL;
    ucs_status_t status;
    int i, num_devices;

    /* Get device list from driver */
    device_list = ibv_get_device_list(&num_devices);
    if (device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    for (i = 0; i < num_devices; ++i) {
        uct_ib_make_pd_name(tmp_pd_name, device_list[i]);
        if (!strcmp(tmp_pd_name, pd_name)) {
            status = uct_ib_device_create(device_list[i], &dev);
            if (status == UCS_OK) {
                dev->super.component = &uct_ib_pd;
                *pd_p = &dev->super;
            }
            goto out_free_dev_list;
        }
    }

    status = UCS_ERR_NO_DEVICE;

out_free_dev_list:
    ibv_free_device_list(device_list);
out:
    return status;
}

UCT_PD_COMPONENT_DEFINE(uct_ib_pd, UCT_IB_PD_PREFIX,
                        uct_ib_query_pd_resources, uct_ib_pd_open,
                        sizeof(uint32_t), uct_ib_rkey_unpack,
                        (void*)ucs_empty_function /* release */)
