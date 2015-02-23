/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#define _GNU_SOURCE /* for CPU_ZERO/CPU_SET in sched.h */
#include "ib_device.h"
#include "ib_context.h"

#include <uct/tl/context.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/async/async.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <sys/poll.h>
#include <sched.h>


#define UCT_IB_RKEY_MAGIC        0x69626962  /* ibib *(const uint32_t*)"ibib" */
#define UCT_IB_MEM_ACCESS_FLAGS  (IBV_ACCESS_LOCAL_WRITE | \
                                  IBV_ACCESS_REMOTE_WRITE | \
                                  IBV_ACCESS_REMOTE_READ | \
                                  IBV_ACCESS_REMOTE_ATOMIC)


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

static ucs_status_t uct_ib_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr)
{
    pd_attr->rkey_packed_size  = sizeof(uint32_t) * 2;
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_map(uct_pd_h pd, void **address_p,
                                   size_t *length_p, unsigned flags,
                                   uct_lkey_t *lkey_p UCS_MEMTRACK_ARG)
{
    uct_ib_device_t *dev = ucs_derived_of(pd, uct_ib_device_t);
    struct ibv_mr *mr;

    if (*address_p == NULL) {
        if (IBV_EXP_HAVE_CONTIG_PAGES(&dev->dev_attr)) {
            struct ibv_exp_reg_mr_in in = {
                dev->pd,
                NULL,
                ucs_memtrack_adjust_alloc_size(*length_p),
                UCT_IB_MEM_ACCESS_FLAGS | IBV_EXP_ACCESS_ALLOCATE_MR,
                0
            };

            mr = ibv_exp_reg_mr(&in);
            if (mr == NULL) {
                ucs_error("ibv_exp_reg_mr(NULL, length=%Zu) failed: %m", *length_p);
                return UCS_ERR_IO_ERROR;
            }

            *address_p = mr->addr;
            *length_p  = mr->length;
            ucs_memtrack_allocated(address_p, length_p UCS_MEMTRACK_VAL);
        } else {
            ucs_error("Contig pages allocator not supported");
            return UCS_ERR_UNSUPPORTED;
        }
    } else {
        mr = ibv_reg_mr(dev->pd, *address_p, *length_p, UCT_IB_MEM_ACCESS_FLAGS);
        if (mr == NULL) {
            ucs_error("ibv_reg_mr(address=%p, length=%zu) failed: %m", *address_p,
                      *length_p);
            return UCS_ERR_IO_ERROR;
        }
    }

    *lkey_p = (uintptr_t)mr;
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_unmap(uct_pd_h pd, uct_lkey_t lkey)
{
    struct ibv_mr *mr = uct_ib_lkey_mr(lkey);
    void UCS_V_UNUSED *address = mr->addr;
    int ret;

    ucs_memtrack_releasing(&address);
    ret = ibv_dereg_mr(mr);
    if (ret != 0) {
        ucs_error("ibv_dereg_mr() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_ib_rkey_pack(uct_pd_h pd, uct_lkey_t lkey,
                                     void *rkey_buffer)
{
    struct ibv_mr *mr = uct_ib_lkey_mr(lkey);
    uint32_t *ptr = rkey_buffer;

    *(ptr++) = UCT_IB_RKEY_MAGIC;
    *(ptr++) = mr->rkey;
    return UCS_OK;
}

ucs_status_t uct_ib_rkey_unpack(uct_context_h context, void *rkey_buffer,
                                uct_rkey_bundle_t *rkey_ob)
{
    uint32_t *ptr = rkey_buffer;
    uint32_t magic;

    magic = *(ptr++);
    if (magic != UCT_IB_RKEY_MAGIC) {
        return UCS_ERR_UNSUPPORTED;
    }

    rkey_ob->rkey = *(ptr++);
    rkey_ob->type = (void*)ucs_empty_function;
    return UCS_OK;
}

uct_pd_ops_t uct_ib_pd_ops = {
    .query        = uct_ib_pd_query,
    .mem_map      = uct_ib_mem_map,
    .mem_unmap    = uct_ib_mem_unmap,
    .rkey_pack    = uct_ib_rkey_pack,
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

    ucs_warn("IB Async event on %s: %s", uct_ib_device_name(dev), event_info);
    ibv_ack_async_event(&event);
}

ucs_status_t uct_ib_device_create(uct_context_h context,
                                  struct ibv_device *ibv_device,
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
    dev->super.context = context;
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

    status = ucs_sys_fcntl_modfl(dev->ibv_context->async_fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_destroy_pd;
    }

    /* Register to IB async events
     * TODO have option to set async mode as signal/thread.
     */
    status = ucs_async_set_event_handler(UCS_ASYNC_MODE_THREAD,
                                         dev->ibv_context->async_fd,
                                         POLLIN, uct_ib_async_event_handler, dev, NULL);
    if (status != UCS_OK) {
        goto err_destroy_pd;
    }

    ucs_debug("created device '%s' (%s) with %d ports", uct_ib_device_name(dev),
              ibv_node_type_str(ibv_device->node_type),
              dev->num_ports);

    *dev_p = dev;
    return UCS_OK;

err_destroy_pd:
    ibv_dealloc_pd(dev->pd);
err_free_device:
    ucs_free(dev);
err_free_context:
    ibv_close_device(ibv_context);
err:
    return status;
}

void uct_ib_device_destroy(uct_ib_device_t *dev)
{
    ucs_async_unset_event_handler(dev->ibv_context->async_fd);
    ibv_dealloc_pd(dev->pd);
    ibv_close_device(dev->ibv_context);
    ucs_free(dev);
}

ucs_status_t uct_ib_device_port_check(uct_ib_device_t *dev, uint8_t port_num,
                                      unsigned flags)
{
    if (port_num < dev->first_port || port_num >= dev->first_port + dev->num_ports) {
        return UCS_ERR_NO_DEVICE;
    }

    if (uct_ib_device_port_attr(dev, port_num)->state != IBV_PORT_ACTIVE) {
        return UCS_ERR_UNREACHABLE;
    }

    if (flags & UCT_IB_RESOURCE_FLAG_DC) {
        if (!IBV_DEVICE_HAS_DC(&dev->dev_attr)) {
            return UCS_ERR_UNSUPPORTED;
        }
    }

    if (flags & UCT_IB_RESOURCE_FLAG_MLX4_PRM) {
        return UCS_ERR_UNSUPPORTED; /* Unsupported yet */
    }

    if (flags & UCT_IB_RESOURCE_FLAG_MLX5_PRM) {
        /* TODO list all devices with their flags */
        if (dev->dev_attr.vendor_id != 0x02c9 || dev->dev_attr.vendor_part_id != 4113) {
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

size_t uct_ib_device_port_active_mtu(uct_ib_device_t *dev, uint8_t port_num)
{
    switch (uct_ib_device_port_attr(dev, port_num)->active_mtu) {
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
    ucs_fatal("Invalid port MTU value (%d)",
              uct_ib_device_port_attr(dev, port_num)->active_mtu);
}

ucs_status_t uct_ib_device_port_get_resource(uct_ib_device_t *dev, uint8_t port_num,
                                             size_t tl_hdr_len, uint64_t tl_overhead_ns,
                                             uct_resource_desc_t *resource)
{
    static unsigned ib_port_widths[] = {
        [0] = 1,
        [1] = 4,
        [2] = 8,
        [3] = 12
    };
    struct sockaddr_in6 *in6_addr;
    double encoding, signal_rate, wire_speed;
    union ibv_gid gid;
    unsigned active_width;
    size_t extra_pkt_len;
    size_t mtu;
    int ret;

    /* HCA:Port is the hardware resource name */
    ucs_snprintf_zero(resource->dev_name, sizeof(resource->dev_name), "%s:%d",
                      uct_ib_device_name(dev), port_num);

    /* Port network address */
    if (uct_ib_device_is_port_ib(dev, port_num)) {
        /*
         * For Infiniband, take the subnet prefix.
         */
        in6_addr = (struct sockaddr_in6 *)&(resource->subnet_addr);
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
    } else {
        return UCS_ERR_UNSUPPORTED;
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

    mtu           = uct_ib_device_port_active_mtu(dev, port_num);
    wire_speed    = (signal_rate * 1e9 * encoding *
                     ib_port_widths[ucs_ilog2(active_width)]) / 8.0;
    extra_pkt_len = UCT_IB_LRH_LEN + UCT_IB_BTH_LEN + tl_hdr_len +
                    UCT_IB_ICRC_LEN + UCT_IB_VCRC_LEN + UCT_IB_DELIM_LEN;

    resource->latency   += tl_overhead_ns;
    resource->bandwidth = (long)((wire_speed * mtu) / (mtu + extra_pkt_len) + 0.5);
    return UCS_OK;
}
