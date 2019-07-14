/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ib_device.h"
#include "ib_md.h"

#include <ucs/arch/bitops.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/async/async.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <sys/poll.h>
#include <sched.h>


typedef struct {
    union ibv_gid       gid;
    struct {
        uint8_t         major;
        uint8_t         minor;
    } roce_version;
} uct_ib_device_gid_info_t;


/* This table is according to "Encoding for RNR NAK Timer Field"
 * in IBTA specification */
const double uct_ib_qp_rnr_time_ms[] = {
    655.36,  0.01,  0.02,   0.03,   0.04,   0.06,   0.08,   0.12,
      0.16,  0.24,  0.32,   0.48,   0.64,   0.96,   1.28,   1.92,
      2.56,  3.84,  5.12,   7.68,  10.24,  15.36,  20.48,  30.72,
     40.96, 61.44, 81.92, 122.88, 163.84, 245.76, 327.68, 491.52
};


/* use both gid + lid data for key generarion (lid - ib based, gid - RoCE) */
static UCS_F_ALWAYS_INLINE
khint32_t uct_ib_kh_ah_hash_func(struct ibv_ah_attr attr)
{
    return kh_int64_hash_func(attr.grh.dgid.global.subnet_prefix ^
                              attr.grh.dgid.global.interface_id  ^
                              attr.dlid);
}

static UCS_F_ALWAYS_INLINE
int uct_ib_kh_ah_hash_equal(struct ibv_ah_attr a, struct ibv_ah_attr b)
{
    return !memcmp(&a, &b, sizeof(a));
}

KHASH_IMPL(uct_ib_ah, struct ibv_ah_attr, struct ibv_ah*, 1,
           uct_ib_kh_ah_hash_func, uct_ib_kh_ah_hash_equal)


#if ENABLE_STATS
static ucs_stats_class_t uct_ib_device_stats_class = {
    .name           = "",
    .num_counters   = UCT_IB_DEVICE_STAT_LAST,
    .counter_names = {
        [UCT_IB_DEVICE_STAT_ASYNC_EVENT] = "async_event"
    }
};
#endif

static uct_ib_device_spec_t uct_ib_builtin_device_specs[] = {
  {0x02c9, 4099, "ConnectX-3",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX4_PRM, 10},
  {0x02c9, 4103, "ConnectX-3 Pro",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX4_PRM, 11},
  {0x02c9, 4113, "Connect-IB",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V1, 20},
  {0x02c9, 4115, "ConnectX-4",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V1, 30},
  {0x02c9, 4116, "ConnectX-4",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V1, 29},
  {0x02c9, 4117, "ConnectX-4 LX",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V1, 28},
  {0x02c9, 4119, "ConnectX-5",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V2, 38},
  {0x02c9, 4121, "ConnectX-5",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V2, 40},
  {0x02c9, 4120, "ConnectX-5",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V2, 39},
  {0x02c9, 41682, "ConnectX-5",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V2, 37},
  {0x02c9, 4122, "ConnectX-5",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V2, 36},
  {0x02c9, 4123, "ConnectX-6",
   UCT_IB_DEVICE_FLAG_MELLANOX | UCT_IB_DEVICE_FLAG_MLX5_PRM |
   UCT_IB_DEVICE_FLAG_DC_V2, 50},
  {0, 0, "Generic HCA", 0, 0}
};

static void uct_ib_device_get_locailty(const char *dev_name, cpu_set_t *cpu_mask,
                                       int *numa_node)
{
    char *p, buf[ucs_max(CPU_SETSIZE, 10)];
    ucs_status_t status;
    ssize_t nread;
    uint32_t word;
    int base, k;
    long n;

    /* Read list of CPUs close to the device */
    CPU_ZERO(cpu_mask);
    nread = ucs_read_file(buf, sizeof(buf) - 1, 1,
                          "/sys/class/infiniband/%s/device/local_cpus",
                          dev_name);
    if (nread >= 0) {
        buf[CPU_SETSIZE - 1] = '\0';
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

    /* Read NUMA node number */
    status = ucs_read_file_number(&n, 1,
                                  "/sys/class/infiniband/%s/device/numa_node",
                                  dev_name);
    *numa_node = (status == UCS_OK) ? n : -1;
}

static void uct_ib_async_event_handler(int fd, void *arg)
{
    uct_ib_device_t *dev = arg;
    struct ibv_async_event event;
    ucs_log_level_t level;
    char event_info[200];
    int ret;

    ret = ibv_get_async_event(dev->ibv_context, &event);
    if (ret != 0) {
        if (errno != EAGAIN) {
            ucs_warn("ibv_get_async_event() failed: %m");
        }
        return;
    }

    switch (event.event_type) {
    case IBV_EVENT_CQ_ERR:
        snprintf(event_info, sizeof(event_info), "%s on CQ %p",
                 ibv_event_type_str(event.event_type), event.element.cq);
        level = UCS_LOG_LEVEL_ERROR;
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
        level = UCS_LOG_LEVEL_ERROR;
        break;
    case IBV_EVENT_QP_LAST_WQE_REACHED:
        snprintf(event_info, sizeof(event_info), "SRQ-attached QP 0x%x was flushed",
                 event.element.qp->qp_num);
        level = UCS_LOG_LEVEL_DEBUG;
        break;
    case IBV_EVENT_SRQ_ERR:
        level = UCS_LOG_LEVEL_ERROR;
        snprintf(event_info, sizeof(event_info), "%s on SRQ %p",
                 ibv_event_type_str(event.event_type), event.element.srq);
        break;
    case IBV_EVENT_SRQ_LIMIT_REACHED:
        snprintf(event_info, sizeof(event_info), "%s on SRQ %p",
                 ibv_event_type_str(event.event_type), event.element.srq);
        level = UCS_LOG_LEVEL_DEBUG;
        break;
    case IBV_EVENT_DEVICE_FATAL:
    case IBV_EVENT_PORT_ERR:
        snprintf(event_info, sizeof(event_info), "%s on port %d",
                 ibv_event_type_str(event.event_type), event.element.port_num);
        level = UCS_LOG_LEVEL_ERROR;
        break;
    case IBV_EVENT_PORT_ACTIVE:
#if HAVE_DECL_IBV_EVENT_GID_CHANGE
    case IBV_EVENT_GID_CHANGE:
#endif
    case IBV_EVENT_LID_CHANGE:
    case IBV_EVENT_PKEY_CHANGE:
    case IBV_EVENT_SM_CHANGE:
    case IBV_EVENT_CLIENT_REREGISTER:
        snprintf(event_info, sizeof(event_info), "%s on port %d",
                 ibv_event_type_str(event.event_type), event.element.port_num);
        level = UCS_LOG_LEVEL_WARN;
        break;
#if HAVE_STRUCT_IBV_ASYNC_EVENT_ELEMENT_DCT
    case IBV_EXP_EVENT_DCT_KEY_VIOLATION:
        snprintf(event_info, sizeof(event_info), "%s on DCTN 0x%x",
                 "DCT key violation", event.element.dct->dct_num);
        level = UCS_LOG_LEVEL_ERROR;
        break;
    case IBV_EXP_EVENT_DCT_ACCESS_ERR:
        if (event.element.dct) {
            snprintf(event_info, sizeof(event_info), "%s on DCTN 0x%x",
                     "DCT access error", event.element.dct->dct_num);
        } else {
            snprintf(event_info, sizeof(event_info), "%s on DCTN UNKNOWN",
                     "DCT access error");
        }
        level = UCS_LOG_LEVEL_ERROR;
        break;
    case IBV_EXP_EVENT_DCT_REQ_ERR:
        snprintf(event_info, sizeof(event_info), "%s on DCTN 0x%x",
                 "DCT requester error", event.element.dct->dct_num);
        level = UCS_LOG_LEVEL_ERROR;
        break;
#endif
    default:
        snprintf(event_info, sizeof(event_info), "%s (%d)",
                 ibv_event_type_str(event.event_type), event.event_type);
        level = UCS_LOG_LEVEL_INFO;
        break;
    };

    UCS_STATS_UPDATE_COUNTER(dev->stats, UCT_IB_DEVICE_STAT_ASYNC_EVENT, +1);
    ucs_log(level, "IB Async event on %s: %s", uct_ib_device_name(dev), event_info);
    ibv_ack_async_event(&event);
}

ucs_status_t uct_ib_device_init(uct_ib_device_t *dev,
                                struct ibv_device *ibv_device, int async_events
                                UCS_STATS_ARG(ucs_stats_node_t *stats_parent))
{
    ucs_status_t status;
    uint8_t i;
    int ret;

    dev->async_events = async_events;

    /* Check device type*/
    switch (ibv_device->node_type) {
    case IBV_NODE_SWITCH:
        dev->first_port = 0;
        dev->num_ports  = 1;
        break;
    case IBV_NODE_CA:
    default:
        dev->first_port = 1;
        dev->num_ports  = IBV_DEV_ATTR(dev, phys_port_cnt);
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
    uct_ib_device_get_locailty(ibv_get_device_name(ibv_device), &dev->local_cpus,
                               &dev->numa_node);

    /* Query all ports */
    for (i = 0; i < dev->num_ports; ++i) {
        ret = ibv_query_port(dev->ibv_context, i + dev->first_port,
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

    /* Register to IB async events */
    if (dev->async_events) {
        status = ucs_async_set_event_handler(UCS_ASYNC_THREAD_LOCK_TYPE,
                                             dev->ibv_context->async_fd,
                                             UCS_EVENT_SET_EVREAD,
                                             uct_ib_async_event_handler, dev,
                                             NULL);
        if (status != UCS_OK) {
            goto err_release_stats;
        }
    }

    kh_init_inplace(uct_ib_ah, &dev->ah_hash);
    ucs_spinlock_init(&dev->ah_lock);

    ucs_debug("initialized device '%s' (%s) with %d ports", uct_ib_device_name(dev),
              ibv_node_type_str(ibv_device->node_type),
              dev->num_ports);
    return UCS_OK;

err_release_stats:
    UCS_STATS_NODE_FREE(dev->stats);
err_free_context:
    ibv_close_device(dev->ibv_context);
    return status;
}

void uct_ib_device_cleanup_ah_cached(uct_ib_device_t *dev)
{
    struct ibv_ah *ah;

    kh_foreach_value(&dev->ah_hash, ah, ibv_destroy_ah(ah));
}

void uct_ib_device_cleanup(uct_ib_device_t *dev)
{
    ucs_debug("destroying ib device %s", uct_ib_device_name(dev));

    kh_destroy_inplace(uct_ib_ah, &dev->ah_hash);
    ucs_spinlock_destroy(&dev->ah_lock);

    if (dev->async_events) {
        ucs_async_remove_handler(dev->ibv_context->async_fd, 1);
    }
    UCS_STATS_NODE_FREE(dev->stats);
    ibv_close_device(dev->ibv_context);
}

static inline int uct_ib_device_spec_match(uct_ib_device_t *dev,
                                           const uct_ib_device_spec_t *spec)
{
    return (spec->vendor_id == IBV_DEV_ATTR(dev, vendor_id)) &&
           (spec->part_id   == IBV_DEV_ATTR(dev, vendor_part_id));
}

const uct_ib_device_spec_t* uct_ib_device_spec(uct_ib_device_t *dev)
{
    uct_ib_md_t *md = ucs_container_of(dev, uct_ib_md_t, dev);
    uct_ib_device_spec_t *spec;

    /* search through devices specified in the configuration */
    for (spec = md->custom_devices.specs;
         spec < md->custom_devices.specs + md->custom_devices.count; ++spec) {
        if (uct_ib_device_spec_match(dev, spec)) {
            return spec;
        }
    }

    /* search through built-in list of device specifications */
    spec = uct_ib_builtin_device_specs;
    while ((spec->vendor_id != 0) && !uct_ib_device_spec_match(dev, spec)) {
        ++spec;
    }
    return spec; /* if no match is found, return the last entry, which contains
                    default settings for unknown devices */
}

static size_t uct_ib_device_get_ib_gid_index(uct_ib_md_t *md)
{
    if (md->config.gid_index == UCS_ULUNITS_AUTO) {
        return UCT_IB_MD_DEFAULT_GID_INDEX;
    } else {
        return md->config.gid_index;
    }
}

static int uct_ib_device_is_iwarp(uct_ib_device_t *dev)
{
    return dev->ibv_context->device->transport_type == IBV_TRANSPORT_IWARP;
}

ucs_status_t uct_ib_device_port_check(uct_ib_device_t *dev, uint8_t port_num,
                                      unsigned flags)
{
    uct_ib_md_t *md = ucs_container_of(dev, uct_ib_md_t, dev);
    const uct_ib_device_spec_t *dev_info;
    uint8_t required_dev_flags;
    ucs_status_t status;
    union ibv_gid gid;
    int is_roce_v2;

    if (port_num < dev->first_port || port_num >= dev->first_port + dev->num_ports) {
        return UCS_ERR_NO_DEVICE;
    }

    if (uct_ib_device_port_attr(dev, port_num)->state != IBV_PORT_ACTIVE) {
        ucs_trace("%s:%d is not active (state: %d)", uct_ib_device_name(dev),
                  port_num, uct_ib_device_port_attr(dev, port_num)->state);
        return UCS_ERR_UNREACHABLE;
    }

    if (uct_ib_device_is_iwarp(dev)) {
        /* TODO: enable it when support is ready */
        ucs_debug("iWarp device %s is not supported", uct_ib_device_name(dev));
        return UCS_ERR_UNSUPPORTED;
    }

    if (!uct_ib_device_is_port_ib(dev, port_num) && (flags & UCT_IB_DEVICE_FLAG_LINK_IB)) {
        ucs_debug("%s:%d is not IB link layer", uct_ib_device_name(dev),
                  port_num);
        return UCS_ERR_UNSUPPORTED;
    }

    if (flags & UCT_IB_DEVICE_FLAG_DC) {
        if (!IBV_DEVICE_HAS_DC(dev)) {
            ucs_trace("%s:%d does not support DC", uct_ib_device_name(dev), port_num);
            return UCS_ERR_UNSUPPORTED;
        }
    }

    /* check generic device flags */
    dev_info           = uct_ib_device_spec(dev);
    required_dev_flags = flags & (UCT_IB_DEVICE_FLAG_MLX4_PRM |
                                  UCT_IB_DEVICE_FLAG_MLX5_PRM);
    if (!ucs_test_all_flags(dev_info->flags, required_dev_flags)) {
        ucs_trace("%s:%d (%s) does not support flags 0x%x", uct_ib_device_name(dev),
                  port_num, dev_info->name, required_dev_flags);
        return UCS_ERR_UNSUPPORTED;
    }

    if (md->check_subnet_filter && uct_ib_device_is_port_ib(dev, port_num)) {
        status = uct_ib_device_query_gid(dev, port_num,
                                         uct_ib_device_get_ib_gid_index(md), &gid,
                                         &is_roce_v2);
        if (status) {
            return status;
        }

        ucs_assert(is_roce_v2 == 0);
        if (md->subnet_filter != gid.global.subnet_prefix) {
            ucs_trace("%s:%d subnet_prefix does not match",
                      uct_ib_device_name(dev), port_num);
            return UCS_ERR_UNSUPPORTED;
        }
    }

    return UCS_OK;
}

static int uct_ib_device_is_addr_ipv4_mcast(const struct in6_addr *raw,
                                            const uint32_t addr_last_bits)
{
    /* IPv4 encoded multicast addresses */
    return (raw->s6_addr32[0] == htonl(0xff0e0000)) &&
           !(raw->s6_addr32[1] | addr_last_bits);
}

static sa_family_t uct_ib_device_get_addr_family(union ibv_gid *gid, int gid_index)
{
    const struct in6_addr *raw    = (struct in6_addr *)gid->raw;
    const uint32_t addr_last_bits = raw->s6_addr32[2] ^ htonl(0x0000ffff);
    char p[128];

    ucs_debug("testing addr_family on gid index %d: %s",
              gid_index, inet_ntop(AF_INET6, gid, p, sizeof(p)));

    if (!((raw->s6_addr32[0] | raw->s6_addr32[1]) | addr_last_bits) ||
        uct_ib_device_is_addr_ipv4_mcast(raw, addr_last_bits)) {
        return AF_INET;
    } else {
        return AF_INET6;
    }
}

static ucs_status_t
uct_ib_device_query_gid_info(uct_ib_device_t *dev, uint8_t port_num,
                             unsigned gid_index, uct_ib_device_gid_info_t *info)
{
    int ret;

#if HAVE_DECL_IBV_EXP_QUERY_GID_ATTR
    struct ibv_exp_gid_attr attr;

    attr.comp_mask = IBV_EXP_QUERY_GID_ATTR_TYPE | IBV_EXP_QUERY_GID_ATTR_GID;
    ret = ibv_exp_query_gid_attr(dev->ibv_context, port_num, gid_index, &attr);
    if (ret == 0) {
        info->gid = attr.gid;
        switch (attr.type) {
        case IBV_EXP_IB_ROCE_V1_GID_TYPE:
            info->roce_version.major = 1;
            info->roce_version.minor = 0;
            return UCS_OK;
        case IBV_EXP_ROCE_V1_5_GID_TYPE:
            info->roce_version.major = 1;
            info->roce_version.minor = 5;
            return UCS_OK;
        case IBV_EXP_ROCE_V2_GID_TYPE:
            info->roce_version.major = 2;
            info->roce_version.minor = 0;
            return UCS_OK;
        default:
            ucs_error("Invalid GID[%d] type on %s:%d: %d",
                      gid_index, uct_ib_device_name(dev), port_num, attr.type);
            return UCS_ERR_IO_ERROR;
        }
    }
#else
    ret = ibv_query_gid(dev->ibv_context, port_num, gid_index, &info->gid);
    if (ret == 0) {
        info->roce_version.major = 1;
        info->roce_version.minor = 0;
        return UCS_OK;
    }
#endif
    ucs_error("ibv_query_gid(dev=%s port=%d index=%d) failed: %m",
              uct_ib_device_name(dev), port_num, gid_index);
    return UCS_ERR_INVALID_PARAM;
}

static ucs_status_t uct_ib_device_set_roce_gid_index(uct_ib_device_t *dev,
                                                     uint8_t port_num,
                                                     uint8_t *gid_index)
{
    static const uct_ib_roce_version_desc_t roce_prio[] = {
        {2, 0, AF_INET},
        {2, 0, AF_INET6},
        {1, 0, AF_INET},
        {1, 0, AF_INET6}
    };
    int gid_tbl_len         = uct_ib_device_port_attr(dev, port_num)->gid_tbl_len;
    ucs_status_t status     = UCS_OK;
    int priorities_arr_len  = ucs_static_array_size(roce_prio);
    uct_ib_device_gid_info_t gid_info;
    int i, prio_idx;

    /* search for matching GID table entries, accroding to the order defined
     * in priorities array
     */
    for (prio_idx = 0; prio_idx < priorities_arr_len; prio_idx++) {
        for (i = 0; i < gid_tbl_len; i++) {
            status = uct_ib_device_query_gid_info(dev, port_num, i, &gid_info);
            if (status != UCS_OK) {
                goto out;
            }

            if ((roce_prio[prio_idx].roce_major     == gid_info.roce_version.major) &&
                (roce_prio[prio_idx].roce_minor     == gid_info.roce_version.minor) &&
                (roce_prio[prio_idx].address_family ==
                                uct_ib_device_get_addr_family(&gid_info.gid, i))) {
                *gid_index = i;
                goto out_print;
            }
        }
    }

    *gid_index = UCT_IB_MD_DEFAULT_GID_INDEX;

out_print:
    ucs_debug("%s:%d using gid_index %d", uct_ib_device_name(dev), port_num,
              *gid_index);
out:
    return status;
}

int uct_ib_device_is_port_ib(uct_ib_device_t *dev, uint8_t port_num)
{
#if HAVE_DECL_IBV_LINK_LAYER_INFINIBAND
    return uct_ib_device_port_attr(dev, port_num)->link_layer == IBV_LINK_LAYER_INFINIBAND;
#else
    return 1;
#endif
}

int uct_ib_device_is_port_roce(uct_ib_device_t *dev, uint8_t port_num)
{
    return IBV_PORT_IS_LINK_LAYER_ETHERNET(uct_ib_device_port_attr(dev, port_num));
}

ucs_status_t uct_ib_device_select_gid_index(uct_ib_device_t *dev,
                                            uint8_t port_num,
                                            size_t md_config_index,
                                            uint8_t *gid_index)
{
    ucs_status_t status = UCS_OK;

    if (md_config_index == UCS_ULUNITS_AUTO) {
        if (uct_ib_device_is_port_roce(dev, port_num)) {
            status = uct_ib_device_set_roce_gid_index(dev, port_num, gid_index);
        } else {
            *gid_index = UCT_IB_MD_DEFAULT_GID_INDEX;
        }
    } else {
        *gid_index = md_config_index;
    }

    return status;
}

const char *uct_ib_device_name(uct_ib_device_t *dev)
{
    return ibv_get_device_name(dev->ibv_context->device);
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

uint8_t uct_ib_to_qp_fabric_time(double time)
{
    double to;

    to = log(time / 4.096e-6) / log(2.0);
    if (to < 1) {
        return 1; /* Very small timeout */
    } else if ((long)(to + 0.5) >= UCT_IB_FABRIC_TIME_MAX) {
        return 0; /* No timeout */
    } else {
        return (long)(to + 0.5);
    }
}

uint8_t uct_ib_to_rnr_fabric_time(double time)
{
    double time_ms = time * UCS_MSEC_PER_SEC;
    uint8_t index, next_index;
    double avg_ms;

    for (index = 1; index < UCT_IB_FABRIC_TIME_MAX; index++) {
        next_index = (index + 1) % UCT_IB_FABRIC_TIME_MAX;

        if (time_ms <= uct_ib_qp_rnr_time_ms[next_index]) {
            avg_ms = (uct_ib_qp_rnr_time_ms[index] +
                      uct_ib_qp_rnr_time_ms[next_index]) * 0.5;

            if (time_ms < avg_ms) {
                /* return previous index */
                return index;
            } else {
                /* return current index */
                return next_index;
            }
        }
    }

    return 0; /* this is a special value that means the maximum value */
}

ucs_status_t uct_ib_modify_qp(struct ibv_qp *qp, enum ibv_qp_state state)
{
    struct ibv_qp_attr qp_attr;

    ucs_debug("modify QP 0x%x to state %d", qp->qp_num, state);
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = state;
    if (ibv_modify_qp(qp, &qp_attr, IBV_QP_STATE)) {
        ucs_warn("modify qp 0x%x to state %d failed: %m", qp->qp_num, state);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
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

ucs_status_t uct_ib_device_mtu(const char *dev_name, uct_md_h md, int *p_mtu)
{

    uct_ib_device_t *dev = &ucs_derived_of(md, uct_ib_md_t)->dev;
    uint8_t port_num;
    ucs_status_t status;

    status = uct_ib_device_find_port(dev, dev_name, &port_num);
    if (status != UCS_OK) {
        return status;
    }

    *p_mtu = uct_ib_mtu_value(uct_ib_device_port_attr(dev, port_num)->active_mtu);
    return UCS_OK;
}

int uct_ib_device_is_gid_raw_empty(uint8_t *gid_raw)
{
    return (*(uint64_t *)gid_raw == 0) && (*(uint64_t *)(gid_raw + 8) == 0);
}

ucs_status_t uct_ib_device_query_gid(uct_ib_device_t *dev, uint8_t port_num,
                                     unsigned gid_index, union ibv_gid *gid,
                                     int *is_roce_v2)
{
    uct_ib_device_gid_info_t gid_info;
    ucs_status_t status;

    status = uct_ib_device_query_gid_info(dev, port_num, gid_index, &gid_info);
    if (status != UCS_OK) {
        return status;
    }

    if (uct_ib_device_is_gid_raw_empty(gid_info.gid.raw)) {
        ucs_error("Invalid gid[%d] on %s:%d", gid_index,
                  uct_ib_device_name(dev), port_num);
        return UCS_ERR_INVALID_ADDR;
    }

    *gid        = gid_info.gid;
    *is_roce_v2 = uct_ib_device_is_port_roce(dev, port_num) &&
                  (gid_info.roce_version.major >= 2);
    return UCS_OK;
}

size_t uct_ib_device_odp_max_size(uct_ib_device_t *dev)
{
#if HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_ODP_CAPS
    const struct ibv_exp_device_attr *dev_attr = &dev->dev_attr;
    uint32_t required_ud_odp_caps = IBV_EXP_ODP_SUPPORT_SEND;
    uint32_t required_rc_odp_caps = IBV_EXP_ODP_SUPPORT_SEND |
                                    IBV_EXP_ODP_SUPPORT_WRITE |
                                    IBV_EXP_ODP_SUPPORT_READ;

    if (RUNNING_ON_VALGRIND ||
        !IBV_EXP_HAVE_ODP(dev_attr) ||
        !ucs_test_all_flags(IBV_EXP_ODP_CAPS(dev_attr, rc), required_rc_odp_caps) ||
        !ucs_test_all_flags(IBV_EXP_ODP_CAPS(dev_attr, ud), required_ud_odp_caps))
    {
        return 0;
    }

    if (IBV_DEVICE_HAS_DC(dev)
#  if HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_ODP_CAPS_PER_TRANSPORT_CAPS_DC_ODP_CAPS
        && !ucs_test_all_flags(IBV_EXP_ODP_CAPS(dev_attr, dc), required_rc_odp_caps)
#  endif
        )
    {
        return 0;
    }

#  if HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_ODP_MR_MAX_SIZE
    return dev_attr->odp_mr_max_size;
#  else
    return 1ul << 28; /* Limit ODP to 256 MB by default */
#  endif /* HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_ODP_MR_MAX_SIZE */

#else
    return 0;
#endif /* HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_ODP_CAPS */
}

static ucs_status_t
uct_ib_device_parse_fw_ver_triplet(uct_ib_device_t *dev, unsigned *major,
                                   unsigned *minor, unsigned *release)
{
    int ret;

    ret = sscanf(IBV_DEV_ATTR(dev, fw_ver), "%u.%u.%u", major, minor, release);
    if (ret != 3) {
        ucs_debug("failed to parse firmware version string '%s'",
                  IBV_DEV_ATTR(dev, fw_ver));
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

int uct_ib_device_odp_has_global_mr(uct_ib_device_t *dev)
{
    unsigned fw_major, fw_minor, fw_release;
    ucs_status_t status;

    if (!uct_ib_device_odp_max_size(dev)) {
        return 0;
    }

#if HAVE_DECL_IBV_EXP_ODP_SUPPORT_IMPLICIT
    if (!(dev->dev_attr.odp_caps.general_odp_caps & IBV_EXP_ODP_SUPPORT_IMPLICIT)) {
        return 0;
    }
#endif

    if (uct_ib_device_spec(dev)->flags & UCT_IB_DEVICE_FLAG_MELLANOX) {
        status = uct_ib_device_parse_fw_ver_triplet(dev, &fw_major, &fw_minor,
                                                    &fw_release);
        if (status != UCS_OK) {
            return 0;
        }

        if ((fw_major < 12) || (fw_minor < 21)) {
            return 0;
        } else if (fw_minor == 21) {
            return (fw_release >= 2031) && (fw_release <= 2099);
        } else if (fw_minor == 22) {
            return (fw_release >= 84);
        } else {
            return 1;
        }
    }

    return 1;
}

const char *uct_ib_wc_status_str(enum ibv_wc_status wc_status)
{
    return ibv_wc_status_str(wc_status);
}

static ucs_status_t uct_ib_device_create_ah(uct_ib_device_t *dev,
                                            struct ibv_ah_attr *ah_attr,
                                            struct ibv_pd *pd,
                                            struct ibv_ah **ah_p)
{
    char buf[128];
    char *p, *endp;
    struct ibv_ah *ah;

    ah = ibv_create_ah(pd, ah_attr);
    if (ah == NULL) {
        p    = buf;
        endp = buf + sizeof(buf);
        snprintf(p, endp - p, "dlid=%d sl=%d port=%d src_path_bits=%d",
                 ah_attr->dlid, ah_attr->sl,
                 ah_attr->port_num, ah_attr->src_path_bits);
        p += strlen(p);

        if (ah_attr->is_global) {
            snprintf(p, endp - p, " dgid=");
            p += strlen(p);
            inet_ntop(AF_INET6, &ah_attr->grh.dgid, p, endp - p);
            p += strlen(p);
            snprintf(p, endp - p, " sgid_index=%d traffic_class=%d",
                     ah_attr->grh.sgid_index, ah_attr->grh.traffic_class);
        }

        ucs_error("ibv_create_ah(%s) failed: %m", buf);
        return UCS_ERR_INVALID_ADDR;
    }

    *ah_p = ah;
    return UCS_OK;
}

ucs_status_t uct_ib_device_create_ah_cached(uct_ib_device_t *dev,
                                            struct ibv_ah_attr *ah_attr,
                                            struct ibv_pd *pd,
                                            struct ibv_ah **ah_p)
{
    ucs_status_t status = UCS_OK;
    khiter_t iter;
    int ret;

    ucs_spin_lock(&dev->ah_lock);

    /* looking for existing AH with same attributes */
    iter = kh_get(uct_ib_ah, &dev->ah_hash, *ah_attr);
    if (iter == kh_end(&dev->ah_hash)) {
        /* new AH */
        status = uct_ib_device_create_ah(dev, ah_attr, pd, ah_p);
        if (status != UCS_OK) {
            goto unlock;
        }

        /* store AH in hash */
        iter = kh_put(uct_ib_ah, &dev->ah_hash, *ah_attr, &ret);

        /* failed to store - rollback */
        if (iter == kh_end(&dev->ah_hash)) {
            ibv_destroy_ah(*ah_p);
            status = UCS_ERR_NO_MEMORY;
            goto unlock;
        }

        kh_value(&dev->ah_hash, iter) = *ah_p;
    } else {
        /* found existing AH */
        *ah_p = kh_value(&dev->ah_hash, iter);
    }

unlock:
    ucs_spin_unlock(&dev->ah_lock);
    return status;
}

int uct_ib_get_cqe_size(int cqe_size_min)
{
    static int cqe_size_max = -1;
    int cqe_size;

    if (cqe_size_max == -1) {
#ifdef __aarch64__
        char arm_board_vendor[128];
        ucs_aarch64_cpuid_t cpuid;
        ucs_aarch64_cpuid(&cpuid);

        arm_board_vendor[0] = '\0';
        ucs_read_file(arm_board_vendor, sizeof(arm_board_vendor), 1,
                      "/sys/devices/virtual/dmi/id/board_vendor");
        ucs_debug("arm_board_vendor is '%s'", arm_board_vendor);

        cqe_size_max = ((strcasestr(arm_board_vendor, "Huawei")) &&
                        (cpuid.implementer == 0x41) && (cpuid.architecture == 8) &&
                        (cpuid.variant == 0)        && (cpuid.part == 0xd08)     &&
                        (cpuid.revision == 2))
                       ? 64 : 128;
#else
        cqe_size_max = 128;
#endif
        ucs_debug("max IB CQE size is %d", cqe_size_max);
    }

    /* Set cqe size according to inline size and cache line size. */
    cqe_size = ucs_max(cqe_size_min, UCS_SYS_CACHE_LINE_SIZE);
    cqe_size = ucs_max(cqe_size, 64);  /* at least 64 */
    cqe_size = ucs_min(cqe_size, cqe_size_max);

    return cqe_size;
}
