/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_DEVICE_H
#define UCT_IB_DEVICE_H

#include "ib_verbs.h"

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/stats/stats.h>
#include <ucs/debug/assert.h>
#include <ucs/datastruct/khash.h>
#include <ucs/type/spinlock.h>

#include <endian.h>


#define UCT_IB_QPN_ORDER            24  /* How many bits can be an IB QP number */
#define UCT_IB_LRH_LEN              8   /* IB Local routing header */
#define UCT_IB_GRH_LEN              40  /* IB GLobal routing header */
#define UCT_IB_BTH_LEN              12  /* IB base transport header */
#define UCT_IB_ROCE_LEN             14  /* Ethernet header -
                                           6B for Destination MAC +
                                           6B for Source MAC + 2B Type (RoCE) */
#define UCT_IB_DETH_LEN             8   /* IB datagram header */
#define UCT_IB_RETH_LEN             16  /* IB RDMA header */
#define UCT_IB_ATOMIC_ETH_LEN       28  /* IB atomic header */
#define UCT_IB_AETH_LEN             4   /* IB ack */
#define UCT_IB_PAYLOAD_ALIGN        4   /* IB payload padding */
#define UCT_IB_ICRC_LEN             4   /* IB invariant crc footer */
#define UCT_IB_VCRC_LEN             2   /* IB variant crc footer */
#define UCT_IB_DELIM_LEN            2   /* IB wire delimiter */
#define UCT_IB_FDR_PACKET_GAP       64  /* Minimal FDR packet gap */
#define UCT_IB_MAX_MESSAGE_SIZE     (2 << 30) /* Maximal IB message size */
#define UCT_IB_PKEY_PARTITION_MASK  0x7fff /* IB partition number mask */
#define UCT_IB_PKEY_MEMBERSHIP_MASK 0x8000 /* Full/send-only member */
#define UCT_IB_DEV_MAX_PORTS        2
#define UCT_IB_FABRIC_TIME_MAX      32
#define UCT_IB_INVALID_RKEY         0xffffffffu
#define UCT_IB_KEY                  0x1ee7a330
#define UCT_IB_LINK_LOCAL_PREFIX    be64toh(0xfe80000000000000ul) /* IBTA 4.1.1 12a */
#define UCT_IB_SITE_LOCAL_PREFIX    be64toh(0xfec0000000000000ul) /* IBTA 4.1.1 12b */
#define UCT_IB_SITE_LOCAL_MASK      be64toh(0xffffffffffff0000ul) /* IBTA 4.1.1 12b */
#define UCT_IB_DEFAULT_ROCEV2_DSCP  106  /* Default DSCP for RoCE v2 */


enum {
    UCT_IB_DEVICE_STAT_ASYNC_EVENT,
    UCT_IB_DEVICE_STAT_LAST
};


enum {
    UCT_IB_DEVICE_FLAG_MLX4_PRM = UCS_BIT(1),   /* Device supports mlx4 PRM */
    UCT_IB_DEVICE_FLAG_MLX5_PRM = UCS_BIT(2),   /* Device supports mlx5 PRM */
    UCT_IB_DEVICE_FLAG_MELLANOX = UCS_BIT(3),   /* Mellanox device */
    UCT_IB_DEVICE_FLAG_LINK_IB  = UCS_BIT(5),   /* Require only IB */
    UCT_IB_DEVICE_FLAG_DC_V1    = UCS_BIT(6),   /* Device supports DC ver 1 */
    UCT_IB_DEVICE_FLAG_DC_V2    = UCS_BIT(7),   /* Device supports DC ver 2 */
    UCT_IB_DEVICE_FLAG_AV       = UCS_BIT(8),   /* Device supports compact AV */
    UCT_IB_DEVICE_FLAG_DC       = UCT_IB_DEVICE_FLAG_DC_V1 |
                                  UCT_IB_DEVICE_FLAG_DC_V2, /* Device supports DC */
    UCT_IB_DEVICE_FLAG_ODP_IMPLICIT = UCS_BIT(9),
};


/**
 * Flags which specify which address fields are present
 */
enum {
    UCT_IB_ADDRESS_FLAG_LID      = UCS_BIT(0),
    UCT_IB_ADDRESS_FLAG_IF_ID    = UCS_BIT(1),
    UCT_IB_ADDRESS_FLAG_SUBNET16 = UCS_BIT(2),
    UCT_IB_ADDRESS_FLAG_SUBNET64 = UCS_BIT(3),
    UCT_IB_ADDRESS_FLAG_GID  = UCS_BIT(4),
    UCT_IB_ADDRESS_FLAG_LINK_LAYER_IB = UCS_BIT(5),
    UCT_IB_ADDRESS_FLAG_LINK_LAYER_ETH = UCS_BIT(6)
};


/**
 * IB network address
 */
typedef struct uct_ib_address {
    uint8_t            flags;
    /* Following fields appear in this order (if specified by flags).
     * The full gid always appears last:
     * - uint16_t lid
     * - uint64_t if_id
     * - uint16_t subnet16
     * - uint64_t subnet64
     * For RoCE:
     * - uint8_t gid[16]
     */
} UCS_S_PACKED uct_ib_address_t;


/**
 * IB device specification.
 */
typedef struct uct_ib_device_spec {
    uint16_t                    vendor_id;
    uint16_t                    part_id;
    const char                  *name;
    unsigned                    flags;
    uint8_t                     priority;
} uct_ib_device_spec_t;


KHASH_TYPE(uct_ib_ah, struct ibv_ah_attr, struct ibv_ah*);

/**
 * IB device (corresponds to HCA)
 */
typedef struct uct_ib_device {
    struct ibv_context          *ibv_context;    /* Verbs context */
    uct_ib_device_attr          dev_attr;        /* Cached device attributes */
    uint8_t                     first_port;      /* Number of first port (usually 1) */
    uint8_t                     num_ports;       /* Amount of physical ports */
    cpu_set_t                   local_cpus;      /* CPUs local to device */
    int                         numa_node;       /* NUMA node of the device */
    int                         async_events;    /* Whether async events are handled */
    int                         max_zcopy_log_sge; /* Maximum sges log for zcopy am */
    UCS_STATS_NODE_DECLARE(stats);
    struct ibv_port_attr        port_attr[UCT_IB_DEV_MAX_PORTS]; /* Cached port attributes */
    unsigned                    flags;
    uint8_t                     atomic_arg_sizes;
    uint8_t                     atomic_arg_sizes_be;
    uint8_t                     ext_atomic_arg_sizes;
    uint8_t                     ext_atomic_arg_sizes_be;
    uint8_t                     pci_fadd_arg_sizes;
    uint8_t                     pci_cswap_arg_sizes;
    /* AH hash */
    khash_t(uct_ib_ah)          ah_hash;
    ucs_spinlock_t              ah_lock;
} uct_ib_device_t;


/**
 * RoCE version priorities
 */
typedef struct uct_ib_roce_version_desc {
    uint8_t     roce_major;
    uint8_t     roce_minor;
    sa_family_t address_family;
} uct_ib_roce_version_desc_t;


extern const double uct_ib_qp_rnr_time_ms[];


/**
 * Check if a port on a device is active and supports the given flags.
 */
ucs_status_t uct_ib_device_port_check(uct_ib_device_t *dev, uint8_t port_num,
                                      unsigned flags);


/*
 * Helper function to list IB transport resources.
 *
 * @param dev              IB device.
 * @param flags            Transport requirements from IB device (see UCT_IB_RESOURCE_FLAG_xx)
 * @param devices_p        Filled with a pointer to an array of devices.
 * @param num_devices_p    Filled with the number of devices.
 */
ucs_status_t uct_ib_device_query_ports(uct_ib_device_t *dev, unsigned flags,
                                       uct_tl_device_resource_t **devices_p,
                                       unsigned *num_devices_p);


ucs_status_t uct_ib_device_init(uct_ib_device_t *dev,
                                struct ibv_device *ibv_device, int async_events
                                UCS_STATS_ARG(ucs_stats_node_t *stats_parent));

void uct_ib_device_cleanup(uct_ib_device_t *dev);


/**
 * @return device specification.
 */
const uct_ib_device_spec_t* uct_ib_device_spec(uct_ib_device_t *dev);


/**
 * Select the IB gid index to use.
 *
 * @param dev                   IB device.
 * @param port_num              Port number.
 * @param md_config_index       Gid index from the md configuration.
 * @param ib_gid_index          Filled with the selected gid index.
 */
ucs_status_t uct_ib_device_select_gid_index(uct_ib_device_t *dev,
                                            uint8_t port_num,
                                            size_t md_config_index,
                                            uint8_t *ib_gid_index);


/**
 * @return device name.
 */
const char *uct_ib_device_name(uct_ib_device_t *dev);


/**
 * @return whether the port is InfiniBand
 */
int uct_ib_device_is_port_ib(uct_ib_device_t *dev, uint8_t port_num);


/**
 * @return whether the port is RoCE
 */
int uct_ib_device_is_port_roce(uct_ib_device_t *dev, uint8_t port_num);


/**
 * @return 1 if the gid_raw is 0, 0 otherwise.
 */
int uct_ib_device_is_gid_raw_empty(uint8_t *gid_raw);


/**
 * Convert time-in-seconds to IB fabric QP time value
 */
uint8_t uct_ib_to_qp_fabric_time(double time);


/**
 * Convert time-in-seconds to IB fabric RNR time value
 */
uint8_t uct_ib_to_rnr_fabric_time(double time);


/**
 * @return MTU in bytes.
 */
size_t uct_ib_mtu_value(enum ibv_mtu mtu);


/**
 * Modify QP to a given state and check for error
 */
ucs_status_t uct_ib_modify_qp(struct ibv_qp *qp, enum ibv_qp_state state);


/**
 * find device mtu. This function can be used before ib
 * interface is created.
 */
ucs_status_t uct_ib_device_mtu(const char *dev_name, uct_md_h md, int *p_mtu);

ucs_status_t uct_ib_device_find_port(uct_ib_device_t *dev,
                                     const char *resource_dev_name,
                                     uint8_t *p_port_num);

size_t uct_ib_device_odp_max_size(uct_ib_device_t *dev);

int uct_ib_device_odp_has_global_mr(uct_ib_device_t *dev);

const char *uct_ib_wc_status_str(enum ibv_wc_status wc_status);

ucs_status_t uct_ib_device_create_ah_cached(uct_ib_device_t *dev,
                                            struct ibv_ah_attr *ah_attr,
                                            struct ibv_pd *pd,
                                            struct ibv_ah **ah_p);

void uct_ib_device_cleanup_ah_cached(uct_ib_device_t *dev);

static inline struct ibv_port_attr*
uct_ib_device_port_attr(uct_ib_device_t *dev, uint8_t port_num)
{
    return &dev->port_attr[port_num - dev->first_port];
}

static inline int uct_ib_device_has_pci_atomics(uct_ib_device_t *dev)
{
    return !!((dev->pci_fadd_arg_sizes | dev->pci_cswap_arg_sizes) &
              (sizeof(uint32_t) | sizeof(uint64_t)));
}

ucs_status_t uct_ib_device_query_gid(uct_ib_device_t *dev, uint8_t port_num,
                                     unsigned gid_index, union ibv_gid *gid,
                                     int *is_roce_v2);

int uct_ib_get_cqe_size(int cqe_size_min);

static inline ucs_status_t uct_ib_poll_cq(struct ibv_cq *cq, unsigned *count, struct ibv_wc *wcs)
{
    int ret;

    ret = ibv_poll_cq(cq, *count, wcs);
    if (ret <= 0) {
        if (ucs_likely(ret == 0)) {
            return UCS_ERR_NO_PROGRESS;
        }
        ucs_fatal("failed to poll receive CQ %d", ret);
    }

    *count = ret;
    return UCS_OK;
}

#endif
