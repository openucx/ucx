/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_DEVICE_H
#define UCT_IB_DEVICE_H

#include "ib_verbs.h"

#include <uct/api/uct.h>
#include <ucs/stats/stats.h>
#include <ucs/debug/assert.h>

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
#define UCT_IB_INVALID_RKEY         0xffffffffu
#define UCT_IB_KEY                  0x1ee7a330
#define UCT_IB_LINK_LOCAL_PREFIX    be64toh(0xfe80000000000000ul) /* IBTA 4.1.1 12a */
#define UCT_IB_SITE_LOCAL_PREFIX    be64toh(0xfec0000000000000ul) /* IBTA 4.1.1 12b */
#define UCT_IB_SITE_LOCAL_MASK      be64toh(0xffffffffffff0000ul) /* IBTA 4.1.1 12b */


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
    UCT_IB_DEVICE_FLAG_DC       = UCT_IB_DEVICE_FLAG_DC_V1 |
                                  UCT_IB_DEVICE_FLAG_DC_V2 /* Device supports DC */
};


typedef enum {
    UCT_IB_ADDRESS_TYPE_LINK_LOCAL,   /* Subnet-local address */
    UCT_IB_ADDRESS_TYPE_SITE_LOCAL,   /* Site local, 16-bit subnet prefix */
    UCT_IB_ADDRESS_TYPE_GLOBAL,       /* Global, 64-bit subnet prefix */
    UCT_IB_ADDRESS_TYPE_ETH,          /* RoCE  address */
    UCT_IB_ADDRESS_TYPE_LAST
} uct_ib_address_type_t;


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


/**
 * IB device (corresponds to HCA)
 */
typedef struct uct_ib_device {
    struct ibv_context          *ibv_context;    /* Verbs context */
    struct ibv_exp_device_attr  dev_attr;        /* Cached device attributes */
    uint8_t                     first_port;      /* Number of first port (usually 1) */
    uint8_t                     num_ports;       /* Amount of physical ports */
    cpu_set_t                   local_cpus;      /* CPUs local to device */
    int                         numa_node;       /* NUMA node of the device */
    int                         async_events;    /* Whether async events are handled */
    int                         max_zcopy_log_sge; /* Maximum sges log for zcopy am */
    UCS_STATS_NODE_DECLARE(stats);
    struct ibv_exp_port_attr    port_attr[UCT_IB_DEV_MAX_PORTS]; /* Cached port attributes */
} uct_ib_device_t;


#if HAVE_DECL_IBV_EXP_QUERY_GID_ATTR
/**
 * RoCE version description
 */
typedef struct uct_ib_roce_version_desc {
    enum ibv_exp_roce_gid_type type;
    int address_family;
    int priority;
} uct_ib_roce_version_desc_t;
#endif


/**
 * Check if a port on a device is active and supports the given flags.
 */
ucs_status_t uct_ib_device_port_check(uct_ib_device_t *dev, uint8_t port_num,
                                      unsigned flags);


/*
 * Helper function to list IB transport resources.
 *
 * @param dev              IB device.
 * @param tl_name          Transport name.
 * @param flags            Transport requirements from IB device (see UCT_IB_RESOURCE_FLAG_xx)
 * @param resources_p      Filled with a pointer to an array of resources.
 * @param num_resources_p  Filled with the number of resources.
 */
ucs_status_t uct_ib_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);


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
 * @return 1 if the port is InfiniBand, 0 if the port is Ethernet.
 */
int uct_ib_device_is_port_ib(uct_ib_device_t *dev, uint8_t port_num);


/**
 * @return 1 if the gid_raw is 0, 0 otherwise.
 */
int uct_ib_device_is_gid_raw_empty(uint8_t *gid_raw);


/**
 * Convert time-in-seconds to IB fabric time value
 */
uint8_t uct_ib_to_fabric_time(double time);


/**
 * @return MTU in bytes.
 */
size_t uct_ib_mtu_value(enum ibv_mtu mtu);


/**
 * @return IB address scope of a given subnet prefix (according to IBTA 4.1.1 12).
 */
uct_ib_address_type_t uct_ib_address_scope(uint64_t subnet_prefix);


/**
 * @return IB address size of the given link scope.
 */
size_t uct_ib_address_size(uct_ib_address_type_t type);


/**
 * Pack IB address.
 *
 * @param [in]  dev        IB device. TODO remove this.
 * @param [in]  scope      Address scope.
 * @param [in]  gid        GID address to pack.
 * @param [in]  lid        LID address to pack.
 * @param [out] ib_addr    Filled with packed ib address. Size of the structure
 *                         must be at least what @ref uct_ib_address_size() returns
 *                         for the given scope.
 */
void uct_ib_address_pack(uct_ib_device_t *dev, uct_ib_address_type_t scope,
                         const union ibv_gid *gid, uint16_t lid,
                         uct_ib_address_t *ib_addr);


/**
 * Unpack IB address.
 *
 * @param [in]  ib_addr    IB address to unpack.
 * @param [out] lid        Filled with address LID, or 0 if not present.
 * @param [out] is_global  Filled with 0, or 1 if the address is IB global
 */
void uct_ib_address_unpack(const uct_ib_address_t *ib_addr, uint16_t *lid,
                           uint8_t *is_global, union ibv_gid *gid);


/**
 * Convert IB address to a human-readable string.
 */
const char *uct_ib_address_str(const uct_ib_address_t *ib_addr, char *buf,
                               size_t max);


/**
 * Modify QP to a given state and check for error
 */
ucs_status_t uct_ib_modify_qp(struct ibv_qp *qp, enum ibv_qp_state state);


/**
 * find device mtu. This function can be used before ib
 * interface is created.
 */
ucs_status_t uct_ib_device_mtu(const char *dev_name, uct_md_h md, int *p_mtu);

int uct_ib_atomic_is_supported(uct_ib_device_t *dev, int ext, size_t size);

int uct_ib_atomic_is_be_reply(uct_ib_device_t *dev, int ext, size_t size);

ucs_status_t uct_ib_device_find_port(uct_ib_device_t *dev,
                                     const char *resource_dev_name,
                                     uint8_t *p_port_num);

size_t uct_ib_device_odp_max_size(uct_ib_device_t *dev);

int uct_ib_device_odp_has_global_mr(uct_ib_device_t *dev);

const char *uct_ib_wc_status_str(enum ibv_wc_status wc_status);

static inline struct ibv_exp_port_attr*
uct_ib_device_port_attr(uct_ib_device_t *dev, uint8_t port_num)
{
    return &dev->port_attr[port_num - dev->first_port];
}

ucs_status_t
uct_ib_device_query_gid(uct_ib_device_t *dev, uint8_t port_num, unsigned gid_index,
                        union ibv_gid *gid);


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
