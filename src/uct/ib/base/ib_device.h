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
#include <infiniband/arch.h>
#include <ucs/debug/log.h>


#define UCT_IB_QPN_ORDER            24  /* How many bits can be an IB QP number */
#define UCT_IB_LRH_LEN              8   /* IB Local routing header */
#define UCT_IB_GRH_LEN              40  /* IB GLobal routing header */
#define UCT_IB_BTH_LEN              12  /* IB base transport header */
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
#define UCT_IB_QKEY                 0x1ee7a330
#define UCT_IB_LINK_LOCAL_PREFIX    ntohll(0xfe80000000000000ul) /* IBTA 4.1.1 12a */
#define UCT_IB_SITE_LOCAL_PREFIX    ntohll(0xfec0000000000000ul) /* IBTA 4.1.1 12b */
#define UCT_IB_SITE_LOCAL_MASK      ntohll(0xffffffffffff0000ul) /* IBTA 4.1.1 12b */
#define UCT_IB_DC_KEY               0x1234


enum {
    UCT_IB_DEVICE_STAT_ASYNC_EVENT,
    UCT_IB_DEVICE_STAT_LAST
};


enum {
    UCT_IB_DEVICE_FLAG_MLX4_PRM = UCS_BIT(1),   /* Device supports mlx4 PRM */
    UCT_IB_DEVICE_FLAG_MLX5_PRM = UCS_BIT(2),   /* Device supports mlx5 PRM */
    UCT_IB_DEVICE_FLAG_DC       = UCS_BIT(3)    /* Device supports DC */
};


typedef enum {
    UCT_IB_ADDRESS_SCOPE_LINK_LOCAL,   /* Subnet-local address */
    UCT_IB_ADDRESS_SCOPE_SITE_LOCAL,   /* Site local, 16-bit subnet prefix */
    UCT_IB_ADDRESS_SCOPE_GLOBAL        /* Global, 64-bit subnet prefix */
} uct_ib_address_scope_t;


/**
 * Flags which specify which address fields are present
 */
enum {
    UCT_IB_ADDRESS_FLAG_LID      = UCS_BIT(0),
    UCT_IB_ADDRESS_FLAG_IF_ID    = UCS_BIT(1),
    UCT_IB_ADDRESS_FLAG_SUBNET16 = UCS_BIT(2),
    UCT_IB_ADDRESS_FLAG_SUBNET64 = UCS_BIT(3)
};


/**
 * IB network address
 */
typedef struct uct_ib_address {
    uint8_t            flags;
    uint16_t           dev_id; /* TODO remove this */
    /* Following fields appear in this order (if specified by flags) :
     * - uint16_t lid
     * - uint64_t if_id
     * - uint16_t subnet16
     * - uint64_t subnet64
     */
} UCS_S_PACKED uct_ib_address_t;


/**
 * IB device (corresponds to HCA)
 */
typedef struct uct_ib_device {
    struct ibv_context          *ibv_context;    /* Verbs context */
    struct ibv_exp_device_attr  dev_attr;        /* Cached device attributes */
    uint8_t                     first_port;      /* Number of first port (usually 1) */
    uint8_t                     num_ports;       /* Amount of physical ports */
    cpu_set_t                   local_cpus;      /* CPUs local to device */
    UCS_STATS_NODE_DECLARE(stats);
    struct ibv_exp_port_attr    port_attr[UCT_IB_DEV_MAX_PORTS]; /* Cached port attributes */
} uct_ib_device_t;


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


ucs_status_t uct_ib_device_init(uct_ib_device_t *dev, struct ibv_device *ibv_device
                                UCS_STATS_ARG(ucs_stats_node_t *stats_parent));

void uct_ib_device_cleanup(uct_ib_device_t *dev);


/**
 * @return device name.
 */
const char *uct_ib_device_name(uct_ib_device_t *dev);


/**
 * @return 1 if the port is InfiniBand, 0 if the port is Ethernet.
 */
int uct_ib_device_is_port_ib(uct_ib_device_t *dev, uint8_t port_num);


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
uct_ib_address_scope_t uct_ib_address_scope(uint64_t subnet_prefix);


/**
 * @return IB address size of the given link scope.
 */
size_t uct_ib_address_size(uct_ib_address_scope_t scope);


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
void uct_ib_address_pack(uct_ib_device_t *dev, uct_ib_address_scope_t scope,
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
 * find device mtu. This function can be used before ib
 * interface is created.
 */
ucs_status_t uct_ib_device_mtu(const char *dev_name, uct_pd_h pd, int *p_mtu);

ucs_status_t uct_ib_device_find_port(uct_ib_device_t *dev,
                                     const char *resource_dev_name,
                                     uint8_t *p_port_num);

static inline struct ibv_exp_port_attr*
uct_ib_device_port_attr(uct_ib_device_t *dev, uint8_t port_num)
{
    return &dev->port_attr[port_num - dev->first_port];
}


static inline ucs_status_t uct_ib_poll_cq(struct ibv_cq *cq, unsigned *count, struct ibv_wc *wcs)
{
    int ret;

    ret = ibv_poll_cq(cq, *count, wcs);
    if (ucs_unlikely(ret < 0)) {
        ucs_fatal("Failed to poll receive CQ");
    }

    if (ret == 0) {
        return UCS_ERR_NO_PROGRESS;
    }
    *count = ret;
    return UCS_OK;
}

#endif
