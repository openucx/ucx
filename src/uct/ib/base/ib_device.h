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


enum {
    UCT_IB_DEVICE_STAT_ASYNC_EVENT,
    UCT_IB_DEVICE_STAT_LAST
};


enum {
    UCT_IB_DEVICE_FLAG_MLX4_PRM = UCS_BIT(1),   /* Device supports mlx4 PRM */
    UCT_IB_DEVICE_FLAG_MLX5_PRM = UCS_BIT(2),   /* Device supports mlx5 PRM */
    UCT_IB_DEVICE_FLAG_DC       = UCS_BIT(3)    /* Device supports DC */
};


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
 * @param tl_hdr_len       How many bytes this transport adds on top of IB header (LRH+BTH+iCRC+vCRC)
 * @param tl_overhead_ns   How much overhead the transport adds to latency.
 * @param resources_p      Filled with a pointer to an array of resources.
 * @param num_resources_p  Filled with the number of resources.
 */
ucs_status_t uct_ib_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              size_t tl_hdr_len, uint64_t tl_overhead_ns,
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


static inline struct ibv_exp_port_attr*
uct_ib_device_port_attr(uct_ib_device_t *dev, uint8_t port_num)
{
    return &dev->port_attr[port_num - dev->first_port];
}

#endif
