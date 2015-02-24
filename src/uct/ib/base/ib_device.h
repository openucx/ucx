/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_DEVICE_H
#define UCT_IB_DEVICE_H

#include "ib_verbs.h"

#include <uct/api/uct.h>
#include <ucs/stats/stats.h>
#include <ucs/type/status.h>

#define UCT_IB_QPN_ORDER         24  /* How many bits can be an IB QP number */
#define UCT_IB_LRH_LEN           8
#define UCT_IB_GRH_LEN           40
#define UCT_IB_BTH_LEN           12
#define UCT_IB_DETH_LEN          8
#define UCT_IB_RETH_LEN          16
#define UCT_IB_ATOMIC_ETH_LEN    28
#define UCT_IB_AETH_LEN          4
#define UCT_IB_PAYLOAD_ALIGN     4
#define UCT_IB_ICRC_LEN          4
#define UCT_IB_VCRC_LEN          2
#define UCT_IB_DELIM_LEN         2
#define UCT_IB_FDR_PACKET_GAP    64
#define UCT_IB_MAX_MESSAGE_SIZE  (2 << 30)

enum {
    UCT_IB_DEVICE_STAT_MEM_MAP,
    UCT_IB_DEVICE_STAT_MEM_UNMAP,
    UCT_IB_DEVICE_STAT_ASYNC_EVENT,
    UCT_IB_DEVICE_STAT_LAST
};


typedef struct uct_ib_device uct_ib_device_t;
struct uct_ib_device {
    uct_pd_t                    super;
    struct ibv_context          *ibv_context;    /* Verbs context */
    struct ibv_pd               *pd;             /* Protection domain */
    struct ibv_exp_device_attr  dev_attr;        /* Cached device attributes */
    uint8_t                     first_port;      /* Number of first port (usually 1) */
    uint8_t                     num_ports;       /* Amount of physical ports */
    cpu_set_t                   local_cpus;      /* CPUs local to device */
    UCS_STATS_NODE_DECLARE(stats);

    struct ibv_exp_port_attr    port_attr[0];    /* Cached port attributes */
};


ucs_status_t uct_ib_device_create(uct_context_h context,
                                  struct ibv_device *ibv_device,
                                  uct_ib_device_t **dev_p);
void uct_ib_device_destroy(uct_ib_device_t *dev);


ucs_status_t uct_ib_device_port_check(uct_ib_device_t *dev, uint8_t port_num,
                                      unsigned flags);

ucs_status_t uct_ib_device_port_get_resource(uct_ib_device_t *dev, uint8_t port_num,
                                             size_t tl_hdr_len, uint64_t tl_overhead_ns,
                                             uct_resource_desc_t *resource);

const char *uct_ib_device_name(uct_ib_device_t *dev);

ucs_status_t uct_ib_rkey_unpack(uct_context_h context, void *rkey_buffer,
                                uct_rkey_bundle_t *rkey_ob);

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


static inline struct ibv_exp_port_attr* uct_ib_device_port_attr(uct_ib_device_t *dev, uint8_t port_num)
{
    return &dev->port_attr[port_num - dev->first_port];
}

static inline struct ibv_mr* uct_ib_lkey_mr(uct_lkey_t lkey)
{
    return (struct ibv_mr *)(void*)lkey;
}

#endif
