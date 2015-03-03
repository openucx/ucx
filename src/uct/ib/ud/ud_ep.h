/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_UD_EP_H
#define UCT_UD_EP_H

#include "ud_def.h"

#include <uct/api/uct.h>
#include <ucs/datastruct/frag_list.h>
#include <ucs/datastruct/queue.h>

#define UCT_UD_EP_NULL_ID (-1)

typedef struct uct_ud_ep_addr {
    uct_ep_addr_t     super;
    uint32_t          ep_id;
} uct_ud_ep_addr_t;


typedef struct uct_ud_ep {
    uct_ep_t                super;
    uint32_t                ep_id;
    uint32_t                dest_ep_id;
    uint32_t                dest_qpn;
    struct ibv_ah          *ah;
    struct {
         uct_ud_psn_t       psn;          /* Next PSN to send */
         uct_ud_psn_t       max_psn;      /* Largest PSN that can be sent - (ack_psn + window) (from incoming packet) */
         uct_ud_psn_t       acked_psn;    /* last psn that was acked by remote side */
         ucs_queue_head_t   window;       /* send window */
         UCS_STATS_NODE_DECLARE(stats);
    } tx;
    struct {
        uct_ud_psn_t        acked_psn;    /* Last psn we acked */
        ucs_frag_list_t     ooo_pkts;     /* Out of order packets that can not be processed yet,
                                            also keeps last psn we successfully received and processed */
        UCS_STATS_NODE_DECLARE(stats);
    } rx;
} uct_ud_ep_t;


ucs_status_t uct_ud_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr);

ucs_status_t uct_ud_ep_connect_to_ep(uct_ep_h tl_ep, uct_iface_addr_t *tl_iface_addr,
                                     uct_ep_addr_t *tl_ep_addr);

static inline void uct_ud_neth_set_type_am(uct_ud_neth_t *neth, uct_ud_ep_t *ep, uint8_t id)
{
    neth->packet_type = (id << UCT_UD_PACKET_AM_ID_SHIFT) |
                        (ep->dest_ep_id << UCT_UD_PACKET_DEST_ID_SHIFT) | 
                        UCT_UD_PACKET_FLAG_AM;
}

void uct_ud_ep_process_rx(uct_ud_iface_t *iface, uct_ud_neth_t *neth, unsigned byte_len, uct_ud_recv_skb_t *skb);

#endif 

