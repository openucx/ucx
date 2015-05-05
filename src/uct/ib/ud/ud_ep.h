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
#include <ucs/datastruct/sglib.h>

#define UCT_UD_EP_NULL_ID     ((1<<24)-1)
#define UCT_UD_EP_ID_MAX      UCT_UD_EP_NULL_ID
#define UCT_UD_EP_CONN_ID_MAX UCT_UD_EP_ID_MAX 

struct uct_ud_ep_addr {
    uct_ep_addr_t     super;
    uint32_t          ep_id;
};

#ifdef UCT_UD_EP_DEBUG_HOOKS
/*
   Hooks that allow packet header inspection and rewriting. UCT user can
   set functions that will be called just before packet is put on wire
   and when packet is received. Packet will be discarded if RX function
   returns status different from UCS_OK.

   Example:

  static ucs_status_t clear_ack_req(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
  {
     neth->packet_type &= ~UCT_UD_PACKET_FLAG_ACK_REQ;
     return UCS_OK;
  }

  uct_ep_t ep;
  ....
  // clear ack request bin on all outgoing packets 
  ucs_derived_of(ep, uct_ud_ep_t)->tx.tx_hook = clear_ack_req;

*/

typedef ucs_status_t (*uct_ud_ep_hook_t)(uct_ud_ep_t *ep, uct_ud_neth_t *neth);

#define UCT_UD_EP_HOOK_DECLARE(name) uct_ud_ep_hook_t name

#define UCT_UD_EP_HOOK_CALL_RX(ep, neth) \
    if ((ep)->rx.rx_hook(ep, neth) != UCS_OK) { \
        return; \
    }

#define UCT_UD_EP_HOOK_CALL_TX(ep, neth) (ep)->tx.tx_hook(ep, neth);

static inline ucs_status_t uct_ud_ep_null_hook(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    return UCS_OK;
}

#define UCT_UD_EP_HOOK_INIT(ep) \
do { \
   (ep)->tx.tx_hook = uct_ud_ep_null_hook; \
   (ep)->rx.rx_hook = uct_ud_ep_null_hook; \
} while(0);

#else 

#define UCT_UD_EP_HOOK_DECLARE(name)
#define UCT_UD_EP_HOOK_CALL_RX(ep, neth) 
#define UCT_UD_EP_HOOK_CALL_TX(ep, neth) 
#define UCT_UD_EP_HOOK_INIT(ep) 

#endif

struct uct_ud_ep {
    uct_base_ep_t           super;
    uint32_t                ep_id;
    uint32_t                dest_ep_id;
    uint32_t                dest_qpn;
    struct {
         uct_ud_psn_t           psn;          /* Next PSN to send */
         uct_ud_psn_t           max_psn;      /* Largest PSN that can be sent - (ack_psn + window) (from incoming packet) */
         uct_ud_psn_t           acked_psn;    /* last psn that was acked by remote side */
         ucs_queue_head_t       window;       /* send window */
         uct_ud_ep_pending_op_t pending;      /* pending control ops */
         UCS_STATS_NODE_DECLARE(stats);
         UCT_UD_EP_HOOK_DECLARE(tx_hook);
    } tx;
    struct {
        uct_ud_psn_t        acked_psn;    /* Last psn we acked */
        ucs_frag_list_t     ooo_pkts;     /* Out of order packets that can not be processed yet,
                                            also keeps last psn we successfully received and processed */
        UCS_STATS_NODE_DECLARE(stats);
        UCT_UD_EP_HOOK_DECLARE(rx_hook);
    } rx;
    ucs_list_link_t          cep_list;
    uint32_t                 conn_id;      /* connection id. assigned in connect_to_iface() */
};

UCS_CLASS_DECLARE(uct_ud_ep_t, uct_ud_iface_t*)


ucs_status_t uct_ud_ep_get_address(uct_ep_h tl_ep, struct sockaddr *addr);

ucs_status_t uct_ud_ep_connect_to_ep(uct_ud_ep_t *ep, const struct sockaddr *addr);

ucs_status_t uct_ud_ep_connect_to_iface(uct_ep_h tl_ep, const uct_iface_addr_t *tl_iface_addr);
ucs_status_t uct_ud_ep_disconnect_from_iface(uct_ep_h tl_ep);

uct_ud_send_skb_t *uct_ud_ep_prepare_creq(uct_ud_ep_t *ep);
uct_ud_send_skb_t *uct_ud_ep_prepare_crep(uct_ud_ep_t *ep);

void uct_ud_ep_clone(uct_ud_ep_t *old_ep, uct_ud_ep_t *new_ep);

static inline void uct_ud_neth_set_type_am(uct_ud_ep_t *ep, uct_ud_neth_t *neth, uint8_t id)
{
    neth->packet_type = (id << UCT_UD_PACKET_AM_ID_SHIFT) |
                        ep->dest_ep_id | 
                        UCT_UD_PACKET_FLAG_AM;
}

static inline void uct_ud_neth_set_type_put(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->packet_type = ep->dest_ep_id | UCT_UD_PACKET_FLAG_PUT;
}
void uct_ud_ep_process_rx(uct_ud_iface_t *iface, uct_ud_neth_t *neth, unsigned byte_len, uct_ud_recv_skb_t *skb);


/*
 * Request ACK once we sent 1/4 of the window. Request another ack once we got to the window end. 
 */
static inline int uct_ud_ep_req_ack(uct_ud_ep_t *ep)
{
    uct_ud_psn_t acked_psn, max_psn, psn;

    max_psn   = ep->tx.max_psn;
    acked_psn = ep->tx.acked_psn;
    psn       = ep->tx.psn;

    return UCT_UD_PSN_COMPARE(psn, ==, ((acked_psn*3 + max_psn)>>2)) ||
        UCT_UD_PSN_COMPARE(psn+1, ==, max_psn);
}

static inline void uct_ud_neth_ctl_ack(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->psn         = ep->tx.psn;
    neth->ack_psn     = ep->rx.acked_psn = ucs_frag_list_sn(&ep->rx.ooo_pkts);
    neth->packet_type = ep->dest_ep_id;
}

static inline void uct_ud_neth_init_data(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->psn = ep->tx.psn;
    neth->ack_psn = ep->rx.acked_psn = ucs_frag_list_sn(&ep->rx.ooo_pkts);
}


static inline void uct_ud_neth_ack_req(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->packet_type |= uct_ud_ep_req_ack(ep) << UCT_UD_PACKET_ACK_REQ_SHIFT;
    ep->tx.pending.ops &= ~UCT_UD_EP_OP_ACK;
}

static inline int uct_ud_ep_compare(uct_ud_ep_t *a, uct_ud_ep_t *b)
{
    return a->conn_id - b->conn_id;
}

static inline int uct_ud_ep_hash(uct_ud_ep_t *ep)
{
    return ep->conn_id % UCT_UD_HASH_SIZE;
}

SGLIB_DEFINE_LIST_PROTOTYPES(uct_ud_ep_t, uct_ud_ep_compare, next)
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_ud_ep_t, UCT_UD_HASH_SIZE, uct_ud_ep_hash)

#endif 

