/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_UD_EP_H
#define UCT_UD_EP_H

#include "ud_def.h"

#include <uct/api/uct.h>
#include <ucs/datastruct/frag_list.h>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/datastruct/sglib.h>
#include <ucs/time/timer_wheel.h>

#define UCT_UD_EP_NULL_ID     ((1<<24)-1)
#define UCT_UD_EP_ID_MAX      UCT_UD_EP_NULL_ID
#define UCT_UD_EP_CONN_ID_MAX UCT_UD_EP_ID_MAX 

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

#define UCT_UD_EP_HOOK_CALL_RX(ep, neth, len) \
    if ((ep)->rx.rx_hook(ep, neth) != UCS_OK) { \
        uct_ud_log_packet(__FILE__, __LINE__, __FUNCTION__, \
                          ucs_derived_of(ep->super.super.iface, uct_ud_iface_t), \
                          ep, \
                          UCT_AM_TRACE_TYPE_RECV_DROP, \
                          neth, len); \
        return; \
    }

#define UCT_UD_EP_HOOK_CALL_TX(ep, neth) (ep)->tx.tx_hook(ep, neth);
#define UCT_UD_EP_HOOK_CALL_TIMER(ep)    (ep)->timer_hook(ep, NULL);

static inline ucs_status_t uct_ud_ep_null_hook(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    return UCS_OK;
}

#define UCT_UD_EP_HOOK_INIT(ep) \
do { \
   (ep)->tx.tx_hook = uct_ud_ep_null_hook; \
   (ep)->rx.rx_hook = uct_ud_ep_null_hook; \
   (ep)->timer_hook = uct_ud_ep_null_hook; \
} while(0);

#else 

#define UCT_UD_EP_HOOK_DECLARE(name)
#define UCT_UD_EP_HOOK_CALL_RX(ep, neth, len) 
#define UCT_UD_EP_HOOK_CALL_TX(ep, neth) 
#define UCT_UD_EP_HOOK_CALL_TIMER(ep) 
#define UCT_UD_EP_HOOK_INIT(ep) 

#endif


/**
 * Slow ep timer 
 * The purpose of the slow timer is to schedule resends and ack replies.
 * The timer is a wheel timer. Timer wheel sweep is done on every async
 * progress invocation. One tick usually happens once in 0.1 seconds. 
 * It is best to avoid to take time in the fast path.
 *
 * wheel_time is the time of last timer wheel sweep. 
 * on send:
 *   - try to start wheel timer. 
 *   - send_time = wheel_time. That is sending a packet resets retransmission
 *   timeout. This does not allow infinite number of resets because number of
 *   outstanding packets is bound by the TX window size. 
 * on ack recv:
 *   - send_time = wheel_time. (advance last send time)
 * on timer expiration:
 *   - if wheel_time - saved_time > 3*one_tick_time 
 *        schedule resend 
 *        send_time = wheel_time
 *   - if window is not empty resched timer
 *   3x is needed to avoid false resends because of errors in timekeeping
 *
 * Fast ep timer
 * The purpose of the fast timer is to detect packet loss as early as
 * possible. The timer is a wheel timer. Fast timer sweep is done on 
 * CQ polling which happens either in explicit polling or in async
 * progress. As a result fast timer resolution may vary.   
 *
 * TODO: adaptive CHK algo description
 *
 * Fast time is relatively expensive. It is best to disable if packet loss
 * is not expected. Usual reasons for packet loss are: slow receiver,
 * many to one traffic pattern. 
 */

/* 
 * Endpoint pending control operations. The operations
 * are executed in time of progress along with
 * pending requests added by uct user. 
 */
enum {
    UCT_UD_EP_OP_NONE       = 0,
    UCT_UD_EP_OP_ACK        = UCS_BIT(0),  /* ack data */
    UCT_UD_EP_OP_ACK_REQ    = UCS_BIT(1),  /* request ack of sent packets */
    UCT_UD_EP_OP_RESEND     = UCS_BIT(2),  /* resend un acked packets */
    UCT_UD_EP_OP_CREP       = UCS_BIT(3),  /* send connection reply */
};

#define UCT_UD_EP_OP_CTL_LOW_PRIO (UCT_UD_EP_OP_ACK_REQ|UCT_UD_EP_OP_ACK)
#define UCT_UD_EP_OP_CTL_HI_PRIO  (UCT_UD_EP_OP_CREP|UCT_UD_EP_OP_RESEND)

typedef struct uct_ud_ep_pending_op {
    ucs_arbiter_group_t   group;  
    uint32_t              ops;    /* bitmask that describes what control ops are sceduled */
    ucs_arbiter_elem_t    elem;
} uct_ud_ep_pending_op_t;


struct uct_ud_ep {
    uct_base_ep_t           super;
    uint32_t                ep_id;
    uint32_t                dest_ep_id;
    uint32_t                dest_qpn;
    struct {
         uct_ud_psn_t           psn;          /* Next PSN to send */
         uct_ud_psn_t           max_psn;      /* Largest PSN that can be sent - (ack_psn + window) (from incoming packet) */
         uct_ud_psn_t           acked_psn;    /* last psn that was acked by remote side */
         uct_ud_psn_t           rt_psn;       /* last psn that was retransmitted */
         ucs_queue_head_t       window;       /* send window */
         uct_ud_ep_pending_op_t pending;      /* pending ops */
         ucs_time_t             send_time;    /* tx time of last packet */
         ucs_queue_iter_t       rt_pos;       /* points to the part of tx window that needs to be resent */
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
    ucs_wtimer_t slow_timer;
    UCT_UD_EP_HOOK_DECLARE(timer_hook);
};

UCS_CLASS_DECLARE(uct_ud_ep_t, uct_ud_iface_t*)

ucs_status_t uct_ud_ep_flush(uct_ep_h ep);
/* internal flush */
ucs_status_t uct_ud_ep_flush_nolock(uct_ud_iface_t *iface, uct_ud_ep_t *ep);

ucs_status_t uct_ud_ep_get_address(uct_ep_h tl_ep, struct sockaddr *addr);

ucs_status_t uct_ud_ep_connect_to_ep(uct_ud_ep_t *ep, 
                                     const struct sockaddr *addr);

ucs_status_t uct_ud_ep_connect_to_iface(uct_ud_ep_t *ep,
                                        const uct_sockaddr_ib_t *addr);

ucs_status_t uct_ud_ep_disconnect_from_iface(uct_ep_h tl_ep);

ucs_status_t uct_ud_ep_pending_add(uct_ep_h ep, uct_pending_req_t *n);

void         uct_ud_ep_pending_purge(uct_ep_h ep, uct_pending_callback_t cb);


/* helper function to create/destroy new connected ep */
ucs_status_t uct_ud_ep_create_connected_common(uct_ud_iface_t *iface,
                                               const uct_sockaddr_ib_t *addr,
                                               uct_ud_ep_t **new_ep_p,
                                               uct_ud_send_skb_t **skb_p);

void uct_ud_ep_destroy_connected(uct_ud_ep_t *ep, 
                                 const uct_sockaddr_ib_t *addr);

uct_ud_send_skb_t *uct_ud_ep_prepare_creq(uct_ud_ep_t *ep);

ucs_arbiter_cb_result_t
uct_ud_ep_do_pending(ucs_arbiter_t *arbiter, ucs_arbiter_elem_t *elem,
                     void *arg);

void uct_ud_ep_clone(uct_ud_ep_t *old_ep, uct_ud_ep_t *new_ep);

static UCS_F_ALWAYS_INLINE void
uct_ud_neth_set_type_am(uct_ud_ep_t *ep, uct_ud_neth_t *neth, uint8_t id)
{
    neth->packet_type = (id << UCT_UD_PACKET_AM_ID_SHIFT) |
                        ep->dest_ep_id | 
                        UCT_UD_PACKET_FLAG_AM;
}

static UCS_F_ALWAYS_INLINE void
uct_ud_neth_set_type_put(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->packet_type = ep->dest_ep_id | UCT_UD_PACKET_FLAG_PUT;
}

void uct_ud_ep_process_rx(uct_ud_iface_t *iface, 
                          uct_ud_neth_t *neth, unsigned byte_len, 
                          uct_ud_recv_skb_t *skb);


static UCS_F_ALWAYS_INLINE void
uct_ud_neth_ctl_ack(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->psn         = ep->tx.psn;
    neth->ack_psn     = ep->rx.acked_psn = ucs_frag_list_sn(&ep->rx.ooo_pkts);
    neth->packet_type = ep->dest_ep_id;
}

static UCS_F_ALWAYS_INLINE void
uct_ud_neth_ctl_ack_req(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->psn         = ep->tx.psn;
    neth->ack_psn     = ep->rx.acked_psn = ucs_frag_list_sn(&ep->rx.ooo_pkts);
    neth->packet_type = ep->dest_ep_id|UCT_UD_PACKET_FLAG_ACK_REQ;
}

static UCS_F_ALWAYS_INLINE void 
uct_ud_neth_init_data(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->psn = ep->tx.psn;
    neth->ack_psn = ep->rx.acked_psn = ucs_frag_list_sn(&ep->rx.ooo_pkts);
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


static UCS_F_ALWAYS_INLINE void
uct_ud_ep_ctl_op_del(uct_ud_ep_t *ep, uint32_t ops)
{
    ep->tx.pending.ops &= ~ops;
}

static UCS_F_ALWAYS_INLINE int
uct_ud_ep_ctl_op_check(uct_ud_ep_t *ep, uint32_t op)
{
    return ep->tx.pending.ops & op;
}

static UCS_F_ALWAYS_INLINE int
uct_ud_ep_ctl_op_isany(uct_ud_ep_t *ep)
{
    return ep->tx.pending.ops; 
}

static UCS_F_ALWAYS_INLINE int
uct_ud_ep_ctl_op_check_ex(uct_ud_ep_t *ep, uint32_t ops)
{
    return ep->tx.pending.ops == ops;
}


/* TODO: relay on window check instead. max_psn = psn  */
static UCS_F_ALWAYS_INLINE int uct_ud_ep_is_connected(uct_ud_ep_t *ep)
{
        return ep->dest_ep_id != UCT_UD_EP_NULL_ID;
}

static UCS_F_ALWAYS_INLINE int uct_ud_ep_no_window(uct_ud_ep_t *ep)
{
        return ep->tx.psn == ep->tx.max_psn;
}

/*
 * Request ACK once we sent 1/4 of the window or once we got to the window end
 * or there is a pending ack request operation
 */
static UCS_F_ALWAYS_INLINE int uct_ud_ep_req_ack(uct_ud_ep_t *ep)
{
    uct_ud_psn_t acked_psn, max_psn, psn;

    max_psn   = ep->tx.max_psn;
    acked_psn = ep->tx.acked_psn;
    psn       = ep->tx.psn;

    return UCT_UD_PSN_COMPARE(psn, ==, ((acked_psn * 3 + max_psn) >> 2)) ||
           UCT_UD_PSN_COMPARE(psn + 1, ==, max_psn) ||
           uct_ud_ep_ctl_op_check(ep, UCT_UD_EP_OP_ACK_REQ);

}


static UCS_F_ALWAYS_INLINE void 
uct_ud_neth_ack_req(uct_ud_ep_t *ep, uct_ud_neth_t *neth)
{
    neth->packet_type |= uct_ud_ep_req_ack(ep) << UCT_UD_PACKET_ACK_REQ_SHIFT;
    uct_ud_ep_ctl_op_del(ep, UCT_UD_EP_OP_ACK|UCT_UD_EP_OP_ACK_REQ);
}

#endif 

