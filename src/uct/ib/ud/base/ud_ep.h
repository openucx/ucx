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
        ucs_trace_data("RX: dropping packet"); \
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
 *        consgestion avoidance decreases tx window
 *   - if window is not empty resched timer
 *   3x is needed to avoid false resends because of errors in timekeeping
 *
 * Fast ep timer (Not implemented)
 *
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

/* Congestion avoidance and retransmits
 *
 * UD uses additive increase/multiplicative decrease algorightm
 * See https://en.wikipedia.org/wiki/Additive_increase/multiplicative_decrease
 *
 * tx window is increased when ack is received and decreased when
 * resend is scheduled. Ack must be a 'new' one that is it must
 * acknowledge packets on window. Increasing window on ack does not casue
 * exponential window increase because, unlike tcp, only two acks
 * per window are sent.
 *
 * Todo:
 *
 * Consider trigering window decrease before resend timeout:
 * - on ECN (explicit congestion notification) from receiever. ECN can
 *   be based on some heuristic. For example on number of rx completions
 *   that receiver picked from CQ.
 * - upon receiving a 'duplicate ack' packet
 *
 * Consider using other algorithm (ex BIC/CUBIC)
 */

/*
 * Handling retransmits
 *
 * On slow timer timeout schedule a retransmit operation for
 * [acked_psn+1, psn-1]. These values are saved as 'resend window'
 *
 * Resend operation will resend no more then the current cwnd
 * If ack arrives when resend window is active it means that
 *  - something new in the resend window was acked. As a
 *  resutlt a new resend operation will be scheduled.
 *  - either resend window or something beyond it was
 *  acked. It means that no more retransmisions are needed.
 *  Current 'resend window' is deactivated
 *
 * When retransmitting, ack is requested if:
 * psn == acked_psn + 1 or
 * psn % UCT_UD_RESENDS_PER_ACK = 0
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
    UCT_UD_EP_OP_CREQ       = UCS_BIT(4)   /* send connection request */
};

#define UCT_UD_EP_OP_CTL_LOW_PRIO (UCT_UD_EP_OP_ACK_REQ|UCT_UD_EP_OP_ACK)
#define UCT_UD_EP_OP_CTL_HI_PRIO  (UCT_UD_EP_OP_CREQ|UCT_UD_EP_OP_CREP|UCT_UD_EP_OP_RESEND)

typedef struct uct_ud_ep_pending_op {
    ucs_arbiter_group_t   group;
    uint32_t              ops;    /* bitmask that describes what control ops are sceduled */
    ucs_arbiter_elem_t    elem;
} uct_ud_ep_pending_op_t;

enum {
    UCT_UD_EP_STAT_TODO
};

/* TODO: optimize endpoint memory footprint */
enum {
    UCT_UD_EP_FLAG_ASYNC_COMPS       = UCS_BIT(0), /* set if there are completions that
                                                    * were picked by async thread and queued */
    UCT_UD_EP_FLAG_DISCONNECTED      = UCS_BIT(1), /* set if the endpoint was disconnected */

    /* debug flags */
    UCT_UD_EP_FLAG_PRIVATE           = UCS_BIT(2), /* EP is was created as internal */
    UCT_UD_EP_FLAG_CREQ_RCVD         = UCS_BIT(3), /* CREQ message was received */
    UCT_UD_EP_FLAG_CREP_RCVD         = UCS_BIT(4), /* CREP message was received */
    UCT_UD_EP_FLAG_CREQ_SENT         = UCS_BIT(5), /* CREQ message was sent */
    UCT_UD_EP_FLAG_CREP_SENT         = UCS_BIT(6), /* CREP message was sent */
    UCT_UD_EP_FLAG_CREQ_NOTSENT      = UCS_BIT(7)  /* CREQ message is NOT sent, because
                                                      connection establishment process
                                                      is driven by remote side. */
};

typedef struct uct_ud_peer_name {
    char name[16];
    int  pid;
} uct_ud_peer_name_t;

struct uct_ud_ep {
    uct_base_ep_t           super;
    uint32_t                ep_id;
    uint32_t                dest_ep_id;
    struct {
         uct_ud_psn_t           psn;          /* Next PSN to send */
         uct_ud_psn_t           max_psn;      /* Largest PSN that can be sent */
         uct_ud_psn_t           acked_psn;    /* last psn that was acked by remote side */
         uint16_t               err_skb_count;/* number of failed SKBs on the ep */
         ucs_queue_head_t       window;       /* send window: [acked_psn+1, psn-1] */
         uct_ud_ep_pending_op_t pending;      /* pending ops */
         ucs_time_t             send_time;    /* tx time of last packet */
         ucs_time_t             slow_tick;    /* timeout to trigger slow timer */
         UCS_STATS_NODE_DECLARE(stats);
         UCT_UD_EP_HOOK_DECLARE(tx_hook);
    } tx;
    struct {
         uct_ud_psn_t           psn;       /* last psn that was retransmitted */
         uct_ud_psn_t           max_psn;   /* max psn that should be retransmitted */
         ucs_queue_iter_t       pos;       /* points to the part of tx window that needs to be resent */
    } resend;
    struct {
        uct_ud_psn_t  wmax;
        uct_ud_psn_t  cwnd;
    } ca;
    struct {
        uct_ud_psn_t        acked_psn;    /* Last psn we acked */
        ucs_frag_list_t     ooo_pkts;     /* Out of order packets that can not be processed yet,
                                            also keeps last psn we successfully received and processed */
        UCS_STATS_NODE_DECLARE(stats);
        UCT_UD_EP_HOOK_DECLARE(rx_hook);
    } rx;
    ucs_list_link_t  cep_list;
    uint32_t         conn_id;      /* connection id. assigned in connect_to_iface() */
    ucs_wtimer_t     slow_timer;
    uint8_t          flags;
    uint8_t          path_bits;
    UCS_STATS_NODE_DECLARE(stats);
    UCT_UD_EP_HOOK_DECLARE(timer_hook);
#if ENABLE_DEBUG_DATA
    uct_ud_peer_name_t  peer;
#endif
};

UCS_CLASS_DECLARE(uct_ud_ep_t, uct_ud_iface_t*)

void uct_ud_tx_wnd_purge_outstanding(uct_ud_iface_t *iface, uct_ud_ep_t *ud_ep,
                                     ucs_status_t status);

ucs_status_t uct_ud_ep_flush(uct_ep_h ep, unsigned flags,
                             uct_completion_t *comp);
/* internal flush */
ucs_status_t uct_ud_ep_flush_nolock(uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                                    uct_completion_t *comp);

ucs_status_t uct_ud_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr);

ucs_status_t uct_ud_ep_connect_to_ep(uct_ud_ep_t *ep,
                                     const uct_ib_address_t *ib_addr,
                                     const uct_ud_ep_addr_t *ep_addr);

ucs_status_t uct_ud_ep_pending_add(uct_ep_h ep, uct_pending_req_t *n);

void   uct_ud_ep_pending_purge(uct_ep_h ep, uct_pending_purge_callback_t cb,
                               void *arg);

void   uct_ud_ep_disconnect(uct_ep_h ep);


/* helper function to create/destroy new connected ep */
ucs_status_t uct_ud_ep_create_connected_common(uct_ud_iface_t *iface,
                                               const uct_ib_address_t *ib_addr,
                                               const uct_ud_iface_addr_t *if_addr,
                                               uct_ud_ep_t **new_ep_p,
                                               uct_ud_send_skb_t **skb_p);

void uct_ud_ep_destroy_connected(uct_ud_ep_t *ep,
                                 const uct_ib_address_t *ib_addr,
                                  const uct_ud_iface_addr_t *if_addr);

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
                          uct_ud_recv_skb_t *skb, int is_async);


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
    /* max_psn can be decreased by CA, so check >= */
    return UCT_UD_PSN_COMPARE(ep->tx.psn, >=, ep->tx.max_psn);
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
