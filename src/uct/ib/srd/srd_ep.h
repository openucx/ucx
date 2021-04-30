/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_SRD_EP_H
#define UCT_SRD_EP_H

#include "srd_def.h"

#include <uct/api/uct.h>
#include <ucs/datastruct/frag_list.h>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/datastruct/sglib.h>
#include <ucs/datastruct/conn_match.h>

#define UCT_SRD_EP_NULL_ID     ((1<<24)-1)
#define UCT_SRD_EP_ID_MAX      UCT_SRD_EP_NULL_ID
#define UCT_SRD_EP_CONN_ID_MAX UCT_SRD_EP_ID_MAX

typedef uint32_t uct_srd_ep_conn_sn_t;


/*
 * Endpoint pending control operations. The operations
 * are executed in time of progress along with
 * pending requests added by uct user.
 */
enum {
    UCT_SRD_EP_OP_NONE       = 0,
    UCT_SRD_EP_OP_CREP       = UCS_BIT(0),  /* send connection reply */
    UCT_SRD_EP_OP_CREQ       = UCS_BIT(1),  /* send connection request */
};

#define UCT_SRD_EP_OP_CTL_HI_PRIO  (UCT_SRD_EP_OP_CREQ|UCT_SRD_EP_OP_CREP)

typedef struct uct_srd_ep_pending_op {
    ucs_arbiter_group_t   group;
    uint32_t              ops;    /* bitmask that describes what control ops are sceduled */
    ucs_arbiter_elem_t    elem;
} uct_srd_ep_pending_op_t;

enum {
    UCT_SRD_EP_STAT_TODO
};

/* TODO: optimize endpoint memory footprint */
enum {
    UCT_SRD_EP_FLAG_DISCONNECTED      = UCS_BIT(0),  /* EP was disconnected */
    UCT_SRD_EP_FLAG_PRIVATE           = UCS_BIT(1),  /* EP was created as internal */
    UCT_SRD_EP_FLAG_HAS_PENDING       = UCS_BIT(2),  /* EP has some pending requests */
    UCT_SRD_EP_FLAG_CONNECTED         = UCS_BIT(3),  /* EP was connected to the peer */
    UCT_SRD_EP_FLAG_ON_CEP            = UCS_BIT(4),  /* EP was inserted to connection
                                                       matching context */

    /* debug flags */
    UCT_SRD_EP_FLAG_CREQ_RCVD         = UCS_BIT(5),  /* CREQ message was received */
    UCT_SRD_EP_FLAG_CREP_RCVD         = UCS_BIT(6),  /* CREP message was received */
    UCT_SRD_EP_FLAG_CREQ_SENT         = UCS_BIT(7),  /* CREQ message was sent */
    UCT_SRD_EP_FLAG_CREP_SENT         = UCS_BIT(8),  /* CREP message was sent */
    UCT_SRD_EP_FLAG_CREQ_NOTSENT      = UCS_BIT(9),  /* CREQ message is NOT sent, because
                                                       connection establishment process
                                                       is driven by remote side. */

    /* Endpoint is currently executing the pending queue */
#if UCS_ENABLE_ASSERT
    UCT_SRD_EP_FLAG_IN_PENDING        = UCS_BIT(11)
#else
    UCT_SRD_EP_FLAG_IN_PENDING        = 0
#endif
};


#define UCT_SRD_EP_ASSERT_PENDING(_ep) \
    ucs_assertv((_ep->flags & UCT_SRD_EP_FLAG_IN_PENDING) ||    \
                !uct_srd_ep_has_pending(_ep),                   \
                "out-of-order send detected for"                \
                " ep %p ep_pending %d arbelem %p",              \
                _ep, (_ep->flags & UCT_SRD_EP_FLAG_IN_PENDING), \
                &_ep->tx.pending.elem);


typedef struct uct_srd_peer_name {
    char name[16];
    int  pid;
} uct_srd_peer_name_t;

typedef struct {
    uint32_t                          dest_qpn;
    struct ibv_ah                     *ah;
} uct_srd_ep_peer_address_t;

struct uct_srd_ep {
    uct_base_ep_t                     super;
    uint32_t                          ep_id;
    uint32_t                          dest_ep_id;
    struct {
        uct_srd_psn_t                 psn;     /* Next PSN to send */
        uct_srd_ep_pending_op_t       pending; /* pending */
        ucs_queue_head_t              outstanding_q; /* queue of outstanding send ops*/
        UCS_STATS_NODE_DECLARE(stats)
    } tx;
    struct {
        ucs_frag_list_t               ooo_pkts; /* Out of order packets that
                                                   can not be processed yet */
        UCS_STATS_NODE_DECLARE(stats)
    } rx;
    ucs_conn_match_elem_t             conn_match;
    /* connection sequence number. assigned in connect_to_iface() */
    uct_srd_ep_conn_sn_t              conn_sn;
    uint16_t                          flags;
    uint8_t                           rx_creq_count;
    uint8_t                           path_index;
    uct_srd_ep_peer_address_t         peer_address;
    UCS_STATS_NODE_DECLARE(stats)
#if ENABLE_DEBUG_DATA
    uct_srd_peer_name_t               peer;
#endif
};

#if ENABLE_DEBUG_DATA
#  define UCT_SRD_EP_PEER_NAME_FMT        "%s:%d"
#  define UCT_SRD_EP_PEER_NAME_ARG(_ep)   (_ep)->peer.name, (_ep)->peer.pid
#else
#  define UCT_SRD_EP_PEER_NAME_FMT        "%s"
#  define UCT_SRD_EP_PEER_NAME_ARG(_ep)   "<no debug data>"
#endif


/**
 * SRD pending request private data
 */
typedef struct {
    uct_pending_req_priv_arb_t arb;
    unsigned                   flags;
} uct_srd_pending_req_priv_t;


static UCS_F_ALWAYS_INLINE uct_srd_pending_req_priv_t *
uct_srd_pending_req_priv(uct_pending_req_t *req)
{
    return (uct_srd_pending_req_priv_t *)&(req)->priv;
}


ucs_status_t uct_srd_ep_create(const uct_ep_params_t *params, uct_ep_h *ep_p);

ucs_status_t uct_srd_ep_flush(uct_ep_h ep, unsigned flags,
                              uct_completion_t *comp);
/* internal flush */
ucs_status_t uct_srd_ep_flush_nolock(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                                     uct_completion_t *comp);

ucs_status_t uct_srd_ep_check(uct_ep_h tl_ep, unsigned flags,
                              uct_completion_t *comp);

ucs_status_t uct_srd_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr);

ucs_status_t uct_srd_ep_create_connected_common(const uct_ep_params_t *params,
                                                uct_ep_h *new_ep_p);

ucs_status_t uct_srd_ep_connect_to_ep(uct_ep_h tl_ep,
                                      const uct_device_addr_t *dev_addr,
                                      const uct_ep_addr_t *uct_ep_addr);

ucs_status_t uct_srd_ep_pending_add(uct_ep_h ep, uct_pending_req_t *n,
                                    unsigned flags);

void uct_srd_ep_pending_purge(uct_ep_h ep, uct_pending_purge_callback_t cb,
                              void *arg);

void uct_srd_ep_disconnect(uct_ep_h ep);

ucs_arbiter_cb_result_t
uct_srd_ep_do_pending(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                      ucs_arbiter_elem_t *elem, void *arg);

void uct_srd_ep_clone(uct_srd_ep_t *old_ep, uct_srd_ep_t *new_ep);

void *uct_srd_ep_get_peer_address(uct_srd_ep_t *srd_ep);

void uct_srd_ep_process_rx(uct_srd_iface_t *iface,
                           uct_srd_neth_t *neth, unsigned byte_len,
                           uct_srd_recv_desc_t *desc);

void uct_srd_ep_send_completion(uct_srd_send_op_t *send_op);

ucs_status_t uct_srd_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                 const void *buffer, unsigned length);

ucs_status_t uct_srd_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                     const uct_iov_t *iov, size_t iovcnt);

ssize_t uct_srd_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags);

ucs_status_t
uct_srd_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                    unsigned header_length, const uct_iov_t *iov,
                    size_t iovcnt, unsigned flags, uct_completion_t *comp);

#ifdef HAVE_DECL_EFA_DV_RDMA_READ
ucs_status_t uct_srd_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                  size_t iovcnt, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp);

ucs_status_t uct_srd_ep_get_bcopy(uct_ep_h tl_ep,
                                  uct_unpack_callback_t unpack_cb,
                                  void *arg, size_t length,
                                  uint64_t remote_addr, uct_rkey_t rkey,
                                  uct_completion_t *comp);
#endif

static UCS_F_ALWAYS_INLINE void
uct_srd_ep_ctl_op_del(uct_srd_ep_t *ep, uint32_t ops)
{
    ep->tx.pending.ops &= ~ops;
}

static UCS_F_ALWAYS_INLINE int
uct_srd_ep_ctl_op_check(uct_srd_ep_t *ep, uint32_t op)
{
    return ep->tx.pending.ops & op;
}

static UCS_F_ALWAYS_INLINE int
uct_srd_ep_ctl_op_isany(uct_srd_ep_t *ep)
{
    return ep->tx.pending.ops;
}

static UCS_F_ALWAYS_INLINE int
uct_srd_ep_ctl_op_check_ex(uct_srd_ep_t *ep, uint32_t ops)
{
    /* check that at least one the given ops is set and
     * all ops not given are not set */
    return (ep->tx.pending.ops & ops) &&
           ((ep->tx.pending.ops & ~ops) == 0);
}

static UCS_F_ALWAYS_INLINE int uct_srd_ep_is_connected(uct_srd_ep_t *ep)
{
    ucs_assert((ep->dest_ep_id == UCT_SRD_EP_NULL_ID) ==
               !(ep->flags & UCT_SRD_EP_FLAG_CONNECTED));
    return ep->flags & UCT_SRD_EP_FLAG_CONNECTED;
}

static UCS_F_ALWAYS_INLINE int
uct_srd_ep_is_connected_and_no_pending(uct_srd_ep_t *ep)
{
    return (ep->flags & (UCT_SRD_EP_FLAG_CONNECTED |
                         UCT_SRD_EP_FLAG_HAS_PENDING))
           == UCT_SRD_EP_FLAG_CONNECTED;
}

UCS_CLASS_DECLARE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);

#endif
