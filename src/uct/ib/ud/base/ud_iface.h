/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCT_UD_IFACE_H
#define UCT_UD_IFACE_H

#include <uct/base/uct_worker.h>
#include <uct/ib/base/ib_iface.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/datastruct/ptr_array.h>
#include <ucs/datastruct/sglib.h>
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/async/async.h>
#include <ucs/time/timer_wheel.h>
#include <ucs/sys/compiler_def.h>

#include "ud_def.h"
#include "ud_ep.h"
#include "ud_iface_common.h"

BEGIN_C_DECLS

#define UCT_UD_MIN_INLINE   48

enum {
    UCT_UD_IFACE_STAT_RX_DROP,
    UCT_UD_IFACE_STAT_LAST
};

/* TODO: maybe tx_moderation can be defined at compile-time since tx completions are used only to know how much space is there in tx qp */

typedef struct uct_ud_iface_config {
    uct_ib_iface_config_t         super;
    uct_ud_iface_common_config_t  ud_common;
    double                        peer_timeout;
    double                        slow_timer_backoff;
    int                           dgid_check;
} uct_ud_iface_config_t;


struct uct_ud_iface_peer {
    uct_ud_iface_peer_t   *next;
    union ibv_gid          dgid;
    uint16_t               dlid;
    uint32_t               dst_qpn;
    uint32_t               conn_id_last;
    ucs_list_link_t        ep_list; /* ep list ordered by connection id */
};


static inline int uct_ud_iface_peer_cmp(uct_ud_iface_peer_t *a, uct_ud_iface_peer_t *b) {
    return (int)a->dst_qpn - (int)b->dst_qpn ||
           memcmp(a->dgid.raw, b->dgid.raw, sizeof(union ibv_gid)) ||
           (int)a->dlid - (int)b->dlid;
}

static inline int uct_ud_iface_peer_hash(uct_ud_iface_peer_t *a) {
    return (a->dlid + a->dgid.global.interface_id + a->dgid.global.subnet_prefix)
                    % UCT_UD_HASH_SIZE;
}

SGLIB_DEFINE_LIST_PROTOTYPES(uct_ud_iface_peer_t, uct_ud_iface_peer_cmp, next)
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_ud_iface_peer_t, UCT_UD_HASH_SIZE,
                                         uct_ud_iface_peer_hash)



#ifdef UCT_UD_EP_DEBUG_HOOKS

typedef ucs_status_t (*uct_ud_iface_hook_t)(uct_ud_iface_t *iface, uct_ud_neth_t *neth);

#define UCT_UD_IFACE_HOOK_DECLARE(_name) \
    uct_ud_iface_hook_t _name

#define UCT_UD_IFACE_HOOK_CALL_RX(_iface, _neth, _len) \
    if ((_iface)->rx.hook(_iface, _neth) != UCS_OK) { \
        ucs_trace_data("RX: dropping packet"); \
        return; \
    }

#define UCT_UD_IFACE_HOOK_INIT(_iface) { \
        (_iface)->rx.hook = uct_ud_iface_null_hook; \
    }

static inline ucs_status_t uct_ud_iface_null_hook(uct_ud_iface_t *iface,
                                                  uct_ud_neth_t *neth)
{
    return UCS_OK;
}

#else

#define UCT_UD_IFACE_HOOK_DECLARE(_name)
#define UCT_UD_IFACE_HOOK_CALL_RX(_iface, _neth, _len)
#define UCT_UD_IFACE_HOOK_INIT(_iface)

#endif

typedef struct uct_ud_iface_ops {
    uct_ib_iface_ops_t        super;
    void                      (*async_progress)(uct_ud_iface_t *iface);
    void                      (*tx_skb)(uct_ud_ep_t *ep, uct_ud_send_skb_t *skb,
                                        int solicited);
} uct_ud_iface_ops_t;

struct uct_ud_iface {
    uct_ib_iface_t           super;
    struct ibv_qp           *qp;
    struct {
        ucs_mpool_t          mp;
        unsigned             available;
        unsigned             quota;
        ucs_queue_head_t     pending_q;
        UCT_UD_IFACE_HOOK_DECLARE(hook);
    } rx;
    struct {
        uct_ud_send_skb_t     *skb; /* ready to use skb */
        uct_ud_send_skb_inl_t  skb_inl;
        ucs_mpool_t            mp;
        int16_t                available;
        unsigned               unsignaled;
        /* pool of skbs that are reserved for retransmissions */
        ucs_queue_head_t       resend_skbs;
        unsigned               resend_skbs_quota;
        ucs_arbiter_t          pending_q;
        int                    pending_q_len;
        int                    in_pending;
        ucs_queue_head_t       async_comp_q;
    } tx;
    struct {
        ucs_time_t           peer_timeout;
        double               slow_timer_backoff;
        unsigned             tx_qp_len;
        unsigned             max_inline;
        int                  check_grh_dgid;
        unsigned             gid_len;
    } config;

    UCS_STATS_NODE_DECLARE(stats);

    ucs_ptr_array_t       eps;
    uct_ud_iface_peer_t  *peers[UCT_UD_HASH_SIZE];
    struct {
        ucs_twheel_t              slow_timer;
        ucs_time_t                slow_tick;
        int                       timer_id;
    } async;
};

UCS_CLASS_DECLARE(uct_ud_iface_t, uct_ud_iface_ops_t*, uct_md_h,
                  uct_worker_h, const uct_iface_params_t*,
                  unsigned, const uct_ud_iface_config_t*)

struct uct_ud_ctl_hdr {
    uint8_t                    type;
    uint8_t                    reserved[3];
    union {
        struct {
            uct_ud_ep_addr_t   ep_addr;
            uint32_t           conn_id;
        } conn_req;
        struct {
            uint32_t           src_ep_id;
        } conn_rep;
        uint32_t               data;
    };
    uct_ud_peer_name_t         peer;
    /* For CREQ packet, IB address follows */
} UCS_S_PACKED;


extern ucs_config_field_t uct_ud_iface_config_table[];

ucs_status_t uct_ud_iface_query(uct_ud_iface_t *iface, uct_iface_attr_t *iface_attr);
void uct_ud_iface_release_desc(uct_recv_desc_t *self, void *desc);

ucs_status_t uct_ud_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *addr);

void uct_ud_iface_add_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep);
void uct_ud_iface_remove_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep);
void uct_ud_iface_replace_ep(uct_ud_iface_t *iface, uct_ud_ep_t *old_ep, uct_ud_ep_t *new_ep);

ucs_status_t uct_ud_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                uct_completion_t *comp);

ucs_status_t uct_ud_iface_complete_init(uct_ud_iface_t *iface);

void uct_ud_iface_remove_async_handlers(uct_ud_iface_t *iface);

void uct_ud_dump_packet(uct_base_iface_t *iface, uct_am_trace_type_t type,
                        void *data, size_t length, size_t valid_length,
                        char *buffer, size_t max);


static UCS_F_ALWAYS_INLINE int uct_ud_iface_can_tx(uct_ud_iface_t *iface)
{
    return iface->tx.available > 0;
}

static UCS_F_ALWAYS_INLINE int uct_ud_iface_has_skbs(uct_ud_iface_t *iface)
{
    return iface->tx.skb || !ucs_mpool_is_empty(&iface->tx.mp);
}


uct_ud_send_skb_t *uct_ud_iface_resend_skb_get(uct_ud_iface_t *iface);

static inline void
uct_ud_iface_resend_skb_put(uct_ud_iface_t *iface, uct_ud_send_skb_t *skb)
{
    if (skb != (void*)&iface->tx.skb_inl.super) {
        ucs_queue_push(&iface->tx.resend_skbs, &skb->queue);
    }
}

static inline uct_ib_address_t* uct_ud_creq_ib_addr(uct_ud_ctl_hdr_t *conn_req)
{
    ucs_assert(conn_req->type == UCT_UD_PACKET_CREQ);
    return (uct_ib_address_t*)(conn_req + 1);
}

static UCS_F_ALWAYS_INLINE void uct_ud_enter(uct_ud_iface_t *iface)
{
    UCS_ASYNC_BLOCK(iface->super.super.worker->async);
}

static UCS_F_ALWAYS_INLINE void uct_ud_leave(uct_ud_iface_t *iface)
{
    UCS_ASYNC_UNBLOCK(iface->super.super.worker->async);
}

static UCS_F_ALWAYS_INLINE int
uct_ud_iface_check_grh(uct_ud_iface_t *iface, void *grh_end, int is_grh_present)
{
    void *dest_gid, *local_gid;

    if (!iface->config.check_grh_dgid) {
        return 1;
    }

    if (ucs_unlikely(!is_grh_present)) {
        ucs_warn("RoCE packet does not contain GRH");
        return 1;
    }

    local_gid = (char*)iface->super.gid.raw + (16 - iface->config.gid_len);
    dest_gid  = (char*)grh_end - iface->config.gid_len;

    if (memcmp(local_gid, dest_gid, iface->config.gid_len)) {
        UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_UD_IFACE_STAT_RX_DROP, 1);
        ucs_trace_data("Drop packet with wrong dgid");
        return 0;
    }

    return 1;
}

/*
management of connecting endpoints (cep)

Such endpoint are created either by explicitely calling ep_create_connected()
or implicitely as a result of UD connection protocol. Calling
ep_create_connected() may reuse already existing endpoint that was implicitely
created.

UD connection protocol

The protocol allows connection establishment in environment where UD packets
can be dropped, duplicated or reordered. The connection is done as 3 way
handshake:

1: CREQ (src_if_addr, src_ep_addr, conn_id)
Connection request. It includes source interface address, source ep address
and connection id.

Connection id is essentially a counter of endpoints that are created by
ep_create_connected(). The counter is per destination interface. Purpose of
conn_id is to ensure order between multiple CREQ packets and to handle
simultanuous connection establishment. The case when both sides call
ep_create_connected(). The rule is that connected endpoints must have
same conn_id.

2: CREP (dest_ep_id)

Connection reply. It includes id of destination endpoint and optinally ACK
request flag. From this point reliability is handled by UD protocol as
source and destination endpoint ids are known.

Endpoint may be created upon reception of CREQ. It is possible that the
endpoint already exists because CREQ is retransmitted or because of
simultaneous connection. In any case endpoint connection id must be
equal to connection id in CREQ.

3: ACK

Ack on connection reply. It may be send as part of the data packet.

Implicit endpoints reuse

Endpoints created upon receive of CREP request can be re-used when
application calls ep_create_connected().

Data structure

Hash table and double linked sorted list:
hash(src_if_addr) -> peer ->ep (list sorted in descending order)

List is used to save memory (8 bytes instead of 500-1000 bytes of hashtable)
In many cases list will provide fast lookup and insertion.
It is expected that most of connect requests will arrive in order. In
such case the insertion is O(1) because it is done to the head of the
list. Lookup is O(number of 'passive' eps) which is expected to be small.

TODO: add and maintain pointer to the list element with conn_id equal to
conn_last_id. This will allow for O(1) endpoint lookup.

Connection id assignment:

  0 1 ... conn_last_id, +1, +2, ... UCT_UD_EP_CONN_ID_MAX

Ids upto (not including) conn_last_id are already assigned to endpoints.
Any endpoint with conn_id >= conn_last_id is created on receive of CREQ
There may be holes because CREQs are not received in order.

Call to ep_create_connected() will try reuse endpoint with
conn_id = conn_last_id

If there is no such endpoint new endpoint with id conn_last_id
will be created.

In both cases conn_last_id = conn_last_id + 1

*/
void uct_ud_iface_cep_init(uct_ud_iface_t *iface);

/* find ep that is connected to (src_if, src_ep),
 * if conn_id == UCT_UD_EP_CONN_ID_MAX then try to
 * reuse ep with conn_id == conn_last_id
 */
uct_ud_ep_t *uct_ud_iface_cep_lookup(uct_ud_iface_t *iface,
                                     const uct_ib_address_t *src_ib_addr,
                                     const uct_ud_iface_addr_t *src_if_addr,
                                     uint32_t conn_id);

/* remove ep */
void uct_ud_iface_cep_remove(uct_ud_ep_t *ep);

/*
 * rollback last ordered insert (conn_id == UCT_UD_EP_CONN_ID_MAX).
 */
void uct_ud_iface_cep_rollback(uct_ud_iface_t *iface,
                               const uct_ib_address_t *src_ib_addr,
                               const uct_ud_iface_addr_t *src_if_addr,
                               uct_ud_ep_t *ep);

/* insert new ep that is connected to src_if_addr */
ucs_status_t uct_ud_iface_cep_insert(uct_ud_iface_t *iface,
                                     const uct_ib_address_t *src_ib_addr,
                                     const uct_ud_iface_addr_t *src_if_addr,
                                     uct_ud_ep_t *ep, uint32_t conn_id);

void uct_ud_iface_cep_cleanup(uct_ud_iface_t *iface);

/* get time of the last async wakeup */
static UCS_F_ALWAYS_INLINE ucs_time_t
uct_ud_iface_get_async_time(uct_ud_iface_t *iface)
{
    return iface->super.super.worker->async->last_wakeup;
}

static UCS_F_ALWAYS_INLINE void
uct_ud_iface_progress_pending(uct_ud_iface_t *iface, const uintptr_t is_async)
{

    if (!uct_ud_iface_can_tx(iface)) {
        return;
    }

    iface->tx.in_pending = 1;
    ucs_arbiter_dispatch(&iface->tx.pending_q, 1,
             uct_ud_ep_do_pending, (void *)is_async);
    iface->tx.in_pending = 0;
}

static UCS_F_ALWAYS_INLINE void
uct_ud_iface_progress_pending_tx(uct_ud_iface_t *iface)
{
    if (ucs_unlikely(iface->tx.pending_q_len > 0 &&
                     iface->tx.in_pending == 0)) {
        iface->tx.in_pending = 1;
        ucs_arbiter_dispatch(&iface->tx.pending_q, 1,
                             uct_ud_ep_do_pending, (void *)0);
        iface->tx.in_pending = 0;
    }
}

/* Go over all active eps and remove them. Do it this way because class destructors are not
 * virtual
 */
#define UCT_UD_IFACE_DELETE_EPS(_iface, _ep_type_t) \
    { \
        int _i; \
        _ep_type_t *_ep; \
        ucs_ptr_array_for_each(_ep, _i, &(_iface)->eps) { \
            UCS_CLASS_DELETE(_ep_type_t, _ep); \
        } \
    }

ucs_status_t uct_ud_iface_dispatch_pending_rx_do(uct_ud_iface_t *iface);

void uct_ud_iface_handle_failure(uct_ib_iface_t *iface, void *arg,
                                 ucs_status_t status);

ucs_status_t uct_ud_iface_event_arm(uct_iface_h tl_iface, unsigned events);

void uct_ud_iface_progress_enable(uct_iface_h tl_iface, unsigned flags);

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_ud_iface_dispatch_pending_rx(uct_ud_iface_t *iface)
{
    if (ucs_likely(ucs_queue_is_empty(&iface->rx.pending_q))) {
        return UCS_OK;
    }
    return uct_ud_iface_dispatch_pending_rx_do(iface);
}

void uct_ud_iface_dispatch_async_comps_do(uct_ud_iface_t *iface);

static UCS_F_ALWAYS_INLINE void
uct_ud_iface_dispatch_zcopy_comps(uct_ud_iface_t *iface)
{
    if (ucs_likely(ucs_queue_is_empty(&iface->tx.async_comp_q))) {
        return;
    }
    uct_ud_iface_dispatch_async_comps_do(iface);
}

#if ENABLE_PARAMS_CHECK
#define UCT_UD_CHECK_LENGTH(iface, header_len, payload_len, msg) \
     do { \
         int mtu; \
         mtu =  uct_ib_mtu_value(uct_ib_iface_port_attr(&(iface)->super)->active_mtu); \
         UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + payload_len + header_len, \
                          0, mtu, msg); \
     } while(0);

#define UCT_UD_CHECK_BCOPY_LENGTH(iface, len) \
    UCT_UD_CHECK_LENGTH(iface, 0, len, "am_bcopy length")

#define UCT_UD_CHECK_ZCOPY_LENGTH(iface, header_len, payload_len) \
    UCT_UD_CHECK_LENGTH(iface, header_len, payload_len, "am_zcopy payload")

#else
#define UCT_UD_CHECK_ZCOPY_LENGTH(iface, header_len, payload_len)
#define UCT_UD_CHECK_BCOPY_LENGTH(iface, len)
#endif

END_C_DECLS

#endif
