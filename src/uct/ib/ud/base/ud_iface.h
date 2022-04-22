/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCT_UD_IFACE_H
#define UCT_UD_IFACE_H

#include <uct/base/uct_worker.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_iface.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/datastruct/ptr_array.h>
#include <ucs/datastruct/sglib.h>
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/async/async.h>
#include <ucs/time/timer_wheel.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/sys/sock.h>

#include "ud_def.h"
#include "ud_ep.h"
#include "ud_iface_common.h"

BEGIN_C_DECLS


#define UCT_UD_MIN_TIMER_TIMER_BACKOFF 1.0


/** @file ud_iface.h */

enum {
    UCT_UD_IFACE_STAT_RX_DROP,
    UCT_UD_IFACE_STAT_LAST
};


/* flags for uct_ud_iface_send_ctl() */
enum {
    UCT_UD_IFACE_SEND_CTL_FLAG_INLINE    = UCS_BIT(0),
    UCT_UD_IFACE_SEND_CTL_FLAG_SOLICITED = UCS_BIT(1),
    UCT_UD_IFACE_SEND_CTL_FLAG_SIGNALED  = UCS_BIT(2)
};


/* TODO: maybe tx_moderation can be defined at compile-time since tx completions are used only to know how much space is there in tx qp */

typedef struct uct_ud_iface_config {
    uct_ib_iface_config_t         super;
    uct_ud_iface_common_config_t  ud_common;
    double                        linger_timeout;
    double                        peer_timeout;
    double                        min_poke_time;
    double                        timer_tick;
    double                        timer_backoff;
    double                        event_timer_tick;
    int                           dgid_check;
    unsigned                      max_window;
    unsigned                      rx_async_max_poll;
} uct_ud_iface_config_t;


#if UCT_UD_EP_DEBUG_HOOKS

typedef ucs_status_t (*uct_ud_iface_hook_t)(uct_ud_iface_t *iface, uct_ud_neth_t *neth);


#define UCT_UD_IFACE_HOOK_DECLARE(_name) \
    uct_ud_iface_hook_t _name;


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
    unsigned                  (*async_progress)(uct_ud_iface_t *iface);
    uint16_t                  (*send_ctl)(uct_ud_ep_t *ud_ep, uct_ud_send_skb_t *skb,
                                          const uct_ud_iov_t *iov, uint16_t iovcnt,
                                          int flags, int max_log_sge);
    ucs_status_t              (*ep_new)(const uct_ep_params_t* params,
                                        uct_ep_h *ep_p);
    void                      (*ep_free)(uct_ep_h ep);
    ucs_status_t              (*create_qp)(uct_ib_iface_t *iface, uct_ib_qp_attr_t *attr,
                                           struct ibv_qp **qp_p);
    void                      (*destroy_qp)(uct_ud_iface_t *ud_iface);
    ucs_status_t              (*unpack_peer_address)(uct_ud_iface_t *iface,
                                                     const uct_ib_address_t *ib_addr,
                                                     const uct_ud_iface_addr_t *if_addr,
                                                     int path_index, void *address_p);
    void*                     (*ep_get_peer_address)(uct_ud_ep_t *ud_ep);
    size_t                    (*get_peer_address_length)();
    const char*               (*peer_address_str)(const uct_ud_iface_t *iface,
                                                  const void *address,
                                                  char *str, size_t max_size);
} uct_ud_iface_ops_t;


/* device GIDs set */
KHASH_TYPE(uct_ud_iface_gid, union ibv_gid, char);


static UCS_F_ALWAYS_INLINE
khint32_t uct_ud_iface_kh_gid_hash_func(union ibv_gid gid)
{
    return kh_int64_hash_func(gid.global.subnet_prefix ^
                              gid.global.interface_id);
}


static UCS_F_ALWAYS_INLINE int
uct_ud_gid_equal(const union ibv_gid *a, const union ibv_gid *b, size_t length)
{
    ucs_assert(length <= sizeof(union ibv_gid));
    return !memcmp(UCS_PTR_BYTE_OFFSET(a, sizeof(*a) - length),
                   UCS_PTR_BYTE_OFFSET(b, sizeof(*b) - length), length);
}


static UCS_F_ALWAYS_INLINE int
uct_ud_iface_kh_gid_hash_equal(union ibv_gid a, union ibv_gid b)
{
    return uct_ud_gid_equal(&a, &b, sizeof(a));
}


KHASH_IMPL(uct_ud_iface_gid, union ibv_gid, char, 0,
           uct_ud_iface_kh_gid_hash_func, uct_ud_iface_kh_gid_hash_equal)


struct uct_ud_iface {
    uct_ib_iface_t           super;
    struct ibv_qp           *qp;
    struct {
        ucs_mpool_t          mp;
        unsigned             available;
        unsigned             quota;
        unsigned             async_max_poll;
        ucs_queue_head_t     pending_q;
        UCT_UD_IFACE_HOOK_DECLARE(hook)
    } rx;
    struct {
        uct_ud_send_skb_t     *skb; /* ready to use skb */
        ucs_mpool_t            mp;
        /* got async events but pending queue was not dispatched */
        uint8_t                async_before_pending;
        int16_t                available;
        unsigned               unsignaled;
        ucs_queue_head_t       outstanding_q;
        ucs_arbiter_t          pending_q;
        ucs_queue_head_t       async_comp_q;
        ucs_twheel_t           timer;
        ucs_time_t             tick;
        double                 timer_backoff;
        unsigned               timer_sweep_count;
    } tx;
    struct {
        ucs_time_t           linger_timeout;
        ucs_time_t           peer_timeout;
        ucs_time_t           min_poke_time;
        unsigned             tx_qp_len;
        unsigned             max_inline;
        int                  check_grh_dgid;
        unsigned             max_window;
    } config;

    UCS_STATS_NODE_DECLARE(stats)

    ucs_conn_match_ctx_t  conn_match_ctx;

    ucs_ptr_array_t       eps;
    struct {
        ucs_time_t                tick;
        int                       timer_id;
        void                      *event_arg;
        uct_async_event_cb_t      event_cb;
        unsigned                  disable;
    } async;

    /* used for GRH GID filter */
    struct {
        union ibv_gid             last;
        unsigned                  last_len;
        khash_t(uct_ud_iface_gid) hash;
    } gid_table;
};


UCS_CLASS_DECLARE(uct_ud_iface_t, uct_ud_iface_ops_t*, uct_iface_ops_t*,
                  uct_md_h, uct_worker_h, const uct_iface_params_t*,
                  const uct_ud_iface_config_t*, uct_ib_iface_init_attr_t*)


struct uct_ud_ctl_hdr {
    uint8_t                     type;
    uint8_t                     reserved[3];
    union {
        struct {
            uct_ud_ep_addr_t    ep_addr;
            uct_ud_ep_conn_sn_t conn_sn;
            uint8_t             path_index;
        } conn_req;
        struct {
            uint32_t            src_ep_id;
        } conn_rep;
        uint32_t                data;
    };
    uct_ud_peer_name_t          peer;
    /* For CREQ packet, IB address follows */
} UCS_S_PACKED;


extern ucs_config_field_t uct_ud_iface_config_table[];


ucs_status_t uct_ud_iface_query(uct_ud_iface_t *iface,
                                uct_iface_attr_t *iface_attr,
                                size_t am_max_iov, size_t am_max_hdr);

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

union ibv_gid* uct_ud_grh_get_dgid(struct ibv_grh *grh, size_t dgid_len);

uct_ud_send_skb_t *uct_ud_iface_ctl_skb_get(uct_ud_iface_t *iface);

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

1: CREQ (src_if_addr, src_ep_addr, conn_sn)
Connection request. It includes source interface address, source ep address
and connection id.

Connection id is essentially a counter of endpoints that are created by
ep_create_connected(). The counter is per destination interface. Purpose of
conn_sn is to ensure order between multiple CREQ packets and to handle
simultaneous connection establishment. The case when both sides call
ep_create_connected(). The rule is that connected endpoints must have
same conn_sn.

2: CREP (dest_ep_id)

Connection reply. It includes id of destination endpoint and optionally ACK
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
application calls ep_create_connected(). */

void uct_ud_iface_cep_cleanup(uct_ud_iface_t *iface);

ucs_status_t
uct_ud_iface_cep_get_conn_sn(uct_ud_iface_t *iface,
                             const uct_ib_address_t *ib_addr,
                             const uct_ud_iface_addr_t *if_addr,
                             int path_index, uct_ud_ep_conn_sn_t *conn_sn_p);

ucs_status_t uct_ud_iface_cep_insert_ep(uct_ud_iface_t *iface,
                                        const uct_ib_address_t *ib_addr,
                                        const uct_ud_iface_addr_t *if_addr,
                                        int path_index,
                                        uct_ud_ep_conn_sn_t conn_sn,
                                        uct_ud_ep_t *ep);

uct_ud_ep_t *uct_ud_iface_cep_get_ep(uct_ud_iface_t *iface,
                                     const uct_ib_address_t *ib_addr,
                                     const uct_ud_iface_addr_t *if_addr,
                                     int path_index,
                                     uct_ud_ep_conn_sn_t conn_sn,
                                     int is_private);

void uct_ud_iface_cep_remove_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep);

unsigned uct_ud_iface_dispatch_pending_rx_do(uct_ud_iface_t *iface);

ucs_status_t uct_ud_iface_event_arm(uct_iface_h tl_iface, unsigned events);

void uct_ud_iface_progress_enable(uct_iface_h tl_iface, unsigned flags);

void uct_ud_iface_progress_disable(uct_iface_h tl_iface, unsigned flags);

void uct_ud_iface_ctl_skb_complete(uct_ud_iface_t *iface,
                                   uct_ud_ctl_desc_t *cdesc, int is_async);

void uct_ud_iface_send_completion(uct_ud_iface_t *iface, uint16_t sn,
                                  int is_async);

unsigned
uct_ud_iface_dispatch_async_comps_do(uct_ud_iface_t *iface, uct_ud_ep_t *ep);


static UCS_F_ALWAYS_INLINE int uct_ud_iface_can_tx(uct_ud_iface_t *iface)
{
    return iface->tx.available > 0;
}


static UCS_F_ALWAYS_INLINE int uct_ud_iface_has_skbs(uct_ud_iface_t *iface)
{
    return iface->tx.skb || !ucs_mpool_is_empty(&iface->tx.mp);
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
uct_ud_iface_check_grh(uct_ud_iface_t *iface, void *packet, int is_grh_present,
                       uint8_t roce_pkt_type)
{
    struct ibv_grh *grh = (struct ibv_grh *)packet;
    size_t gid_len;
    union ibv_gid *gid;
    khiter_t khiter;
    char gid_str[128] UCS_V_UNUSED;

    if (!iface->config.check_grh_dgid) {
        return 1;
    }

    if (ucs_unlikely(!is_grh_present)) {
        ucs_warn("RoCE packet does not contain GRH");
        return 1;
    }

    /*
     * Take the packet type from CQE, because:
     * 1. According to Annex17_RoCEv2 (A17.4.5.1):
     * For UD, the Completion Queue Entry (CQE) includes remote address
     * information (InfiniBand Specification Vol. 1 Rev 1.2.1 Section 11.4.2.1).
     * For RoCEv2, the remote address information comprises the source L2
     * Address and a flag that indicates if the received frame is an IPv4,
     * IPv6 or RoCE packet.
     * 2. According to PRM, for responder UD/DC over RoCE sl represents RoCE
     * packet type as:
     * bit 3    : when set R-RoCE frame contains an UDP header otherwise not
     * Bits[2:0]: L3_Header_Type, as defined below
     *     - 0x0 : GRH - (RoCE v1.0)
     *     - 0x1 : IPv6 - (RoCE v1.5/v2.0)
     *     - 0x2 : IPv4 - (RoCE v1.5/v2.0)
     */
    gid_len = ((roce_pkt_type & UCT_IB_CQE_SL_PKTYPE_MASK) == 0x2) ?
              UCS_IPV4_ADDR_LEN : UCS_IPV6_ADDR_LEN;

    if (ucs_likely((gid_len == iface->gid_table.last_len) &&
                    uct_ud_gid_equal(&grh->dgid, &iface->gid_table.last,
                                     gid_len))) {
        return 1;
    }

    gid    = uct_ud_grh_get_dgid(grh, gid_len);
    khiter = kh_get(uct_ud_iface_gid, &iface->gid_table.hash, *gid);
    if (ucs_likely(khiter != kh_end(&iface->gid_table.hash))) {
        iface->gid_table.last     = *gid;
        iface->gid_table.last_len = gid_len;
        return 1;
    }

    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_UD_IFACE_STAT_RX_DROP, 1);
    ucs_trace_data("iface %p: drop packet with wrong dgid %s", iface,
                   uct_ib_gid_str(gid, gid_str, sizeof(gid_str)));
    return 0;
}


/* get time of the last async wakeup */
static UCS_F_ALWAYS_INLINE ucs_time_t
uct_ud_iface_get_async_time(uct_ud_iface_t *iface)
{
    return iface->super.super.worker->async->last_wakeup;
}


static UCS_F_ALWAYS_INLINE ucs_time_t
uct_ud_iface_get_time(uct_ud_iface_t *iface)
{
    return ucs_get_time();
}


static UCS_F_ALWAYS_INLINE void
uct_ud_iface_twheel_sweep(uct_ud_iface_t *iface)
{
    if (iface->tx.timer_sweep_count++ % UCT_UD_SKIP_SWEEP) {
        return;
    }

    if (ucs_twheel_is_empty(&iface->tx.timer)) {
        return;
    }

    ucs_twheel_sweep(&iface->tx.timer, uct_ud_iface_get_time(iface));
}


static UCS_F_ALWAYS_INLINE void
uct_ud_iface_progress_pending(uct_ud_iface_t *iface, const uintptr_t is_async)
{
    uct_ud_iface_twheel_sweep(iface);

    if (!is_async) {
        iface->tx.async_before_pending = 0;
    }

    if (!uct_ud_iface_can_tx(iface)) {
        return;
    }

    ucs_arbiter_dispatch(&iface->tx.pending_q, 1, uct_ud_ep_do_pending,
                         (void *)is_async);
}


static UCS_F_ALWAYS_INLINE int
uct_ud_iface_has_pending_async_ev(uct_ud_iface_t *iface)
{
    return iface->tx.async_before_pending;
}


static UCS_F_ALWAYS_INLINE void
uct_ud_iface_raise_pending_async_ev(uct_ud_iface_t *iface)
{
    if (!ucs_arbiter_is_empty(&iface->tx.pending_q)) {
        iface->tx.async_before_pending = 1;
    }
}


static UCS_F_ALWAYS_INLINE const void *
uct_ud_ep_get_peer_address(const ucs_conn_match_elem_t *elem)
{
    uct_ud_ep_t *ep            = ucs_container_of(elem, uct_ud_ep_t,
                                                  conn_match);
    uct_ib_iface_t *ib_iface   = ucs_derived_of(ep->super.super.iface,
                                                uct_ib_iface_t);
    return uct_iface_invoke_ops_func(ib_iface, uct_ud_iface_ops_t,
                                     ep_get_peer_address, ep);
}


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_ud_iface_unpack_peer_address(uct_ud_iface_t *iface,
                                 const uct_ib_address_t *ib_addr,
                                 const uct_ud_iface_addr_t *if_addr,
                                 unsigned path_index, void *address_p)
{
    uct_ud_iface_ops_t *ud_ops = ucs_derived_of(iface->super.ops,
                                                uct_ud_iface_ops_t);
    return ud_ops->unpack_peer_address(iface, ib_addr, if_addr,
                                       path_index, address_p);
}


static UCS_F_ALWAYS_INLINE uint16_t
uct_ud_iface_send_ctl(uct_ud_iface_t *iface, uct_ud_ep_t *ep, uct_ud_send_skb_t *skb,
                      const uct_ud_iov_t *iov, uint16_t iovcnt, int flags,
                      int max_log_sge)
{
    uct_ud_iface_ops_t *ud_ops = ucs_derived_of(iface->super.ops,
                                                uct_ud_iface_ops_t);
    return ud_ops->send_ctl(ep, skb, iov, iovcnt, flags, max_log_sge);
}


static UCS_F_ALWAYS_INLINE void
uct_ud_iface_add_ctl_desc(uct_ud_iface_t *iface, uct_ud_ctl_desc_t *cdesc)
{
    ucs_queue_push(&iface->tx.outstanding_q, &cdesc->queue);
}


static UCS_F_ALWAYS_INLINE unsigned
uct_ud_iface_dispatch_pending_rx(uct_ud_iface_t *iface)
{
    if (ucs_likely(ucs_queue_is_empty(&iface->rx.pending_q))) {
        return 0;
    }

    return uct_ud_iface_dispatch_pending_rx_do(iface);
}


static UCS_F_ALWAYS_INLINE unsigned
uct_ud_iface_dispatch_async_comps(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    if (ucs_likely(ucs_queue_is_empty(&iface->tx.async_comp_q))) {
        return 0;
    }

    return uct_ud_iface_dispatch_async_comps_do(iface, ep);
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
    UCT_UD_CHECK_LENGTH(iface, 0, len, "am_bcopy")

#define UCT_UD_CHECK_ZCOPY_LENGTH(iface, header_len, payload_len) \
    UCT_UD_CHECK_LENGTH(iface, header_len, payload_len, "am_zcopy payload")

#else
#define UCT_UD_CHECK_ZCOPY_LENGTH(iface, header_len, payload_len)
#define UCT_UD_CHECK_BCOPY_LENGTH(iface, len)
#endif

END_C_DECLS

#endif
