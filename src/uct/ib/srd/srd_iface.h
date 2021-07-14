/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_SRD_IFACE_H
#define UCT_SRD_IFACE_H

#include "srd_def.h"
#include "srd_ep.h"

#include <uct/base/uct_worker.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/ib/base/ib_verbs.h>
#include <uct/ib/ud/base/ud_iface_common.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/datastruct/ptr_array.h>
#include <ucs/datastruct/sglib.h>
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/async/async.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/sys/sock.h>


BEGIN_C_DECLS


/** @file srd_iface.h */


enum {
    UCT_SRD_IFACE_STAT_RX_DROP,
    UCT_SRD_IFACE_STAT_LAST
};


typedef struct uct_srd_iface_config {
    uct_ib_iface_config_t         super;
    uct_ud_iface_common_config_t  ud_common;
    struct {
        size_t max_get_zcopy;
    } tx;

    struct {
        double               soft_thresh;
        double               hard_thresh;
        unsigned             wnd_size;
    } fc;

} uct_srd_iface_config_t;


struct uct_srd_iface {
    uct_ib_iface_t             super;
    struct ibv_qp              *qp;
#ifdef HAVE_DECL_EFA_DV_RDMA_READ
    struct ibv_qp_ex           *qp_ex;
#endif
    struct {
        ucs_mpool_t            mp;
        unsigned               available;
        unsigned               quota;
    } rx;
    struct {
        uct_srd_am_short_hdr_t am_inl_hdr;
        uct_srd_put_hdr_t      put_hdr;     /* to emulate put with send/recv */
        uct_srd_send_desc_t    *desc;       /* cached ready to use desc */
        uct_srd_send_op_t      *send_op;    /* cached ready to use send op */
        ucs_mpool_t            desc_mp;
        ucs_mpool_t            send_op_mp;
        int32_t                available;
        ucs_arbiter_t          pending_q;
        struct ibv_sge         sge[UCT_IB_MAX_IOV];
        struct ibv_send_wr     wr_inl;
        struct ibv_send_wr     wr_desc;
    } tx;
    struct {
        unsigned               tx_qp_len;
        unsigned               max_inline;
        size_t                 max_send_sge;
        size_t                 max_get_bcopy;
        size_t                 max_get_zcopy;

        /* Threshold to send "soft" FC credit request. The peer will try to
         * piggy-back credits grant to the counter AM, if any. */
        int16_t              fc_soft_thresh;

        /* Threshold to sent "hard" credits request. The peer will grant
         * credits in a separate ctrl message as soon as it handles this request. */
        int16_t              fc_hard_thresh;

        uint16_t             fc_wnd_size;
    } config;

    UCS_STATS_NODE_DECLARE(stats)

    ucs_conn_match_ctx_t       conn_match_ctx;

    ucs_ptr_array_t            eps;
};


struct uct_srd_ctl_hdr {
    uint8_t                         type;
    union {
        struct {
            uct_srd_ep_addr_t       ep_addr;
            uct_srd_ep_conn_sn_t    conn_sn;
            uint8_t                 path_index;
        } conn_req;
        struct {
            uint32_t                src_ep_id;
        } conn_rep;
    };
    uct_srd_peer_name_t             peer;
    /* For CREQ packet, IB address follows */
} UCS_S_PACKED;


extern ucs_config_field_t uct_srd_iface_config_table[];


ucs_status_t uct_srd_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_srd_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *addr);

ucs_status_t uct_srd_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                 uct_completion_t *comp);

unsigned uct_srd_iface_dispatch_pending_rx_do(uct_srd_iface_t *iface);

void uct_srd_iface_add_ep(uct_srd_iface_t *iface, uct_srd_ep_t *ep);

void uct_srd_iface_remove_ep(uct_srd_iface_t *iface, uct_srd_ep_t *ep);

void uct_srd_iface_replace_ep(uct_srd_iface_t *iface, uct_srd_ep_t *old_ep,
                              uct_srd_ep_t *new_ep);

void uct_srd_iface_cep_remove_ep(uct_srd_iface_t *iface, uct_srd_ep_t *ep);

void uct_srd_iface_send_op_release(uct_srd_send_op_t *send_op);

void uct_srd_iface_send_op_ucomp_release(uct_srd_send_op_t *send_op);

void uct_srd_dump_packet(uct_base_iface_t *iface, uct_am_trace_type_t type,
                         void *data, size_t length, size_t valid_length,
                         char *buffer, size_t max);

/* management of connecting endpoints (cep) is similar to UD */
void uct_srd_iface_cep_cleanup(uct_srd_iface_t *iface);

uct_srd_ep_conn_sn_t
uct_srd_iface_cep_get_conn_sn(uct_srd_iface_t *iface,
                              const uct_ib_address_t *ib_addr,
                              const uct_srd_iface_addr_t *if_addr,
                              int path_index);

void uct_srd_iface_cep_insert_ep(uct_srd_iface_t *iface,
                                 const uct_ib_address_t *ib_addr,
                                 const uct_srd_iface_addr_t *if_addr,
                                 int path_index, uct_srd_ep_conn_sn_t conn_sn,
                                 uct_srd_ep_t *ep);

uct_srd_ep_t *uct_srd_iface_cep_get_ep(uct_srd_iface_t *iface,
                                       const uct_ib_address_t *ib_addr,
                                       const uct_srd_iface_addr_t *if_addr,
                                       int path_index,
                                       uct_srd_ep_conn_sn_t conn_sn,
                                       int is_private);

ucs_status_t
uct_srd_iface_unpack_peer_address(uct_srd_iface_t *iface,
                                  const uct_ib_address_t *ib_addr,
                                  const uct_srd_iface_addr_t *if_addr,
                                  int path_index, void *address_p);

static UCS_F_ALWAYS_INLINE int
uct_srd_iface_can_tx(const uct_srd_iface_t *iface)
{
    return iface->tx.available > 0;
}

static UCS_F_ALWAYS_INLINE int
uct_srd_iface_has_desc(uct_srd_iface_t *iface)
{
    return iface->tx.desc || !ucs_mpool_is_empty(&iface->tx.desc_mp);
}

static UCS_F_ALWAYS_INLINE int
uct_srd_iface_has_send_op(uct_srd_iface_t *iface)
{
    return iface->tx.send_op || !ucs_mpool_is_empty(&iface->tx.send_op_mp);
}

static UCS_F_ALWAYS_INLINE int
uct_srd_iface_has_all_tx_resources(uct_srd_iface_t *iface)
{
    return uct_srd_iface_can_tx(iface) &&
           uct_srd_iface_has_desc(iface) &&
           uct_srd_iface_has_send_op(iface);
}

#define UCT_SRD_IFACE_GET_CACHED_TX_RES(_iface, _cache, _mpool, _res)   \
    do {                                                                \
        if (ucs_unlikely(!uct_srd_iface_can_tx(_iface))) {              \
            return NULL;                                                \
        }                                                               \
        _res = _cache;                                                  \
        if (ucs_unlikely(_res == NULL)) {                               \
            _res = ucs_mpool_get(_mpool);                               \
            if (_res == NULL) {                                         \
                ucs_trace_data("iface=%p out of tx descs", _iface);     \
                UCT_TL_IFACE_STAT_TX_NO_DESC(&_iface->super.super);     \
            }                                                           \
            _cache = _res;                                              \
        }                                                               \
    } while (0);

/*
 * NOTE: caller must NOT return desc to mpool until it is
 * removed from the cache, which is done by uct_srd_iface_complete_tx_desc().
 *
 * In case of error flow, caller must do nothing with the desc
 */
static UCS_F_ALWAYS_INLINE uct_srd_send_desc_t *
uct_srd_iface_get_send_desc(uct_srd_iface_t *iface)
{
    uct_srd_send_desc_t* desc;

    UCT_SRD_IFACE_GET_CACHED_TX_RES(iface, iface->tx.desc,
                                    &iface->tx.desc_mp, desc);
    VALGRIND_MAKE_MEM_DEFINED(&desc->lkey, sizeof(desc->lkey));

    if (desc) {
        desc->super.flags = 0;
        ucs_prefetch(desc + 1);
    }

    return desc;
}

/*
 * NOTE: caller must NOT return send_op to mpool until it is
 * removed from the cache, which is done by uct_srd_iface_complete_tx_op().
 *
 * In case of error flow, caller must do nothing with the send_op 
 */
static UCS_F_ALWAYS_INLINE uct_srd_send_op_t *
uct_srd_iface_get_send_op(uct_srd_iface_t *iface)
{
    uct_srd_send_op_t* send_op;

    UCT_SRD_IFACE_GET_CACHED_TX_RES(iface, iface->tx.send_op,
                                    &iface->tx.send_op_mp, send_op);
    if (send_op) {
        send_op->flags = 0;
    }

    return send_op;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_iface_complete_tx_desc(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                               uct_srd_send_desc_t *desc)
{
    ucs_assert(!(desc->super.flags & UCT_SRD_SEND_OP_FLAG_INVALID));
    iface->tx.desc = ucs_mpool_get(&iface->tx.desc_mp);
}

static UCS_F_ALWAYS_INLINE void
uct_srd_iface_complete_tx_op(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                             uct_srd_send_op_t *send_op)
{
    ucs_assert(!(send_op->flags & UCT_SRD_SEND_OP_FLAG_INVALID));
    iface->tx.send_op = ucs_mpool_get(&iface->tx.send_op_mp);
}

static inline uct_ib_address_t* uct_srd_creq_ib_addr(uct_srd_ctl_hdr_t *conn_req)
{
    ucs_assert(conn_req->type == UCT_SRD_PACKET_CREQ);
    return (uct_ib_address_t*)(conn_req + 1);
}

static UCS_F_ALWAYS_INLINE void uct_srd_enter(uct_srd_iface_t *iface)
{
    UCS_ASYNC_BLOCK(iface->super.super.worker->async);
}

static UCS_F_ALWAYS_INLINE void uct_srd_leave(uct_srd_iface_t *iface)
{
    UCS_ASYNC_UNBLOCK(iface->super.super.worker->async);
}

static UCS_F_ALWAYS_INLINE void
uct_srd_iface_progress_pending(uct_srd_iface_t *iface)
{
    if (!uct_srd_iface_can_tx(iface)) {
        return;
    }

    ucs_arbiter_dispatch(&iface->tx.pending_q, 1, uct_srd_ep_do_pending, NULL);
}


#if ENABLE_PARAMS_CHECK
#define UCT_SRD_CHECK_LENGTH_MTU(_iface, _tx_len, _msg) \
     do { \
         int mtu; \
         mtu =  uct_ib_mtu_value(uct_ib_iface_port_attr(&(_iface)->super)->active_mtu); \
         UCT_CHECK_LENGTH(_tx_len, 0, mtu, _msg); \
     } while(0);

#else
#define UCT_SRD_CHECK_LENGTH_MTU(_iface, _tx_len, _msg)
#endif

#define UCT_SRD_CHECK_AM_SHORT(_iface, _id, _hdr_len, _data_len, _msg) \
    UCT_CHECK_AM_ID(_id); \
    UCT_SRD_CHECK_AM_LEN(_iface, _id, _hdr_len + _data_len, \
                         _iface->config.max_inline, _msg);

#define UCT_SRD_CHECK_AM_BCOPY(_iface, _id, _data_len) \
    UCT_CHECK_AM_ID(_id); \
    UCT_SRD_CHECK_AM_LEN(_iface, _id, _data_len, \
                         _iface->super.config.seg_size, "am_bcopy");

#define UCT_SRD_CHECK_AM_ZCOPY(_iface, _id, _hdr_len, _data_len) \
    UCT_CHECK_AM_ID(_id); \
    UCT_SRD_CHECK_AM_LEN(_iface, _id, _hdr_len, \
                         _iface->super.config.seg_size, \
                         "am_zcopy_header"); \
    UCT_SRD_CHECK_AM_LEN(_iface, _id, _data_len, \
                         _iface->super.config.seg_size, \
                         "am_zcopy_data"); \
    UCT_SRD_CHECK_AM_LEN(_iface, _id, _hdr_len + _data_len, \
                         _iface->super.config.seg_size, \
                         "am_zcopy_header_and_data");


#define UCT_SRD_CHECK_AM_LEN(_iface, _id, _data_len, _max_len, _msg) \
    UCT_CHECK_LENGTH(sizeof(uct_srd_neth_t) + _data_len, 0, _max_len, _msg); \
    UCT_SRD_CHECK_LENGTH_MTU(_iface, sizeof(uct_srd_neth_t) + _data_len, _msg"_mtu");
    


END_C_DECLS

#endif
