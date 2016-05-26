/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_IFACE_H
#define UCT_RC_IFACE_H

#include "rc_def.h"

#include <uct/base/uct_iface.h>
#include <uct/ib/base/ib_log.h>
#include <uct/ib/base/ib_iface.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>


#define UCT_RC_QP_TABLE_ORDER       12
#define UCT_RC_QP_TABLE_SIZE        UCS_BIT(UCT_RC_QP_TABLE_ORDER)
#define UCT_RC_QP_TABLE_MEMB_ORDER  (UCT_IB_QPN_ORDER - UCT_RC_QP_TABLE_ORDER)
#define UCT_RC_MAX_ATOMIC_SIZE      sizeof(uint64_t)
#define UCR_RC_QP_MAX_RETRY_COUNT   7

#define UCT_RC_CHECK_AM_SHORT(am_id, length, max_inline) \
     UCT_CHECK_AM_ID(am_id); \
     UCT_CHECK_LENGTH(sizeof(uct_rc_am_short_hdr_t) + length, max_inline, "am_short");


#define UCT_RC_CHECK_AM_ZCOPY(_id, _header_length, _length, _desc_size, _seg_size) \
    UCT_CHECK_AM_ID(_id); \
    UCT_CHECK_LENGTH(sizeof(uct_rc_hdr_t) + _header_length, _desc_size, "am_zcopy header"); \
    UCT_CHECK_LENGTH(_header_length + _length, _seg_size, "am_zcopy payload");


#define UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    UCT_TL_IFACE_GET_TX_DESC(&(_iface)->super.super, _mp, _desc, \
                             return UCS_ERR_NO_RESOURCE);

#define UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(_iface, _mp, _desc, _id, _pack_cb, _arg, _length) \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    uct_rc_bcopy_desc_fill(_desc, _id, _pack_cb, _arg, _length);

#define UCT_RC_IFACE_GET_TX_AM_ZCOPY_DESC(_iface, _mp, _desc, \
                                          _id, _header, _header_length, _comp, _send_flags) \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc); \
    uct_rc_zcopy_desc_set_comp(_desc, _comp, _send_flags); \
    uct_rc_zcopy_desc_set_header(_desc, _id, _header, _header_length);

#define UCT_RC_IFACE_GET_TX_PUT_BCOPY_DESC(_iface, _mp, _desc, _pack_cb, _arg, _length) \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; \
    _length = pack_cb(_desc + 1, _arg); \
    UCT_SKIP_ZERO_LENGTH(_length, _desc);

#define UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(_iface, _mp, _desc, _unpack_cb, _comp, _arg, _length) \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    ucs_assert(_length <= (_iface)->super.config.seg_size); \
    _desc->super.handler     = (_comp == NULL) ? \
                                uct_rc_ep_get_bcopy_handler_no_completion : \
                                uct_rc_ep_get_bcopy_handler; \
    _desc->super.unpack_arg  = _arg; \
    _desc->super.user_comp   = _comp; \
    _desc->super.length      = _length; \
    _desc->unpack_cb         = _unpack_cb;


#define UCT_RC_IFACE_GET_TX_ATOMIC_ADD_DESC(_iface, _mp, _desc) \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    _desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; 

#define UCT_RC_IFACE_GET_TX_ATOMIC_DESC(_iface, _mp, _desc, _handler, _result, _comp) \
    UCT_CHECK_PARAM(_comp != NULL, "completion must be non-NULL"); \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    _desc->super.handler   = _handler; \
    _desc->super.buffer    = _result; \
    _desc->super.user_comp = _comp; 


enum {
    UCT_RC_IFACE_STAT_RX_COMPLETION,
    UCT_RC_IFACE_STAT_TX_COMPLETION,
    UCT_RC_IFACE_STAT_NO_CQE,
    UCT_RC_IFACE_STAT_LAST
};


struct uct_rc_iface_config {
    uct_ib_iface_config_t    super;
    uct_ib_mtu_t             path_mtu;
    unsigned                 max_rd_atomic;

    struct {
        double               timeout;
        unsigned             retry_count;
        double               rnr_timeout;
        unsigned             rnr_retry_count;
        unsigned             cq_len;
    } tx;

    struct {
        double               soft_thresh;
        double               hard_thresh;
        unsigned             wnd_size;
    } fc;
};


typedef struct uct_rc_iface_ops {
    uct_ib_iface_ops_t   super;
    ucs_status_t         (*fc_ctrl)(uct_rc_ep_t *ep);
} uct_rc_iface_ops_t;


struct uct_rc_iface {
    uct_ib_iface_t           super;

    struct {
        ucs_mpool_t          mp;
        uct_rc_iface_send_op_t *ops;
        unsigned             cq_available;
        unsigned             next_op;
        ucs_arbiter_t        arbiter;
    } tx;

    struct {
        ucs_mpool_t          mp;
        struct ibv_srq       *srq;
        unsigned             available;
    } rx;

    struct {
        unsigned             tx_qp_len;
        unsigned             tx_min_sge;
        unsigned             tx_min_inline;
        unsigned             tx_ops_mask;
        unsigned             rx_inline;
        uint16_t             tx_moderation;

        /* Threshold to send "soft" FC credit request. The peer will try to
         * piggy-back credits grant to the counter AM, if any. */
        uint16_t             fc_soft_thresh;

        /* Threshold to sent "hard" credits request. The peer will grant
         * credits in a separate AM as soon as it handles this request. */
        uint16_t             fc_hard_thresh;

        uint16_t             fc_wnd_size;
        uint8_t              min_rnr_timer;
        uint8_t              timeout;
        uint8_t              rnr_retry;
        uint8_t              retry_cnt;
        uint8_t              max_rd_atomic;
        enum ibv_mtu         path_mtu;
    } config;

    UCS_STATS_NODE_DECLARE(stats);

    uct_rc_ep_t              **eps[UCT_RC_QP_TABLE_SIZE];
    ucs_list_link_t          ep_list;
};
UCS_CLASS_DECLARE(uct_rc_iface_t, uct_rc_iface_ops_t*, uct_pd_h, uct_worker_h,
                  const char*, unsigned, unsigned, uct_rc_iface_config_t*)


typedef void (*uct_rc_send_handler_t)(uct_rc_iface_send_op_t *op /*, void *inline_data */);


struct uct_rc_iface_send_op {
    ucs_queue_elem_t              queue;
    uct_rc_send_handler_t         handler;
    uint16_t                      sn;
    unsigned                      length;
    union {
        void                      *buffer;
        void                      *unpack_arg;
    };
    uct_completion_t              *user_comp;
};


struct uct_rc_iface_send_desc {
    uct_rc_iface_send_op_t        super;
    uct_unpack_callback_t         unpack_cb;
    uint32_t                      lkey;
};


/**
 * RC network header.
 */
typedef struct uct_rc_hdr {
    uint8_t           am_id;  /* Active message ID */
} UCS_S_PACKED uct_rc_hdr_t;


/*
 * Short active message header (active message header is always 64 bit).
 */
typedef struct uct_rc_am_short_hdr {
    uct_rc_hdr_t      rc_hdr;
    uint64_t          am_hdr;
} UCS_S_PACKED uct_rc_am_short_hdr_t;


extern ucs_config_field_t uct_rc_iface_config_table[];

void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr);

void uct_rc_iface_add_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep);
void uct_rc_iface_remove_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep);

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface);
void uct_rc_iface_send_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh);

/**
 * Creates an RC or DCI QP and fills 'cap' with QP capabilities;
 */
ucs_status_t uct_rc_iface_qp_create(uct_rc_iface_t *iface, int qp_type, struct ibv_qp **qp_p,
                                    struct ibv_qp_cap *cap);

ucs_status_t uct_rc_iface_handle_fc(uct_rc_iface_t *iface, unsigned qp_num,
                                    uct_rc_hdr_t *hdr, unsigned length,
                                    void *desc);

static inline uct_rc_ep_t *uct_rc_iface_lookup_ep(uct_rc_iface_t *iface,
                                                  unsigned qp_num)
{
    ucs_assert(qp_num < UCS_BIT(UCT_IB_QPN_ORDER));
    return iface->eps[qp_num >> UCT_RC_QP_TABLE_ORDER]
                     [qp_num &  UCS_MASK(UCT_RC_QP_TABLE_MEMB_ORDER)];
}


static UCS_F_ALWAYS_INLINE int
uct_rc_iface_have_tx_cqe_avail(uct_rc_iface_t* iface)
{
    return iface->tx.cq_available > 0;
}

static UCS_F_ALWAYS_INLINE uct_rc_iface_send_op_t*
uct_rc_iface_get_send_op(uct_rc_iface_t *iface)
{
    return &iface->tx.ops[(iface->tx.next_op++) & iface->config.tx_ops_mask];
}


static inline void 
uct_rc_bcopy_desc_fill(uct_rc_iface_send_desc_t *desc, uint8_t id, 
                       uct_pack_callback_t pack_cb, void *arg, size_t *length)
{
    uct_rc_hdr_t *rch;

    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; 
    rch = (uct_rc_hdr_t *)(desc + 1);
    rch->am_id = id;
    *length = pack_cb(rch + 1, arg);
}

void uct_rc_am_zcopy_handler(uct_rc_iface_send_op_t *op);

static inline void uct_rc_zcopy_desc_set_comp(uct_rc_iface_send_desc_t *desc,
                                                    uct_completion_t *comp,
                                                    int *send_flags)
{
    if (comp == NULL) {
        desc->super.handler   = (uct_rc_send_handler_t)ucs_mpool_put;
        *send_flags           = 0;
    } else {
        desc->super.handler   = uct_rc_am_zcopy_handler;
        desc->super.user_comp = comp;
        *send_flags           = IBV_SEND_SIGNALED;
    }
}

static inline void uct_rc_zcopy_desc_set_header(uct_rc_iface_send_desc_t *desc,
                                                uint8_t id, const void *header,
                                                unsigned header_length)
{
     uct_rc_hdr_t *rch;

    /* Header buffer: active message ID + user header */
    rch = (uct_rc_hdr_t *)(desc + 1);
    rch->am_id = id;
    memcpy(rch + 1, header, header_length);
}

#endif
