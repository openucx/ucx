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
#include <ucs/datastruct/ptr_array.h>
#include <ucs/debug/log.h>


#define UCT_RC_QP_TABLE_ORDER       12
#define UCT_RC_QP_TABLE_SIZE        UCS_BIT(UCT_RC_QP_TABLE_ORDER)
#define UCT_RC_QP_TABLE_MEMB_ORDER  (UCT_IB_QPN_ORDER - UCT_RC_QP_TABLE_ORDER)
#define UCT_RC_MAX_ATOMIC_SIZE      sizeof(uint64_t)
#define UCR_RC_QP_MAX_RETRY_COUNT   7

#define UCT_RC_CHECK_AM_SHORT(_am_id, _length, _max_inline) \
     UCT_CHECK_AM_ID(_am_id); \
     UCT_CHECK_LENGTH(sizeof(uct_rc_am_short_hdr_t) + _length, 0, _max_inline, "am_short");

#define UCT_RC_CHECK_ZCOPY_DATA(_header_length, _length, _seg_size) \
    UCT_CHECK_LENGTH(_header_length + _length, 0, _seg_size, "am_zcopy payload"); \
    UCT_CHECK_LENGTH(_header_length + _length, 0, UCT_IB_MAX_MESSAGE_SIZE, "am_zcopy ib max message");

#define UCT_RC_CHECK_AM_ZCOPY(_id, _header_length, _length, _desc_size, _seg_size) \
    UCT_CHECK_AM_ID(_id); \
    UCT_RC_CHECK_ZCOPY_DATA(_header_length, _length, _seg_size) \
    UCT_CHECK_LENGTH(sizeof(uct_rc_hdr_t) + _header_length, 0, _desc_size, "am_zcopy header");


#define UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    UCT_TL_IFACE_GET_TX_DESC(&(_iface)->super.super, _mp, _desc, \
                             return UCS_ERR_NO_RESOURCE);

#define UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(_iface, _mp, _desc, _id, _pack_cb, _arg, _length) \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    (_desc)->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; \
    uct_rc_bcopy_desc_fill((uct_rc_hdr_t*)(_desc + 1), _id, _pack_cb, _arg, _length);

#define UCT_RC_IFACE_GET_TX_AM_ZCOPY_DESC(_iface, _mp, _desc, \
                                          _id, _header, _header_length, _comp, _send_flags) \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc); \
    uct_rc_zcopy_desc_set_comp(_desc, _comp, _send_flags); \
    uct_rc_zcopy_desc_set_header((uct_rc_hdr_t*)(_desc + 1), _id, _header, _header_length);

#define UCT_RC_IFACE_GET_TX_PUT_BCOPY_DESC(_iface, _mp, _desc, _pack_cb, _arg, _length) \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    (_desc)->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; \
    _length = _pack_cb(_desc + 1, _arg); \
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


#define UCT_RC_IFACE_GET_TX_ATOMIC_DESC(_iface, _mp, _desc) \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    _desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;

#define UCT_RC_IFACE_GET_TX_ATOMIC_FETCH_DESC(_iface, _mp, _desc, _handler, _result, _comp) \
    UCT_CHECK_PARAM(_comp != NULL, "completion must be non-NULL"); \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    _desc->super.handler   = _handler; \
    _desc->super.buffer    = _result; \
    _desc->super.user_comp = _comp;


enum {
    UCT_RC_IFACE_ADDR_TYPE_BASIC,

    /* Tag Matching address. It additionaly contains QP number which
     * is used for hardware offloads. */
    UCT_RC_IFACE_ADDR_TYPE_TM,
    UCT_RC_IFACE_ADDR_TYPE_LAST
};


enum {
    UCT_RC_IFACE_STAT_RX_COMPLETION,
    UCT_RC_IFACE_STAT_TX_COMPLETION,
    UCT_RC_IFACE_STAT_NO_CQE,
    UCT_RC_IFACE_STAT_LAST
};


/* flags for uct_rc_iface_send_op_t */
enum {
#if ENABLE_ASSERT
    UCT_RC_IFACE_SEND_OP_FLAG_IFACE = UCS_BIT(14), /* belongs to iface ops buffer */
    UCT_RC_IFACE_SEND_OP_FLAG_INUSE = UCS_BIT(15)  /* queued on a txqp */
#else
    UCT_RC_IFACE_SEND_OP_FLAG_IFACE = 0,
    UCT_RC_IFACE_SEND_OP_FLAG_INUSE = 0
#endif
};


typedef void (*uct_rc_send_handler_t)(uct_rc_iface_send_op_t *op, const void *resp);


/**
 * RC network header.
 */
typedef struct uct_rc_hdr {
#if IBV_EXP_HW_TM
    uint8_t           tmh_opcode; /* reserved for TMH.opcode */
#endif
    uint8_t           am_id;     /* Active message ID */
} UCS_S_PACKED uct_rc_hdr_t;


typedef struct uct_rc_fc_request {
    uct_pending_req_t super;
    uct_ep_t          *ep;
} uct_rc_fc_request_t;


typedef struct uct_rc_fc_config {
    double            soft_thresh;
} uct_rc_fc_config_t;


struct uct_rc_iface_config {
    uct_ib_iface_config_t    super;
    uct_ib_mtu_t             path_mtu;
    unsigned                 max_rd_atomic;
    int                      ooo_rw; /* Enable out-of-order RDMA data placement */

    struct {
        double               timeout;
        unsigned             retry_count;
        double               rnr_timeout;
        unsigned             rnr_retry_count;
        unsigned             cq_len;
    } tx;

    struct {
        int                  enable;
        double               hard_thresh;
        unsigned             wnd_size;
    } fc;

#if IBV_EXP_HW_TM
    struct {
        int                  enable;
        unsigned             list_size;
        size_t               max_bcopy;
    } tm;
#endif

};


typedef struct uct_rc_iface_ops {
    uct_ib_iface_ops_t   super;
    ucs_status_t         (*fc_ctrl)(uct_ep_t *ep, unsigned op,
                                    uct_rc_fc_request_t *req);
    ucs_status_t         (*fc_handler)(uct_rc_iface_t *iface, unsigned qp_num,
                                       uct_rc_hdr_t *hdr, unsigned length,
                                       uint32_t imm_data, uint16_t lid,
                                       unsigned flags);
} uct_rc_iface_ops_t;


typedef struct uct_rc_srq {
    struct ibv_srq           *srq;
    unsigned                 available;
    unsigned                 quota;
} uct_rc_srq_t;


#if IBV_EXP_HW_TM

typedef struct uct_rc_iface_tmh_priv_data {
    uint8_t                     length;
    uint16_t                    data;
} UCS_S_PACKED uct_rc_iface_tmh_priv_data_t;


typedef struct uct_rc_iface_release_desc {
    uct_recv_desc_t             super;
    unsigned                    offset;
} uct_rc_iface_release_desc_t;


typedef struct uct_rc_iface_ctx_priv {
    uint64_t                    tag;
    void                        *buffer;
    uint32_t                    app_ctx;
    uint32_t                    length;
    uint32_t                    tag_handle;
} uct_rc_iface_ctx_priv_t;

#endif


struct uct_rc_iface {
    uct_ib_iface_t              super;

    struct {
        ucs_mpool_t             mp;      /* pool for send descriptors */
        ucs_mpool_t             fc_mp;   /* pool for FC grant pending requests */
        /* Credits for completions.
         * May be negative in case mlx5 because we take "num_bb" credits per
         * post to be able to calculate credits of outstanding ops on failure.
         * In case of verbs TL we use QWE number, so 1 post always takes 1
         * credit */
        signed                  cq_available;
        uct_rc_iface_send_op_t  *free_ops; /* stack of free send operations */
        ucs_arbiter_t           arbiter;
        uct_rc_iface_send_op_t  *ops_buffer;
    } tx;

    struct {
        ucs_mpool_t          mp;
        uct_rc_srq_t         srq;
    } rx;

#if IBV_EXP_HW_TM
    struct {
        ucs_ptr_array_t              rndv_comps;
        unsigned                     num_tags;
        unsigned                     num_outstanding;
        unsigned                     max_rndv_data;
        uint16_t                     unexpected_cnt;
        uint16_t                     cmd_qp_len;
        uint8_t                      enabled;
        struct {
            void                     *arg; /* User defined arg */
            uct_tag_unexp_eager_cb_t cb;   /* Callback for unexpected eager messages */
        } eager_unexp;

        struct {
            void                     *arg; /* User defined arg */
            uct_tag_unexp_rndv_cb_t  cb;   /* Callback for unexpected rndv messages */
        } rndv_unexp;
        uct_rc_iface_release_desc_t  eager_desc;
        uct_rc_iface_release_desc_t  rndv_desc;

    } tm;
#endif

    struct {
        unsigned             tx_qp_len;
        unsigned             tx_min_sge;
        unsigned             tx_min_inline;
        unsigned             tx_ops_count;
        unsigned             rx_inline;
        uint16_t             tx_moderation;

        /* Threshold to send "soft" FC credit request. The peer will try to
         * piggy-back credits grant to the counter AM, if any. */
        int16_t              fc_soft_thresh;

        /* Threshold to sent "hard" credits request. The peer will grant
         * credits in a separate AM as soon as it handles this request. */
        int16_t              fc_hard_thresh;

        uint16_t             fc_wnd_size;
        uint8_t              fc_enabled;

        uint8_t              min_rnr_timer;
        uint8_t              timeout;
        uint8_t              rnr_retry;
        uint8_t              retry_cnt;
        uint8_t              max_rd_atomic;
        enum ibv_mtu         path_mtu;
        /* Enable out-of-order RDMA data placement */
        uint8_t              ooo_rw;
#if ENABLE_ASSERT
        int                  tx_cq_len;
#endif

        /* Atomic callbacks */
        uct_rc_send_handler_t  atomic64_handler;      /* 64bit ib-spec */
        uct_rc_send_handler_t  atomic32_ext_handler;  /* 32bit extended */
        uct_rc_send_handler_t  atomic64_ext_handler;  /* 64bit extended */
    } config;

    UCS_STATS_NODE_DECLARE(stats);

    uct_rc_ep_t              **eps[UCT_RC_QP_TABLE_SIZE];
    ucs_list_link_t          ep_list;

    /* Progress function (either regular or TM aware) */
    ucs_callback_t           progress;
};
UCS_CLASS_DECLARE(uct_rc_iface_t, uct_rc_iface_ops_t*, uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, const uct_rc_iface_config_t*,
                  uct_ib_iface_init_attr_t*);


struct uct_rc_iface_send_op {
    union {
        ucs_queue_elem_t          queue;  /* used when enqueued on a txqp */
        uct_rc_iface_send_op_t    *next;  /* used when on free list */
    };
    uct_rc_send_handler_t         handler;
    uint16_t                      sn;
    uint16_t                      flags;
    unsigned                      length;
    union {
        void                      *buffer;        /* atomics / desc */
        void                      *unpack_arg;    /* get_bcopy / desc */
        uct_rc_iface_t            *iface;         /* zcopy / op */
    };
    uct_completion_t              *user_comp;
};


struct uct_rc_iface_send_desc {
    uct_rc_iface_send_op_t        super;
    uct_unpack_callback_t         unpack_cb;
    uint32_t                      lkey;
};


/*
 * Short active message header (active message header is always 64 bit).
 */
typedef struct uct_rc_am_short_hdr {
    uct_rc_hdr_t      rc_hdr;
    uint64_t          am_hdr;
} UCS_S_PACKED uct_rc_am_short_hdr_t;


#if IBV_EXP_HW_TM

#  define UCT_RC_IFACE_TM_ENABLED(_iface) (_iface)->tm.enabled

/* TMH can carry 2 bytes of data in its reserved filed */
#  define UCT_RC_IFACE_TMH_PRIV_LEN       ucs_field_sizeof(uct_rc_iface_tmh_priv_data_t, \
                                                           data)

#  define UCT_RC_IFACE_CHECK_RES_PTR(_iface, _ep) \
       UCT_RC_CHECK_CQE_RET(_iface, _ep, &(_ep)->txqp, \
                            UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE)) \
       UCT_RC_CHECK_TXQP_RET(_iface, _ep, &(_ep)->txqp, \
                             UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE))

#  define UCT_RC_IFACE_CHECK_RNDV_PARAMS(_iovcnt, _header_len, _tm_len, \
                                         _max_inline, _max_rndv_hdr) \
       { \
           UCT_CHECK_PARAM_PTR(_iovcnt <= 1ul, "Wrong iovcnt %lu", iovcnt); \
           UCT_CHECK_PARAM_PTR(_header_len <= _max_rndv_hdr, \
                               "Invalid header len %u", _header_len); \
           UCT_CHECK_PARAM_PTR((_header_len + _tm_len) <= _max_inline, \
                               "Invalid RTS len gth %u", \
                               _header_len + _tm_len); \
       }

#  define UCT_RC_IFACE_FILL_TM_IMM(_imm_data, _app_ctx, _ib_imm, _res_op, \
                                   _op, _imm_suffix) \
       if (_imm_data == 0) { \
           _res_op  = _op; \
           _app_ctx = 0; \
           _ib_imm  = 0; \
       } else { \
           _res_op = UCS_PP_TOKENPASTE(_op, _imm_suffix); \
           uct_rc_iface_tag_imm_data_pack(&(_ib_imm), &(_app_ctx), _imm_data); \
       }

#  define UCT_RC_IFACE_GET_TX_TM_DESC(_iface, _mp, _desc, _tag, _app_ctx, _hdr) \
      { \
          UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
          _hdr = _desc + 1; \
          uct_rc_iface_fill_tmh(_hdr, _tag, _app_ctx, IBV_EXP_TMH_EAGER); \
          _hdr += sizeof(struct ibv_exp_tmh); \
      }

#  define UCT_RC_IFACE_GET_TM_BCOPY_DESC(_iface, _mp, _desc, _tag, _app_ctx, \
                                         _pack_cb, _arg, _length) \
       { \
           void *hdr; \
           UCT_RC_IFACE_GET_TX_TM_DESC(_iface, _mp, _desc, _tag, _app_ctx, hdr) \
           (_desc)->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; \
           _length = _pack_cb(hdr, _arg); \
       }

ucs_status_t uct_rc_iface_handle_rndv(uct_rc_iface_t *iface,
                                      struct ibv_exp_tmh *tmh, uct_tag_t tag,
                                      unsigned byte_len);


static UCS_F_ALWAYS_INLINE void
uct_rc_iface_fill_tmh(struct ibv_exp_tmh *tmh, uct_tag_t tag,
                      uint32_t app_ctx, unsigned op)
{
    tmh->opcode  = op;
    tmh->app_ctx = htonl(app_ctx);
    tmh->tag     = htobe64(tag);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_iface_fill_rvh(struct ibv_exp_tmh_rvh *rvh, const void *vaddr,
                      uint32_t rkey, uint32_t len)
{
    rvh->va   = htobe64((uint64_t)vaddr);
    rvh->rkey = htonl(rkey);
    rvh->len  = htonl(len);
}

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_iface_tag_get_op_id(uct_rc_iface_t *iface, uct_completion_t *comp)
{
    uint32_t prev_ph;
    return ucs_ptr_array_insert(&iface->tm.rndv_comps, comp, &prev_ph);
}


static UCS_F_ALWAYS_INLINE unsigned
uct_rc_iface_fill_tmh_priv_data(struct ibv_exp_tmh *tmh, const void *hdr,
                                unsigned hdr_len, unsigned max_rndv_priv_data)
{
    uct_rc_iface_tmh_priv_data_t *priv = (uct_rc_iface_tmh_priv_data_t*)tmh->reserved;

    /* If header length is bigger tha max_rndv_priv_data size, need to add the
     * rest to the TMH reserved field. */
    if (hdr_len > max_rndv_priv_data) {
        priv->length = hdr_len - max_rndv_priv_data;
        ucs_assert(priv->length <= UCT_RC_IFACE_TMH_PRIV_LEN);
        memcpy(&priv->data, (char*)hdr, priv->length);
    } else {
        priv->length = 0;
    }

    return priv->length;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_iface_tag_imm_data_pack(uint32_t *ib_imm, uint32_t *app_ctx,
                               uint64_t imm_val)
{
    *ib_imm  = (uint32_t)(imm_val & 0xFFFFFFFF);
    *app_ctx = (uint32_t)(imm_val >> 32);
}

static UCS_F_ALWAYS_INLINE uint64_t
uct_rc_iface_tag_imm_data_unpack(uint32_t ib_imm, uint32_t app_ctx, int is_imm)
{
    return is_imm ? (((uint64_t)app_ctx << 32) | ib_imm) : 0ul;
}

static UCS_F_ALWAYS_INLINE uct_rc_iface_ctx_priv_t*
uct_rc_iface_ctx_priv(uct_tag_context_t *ctx)
{
    return (uct_rc_iface_ctx_priv_t*)ctx->priv;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_iface_handle_rndv_fin(uct_rc_iface_t *iface, uint32_t app_ctx)
{
    int found;
    void *rndv_comp;

    found = ucs_ptr_array_lookup(&iface->tm.rndv_comps, app_ctx, rndv_comp);
    ucs_assert_always(found > 0);
    uct_invoke_completion((uct_completion_t*)rndv_comp, UCS_OK);
    ucs_ptr_array_remove(&iface->tm.rndv_comps, app_ctx, 0);
}

#else

#  define UCT_RC_IFACE_TM_ENABLED(_iface) 0

#endif


extern ucs_config_field_t uct_rc_iface_config_table[];
extern ucs_config_field_t uct_rc_fc_config_table[];

unsigned uct_rc_iface_do_progress(uct_iface_h tl_iface);

ucs_status_t uct_rc_iface_query(uct_rc_iface_t *iface,
                                uct_iface_attr_t *iface_attr,
                                size_t put_max_short, size_t max_inline,
                                size_t am_max_hdr, size_t am_max_iov,
                                size_t tag_max_iov);

ucs_status_t uct_rc_iface_get_address(uct_iface_h tl_iface,
                                      uct_iface_addr_t *addr);

int uct_rc_iface_is_reachable(const uct_iface_h tl_iface,
                              const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr);

ucs_status_t uct_rc_iface_tag_init(uct_rc_iface_t *iface,
                                   uct_rc_iface_config_t *config,
                                   struct ibv_exp_create_srq_attr *srq_init_attr,
                                   unsigned rndv_hdr_len,
                                   unsigned max_cancel_sync_ops);

void uct_rc_iface_tag_cleanup(uct_rc_iface_t *iface);

void uct_rc_iface_add_qp(uct_rc_iface_t *iface, uct_rc_ep_t *ep,
                         unsigned qp_num);

void uct_rc_iface_remove_qp(uct_rc_iface_t *iface, unsigned qp_num);

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                uct_completion_t *comp);

void uct_rc_iface_send_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh);

void uct_rc_ep_am_zcopy_handler(uct_rc_iface_send_op_t *op, const void *resp);

/**
 * Creates an RC or DCI QP and fills 'cap' with QP capabilities;
 */
ucs_status_t uct_rc_iface_qp_create(uct_rc_iface_t *iface, int qp_type,
                                    struct ibv_qp **qp_p, struct ibv_qp_cap *cap,
                                    unsigned max_send_wr);

ucs_status_t uct_rc_iface_qp_init(uct_rc_iface_t *iface, struct ibv_qp *qp);

ucs_status_t uct_rc_iface_qp_connect(uct_rc_iface_t *iface, struct ibv_qp *qp,
                                     const uint32_t qp_num,
                                     struct ibv_ah_attr *ah_attr);

ucs_status_t uct_rc_iface_fc_handler(uct_rc_iface_t *iface, unsigned qp_num,
                                     uct_rc_hdr_t *hdr, unsigned length,
                                     uint32_t imm_data, uint16_t lid, unsigned flags);

ucs_status_t uct_rc_init_fc_thresh(uct_rc_fc_config_t *fc_cfg,
                                   uct_rc_iface_config_t *rc_cfg,
                                   uct_rc_iface_t *iface);

ucs_status_t uct_rc_iface_event_arm(uct_iface_h tl_iface, unsigned events);

ucs_status_t uct_rc_iface_common_event_arm(uct_iface_h tl_iface,
                                           unsigned events, int force_rx_all);


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_fc_ctrl(uct_ep_t *ep, unsigned op, uct_rc_fc_request_t *req)
{
    uct_rc_iface_t *iface   = ucs_derived_of(ep->iface, uct_rc_iface_t);
    uct_rc_iface_ops_t *ops = ucs_derived_of(iface->super.ops,
                                             uct_rc_iface_ops_t);
    return ops->fc_ctrl(ep, op, req);
}

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
    uct_rc_iface_send_op_t *op;
    op = iface->tx.free_ops;
    iface->tx.free_ops = op->next;
    return op;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_iface_put_send_op(uct_rc_iface_send_op_t *op)
{
    uct_rc_iface_t *iface = op->iface;
    ucs_assert(op->flags == UCT_RC_IFACE_SEND_OP_FLAG_IFACE);
    op->next = iface->tx.free_ops;
    iface->tx.free_ops = op;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_am_hdr_fill(uct_rc_hdr_t *rch, uint8_t id)
{
#if IBV_EXP_HW_TM
    rch->tmh_opcode = IBV_EXP_TMH_NO_TAG;
#endif
    rch->am_id      = id;
}

static inline void
uct_rc_bcopy_desc_fill(uct_rc_hdr_t *rch, uint8_t id,
                       uct_pack_callback_t pack_cb, void *arg, size_t *length)
{
    uct_rc_am_hdr_fill(rch, id);
    *length = pack_cb(rch + 1, arg);
}

static inline void uct_rc_zcopy_desc_set_comp(uct_rc_iface_send_desc_t *desc,
                                              uct_completion_t *comp,
                                              int *send_flags)
{
    if (comp == NULL) {
        desc->super.handler   = (uct_rc_send_handler_t)ucs_mpool_put;
        *send_flags           = 0;
    } else {
        desc->super.handler   = uct_rc_ep_am_zcopy_handler;
        desc->super.user_comp = comp;
        *send_flags           = IBV_SEND_SIGNALED;
    }
}

static inline void uct_rc_zcopy_desc_set_header(uct_rc_hdr_t *rch,
                                                uint8_t id, const void *header,
                                                unsigned header_length)
{
    uct_rc_am_hdr_fill(rch, id);
    memcpy(rch + 1, header, header_length);
}

static inline int uct_rc_iface_has_tx_resources(uct_rc_iface_t *iface)
{
    return uct_rc_iface_have_tx_cqe_avail(iface) &&
           !ucs_mpool_is_empty(&iface->tx.mp);
}

static UCS_F_ALWAYS_INLINE uct_rc_send_handler_t
uct_rc_iface_atomic_handler(uct_rc_iface_t *iface, int ext, unsigned length)
{
    ucs_assert((length == sizeof(uint32_t)) || (length == sizeof(uint64_t)));
    switch (length) {
    case sizeof(uint32_t):
        return iface->config.atomic32_ext_handler;
    case sizeof(uint64_t):
        return ext ? iface->config.atomic64_ext_handler :
                     iface->config.atomic64_handler;
    }
    return NULL;
}

#endif
