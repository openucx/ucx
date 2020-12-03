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
#define UCT_RC_QP_MAX_RETRY_COUNT   7

#define UCT_RC_CHECK_AM_SHORT(_am_id, _length, _header_t, _max_inline) \
     UCT_CHECK_AM_ID(_am_id); \
     UCT_CHECK_LENGTH(sizeof(_header_t) + _length, 0, _max_inline, "am_short");

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

#define UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(_iface, _mp, _desc, _id, _pk_hdr_cb, \
                                          _hdr, _pack_cb, _arg, _length) ({ \
    _hdr *rch; \
    UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    (_desc)->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; \
    rch = (_hdr *)(_desc + 1); \
    _pk_hdr_cb(rch, _id); \
    *(_length) = _pack_cb(rch + 1, _arg); \
})

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
    UCT_RC_IFACE_STAT_RX_COMPLETION,
    UCT_RC_IFACE_STAT_TX_COMPLETION,
    UCT_RC_IFACE_STAT_NO_CQE,
    UCT_RC_IFACE_STAT_NO_READS,
    UCT_RC_IFACE_STAT_LAST
};


/* flags for uct_rc_iface_send_op_t */
enum {
    UCT_RC_IFACE_SEND_OP_FLAG_FC_GRANT = UCS_BIT(11),
#ifdef NVALGRIND
    UCT_RC_IFACE_SEND_OP_FLAG_IOV      = 0,
#else
    UCT_RC_IFACE_SEND_OP_FLAG_IOV      = UCS_BIT(12), /* save iovec to make mem defined */
#endif
#if UCS_ENABLE_ASSERT
    UCT_RC_IFACE_SEND_OP_FLAG_ZCOPY    = UCS_BIT(13), /* zcopy */
    UCT_RC_IFACE_SEND_OP_FLAG_IFACE    = UCS_BIT(14), /* belongs to iface ops buffer */
    UCT_RC_IFACE_SEND_OP_FLAG_INUSE    = UCS_BIT(15)  /* queued on a txqp */
#else
    UCT_RC_IFACE_SEND_OP_FLAG_ZCOPY    = 0,
    UCT_RC_IFACE_SEND_OP_FLAG_IFACE    = 0,
    UCT_RC_IFACE_SEND_OP_FLAG_INUSE    = 0
#endif
};


typedef void (*uct_rc_send_handler_t)(uct_rc_iface_send_op_t *op, const void *resp);


/**
 * RC network header.
 */
typedef struct uct_rc_hdr {
    uint8_t           am_id;     /* Active message ID */
} UCS_S_PACKED uct_rc_hdr_t;


typedef struct uct_rc_pending_req {
    uct_pending_req_t super;
    uct_ep_t          *ep;
} uct_rc_pending_req_t;


/**
 * RC fence type.
 */
typedef enum uct_rc_fence_mode {
    UCT_RC_FENCE_MODE_NONE,
    UCT_RC_FENCE_MODE_WEAK,
    UCT_RC_FENCE_MODE_AUTO,
    UCT_RC_FENCE_MODE_LAST
} uct_rc_fence_mode_t;


/* Common configuration used for rc verbs, rcx and dc transports */
typedef struct uct_rc_iface_common_config {
    uct_ib_iface_config_t    super;
    unsigned                 max_rd_atomic;
    int                      ooo_rw; /* Enable out-of-order RDMA data placement */
    int                      fence_mode;

    struct {
        double               timeout;
        unsigned             retry_count;
        double               rnr_timeout;
        unsigned             rnr_retry_count;
        size_t               max_get_zcopy;
        size_t               max_get_bytes;
        int                  poll_always;
    } tx;

    struct {
        int                  enable;
        double               hard_thresh;
        unsigned             wnd_size;
    } fc;
} uct_rc_iface_common_config_t;


/* RC specific configuration used for rc verbs and rcx transports only */
struct uct_rc_iface_config {
    uct_rc_iface_common_config_t   super;
    double                         soft_thresh;
    unsigned                       tx_cq_moderation; /* How many TX messages are
                                                        batched to one CQE */
    unsigned                       tx_cq_len;
};


typedef struct uct_rc_iface_ops {
    uct_ib_iface_ops_t   super;
    ucs_status_t         (*init_rx)(uct_rc_iface_t *iface,
                                    const uct_rc_iface_common_config_t *config);
    void                 (*cleanup_rx)(uct_rc_iface_t *iface);
    ucs_status_t         (*fc_ctrl)(uct_ep_t *ep, unsigned op,
                                    uct_rc_pending_req_t *req);
    ucs_status_t         (*fc_handler)(uct_rc_iface_t *iface, unsigned qp_num,
                                       uct_rc_hdr_t *hdr, unsigned length,
                                       uint32_t imm_data, uint16_t lid,
                                       unsigned flags);
    void                 (*cleanup_qp)(uct_ib_async_event_wait_t *cleanup_ctx);
    void                 (*ep_post_check)(uct_ep_h tl_ep);
} uct_rc_iface_ops_t;


typedef struct uct_rc_srq {
    unsigned                 available;
    unsigned                 quota;
} uct_rc_srq_t;


struct uct_rc_iface {
    uct_ib_iface_t              super;

    struct {
        ucs_mpool_t             mp;         /* pool for send descriptors */
        ucs_mpool_t             pending_mp; /* pool for FC grant and keepalive
                                               pending requests */
        ucs_mpool_t             send_op_mp; /* pool for send_op completions */
        /* Credits for completions.
         * May be negative in case mlx5 because we take "num_bb" credits per
         * post to be able to calculate credits of outstanding ops on failure.
         * In case of verbs TL we use QWE number, so 1 post always takes 1
         * credit */
        signed                  cq_available;
        ssize_t                 reads_available;
        ssize_t                 reads_completed;
        uct_rc_iface_send_op_t  *free_ops; /* stack of free send operations */
        ucs_arbiter_t           arbiter;
        uct_rc_iface_send_op_t  *ops_buffer;
        uct_ib_fence_info_t     fi;
#if UCS_ENABLE_ASSERT
        int                     in_pending;
#endif
    } tx;

    struct {
        ucs_mpool_t          mp;
        uct_rc_srq_t         srq;
    } rx;

    struct {
        unsigned             tx_qp_len;
        unsigned             tx_min_sge;
        unsigned             tx_min_inline;
        unsigned             tx_ops_count;
        uint16_t             tx_moderation;
        uint8_t              tx_poll_always;

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
        /* Enable out-of-order RDMA data placement */
        uint8_t              ooo_rw;
#if UCS_ENABLE_ASSERT
        int                  tx_cq_len;
#endif
        uct_rc_fence_mode_t  fence_mode;
        unsigned             exp_backoff;
        size_t               max_get_zcopy;

        /* Atomic callbacks */
        uct_rc_send_handler_t  atomic64_handler;      /* 64bit ib-spec */
        uct_rc_send_handler_t  atomic32_ext_handler;  /* 32bit extended */
        uct_rc_send_handler_t  atomic64_ext_handler;  /* 64bit extended */
    } config;

    UCS_STATS_NODE_DECLARE(stats)

    ucs_spinlock_t           eps_lock; /* common lock for eps and ep_list */
    uct_rc_ep_t              **eps[UCT_RC_QP_TABLE_SIZE];
    ucs_list_link_t          ep_list;
    ucs_list_link_t          ep_gc_list;

    /* Progress function (either regular or TM aware) */
    ucs_callback_t           progress;
};
UCS_CLASS_DECLARE(uct_rc_iface_t, uct_rc_iface_ops_t*, uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, const uct_rc_iface_common_config_t*,
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
        void                      *buffer;     /* atomics / desc */
        void                      *unpack_arg; /* get_bcopy / desc */
        uct_rc_iface_t            *iface;      /* should not be used with
                                                  get_bcopy completions */
        uct_ep_h                  ep;          /* ep on which we sent ep_check */
    };
    uct_completion_t              *user_comp;
#ifndef NVALGRIND
    struct iovec                  *iov;        /* get_zcopy with valgrind */
#endif
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


extern ucs_config_field_t uct_rc_iface_config_table[];
extern ucs_config_field_t uct_rc_iface_common_config_table[];

unsigned uct_rc_iface_do_progress(uct_iface_h tl_iface);

ucs_status_t uct_rc_iface_query(uct_rc_iface_t *iface,
                                uct_iface_attr_t *iface_attr,
                                size_t put_max_short, size_t max_inline,
                                size_t am_max_hdr, size_t am_max_iov,
                                size_t am_min_hdr, size_t rma_max_iov);

void uct_rc_iface_add_qp(uct_rc_iface_t *iface, uct_rc_ep_t *ep,
                         unsigned qp_num);

void uct_rc_iface_remove_qp(uct_rc_iface_t *iface, unsigned qp_num);

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                uct_completion_t *comp);

void uct_rc_iface_send_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh);

void uct_rc_ep_am_zcopy_handler(uct_rc_iface_send_op_t *op, const void *resp);

void uct_rc_iface_cleanup_eps(uct_rc_iface_t *iface);

/**
 * Creates an RC or DCI QP
 */
ucs_status_t uct_rc_iface_qp_create(uct_rc_iface_t *iface, struct ibv_qp **qp_p,
                                    uct_ib_qp_attr_t *attr, unsigned max_send_wr,
                                    struct ibv_srq *srq);

void uct_rc_iface_fill_attr(uct_rc_iface_t *iface,
                            uct_ib_qp_attr_t *qp_init_attr,
                            unsigned max_send_wr,
                            struct ibv_srq *srq);

ucs_status_t uct_rc_iface_qp_init(uct_rc_iface_t *iface, struct ibv_qp *qp);

ucs_status_t uct_rc_iface_qp_connect(uct_rc_iface_t *iface, struct ibv_qp *qp,
                                     const uint32_t qp_num,
                                     struct ibv_ah_attr *ah_attr,
                                     enum ibv_mtu path_mtu);

ucs_status_t uct_rc_iface_fc_handler(uct_rc_iface_t *iface, unsigned qp_num,
                                     uct_rc_hdr_t *hdr, unsigned length,
                                     uint32_t imm_data, uint16_t lid, unsigned flags);

ucs_status_t uct_rc_init_fc_thresh(uct_rc_iface_config_t *rc_cfg,
                                   uct_rc_iface_t *iface);

ucs_status_t uct_rc_iface_event_arm(uct_iface_h tl_iface, unsigned events);

ucs_status_t uct_rc_iface_common_event_arm(uct_iface_h tl_iface,
                                           unsigned events, int force_rx_all);

ucs_status_t uct_rc_iface_init_rx(uct_rc_iface_t *iface,
                                  const uct_rc_iface_common_config_t *config,
                                  struct ibv_srq **p_srq);

ucs_status_t uct_rc_iface_fence(uct_iface_h tl_iface, unsigned flags);

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_fc_ctrl(uct_ep_t *ep, unsigned op, uct_rc_pending_req_t *req)
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

/**
 * Release RDMA_READ credits back to RC iface.
 * RDMA_READ credits are freed in completion callbacks, but not released to
 * RC iface to avoid OOO sends. Otherwise, if read credit is the only missing
 * resource and is released in completion callback, next completion callback
 * will be able to send even if pedning queue is not empty.
 */
static UCS_F_ALWAYS_INLINE void
uct_rc_iface_update_reads(uct_rc_iface_t *iface)
{
    ucs_assert(iface->tx.reads_completed >= 0);

    iface->tx.reads_available += iface->tx.reads_completed;
    iface->tx.reads_completed  = 0;
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
    rch->am_id = id;
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
           !ucs_mpool_is_empty(&iface->tx.mp) &&
           (iface->tx.reads_available > 0);
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

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_iface_fence_relaxed_order(uct_iface_h tl_iface)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    uct_ib_md_t *md         = ucs_derived_of(iface->md, uct_ib_md_t);

    ucs_assert(tl_iface->ops.iface_fence == uct_rc_iface_fence);

    if (!md->relaxed_order) {
        return UCS_OK;
    }

    return uct_rc_iface_fence(tl_iface, 0);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_iface_check_pending(uct_rc_iface_t *iface, ucs_arbiter_group_t *arb_group)
{
    ucs_assert(iface->tx.in_pending || ucs_arbiter_group_is_empty(arb_group));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_iface_invoke_pending_cb(uct_rc_iface_t *iface, uct_pending_req_t *req)
{
    ucs_status_t status;

    ucs_trace_data("progressing pending request %p", req);
#if UCS_ENABLE_ASSERT
    iface->tx.in_pending = 1;
#endif

    status = req->func(req);

#if UCS_ENABLE_ASSERT
    iface->tx.in_pending = 0;
#endif
    ucs_trace_data("status returned from progress pending: %s",
                   ucs_status_string(status));

    return status;
}

static UCS_F_ALWAYS_INLINE int
uct_rc_iface_poll_tx(uct_rc_iface_t *iface, unsigned count)
{
    return (count == 0) || iface->config.tx_poll_always;
}

#endif
