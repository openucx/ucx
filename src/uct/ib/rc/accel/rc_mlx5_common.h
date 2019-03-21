/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_MLX5_COMMON_H
#define UCT_RC_MLX5_COMMON_H

#include <uct/ib/base/ib_device.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <uct/ib/mlx5/ib_mlx5.h>


#define UCT_RC_MLX5_OPCODE_FLAG_RAW   0x100
#define UCT_RC_MLX5_OPCODE_FLAG_TM    0x200
#define UCT_RC_MLX5_OPCODE_MASK       0xff

#define UCT_RC_MLX5_CHECK_AM_ZCOPY(_id, _header_length, _length, _seg_size, _av_size) \
    UCT_CHECK_AM_ID(_id); \
    UCT_RC_CHECK_ZCOPY_DATA(_header_length, _length, _seg_size) \
    UCT_CHECK_LENGTH(sizeof(uct_rc_mlx5_hdr_t) + _header_length, 0, \
                     UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(_av_size), "am_zcopy header");


#define UCT_RC_MLX5_CHECK_AM_SHORT(_id, _length, _av_size) \
    UCT_CHECK_AM_ID(_id); \
    UCT_CHECK_LENGTH(sizeof(uct_rc_mlx5_am_short_hdr_t) + _length, 0, \
        UCT_IB_MLX5_AM_MAX_SHORT(_av_size), "am_short");


/* there is no need to do a special check for length == 0 because in that
 * case wqe size is valid: inl + raddr + dgram + ctrl fit in 2 WQ BB
 */
#define UCT_RC_MLX5_CHECK_PUT_SHORT(_length, _av_size) \
    UCT_CHECK_LENGTH(_length, 0, UCT_IB_MLX5_PUT_MAX_SHORT(_av_size), "put_short")

#define UCT_RC_MLX5_ATOMIC_OPS (UCS_BIT(UCT_ATOMIC_OP_ADD) | \
                                UCS_BIT(UCT_ATOMIC_OP_AND) | \
                                UCS_BIT(UCT_ATOMIC_OP_OR)  | \
                                UCS_BIT(UCT_ATOMIC_OP_XOR))

#define UCT_RC_MLX5_ATOMIC_FOPS (UCT_RC_MLX5_ATOMIC_OPS | UCS_BIT(UCT_ATOMIC_OP_SWAP))

#define UCT_RC_MLX5_CHECK_ATOMIC_OPS(_op, _size, _flags)                        \
    if (ucs_unlikely(!(UCS_BIT(_op) & (_flags)))) {                             \
        ucs_assertv(0, "incorrect opcode for atomic: %d", _op);                 \
        return UCS_ERR_UNSUPPORTED;                                             \
    } else {                                                                    \
        ucs_assert((_size == sizeof(uint64_t)) || (_size == sizeof(uint32_t))); \
    }

#define UCT_RC_MLX5_TO_BE(_val, _size) \
    ((_size) == sizeof(uint64_t) ? htobe64(_val) : htobe32(_val))

#define UCT_RC_MLX5_DECLARE_ATOMIC_LE_HANDLER(_bits) \
    void \
    uct_rc_mlx5_common_atomic##_bits##_le_handler(uct_rc_iface_send_op_t *op, \
                                                  const void *resp);

UCT_RC_MLX5_DECLARE_ATOMIC_LE_HANDLER(32)
UCT_RC_MLX5_DECLARE_ATOMIC_LE_HANDLER(64)

enum {
    UCT_RC_MLX5_IFACE_STAT_RX_INL_32,
    UCT_RC_MLX5_IFACE_STAT_RX_INL_64,
    UCT_RC_MLX5_IFACE_STAT_LAST
};

enum {
    UCT_RC_MLX5_TM_OPCODE_NOP              = 0x00,
    UCT_RC_MLX5_TM_OPCODE_APPEND           = 0x01,
    UCT_RC_MLX5_TM_OPCODE_REMOVE           = 0x02
};

/* TODO: Remove/replace this enum when mlx5dv.h is included */
enum {
    UCT_RC_MLX5_OPCODE_TAG_MATCHING        = 0x28,
    UCT_RC_MLX5_CQE_APP_TAG_MATCHING       = 1,

    /* tag segment flags */
    UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT         = (1 << 6),
    UCT_RC_MLX5_SRQ_FLAG_TM_CQE_REQ        = (1 << 7),

    /* tag CQE codes */
    UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED     = 0x1,
    UCT_RC_MLX5_CQE_APP_OP_TM_EXPECTED     = 0x2,
    UCT_RC_MLX5_CQE_APP_OP_TM_UNEXPECTED   = 0x3,
    UCT_RC_MLX5_CQE_APP_OP_TM_NO_TAG       = 0x4,
    UCT_RC_MLX5_CQE_APP_OP_TM_APPEND       = 0x5,
    UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE       = 0x6,
    UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED_MSG = 0xA
};

#ifdef IBV_HW_TM
#  define UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(_av_size) \
       (UCT_IB_MLX5_AM_MAX_SHORT(_av_size + sizeof(struct ibv_tmh))/ \
        sizeof(struct mlx5_wqe_data_seg))
# else
#  define UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(_av_size)   0
#endif /* IBV_HW_TM  */

#define UCT_RC_MLX5_TM_CQE_WITH_IMM(_cqe64) \
   (((_cqe64)->op_own >> 4) == MLX5_CQE_RESP_SEND_IMM)

#define UCT_RC_MLX5_TM_IS_SW_RNDV(_cqe64, _imm_data) \
   (ucs_unlikely(UCT_RC_MLX5_TM_CQE_WITH_IMM(_cqe64) && !(_imm_data)))

#define UCT_RC_MLX5_CHECK_TAG(_mlx5_common_iface) \
   if (ucs_unlikely((_mlx5_common_iface)->tm.head->next == NULL)) {  \
       return UCS_ERR_EXCEEDS_LIMIT; \
   }


typedef struct uct_rc_mlx5_hdr {
    uint8_t           tmh_opcode; /* TMH.opcode */
    uct_rc_hdr_t      rc_hdr;
} UCS_S_PACKED uct_rc_mlx5_hdr_t;

/*
 * Short active message header (active message header is always 64 bit).
 */
typedef struct uct_rc_mlx5_am_short_hdr {
    uct_rc_mlx5_hdr_t  rc_hdr;
    uint64_t           am_hdr;
} UCS_S_PACKED uct_rc_mlx5_am_short_hdr_t;


/* TODO: Remove this struct when mlx5dv.h is included! */
typedef struct uct_rc_mlx5_wqe_tm_seg {
    uint8_t                       opcode;
    uint8_t                       flags;
    uint16_t                      index;
    uint8_t                       rsvd0[2];
    uint16_t                      sw_cnt;
    uint8_t                       rsvd1[8];
    uint64_t                      append_tag;
    uint64_t                      append_mask;
} uct_rc_mlx5_wqe_tm_seg_t;


/* Tag matching list entry */
typedef struct uct_rc_mlx5_tag_entry {
    struct uct_rc_mlx5_tag_entry  *next;
    uct_tag_context_t             *ctx;     /* the corresponding UCT context */
    unsigned                      num_cqes; /* how many CQEs is expected for this entry */
} uct_rc_mlx5_tag_entry_t;


/* Pending operation on the command QP */
typedef struct uct_rc_mlx5_srq_op {
    uct_rc_mlx5_tag_entry_t       *tag;
} uct_rc_mlx5_srq_op_t;


/* Command QP work-queue. All tag matching list operations are posted on it. */
typedef struct uct_rc_mlx5_cmd_wq {
    uct_ib_mlx5_txwq_t            super;
    uint32_t                      qp_num;   /* command QP num */
    uct_rc_mlx5_srq_op_t          *ops;     /* array of operations on command QP */
    int                           ops_head; /* points to the next operation to be completed */
    int                           ops_tail; /* points to the last adde operation*/
    int                           ops_mask; /* mask which bounds head and tail by
                                               ops array size */
} uct_rc_mlx5_cmd_wq_t;

#ifdef IBV_HW_TM
#  define UCT_RC_MLX5_IFACE_GET_TM_BCOPY_DESC(_iface, _mp, _desc, _tag, _app_ctx, \
                                              _pack_cb, _arg, _length) \
       { \
           void *hdr; \
           UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
           (_desc)->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; \
           hdr = (_desc) + 1; \
           uct_rc_mlx5_fill_tmh(hdr, _tag, _app_ctx, IBV_TMH_EAGER); \
           hdr = (char*)hdr + sizeof(struct ibv_tmh); \
           _length = _pack_cb(hdr, _arg); \
       }
#endif

enum {
    UCT_RC_MLX5_STAT_TAG_RX_EXP,
    UCT_RC_MLX5_STAT_TAG_RX_EAGER_UNEXP,
    UCT_RC_MLX5_STAT_TAG_RX_RNDV_UNEXP,
    UCT_RC_MLX5_STAT_TAG_RX_RNDV_REQ_EXP,
    UCT_RC_MLX5_STAT_TAG_RX_RNDV_REQ_UNEXP,
    UCT_RC_MLX5_STAT_TAG_RX_RNDV_FIN,
    UCT_RC_MLX5_STAT_TAG_LIST_ADD,
    UCT_RC_MLX5_STAT_TAG_LIST_DEL,
    UCT_RC_MLX5_STAT_TAG_LIST_SYNC,
    UCT_RC_MLX5_STAT_TAG_LAST
};

typedef struct uct_rc_mlx5_tmh_priv_data {
    uint8_t                     length;
    uint16_t                    data;
} UCS_S_PACKED uct_rc_mlx5_tmh_priv_data_t;


typedef struct uct_rc_mlx5_release_desc {
    uct_recv_desc_t             super;
    unsigned                    offset;
} uct_rc_mlx5_release_desc_t;


typedef struct uct_rc_mlx5_ctx_priv {
    uint64_t                    tag;
    void                        *buffer;
    uint32_t                    app_ctx;
    uint32_t                    length;
    uint32_t                    tag_handle;
} uct_rc_mlx5_ctx_priv_t;

#if HAVE_IBV_EXP_DM
typedef struct uct_mlx5_dm_data {
    uct_worker_tl_data_t super;
    ucs_mpool_t          mp;
    struct ibv_mr        *mr;
    struct ibv_exp_dm    *dm;
    void                 *start_va;
    size_t               seg_len;
    unsigned             seg_count;
    unsigned             seg_attached;
    uct_ib_device_t      *device;
} uct_mlx5_dm_data_t;

typedef union uct_rc_mlx5_dm_copy_data {
    uct_rc_mlx5_am_short_hdr_t am_hdr;
    struct ibv_tmh             tm_hdr;
    char                       bytes[sizeof(uint64_t) * 2];
} UCS_S_PACKED uct_rc_mlx5_dm_copy_data_t;
#endif

typedef struct uct_rc_mlx5_iface_common {
    uct_rc_iface_t                   super;
    uct_ib_mlx5_iface_common_t       mlx5_common;
    struct {
        ucs_mpool_t                  atomic_desc_mp;
        uct_ib_mlx5_mmio_mode_t      mmio_mode;
        uint16_t                     bb_max;     /* limit number of outstanding WQE BBs */
        uint16_t                     fence_beat; /* 16bit is enough because if it wraps around,
                                                  * it means the older ops are already completed
                                                  * because QP size is less than 64k */
        uint8_t                      next_fm;
    } tx;
    struct {
        uct_ib_mlx5_srq_t            srq;
        void                         *pref_ptr;
    } rx;
    uct_ib_mlx5_cq_t                 cq[UCT_IB_DIR_NUM];
    struct {
        struct ibv_qp                *cmd_qp;    /* set if QP was created by UCX */
        uct_rc_mlx5_cmd_wq_t         cmd_wq;
        uct_rc_mlx5_tag_entry_t      *head;
        uct_rc_mlx5_tag_entry_t      *tail;
        uct_rc_mlx5_tag_entry_t      *list;

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
        uct_rc_mlx5_release_desc_t  eager_desc;
        uct_rc_mlx5_release_desc_t  rndv_desc;
        UCS_STATS_NODE_DECLARE(stats);
    } tm;
#if HAVE_IBV_EXP_DM
    struct {
        uct_mlx5_dm_data_t           *dm;
        size_t                       seg_len; /* cached value to avoid double-pointer access */
        ucs_status_t                 (*am_short)(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                                 const void *payload, unsigned length);
        ucs_status_t                 (*tag_short)(uct_ep_h tl_ep, uct_tag_t tag,
                                                  const void *data, size_t length);
    } dm;
#endif
    UCS_STATS_NODE_DECLARE(stats);
} uct_rc_mlx5_iface_common_t;

/**
 * Common RC/DC mlx5 interface configuration
 */
typedef struct uct_rc_mlx5_iface_common_config {
    uct_rc_iface_config_t             super;
    uct_ib_mlx5_iface_config_t        mlx5_common;
    unsigned                          tx_max_bb;
    struct {
        int                  enable;
        unsigned             list_size;
        size_t               max_bcopy;
    } tm;
} uct_rc_mlx5_iface_common_config_t;


UCS_CLASS_DECLARE(uct_rc_mlx5_iface_common_t);
UCS_CLASS_DECLARE_INIT_FUNC(uct_rc_mlx5_iface_common_t, uct_rc_iface_ops_t*,
                  uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, uct_rc_mlx5_iface_common_config_t*,
                  uct_ib_iface_init_attr_t*);


#define UCT_RC_MLX5_TM_STAT(_iface, _op) \
    UCS_STATS_UPDATE_COUNTER((_iface)->tm.stats, UCT_RC_MLX5_STAT_TAG_##_op, 1)

#define UCT_RC_MLX5_TM_ENABLED(_iface) (_iface)->tm.enabled

/* TMH can carry 2 bytes of data in its reserved filed */
#define UCT_RC_MLX5_TMH_PRIV_LEN       ucs_field_sizeof(uct_rc_mlx5_tmh_priv_data_t, \
                                                        data)

#define UCT_RC_MLX5_CHECK_RES_PTR(_iface, _ep) \
    UCT_RC_CHECK_CQE_RET(&(_iface)->super, _ep, &(_ep)->txqp, \
                         UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE)) \
    UCT_RC_CHECK_TXQP_RET(&(_iface)->super, _ep, &(_ep)->txqp, \
                          UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE))

#define UCT_RC_MLX5_CHECK_RNDV_PARAMS(_iovcnt, _header_len, _tm_len, \
                                      _max_inline, _max_rndv_hdr) \
   { \
       UCT_CHECK_PARAM_PTR(_iovcnt <= 1ul, "Wrong iovcnt %lu", iovcnt); \
       UCT_CHECK_PARAM_PTR(_header_len <= _max_rndv_hdr, \
                           "Invalid header len %u", _header_len); \
       UCT_CHECK_PARAM_PTR((_header_len + _tm_len) <= _max_inline, \
                           "Invalid RTS len gth %u", \
                           _header_len + _tm_len); \
   }

#define UCT_RC_MLX5_FILL_TM_IMM(_imm_data, _app_ctx, _ib_imm, _res_op, \
                                _op, _imm_suffix) \
   if (_imm_data == 0) { \
       _res_op  = _op; \
       _app_ctx = 0; \
       _ib_imm  = 0; \
   } else { \
       _res_op = UCS_PP_TOKENPASTE(_op, _imm_suffix); \
       uct_rc_mlx5_tag_imm_data_pack(&(_ib_imm), &(_app_ctx), _imm_data); \
   }

#define UCT_RC_MLX5_GET_TX_TM_DESC(_iface, _mp, _desc, _tag, _app_ctx, _hdr) \
   { \
       UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
       _hdr = _desc + 1; \
       uct_rc_mlx5_fill_tmh(_hdr, _tag, _app_ctx, IBV_EXP_TMH_EAGER); \
       _hdr += sizeof(struct ibv_tmh); \
   }

#define UCT_RC_MLX5_GET_TM_BCOPY_DESC(_iface, _mp, _desc, _tag, _app_ctx, \
                                      _pack_cb, _arg, _length) \
   { \
       void *hdr; \
       UCT_RC_MLX5_GET_TX_TM_DESC(_iface, _mp, _desc, _tag, _app_ctx, hdr) \
       (_desc)->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; \
       _length = _pack_cb(hdr, _arg); \
   }

#ifdef IBV_HW_TM
ucs_status_t uct_rc_mlx5_handle_rndv(uct_rc_mlx5_iface_common_t *iface,
                                     struct ibv_tmh *tmh, uct_tag_t tag,
                                     unsigned byte_len);


static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_fill_tmh(struct ibv_tmh *tmh, uct_tag_t tag,
                     uint32_t app_ctx, unsigned op)
{
    tmh->opcode  = op;
    tmh->app_ctx = app_ctx;
    tmh->tag     = tag;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_fill_rvh(struct ibv_rvh *rvh, const void *vaddr,
                     uint32_t rkey, uint32_t len)
{
    rvh->va   = htobe64((uint64_t)vaddr);
    rvh->rkey = htonl(rkey);
    rvh->len  = htonl(len);
}

static UCS_F_ALWAYS_INLINE unsigned
uct_rc_mlx5_tag_get_op_id(uct_rc_mlx5_iface_common_t *iface, uct_completion_t *comp)
{
    uint32_t prev_ph;
    return ucs_ptr_array_insert(&iface->tm.rndv_comps, comp, &prev_ph);
}


static UCS_F_ALWAYS_INLINE unsigned
uct_rc_mlx5_fill_tmh_priv_data(struct ibv_tmh *tmh, const void *hdr,
                               unsigned hdr_len, unsigned max_rndv_priv_data)
{
    uct_rc_mlx5_tmh_priv_data_t *priv = (uct_rc_mlx5_tmh_priv_data_t*)tmh->reserved;

    /* If header length is bigger tha max_rndv_priv_data size, need to add the
     * rest to the TMH reserved field. */
    if (hdr_len > max_rndv_priv_data) {
        priv->length = hdr_len - max_rndv_priv_data;
        ucs_assert(priv->length <= UCT_RC_MLX5_TMH_PRIV_LEN);
        memcpy(&priv->data, (char*)hdr, priv->length);
    } else {
        priv->length = 0;
    }

    return priv->length;
}
#endif

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_tag_imm_data_pack(uint32_t *ib_imm, uint32_t *app_ctx,
                              uint64_t imm_val)
{
    *ib_imm  = (uint32_t)(imm_val & 0xFFFFFFFF);
    *app_ctx = (uint32_t)(imm_val >> 32);
}

static UCS_F_ALWAYS_INLINE uint64_t
uct_rc_mlx5_tag_imm_data_unpack(uint32_t ib_imm, uint32_t app_ctx, int is_imm)
{
    return is_imm ? (((uint64_t)app_ctx << 32) | ib_imm) : 0ul;
}

static UCS_F_ALWAYS_INLINE uct_rc_mlx5_ctx_priv_t*
uct_rc_mlx5_ctx_priv(uct_tag_context_t *ctx)
{
    return (uct_rc_mlx5_ctx_priv_t*)ctx->priv;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_handle_rndv_fin(uct_rc_mlx5_iface_common_t *iface, uint32_t app_ctx)
{
    int found;
    void *rndv_comp;

    found = ucs_ptr_array_lookup(&iface->tm.rndv_comps, app_ctx, rndv_comp);
    ucs_assert_always(found > 0);
    uct_invoke_completion((uct_completion_t*)rndv_comp, UCS_OK);
    ucs_ptr_array_remove(&iface->tm.rndv_comps, app_ctx, 0);
}

extern ucs_config_field_t uct_rc_mlx5_common_config_table[];

unsigned uct_rc_mlx5_iface_srq_post_recv(uct_rc_iface_t *iface, uct_ib_mlx5_srq_t *srq);

void uct_rc_mlx5_iface_common_prepost_recvs(uct_rc_mlx5_iface_common_t *iface);

ucs_status_t uct_rc_mlx5_iface_common_init(uct_rc_mlx5_iface_common_t *iface,
                                           uct_rc_iface_t *rc_iface,
                                           const uct_rc_iface_config_t *config,
                                           const uct_ib_mlx5_iface_config_t *mlx5_config);

void uct_rc_mlx5_iface_common_cleanup(uct_rc_mlx5_iface_common_t *iface);

ucs_status_t uct_rc_mlx5_iface_common_dm_init(uct_rc_mlx5_iface_common_t *iface,
                                              uct_rc_iface_t *rc_iface,
                                              const uct_ib_mlx5_iface_config_t *mlx5_config);

void uct_rc_mlx5_iface_common_dm_cleanup(uct_rc_mlx5_iface_common_t *iface);

void uct_rc_mlx5_iface_common_query(uct_ib_iface_t *ib_iface,
                                    uct_iface_attr_t *iface_attr,
                                    size_t max_inline, size_t av_size);

void uct_rc_mlx5_iface_common_update_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                            uct_ib_iface_t *ib_iface);

void uct_rc_mlx5_iface_common_sync_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                          uct_ib_iface_t *ib_iface);

int uct_rc_mlx5_iface_commom_clean(uct_ib_mlx5_cq_t *mlx5_cq,
                                   uct_ib_mlx5_srq_t *srq, uint32_t qpn);

ucs_status_t uct_rc_mlx5_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                                    const uct_rc_mlx5_iface_common_config_t *config,
                                    struct ibv_exp_create_srq_attr *srq_init_attr,
                                    unsigned rndv_hdr_len,
                                    unsigned max_cancel_sync_ops);

void uct_rc_mlx5_tag_cleanup(uct_rc_mlx5_iface_common_t *iface);

ucs_status_t
uct_rc_mlx5_iface_common_tag_init(uct_rc_mlx5_iface_common_t *iface,
                                  uct_rc_mlx5_iface_common_config_t *config);

void uct_rc_mlx5_iface_common_tag_cleanup(uct_rc_mlx5_iface_common_t *iface);

ucs_status_t uct_rc_mlx5_ep_tag_rndv_cancel(uct_ep_h tl_ep, void *op);

ucs_status_t uct_rc_mlx5_iface_fence(uct_iface_h tl_iface, unsigned flags);

ucs_status_t uct_rc_mlx5_init_res_domain(uct_ib_iface_t *ib_iface);

void uct_rc_mlx5_cleanup_res_domain(uct_ib_iface_t *ib_iface);

void uct_rc_mlx5_common_packet_dump(uct_base_iface_t *iface, uct_am_trace_type_t type,
                                    void *data, size_t length, size_t valid_length,
                                    char *buffer, size_t max);

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_am_hdr_fill(uct_rc_mlx5_hdr_t *rch, uint8_t id)
{
#ifdef IBV_HW_TM
    rch->tmh_opcode   = IBV_TMH_NO_TAG;
#endif
    rch->rc_hdr.am_id = id;
}
#endif
