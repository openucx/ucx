/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_MLX5_COMMON_H
#define UCT_RC_MLX5_COMMON_H

#include <uct/ib/base/ib_device.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <uct/ib/mlx5/ib_mlx5.h>


/*
 * HW tag matching
 */
#if IBV_HW_TM
#  define UCT_RC_RNDV_HDR_LEN         sizeof(struct ibv_rvh)
#else
#  define UCT_RC_RNDV_HDR_LEN         0
#endif

#if IBV_HW_TM
#  if HAVE_INFINIBAND_TM_TYPES_H
#    include <infiniband/tm_types.h>
#  else
#    define ibv_tmh                         ibv_exp_tmh
#    define ibv_rvh                         ibv_exp_tmh_rvh
#    define IBV_TM_CAP_RC                   IBV_EXP_TM_CAP_RC
#    define IBV_TMH_EAGER                   IBV_EXP_TMH_EAGER
#    define IBV_TMH_RNDV                    IBV_EXP_TMH_RNDV
#    define IBV_TMH_FIN                     IBV_EXP_TMH_FIN
#    define IBV_TMH_NO_TAG                  IBV_EXP_TMH_NO_TAG
#  endif
#  define IBV_DEVICE_TM_CAPS(_dev, _field)  ((_dev)->dev_attr.tm_caps._field)
#else
#  define IBV_TM_CAP_RC                     0
#  define IBV_DEVICE_TM_CAPS(_dev, _field)  0
#endif

#if HAVE_STRUCT_IBV_TM_CAPS_FLAGS
#  define IBV_DEVICE_TM_FLAGS(_dev)         IBV_DEVICE_TM_CAPS(_dev, flags)
#else
#  define IBV_DEVICE_TM_FLAGS(_dev)         IBV_DEVICE_TM_CAPS(_dev, capability_flags)
#endif

#define IBV_DEVICE_MAX_UNEXP_COUNT          UCS_BIT(14)

#if HAVE_DECL_IBV_EXP_CREATE_SRQ
#  define ibv_srq_init_attr_ex              ibv_exp_create_srq_attr
#endif

#define UCT_RC_MLX5_OPCODE_FLAG_RAW         0x100
#define UCT_RC_MLX5_OPCODE_FLAG_TM          0x200
#define UCT_RC_MLX5_OPCODE_MASK             0xff
#define UCT_RC_MLX5_SINGLE_FRAG_MSG(_flags) \
    (((_flags) & UCT_CB_PARAM_FLAG_FIRST) && !((_flags) & UCT_CB_PARAM_FLAG_MORE))

#define UCT_RC_MLX5_CHECK_AM_ZCOPY(_id, _header_length, _length, _seg_size, _av_size) \
    UCT_CHECK_AM_ID(_id); \
    UCT_RC_CHECK_ZCOPY_DATA(_header_length, _length, _seg_size) \
    UCT_CHECK_LENGTH(sizeof(uct_rc_mlx5_hdr_t) + _header_length, 0, \
                     UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(_av_size), "am_zcopy header");


#define UCT_RC_MLX5_CHECK_AM_SHORT(_id, _header_t, _length, _av_size) \
    UCT_CHECK_AM_ID(_id); \
    UCT_CHECK_LENGTH(sizeof(_header_t) + _length, 0, \
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


typedef enum {
    UCT_RC_MLX5_SRQ_TOPO_LIST,
    UCT_RC_MLX5_SRQ_TOPO_CYCLIC,
    UCT_RC_MLX5_SRQ_TOPO_CYCLIC_EMULATED,
    UCT_RC_MLX5_SRQ_TOPO_LAST
} uct_rc_mlx5_srq_topo_t;


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
    UCT_RC_MLX5_OPCODE_TAG_MATCHING          = 0x28,
    UCT_RC_MLX5_CQE_APP_TAG_MATCHING         = 1,

    /* tag segment flags */
    UCT_RC_MLX5_SRQ_FLAG_TM_SW_CNT           = (1 << 6),
    UCT_RC_MLX5_SRQ_FLAG_TM_CQE_REQ          = (1 << 7),

    /* tag CQE codes */
    UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED       = 0x1,
    UCT_RC_MLX5_CQE_APP_OP_TM_EXPECTED       = 0x2,
    UCT_RC_MLX5_CQE_APP_OP_TM_UNEXPECTED     = 0x3,
    UCT_RC_MLX5_CQE_APP_OP_TM_NO_TAG         = 0x4,
    UCT_RC_MLX5_CQE_APP_OP_TM_APPEND         = 0x5,
    UCT_RC_MLX5_CQE_APP_OP_TM_REMOVE         = 0x6,
    UCT_RC_MLX5_CQE_APP_OP_TM_CONSUMED_MSG   = 0xA
};

enum {
    UCT_RC_MLX5_POLL_FLAG_TM                 = UCS_BIT(0),
    UCT_RC_MLX5_POLL_FLAG_HAS_EP             = UCS_BIT(1),
    UCT_RC_MLX5_POLL_FLAG_TAG_CQE            = UCS_BIT(2),
    UCT_RC_MLX5_POLL_FLAG_LINKED_LIST        = UCS_BIT(3)
};


#define UCT_RC_MLX5_RMA_MAX_IOV(_av_size) \
    ((UCT_IB_MLX5_MAX_SEND_WQE_SIZE - ((_av_size) + \
     sizeof(struct mlx5_wqe_raddr_seg) + sizeof(struct mlx5_wqe_ctrl_seg))) / \
     sizeof(struct mlx5_wqe_data_seg))


#if IBV_HW_TM
#  define UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(_av_size) \
       (UCT_IB_MLX5_AM_MAX_SHORT(_av_size + sizeof(struct ibv_tmh))/ \
        sizeof(struct mlx5_wqe_data_seg))
#else
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
    uct_rc_mlx5_srq_op_t          *ops;     /* array of operations on command QP */
    int                           ops_head; /* points to the next operation to be completed */
    int                           ops_tail; /* points to the last adde operation*/
    int                           ops_mask; /* mask which bounds head and tail by
                                               ops array size */
} uct_rc_mlx5_cmd_wq_t;


/* Message context used with multi-packet XRQ */
typedef struct uct_rc_mlx5_mp_context {
    /* Storage for a per-message user-defined context. Must be passed unchanged
     * to the user in uct_tag_unexp_eager_cb_t. */
    void                          *context;

    /* Tag is saved when first fragment (with TMH) arrives and then passed to
     * the eager unexpected callback for subsequent fragments. */
    uct_tag_t                     tag;

    /* With MP XRQ immediate value is delivered with the last fragment, while
     * TMH is present in the first fragment only. Need to save app_context
     * from TMH in this field and construct immediate data for unexpected
     * eager callback when the last message fragment arrives. */
    uint32_t                      app_ctx;

    /* Used when local EP can be found by sender QP number (rc_mlx5 tl).
     * When 0, it means that tag eager unexpected multi-fragmented message
     * is being processed (not all fragments are delivered to the user via
     * uct_tag_unexp_eager_cb_t callback yet). Otherwise, any incoming tag
     * eager message should be either a single fragment message or the first
     * fragment of multi-fragmeneted message. */
    uint8_t                       free;
} uct_rc_mlx5_mp_context_t;


typedef struct uct_rc_mlx5_mp_hash_key {
    uint64_t                      guid;
    uint32_t                      qp_num;
} uct_rc_mlx5_mp_hash_key_t;


static UCS_F_ALWAYS_INLINE int
uct_rc_mlx5_mp_hash_equal(uct_rc_mlx5_mp_hash_key_t key1,
                          uct_rc_mlx5_mp_hash_key_t key2)
{
    return (key1.qp_num == key2.qp_num) && (key1.guid == key2.guid);
}


static UCS_F_ALWAYS_INLINE khint32_t
uct_rc_mlx5_mp_hash_func(uct_rc_mlx5_mp_hash_key_t key)
{
    return kh_int64_hash_func(key.guid ^ key.qp_num);
}


KHASH_MAP_INIT_INT64(uct_rc_mlx5_mp_hash_lid, uct_rc_mlx5_mp_context_t);


KHASH_INIT(uct_rc_mlx5_mp_hash_gid, uct_rc_mlx5_mp_hash_key_t,
           uct_rc_mlx5_mp_context_t, 1, uct_rc_mlx5_mp_hash_func,
           uct_rc_mlx5_mp_hash_equal);


#if IBV_HW_TM
#  define UCT_RC_MLX5_IFACE_GET_TM_BCOPY_DESC(_iface, _mp, _desc, _tag, _app_ctx, \
                                              _pack_cb, _arg, _length) \
       { \
           void *hdr; \
           UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
           (_desc)->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; \
           hdr = (_desc) + 1; \
           uct_rc_mlx5_fill_tmh(hdr, _tag, _app_ctx, IBV_TMH_EAGER); \
           hdr = UCS_PTR_BYTE_OFFSET(hdr, sizeof(struct ibv_tmh)); \
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

void uct_rc_mlx5_release_desc(uct_recv_desc_t *self, void *desc);

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

#if HAVE_IBV_DM
typedef struct uct_mlx5_dm_data {
    uct_worker_tl_data_t super;
    ucs_mpool_t          mp;
    struct ibv_mr        *mr;
    struct ibv_dm        *dm;
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

#define uct_rc_mlx5_tag_addr_hash(_ptr) kh_int64_hash_func((uintptr_t)(_ptr))
KHASH_INIT(uct_rc_mlx5_tag_addrs, void*, char, 0, uct_rc_mlx5_tag_addr_hash,
           kh_int64_hash_equal)

typedef struct uct_rc_mlx5_iface_common {
    uct_rc_iface_t                     super;
    struct {
        ucs_mpool_t                    atomic_desc_mp;
        uct_ib_mlx5_mmio_mode_t        mmio_mode;
        uint16_t                       bb_max;     /* limit number of outstanding WQE BBs */
    } tx;
    struct {
        uct_ib_mlx5_srq_t              srq;
        void                           *pref_ptr;
    } rx;
    uct_ib_mlx5_cq_t                   cq[UCT_IB_DIR_NUM];
    struct {
        uct_rc_mlx5_cmd_wq_t           cmd_wq;
        uct_rc_mlx5_tag_entry_t        *head;
        uct_rc_mlx5_tag_entry_t        *tail;
        uct_rc_mlx5_tag_entry_t        *list;
        ucs_mpool_t                    *bcopy_mp;
        khash_t(uct_rc_mlx5_tag_addrs) tag_addrs;

        ucs_ptr_array_t                rndv_comps;
        size_t                         max_bcopy;
        size_t                         max_zcopy;
        unsigned                       num_tags;
        unsigned                       num_outstanding;
        unsigned                       max_rndv_data;
        uint16_t                       unexpected_cnt;
        uint16_t                       cmd_qp_len;
        uint8_t                        enabled;
        struct {
            uint8_t                    num_strides;
            ucs_mpool_t                tx_mp;
            uct_rc_mlx5_mp_context_t   last_frag_ctx;
            khash_t(uct_rc_mlx5_mp_hash_lid) hash_lid;
            khash_t(uct_rc_mlx5_mp_hash_gid) hash_gid;
        } mp;
        struct {
            void                       *arg; /* User defined arg */
            uct_tag_unexp_eager_cb_t   cb;   /* Callback for unexpected eager messages */
        } eager_unexp;

        struct {
            void                       *arg; /* User defined arg */
            uct_tag_unexp_rndv_cb_t    cb;   /* Callback for unexpected rndv messages */
        } rndv_unexp;
        uct_rc_mlx5_release_desc_t     eager_desc;
        uct_rc_mlx5_release_desc_t     rndv_desc;
        uct_rc_mlx5_release_desc_t     am_desc;
        UCS_STATS_NODE_DECLARE(stats)
    } tm;
#if HAVE_IBV_DM
    struct {
        uct_mlx5_dm_data_t             *dm;
        size_t                         seg_len; /* cached value to avoid double-pointer access */
        ucs_status_t                   (*am_short)(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                                   const void *payload, unsigned length);
        ucs_status_t                   (*tag_short)(uct_ep_h tl_ep, uct_tag_t tag,
                                                    const void *data, size_t length);
    } dm;
#endif
#if HAVE_DECL_MLX5DV_DEVX_SUBSCRIBE_DEVX_EVENT
    struct mlx5dv_devx_event_channel   *event_channel;
#endif
    struct {
        uint8_t                        atomic_fence_flag;
        uct_rc_mlx5_srq_topo_t         srq_topo;
        uint8_t                        log_ack_req_freq;
    } config;
    UCS_STATS_NODE_DECLARE(stats)
} uct_rc_mlx5_iface_common_t;

/**
 * Common RC/DC mlx5 interface configuration
 */
typedef struct uct_rc_mlx5_iface_common_config {
    uct_ib_mlx5_iface_config_t           super;
    unsigned                             tx_max_bb;
    struct {
        int                              enable;
        unsigned                         list_size;
        size_t                           seg_size;
        ucs_ternary_auto_value_t         mp_enable;
        size_t                           mp_num_strides;
    } tm;
    unsigned                             exp_backoff;
    uint8_t                              log_ack_req_freq;
    UCS_CONFIG_STRING_ARRAY_FIELD(types) srq_topo;
} uct_rc_mlx5_iface_common_config_t;


UCS_CLASS_DECLARE(uct_rc_mlx5_iface_common_t, uct_iface_ops_t*,
                  uct_rc_iface_ops_t*, uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, uct_rc_iface_common_config_t*,
                  uct_rc_mlx5_iface_common_config_t*,
                  uct_ib_iface_init_attr_t*);


#define UCT_RC_MLX5_TM_STAT(_iface, _op) \
    UCS_STATS_UPDATE_COUNTER((_iface)->tm.stats, UCT_RC_MLX5_STAT_TAG_##_op, 1)

#define UCT_RC_MLX5_TM_ENABLED(_iface) (_iface)->tm.enabled

#define UCT_RC_MLX5_MP_ENABLED(_iface) ((_iface)->tm.mp.num_strides > 1)

/* TMH can carry 2 bytes of data in its reserved filed */
#define UCT_RC_MLX5_TMH_PRIV_LEN       ucs_field_sizeof(uct_rc_mlx5_tmh_priv_data_t, \
                                                        data)

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


/* Max value for log_ack_req_freq field in QPC */
#define UCT_RC_MLX5_MAX_LOG_ACK_REQ_FREQ 8


#if IBV_HW_TM
void uct_rc_mlx5_handle_unexp_rndv(uct_rc_mlx5_iface_common_t *iface,
                                   struct ibv_tmh *tmh, uct_tag_t tag,
                                   struct mlx5_cqe64 *cqe, unsigned flags,
                                   unsigned byte_len, int poll_flags);


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
    return ucs_ptr_array_insert(&iface->tm.rndv_comps, comp);
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
    void *rndv_comp = NULL;
    int found;

    found = ucs_ptr_array_lookup(&iface->tm.rndv_comps, app_ctx, rndv_comp);
    ucs_assert_always(found > 0);
    uct_invoke_completion((uct_completion_t*)rndv_comp, UCS_OK);
    ucs_ptr_array_remove(&iface->tm.rndv_comps, app_ctx);
}

extern ucs_config_field_t uct_rc_mlx5_common_config_table[];

unsigned uct_rc_mlx5_iface_srq_post_recv(uct_rc_mlx5_iface_common_t *iface);
unsigned uct_rc_mlx5_iface_srq_post_recv_ll(uct_rc_mlx5_iface_common_t *iface);

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
                                    size_t max_inline, size_t max_tag_eager_iov);

void uct_rc_mlx5_iface_common_update_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                            uct_ib_iface_t *ib_iface);

void uct_rc_mlx5_iface_common_sync_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                          uct_ib_iface_t *ib_iface);

ucs_status_t
uct_rc_mlx5_iface_common_arm_cq(uct_ib_iface_t *ib_iface, uct_ib_dir_t dir,
                                int solicited_only);

void uct_rc_mlx5_iface_common_event_cq(uct_ib_iface_t *ib_iface,
                                       uct_ib_dir_t dir);

int uct_rc_mlx5_iface_commom_clean(uct_ib_mlx5_cq_t *mlx5_cq,
                                   uct_ib_mlx5_srq_t *srq, uint32_t qpn);

static UCS_F_MAYBE_UNUSED void
uct_rc_mlx5_iface_tm_set_cmd_qp_len(uct_rc_mlx5_iface_common_t *iface)
{
    /* 2 ops for each tag (ADD + DEL) and extra ops for SYNC. */
    iface->tm.cmd_qp_len = (2 * iface->tm.num_tags) + 2;
}

#if IBV_HW_TM
void uct_rc_mlx5_init_rx_tm_common(uct_rc_mlx5_iface_common_t *iface,
                                   const uct_rc_iface_common_config_t *config,
                                   unsigned rndv_hdr_len);

ucs_status_t uct_rc_mlx5_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                                    const uct_rc_iface_common_config_t *config,
                                    struct ibv_srq_init_attr_ex *srq_init_attr,
                                    unsigned rndv_hdr_len);
#else
static UCS_F_MAYBE_UNUSED ucs_status_t
uct_rc_mlx5_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                       const uct_rc_iface_common_config_t *config,
                       struct ibv_srq_init_attr_ex *srq_init_attr,
                       unsigned rndv_hdr_len)
{
    return UCS_ERR_UNSUPPORTED;
}
#endif

#if IBV_HW_TM && HAVE_DEVX
ucs_status_t uct_rc_mlx5_devx_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                                         const uct_rc_iface_common_config_t *config,
                                         int dc, unsigned rndv_hdr_len);
#else
static UCS_F_MAYBE_UNUSED ucs_status_t
uct_rc_mlx5_devx_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                            const uct_rc_iface_common_config_t *config,
                            int dc, unsigned rndv_hdr_len)
{
    return UCS_ERR_UNSUPPORTED;
}
#endif

#if HAVE_DEVX
ucs_status_t uct_rc_mlx5_devx_init_rx(uct_rc_mlx5_iface_common_t *iface,
                                      const uct_rc_iface_common_config_t *config);

void uct_rc_mlx5_devx_cleanup_srq(uct_ib_mlx5_md_t *md, uct_ib_mlx5_srq_t *srq);
#else
static UCS_F_MAYBE_UNUSED ucs_status_t
uct_rc_mlx5_devx_init_rx(uct_rc_mlx5_iface_common_t *iface,
                         const uct_rc_iface_common_config_t *config)
{
    return UCS_ERR_UNSUPPORTED;
}

static UCS_F_MAYBE_UNUSED void
uct_rc_mlx5_devx_cleanup_srq(uct_ib_mlx5_md_t *md, uct_ib_mlx5_srq_t *srq)
{
    ucs_bug("DEVX SRQ cleanup has to be done only if DEVX support is enabled");
}
#endif

void uct_rc_mlx5_tag_cleanup(uct_rc_mlx5_iface_common_t *iface);

ucs_status_t uct_rc_mlx5_iface_common_tag_init(uct_rc_mlx5_iface_common_t *iface);

void uct_rc_mlx5_iface_common_tag_cleanup(uct_rc_mlx5_iface_common_t *iface);

ucs_status_t uct_rc_mlx5_ep_tag_rndv_cancel(uct_ep_h tl_ep, void *op);

void uct_rc_mlx5_common_packet_dump(uct_base_iface_t *iface, uct_am_trace_type_t type,
                                    void *data, size_t length, size_t valid_length,
                                    char *buffer, size_t max);

static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_am_hdr_fill(uct_rc_mlx5_hdr_t *rch, uint8_t id)
{
#if IBV_HW_TM
    rch->tmh_opcode   = IBV_TMH_NO_TAG;
#endif
    rch->rc_hdr.am_id = id;
}

#if HAVE_DECL_MLX5DV_CREATE_QP
void uct_rc_mlx5_common_fill_dv_qp_attr(uct_rc_mlx5_iface_common_t *iface,
                                        struct ibv_qp_init_attr_ex *qp_attr,
                                        struct mlx5dv_qp_init_attr *dv_attr,
                                        unsigned scat2cqe_dir_mask);
#endif

#if HAVE_DEVX
ucs_status_t
uct_rc_mlx5_iface_common_devx_connect_qp(uct_rc_mlx5_iface_common_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         uint32_t dest_qp_num,
                                         struct ibv_ah_attr *ah_attr,
                                         enum ibv_mtu path_mtu,
                                         uint8_t path_index);

#else
static UCS_F_MAYBE_UNUSED ucs_status_t
uct_rc_mlx5_iface_common_devx_connect_qp(uct_rc_mlx5_iface_common_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         uint32_t dest_qp_num,
                                         struct ibv_ah_attr *ah_attr,
                                         enum ibv_mtu path_mtu,
                                         uint8_t path_index)
{
    return UCS_ERR_UNSUPPORTED;
}
#endif

ucs_status_t uct_rc_mlx5_devx_iface_init_events(uct_rc_mlx5_iface_common_t *iface);

void uct_rc_mlx5_devx_iface_free_events(uct_rc_mlx5_iface_common_t *iface);

ucs_status_t uct_rc_mlx5_devx_iface_subscribe_event(uct_rc_mlx5_iface_common_t *iface,
                                                    uct_ib_mlx5_qp_t *qp,
                                                    unsigned event_num,
                                                    enum ibv_event_type event_type,
                                                    unsigned event_data);

void uct_rc_mlx5_iface_fill_attr(uct_rc_mlx5_iface_common_t *iface,
                                 uct_ib_mlx5_qp_attr_t *qp_attr,
                                 unsigned max_send_wr,
                                 uct_ib_mlx5_srq_t *srq);

ucs_status_t
uct_rc_mlx5_common_iface_init_rx(uct_rc_mlx5_iface_common_t *iface,
                                 const uct_rc_iface_common_config_t *rc_config);

void uct_rc_mlx5_destroy_srq(uct_ib_mlx5_md_t *md, uct_ib_mlx5_srq_t *srq);

#endif
