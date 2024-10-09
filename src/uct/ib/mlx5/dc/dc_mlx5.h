/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2016-2018. ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_IFACE_H
#define UCT_DC_IFACE_H

#include <ucs/datastruct/array.h>
#include <ucs/debug/assert.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <uct/ib/rc/verbs/rc_verbs.h>
#include <uct/ib/mlx5/rc/rc_mlx5_common.h>
#include <uct/ib/ud/base/ud_iface_common.h>
#include <uct/ib/mlx5/ud/ud_mlx5_common.h>


/*
 * HW tag matching
 */
#if IBV_HW_TM
/* upstream tm_types.h doesn't provide RAVH header */
struct ibv_ravh {
    uint32_t    sl_dct;
    uint32_t    reserved;    /* must be zero */
    uint64_t    dc_access_key;
};
#  define UCT_DC_RNDV_HDR_LEN   (sizeof(struct ibv_rvh) + \
                                 sizeof(struct ibv_ravh))
#else
#  define UCT_DC_RNDV_HDR_LEN   0
#endif

#define UCT_DC_MLX5_IFACE_MAX_DCI_POOLS      16

#define UCT_DC_MLX5_IFACE_ADDR_TM_ENABLED(_addr) \
    (!!((_addr)->flags & UCT_DC_MLX5_IFACE_ADDR_HW_TM))

#define UCT_DC_MLX5_IFACE_TXQP_DCI_GET(_iface, _dci, _txqp, _txwq) \
    { \
        ucs_assert(_dci != UCT_DC_MLX5_EP_NO_DCI); \
        _txqp = &uct_dc_mlx5_iface_dci(_iface, _dci)->txqp; \
        _txwq = &uct_dc_mlx5_iface_dci(_iface, _dci)->txwq; \
    }

/**
 * Set iface config flag for enabling full handshake on DCI/DCT,
 * according to user configuration. Fail if the user requests to
 * force full-handshake, while the HW does not support it.
 */
#define UCT_DC_MLX5_CHECK_FORCE_FULL_HANDSHAKE(_self, _config, _config_name, \
                                               _flag_name, _status, _err) \
    if (!((_self)->version_flag & UCT_DC_MLX5_IFACE_ADDR_DC_V2) && \
        ((_config)->_config_name##_full_handshake == UCS_YES)) { \
        _status = UCS_ERR_UNSUPPORTED; \
        goto _err; \
    } \
    if ((_config)->_config_name##_full_handshake != UCS_NO) { \
        (_self)->flags |= UCT_DC_MLX5_IFACE_FLAG_##_flag_name##_FULL_HANDSHAKE; \
    }


typedef struct uct_dc_mlx5_ep     uct_dc_mlx5_ep_t;
typedef struct uct_dc_mlx5_iface  uct_dc_mlx5_iface_t;


typedef enum {
    UCT_DC_MLX5_IFACE_ADDR_HW_TM            = UCS_BIT(0),
    UCT_DC_MLX5_IFACE_ADDR_DC_V1            = UCS_BIT(1),
    UCT_DC_MLX5_IFACE_ADDR_DC_V2            = UCS_BIT(2),
    UCT_DC_MLX5_IFACE_ADDR_FLUSH_RKEY       = UCS_BIT(3),
    UCT_DC_MLX5_IFACE_ADDR_MAX_RD_ATOMIC_16 = UCS_BIT(4),
    UCT_DC_MLX5_IFACE_ADDR_DC_VERS          = UCT_DC_MLX5_IFACE_ADDR_DC_V1 |
                                              UCT_DC_MLX5_IFACE_ADDR_DC_V2
} uct_dc_mlx5_iface_addr_flags_t;


typedef enum {
    /** uidx is set to dci idx */
    UCT_DC_MLX5_IFACE_FLAG_UIDX                     = UCS_BIT(0),

    /** Flow control endpoint is using a DCI in error state */
    UCT_DC_MLX5_IFACE_FLAG_FC_EP_FAILED             = UCS_BIT(1),

    /** Enable full handshake for DCI */
    UCT_DC_MLX5_IFACE_FLAG_DCI_FULL_HANDSHAKE       = UCS_BIT(2),

    /** Enable full handshake for DCT */
    UCT_DC_MLX5_IFACE_FLAG_DCT_FULL_HANDSHAKE       = UCS_BIT(3),

    /** Disable PUT capability (RDMA_WRITE) */
    UCT_DC_MLX5_IFACE_FLAG_DISABLE_PUT              = UCS_BIT(4)
} uct_dc_mlx5_iface_flags_t;


typedef struct uct_dc_mlx5_iface_addr {
    uct_ib_uint24_t   qp_num;
    uint8_t           atomic_mr_id;
    uint8_t           flags;
} UCS_S_PACKED uct_dc_mlx5_iface_addr_t;


typedef struct uct_dc_mlx5_iface_flush_addr {
    uct_dc_mlx5_iface_addr_t super;
    /* this is upper 16 bit of rkey used for flush_remote operation,
     * middle 8 bit is stored in atomic_mr_id of uct_dc_mlx5_iface_addr_t
     * structure, the lowest 8 bit must be 0 (not stored) */
    uint16_t                 flush_rkey_hi;
} UCS_S_PACKED uct_dc_mlx5_iface_flush_addr_t;


/**
 * dci policies:
 * - fixed: all eps always use same dci no matter what
 * - dcs:
 *    - ep uses already assigned dci or
 *    - free dci is assigned in LIFO (stack) order or
 *    - ep has not resources to transmit
 *    - on FULL completion (once there are no outstanding ops)
 *      dci is pushed to the stack of free dcis
 *    it is possible that ep will never release its dci:
 *      ep send, gets some completion, sends more, repeat
 * - dcs + quota:
 *    - same as dcs with following addition:
 *    - if dci can not tx, and there are eps waiting for dci
 *      allocation ep goes into tx_wait state
 *    - in tx_wait state:
 *          - ep can not transmit while there are eps
 *            waiting for dci allocation. This will break
 *            starvation.
 *          - if there are no eps that are waiting for dci allocation
 *            ep goes back to normal state
 * - random
 *    - dci is chosen by random() % ndci
 *    - ep keeps using dci as long as it has outstanding sends
 *
 * Not implemented policies:
 *
 * - hash:
 *    - dci is allocated to ep by some hash function
 *      for example dlid % ndci
 *
 */
typedef enum {
    /* Policies with dedicated DCI per active connection */
    UCT_DC_TX_POLICY_DCS,
    UCT_DC_TX_POLICY_DCS_QUOTA,
    UCT_DC_TX_POLICY_DCS_HYBRID,
    /* Policies with shared DCI */
    UCT_DC_TX_POLICY_SHARED_FIRST,
    UCT_DC_TX_POLICY_RAND = UCT_DC_TX_POLICY_SHARED_FIRST,
    UCT_DC_TX_POLICY_HW_DCS,
    UCT_DC_TX_POLICY_LAST
} uct_dc_tx_policy_t;


/**
 * dct port affinity policies for RoCE LAG device
 * - default: use the first physical port number for affinity
 * - random : use random slave port number for affinity
 * - [1, lag_level]: use given value as the slave port number for affinity
 */
typedef enum {
    UCT_DC_MLX5_DCT_AFFINITY_DEFAULT,
    UCT_DC_MLX5_DCT_AFFINITY_RANDOM,
    UCT_DC_MLX5_DCT_AFFINITY_LAST
} uct_dc_mlx5_dct_affinity_t;


typedef struct uct_dc_mlx5_iface_config {
    uct_rc_iface_common_config_t        super;
    uct_rc_mlx5_iface_common_config_t   rc_mlx5_common;
    uct_ud_iface_common_config_t        ud_common;
    int                                 ndci;
    int                                 tx_policy;
    ucs_on_off_auto_value_t             tx_port_affinity;
    ucs_ternary_auto_value_t            dci_full_handshake;
    ucs_ternary_auto_value_t            dci_ka_full_handshake;
    ucs_ternary_auto_value_t            dct_full_handshake;
    unsigned                            dct_affinity;
    unsigned                            quota;
    unsigned                            rand_seed;
    ucs_time_t                          fc_hard_req_timeout;
    uct_ud_mlx5_iface_common_config_t   mlx5_ud;
    unsigned                            num_dci_channels;
    unsigned                            dcis_initial_capacity;
} uct_dc_mlx5_iface_config_t;


typedef void (*uct_dc_dci_handle_failure_func_t)(uct_dc_mlx5_iface_t *iface,
                                                 struct mlx5_cqe64 *cqe,
                                                 uint8_t dci_index,
                                                 ucs_status_t status);


typedef enum {
    /* Indicates that this specific dci is shared, regardless of policy */
    UCT_DC_DCI_FLAG_SHARED = UCS_BIT(0),
} uct_dc_dci_flags_t;


typedef struct uct_dc_dci {
    uct_rc_txqp_t                 txqp; /* DCI qp */
    uct_ib_mlx5_txwq_t            txwq; /* DCI mlx5 wq */
    union {
        uct_dc_mlx5_ep_t          *ep;  /* points to an endpoint that currently
                                           owns the dci. Relevant only for dcs
                                           and dcs quota policies. */
        ucs_arbiter_group_t       arb_group; /* pending group, relevant for rand
                                                policy. With rand, groups are not
                                                descheduled until all elements
                                                processed. Better have dci num
                                                groups scheduled than ep num. */
    };
    uint8_t                       pool_index; /* DCI pool index. */
    uint8_t                       path_index; /* Path index */
    uint8_t                       next_channel_index; /* next DCI channel index
                                                         to be used by EP */
    uint8_t                       flags; /* See uct_dc_dci_flags_t */
} uct_dc_dci_t;


typedef struct uct_dc_fc_sender_data {
    uint64_t                      ep;
    struct {
        uint64_t                  seq;
        int                       is_global;
        union ibv_gid             gid;
    } UCS_S_PACKED payload;
} UCS_S_PACKED uct_dc_fc_sender_data_t;

typedef struct uct_dc_fc_request {
    uct_rc_pending_req_t          super;
    uct_dc_fc_sender_data_t       sender;
    uint32_t                      dct_num;

    /* Lid can be stored either in BE or in LE order. The endianness depends
     * on the transport (BE for mlx5 and LE for dc verbs) */
    uint16_t                      lid;
} uct_dc_fc_request_t;


typedef struct uct_dc_mlx5_ep_fc_entry {
    uint64_t   seq; /* Sequence number in FC_HARD_REQ's sender data */
    ucs_time_t send_time; /* Last time FC_HARD_REQ was sent */
} uct_dc_mlx5_ep_fc_entry_t;


KHASH_MAP_INIT_INT64(uct_dc_mlx5_fc_hash, uct_dc_mlx5_ep_fc_entry_t);

typedef struct uct_dc_mlx5_dci_config {
    uint8_t path_index;
    uint8_t max_rd_atomic;
} uct_dc_mlx5_dci_config_t;

KHASH_MAP_INIT_INT64(uct_dc_mlx5_config_hash, uint8_t);

UCS_ARRAY_DECLARE_TYPE(uct_dc_mlx5_pool_stack_t, uint8_t, uint8_t);

/* DCI pool
 * same array is used to store DCI's to allocate and DCI's to release:
 *
 * +--------------+-----+-------------+
 * | to release   |     | to allocate |
 * +--------------+-----+-------------+
 * ^              ^     ^             ^
 * |              |     |             |
 * 0        release     stack      ndci
 *              top     top
 *
 * Overall count of DCI's to release and allocated DCI's could not be more than
 * ndci and these stacks are not intersected
 */
typedef struct {
    int8_t                   stack_top;         /* dci stack top */
    int8_t                   release_stack_top; /* releasing dci's stack,
                                                   points to last DCI to release
                                                   or -1 if no DCI's to release */
    uct_dc_mlx5_pool_stack_t stack;             /* LIFO of indexes of available dcis */
    ucs_arbiter_t            arbiter;           /* queue of requests waiting for DCI */
    uct_dc_mlx5_dci_config_t config;
} uct_dc_mlx5_dci_pool_t;


UCS_ARRAY_DECLARE_TYPE(uct_dc_dci_array_t, uint16_t, uct_dc_dci_t);

struct uct_dc_mlx5_iface {
    uct_rc_mlx5_iface_common_t       super;
    struct {
        /* Array of dcis */
        uct_dc_dci_array_t           dcis;

        /* Number of DCIs */
        uint8_t                      ndci;

        /* Whether to set port affinity */
        uint8_t                      port_affinity;

        /* LIFO is only relevant for dcs allocation policy */
        uct_dc_mlx5_dci_pool_t       dci_pool[UCT_DC_MLX5_IFACE_MAX_DCI_POOLS];
        uint8_t                      num_dci_pools;

        /* dci selection algorithm */
        uint8_t                      policy;

        /* if available tx is lower, let another endpoint use the dci */
        int16_t                      available_quota;

        /* DCI max elements */
        unsigned                     bb_max;

        /* Used to send grant messages for all peers */
        uct_dc_mlx5_ep_t             *fc_ep;

        /* Hash of expected FC grants */
        khash_t(uct_dc_mlx5_fc_hash) fc_hash;

        /* Sequence number of expected FC grants */
        uint64_t                     fc_seq;

        /* Timeout for sending FC_HARD_REQ when FC window is empty */
        ucs_time_t                   fc_hard_req_timeout;

        /* Next time when FC_HARD_REQ operations should be resent */
        ucs_time_t                   fc_hard_req_resend_time;

        /* Callback ID of FC_HARD_REQ resend operation */
        uct_worker_cb_id_t           fc_hard_req_progress_cb_id;

        /* Seed used for random dci allocation */
        unsigned                     rand_seed;

        ucs_arbiter_callback_t       pend_cb;

        uint8_t                      dci_pool_release_bitmap;

        uint8_t                      av_fl_mlid;

        uint8_t                      num_dci_channels;

        uint16_t                     dcis_initial_capacity;

        /* used in hybrid dcs policy otherwise -1 */
        uint16_t                     hybrid_hw_dci;
    } tx;

    struct {
        uct_ib_mlx5_qp_t             dct;

        uint8_t                      port_affinity;
    } rx;

    khash_t(uct_dc_mlx5_config_hash) dc_config_hash;

    uint8_t                          version_flag;

    /* iface flags, see uct_dc_mlx5_iface_flags_t */
    uint16_t                         flags;

    uct_ud_mlx5_iface_common_t       ud_common;
};


extern ucs_config_field_t uct_dc_mlx5_iface_config_table[];

extern const char *uct_dc_tx_policy_names[];


ucs_status_t
uct_dc_mlx5_iface_create_dct(uct_dc_mlx5_iface_t *iface,
                             const uct_dc_mlx5_iface_config_t *config);

ucs_status_t uct_dc_mlx5_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

ucs_status_t uct_dc_mlx5_iface_flush(uct_iface_h tl_iface, unsigned flags, uct_completion_t *comp);

void uct_dc_mlx5_iface_set_quota(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_iface_config_t *config);

ucs_status_t uct_dc_mlx5_iface_init_fc_ep(uct_dc_mlx5_iface_t *iface);

ucs_status_t uct_dc_mlx5_iface_fc_grant(uct_pending_req_t *self);

const char *
uct_dc_mlx5_fc_req_str(uct_dc_fc_request_t *dc_req, char *buf, size_t max);

void uct_dc_mlx5_fc_entry_iter_del(uct_dc_mlx5_iface_t *iface, khiter_t it);

void uct_dc_mlx5_destroy_dct(uct_dc_mlx5_iface_t *iface);

void uct_dc_mlx5_iface_init_version(uct_dc_mlx5_iface_t *iface, uct_md_h md);

ucs_status_t uct_dc_mlx5_iface_dci_connect(uct_dc_mlx5_iface_t *iface,
                                           uct_dc_dci_t *dci);

void uct_dc_mlx5_iface_set_ep_failed(uct_dc_mlx5_iface_t *iface,
                                     uct_dc_mlx5_ep_t *ep,
                                     struct mlx5_cqe64 *cqe,
                                     uct_ib_mlx5_txwq_t *txwq,
                                     ucs_status_t ep_status);

void uct_dc_mlx5_iface_reset_dci(uct_dc_mlx5_iface_t *iface, uint8_t dci_index);

ucs_status_t uct_dc_mlx5_iface_create_dci(uct_dc_mlx5_iface_t *iface,
                                          uint8_t dci_index, int connect,
                                          uint8_t num_dci_channels);

ucs_status_t uct_dc_mlx5_iface_resize_and_fill_dcis(uct_dc_mlx5_iface_t *iface,
                                                    uint16_t size);

/**
 * Checks whether dci pool config is present in dc_config_hash and returns 
 * the matching pool index or creates a new one
*/
ucs_status_t
uct_dc_mlx5_dci_pool_get_or_create(uct_dc_mlx5_iface_t *iface,
                                   const uct_dc_mlx5_dci_config_t *dci_config,
                                   uint8_t *pool_index_p);

uint32_t
uct_dc_mlx5_dci_config_hash(const uct_dc_mlx5_dci_config_t *dci_config);

static UCS_F_ALWAYS_INLINE uint8_t uct_dc_mlx5_is_dci_valid(const uct_dc_dci_t *dci)
{
    return dci->txwq.super.qp_num != UCT_IB_INVALID_QPN;
}

static UCS_F_ALWAYS_INLINE uct_dc_dci_t *
uct_dc_mlx5_iface_dci(uct_dc_mlx5_iface_t *iface, uint8_t dci_index)
{
    return &ucs_array_elem(&iface->tx.dcis, dci_index);
}

#if HAVE_DEVX

ucs_status_t uct_dc_mlx5_iface_devx_create_dct(uct_dc_mlx5_iface_t *iface);

ucs_status_t uct_dc_mlx5_iface_devx_set_srq_dc_params(uct_dc_mlx5_iface_t *iface);

ucs_status_t
uct_dc_mlx5_iface_devx_dci_connect(uct_dc_mlx5_iface_t *iface,
                                   uct_ib_mlx5_qp_t *qp,
                                   const uct_dc_mlx5_dci_config_t *dci_config);

#else

static UCS_F_MAYBE_UNUSED ucs_status_t
uct_dc_mlx5_iface_devx_create_dct(uct_dc_mlx5_iface_t *iface)
{
    return UCS_ERR_UNSUPPORTED;
}

static UCS_F_MAYBE_UNUSED ucs_status_t
uct_dc_mlx5_iface_devx_set_srq_dc_params(uct_dc_mlx5_iface_t *iface)
{
    return UCS_ERR_UNSUPPORTED;
}

static UCS_F_MAYBE_UNUSED ucs_status_t uct_dc_mlx5_iface_devx_dci_connect(
        uct_dc_mlx5_iface_t *iface, uct_ib_mlx5_qp_t *qp,
        const uct_dc_mlx5_dci_config_t *dci_config)
{
    return UCS_ERR_UNSUPPORTED;
}

#endif

#if IBV_HW_TM
static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_fill_ravh(struct ibv_ravh *ravh, uint32_t dct_num)
{
    ravh->sl_dct        = htobe32(dct_num);
    ravh->dc_access_key = htobe64(UCT_IB_KEY);
    ravh->reserved      = 0;
}
#endif

/* TODO:
 * use a better search algorithm (perfect hash, bsearch, hash) ???
 *
 * linear search is most probably the best way to go
 * because the number of dcis is usually small
 */
static UCS_F_ALWAYS_INLINE uint8_t
uct_dc_mlx5_iface_dci_find(uct_dc_mlx5_iface_t *iface, struct mlx5_cqe64 *cqe)
{
    uint32_t qp_num;
    int i, ndci;

    if (ucs_likely(iface->flags & UCT_DC_MLX5_IFACE_FLAG_UIDX)) {
        return cqe->srqn_uidx >> UCT_IB_UIDX_SHIFT;
    }

    qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    ndci   = ucs_array_length(&iface->tx.dcis);
    for (i = 0; i < ndci; i++) {
        if (uct_dc_mlx5_iface_dci(iface, i)->txwq.super.qp_num == qp_num) {
            return i;
        }
    }

    ucs_fatal("DCI (qpnum=%d) does not exist", qp_num);
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_has_tx_resources(uct_dc_mlx5_iface_t *iface)
{
    return !ucs_mpool_is_empty(&iface->super.super.tx.mp) &&
           (iface->super.super.tx.reads_available > 0);
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_dci_has_tx_resources(uct_dc_mlx5_iface_t *iface,
                                       uint8_t dci_index)
{
    return uct_rc_txqp_available(
                   &uct_dc_mlx5_iface_dci(iface, dci_index)->txqp) > 0;
}

/* returns pending queue of eps waiting for tx resources */
static UCS_F_ALWAYS_INLINE ucs_arbiter_t *
uct_dc_mlx5_iface_tx_waitq(uct_dc_mlx5_iface_t *iface)
{
    return &iface->super.super.tx.arbiter;
}

/* returns pending queue of eps waiting for the dci allocation */
static UCS_F_ALWAYS_INLINE ucs_arbiter_t *
uct_dc_mlx5_iface_dci_waitq(uct_dc_mlx5_iface_t *iface, uint8_t pool_index)
{
    return &iface->tx.dci_pool[pool_index].arbiter;
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_dci_has_outstanding(uct_dc_mlx5_iface_t *iface, int dci_index)
{
    uct_dc_dci_t *dci = uct_dc_mlx5_iface_dci(iface, dci_index);

    return uct_rc_txqp_available(&dci->txqp) < (int16_t)iface->tx.bb_max;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_mlx5_iface_flush_dci(uct_dc_mlx5_iface_t *iface, int dci_index)
{

    if (!uct_dc_mlx5_iface_dci_has_outstanding(iface, dci_index)) {
        return UCS_OK;
    }

    ucs_trace_poll("dci %d is not flushed %d/%d", dci_index,
                   uct_dc_mlx5_iface_dci(iface, dci_index)->txqp.available,
                   iface->tx.bb_max);
    ucs_assertv(uct_rc_txqp_unsignaled(
                        &uct_dc_mlx5_iface_dci(iface, dci_index)->txqp) == 0,
                "unsignalled send is not supported!!!");
    return UCS_INPROGRESS;
}

#endif
