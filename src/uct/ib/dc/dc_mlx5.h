/**
* Copyright (C) Mellanox Technologies Ltd. 2016-2018.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_IFACE_H
#define UCT_DC_IFACE_H

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <uct/ib/rc/verbs/rc_verbs.h>
#include <uct/ib/rc/accel/rc_mlx5_common.h>
#include <uct/ib/ud/base/ud_iface_common.h>
#include <uct/ib/ud/accel/ud_mlx5_common.h>
#include <ucs/debug/assert.h>
#include <ucs/datastruct/bitmap.h>


/*
 * HW tag matching
 */
#if IBV_HW_TM
#  if HAVE_INFINIBAND_TM_TYPES_H
/* upstream tm_types.h doesn't provide RAVH header */
struct ibv_ravh {
    uint32_t    sl_dct;
    uint32_t    reserved;    /* must be zero */
    uint64_t    dc_access_key;
};
#  else
#    define ibv_ravh            ibv_exp_tmh_ravh
#  endif
#  define UCT_DC_RNDV_HDR_LEN   (sizeof(struct ibv_rvh) + \
                                 sizeof(struct ibv_ravh))
#else
#  define UCT_DC_RNDV_HDR_LEN   0
#endif

#define UCT_DC_MLX5_KEEPALIVE_NUM_DCIS  1
#define UCT_DC_MLX5_IFACE_MAX_DCI_POOLS 8

#define UCT_DC_MLX5_IFACE_ADDR_TM_ENABLED(_addr) \
    (!!((_addr)->flags & UCT_DC_MLX5_IFACE_ADDR_HW_TM))

#define UCT_DC_MLX5_IFACE_TXQP_DCI_GET(_iface, _dci, _txqp, _txwq) \
    { \
        ucs_assert(_dci != UCT_DC_MLX5_EP_NO_DCI); \
        _txqp = &(_iface)->tx.dcis[_dci].txqp; \
        _txwq = &(_iface)->tx.dcis[_dci].txwq; \
    }

/**
 * Set iface config flag for enabling full handshake on DCI/DCT,
 * according to user configuration. Fail if the user requests to
 * force full-handlshake, while the HW does not support it.
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
    UCT_DC_MLX5_IFACE_ADDR_HW_TM   = UCS_BIT(0),
    UCT_DC_MLX5_IFACE_ADDR_DC_V1   = UCS_BIT(1),
    UCT_DC_MLX5_IFACE_ADDR_DC_V2   = UCS_BIT(2),
    UCT_DC_MLX5_IFACE_ADDR_DC_VERS = UCT_DC_MLX5_IFACE_ADDR_DC_V1 |
                                     UCT_DC_MLX5_IFACE_ADDR_DC_V2
} uct_dc_mlx5_iface_addr_flags_t;


typedef enum {
    /** Keepalive dci is created */
    UCT_DC_MLX5_IFACE_FLAG_KEEPALIVE                = UCS_BIT(0),

    /** Enable full handshake for keepalive DCI */
    UCT_DC_MLX5_IFACE_FLAG_KEEPALIVE_FULL_HANDSHAKE = UCS_BIT(1),

    /** uidx is set to dci idx */
    UCT_DC_MLX5_IFACE_FLAG_UIDX                     = UCS_BIT(2),

    /** Flow control endpoint is using a DCI in error state */
    UCT_DC_MLX5_IFACE_FLAG_FC_EP_FAILED             = UCS_BIT(3),

    /** Enable full handshake for DCI */
    UCT_DC_MLX5_IFACE_FLAG_DCI_FULL_HANDSHAKE       = UCS_BIT(4),

    /** Enable full handshake for DCT */
    UCT_DC_MLX5_IFACE_FLAG_DCT_FULL_HANDSHAKE       = UCS_BIT(5),

    /** Disable PUT capability (RDMA_WRITE) */
    UCT_DC_MLX5_IFACE_FLAG_DISABLE_PUT              = UCS_BIT(6)
} uct_dc_mlx5_iface_flags_t;


typedef struct uct_dc_mlx5_iface_addr {
    uct_ib_uint24_t   qp_num;
    uint8_t           atomic_mr_id;
    uint8_t           flags;
} UCS_S_PACKED uct_dc_mlx5_iface_addr_t;


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
 *    - dci is choosen by random() % ndci
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
    UCT_DC_TX_POLICY_DCS,
    UCT_DC_TX_POLICY_DCS_QUOTA,
    UCT_DC_TX_POLICY_RAND,
    UCT_DC_TX_POLICY_LAST
} uct_dc_tx_policy_t;


typedef struct uct_dc_mlx5_iface_config {
    uct_rc_iface_common_config_t        super;
    uct_rc_mlx5_iface_common_config_t   rc_mlx5_common;
    uct_ud_iface_common_config_t        ud_common;
    int                                 ndci;
    int                                 tx_policy;
    ucs_ternary_auto_value_t            dci_full_handshake;
    ucs_ternary_auto_value_t            dci_ka_full_handshake;
    ucs_ternary_auto_value_t            dct_full_handshake;
    unsigned                            quota;
    unsigned                            rand_seed;
    ucs_time_t                          fc_hard_req_timeout;
    uct_ud_mlx5_iface_common_config_t   mlx5_ud;
} uct_dc_mlx5_iface_config_t;


typedef void (*uct_dc_dci_handle_failure_func_t)(uct_dc_mlx5_iface_t *iface,
                                                 struct mlx5_cqe64 *cqe,
                                                 uint8_t dci_index,
                                                 ucs_status_t status);


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

    /* Lid can be stored either in BE or in LE order. The endianess depends
     * on the transport (BE for mlx5 and LE for dc verbs) */
    uint16_t                      lid;
} uct_dc_fc_request_t;


typedef struct uct_dc_mlx5_ep_fc_entry {
    uint64_t   seq; /* Sequence number in FC_HARD_REQ's sender data */
    ucs_time_t send_time; /* Last time FC_HARD_REQ was sent */
} uct_dc_mlx5_ep_fc_entry_t;


KHASH_MAP_INIT_INT64(uct_dc_mlx5_fc_hash, uct_dc_mlx5_ep_fc_entry_t);


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
 * Overall count of DCI's to relase and allocated DCI's could not be more than
 * ndci and these stacks are not intersected
 */
typedef struct {
    int8_t        stack_top;         /* dci stack top */
    uint8_t       *stack;            /* LIFO of indexes of available dcis */
    ucs_arbiter_t arbiter;           /* queue of requests waiting for DCI */
    int8_t        release_stack_top; /* releasing dci's stack,
                                        points to last DCI to release
                                        or -1 if no DCI's to release */
} uct_dc_mlx5_dci_pool_t;


struct uct_dc_mlx5_iface {
    uct_rc_mlx5_iface_common_t    super;
    struct {
        /* Array of dcis */
        uct_dc_dci_t              *dcis;

        uint8_t                   ndci;                        /* Number of DCIs */

        /* LIFO is only relevant for dcs allocation policy */
        uct_dc_mlx5_dci_pool_t    dci_pool[UCT_DC_MLX5_IFACE_MAX_DCI_POOLS];
        uint8_t                   num_dci_pools;

        uint8_t                   policy;                      /* dci selection algorithm */
        int16_t                   available_quota;             /* if available tx is lower, let
                                                                  another endpoint use the dci */
        /* DCI max elements */
        unsigned                  bb_max;

        /* Used to send grant messages for all peers */
        uct_dc_mlx5_ep_t          *fc_ep;

        /* Hash of expected FC grants */
        khash_t(uct_dc_mlx5_fc_hash) fc_hash;

        /* Sequence number of expected FC grants */
        uint64_t                  fc_seq;

        /* Timeout for sending FC_HARD_REQ when FC window is empty */
        ucs_time_t                fc_hard_req_timeout;

        /* Next time when FC_HARD_REQ operations should be resent */
        ucs_time_t                fc_hard_req_resend_time;

        /* Callback ID of FC_HARD_REQ resend operation */
        uct_worker_cb_id_t        fc_hard_req_progress_cb_id;

        /* Seed used for random dci allocation */
        unsigned                  rand_seed;

        ucs_arbiter_callback_t    pend_cb;

        uct_worker_cb_id_t        dci_release_prog_id;

        uint8_t                   dci_pool_release_bitmap;
    } tx;

    struct {
        uct_ib_mlx5_qp_t          dct;
    } rx;

    uint8_t                       version_flag;

    /* iface flags, see uct_dc_mlx5_iface_flags_t */
    uint8_t                       flags;

    uint8_t                       keepalive_dci;

    uct_ud_mlx5_iface_common_t    ud_common;
};


extern ucs_config_field_t uct_dc_mlx5_iface_config_table[];

ucs_status_t
uct_dc_mlx5_iface_create_dct(uct_dc_mlx5_iface_t *iface,
                             const uct_dc_mlx5_iface_config_t *config);

int uct_dc_mlx5_iface_is_reachable(const uct_iface_h tl_iface,
                                   const uct_device_addr_t *dev_addr,
                                   const uct_iface_addr_t *iface_addr);

ucs_status_t uct_dc_mlx5_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

ucs_status_t uct_dc_mlx5_iface_flush(uct_iface_h tl_iface, unsigned flags, uct_completion_t *comp);

void uct_dc_mlx5_iface_set_quota(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_iface_config_t *config);

ucs_status_t uct_dc_mlx5_iface_init_fc_ep(uct_dc_mlx5_iface_t *iface);

ucs_status_t uct_dc_mlx5_iface_fc_grant(uct_pending_req_t *self);

ucs_status_t uct_dc_mlx5_iface_fc_handler(uct_rc_iface_t *rc_iface, unsigned qp_num,
                                          uct_rc_hdr_t *hdr, unsigned length,
                                          uint32_t imm_data, uint16_t lid, unsigned flags);

void uct_dc_mlx5_fc_entry_iter_del(uct_dc_mlx5_iface_t *iface, khiter_t it);

void uct_dc_mlx5_destroy_dct(uct_dc_mlx5_iface_t *iface);

void uct_dc_mlx5_iface_init_version(uct_dc_mlx5_iface_t *iface, uct_md_h md);

ucs_status_t uct_dc_mlx5_iface_dci_connect(uct_dc_mlx5_iface_t *iface,
                                           uct_dc_dci_t *dci);

ucs_status_t uct_dc_mlx5_iface_keepalive_init(uct_dc_mlx5_iface_t *iface);

void uct_dc_mlx5_iface_set_ep_failed(uct_dc_mlx5_iface_t *iface,
                                     uct_dc_mlx5_ep_t *ep,
                                     struct mlx5_cqe64 *cqe,
                                     uct_ib_mlx5_txwq_t *txwq,
                                     ucs_status_t ep_status);

void uct_dc_mlx5_iface_reset_dci(uct_dc_mlx5_iface_t *iface, uint8_t dci_index);

#if HAVE_DEVX

ucs_status_t uct_dc_mlx5_iface_devx_create_dct(uct_dc_mlx5_iface_t *iface);

ucs_status_t uct_dc_mlx5_iface_devx_set_srq_dc_params(uct_dc_mlx5_iface_t *iface);

ucs_status_t uct_dc_mlx5_iface_devx_dci_connect(uct_dc_mlx5_iface_t *iface,
                                                uct_ib_mlx5_qp_t *qp,
                                                uint8_t path_index);

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
        uct_dc_mlx5_iface_t *iface, uct_ib_mlx5_qp_t *qp, uint8_t path_index)
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

static UCS_F_ALWAYS_INLINE uint8_t
uct_dc_mlx5_iface_total_ndci(uct_dc_mlx5_iface_t *iface)
{
    return (iface->tx.ndci * iface->tx.num_dci_pools) +
        ((iface->flags & UCT_DC_MLX5_IFACE_FLAG_KEEPALIVE) ?
         UCT_DC_MLX5_KEEPALIVE_NUM_DCIS : 0);
}

/* TODO:
 * use a better seach algorithm (perfect hash, bsearch, hash) ???
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
    ndci   = uct_dc_mlx5_iface_total_ndci(iface);
    for (i = 0; i < ndci; i++) {
        if (iface->tx.dcis[i].txwq.super.qp_num == qp_num) {
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
    return uct_rc_txqp_available(&iface->tx.dcis[dci_index].txqp) > 0;
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
    uct_rc_txqp_t *txqp;

    txqp = &iface->tx.dcis[dci_index].txqp;
    return uct_rc_txqp_available(txqp) < (int16_t)iface->tx.bb_max;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_mlx5_iface_flush_dci(uct_dc_mlx5_iface_t *iface, int dci_index)
{

    if (!uct_dc_mlx5_iface_dci_has_outstanding(iface, dci_index)) {
        return UCS_OK;
    }

    ucs_trace_poll("dci %d is not flushed %d/%d", dci_index,
                   iface->tx.dcis[dci_index].txqp.available, iface->tx.bb_max);
    ucs_assertv(uct_rc_txqp_unsignaled(&iface->tx.dcis[dci_index].txqp) == 0,
                "unsignalled send is not supported!!!");
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_is_dci_keepalive(uct_dc_mlx5_iface_t *iface, int dci_index)
{
    return dci_index == iface->keepalive_dci;
}

#endif
