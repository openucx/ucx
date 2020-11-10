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

#define UCT_DC_MLX5_IFACE_MAX_USER_DCIS 15
#define UCT_DC_MLX5_KEEPALIVE_NUM_DCIS  1
#define UCT_DC_MLX5_IFACE_MAX_DCIS      (UCT_DC_MLX5_IFACE_MAX_USER_DCIS + \
                                         UCT_DC_MLX5_KEEPALIVE_NUM_DCIS)

#define UCT_DC_MLX5_IFACE_ADDR_TM_ENABLED(_addr) \
    (!!((_addr)->flags & UCT_DC_MLX5_IFACE_ADDR_HW_TM))

#define UCT_DC_MLX5_IFACE_TXQP_DCI_GET(_iface, _dci, _txqp, _txwq) \
    { \
        _txqp = &(_iface)->tx.dcis[_dci].txqp; \
        _txwq = &(_iface)->tx.dcis[_dci].txwq; \
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
    UCT_DC_MLX5_IFACE_FLAG_KEEPALIVE = UCS_BIT(0) /**< keepalive dci is created */
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
 *    - ep keeps using dci as long as it has oustanding sends
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
    unsigned                            quota;
    unsigned                            rand_seed;
    uct_ud_mlx5_iface_common_config_t   mlx5_ud;
} uct_dc_mlx5_iface_config_t;


typedef void (*uct_dc_dci_handle_failure_func_t)(uct_dc_mlx5_iface_t *iface,
                                                 struct mlx5_cqe64 *cqe,
                                                 uint8_t dci,
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
#if UCS_ENABLE_ASSERT
    uint8_t                       flags; /* debug state, @ref uct_dc_dci_state_t */
#endif
} uct_dc_dci_t;


typedef struct uct_dc_fc_sender_data {
    uint64_t                      ep;
    struct {
        int                       is_global;
        union ibv_gid             gid;
    } UCS_S_PACKED global;
} UCS_S_PACKED uct_dc_fc_sender_data_t;

typedef struct uct_dc_fc_request {
    uct_rc_pending_req_t          super;
    uct_dc_fc_sender_data_t       sender;
    uint32_t                      dct_num;

    /* Lid can be stored either in BE or in LE order. The endianess depends
     * on the transport (BE for mlx5 and LE for dc verbs) */
    uint16_t                      lid;
} uct_dc_fc_request_t;


struct uct_dc_mlx5_iface {
    uct_rc_mlx5_iface_common_t    super;
    struct {
        /* Array of dcis */
        uct_dc_dci_t              dcis[UCT_DC_MLX5_IFACE_MAX_DCIS];

        uint8_t                   ndci;                        /* Number of DCIs */
        uct_dc_tx_policy_t        policy;                      /* dci selection algorithm */
        int16_t                   available_quota;             /* if available tx is lower, let
                                                                  another endpoint use the dci */

        /* LIFO is only relevant for dcs allocation policy */
        uint8_t                   stack_top;                   /* dci stack top */
        uint8_t                   dcis_stack[UCT_DC_MLX5_IFACE_MAX_DCIS];  /* LIFO of indexes of available dcis */

        ucs_arbiter_t             dci_arbiter;

        /* Used to send grant messages for all peers */
        uct_dc_mlx5_ep_t          *fc_ep;

        /* List of destroyed endpoints waiting for credit grant */
        ucs_list_link_t           gc_list;

        /* Number of expected FC grants */
        unsigned                  fc_grants;

        /* Seed used for random dci allocation */
        unsigned                  rand_seed;

        ucs_arbiter_callback_t    pend_cb;
    } tx;

    struct {
        uct_ib_mlx5_qp_t          dct;
    } rx;

    uint8_t                       version_flag;

    /* iface flags, see uct_dc_mlx5_iface_flags_t */
    uint8_t                       flags;

    uct_ud_mlx5_iface_common_t    ud_common;
};


extern ucs_config_field_t uct_dc_mlx5_iface_config_table[];

ucs_status_t uct_dc_mlx5_iface_create_dct(uct_dc_mlx5_iface_t *iface);

int uct_dc_mlx5_iface_is_reachable(const uct_iface_h tl_iface,
                                   const uct_device_addr_t *dev_addr,
                                   const uct_iface_addr_t *iface_addr);

ucs_status_t uct_dc_mlx5_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

ucs_status_t uct_dc_mlx5_iface_flush(uct_iface_h tl_iface, unsigned flags, uct_completion_t *comp);

void uct_dc_mlx5_iface_set_quota(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_iface_config_t *config);

ucs_status_t uct_dc_mlx5_iface_init_fc_ep(uct_dc_mlx5_iface_t *iface);

void uct_dc_mlx5_iface_cleanup_fc_ep(uct_dc_mlx5_iface_t *iface);

ucs_status_t uct_dc_mlx5_iface_fc_grant(uct_pending_req_t *self);

ucs_status_t uct_dc_mlx5_iface_fc_handler(uct_rc_iface_t *rc_iface, unsigned qp_num,
                                          uct_rc_hdr_t *hdr, unsigned length,
                                          uint32_t imm_data, uint16_t lid, unsigned flags);

void uct_dc_mlx5_destroy_dct(uct_dc_mlx5_iface_t *iface);

void uct_dc_mlx5_iface_init_version(uct_dc_mlx5_iface_t *iface, uct_md_h md);

ucs_status_t uct_dc_mlx5_iface_dci_connect(uct_dc_mlx5_iface_t *iface,
                                           uct_dc_dci_t *dci);

void uct_dc_mlx5_iface_dcis_destroy(uct_dc_mlx5_iface_t *iface, int max);

ucs_status_t uct_dc_mlx5_iface_keepalive_init(uct_dc_mlx5_iface_t *iface);

void uct_dc_mlx5_iface_set_ep_failed(uct_dc_mlx5_iface_t *iface,
                                     uct_dc_mlx5_ep_t *ep,
                                     struct mlx5_cqe64 *cqe,
                                     uct_ib_mlx5_txwq_t *txwq,
                                     ucs_status_t ep_status);

void uct_dc_mlx5_iface_reset_dci(uct_dc_mlx5_iface_t *iface,
                                 uint8_t dci,
                                 ucs_status_t ep_status);

#if HAVE_DEVX

ucs_status_t uct_dc_mlx5_iface_devx_create_dct(uct_dc_mlx5_iface_t *iface);

ucs_status_t uct_dc_mlx5_iface_devx_set_srq_dc_params(uct_dc_mlx5_iface_t *iface);

ucs_status_t uct_dc_mlx5_iface_devx_dci_connect(uct_dc_mlx5_iface_t *iface,
                                                uct_ib_mlx5_qp_t *qp);

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

static UCS_F_MAYBE_UNUSED ucs_status_t
uct_dc_mlx5_iface_devx_dci_connect(uct_dc_mlx5_iface_t *iface,
                                   uct_ib_mlx5_qp_t *qp)
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

static inline uint8_t
uct_dc_mlx5_iface_total_ndci(uct_dc_mlx5_iface_t *iface)
{
    return iface->tx.ndci + ((iface->flags & UCT_DC_MLX5_IFACE_FLAG_KEEPALIVE) ?
                             UCT_DC_MLX5_KEEPALIVE_NUM_DCIS : 0);
}

/* TODO:
 * use a better seach algorithm (perfect hash, bsearch, hash) ???
 *
 * linear search is most probably the best way to go
 * because the number of dcis is usually small
 */
static inline uint8_t uct_dc_mlx5_iface_dci_find(uct_dc_mlx5_iface_t *iface, uint32_t qp_num)
{
    uct_dc_dci_t *dcis = iface->tx.dcis;
    int i, ndci = uct_dc_mlx5_iface_total_ndci(iface);

    for (i = 0; i < ndci; i++) {
        if (dcis[i].txwq.super.qp_num == qp_num) {
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

static inline int uct_dc_mlx5_iface_dci_has_tx_resources(uct_dc_mlx5_iface_t *iface, uint8_t dci)
{
    return uct_rc_txqp_available(&iface->tx.dcis[dci].txqp) > 0;
}

/* returns pending queue of eps waiting for tx resources */
static inline ucs_arbiter_t *uct_dc_mlx5_iface_tx_waitq(uct_dc_mlx5_iface_t *iface)
{
    return &iface->tx.dci_arbiter;
}

/* returns pending queue of eps waiting for the dci allocation */
static inline ucs_arbiter_t *uct_dc_mlx5_iface_dci_waitq(uct_dc_mlx5_iface_t *iface)
{
    return &iface->super.super.tx.arbiter;
}

static inline int
uct_dc_mlx5_iface_dci_has_outstanding(uct_dc_mlx5_iface_t *iface, int dci)
{
    uct_rc_txqp_t *txqp;

    txqp = &iface->tx.dcis[dci].txqp;
    return uct_rc_txqp_available(txqp) < (int16_t)iface->super.super.config.tx_qp_len;
}

static inline ucs_status_t uct_dc_mlx5_iface_flush_dci(uct_dc_mlx5_iface_t *iface, int dci)
{

    if (!uct_dc_mlx5_iface_dci_has_outstanding(iface, dci)) {
        return UCS_OK;
    }
    ucs_trace_poll("dci %d is not flushed %d/%d", dci,
                   iface->tx.dcis[dci].txqp.available,
                   iface->super.super.config.tx_qp_len);
    ucs_assertv(uct_rc_txqp_unsignaled(&iface->tx.dcis[dci].txqp) == 0,
                "unsignalled send is not supported!!!");
    return UCS_INPROGRESS;
}

static inline int
uct_dc_mlx5_iface_is_dci_keepalive(uct_dc_mlx5_iface_t *iface, int dci)
{
    ucs_assert(dci < uct_dc_mlx5_iface_total_ndci(iface));

    return dci == iface->tx.ndci;
}

#endif
