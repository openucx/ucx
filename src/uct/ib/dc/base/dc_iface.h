/**
* Copyright (C) Mellanox Technologies Ltd. 2016-.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_IFACE_H
#define UCT_DC_IFACE_H

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <uct/ib/rc/verbs/rc_verbs_common.h>
#include <uct/ib/ud/base/ud_iface_common.h>


#define UCT_DC_IFACE_MAX_DCIS   16

#define UCT_DC_IFACE_ADDR_TM_ENABLED(_addr) \
    (!!((_addr)->flags & UCT_DC_IFACE_ADDR_HW_TM))

typedef struct uct_dc_ep     uct_dc_ep_t;
typedef struct uct_dc_iface  uct_dc_iface_t;


typedef enum {
    UCT_DC_IFACE_ADDR_HW_TM   = UCS_BIT(0),
    UCT_DC_IFACE_ADDR_DC_V1   = UCS_BIT(1),
    UCT_DC_IFACE_ADDR_DC_V2   = UCS_BIT(2),
    UCT_DC_IFACE_ADDR_DC_VERS = UCT_DC_IFACE_ADDR_DC_V1 |
                                UCT_DC_IFACE_ADDR_DC_V2
} uct_dc_iface_addr_flags_t;


typedef struct uct_dc_iface_addr {
    uct_ib_uint24_t   qp_num;
    uint8_t           atomic_mr_id;
    uint8_t           flags;
} UCS_S_PACKED uct_dc_iface_addr_t;


typedef enum {
    UCT_DC_TX_POLICY_DCS,
    UCT_DC_TX_POLICY_DCS_QUOTA,
    UCT_DC_TX_POLICY_LAST
} uct_dc_tx_policty_t;


typedef struct uct_dc_iface_config {
    uct_rc_iface_config_t         super;
    uct_ud_iface_common_config_t  ud_common;
    int                           ndci;
    int                           tx_policy;
    unsigned                      quota;
} uct_dc_iface_config_t;


typedef enum {
    UCT_DC_DCI_FLAG_EP_CANCELED         = UCS_BIT(0),
    UCT_DC_DCI_FLAG_EP_DESTROYED        = UCS_BIT(1)
} uct_dc_dci_state_t;


typedef struct uct_dc_dci {
    uct_rc_txqp_t                 txqp; /* DCI qp */
    uct_dc_ep_t                   *ep;  /* points to an endpoint that currently
                                           owns the dci. Relevant only for dcs
                                           and dcs quota policies. */
#if ENABLE_ASSERT
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
    uct_rc_fc_request_t           super;
    uct_dc_fc_sender_data_t       sender;
    uint32_t                      dct_num;

    /* Lid can be stored either in BE or in LE order. The endianess depends
     * on the transport (BE for mlx5 and LE for dc verbs) */
    uint16_t                      lid;
} uct_dc_fc_request_t;


typedef struct uct_dc_iface_ops {
    uct_rc_iface_ops_t            super;
    ucs_status_t                  (*reset_dci)(uct_dc_iface_t *iface, int dci);
} uct_dc_iface_ops_t;


struct uct_dc_iface {
    uct_rc_iface_t                super;
    struct {
        uct_dc_dci_t              dcis[UCT_DC_IFACE_MAX_DCIS]; /* Array of dcis */
        uint8_t                   ndci;                        /* Number of DCIs */
        uct_dc_tx_policty_t       policy;                      /* dci selection algorithm */
        int16_t                   available_quota;             /* if available tx is lower, let
                                                                  another endpoint use the dci */

        /* LIFO is only relevant for dcs allocation policy */
        uint8_t                   stack_top;                   /* dci stack top */
        uint8_t                   dcis_stack[UCT_DC_IFACE_MAX_DCIS];  /* LIFO of indexes of available dcis */

        ucs_arbiter_t             dci_arbiter;

        /* Used to send grant messages for all peers */
        uct_dc_ep_t               *fc_ep;

        /* List of destroyed endpoints waiting for credit grant */
        ucs_list_link_t           gc_list;
    } tx;

    struct {
        struct ibv_exp_dct        *dct;
    } rx;

    uint8_t                       version_flag;
};


UCS_CLASS_DECLARE(uct_dc_iface_t, uct_dc_iface_ops_t*, uct_md_h,
                  uct_worker_h, const uct_iface_params_t*, unsigned,
                  uct_dc_iface_config_t*, int)

extern ucs_config_field_t uct_dc_iface_config_table[];

ucs_status_t uct_dc_iface_create_dct(uct_dc_iface_t *iface);

ucs_status_t uct_dc_iface_query(uct_dc_iface_t *iface, uct_iface_attr_t *iface_attr,
                                size_t put_max_short, size_t max_inline,
                                size_t am_max_hdr, size_t am_max_iov);

int uct_dc_iface_is_reachable(const uct_iface_h tl_iface,
                              const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr);

ucs_status_t uct_dc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

ucs_status_t uct_dc_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);

ucs_status_t uct_dc_iface_flush(uct_iface_h tl_iface, unsigned flags, uct_completion_t *comp);

void uct_dc_iface_set_quota(uct_dc_iface_t *iface, uct_dc_iface_config_t *config);

ucs_status_t uct_dc_iface_init_fc_ep(uct_dc_iface_t *iface);

void uct_dc_iface_cleanup_fc_ep(uct_dc_iface_t *iface);

ucs_status_t uct_dc_iface_fc_grant(uct_pending_req_t *self);

ucs_status_t uct_dc_iface_fc_handler(uct_rc_iface_t *rc_iface, unsigned qp_num,
                                     uct_rc_hdr_t *hdr, unsigned length,
                                     uint32_t imm_data, uint16_t lid, unsigned flags);

ucs_status_t uct_dc_handle_failure(uct_ib_iface_t *ib_iface, uint32_t qp_num,
                                   ucs_status_t status);

#if IBV_EXP_HW_TM_DC
void uct_dc_iface_fill_xrq_init_attrs(uct_rc_iface_t *rc_iface,
                                      struct ibv_exp_create_srq_attr *srq_attr,
                                      struct ibv_exp_srq_dc_offload_params *dc_op);

static UCS_F_ALWAYS_INLINE void
uct_dc_iface_fill_ravh(struct ibv_exp_tmh_ravh *ravh, uint32_t dct_num)
{
    ravh->sl_dct        = htobe32(dct_num);
    ravh->dc_access_key = htobe64(UCT_IB_KEY);
    ravh->reserved      = 0;
}
#endif

/* TODO:
 * use a better seach algorithm (perfect hash, bsearch, hash) ???
 *
 * linear search is most probably the best way to go
 * because the number of dcis is usually small
 */
static inline uint8_t uct_dc_iface_dci_find(uct_dc_iface_t *iface, uint32_t qp_num)
{
    uct_dc_dci_t *dcis = iface->tx.dcis;
    int i, ndci = iface->tx.ndci;

    for (i = 0; i < ndci; i++) {
        if (dcis[i].txqp.qp->qp_num == qp_num) {
            return i;
        }
    }
    ucs_fatal("DCI (qpnum=%d) does not exist", qp_num);
}

static inline int uct_dc_iface_dci_has_tx_resources(uct_dc_iface_t *iface, uint8_t dci)
{
    return uct_rc_txqp_available(&iface->tx.dcis[dci].txqp) > 0;
}

/* returns pending queue of eps waiting for tx resources */
static inline ucs_arbiter_t *uct_dc_iface_tx_waitq(uct_dc_iface_t *iface)
{
    return &iface->tx.dci_arbiter;
}

/* returns pending queue of eps waiting for the dci allocation */
static inline ucs_arbiter_t *uct_dc_iface_dci_waitq(uct_dc_iface_t *iface)
{
    return &iface->super.tx.arbiter;
}

static inline int
uct_dc_iface_dci_has_outstanding(uct_dc_iface_t *iface, int dci)
{
    uct_rc_txqp_t *txqp;

    txqp = &iface->tx.dcis[dci].txqp;
    return uct_rc_txqp_available(txqp) < (int16_t)iface->super.config.tx_qp_len;
}

static inline ucs_status_t uct_dc_iface_flush_dci(uct_dc_iface_t *iface, int dci)
{

    if (!uct_dc_iface_dci_has_outstanding(iface, dci)) {
        return UCS_OK;
    }
    ucs_trace_poll("dci %d is not flushed %d/%d", dci,
                   iface->tx.dcis[dci].txqp.available,
                   iface->super.config.tx_qp_len);
    ucs_assertv(uct_rc_txqp_unsignaled(&iface->tx.dcis[dci].txqp) == 0,
                "unsignalled send is not supported!!!");
    return UCS_INPROGRESS;
}

#endif
