/**
* Copyright (C) Mellanox Technologies Ltd. 2016-.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_IFACE_H
#define UCT_DC_IFACE_H
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>

#include <uct/ib/rc/verbs/rc_verbs_common.h>

typedef uct_ib_qpnum_t uct_dc_iface_addr_t;
typedef struct uct_dc_ep uct_dc_ep_t;

typedef enum {
    UCT_DC_TX_POLICY_DCS,
    UCT_DC_TX_POLICY_DCS_QUOTA,
    UCT_DC_TX_POLICY_LAST
} uct_dc_tx_policty_t;

typedef struct uct_dc_iface_config {
    /* work around to do multiple inheritance:
     * dc_verbs needs both dc_iface_config and verbs_common_iface config 
     */
    uct_rc_verbs_iface_config_t   super;
    int                           ndci;
    int                           tx_policy;
    int                           max_inline;
} uct_dc_iface_config_t;

typedef struct uct_dc_dci {
    uct_rc_txqp_t   txqp;       /* DCI qp */
    uct_dc_ep_t     *ep;        /* points to an endpoint that currently own
                                   the dci. Relevant only for dcs and dcs quota policies */
} uct_dc_dci_t;

typedef struct uct_dc_iface { 
    uct_rc_iface_t super;
    struct {
        uct_dc_dci_t  *dcis;        /* array of dcis. dcis[ndci] */
        /* LIFO is only relevant for dcs allocation policy  */
        uint8_t       *dcis_stack;  /* LIFO of indexes of available dcis */
        uint8_t       stack_top;    /* dci stack top */
        uint8_t       ndci;         /* Number of DCIs */
        uint8_t       policy;       /* dci selection algorithm */
        ucs_arbiter_t dci_arbiter;   
    } tx;
    struct { 
        struct ibv_exp_dct *dct;    
    } rx;
} uct_dc_iface_t;


UCS_CLASS_DECLARE(uct_dc_iface_t, uct_rc_iface_ops_t*, uct_md_h, uct_worker_h,
                  const char *, unsigned, unsigned, uct_dc_iface_config_t*)

extern ucs_config_field_t uct_dc_iface_config_table[];

void uct_dc_iface_query(uct_dc_iface_t *iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_dc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

ucs_status_t uct_dc_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);

/* TODO:
 * use a better seach algorithm (perfect hash, bsearch, hash) ???
 *
 * linear search is most probably the best way to go
 * because the number of dcis is usually small 
 */
static inline uint8_t uct_dc_iface_dci_find(uct_dc_iface_t *iface, uint32_t qp_num)
{
    int i;

    for (i = 0; i < iface->tx.ndci; i++) {
        if (iface->tx.dcis[i].txqp.qp->qp_num == qp_num) {
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
#endif
