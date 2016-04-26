/**
* Copyright (C) Mellanox Technologies Ltd. 2016-.  ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_IFACE_H
#define UCT_DC_IFACE_H
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>

typedef uct_ib_qpnum_t uct_dc_iface_addr_t;
typedef struct uct_dc_ep uct_dc_ep_t;

/* TODO: derive from rc config */
typedef struct uct_dc_iface_config {
    uct_rc_iface_config_t    super;
    int ndcis;
    int tx_policy;
} uct_dc_iface_config_t;

typedef struct uct_dc_iface { 
    uct_rc_iface_t super;
    struct {
        uct_rc_txqp_t *dcis;
        int ndci;
    } tx;
    struct { 
        struct ibv_exp_dct *dct;
    } rx;
    struct {
        int max_inline;
        int tx_policy;
    } config;
} uct_dc_iface_t;


UCS_CLASS_DECLARE(uct_dc_iface_t, uct_rc_iface_ops_t*, uct_pd_h, uct_worker_h,
                  const char *, unsigned, unsigned, uct_dc_iface_config_t*)

extern ucs_config_field_t uct_dc_iface_config_table[];

void uct_dc_iface_query(uct_dc_iface_t *iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_dc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

ucs_status_t uct_dc_device_query_tl_resources(uct_ib_device_t *dev,
                                              const char *tl_name, unsigned flags,
                                              uct_tl_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);

static inline int uct_dc_iface_dci_find(uct_dc_iface_t *iface, uint32_t qp_num)
{
    /* linear search is most probably the best way to go
     * because the number of dcis is usually small 
     */
    int i;

    for (i = 0; i < iface->tx.ndci; i++) {
        if (iface->tx.dcis[i].qp->qp_num == qp_num) {
            return i;
        }
    }
    ucs_fatal("DCI (qpnum=%d) does not exist", qp_num); 
}

static inline ucs_status_t uct_dc_iface_dci_get(uct_dc_iface_t *iface, uct_dc_ep_t *ep, int *dci)
{
    
    /* TODO: actual dci selection logic */
    *dci = 0;
    UCT_RC_CHECK_TXQP(&iface->super, &iface->tx.dcis[*dci]);
    return UCS_OK;
}

#define UCT_DC_CHECK_RES(_iface, _ep, _dci) \
    { \
        ucs_status_t status; \
        UCT_RC_CHECK_CQE(&(_iface)->super); \
        status = uct_dc_iface_dci_get(_iface, _ep, &_dci); \
        if (ucs_unlikely(status != UCS_OK)) { \
            return status; \
        } \
    }

#endif
