/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "srd_ep.h"
#include "srd_iface.h"

#include <ucs/arch/atomic.h>


static void uct_srd_ep_set_dest_ep_id(uct_srd_ep_t *ep, uint32_t dest_id)
{
    ucs_assert(dest_id != UCT_SRD_EP_NULL_ID);
    ep->dest_ep_id = dest_id;
    ep->flags     |= UCT_SRD_EP_FLAG_CONNECTED;
}

static UCS_CLASS_INIT_FUNC(uct_srd_ep_t, const uct_ep_params_t *params)
{
    uct_srd_iface_t *iface = ucs_derived_of(params->iface, uct_srd_iface_t);

    memset(self, 0, sizeof(*self));
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    self->dest_ep_id    = UCT_SRD_EP_NULL_ID;
    self->path_index    = UCT_EP_PARAMS_GET_PATH_INDEX(params);
    self->tx.psn        = UCT_SRD_INITIAL_PSN;
    self->rx_creq_count = 0;
    ucs_frag_list_init(self->tx.psn - 1, &self->rx.ooo_pkts,
                       -1 UCS_STATS_ARG(self->super.stats));

    uct_srd_iface_add_ep(iface, self);
    ucs_debug("created ep ep=%p iface=%p ep_id=%d", self, iface, self->ep_id);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_srd_ep_t)
{
    uct_srd_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                            uct_srd_iface_t);

    uct_srd_iface_remove_ep(iface, self);
    ucs_frag_list_cleanup(&self->rx.ooo_pkts);
}

UCS_CLASS_DEFINE(uct_srd_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);

ucs_status_t
uct_srd_ep_connect_to_ep_v2(uct_ep_h tl_ep, const uct_device_addr_t *dev_addr,
                            const uct_ep_addr_t *uct_ep_addr,
                            const uct_ep_connect_to_ep_params_t *params)
{
    uct_srd_ep_t *ep                  = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface            = ucs_derived_of(ep->super.super.iface,
                                                       uct_srd_iface_t);
    const uct_ib_address_t *ib_addr   = (const uct_ib_address_t*)dev_addr;
    const uct_srd_ep_addr_t *ep_addr  = (const uct_srd_ep_addr_t*)uct_ep_addr;
    uct_ib_device_t UCS_V_UNUSED *dev = uct_ib_iface_device(&iface->super);
    char buf[128];

    ucs_assert_always(ep->dest_ep_id == UCT_SRD_EP_NULL_ID);
    ucs_trace_func("");

    uct_srd_ep_set_dest_ep_id(ep, uct_ib_unpack_uint24(ep_addr->ep_id));

    ucs_debug(UCT_IB_IFACE_FMT " slid %d qpn 0x%x epid %u connected to %s "
                               "qpn 0x%x epid %u",
              UCT_IB_IFACE_ARG(&iface->super),
              dev->port_attr[iface->super.config.port_num - dev->first_port].lid,
              iface->qp->qp_num, ep->ep_id,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(ep_addr->iface_addr.qp_num), ep->dest_ep_id);

    return uct_srd_iface_unpack_peer_address(iface, ib_addr,
                                             &ep_addr->iface_addr,
                                             ep->path_index, &ep->peer_address);
}

static ucs_status_t
uct_srd_ep_connect_to_iface(uct_srd_ep_t *ep, const uct_ib_address_t *ib_addr,
                            const uct_srd_iface_addr_t *if_addr)
{
    uct_srd_iface_t *iface            = ucs_derived_of(ep->super.super.iface,
                                                       uct_srd_iface_t);
    uct_ib_device_t UCS_V_UNUSED *dev = uct_ib_iface_device(&iface->super);
    char buf[128];

    ucs_debug(UCT_IB_IFACE_FMT " lid %d qpn 0x%x epid %u ep %p connected to "
                               "IFACE %s qpn 0x%x",
              UCT_IB_IFACE_ARG(&iface->super),
              dev->port_attr[iface->super.config.port_num - dev->first_port].lid,
              iface->qp->qp_num, ep->ep_id, ep,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(if_addr->qp_num));

    return UCS_OK;
}

ucs_status_t uct_srd_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_srd_ep_t *ep           = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface     = ucs_derived_of(ep->super.super.iface,
                                                uct_srd_iface_t);
    uct_srd_ep_addr_t *ep_addr = (uct_srd_ep_addr_t*)addr;

    uct_ib_pack_uint24(ep_addr->iface_addr.qp_num, iface->qp->qp_num);
    uct_ib_pack_uint24(ep_addr->ep_id, ep->ep_id);
    return UCS_OK;
}

/* FIXME: Replace by CEP connection management */
static uint64_t uct_srd_ep_conn_sn;

static ucs_status_t
uct_srd_ep_create_connected(const uct_ep_params_t *ep_params,
                            uct_ep_h *new_ep_p)
{
    uct_srd_iface_t *iface              = ucs_derived_of(ep_params->iface,
                                                         uct_srd_iface_t);
    const uct_ib_address_t *ib_addr     = (const uct_ib_address_t*)
                                                  ep_params->dev_addr;
    const uct_srd_iface_addr_t *if_addr = (const uct_srd_iface_addr_t*)
                                                  ep_params->iface_addr;
    uct_ep_params_t params;
    ucs_status_t status;
    uct_srd_ep_t *ep;
    uct_ep_h new_ep_h;

    *new_ep_p = NULL;

    /* First create endpoint */
    params.field_mask = UCT_EP_PARAM_FIELD_IFACE |
                        UCT_EP_PARAM_FIELD_PATH_INDEX;
    params.iface      = &iface->super.super.super;
    params.path_index = UCT_EP_PARAMS_GET_PATH_INDEX(ep_params);

    status = uct_srd_ep_create(&params, &new_ep_h);
    if (status != UCS_OK) {
        return status;
    }

    ep          = ucs_derived_of(new_ep_h, uct_srd_ep_t);
    ep->conn_sn = ucs_atomic_fadd64(&uct_srd_ep_conn_sn, 1);

    /* Connect it to the interface */
    status = uct_srd_ep_connect_to_iface(ep, ib_addr, if_addr);
    if (status != UCS_OK) {
        goto err_ep_destroy;
    }

    /* Generate peer address */
    status = uct_srd_iface_unpack_peer_address(iface, ib_addr, if_addr,
                                               ep->path_index,
                                               &ep->peer_address);
    if (status != UCS_OK) {
        goto err_ep_destroy;
    }

    *new_ep_p = &ep->super.super;
    return status;

err_ep_destroy:
    uct_ep_destroy(&ep->super.super);
    return status;
}

ucs_status_t uct_srd_ep_create(const uct_ep_params_t *params, uct_ep_h *ep_p)
{
    if (ucs_test_all_flags(params->field_mask,
                           UCT_EP_PARAM_FIELD_DEV_ADDR |
                           UCT_EP_PARAM_FIELD_IFACE_ADDR)) {
        return uct_srd_ep_create_connected(params, ep_p);
    }

    return UCS_CLASS_NEW_FUNC_NAME(uct_srd_ep_t)(params, ep_p);
}
