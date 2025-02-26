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


static UCS_CLASS_INIT_FUNC(uct_srd_ep_t, const uct_ep_params_t *params)
{
    uct_srd_iface_t *iface              = ucs_derived_of(params->iface,
                                                         uct_srd_iface_t);
    const uct_ib_address_t *ib_addr     = (const uct_ib_address_t*)
                                                  params->dev_addr;
    const uct_srd_iface_addr_t *if_addr = (const uct_srd_iface_addr_t*)
                                                  params->iface_addr;
    char buf[128];
    uct_ib_device_t UCS_V_UNUSED *dev;
    ucs_status_t status;

    ucs_trace_func("");

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);

    memset(self, 0, sizeof(*self));
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    self->ep_uuid    = ucs_generate_uuid((uintptr_t)self);
    self->ep_id      = UCT_SRD_EP_NULL_ID;
    self->path_index = UCT_EP_PARAMS_GET_PATH_INDEX(params);
    self->tx.psn     = UCT_SRD_INITIAL_PSN;

    uct_srd_iface_add_ep(iface, self);

    status = uct_srd_iface_unpack_peer_address(iface, ib_addr, if_addr,
                                               self->path_index,
                                               &self->peer_address);
    if (status != UCS_OK) {
        goto err_ep_destroy;
    }

    dev = uct_ib_iface_device(&iface->super);
    ucs_debug(UCT_IB_IFACE_FMT
              " lid=%d qpn=0x%x ep_id=%u ep_uuid=0x%"PRIx64" ep=%p connected "
              "to IFACE %s qpn=0x%x",
              UCT_IB_IFACE_ARG(&iface->super),
              dev->port_attr[iface->super.config.port_num - dev->first_port].lid,
              iface->qp->qp_num, self->ep_id, self->ep_uuid, self,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(if_addr->qp_num));

    return UCS_OK;

err_ep_destroy:
    UCS_CLASS_DELETE_FUNC_NAME(uct_srd_ep_t)(&self->super.super);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_srd_ep_t)
{
    uct_srd_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                            uct_srd_iface_t);

    ucs_trace_func("");
    uct_srd_iface_remove_ep(iface, self);
}

UCS_CLASS_DEFINE(uct_srd_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);
