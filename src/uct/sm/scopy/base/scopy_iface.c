/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "scopy_iface.h"

#include <uct/sm/base/sm_iface.h>


ucs_config_field_t uct_scopy_iface_config_table[] = {
    {"SM_", "", NULL,
     ucs_offsetof(uct_scopy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_sm_iface_config_table)},

    {"MAX_IOV", "16",
     "Maximum IOV count that can contain user-defined payload in a single\n"
     "call to GET/PUT Zcopy operation",
     ucs_offsetof(uct_scopy_iface_config_t, max_iov), UCS_CONFIG_TYPE_ULONG},

    {NULL}
};

void uct_scopy_iface_query(uct_scopy_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    uct_base_iface_query(&iface->super.super, iface_attr);

    /* default values for all shared memory transports */
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = iface->config.max_iov;

    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = iface->config.max_iov;

    iface_attr->device_addr_len         = uct_sm_iface_get_device_addr_len();
    iface_attr->ep_addr_len             = 0;
    iface_attr->max_conn_priv           = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY |
                                          UCT_IFACE_FLAG_PENDING   |
                                          UCT_IFACE_FLAG_CONNECT_TO_IFACE;
    iface_attr->latency.overhead        = 80e-9; /* 80 ns */
    iface_attr->latency.growth          = 0;
}

UCS_CLASS_INIT_FUNC(uct_scopy_iface_t, uct_scopy_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config)
{
    uct_scopy_iface_config_t *config = ucs_derived_of(tl_config,
                                                      uct_scopy_iface_config_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_sm_iface_t, &ops->super, md, worker, params, tl_config);

    self->config.max_iov = ucs_min(config->max_iov, ucs_iov_get_max());

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_scopy_iface_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_scopy_iface_t, uct_sm_iface_t);
