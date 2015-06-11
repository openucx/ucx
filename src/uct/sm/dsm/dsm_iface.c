/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "dsm_iface.h"
#include "dsm_ep.h"

#include <uct/api/addr.h>


ucs_config_field_t uct_dsm_iface_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_dsm_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    {NULL}
};

void uct_dsm_iface_get_address(uct_dsm_iface_t *iface,
                              uct_sockaddr_process_t *iface_addr)
{
    iface_addr->sp_family = UCT_AF_PROCESS;
    iface_addr->node_guid = ucs_machine_guid();
}

int uct_dsm_iface_is_reachable(uct_iface_t *tl_iface,
                              const struct sockaddr *addr)
{
    return (addr->sa_family == UCT_AF_PROCESS) &&
           (((uct_sockaddr_process_t*)addr)->node_guid == ucs_machine_guid());
}

ucs_status_t uct_dsm_iface_flush(uct_iface_h tl_iface)
{
    return UCS_OK;
}

ucs_status_t uct_dsm_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_dsm_iface_t *iface = ucs_derived_of(tl_iface, uct_dsm_iface_t);
    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    /* default values for all shared memory transports */
    iface_attr->cap.put.max_short      = iface->config.max_put;
    iface_attr->cap.put.max_bcopy      = iface->config.max_bcopy;
    iface_attr->cap.put.max_zcopy      = iface->config.max_zcopy;
    iface_attr->cap.get.max_bcopy      = iface->config.max_bcopy;
    iface_attr->cap.get.max_zcopy      = iface->config.max_zcopy;
    iface_attr->iface_addr_len         = sizeof(uct_sockaddr_process_t);
    iface_attr->ep_addr_len            = 0;
    iface_attr->cap.flags              = 0; /* force actual TL to set its own */
    iface_attr->completion_priv_len    = 0; /* TBD */

    return UCS_OK;
}

UCS_CLASS_INIT_FUNC(uct_dsm_iface_t, uct_iface_ops_t *ops, uct_pd_h pd,
                    uct_worker_h worker, const uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, ops, pd, worker,
                              tl_config UCS_STATS_ARG(NULL));

    /* default values for all shared memory transports
     * (force each TL to set for itself) */
    self->config.max_put     = -1;
    self->config.max_bcopy   = -1;
    self->config.max_zcopy   = -1;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dsm_iface_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_dsm_iface_t, uct_base_iface_t);
