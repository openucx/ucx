/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "sm_iface.h"
#include "sm_ep.h"

/* default values for all shared memory transports */
#define UCT_SM_MAX_SHORT_LENGTH 2048 /* FIXME temp value for now */
#define UCT_SM_MAX_BCOPY_LENGTH 40960 /* FIXME temp value for now */
#define UCT_SM_MAX_ZCOPY_LENGTH 81920 /* FIXME temp value for now */

ucs_status_t uct_sm_iface_flush(uct_iface_h tl_iface)
{
    return UCS_OK;
}

ucs_status_t uct_sm_iface_get_address(uct_iface_h tl_iface, 
                                      uct_iface_addr_t *iface_addr)
{
    /* fake uuid address */
    uct_sm_iface_t *iface = ucs_derived_of(tl_iface, uct_sm_iface_t);

    *(uct_sm_iface_addr_t*)iface_addr = iface->addr;
    return UCS_OK;
}

ucs_status_t uct_sm_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_sm_iface_t *iface = ucs_derived_of(tl_iface, uct_sm_iface_t);

    memset(iface_attr, 0, sizeof(uct_iface_attr_t));

    /* default values for all shared memory transports */
    iface_attr->cap.put.max_short      = iface->config.max_put;
    iface_attr->cap.put.max_bcopy      = iface->config.max_bcopy;
    iface_attr->cap.put.max_zcopy      = iface->config.max_zcopy;
    iface_attr->cap.get.max_bcopy      = iface->config.max_bcopy;
    iface_attr->cap.get.max_zcopy      = iface->config.max_zcopy;
    iface_attr->iface_addr_len         = sizeof(uct_sm_iface_addr_t);
    iface_attr->ep_addr_len            = sizeof(uct_sm_ep_addr_t);
    iface_attr->cap.flags              = UCT_IFACE_FLAG_PUT_SHORT       |
                                         UCT_IFACE_FLAG_PUT_BCOPY       |
                                         UCT_IFACE_FLAG_ATOMIC_ADD32    |
                                         UCT_IFACE_FLAG_ATOMIC_ADD64    |
                                         UCT_IFACE_FLAG_ATOMIC_FADD64   |
                                         UCT_IFACE_FLAG_ATOMIC_FADD32   |
                                         UCT_IFACE_FLAG_ATOMIC_SWAP64   |
                                         UCT_IFACE_FLAG_ATOMIC_SWAP32   |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP64  |
                                         UCT_IFACE_FLAG_ATOMIC_CSWAP32  |
                                         UCT_IFACE_FLAG_PUT_ZCOPY       |
                                         UCT_IFACE_FLAG_GET_BCOPY       |
                                         UCT_IFACE_FLAG_GET_ZCOPY;

    iface_attr->completion_priv_len    = 0; /* TBD */

    return UCS_OK;
}

UCS_CLASS_INIT_FUNC(uct_sm_iface_t, uct_iface_ops_t *ops, uct_worker_h worker, 
                    uct_pd_h pd, const uct_iface_config_t *tl_config, 
                    const char *dev_name, const char *tl_name)
{
    int addr;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, ops, worker, pd, 
                              tl_config UCS_STATS_ARG(NULL));

    if(strcmp(dev_name, tl_name) != 0) {
        ucs_error("No device was found: %s", dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    /* set default values for all derived shared memory transports */
    self->config.max_put     = UCT_SM_MAX_SHORT_LENGTH;
    self->config.max_bcopy   = UCT_SM_MAX_BCOPY_LENGTH;
    self->config.max_zcopy   = UCT_SM_MAX_ZCOPY_LENGTH;

    addr = ucs_generate_uuid((intptr_t)self);

    self->addr.nic_addr = addr;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sm_iface_t)
{
    /* No op */
}

UCS_CLASS_DEFINE(uct_sm_iface_t, uct_base_iface_t);
