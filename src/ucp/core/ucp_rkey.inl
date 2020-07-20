/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RKEY_INL_
#define UCP_RKEY_INL_

#include "ucp_rkey.h"
#include "ucp_worker.h"


static UCS_F_ALWAYS_INLINE ucp_rkey_config_t *
ucp_rkey_config(ucp_worker_h worker, ucp_rkey_h rkey)
{
    ucs_assert(rkey->cfg_index != UCP_WORKER_CFG_INDEX_NULL);
    return &worker->rkey_config[rkey->cfg_index];
}

#endif
