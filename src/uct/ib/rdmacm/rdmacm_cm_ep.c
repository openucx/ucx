/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rdmacm_cm_ep.h"


UCS_CLASS_INIT_FUNC(uct_rdmacm_cm_ep_t, const uct_ep_params_t *params)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_cm_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rdmacm_cm_ep_t, uct_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_cm_ep_t, uct_ep_t);
