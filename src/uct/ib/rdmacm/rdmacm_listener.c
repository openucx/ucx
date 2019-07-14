/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rdmacm_listener.h"


UCS_CLASS_INIT_FUNC(uct_rdmacm_listener_t, uct_cm_h cm,
                    const struct sockaddr *saddr, socklen_t socklen,
                    const uct_listener_params_t *params)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

UCS_CLASS_CLEANUP_FUNC(uct_rdmacm_listener_t){}

UCS_CLASS_DEFINE(uct_rdmacm_listener_t, uct_listener_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rdmacm_listener_t, uct_listener_t,
                          uct_cm_h , const struct sockaddr *, socklen_t ,
                          const uct_listener_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rdmacm_listener_t, uct_listener_t);
