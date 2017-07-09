/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "dt_generic.h"

#include <ucs/debug/memtrack.h>


void ucp_dt_generic_create(ucp_dt_generic_t *dt,
                           const ucp_generic_dt_ops_t *ops,
                           void *context)
{
    dt->ops      = *ops;
    dt->context  = context;
}
