/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "dt_generic.h"

#include <ucs/debug/memtrack.h>

ucs_status_t ucp_dt_create_generic(const ucp_generic_dt_ops_t *ops, void *context,
                                   ucp_datatype_t *datatype_p)
{
    ucp_datatype_t dt = ucp_dt_create(UCP_DATATYPE_GENERIC, ops, context);
    if (!dt) {
        return UCS_ERR_NO_MEMORY;
    }

    *datatype_p = dt;
    return UCS_OK;
}

void ucp_dt_generic_create(ucp_dt_generic_t *dt,
                           const ucp_generic_dt_ops_t *ops,
                           void *context)
{
    dt->ops      = *ops;
    dt->context  = context;
}

void ucp_dt_destroy(ucp_datatype_t datatype)
{
    ucp_dt_generic_t *dt;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        break;
    case UCP_DATATYPE_GENERIC:
        dt = ucp_dt_generic(datatype);
        ucs_free(dt);
        break;
    default:
        break;
    }
}
