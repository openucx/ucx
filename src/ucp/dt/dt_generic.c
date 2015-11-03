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
    ucp_dt_generic_t *dt;

    dt = ucs_memalign(UCS_BIT(UCP_DATATYPE_SHIFT), sizeof(*dt), "generic_dt");
    if (dt == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    dt->ops      = *ops;
    dt->context  = context;
    *datatype_p = ((uintptr_t)dt) | UCP_DATATYPE_GENERIC;
    return UCS_OK;
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
