/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "dt_generic.h"

#include <ucs/sys/math.h>
#include <ucs/debug/memtrack_int.h>


ucs_status_t ucp_dt_create_generic(const ucp_generic_dt_ops_t *ops, void *context,
                                   ucp_datatype_t *datatype_p)
{
    ucp_dt_generic_t *dt_gen;
    int ret;

    ret = ucs_posix_memalign((void **)&dt_gen,
                             ucs_max(sizeof(void *), UCS_BIT(UCP_DATATYPE_SHIFT)),
                             sizeof(*dt_gen), "generic_dt");
    if (ret != 0) {
        return UCS_ERR_NO_MEMORY;
    }

    dt_gen->ops     = *ops;
    dt_gen->context = context;
    *datatype_p     = ucp_dt_from_generic(dt_gen);
    return UCS_OK;
}

void ucp_dt_destroy(ucp_datatype_t datatype)
{
    ucp_dt_generic_t *dt_gen;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        break;
    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_to_generic(datatype);
        ucs_free(dt_gen);
        break;
    default:
        break;
    }
}
