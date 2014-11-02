/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_iface.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_context.h>
#include <uct/ib/base/ib_iface.h>


ucs_status_t uct_rc_query_resources(uct_context_h context, uct_resource_desc_t **resources_p,
                                    unsigned *num_resources_p)
{
    /* TODO take transport overhead into account */
    return uct_ib_query_resources(context, 0, resources_p, num_resources_p);
}

uct_tl_ops_t uct_rc_tl_ops = {
    .query_resources = uct_rc_query_resources,
    .iface_open      = uct_rc_iface_open,
    .iface_close     = uct_rc_iface_close,
};

