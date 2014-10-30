/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "context.h"

#include <uct/api/uct.h>
#include <ucs/debug/memtrack.h>

UCS_COMPONENT_LIST_DEFINE(uct_context_t);

ucs_status_t uct_init(uct_context_h *context_p)
{
    ucs_status_t status;
    uct_context_t *context;

    context = ucs_malloc(ucs_components_total_size(uct_context_t), "uct context");

    status = ucs_components_init_all(uct_context_t, context);
    if (status != UCS_OK) {
        return status;
    }

    *context_p = context;
    return UCS_OK;
}

void uct_cleanup(uct_context_h context)
{
    ucs_components_cleanup_all(uct_context_t, context);
    ucs_free(context);
}
