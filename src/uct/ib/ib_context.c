/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ib_context.h"

#include <uct/tl/context.h>
#include <ucs/type/component.h>


ucs_status_t uct_ib_init(uct_context_t *context)
{
    return UCS_OK;
}

void uct_ib_cleanup(uct_context_t *context)
{
}

UCS_COMPONENT_DEFINE(uct_context_t, ib, uct_ib_init, uct_ib_cleanup, uct_ib_context_t)

