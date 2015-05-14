/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_SYSV_CONTEXT_H
#define UCT_SYSV_CONTEXT_H

#include <uct/sm/base/sm_context.h>
#include "sysv_iface.h"

#define UCT_SYSV_TL_NAME    "sysv"

ucs_status_t uct_sysv_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p);

#endif
