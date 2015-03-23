/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_SYSV_CONTEXT_H
#define UCT_SYSV_CONTEXT_H

#define MAX_TYPE_NAME     (10)
#define UCT_TL_NAME           "sysv"

/*
 * Helper function to list sysv resources
 */
ucs_status_t uct_sysv_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p);

#endif
