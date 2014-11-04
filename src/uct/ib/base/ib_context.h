/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_IB_CONTEXT_H_
#define UCT_IB_CONTEXT_H_

#include "ib_device.h"


typedef struct uct_ib_context uct_ib_context_t;
struct uct_ib_context {
    unsigned                    num_devices;     /* Number of devices */
    uct_ib_device_t             **devices;       /* Array of devices */
};

/*
 * Helper function to list IB resources
 */
ucs_status_t uct_ib_query_resources(uct_context_h context, unsigned flags,
                                    uct_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);


#endif
