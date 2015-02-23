/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_MMP_CONTEXT_H
#define UCT_MMP_CONTEXT_H

#include "mmp_device.h"

typedef struct uct_mmp_context {
    int                 num_devices;        /**< Number of devices */
    int                 num_ifaces;         /**< Number of active interfaces */
    bool                activated;          /**< Context status */
} uct_mmp_context_t;

/*
 * Helper function to list MMP resources
 */
ucs_status_t uct_mmp_query_resources(uct_context_h context,
        uct_resource_desc_t **resources_p,
        unsigned *num_resources_p);

ucs_status_t mmp_activate_domain(uct_mmp_context_t *mmp_ctx);
#endif
