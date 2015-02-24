/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#ifndef UCT_SYSV_CONTEXT_H
#define UCT_SYSV_CONTEXT_H

#include "sysv_device.h"

typedef struct uct_sysv_context {
    int                 num_devices;        /**< Number of devices */
    int                 num_ifaces;         /**< Number of active interfaces */
    bool                activated;          /**< Context status */
    uct_sysv_device_t    device;             /**< Device belonging to this domain */
} uct_sysv_context_t;

/*
 * Helper function to list sysv resources
 */
ucs_status_t uct_sysv_query_resources(uct_context_h context,
        uct_resource_desc_t **resources_p,
        unsigned *num_resources_p);

ucs_status_t sysv_activate_domain(uct_sysv_context_t *sysv_ctx);
#endif
