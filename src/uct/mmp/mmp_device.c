/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#define _GNU_SOURCE /* for CPU_ZERO/CPU_SET in sched.h */
#include "mmp_device.h"

#include <uct/tl/context.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>


void uct_device_get_resource(uct_mmp_device_t *dev,
        uct_resource_desc_t *resource)
{
}

static ucs_status_t get_nic_address(uct_mmp_device_t *dev_p)
{
}

ucs_status_t uct_mmp_device_create(uct_context_h context, int dev_id, 
                                   uct_mmp_device_t *dev_p)
{
}

void uct_mmp_device_destroy(uct_mmp_device_t *dev)
{
    /* Nop */
}
