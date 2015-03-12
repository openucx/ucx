/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#define _GNU_SOURCE /* for CPU_ZERO/CPU_SET in sched.h */
#include "sysv_device.h"

#include <uct/tl/context.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>


void uct_device_get_resource(uct_sysv_device_t *dev,
        uct_resource_desc_t *resource)
{
    ucs_snprintf_zero(resource->tl_name,  
                      sizeof(resource->tl_name), "%s", TL_NAME);
    ucs_snprintf_zero(resource->dev_name, 
                      sizeof(resource->dev_name), "%s", dev->fname);
    resource->local_cpus = dev->cpu_mask; /* FIXME */
    resource->latency    = 1; /* FIXME */
    resource->bandwidth  = (long) (6911 * pow(1024,2)); /* FIXME */
    memset(&resource->subnet_addr, 0, sizeof(resource->subnet_addr));
}

ucs_status_t uct_sysv_device_create(uct_context_h context, int dev_id, 
                                   uct_sysv_device_t *dev_p)
{
    dev_p->device_id = (uint32_t)dev_id;

    /* create names for the dummy sysv device */

    ucs_snprintf_zero(dev_p->type_name, sizeof(dev_p->type_name), "%s",
                      "SM_SYSV");

    ucs_snprintf_zero(dev_p->fname, sizeof(dev_p->fname), "%s:%u",
                      dev_p->type_name, dev_p->device_id);

    return UCS_OK;
}

/* currently unused */
void uct_sysv_device_destroy(uct_sysv_device_t *dev)
{
    /* No op */
}
