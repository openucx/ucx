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


/* called by uct_mmp_query_resources to populate resources array */
void uct_device_get_resource(uct_mmp_device_t *dev,
        uct_resource_desc_t *resource)
{
    ucs_snprintf_zero(resource->tl_name,  
                      sizeof(resource->tl_name), "%s", TL_NAME);
    ucs_snprintf_zero(resource->dev_name, 
                      sizeof(resource->dev_name), "%s", dev->fname);
    resource->local_cpus ; /* FIXME */
    resource->latency    ; /* FIXME */
    resource->bandwidth  ; /* FIXME */
    memset(&resource->subnet_addr, 0, sizeof(resource->subnet_addr));
}

/* called by uct_mmp_device_create at the very beginning */
static ucs_status_t get_nic_address(uct_mmp_device_t *dev_p)
{
    /* FIXME not sure yet what do here */
}

/* called by uct_mmp_init to populate devices array */
ucs_status_t uct_mmp_device_create(uct_context_h context, int dev_id, 
                                   uct_mmp_device_t *dev_p)
{
    ucs_status_t rc;

    dev_p->device_id = (uint32_t)dev_id;

    rc = get_nic_address(dev_p);
    if (rc != UCS_OK) {
        ucs_error("Failed to get NIC address");
        return rc;
    }

    /* FIXME create a name out of device type and address */

    dev_p->attached = false;
    return UCS_OK;
}

/* called by uct_mmp_cleanup */
void uct_mmp_device_destroy(uct_mmp_device_t *dev)
{
    /* FIXME possibly unmap files, free memory, etc. */
}
