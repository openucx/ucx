/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "sm_context.h"

/* general query for all shared memory transports */
ucs_status_t uct_sm_query_resources(uct_resource_desc_t **resource_p, const char *name)
{
    uct_resource_desc_t *resource;

    /* set default values for shared memory transports */
    int latency         = 1;                           /* FIXME temp value */
    long int bandwidth  = (long) (6911 * pow(1024,2)); /* FIXME temp value */

    resource = ucs_calloc(1, sizeof(uct_resource_desc_t), "resource desc");
    if (NULL == resource) {
        ucs_error("Failed to allocate memory");
        return UCS_ERR_NO_MEMORY;
    }

    *resource_p = resource;

    ucs_snprintf_zero(resource->tl_name,  
                      sizeof(resource->tl_name), "%s", name);
    ucs_snprintf_zero(resource->dev_name, 
                      sizeof(resource->dev_name), "%s", name);
    resource->latency    = latency;
    resource->bandwidth  = bandwidth;
    memset(&resource->subnet_addr, 0, sizeof(resource->subnet_addr));

    return UCS_OK;
}
