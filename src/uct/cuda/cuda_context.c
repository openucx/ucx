/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "cuda_iface.h"
#include "cuda_context.h"
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <uct/tl/context.h>

ucs_status_t uct_cuda_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p);

ucs_config_field_t uct_cuda_iface_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_cuda_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    {NULL}
};

ucs_status_t uct_cuda_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resource_p,
                                      unsigned *num_resources_p)
{
    uct_resource_desc_t *resource = NULL;
    resource = ucs_calloc(1, sizeof(uct_resource_desc_t), "resource desc");
    if (NULL == resource) {
      ucs_error("Failed to allocate memory");
      return UCS_ERR_NO_MEMORY;
    }
    ucs_snprintf_zero(resource->tl_name,
                      sizeof(resource->tl_name), "%s", UCT_CUDA_TL_NAME);
    ucs_snprintf_zero(resource->dev_name,
                      sizeof(resource->dev_name), "%s", UCT_CUDA_TL_NAME);
    resource->latency    = 1; /* FIXME temp value */
    resource->bandwidth  = (long) (6911 * pow(1024,2)); /* FIXME temp value */
    *num_resources_p = 1;
    *resource_p     = resource;

    return UCS_OK;
}

ucs_status_t uct_cuda_init(uct_context_h context)
{
    ucs_status_t status;

    status = uct_register_tl(context, "cuda", uct_cuda_iface_config_table,
                             sizeof(uct_cuda_iface_config_t), "cuda_",
                             &uct_cuda_tl_ops);
    if (UCS_OK != status) {
        ucs_error("Failed to register context (%s)",
                  ucs_status_string(status));
        return status;
    }

    ucs_debug("Initialized cuda component");

    return UCS_OK;

}

void uct_cuda_cleanup(uct_context_t *context)
{
    /* no-op */
}
UCS_COMPONENT_DEFINE(uct_context_t, cuda, uct_cuda_init, uct_cuda_cleanup, 0)
