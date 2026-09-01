/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_md.h"
#include "cuda_util.h"

#include <ucs/sys/module.h>


ucs_status_t
uct_cuda_base_query_md_resources(uct_component_t *component,
                                 uct_md_resource_desc_t **resources_p,
                                 unsigned *num_resources_p)
{
    unsigned num_visible_gpus = uct_cuda_init_devices();

    if (num_visible_gpus == 0) {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    } else {
        return uct_md_query_single_md_resource(component, resources_p,
                                               num_resources_p);
    }
}

UCS_STATIC_INIT
{
    UCT_CUDADRV_FUNC_LOG_DEBUG(cuInit(0));
}

UCS_STATIC_CLEANUP
{
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda, 0);
    return UCS_OK;
}
