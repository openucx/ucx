/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_CMA_PD_H_
#define UCT_CMA_PD_H_

#include <ucs/config/types.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/status.h>
#include <uct/tl/context.h>

#include <sys/types.h>
#include <unistd.h>

/*
 * Define a memory-mapper component for MM.
 *
 * @param _var     Variable for PD component.
 * @param _name    String which is the component name.
 * @param _ops     Mapper operations, of type uct_cma_mapper_ops_t.
 */
#define UCT_CMA_COMPONENT_DEFINE(_var, _name, _ops) \
    \
    uct_pd_component_t _var; \
    \
    static ucs_status_t _var##_query_pd_resources(uct_pd_resource_desc_t **resources_p, \
                                                   unsigned *num_resources_p) { \
        return uct_single_pd_resource(&_var, resources_p, num_resources_p); \
    } \
    \
    static ucs_status_t _var##_pd_open(const char *pd_name, uct_pd_h *pd_p) \
    { \
        static uct_pd_ops_t pd_ops = { \
            .close        = (void*)ucs_empty_function, \
            .query        = uct_cma_pd_query, \
            .mem_alloc    = (void*)ucs_empty_function_return_success, \
            .mem_free     = (void*)ucs_empty_function_return_success, \
            .mkey_pack    = (void*)ucs_empty_function_return_success, \
            .mem_reg      = (void*)ucs_empty_function_return_success, \
            .mem_dereg    = (void*)ucs_empty_function_return_success  \
        }; \
        static uct_pd_t pd = { \
            .ops          = &pd_ops, \
            .component    = &_var \
        }; \
        \
        *pd_p = &pd; \
        return UCS_OK; \
    } \
    \
    UCT_PD_COMPONENT_DEFINE(_var, _name, \
                            _var##_query_pd_resources, _var##_pd_open, _ops, \
                            0, ucs_empty_function_return_success, \
                            ucs_empty_function_return_success)

ucs_status_t uct_cma_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr);

#endif
