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
            .mem_alloc    = (void*)ucs_empty_function, \
            .mem_free     = (void*)ucs_empty_function, \
            .mkey_pack    = uct_cma_mkey_pack, \
            .mem_reg      = uct_cma_mem_reg, \
            .mem_dereg    = uct_cma_mem_dereg, \
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
                            sizeof(uct_cma_packed_rkey_t), uct_cma_rkey_unpack, \
                            uct_cma_rkey_release)


/**
 * Local memory segment structure.
 */
typedef struct uct_cma_seg {
    pid_t      cma_id;         /* Shared memory ID */
} uct_cma_seg_t;


/**
 * Packed remote key
 */
typedef struct uct_cma_packed_rkey {
    pid_t     cma_id;       /* PID */
} uct_cma_packed_rkey_t;

ucs_status_t uct_cma_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr);

ucs_status_t uct_cma_mkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer);

ucs_status_t uct_cma_rkey_unpack(uct_pd_component_t *pdc, const void *rkey_buffer,
                                uct_rkey_t *rkey_p, void **handle_p);

ucs_status_t uct_cma_rkey_release(uct_pd_component_t *pdc, uct_rkey_t rkey, void *handle);

ucs_status_t uct_cma_mem_reg(uct_pd_h pd, void *address, size_t length,
                                     uct_mem_h *memh_p);

ucs_status_t uct_cma_mem_dereg(uct_pd_h pd, uct_mem_h memh);

#endif
