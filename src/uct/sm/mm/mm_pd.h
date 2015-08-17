/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_MM_PD_H_
#define UCT_MM_PD_H_

#include <ucs/config/types.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/status.h>
#include <uct/tl/context.h>


/* Shared memory ID */
typedef uint64_t uct_mm_id_t;

/*
 * Memory mapper operations - MM uses them to implement PD and TL functionality.
 */
typedef struct uct_mm_mapper_ops {

    ucs_status_t (*query)();

    ucs_status_t (*reg)(void *address, size_t size, uct_mm_id_t *mmid_p);

    ucs_status_t (*dereg)(uct_mm_id_t mm_id);

    ucs_status_t (*alloc)(size_t *length_p, ucs_ternary_value_t hugetlb,
                          void **address_p, uct_mm_id_t *mmid_p UCS_MEMTRACK_ARG);

    ucs_status_t (*attach)(uct_mm_id_t mmid, size_t length, void **address_p);

    ucs_status_t (*detach)(void *address);

    ucs_status_t (*free)(void *address);

} uct_mm_mapper_ops_t;


/* Extract mapper ops from PD component */
#define uct_mm_pdc_mapper_ops(_pdc) \
    ((uct_mm_mapper_ops_t*)(_pdc)->priv)

/* Extract mapped ops from PD */
#define uct_mm_pd_mapper_ops(_pd) \
    uct_mm_pdc_mapper_ops((_pd)->component)


/*
 * Define a memory-mapper component for MM.
 *
 * @param _var     Variable for PD component.
 * @param _name    String which is the component name.
 * @param _ops     Mapper operations, of type uct_mm_mapper_ops_t.
 */
#define UCT_MM_COMPONENT_DEFINE(_var, _name, _ops) \
    \
    uct_pd_component_t _var; \
    \
    static ucs_status_t _var##_query_pd_resources(uct_pd_resource_desc_t **resources_p, \
                                                   unsigned *num_resources_p) { \
        if ((_ops)->query() == UCS_OK) { \
            return uct_single_pd_resource(&_var, resources_p, num_resources_p); \
        } else { \
            *resources_p = NULL; \
            *num_resources_p = 0; \
            return UCS_OK; \
        } \
    } \
    \
    static ucs_status_t _var##_pd_open(const char *pd_name, uct_pd_h *pd_p) \
    { \
        static uct_pd_ops_t pd_ops = { \
            .close        = (void*)ucs_empty_function, \
            .query        = uct_mm_pd_query, \
            .mem_alloc    = uct_mm_mem_alloc, \
            .mem_free     = uct_mm_mem_free, \
            .mem_reg      = uct_mm_mem_reg, \
            .mem_dereg    = uct_mm_mem_dereg, \
            .mkey_pack    = uct_mm_mkey_pack, \
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
                            sizeof(uct_mm_packed_rkey_t), uct_mm_rkey_unpack, \
                            uct_mm_rkey_release)


/**
 * Local memory segment structure.
 */
typedef struct uct_mm_seg {
    uct_mm_id_t      mmid;         /* Shared memory ID */
    void             *address;     /* Virtual address */
    size_t           length;       /* Size of the memory */
} uct_mm_seg_t;


/**
 * Packed remote key
 */
typedef struct uct_mm_packed_rkey {
    uct_mm_id_t      mmid;         /* Shared memory ID */
    uintptr_t        owner_ptr;    /* VA of in allocating process */
    size_t           len;          /* Size of the memory */
} uct_mm_packed_rkey_t;


ucs_status_t uct_mm_mem_alloc(uct_pd_h pd, size_t *length_p, void **address_p,
                              uct_mem_h *memh_p UCS_MEMTRACK_ARG);

ucs_status_t uct_mm_mem_free(uct_pd_h pd, uct_mem_h memh);

ucs_status_t uct_mm_mem_reg(uct_pd_h pd, void *address, size_t length,
                            uct_mem_h *memh_p);

ucs_status_t uct_mm_mem_dereg(uct_pd_h pd, uct_mem_h memh);

ucs_status_t uct_mm_pd_query(uct_pd_h pd, uct_pd_attr_t *pd_attr);

ucs_status_t uct_mm_mkey_pack(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer);

ucs_status_t uct_mm_rkey_unpack(uct_pd_component_t *pdc, const void *rkey_buffer,
                                uct_rkey_t *rkey_p, void **handle_p);

ucs_status_t uct_mm_rkey_release(uct_pd_component_t *pdc, uct_rkey_t rkey, void *handle);


#endif
