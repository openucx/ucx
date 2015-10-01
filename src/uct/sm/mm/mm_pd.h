/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_MM_PD_H_
#define UCT_MM_PD_H_

#include <ucs/config/types.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/status.h>
#include <uct/tl/context.h>
#include "mm_def.h"


/* Shared memory ID */
typedef uint64_t uct_mm_id_t;

extern ucs_config_field_t uct_mm_pd_config_table[];

/*
 * Descriptor of the mapped memory
 */
struct uct_mm_remote_seg {
    uct_mm_remote_seg_t *next;
    uct_mm_id_t mmid;        /**< mmid of the remote memory chunk */
    void        *address;    /**< local memory address */
    uint64_t    cookie;      /**< cookie for mmap, xpmem, etc. */
    size_t      length;      /**< size of the memory */
};

/*
 * Memory mapper operations - MM uses them to implement PD and TL functionality.
 */
typedef struct uct_mm_mapper_ops {

    ucs_status_t (*query)();

    ucs_status_t (*reg)(void *address, size_t size, 
                        uct_mm_id_t *mmid_p);

    ucs_status_t (*dereg)(uct_mm_id_t mm_id);

    ucs_status_t (*alloc)(uct_pd_h pd, size_t *length_p, ucs_ternary_value_t hugetlb,
                          void **address_p, uct_mm_id_t *mmid_p UCS_MEMTRACK_ARG);

    ucs_status_t (*attach)(uct_mm_id_t mmid, size_t length, 
                           void *remote_address, void **address, uint64_t *cookie);

    ucs_status_t (*detach)(uct_mm_remote_seg_t *mm_desc);

    ucs_status_t (*free)(void *address, uct_mm_id_t mm_id, size_t length);

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
 * @param _var          Variable for PD component.
 * @param _name         String which is the component name.
 * @param _ops          Mapper operations, of type uct_mm_mapper_ops_t.
 * @param _prefix       Prefix for defining the vars config table and config struct.
 * @param _cfg_prefix   Prefix for configuration environment vars.
 */
#define UCT_MM_COMPONENT_DEFINE(_var, _name, _ops, _prefix, _cfg_prefix) \
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
    static ucs_status_t _var##_pd_open(const char *pd_name, const uct_pd_config_t *pd_config, \
                                       uct_pd_h *pd_p) \
    { \
        return uct_mm_pd_open(pd_name, pd_config, pd_p, &_var); \
    } \
    \
    UCT_PD_COMPONENT_DEFINE(_var, _name, \
                            _var##_query_pd_resources, _var##_pd_open, _ops, \
                            sizeof(uct_mm_packed_rkey_t), uct_mm_rkey_unpack, \
                            uct_mm_rkey_release, _cfg_prefix, _prefix##_pd_config_table, \
                            _prefix##_pd_config_t)


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
    size_t           length;       /* Size of the memory */
} uct_mm_packed_rkey_t;


/**
 * MM PD
 */
typedef struct uct_mm_pd {
    uct_pd_t           super;
    uct_mm_pd_config_t *config;
} uct_mm_pd_t;


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

ucs_status_t uct_mm_pd_open(const char *pd_name, const uct_pd_config_t *pd_config,
                            uct_pd_h *pd_p, uct_pd_component_t *_var);

#endif
