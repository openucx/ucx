/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_MM_MD_H_
#define UCT_MM_MD_H_

#include <uct/base/uct_md.h>
#include <ucs/config/types.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/status.h>


/* Shared memory ID */
typedef uint64_t uct_mm_id_t;


/**
 * Local memory segment structure.
 */
typedef struct uct_mm_seg {
    uct_mm_id_t           mmid;       /* Shared memory ID */
    void                  *address;   /* Virtual address */
    size_t                length;     /* Size of the memory */
    const char            *path;      /* Path to the backing file when using posix */
} uct_mm_seg_t;


/**
 * Packed remote key
 */
typedef struct uct_mm_packed_rkey {
    uct_mm_id_t           mmid;       /* Shared memory ID */
    uintptr_t             owner_ptr;  /* VA of in allocating process */
    size_t                length;     /* Size of the memory */
    char                  path[0];    /* path to the backing file when using posix */
} uct_mm_packed_rkey_t;


/*
 * Descriptor of the mapped memory
 */
typedef struct uct_mm_remote_seg uct_mm_remote_seg_t;
struct uct_mm_remote_seg {
    uct_mm_remote_seg_t   *next;
    uct_mm_id_t           mmid;        /* mmid of the remote memory chunk */
    void                  *address;    /* Local address of attached memory */
    uint64_t              cookie;      /* Cookie for mmap, xpmem, etc. */
    size_t                length;      /* Size of the memory */
};


/**
 * MM memory domain configuration
 */
typedef struct uct_mm_md_config {
    uct_md_config_t       super;
    ucs_ternary_value_t   hugetlb_mode;     /* Enable using huge pages */
} uct_mm_md_config_t;


/**
 * MM memory domain
 */
typedef struct uct_mm_md {
    uct_md_t              super;
    uct_mm_md_config_t    *config;
} uct_mm_md_t;


/*
 * Memory mapper operations - used to implement MD and TL functionality
 */
typedef struct uct_mm_mapper_ops {
    uct_md_ops_t           super;

    ucs_status_t (*query)();

    size_t       (*get_path_size)(uct_md_h md);

    uint8_t      (*get_priority)();

    ucs_status_t (*reg)(void *address, size_t size,
                        uct_mm_id_t *mmid_p);

    ucs_status_t (*dereg)(uct_mm_id_t mm_id);

    ucs_status_t (*alloc)(uct_md_h md, size_t *length_p, ucs_ternary_value_t hugetlb,
                          unsigned flags, const char *alloc_name, void **address_p,
                          uct_mm_id_t *mmid_p, const char **path_p);

    ucs_status_t (*attach)(uct_mm_id_t mmid, size_t length,
                           void *remote_address, void **address, uint64_t *cookie,
                           const char *path);

    ucs_status_t (*detach)(uct_mm_remote_seg_t *mm_desc);

    ucs_status_t (*free)(void *address, uct_mm_id_t mm_id, size_t length,
                         const char *path);

} uct_mm_md_mapper_ops_t;


/**
 * Memory mapper component
 */
typedef struct uct_mm_component {
    uct_component_t        super;
    uct_mm_md_mapper_ops_t *md_ops;
} uct_mm_component_t;


/* Extract mapper ops from MM component */
#define uct_mm_mdc_mapper_ops(_component) \
    (ucs_derived_of(_component, uct_mm_component_t)->md_ops)


/* Extract mapper ops from MM memory domain */
#define uct_mm_md_mapper_ops(_md) \
    ucs_derived_of((_md)->ops, uct_mm_md_mapper_ops_t)


/* Call mapper operation */
#define uct_mm_md_mapper_call(_md, _func, ...) \
    uct_mm_md_mapper_ops(_md)->_func(__VA_ARGS__)


/*
 * Define a memory-mapper component for MM.
 *
 * @param _var          Variable for MM component.
 * @param _name         String which is the component name.
 * @param _md_ops       Mapper operations, of type uct_mm_mapper_ops_t.
 * @param _cfg_prefix   Prefix for configuration environment vars.
 */
#define UCT_MM_COMPONENT_DEFINE(_var, _name, _md_ops, _rkey_unpack, \
                                _rkey_release, _cfg_prefix) \
    \
    static uct_mm_component_t _var = { \
        .super = { \
            .query_md_resources = uct_mm_query_md_resources, \
            .md_open            = uct_mm_md_open, \
            .cm_open            = ucs_empty_function_return_unsupported, \
            .rkey_unpack        = _rkey_unpack, \
            .rkey_ptr           = uct_mm_rkey_ptr, \
            .rkey_release       = _rkey_release, \
            .name               = #_name, \
            .md_config          = { \
                .name           = #_name " memory domain", \
                .prefix         = _cfg_prefix, \
                .table          = uct_##_name##_md_config_table, \
                .size           = sizeof(uct_##_name##_md_config_t), \
            }, \
            .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER( \
                                      &(_var).super), \
            .flags              = 0, \
       }, \
       .md_ops                  = (_md_ops) \
    }; \
    UCT_COMPONENT_REGISTER(&(_var).super); \


extern ucs_config_field_t uct_mm_md_config_table[];


ucs_status_t uct_mm_query_md_resources(uct_component_t *component,
                                       uct_md_resource_desc_t **resources_p,
                                       unsigned *num_resources_p);

ucs_status_t uct_mm_mem_alloc(uct_md_h md, size_t *length_p, void **address_p,
                              unsigned flags, const char *alloc_name,
                              uct_mem_h *memh_p);

ucs_status_t uct_mm_mem_free(uct_md_h md, uct_mem_h memh);

ucs_status_t uct_mm_mem_reg(uct_md_h md, void *address, size_t length,
                            unsigned flags, uct_mem_h *memh_p);

ucs_status_t uct_mm_mem_dereg(uct_md_h md, uct_mem_h memh);

ucs_status_t uct_mm_md_query(uct_md_h md, uct_md_attr_t *md_attr);

ucs_status_t uct_mm_mkey_pack(uct_md_h md, uct_mem_h memh, void *rkey_buffer);

ucs_status_t uct_mm_rkey_unpack(uct_component_t *component,
                                const void *rkey_buffer, uct_rkey_t *rkey_p,
                                void **handle_p);

ucs_status_t uct_mm_rkey_ptr(uct_component_t *component, uct_rkey_t rkey,
                             void *handle, uint64_t raddr, void **laddr_p);

ucs_status_t uct_mm_rkey_release(uct_component_t *component, uct_rkey_t rkey,
                                 void *handle);

ucs_status_t uct_mm_md_open(uct_component_t *component, const char *md_name,
                            const uct_md_config_t *config, uct_md_h *md_p);

void uct_mm_md_close(uct_md_h md);

#endif
