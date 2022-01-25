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
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/status.h>


/* Memory mapper segment unique id, used for both FIFO and bcopy descriptors.
 * The exact structure depends on specific mapper */
typedef uint64_t          uct_mm_seg_id_t;


/**
 * Local memory segment structure.
 * The mappers must implement memory allocation functions so that they will
 * return this structure as uct_memh.
 */
typedef struct uct_mm_seg {
    uct_mm_seg_id_t       seg_id;     /* Shared memory ID */
    void                  *address;   /* Virtual address */
    size_t                length;     /* Size of the memory */
} uct_mm_seg_t;


/*
 * Descriptor of remote attached memory
 */
typedef struct uct_mm_remote_seg {
    void                  *address;    /* Local address of attached memory */
    void                  *cookie;     /* Mapper-specific data */
} uct_mm_remote_seg_t;


/**
 * MM memory domain configuration
 */
typedef struct uct_mm_md_config {
    uct_md_config_t          super;
    ucs_ternary_auto_value_t hugetlb_mode;     /* Enable using huge pages */
} uct_mm_md_config_t;


/**
 * MM memory domain
 */
typedef struct uct_mm_md {
    uct_md_t              super;
    uct_mm_md_config_t    *config;         /* Clone of MD configuration */
    size_t                iface_addr_len;  /* As returned from
                                              uct_mm_md_mapper_ops_t::iface_addr_length */
} uct_mm_md_t;


/* Check if available on current machine.
 *
 * @param [in/out] attach_shm_file_p     Flag which shows whether MM transport
 *                                       attaches to a SHM file or to a process
 *                                       region.
 *
 * @return UCS_OK - if MM transport is available on the machine, otherwise -
 *         error code.
 */
typedef ucs_status_t (*uct_mm_mapper_query_func_t)(int *attach_shm_file_p);


/* Return the size of memory-domain specific iface address (e.g mmap path) */
typedef size_t (*uct_mm_mapper_iface_addr_length_func_t)(uct_mm_md_t *md);


/* Pack interface address. Holds common information for all memory segments
 * allocated on the same interface. 'buffer' must be at least the size returned
 * from iface_addr_length()
 */
typedef ucs_status_t
(*uct_mm_mapper_iface_addr_pack_func_t)(uct_mm_md_t *md, void *buffer);


/* Attach memory allocated by mem_alloc(). seg_id is from 'uct_mm_seg_t'
 * structure, and iface_addr is from iface_addr_pack() on the remote process
 *
 * This function is used only for active messages memory (FIFO and receive
 * descriptors).
 */
typedef ucs_status_t
(*uct_mm_mapper_mem_attach_func_t)(uct_mm_md_t *md, uct_mm_seg_id_t seg_id,
                                   size_t length, const void *iface_addr,
                                   uct_mm_remote_seg_t *rseg);


/* Check if memory may be attached using mem_attach. seg_id is from
 * 'uct_mm_seg_t' structure, and iface_addr is from iface_addr_pack() on the
 * remote process
 */
typedef int
(*uct_mm_mapper_is_reachable_func_t)(uct_mm_md_t *md, uct_mm_seg_id_t seg_id,
                                     const void *iface_addr);


/* Clean up the remote segment handle created by mem_attach() */
typedef void
(*uct_mm_mapper_mem_detach_func_t)(uct_mm_md_t *md,
                                   const uct_mm_remote_seg_t *rseg);


/*
 * Memory mapper operations - used to implement MD and TL functionality
 */
typedef struct uct_mm_mapper_ops {
    uct_md_ops_t                           super;
    uct_mm_mapper_query_func_t             query;
    uct_mm_mapper_iface_addr_length_func_t iface_addr_length;
    uct_mm_mapper_iface_addr_pack_func_t   iface_addr_pack;
    uct_mm_mapper_mem_attach_func_t        mem_attach;
    uct_mm_mapper_mem_detach_func_t        mem_detach;
    uct_mm_mapper_is_reachable_func_t      is_reachable;
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
    ucs_derived_of((_md)->super.ops, uct_mm_md_mapper_ops_t)


/* Call mapper operation */
#define uct_mm_md_mapper_call(_md, _func, ...) \
    uct_mm_md_mapper_ops(_md)->_func(_md, ## __VA_ARGS__)


/*
 * Define a memory-mapper component for MM.
 *
 * @param _var          Variable for MM component.
 * @param _name         String which is the component name.
 * @param _md_ops       Mapper operations, of type uct_mm_mapper_ops_t.
 * @param _rkey_unpack  Remote key unpack function.
 * @param _rkey_release Remote key release function.
 * @param _cfg_prefix   Prefix for configuration environment vars.
 */
#define UCT_MM_COMPONENT_DEFINE(_name, _md_ops, _rkey_unpack, _rkey_release, \
                                _cfg_prefix) \
    \
    static uct_mm_component_t UCT_COMPONENT_NAME(_name) = { \
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
            .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY, \
            .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER( \
                                      &UCT_COMPONENT_NAME(_name).super), \
            .flags              = 0, \
            .md_vfs_init        = \
                    (uct_component_md_vfs_init_func_t)ucs_empty_function \
       }, \
       .md_ops                  = (_md_ops) \
    };


extern ucs_config_field_t uct_mm_md_config_table[];


ucs_status_t uct_mm_query_md_resources(uct_component_t *component,
                                       uct_md_resource_desc_t **resources_p,
                                       unsigned *num_resources_p);

ucs_status_t uct_mm_seg_new(void *address, size_t length, uct_mm_seg_t **seg_p);

void uct_mm_md_query(uct_md_h md, uct_md_attr_t *md_attr, uint64_t max_alloc);

ucs_status_t uct_mm_rkey_ptr(uct_component_t *component, uct_rkey_t rkey,
                             void *handle, uint64_t raddr, void **laddr_p);

ucs_status_t uct_mm_md_open(uct_component_t *component, const char *md_name,
                            const uct_md_config_t *config, uct_md_h *md_p);

void uct_mm_md_close(uct_md_h md);

static inline void
uct_mm_md_make_rkey(void *local_address, uintptr_t remote_address,
                    uct_rkey_t *rkey_p)
{
    *rkey_p = (uintptr_t)local_address - remote_address;
}

#endif
