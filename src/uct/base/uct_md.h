/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_MD_H_
#define UCT_MD_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "uct_component.h"

#include <uct/api/uct.h>
#include <uct/api/v2/uct_v2.h>
#include <ucs/config/parser.h>
#include <ucs/memory/rcache.h>
#include <ucs/type/param.h>
#include <string.h>


#define uct_md_log_mem_reg_error(_flags, _fmt, ...) \
    ucs_log(uct_md_reg_log_lvl(_flags), _fmt, ## __VA_ARGS__)


#define UCT_MD_MEM_DEREG_FIELD_VALUE(_params, _name, _flag, _default) \
    UCS_PARAM_VALUE(UCT_MD_MEM_DEREG, _params, _name, _flag, _default)


#define UCT_MD_MEM_DEREG_CHECK_PARAMS(_params, _invalidate_supported) \
    if (!UCT_MD_MEM_DEREG_FIELD_VALUE(_params, memh, FIELD_MEMH, NULL)) { \
        return UCS_ERR_INVALID_PARAM; \
    } \
    if (ENABLE_PARAMS_CHECK) { \
        if (UCT_MD_MEM_DEREG_FIELD_VALUE(_params, flags, FIELD_FLAGS, 0) & \
            UCT_MD_MEM_DEREG_FLAG_INVALIDATE) { \
            if (!(_invalidate_supported)) { \
                return UCS_ERR_UNSUPPORTED; \
            } \
            if (!UCT_MD_MEM_DEREG_FIELD_VALUE(params, comp, FIELD_COMPLETION, \
                                              NULL)) { \
                return UCS_ERR_INVALID_PARAM; \
            } \
        } \
    }


typedef struct uct_md_rcache_config {
    size_t        alignment;      /**< Force address alignment */
    unsigned      event_prio;     /**< Memory events priority */
    ucs_time_t    overhead;       /**< Lookup overhead estimation */
    unsigned long max_regions;    /**< Maximal number of rcache regions */
    size_t        max_size;       /**< Maximal size of mapped memory */
    size_t        max_unreleased; /**< Threshold for triggering a cleanup */
    int           purge_on_fork;  /**< Enable/disable rcache purge on fork */
} uct_md_rcache_config_t;


extern ucs_config_field_t uct_md_config_rcache_table[];
extern const char *uct_device_type_names[];

/**
 * "Base" structure which defines MD configuration options.
 * Specific MDs extend this structure.
 */
struct uct_md_config {
    /* C standard prohibits empty structures */
    char                   __dummy;
};


typedef void (*uct_md_close_func_t)(uct_md_h md);

typedef ucs_status_t (*uct_md_query_func_t)(uct_md_h md,
                                            uct_md_attr_t *md_attr);

typedef ucs_status_t (*uct_md_mem_alloc_func_t)(uct_md_h md,
                                                size_t *length_p,
                                                void **address_p,
                                                ucs_memory_type_t mem_type,
                                                unsigned flags,
                                                const char *alloc_name,
                                                uct_mem_h *memh_p);

typedef ucs_status_t (*uct_md_mem_free_func_t)(uct_md_h md, uct_mem_h memh);

typedef ucs_status_t (*uct_md_mem_advise_func_t)(uct_md_h md,
                                                 uct_mem_h memh,
                                                 void *addr,
                                                 size_t length,
                                                 unsigned advice);

typedef ucs_status_t (*uct_md_mem_reg_func_t)(uct_md_h md, void *address,
                                              size_t length,
                                              unsigned flags,
                                              uct_mem_h *memh_p);

typedef ucs_status_t
(*uct_md_mem_dereg_func_t)(uct_md_h md,
                           const uct_md_mem_dereg_params_t *param);

typedef ucs_status_t (*uct_md_mem_query_func_t)(uct_md_h md,
                                                const void *address,
                                                size_t length,
                                                uct_md_mem_attr_t *mem_attr);

typedef ucs_status_t (*uct_md_mkey_pack_func_t)(
        uct_md_h md, uct_mem_h memh, const uct_md_mkey_pack_params_t *params,
        void *rkey_buffer);

typedef int (*uct_md_is_sockaddr_accessible_func_t)(uct_md_h md,
                                                    const ucs_sock_addr_t *sockaddr,
                                                    uct_sockaddr_accessibility_t mode);

typedef ucs_status_t (*uct_md_detect_memory_type_func_t)(uct_md_h md,
                                                         const void *addr,
                                                         size_t length,
                                                         ucs_memory_type_t *mem_type_p);


/**
 * Memory domain operations
 */
struct uct_md_ops {
    uct_md_close_func_t                  close;
    uct_md_query_func_t                  query;
    uct_md_mem_alloc_func_t              mem_alloc;
    uct_md_mem_free_func_t               mem_free;
    uct_md_mem_advise_func_t             mem_advise;
    uct_md_mem_reg_func_t                mem_reg;
    uct_md_mem_dereg_func_t              mem_dereg;
    uct_md_mem_query_func_t              mem_query;
    uct_md_mkey_pack_func_t              mkey_pack;
    uct_md_is_sockaddr_accessible_func_t is_sockaddr_accessible;
    uct_md_detect_memory_type_func_t     detect_memory_type;
};


/**
 * Memory domain
 */
struct uct_md {
    uct_md_ops_t           *ops;
    uct_component_t        *component;
};


#define UCT_MD_DEFAULT_CONFIG_INITIALIZER \
    { \
        .name        = "Default memory domain", \
        .prefix      =  "", \
        .table       = uct_md_config_table, \
        .size        = sizeof(uct_md_config_t), \
    }


/*
 * Base implementation of query_md_resources(), which returns a single md
 * resource whose name is identical to component name.
 */
ucs_status_t
uct_md_query_single_md_resource(uct_component_t *component,
                                uct_md_resource_desc_t **resources_p,
                                unsigned *num_resources_p);

ucs_status_t
uct_md_query_empty_md_resource(uct_md_resource_desc_t **resources_p,
                               unsigned *num_resources_p);


/**
 * @ingroup UCT_MD
 * @brief Allocate memory for zero-copy sends and remote access.
 *
 * Allocate memory on the memory domain. In order to use this function, MD
 * must support @ref UCT_MD_FLAG_ALLOC flag.
 *
 * @param [in]     md          Memory domain to allocate memory on.
 * @param [in,out] length_p    Points to the size of memory to allocate. Upon successful
 *                             return, filled with the actual size that was allocated,
 *                             which may be larger than the one requested. Must be >0.
 * @param [in,out] address_p   The address
 * @param [in]     mem_type    Memory type of the allocation
 * @param [in]     flags       Memory allocation flags, see @ref uct_md_mem_flags.
 * @param [in]     name        Name of the allocated region, used to track memory
 *                             usage for debugging and profiling.
 * @param [out]    memh_p      Filled with handle for allocated region.
 */
ucs_status_t uct_md_mem_alloc(uct_md_h md, size_t *length_p, void **address_p,
                              ucs_memory_type_t mem_type, unsigned flags,
                              const char *alloc_name, uct_mem_h *memh_p);

/**
 * @ingroup UCT_MD
 * @brief Release memory allocated by @ref uct_md_mem_alloc.
 *
 * @param [in]     md          Memory domain memory was allocated on.
 * @param [in]     memh        Memory handle, as returned from @ref uct_md_mem_alloc.
 */
ucs_status_t uct_md_mem_free(uct_md_h md, uct_mem_h memh);


/**
 * @brief Dummy function
 * Dummy function to emulate unpacking a remote key buffer to handle.
 *
 */
ucs_status_t uct_md_stub_rkey_unpack(uct_component_t *component,
                                     const void *rkey_buffer, uct_rkey_t *rkey_p,
                                     void **handle_p);

/**
 * Check allocation parameters and return an appropriate error if parameters
 * cannot be used for an allocation
 */
ucs_status_t uct_mem_alloc_check_params(size_t length,
                                        const uct_alloc_method_t *methods,
                                        unsigned num_methods,
                                        const uct_mem_alloc_params_t *params);


void uct_md_set_rcache_params(ucs_rcache_params_t *rcache_params,
                              const uct_md_rcache_config_t *rcache_config);

double uct_md_rcache_overhead(const uct_md_rcache_config_t *rcache_config);

extern ucs_config_field_t uct_md_config_table[];

static inline ucs_log_level_t uct_md_reg_log_lvl(unsigned flags)
{
    return (flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ? UCS_LOG_LEVEL_DIAG :
            UCS_LOG_LEVEL_ERROR;
}


void uct_md_vfs_init(uct_component_h component, uct_md_h md,
                     const char *md_name);

#endif
