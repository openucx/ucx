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
#include <ucs/config/parser.h>
#include <string.h>


typedef struct uct_md_rcache_config {
    size_t               alignment;    /**< Force address alignment */
    unsigned             event_prio;   /**< Memory events priority */
    double               overhead;     /**< Lookup overhead estimation */
} uct_md_rcache_config_t;


extern ucs_config_field_t uct_md_config_rcache_table[];

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

typedef ucs_status_t (*uct_md_mem_dereg_func_t)(uct_md_h md, uct_mem_h memh);

typedef ucs_status_t (*uct_md_mem_query_func_t)(uct_md_h md,
                                                const void *addr,
                                                const size_t length,
                                                uct_md_mem_attr_t *mem_attr_p);

typedef ucs_status_t (*uct_md_mkey_pack_func_t)(uct_md_h md, uct_mem_h memh,
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


static UCS_F_ALWAYS_INLINE void*
uct_md_fill_md_name(uct_md_h md, void *buffer)
{
#if ENABLE_DEBUG_DATA
    memcpy(buffer, md->component->name, UCT_COMPONENT_NAME_MAX);
    return (char*)buffer + UCT_COMPONENT_NAME_MAX;
#else
    return buffer;
#endif
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
 * @brief Dummy function
 * Dummy function to emulate unpacking a remote key buffer to handle.
 *
 */
ucs_status_t uct_md_stub_rkey_unpack(uct_component_t *component,
                                     const void *rkey_buffer, uct_rkey_t *rkey_p,
                                     void **handle_p);

extern ucs_config_field_t uct_md_config_table[];

#endif
