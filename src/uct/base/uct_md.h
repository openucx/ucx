/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_MD_H_
#define UCT_MD_H_

#include "uct_iface.h"

#include <uct/api/uct.h>
#include <ucs/type/component.h>
#include <ucs/config/parser.h>


typedef struct uct_md_component uct_md_component_t;
struct uct_md_component {
    ucs_status_t           (*query_resources)(uct_md_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);

    ucs_status_t           (*md_open)(const char *md_name, const uct_md_config_t *config,
                                      uct_md_h *md_p);

    ucs_status_t           (*rkey_unpack)(uct_md_component_t *mdc, const void *rkey_buffer,
                                          uct_rkey_t *rkey_p, void **handle_p);

    ucs_status_t           (*rkey_ptr)(uct_md_component_t *mdc, uct_rkey_t rkey, void *handle,
                                       uint64_t raddr, void **laddr_p);

    ucs_status_t           (*rkey_release)(uct_md_component_t *mdc, uct_rkey_t rkey,
                                           void *handle);

    const char             name[UCT_MD_COMPONENT_NAME_MAX];
    void                   *priv;
    const char             *cfg_prefix;        /**< Prefix for configuration environment vars */
    ucs_config_field_t     *md_config_table;   /**< Defines MD configuration options */
    size_t                 md_config_size;     /**< MD configuration structure size */
    ucs_list_link_t        tl_list;            /* List of uct_md_registered_tl_t */
    ucs_list_link_t        list;
};


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
#ifdef __cplusplus
    char __dummy;
#endif
};


/**
 * MD->Transport
 */
typedef struct uct_md_registered_tl {
    ucs_list_link_t        list;
    uct_tl_component_t     *tl;
} uct_md_registered_tl_t;


/**
 * Define a MD component.
 *
 * @param _mdc           MD component structure to initialize.
 * @param _name          MD component name.
 * @param _query         Function to query MD resources.
 * @param _open          Function to open a MD.
 * @param _priv          Custom private data.
 * @param _rkey_unpack   Function to unpack a remote key buffer to handle.
 * @param _rkey_release  Function to release a remote key handle.
 * @param _cfg_prefix    Prefix for configuration environment vars.
 * @param _cfg_table     Defines the MDC's configuration values.
 * @param _cfg_struct    MDC configuration structure.
 */
#define UCT_MD_COMPONENT_DEFINE(_mdc, _name, _query, _open, _priv, \
                                _rkey_unpack, _rkey_release, \
                                _cfg_prefix, _cfg_table, _cfg_struct) \
    \
    uct_md_component_t _mdc = { \
        .query_resources = _query, \
        .md_open         = _open, \
        .cfg_prefix      = _cfg_prefix, \
        .md_config_table = _cfg_table, \
        .md_config_size  = sizeof(_cfg_struct), \
        .priv            = _priv, \
        .rkey_unpack     = _rkey_unpack, \
        .rkey_ptr        = ucs_empty_function_return_unsupported, \
        .rkey_release    = _rkey_release, \
        .name            = _name, \
        .tl_list         = { &_mdc.tl_list, &_mdc.tl_list } \
    }; \
    UCS_STATIC_INIT { \
        ucs_list_add_tail(&uct_md_components_list, &_mdc.list); \
    }


/**
 * Add a transport component to a md component
 * (same transport component can be added to multiple md components).
 *
 * @param _mdc           Pointer to MD component to add the TL component to.
 * @param _tlc           Pointer to TL component.
 */
#define UCT_MD_REGISTER_TL(_mdc, _tlc) \
    UCS_STATIC_INIT { \
        static uct_md_registered_tl_t reg; \
        reg.tl = (_tlc); \
        ucs_list_add_tail(&(_mdc)->tl_list, &reg.list); \
    }


/**
 * Memory domain operations
 */
struct uct_md_ops {
    void         (*close)(uct_md_h md);

    ucs_status_t (*query)(uct_md_h md, uct_md_attr_t *md_attr);

    ucs_status_t (*mem_alloc)(uct_md_h md, size_t *length_p, void **address_p,
                              unsigned flags, uct_mem_h *memh_p UCS_MEMTRACK_ARG);

    ucs_status_t (*mem_free)(uct_md_h md, uct_mem_h memh);
    ucs_status_t (*mem_advise)(uct_md_h md, uct_mem_h memh, void *addr,
                               size_t length, unsigned advice);

    ucs_status_t (*mem_reg)(uct_md_h md, void *address, size_t length,
                            unsigned flags, uct_mem_h *memh_p);

    ucs_status_t (*mem_dereg)(uct_md_h md, uct_mem_h memh);

    ucs_status_t (*mkey_pack)(uct_md_h md, uct_mem_h memh, void *rkey_buffer);

    int          (*is_sockaddr_accessible)(uct_md_h md, const ucs_sock_addr_t *sockaddr,
                                           uct_sockaddr_accessibility_t mode);

    int          (*is_mem_type_owned)(uct_md_h md, void *addr, size_t length);
};


/**
 * Memory domain
 */
struct uct_md {
    uct_md_ops_t           *ops;
    uct_md_component_t     *component;
};


static UCS_F_ALWAYS_INLINE void*
uct_md_fill_md_name(uct_md_h md, void *buffer)
{
    memcpy(buffer, md->component->name, UCT_MD_COMPONENT_NAME_MAX);
    return (char*)buffer + UCT_MD_COMPONENT_NAME_MAX;
}


ucs_status_t uct_single_md_resource(uct_md_component_t *mdc,
                                    uct_md_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);

/**
 * @brief Dummy function
 * Dummy function to emulate unpacking a remote key buffer to handle.
 *
 */
ucs_status_t uct_md_stub_rkey_unpack(uct_md_component_t *mdc,
                                     const void *rkey_buffer, uct_rkey_t *rkey_p,
                                     void **handle_p);

uct_tl_component_t *uct_find_tl_on_md(uct_md_component_t *mdc,
                                      uint64_t md_flags,
                                      const char *tl_name);


extern ucs_list_link_t uct_md_components_list;
extern ucs_config_field_t uct_md_config_table[];

#endif
