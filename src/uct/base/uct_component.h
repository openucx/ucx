/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCT_COMPONENT_H_
#define UCT_COMPONENT_H_

#include <uct/api/uct.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/list.h>


extern ucs_list_link_t uct_md_components_list;


typedef struct uct_md_component uct_md_component_t;
struct uct_md_component {
    ucs_status_t           (*query_resources)(uct_md_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);

    ucs_status_t           (*md_open)(const char *md_name, const uct_md_config_t *config,
                                      uct_md_h *md_p);

    ucs_status_t           (*cm_open)(uct_component_h component,
                                      uct_worker_h worker, uct_cm_h *cm_p);

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
    ucs_list_link_t        tl_list;            /**< List of uct_md_registered_tl_t */
    ucs_list_link_t        list;
};


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
 * @param _cm_open       Function to open a CM.
 */
#define UCT_MD_COMPONENT_DEFINE(_mdc, _name, _query, _open, _priv, \
                                _rkey_unpack, _rkey_release, \
                                _cfg_prefix, _cfg_table, _cfg_struct, _cm_open) \
    \
    uct_md_component_t _mdc = { \
        .query_resources = _query, \
        .md_open         = _open, \
        .cm_open         = _cm_open, \
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
    } \
    UCS_CONFIG_REGISTER_TABLE(_cfg_table, _name" memory domain", _cfg_prefix, \
                              _cfg_struct)


#endif
