/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_PD_H_
#define UCT_PD_H_

#include "uct_iface.h"

#include <uct/api/uct.h>
#include <ucs/datastruct/notifier.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/component.h>
#include <ucs/config/parser.h>


typedef struct uct_pd_component uct_pd_component_t;
struct uct_pd_component {
    ucs_status_t           (*query_resources)(uct_pd_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);

    ucs_status_t           (*pd_open)(const char *pd_name, const uct_pd_config_t *config,
                                      uct_pd_h *pd_p);

    ucs_status_t           (*rkey_unpack)(uct_pd_component_t *pdc, const void *rkey_buffer,
                                          uct_rkey_t *rkey_p, void **handle_p);

    ucs_status_t           (*rkey_release)(uct_pd_component_t *pdc, uct_rkey_t rkey,
                                           void *handle);

    const char             name[UCT_PD_COMPONENT_NAME_MAX];
    void                   *priv;
    const char             *cfg_prefix;        /**< Prefix for configuration environment vars */
    ucs_config_field_t     *pd_config_table;   /**< Defines PD configuration options */
    size_t                 pd_config_size;     /**< PD configuration structure size */
    size_t                 rkey_buf_size;
    ucs_list_link_t        tl_list;            /* List of uct_pd_registered_tl_t */
    ucs_list_link_t        list;
};


/**
 * "Base" structure which defines PD configuration options.
 * Specific PDs extend this structure.
 */
struct uct_pd_config {
};


/**
 * PD->Transport
 */
typedef struct uct_pd_registered_tl {
    ucs_list_link_t        list;
    uct_tl_component_t     *tl;
} uct_pd_registered_tl_t;


/**
 * Define a PD component.
 *
 * @param _pdc           PD component structure to initialize.
 * @param _name          PD component name.
 * @param _query         Function to query PD resources.
 * @param _open          Function to open a PD.
 * @param _priv          Custom private data.
 * @param _rkey_buf_size Size of buffer needed for packed rkey.
 * @param _rkey_unpack   Function to unpack a remote key buffer to handle.
 * @param _rkey_release  Function to release a remote key handle.
 * @param _cfg_prefix    Prefix for configuration environment vars.
 * @param _cfg_table     Defines the PDC's configuration values.
 * @param _cfg_struct    PDC configuration structure.
 */
#define UCT_PD_COMPONENT_DEFINE(_pdc, _name, _query, _open, _priv, \
                                _rkey_buf_size, _rkey_unpack, _rkey_release, \
                                _cfg_prefix, _cfg_table, _cfg_struct) \
    \
    uct_pd_component_t _pdc = { \
        .query_resources = _query, \
        .pd_open         = _open, \
        .cfg_prefix      = _cfg_prefix, \
        .pd_config_table = _cfg_table, \
        .pd_config_size  = sizeof(_cfg_struct), \
        .priv            = _priv, \
        .rkey_unpack     = _rkey_unpack, \
        .rkey_release    = _rkey_release, \
        .name            = _name, \
        .rkey_buf_size   = _rkey_buf_size, \
        .tl_list         = { &_pdc.tl_list, &_pdc.tl_list } \
    }; \
    UCS_STATIC_INIT { \
        ucs_list_add_tail(&uct_pd_components_list, &_pdc.list); \
    }


/**
 * Add a transport component to a pd component
 * (same transport component can be added to multiple pd components).
 *
 * @param _pdc           Pointer to PD component to add the TL component to.
 * @param _tlc           Pointer to TL component.
 */
#define UCT_PD_REGISTER_TL(_pdc, _tlc) \
    UCS_STATIC_INIT { \
        static uct_pd_registered_tl_t reg; \
        reg.tl = (_tlc); \
        ucs_list_add_tail(&(_pdc)->tl_list, &reg.list); \
    }


/**
 * Protection domain operations
 */
struct uct_pd_ops {
    void         (*close)(uct_pd_h pd);

    ucs_status_t (*query)(uct_pd_h pd, uct_pd_attr_t *pd_attr);

    ucs_status_t (*mem_alloc)(uct_pd_h pd, size_t *length_p, void **address_p,
                              uct_mem_h *memh_p UCS_MEMTRACK_ARG);

    ucs_status_t (*mem_free)(uct_pd_h pd, uct_mem_h memh);

    ucs_status_t (*mem_reg)(uct_pd_h pd, void *address, size_t length,
                            uct_mem_h *memh_p);

    ucs_status_t (*mem_dereg)(uct_pd_h pd, uct_mem_h memh);

    ucs_status_t (*mkey_pack)(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer);
};


/**
 * Protection domain
 */
struct uct_pd {
    uct_pd_ops_t           *ops;
    uct_pd_component_t     *component;
};


/**
 * Transport-specific data on a worker
 */
typedef struct uct_worker_tl_data {
    ucs_list_link_t        list;
    uint32_t               refcount;
    uint32_t               key;
    void                   *ptr;
} uct_worker_tl_data_t;


typedef struct uct_worker uct_worker_t;
struct uct_worker {
    ucs_async_context_t    *async;
    ucs_notifier_chain_t   progress_chain;
    ucs_thread_mode_t      thread_mode;
    ucs_list_link_t        tl_data;
};


ucs_status_t uct_single_pd_resource(uct_pd_component_t *pdc,
                                    uct_pd_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);


#define uct_worker_tl_data_get(_worker, _key, _type, _cmp_fn, _init_fn, ...) \
    ({ \
        uct_worker_tl_data_t *data; \
        \
        ucs_list_for_each(data, &(_worker)->tl_data, list) { \
            if ((data->key == (_key)) && _cmp_fn(ucs_derived_of(data, _type), \
                                                 ## __VA_ARGS__)) \
            { \
                ++data->refcount; \
                break; \
            } \
        } \
        \
        if (&data->list == &(_worker)->tl_data) { \
            data = ucs_malloc(sizeof(_type), UCS_PP_QUOTE(_type)); \
            if (data != NULL) { \
                data->key      = (_key); \
                data->refcount = 1; \
                _init_fn(ucs_derived_of(data, _type), ## __VA_ARGS__); \
                ucs_list_add_tail(&(_worker)->tl_data, &data->list); \
            } \
        } \
        ucs_derived_of(data, _type); \
    })

#define uct_worker_tl_data_put(_data, _cleanup_fn, ...) \
    { \
        uct_worker_tl_data_t *data = (uct_worker_tl_data_t*)(_data); \
        if (--data->refcount == 0) { \
            ucs_list_del(&data->list); \
            _cleanup_fn((_data), ## __VA_ARGS__); \
            ucs_free(data); \
        } \
    }


extern ucs_list_link_t uct_pd_components_list;
extern ucs_config_field_t uct_pd_config_table[];

#endif
