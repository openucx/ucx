/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_CONTEXT_H
#define UCT_CONTEXT_H

#include <uct/api/uct.h>
#include <ucs/datastruct/notifier.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/component.h>
#include <ucs/config/parser.h>


typedef struct uct_pd_component {
    ucs_status_t           (*query_resources)(uct_pd_resource_desc_t **resources_p,
                                              unsigned *num_resources_p);

    ucs_status_t           (*pd_open)(const char *pd_name, uct_pd_h *pd_p);

    ucs_status_t           (*rkey_unpack)(const void *rkey_buffer,
                                          uct_rkey_t *rkey_p, void **handle_p);

    void                   (*rkey_release)(uct_rkey_t rkey, void *handle);

    const char             name[UCT_PD_COMPONENT_NAME_MAX];
    size_t                 rkey_buf_size;
    ucs_list_link_t        tl_list;
    ucs_list_link_t        list;
} uct_pd_component_t;


/**
 * Define a PD component.
 *
 * @param _pdc           PD component structure to initialize.
 * @param _name          PD component name.
 * @param _query         Function to query PD resources.
 * @param _open          Function to open a PD.
 * @param _rkey_buf_size Size of buffer needed for packed rkey.
 * @param _rkey_unpack   Function to unpack a remote key buffer to handle.
 * @param _rkey_release  Function to release a remote key handle.
 */
#define UCT_PD_COMPONENT_DEFINE(_pdc, _name, _query, _open, \
                                _rkey_buf_size, _rkey_unpack, _rkey_release) \
    \
    uct_pd_component_t _pdc = { \
        .query_resources = _query, \
        .pd_open         = _open, \
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


typedef struct uct_worker uct_worker_t;
struct uct_worker {
    ucs_async_context_t    *async;
    ucs_notifier_chain_t   progress_chain;
    ucs_thread_mode_t      thread_mode;
};


ucs_status_t uct_single_pd_resource(uct_pd_component_t *pdc,
                                    uct_pd_resource_desc_t **resources_p,
                                    unsigned *num_resources_p);


extern ucs_list_link_t uct_pd_components_list;


#endif
