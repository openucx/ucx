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


/**
 * Transport memory operations
 */
struct uct_pd_ops {
    ucs_status_t (*query)(uct_pd_h pd, uct_pd_attr_t *pd_attr);

    ucs_status_t (*mem_alloc)(uct_pd_h pd, size_t *length_p, void **address_p,
                              uct_mem_h *memh_p UCS_MEMTRACK_ARG);

    ucs_status_t (*mem_free)(uct_pd_h pd, uct_mem_h memh);

    ucs_status_t (*mem_reg)(uct_pd_h pd, void *address, size_t length,
                            uct_mem_h *memh_p);

    ucs_status_t (*mem_dereg)(uct_pd_h pd, uct_mem_h memh);

    ucs_status_t (*rkey_pack)(uct_pd_h pd, uct_mem_h memh, void *rkey_buffer);

    ucs_status_t (*rkey_unpack)(uct_pd_h pd, void *rkey_buffer,
                                uct_rkey_bundle_t *rkey_ob);

    void         (*rkey_release)(uct_pd_h pd, uct_rkey_bundle_t *rkey_ob);

};


typedef struct uct_context_tl_info {
    uct_tl_ops_t           *ops;
    const char             *name;
    ucs_config_field_t     *iface_config_table;
    size_t                 iface_config_size;
    const char             *config_prefix;
} uct_context_tl_info_t;


typedef struct uct_context uct_context_t;
struct uct_context {
    unsigned               num_tls;
    uct_context_tl_info_t  *tls;
};


typedef struct uct_worker uct_worker_t;
struct uct_worker {
    uct_context_h          context;
    ucs_notifier_chain_t   progress_chain;
    ucs_thread_mode_t      thread_mode;
};


#define uct_component_get(_context, _name) \
    ucs_component_get(_context, _name, uct_context_t) \


extern ucs_config_field_t uct_iface_config_table[];
extern const char *uct_alloc_method_names[];


/**
 * Add a transport to the list of existing transports on this context.
 *
 * @param context        UCT context to add the transport to.
 * @param tl_name        Transport name.
 * @param config_table   Defines transport configuration options.
 * @param config_size    Transport configuration struct size.
 * @param tl_ops         Pointer to transport operations. Must be valid as long
 *                       as  context is still alive.
 */
ucs_status_t uct_register_tl(uct_context_h context, const char *tl_name,
                             ucs_config_field_t *config_table, size_t config_size,
                             const char *config_prefix, uct_tl_ops_t *tl_ops);

#endif
