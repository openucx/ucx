/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_CONTEXT_H
#define UCT_CONTEXT_H

#include <uct/api/uct.h>
#include <ucs/type/component.h>
#include <ucs/type/callback.h>
#include <ucs/config/parser.h>


/**
 * Transport memory operations
 */
struct uct_pd_ops {
    ucs_status_t (*query)(uct_pd_h pd, uct_pd_attr_t *pd_attr);

    /* TODO
     * - support "mem attach", MPI-3 style, e.g by passing rkey
     */
    ucs_status_t (*mem_map)(uct_pd_h pd, void *address, size_t length,
                            unsigned flags, uct_lkey_t *lkey_p);

    ucs_status_t (*mem_alloc)(uct_pd_h pd, size_t *length_p, unsigned flags,
                              void **address_p, uct_lkey_t *lkey_p UCS_MEMTRACK_ARG);

    ucs_status_t (*mem_unmap)(uct_pd_h pd, uct_lkey_t lkey);

    ucs_status_t (*rkey_pack)(uct_pd_h pd, uct_lkey_t lkey, void *rkey_buffer);
};


typedef struct uct_context_tl_info {
    uct_tl_ops_t           *ops;
    const char             *name;
    ucs_config_field_t     *iface_config_table;
    size_t                 iface_config_size;
} uct_context_tl_info_t;


typedef struct uct_context uct_context_t;
struct uct_context {
    ucs_notifier_chain_t   progress_chain;
    unsigned               num_tls;
    uct_context_tl_info_t  *tls;
    UCS_STATS_NODE_DECLARE(stats);
};


#define uct_component_get(_context, _name) \
    ucs_component_get(_context, _name, uct_context_t) \


extern ucs_config_field_t uct_iface_config_table[];

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
                             uct_tl_ops_t *tl_ops);




#endif
