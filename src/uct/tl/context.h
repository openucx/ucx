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


typedef struct uct_context_tl_info {
    uct_tl_ops_t           *ops;
    const char             *name;
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


/**
 * Add a transport to the list of existing transports on this context.
 *
 * @param context  UCT context to add the transport to.
 * @param tl_name  Transport name.
 * @param tl_ops   Pointer to transport operations. Must be valid as long as
 *                 context is still alive.
 */
ucs_status_t uct_register_tl(uct_context_h context, const char *tl_name,
                             uct_tl_ops_t *tl_ops);




#endif
