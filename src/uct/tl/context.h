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


typedef struct uct_context uct_context_t;
struct uct_context {
    ucs_notifier_chain_t progress_chain;
    UCS_STATS_NODE_DECLARE(stats);
};

#define uct_component_get(_context, _name) \
    ucs_component_get(_context, _name, uct_context_t) \

#endif
