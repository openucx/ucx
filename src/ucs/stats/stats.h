/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCS_STATS_H_
#define UCS_STATS_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

void ucs_stats_init();
void ucs_stats_cleanup();
void ucs_stats_dump();
int ucs_stats_is_active();
#include "stats_fwd.h"
#if ENABLE_STATS

#include "libstats.h"

/**
 * Allocate statistics node.
 *
 * @param p_node         Filled with a pointer to new node, or NULL if stats are off.
 * @param cls            Node class / type.
 * @param parent         Parent node.
 * @param name           Node name format.
 */
ucs_status_t ucs_stats_node_alloc(ucs_stats_node_t** p_node, ucs_stats_class_t *cls,
                                 ucs_stats_node_t *parent, const char *name, ...);
void ucs_stats_node_free(ucs_stats_node_t *node);

#define UCS_STATS_ARG(_arg) , _arg

#define UCS_STATS_RVAL(_rval) _rval

#define UCS_STATS_NODE_DECLARE(_node) \
    ucs_stats_node_t* _node

#define UCS_STATS_NODE_ALLOC(_p_node, _class, _parent, ...) \
    ucs_stats_node_alloc(_p_node, _class, _parent, ## __VA_ARGS__ , "")

#define UCS_STATS_NODE_FREE(_node) \
    ucs_stats_node_free(_node)

#define UCS_STATS_UPDATE_COUNTER(_node, _index, _delta) \
    if (((_delta) != 0) && ((_node) != NULL)) { \
        (_node)->counters[(_index)] += (_delta); \
    }

#define UCS_STATS_SET_COUNTER(_node, _index, _value) \
    if ((_node) != NULL) { \
        (_node)->counters[(_index)] = (_value); \
    }

#define UCS_STATS_GET_COUNTER(_node, _index) \
    (((_node) != NULL) ?  \
    (_node)->counters[(_index)] : 0)

#define UCS_STATS_UPDATE_MAX(_node, _index, _value) \
    if ((_node) != NULL) { \
        if ((_node)->counters[(_index)] < (_value)) { \
            (_node)->counters[(_index)] = (_value); \
        } \
    }

#define UCS_STATS_START_TIME(_start_time) \
    { \
        _start_time = ucs_get_time(); \
        ucs_compiler_fence(); \
    }

#define UCS_STATS_UPDATE_TIME(_node, _index, _start_time) \
    { \
        ucs_compiler_fence(); \
        UCS_STATS_UPDATE_COUNTER(_node, _index, \
                                 (long)ucs_time_to_nsec(ucs_get_time() - (_start_time))); \
    }

#define UCS_STATS_SET_TIME(_node, _index, _start_time) \
   { \
        ucs_compiler_fence(); \
        UCS_STATS_SET_COUNTER(_node, _index, \
                              (long)ucs_time_to_nsec(ucs_get_time() - (_start_time))); \
   }

#else

#define UCS_STATS_ARG(_arg)
#define UCS_STATS_RVAL(_rval) NULL
#define UCS_STATS_NODE_DECLARE(_node)
#define UCS_STATS_NODE_ALLOC(_p_node, _class, _parent, ...) UCS_OK
#define UCS_STATS_NODE_FREE(_node)
#define UCS_STATS_UPDATE_COUNTER(_node, _index, _delta)
#define UCS_STATS_SET_COUNTER(_node, _index, _value)
#define UCS_STATS_GET_COUNTER(_node, _index)    0
#define UCS_STATS_UPDATE_MAX(_node, _index, _value)
#define UCS_STATS_START_TIME(_start_time)
#define UCS_STATS_UPDATE_TIME(_node, _index, _start_time)
#define UCS_STATS_SET_TIME(_node, _index, _start_time)

#endif

END_C_DECLS

#endif
