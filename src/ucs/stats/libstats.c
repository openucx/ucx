/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "libstats.h"

#include <ucs/debug/log.h>
#include <errno.h>
#include <string.h>


#define UCS_STATS_NAME_VALID_CHARS \
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"


static ucs_status_t ucs_stats_name_check(const char *name)
{
    size_t length, valid_length;

    length = strlen(name);
    if (length > UCS_STAT_NAME_MAX) {
        ucs_error("stats name '%s' is too long (%zu)", name, length);
        return UCS_ERR_INVALID_PARAM;
    }

    valid_length = strspn(name, UCS_STATS_NAME_VALID_CHARS);
    if (valid_length != length) {
        ucs_error("stats name '%s' contains invalid character at offset %zu",
                  name, valid_length);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;;
}

ucs_status_t ucs_stats_node_initv(ucs_stats_node_t *node, ucs_stats_class_t *cls,
                                  const char *name, va_list ap)
{
    ucs_status_t status;
    unsigned i;

    /* Check class */
    status = ucs_stats_name_check(cls->name);
    if (status != UCS_OK) {
        return status;
    }
    for (i = 0; i < cls->num_counters; ++i) {
        status = ucs_stats_name_check(cls->counter_names[i]);
        if (status != UCS_OK) {
            return status;
        }
    }

    /* Set up node */
    node->cls = cls;
    vsnprintf(node->name, UCS_STAT_NAME_MAX, name, ap);
    ucs_list_head_init(&node->children[UCS_STATS_INACTIVE_CHILDREN]);
    ucs_list_head_init(&node->children[UCS_STATS_ACTIVE_CHILDREN]);
    memset(node->counters, 0, cls->num_counters * sizeof(ucs_stats_counter_t));

    return UCS_OK;
}

