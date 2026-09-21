/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "interval_map.h"

#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>

#include <inttypes.h>


static UCS_F_ALWAYS_INLINE ucs_interval_map_node_t *
ucs_interval_map_node(ucs_rbtree_node_t *rb_node)
{
    return ucs_derived_of(rb_node, ucs_interval_map_node_t);
}

static UCS_F_ALWAYS_INLINE uint64_t
ucs_interval_map_subtree_max(const ucs_rbtree_node_t *rb_node)
{
    return (rb_node == NULL) ?
                   0 :
                   ucs_derived_of(rb_node, ucs_interval_map_node_t)->max_end;
}

/*
 * Recompute one node's maximum end from its own interval and its children.
 */
static void ucs_interval_map_augment(ucs_rbtree_node_t *rb_node)
{
    ucs_interval_map_node_t *node = ucs_interval_map_node(rb_node);
    uint64_t max_end;

    max_end       = ucs_max(node->end,
                            ucs_interval_map_subtree_max(rb_node->left));
    node->max_end = ucs_max(max_end,
                            ucs_interval_map_subtree_max(rb_node->right));
}

void ucs_interval_map_init(ucs_interval_map_t *map)
{
    ucs_rbtree_init(&map->rb, ucs_interval_map_augment);
    map->num_nodes = 0;
}

void ucs_interval_map_insert(ucs_interval_map_t *map,
                             ucs_interval_map_node_t *node, uint64_t start,
                             uint64_t end)
{
    ucs_rbtree_node_t *parent = NULL;
    ucs_rbtree_node_t **link  = &map->rb.root;

    ucs_assertv(start < end, "start=%" PRIu64 " end=%" PRIu64, start, end);

    while (*link != NULL) {
        parent = *link;
        /* Equal starts go right, so duplicates stay distinct */
        link   = (start < ucs_interval_map_node(parent)->start) ?
                           &parent->left :
                           &parent->right;
    }

    node->start = start;
    node->end   = end;
    ucs_rbtree_insert_at(&map->rb, parent, link, &node->super);
    map->num_nodes++;
}

void ucs_interval_map_remove(ucs_interval_map_t *map,
                             ucs_interval_map_node_t *node)
{
    ucs_assertv(map->num_nodes > 0, "map=%p node=%p", map, node);
    ucs_rbtree_remove(&map->rb, &node->super);
    map->num_nodes--;
}

ucs_interval_map_node_t *
ucs_interval_map_find_containing(const ucs_interval_map_t *map, uint64_t start,
                                 uint64_t end)
{
    ucs_rbtree_node_t *rb_node = map->rb.root;
    ucs_interval_map_node_t *node;

    ucs_assertv(start < end, "start=%" PRIu64 " end=%" PRIu64, start, end);

    while (rb_node != NULL) {
        node = ucs_interval_map_node(rb_node);

        if (node->start > start) {
            /* This node and its whole right subtree start too late */
            rb_node = rb_node->left;
            continue;
        }

        if (node->end >= end) {
            return node;
        }

        /* Every node on the left starts at or before this one, so all of them
         * satisfy the start condition; descend if any reaches far enough. */
        if (ucs_interval_map_subtree_max(rb_node->left) >= end) {
            rb_node = rb_node->left;
        } else {
            rb_node = rb_node->right;
        }
    }

    return NULL;
}

static void
ucs_interval_map_foreach_node(ucs_rbtree_node_t *rb_node, uint64_t start,
                              uint64_t end, ucs_interval_map_cb_t cb, void *arg)
{
    ucs_interval_map_node_t *node;

    if ((rb_node == NULL) ||
        (ucs_interval_map_subtree_max(rb_node) <= start)) {
        return;
    }

    ucs_interval_map_foreach_node(rb_node->left, start, end, cb, arg);

    node = ucs_interval_map_node(rb_node);
    if (node->start >= end) {
        /* This node and everything to its right start at or after the range */
        return;
    }

    if (node->end > start) {
        cb(node, arg);
    }

    ucs_interval_map_foreach_node(rb_node->right, start, end, cb, arg);
}

void ucs_interval_map_foreach_overlapping(const ucs_interval_map_t *map,
                                          uint64_t start, uint64_t end,
                                          ucs_interval_map_cb_t cb, void *arg)
{
    ucs_assertv(start < end, "start=%" PRIu64 " end=%" PRIu64, start, end);

    ucs_interval_map_foreach_node(map->rb.root, start, end, cb, arg);
}
