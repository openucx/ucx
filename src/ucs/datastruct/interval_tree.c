/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "interval_tree.h"
#include <ucs/datastruct/mpool.inl>
#include <ucs/sys/math.h>


static UCS_F_ALWAYS_INLINE ucs_interval_node_t *
ucs_interval_tree_node(ucs_rbtree_node_t *rb_node)
{
    return ucs_derived_of(rb_node, ucs_interval_node_t);
}

static ucs_interval_node_t *
ucs_interval_tree_node_create(ucs_interval_tree_t *tree, uint64_t start,
                              uint64_t end)
{
    ucs_interval_node_t *node;

    node = ucs_mpool_get_inline(tree->mpool);
    if (ucs_unlikely(node == NULL)) {
        return NULL;
    }

    node->start = start;
    node->end   = end;

    tree->num_nodes++;
    tree->total_size += (end - start);
    return node;
}

static void ucs_interval_tree_node_free(ucs_interval_tree_t *tree,
                                        ucs_interval_node_t *node)
{
    ucs_assertv(tree->num_nodes > 0, "tree=%p, node=%p", tree, node);
    tree->total_size -= (node->end - node->start);
    tree->num_nodes--;
    ucs_mpool_put_inline(node);
}

void ucs_interval_tree_init(ucs_interval_tree_t *tree, ucs_mpool_t *mpool)
{
    ucs_rbtree_init(&tree->rb);
    tree->mpool      = mpool;
    tree->num_nodes  = 0;
    tree->total_size = 0;
}

static void ucs_interval_tree_cleanup_recursive(ucs_interval_tree_t *tree,
                                                ucs_rbtree_node_t *rb_node)
{
    if (rb_node == NULL) {
        return;
    }

    ucs_interval_tree_cleanup_recursive(tree, rb_node->left);
    ucs_interval_tree_cleanup_recursive(tree, rb_node->right);
    ucs_interval_tree_node_free(tree, ucs_interval_tree_node(rb_node));
}

void ucs_interval_tree_cleanup(ucs_interval_tree_t *tree)
{
    ucs_interval_tree_cleanup_recursive(tree, tree->rb.root);
    ucs_rbtree_init(&tree->rb);
}

/**
 * Find any single node that overlaps or is adjacent to @a range.
 * Returns NULL if none found. Does not modify the tree.
 */
static ucs_interval_node_t *
ucs_interval_tree_find_overlap(ucs_rbtree_node_t *rb_node,
                               ucs_interval_tree_range_t range)
{
    ucs_interval_node_t *node, *found;

    if (rb_node == NULL) {
        return NULL;
    }

    node = ucs_interval_tree_node(rb_node);

    if ((range.start <= (node->end + 1)) && (node->start <= (range.end + 1))) {
        return node;
    }

    found = ucs_interval_tree_find_overlap(rb_node->left, range);
    if (found != NULL) {
        return found;
    }

    /* Right subtree: all nodes have start >= node->start, so if
     * node->start > range.end + 1, no right descendant can overlap. */
    if (node->start <= (range.end + 1)) {
        return ucs_interval_tree_find_overlap(rb_node->right, range);
    }

    return NULL;
}

/**
 * Iteratively find and remove all overlapping/adjacent nodes, expanding the
 * merged range. Each removal properly rebalances the RB tree before the next
 * search restarts from the root.
 */
static void ucs_interval_tree_remove_overlapping(ucs_interval_tree_t *tree,
                                                 ucs_interval_tree_range_t *range)
{
    ucs_interval_node_t *overlap;

    while ((overlap = ucs_interval_tree_find_overlap(tree->rb.root,
                                                     *range)) != NULL) {
        range->start = ucs_min(range->start, overlap->start);
        range->end   = ucs_max(range->end, overlap->end);
        ucs_rbtree_remove(&tree->rb, &overlap->super);
        ucs_interval_tree_node_free(tree, overlap);
    }
}

ucs_status_t ucs_interval_tree_insert_slow(ucs_interval_tree_t *tree,
                                           ucs_interval_tree_range_t range)
{
    ucs_interval_tree_range_t merged = range;
    ucs_rbtree_node_t *parent        = NULL;
    ucs_rbtree_node_t **link         = &tree->rb.root;
    ucs_interval_node_t *new_node;

    ucs_interval_tree_remove_overlapping(tree, &merged);

    new_node = ucs_interval_tree_node_create(tree, merged.start, merged.end);
    if (ucs_unlikely(new_node == NULL)) {
        return UCS_ERR_NO_MEMORY;
    }

    while (*link != NULL) {
        parent = *link;
        link   = (merged.start < ucs_interval_tree_node(parent)->start) ?
                         &parent->left :
                         &parent->right;
    }

    ucs_rbtree_insert_at(&tree->rb, parent, link, &new_node->super);
    return UCS_OK;
}

int ucs_interval_tree_pop_any(ucs_interval_tree_t *tree,
                              ucs_interval_tree_range_t *range)
{
    ucs_rbtree_node_t *rb_node = ucs_rbtree_first(&tree->rb);
    ucs_interval_node_t *node;

    if (rb_node == NULL) {
        return 0;
    }

    node = ucs_interval_tree_node(rb_node);

    range->start = node->start;
    range->end   = node->end;
    ucs_rbtree_remove(&tree->rb, &node->super);
    ucs_interval_tree_node_free(tree, node);
    return 1;
}
