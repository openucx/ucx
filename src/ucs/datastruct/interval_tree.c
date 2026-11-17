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

    tree->total_size += (end - start);
    return node;
}

static void ucs_interval_tree_node_free(ucs_interval_tree_t *tree,
                                        ucs_interval_node_t *node)
{
    tree->total_size -= (node->end - node->start);
    ucs_mpool_put_inline(node);
}

void ucs_interval_tree_init(ucs_interval_tree_t *tree, ucs_mpool_t *mpool)
{
    ucs_rbtree_init(&tree->rb, NULL);
    tree->mpool      = mpool;
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
    ucs_rbtree_init(&tree->rb, NULL);
}

/*
 * Nodes are disjoint and non-adjacent, since inserts merge them. So if the
 * range ends before a node with a gap, only the left subtree can overlap it,
 * and if it starts after the node with a gap, only the right subtree can. When
 * the descent reaches a NULL link, that is where the merged range belongs.
 */
ucs_status_t ucs_interval_tree_insert_slow(ucs_interval_tree_t *tree,
                                           ucs_interval_tree_range_t range)
{
    ucs_rbtree_node_t *parent = NULL;
    ucs_rbtree_node_t **link  = &tree->rb.root;
    ucs_interval_node_t *node;

    while (*link != NULL) {
        parent = *link;
        node   = ucs_interval_tree_node(parent);
        if ((range.end + 1) < node->start) {
            link = &parent->left;
        } else if ((node->end + 1) < range.start) {
            link = &parent->right;
        } else {
            /* Merge the node and search again from the rebalanced root */
            range.start = ucs_min(range.start, node->start);
            range.end   = ucs_max(range.end, node->end);
            ucs_rbtree_remove(&tree->rb, parent);
            ucs_interval_tree_node_free(tree, node);
            parent = NULL;
            link   = &tree->rb.root;
        }
    }

    node = ucs_interval_tree_node_create(tree, range.start, range.end);
    if (ucs_unlikely(node == NULL)) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_rbtree_insert_at(&tree->rb, parent, link, &node->super);
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
