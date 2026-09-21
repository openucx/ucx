/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_INTERVAL_TREE_H_
#define UCS_INTERVAL_TREE_H_

#include <ucs/datastruct/rbtree.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>
#include <ucs/type/status.h>
#include <ucs/datastruct/mpool.h>
#include <stdint.h>
#include <stddef.h>


/**
 * Interval range (start, end pair)
 */
typedef struct {
    uint64_t start;
    uint64_t end;
} ucs_interval_tree_range_t;


/**
 * Interval tree node structure
 *
 * Merges overlapping or touching intervals into one node.
 */
typedef struct ucs_interval_node {
    ucs_rbtree_node_t super; /**< Balancing links, must be first */
    uint64_t          start; /**< Start of interval */
    uint64_t          end;   /**< End of interval */
} ucs_interval_node_t;


typedef struct {
    ucs_rbtree_t rb;         /**< Balanced tree ordered by 'start' */
    ucs_mpool_t  *mpool;     /**< Memory pool for node allocation */
    size_t       total_size; /**< Sum of (end - start) across all nodes */
} ucs_interval_tree_t;


/**
 * Return the interval node embedding a balancing node
 */
static UCS_F_ALWAYS_INLINE ucs_interval_node_t *
ucs_interval_tree_node(ucs_rbtree_node_t *rb_node)
{
    return ucs_derived_of(rb_node, ucs_interval_node_t);
}


/**
 * Return the tree's root node, or NULL if the tree is empty.
 */
static UCS_F_ALWAYS_INLINE ucs_interval_node_t *
ucs_interval_tree_root(const ucs_interval_tree_t *tree)
{
    return ucs_interval_tree_node(tree->rb.root);
}


/**
 * Initialize an interval tree
 *
 * @param [in]  tree   Interval tree to initialize
 * @param [in]  mpool  Memory pool for node allocation
 */
void ucs_interval_tree_init(ucs_interval_tree_t *tree, ucs_mpool_t *mpool);


/**
 * Cleanup and free all nodes in the interval tree
 *
 * @param [in]  tree  Interval tree to clean up
 */
void ucs_interval_tree_cleanup(ucs_interval_tree_t *tree);

/* TODO: remove this forward declaration when file is refactored to minimize 
 * exposing private logic */
ucs_status_t ucs_interval_tree_insert_slow(ucs_interval_tree_t *tree,
                                           ucs_interval_tree_range_t range);


/**
 * Insert a new interval into the tree
 *
 * @param [in]  tree   Interval tree
 * @param [in]  range  Interval range to insert
 *
 * @return UCS_OK on success, or error code on failure
 */
static UCS_F_ALWAYS_INLINE ucs_status_t ucs_interval_tree_insert(
        ucs_interval_tree_t *tree, ucs_interval_tree_range_t range)
{
    ucs_interval_node_t *root = ucs_interval_tree_root(tree);

    ucs_assertv(range.start <= (range.end + 1),
                "tree=%p, start=%lu, end=%lu", tree, range.start, range.end);

    /* Fast path: if tree has only root and new interval overlaps/touches it, extend it */
    if (ucs_likely(tree->rb.num_nodes == 1) &&
        ucs_likely(range.start <= (root->end + 1)) &&
        ucs_likely(root->start <= (range.end + 1))) {
        uint64_t old_size = root->end - root->start;
        root->start        = ucs_min(root->start, range.start);
        root->end          = ucs_max(root->end, range.end);
        tree->total_size  += (root->end - root->start) - old_size;
        return UCS_OK;
    }

    /* Slow path: handle complex merging or empty tree */
    return ucs_interval_tree_insert_slow(tree, range);
}

/**
 * Check if the interval tree is empty
 *
 * @param [in]  tree  Interval tree
 *
 * @return Non-zero if tree has no nodes, 0 otherwise
 */
static UCS_F_ALWAYS_INLINE int
ucs_interval_tree_is_empty(const ucs_interval_tree_t *tree)
{
    return tree->rb.num_nodes == 0;
}


/**
 * Return the number of intervals (nodes) in the tree
 *
 * @param [in]  tree  Interval tree
 *
 * @return Number of nodes
 */
static UCS_F_ALWAYS_INLINE size_t
ucs_interval_tree_count(const ucs_interval_tree_t *tree)
{
    return tree->rb.num_nodes;
}


/**
 * Remove one interval from the tree (the leftmost) and return its range.
 * The node is freed via the tree's memory pool.
 *
 * @param [in]   tree   Interval tree
 * @param [out]  range  Range of the removed interval
 *
 * @return Non-zero if an interval was removed, 0 if tree was empty
 */
int ucs_interval_tree_pop_any(ucs_interval_tree_t *tree,
                              ucs_interval_tree_range_t *range);


/**
 * Check if tree contains exactly one interval with the given range
 *
 * @param [in]  tree   Interval tree
 * @param [in]  range  Range to check
 *
 * @return Non-zero if tree has exactly one interval matching range, 0 otherwise
 */
static UCS_F_ALWAYS_INLINE int
ucs_interval_tree_is_equal_range(const ucs_interval_tree_t *tree,
                                 ucs_interval_tree_range_t range)
{
    ucs_interval_node_t *root = ucs_interval_tree_root(tree);

    ucs_assertv(range.start <= (range.end + 1),
                "tree=%p, start=%lu, end=%lu", tree, range.start, range.end);

    return (tree->rb.num_nodes == 1) && (root->start == range.start) &&
           (root->end == range.end);
}

#endif
