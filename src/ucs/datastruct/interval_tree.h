/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_INTERVAL_TREE_H_
#define UCS_INTERVAL_TREE_H_

#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>
#include <ucs/type/status.h>
#include <ucs/datastruct/mpool.h>
#include <stdint.h>
#include <stddef.h>


/** Red-Black tree node color */
enum {
    UCS_INTERVAL_NODE_BLACK = 0,
    UCS_INTERVAL_NODE_RED   = 1
};

/**
 * Interval tree node structure (Red-Black tree)
 */
typedef struct ucs_interval_node {
    struct ucs_interval_node *left;   /**< Left child node */
    struct ucs_interval_node *right;  /**< Right child node */
    struct ucs_interval_node *parent; /**< Parent node (NULL for root) */
    uint64_t                 start;  /**< Start of interval */
    uint64_t                 end;    /**< End of interval */
    uint8_t                  color;  /**< UCS_INTERVAL_NODE_BLACK or UCS_INTERVAL_NODE_RED */
} ucs_interval_node_t;


typedef struct {
    ucs_interval_node_t *root;       /**< Root node of the tree */
    ucs_mpool_t         *mpool;      /**< Memory pool for node allocation */
    size_t              num_nodes;   /**< Number of nodes in the tree */
    size_t              total_size;  /**< Sum of (end - start) across all nodes */
} ucs_interval_tree_t;


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
                                           uint64_t start, uint64_t end);


/**
 * Insert a new interval into the tree
 *
 * @param [in]  tree   Interval tree
 * @param [in]  start  Start of interval 
 * @param [in]  end    End of interval 
 *
 * @return UCS_OK on success, or error code on failure
 */
static UCS_F_ALWAYS_INLINE ucs_status_t ucs_interval_tree_insert(
        ucs_interval_tree_t *tree, uint64_t start, uint64_t end)
{
    ucs_interval_node_t *root = tree->root;

    ucs_assertv(start <= (end + 1), "tree=%p, start=%lu, end=%lu", tree, start,
                end);

    /* Fast path: if tree has only root and new interval overlaps/touches it, extend it */
    if (ucs_likely(tree->num_nodes == 1) &&
        ucs_likely(start <= (root->end + 1)) &&
        ucs_likely(root->start <= (end + 1))) {
        uint64_t old_size = root->end - root->start;
        root->start        = ucs_min(root->start, start);
        root->end          = ucs_max(root->end, end);
        tree->total_size  += (root->end - root->start) - old_size;
        return UCS_OK;
    }

    /* Slow path: handle complex merging or empty tree */
    return ucs_interval_tree_insert_slow(tree, start, end);
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
    return tree->num_nodes == 0;
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
    return tree->num_nodes;
}


/**
 * Remove one interval from the tree (the root) and return its range.
 * The node is freed via the tree's memory pool.
 *
 * @param [in]   tree   Interval tree
 * @param [out]  start  Start of the removed interval
 * @param [out]  end    End of the removed interval
 *
 * @return Non-zero if an interval was removed, 0 if tree was empty
 */
int ucs_interval_tree_pop_any(ucs_interval_tree_t *tree, uint64_t *start,
                              uint64_t *end);


/**
 * Check if tree contains exactly one interval with the given range
 *
 * @param [in]  tree   Interval tree
 * @param [in]  start  Start of range to check
 * @param [in]  end    End of range to check
 *
 * @return Non-zero if tree has exactly one interval [start, end], 0 otherwise
 */
static UCS_F_ALWAYS_INLINE int
ucs_interval_tree_is_equal_range(const ucs_interval_tree_t *tree,
                                  uint64_t start, uint64_t end)
{
    ucs_interval_node_t *root = tree->root;

    ucs_assertv(start <= (end + 1), "tree=%p, start=%lu, end=%lu", tree, start,
                end);

    return (tree->num_nodes == 1) && (root->start == start) &&
           (root->end == end);
}

#endif
