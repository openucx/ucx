/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_INTERVAL_TREE_H_
#define UCS_INTERVAL_TREE_H_

#include <ucs/type/status.h>
#include <stdint.h>
#include <stddef.h>


/**
 * Callback for allocating an interval tree node
 *
 * @param [in]  size  Size of the node to allocate
 * @param [in]  arg   User-defined argument passed during tree initialization

 * @return Pointer to allocated node, or NULL on failure
 */
typedef void *(*ucs_interval_tree_alloc_node_func_t)(size_t size, void *arg);


/**
 * Callback for deallocating an interval tree node
 *
 * @param [in]  node  Node to deallocate
 * @param [in]  arg   User-defined argument passed during tree initialization
 */
typedef void (*ucs_interval_tree_free_node_func_t)(void *node, void *arg);


/**
 * Interval tree node allocation/deallocation operations
 */
typedef struct ucs_interval_tree_ops {
    ucs_interval_tree_alloc_node_func_t alloc_node; /* Node allocation callback */
    ucs_interval_tree_free_node_func_t  free_node;  /* Node deallocation callback */
    void                               *arg;        /* User-defined argument for
                                                     * callbacks */
} ucs_interval_tree_ops_t;


typedef struct ucs_interval_node ucs_interval_node_t;


typedef struct ucs_interval_tree {
    ucs_interval_node_t     *root;  /**< Root node of the tree */
    ucs_interval_tree_ops_t  ops;   /**< Memory management callbacks */
} ucs_interval_tree_t;


/**
 * Initialize an interval tree
 *
 * @param [in]  tree        Interval tree to initialize
 * @param [in]  ops         Memory management operations (alloc, free, arg)
 * @param [in]  root_start  Start of initial root interval 
 * @param [in]  root_end    End of initial root interval 
 *
 * @return UCS_OK on success, or error code on failure
 */
ucs_status_t ucs_interval_tree_init(ucs_interval_tree_t *tree,
                                    const ucs_interval_tree_ops_t *ops,
                                    uint64_t root_start, uint64_t root_end);


/**
 * Cleanup and free all nodes in the interval tree
 *
 * @param [in]  tree  Interval tree to clean up
 */
void ucs_interval_tree_cleanup(ucs_interval_tree_t *tree);


/**
 * Insert a new interval into the tree
 *
 * @param [in]  tree   Interval tree
 * @param [in]  start  Start of interval 
 * @param [in]  end    End of interval 
 *
 * @return UCS_OK on success, or error code on failure
 */
ucs_status_t ucs_interval_tree_insert(ucs_interval_tree_t *tree, uint64_t start,
                                      uint64_t end);


/**
 * Check if a range is fully covered by intervals in the tree
 *
 * @param [in]  tree   Interval tree
 * @param [in]  start  Start of range to check
 * @param [in]  end    End of range to check
 *
 * @return Non-zero if the range [start, end] is fully covered, 0 otherwise
 */
int ucs_interval_tree_range_covered(const ucs_interval_tree_t *tree,
                                    uint64_t start, uint64_t end);

#endif
