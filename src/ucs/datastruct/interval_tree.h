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


typedef void *(*ucs_interval_tree_alloc_node_func_t)(size_t size, void *arg);


typedef void (*ucs_interval_tree_free_node_func_t)(void *node, void *arg);


typedef struct ucs_interval_tree_ops {
    ucs_interval_tree_alloc_node_func_t alloc_node; /* Node allocation callback */
    ucs_interval_tree_free_node_func_t  free_node;  /* Node deallocation callback */
    void                               *arg;        /* User-defined argument for
                                                     * callbacks */
} ucs_interval_tree_ops_t;

typedef struct ucs_interval_tree {
    uint64_t total;
    uint64_t size;
    int      init;
    int      received_first;
} ucs_interval_tree_t;


static inline void ucs_interval_tree_init(ucs_interval_tree_t *tree,
                                          const ucs_interval_tree_ops_t *ops) {
    tree->total = 0;
    tree->size = 0;
    tree->init = 0;
    tree->received_first = 0;
}

typedef struct ucs_interval_node {
    struct ucs_interval_node *left;  /**< Left child node */
    struct ucs_interval_node *right; /**< Right child node */
    uint64_t                  start; /**< Start of interval */
    uint64_t                  end;   /**< End of interval */
} ucs_interval_node_t;

static inline void ucs_interval_tree_cleanup(ucs_interval_tree_t *tree) {}


static inline ucs_status_t ucs_interval_tree_insert(ucs_interval_tree_t *tree, uint64_t start,
                                      uint64_t end) {
    if (tree->received_first == 1 && start == 0) {
        return UCS_OK;      
    }

    tree->received_first = 1;
    tree->total += (end - start);
    return UCS_OK;
}


static inline int ucs_interval_tree_range_covered(ucs_interval_tree_t *tree,
                                    uint64_t start, uint64_t end) {
    if (tree->init == 0) {
        tree->size = (end - start);
        tree->init = 1;
    }
              
    return tree->total == tree->size;
}

#endif
