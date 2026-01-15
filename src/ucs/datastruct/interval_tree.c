/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "interval_tree.h"
#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>


/**
 * Interval tree node structure
 *
 * The tree maintains the augmented interval tree invariant: each node stores
 * the maximum 'end' value among all intervals in its subtree. This enables
 * efficient pruning during range coverage queries.
 */
typedef struct ucs_interval_node {
    struct ucs_interval_node *left; /**< Left child node */
    struct ucs_interval_node *right; /**< Right child node */
    struct ucs_interval_node *next; /**< used for stack push/pop */
    uint64_t                 start; /**< Start of interval */
    uint64_t                 end; /**< End of interval */
    uint64_t                 max_end; /**< Maximum end value in this subtree,
                                            used for pruning optimization */
} ucs_interval_node_t;


static ucs_interval_node_t *
ucs_interval_tree_node_create(ucs_interval_tree_t *tree, uint64_t start,
                              uint64_t end)
{
    ucs_interval_node_t *node;

    node = tree->ops.alloc_node(sizeof(*node), tree->ops.arg);
    if (node == NULL) {
        return NULL;
    }

    node->left    = NULL;
    node->right   = NULL;
    node->next    = NULL;
    node->start   = start;
    node->end     = end;
    node->max_end = end;
    return node;
}

ucs_status_t ucs_interval_tree_init(ucs_interval_tree_t *tree,
                                    const ucs_interval_tree_ops_t *ops,
                                    uint64_t root_start, uint64_t root_end)
{
    if (root_start > root_end) {
        return UCS_ERR_INVALID_PARAM;
    }

    tree->ops  = *ops;
    tree->root = ucs_interval_tree_node_create(tree, root_start, root_end);
    if (tree->root == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static inline void ucs_interval_tree_stack_push(ucs_interval_node_t **stack_top,
                                                ucs_interval_node_t *node)
{
    node->next = *stack_top;
    *stack_top = node;
}

static inline ucs_interval_node_t *
ucs_interval_tree_stack_pop(ucs_interval_node_t **stack_top)
{
    ucs_interval_node_t *node = *stack_top;

    *stack_top = node->next;
    return node;
}

void ucs_interval_tree_cleanup(ucs_interval_tree_t *tree)
{
    ucs_interval_node_t *stack_top = NULL;
    ucs_interval_node_t *node;

    ucs_interval_tree_stack_push(&stack_top, tree->root);

    while (stack_top != NULL) {
        node = ucs_interval_tree_stack_pop(&stack_top);

        if (node->right != NULL) {
            ucs_interval_tree_stack_push(&stack_top, node->right);
        }

        if (node->left != NULL) {
            ucs_interval_tree_stack_push(&stack_top, node->left);
        }

        tree->ops.free_node(node, tree->ops.arg);
    }
}

ucs_status_t ucs_interval_tree_insert(ucs_interval_tree_t *tree, uint64_t start,
                                      uint64_t end)
{
    ucs_interval_node_t *current = tree->root;
    ucs_interval_node_t *parent  = NULL;
    ucs_interval_node_t *new_node;

    ucs_assert(current != NULL);

    if (start > end) {
        return UCS_ERR_INVALID_PARAM;
    }

    while (current != NULL) {
        /* Update max_end for pruning optimization */
        current->max_end = ucs_max(current->max_end, end);

        /* If new interval overlaps with current, extend the current interval
         * instead of creating a new node */
        if ((start >= current->start) && (start <= current->end)) {
            current->end = ucs_max(current->end, end);
            return UCS_OK;
        }

        parent  = current;
        current = (start < current->start) ? current->left : current->right;
    }

    new_node = ucs_interval_tree_node_create(tree, start, end);
    if (new_node == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    if (start < parent->start) {
        parent->left = new_node;
    } else {
        parent->right = new_node;
    }

    return UCS_OK;
}

int ucs_interval_tree_range_covered(const ucs_interval_tree_t *tree,
                                    uint64_t start, uint64_t end)
{
    ucs_interval_node_t *stack_top = NULL;
    ucs_interval_node_t *node      = tree->root;
    uint64_t covered_end           = start; /* Tracks the rightmost point 
                                             * covered so far */

    if (start > end) {
        return 0;
    }

    /* In-order traversal with pruning optimization */
    while ((node != NULL) || (stack_top != NULL)) {
        if (node != NULL) {
            ucs_interval_tree_stack_push(&stack_top, node);
            /* Skip left subtree if max_end <= covered_end, as no intervals there
             * can extend our coverage. */
            node = (node->max_end > covered_end) ? node->left : NULL;
        } else {
            node = ucs_interval_tree_stack_pop(&stack_top);
            if (node->start > covered_end) {
                /* Gap detected: current node starts after what we've covered
                 * so far */
                return 0;
            }

            /* Extend coverage with current node's end */
            covered_end = ucs_max(node->end, covered_end);
            if (covered_end >= end) {
                /* We've fully covered the requested range */
                return 1;
            }

            /* Skip right subtree if node->start >= end, as all intervals there
             * are beyond our query range. */
            node = (node->start < end) ? node->right : NULL;
        }
    }

    /* Traversal complete: range is not fully covered */
    return 0;
}
