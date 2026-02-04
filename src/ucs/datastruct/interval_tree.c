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
 */

static ucs_interval_node_t *
ucs_interval_tree_node_create(ucs_interval_tree_t *tree, uint64_t start,
                              uint64_t end)
{
    ucs_interval_node_t *node;

    node = tree->ops.alloc_node(sizeof(*node), tree->ops.arg);
    if (node == NULL) {
        return NULL;
    }

    node->left  = NULL;
    node->right = NULL;
    node->start = start;
    node->end   = end;
    return node;
}

void ucs_interval_tree_init(ucs_interval_tree_t *tree,
                            const ucs_interval_tree_ops_t *ops)
{
    tree->ops  = *ops;
    tree->root = NULL;
}

static void ucs_interval_tree_cleanup_recursive(ucs_interval_tree_t *tree,
                                                ucs_interval_node_t *node)
{
    if (node == NULL) {
        return;
    }

    ucs_interval_tree_cleanup_recursive(tree, node->left);
    ucs_interval_tree_cleanup_recursive(tree, node->right);
    tree->ops.free_node(node, tree->ops.arg);
}

void ucs_interval_tree_cleanup(ucs_interval_tree_t *tree)
{
    ucs_interval_tree_cleanup_recursive(tree, tree->root);
}


/**
 * Helper function to remove a node from the tree
 *
 * @param [in]  tree   Interval tree
 * @param [in]  root   Root of the current subtree being searched (used for
 *                     recursion - initially the tree's root)
 * @param [in]  target Node to remove
 *
 * @return New root of the subtree
 */
static ucs_interval_node_t *
ucs_interval_tree_remove_node(ucs_interval_tree_t *tree,
                              ucs_interval_node_t *root,
                              ucs_interval_node_t *target)
{
    ucs_interval_node_t *temp, *successor;

    if (root == NULL) {
        return NULL;
    }

    if (target->start < root->start) {
        root->left = ucs_interval_tree_remove_node(tree, root->left, target);
        return root;
    } else if (target->start > root->start) {
        root->right = ucs_interval_tree_remove_node(tree, root->right, target);
        return root;
    }

    /* Found target - handle deletion based on number of children */
    if (root->left == NULL) {
        temp = root->right;
        tree->ops.free_node(root, tree->ops.arg);
        return temp;
    } else if (root->right == NULL) {
        temp = root->left;
        tree->ops.free_node(root, tree->ops.arg);
        return temp;
    }

    /* Node has two children: find in-order successor (leftmost in right subtree) */
    successor = root->right;
    while (successor->left != NULL) {
        successor = successor->left;
    }

    /* Copy successor data to current node */
    root->start = successor->start;
    root->end   = successor->end;

    /* Delete the successor recursively */
    root->right = ucs_interval_tree_remove_node(tree, root->right, successor);
    return root;
}

/**
 * Recursively find and remove all nodes that overlap with given interval.
 * The start/end parameters are updated to reflect the merged range.
 * Returns the new root of the subtree. Notice that we can safely remove 
   partially overlapping nodes because we create a new node with the merged 
   range.
 */
static ucs_interval_node_t *
ucs_interval_tree_remove_overlapping(ucs_interval_tree_t *tree,
                                     ucs_interval_node_t *node, uint64_t *start,
                                     uint64_t *end)
{
    if (node == NULL) {
        return NULL;
    }

    /* Conditionally process left subtree: only if node's range might reach our start */
    if ((node->end + 1) >= *start) {
        node->left = ucs_interval_tree_remove_overlapping(tree, node->left,
                                                          start, end);
    }

    /* Conditionally process right subtree: only if node's range might reach our end */
    if (node->start <= (*end + 1)) {
        node->right = ucs_interval_tree_remove_overlapping(tree, node->right,
                                                           start, end);
    }

    /* Check if current node overlaps or touches (continuous range) */
    if ((*start <= (node->end + 1)) && (node->start <= (*end + 1))) {
        /* Extend the range to cover this node */
        *start = ucs_min(*start, node->start);
        *end   = ucs_max(*end, node->end);

        /* Delete this overlapping node */
        return ucs_interval_tree_remove_node(tree, node, node);
    }

    return node;
}

/**
 * Insert a node into the tree using standard BST insertion
 *
 * The tree is organized by interval start values. This function assumes the
 * new_node does not overlap with any existing nodes in the tree (overlapping
 * nodes should be removed before calling this function).
 *
 * @param [in]  root      Root of the current subtree (NULL for empty tree)
 * @param [in]  new_node  Node to insert
 *
 * @return New root of the subtree
 */
static ucs_interval_node_t *
ucs_interval_tree_insert_node(ucs_interval_node_t *root,
                              ucs_interval_node_t *new_node)
{
    if (root == NULL) {
        return new_node;
    }

    if (new_node->start < root->start) {
        root->left = ucs_interval_tree_insert_node(root->left, new_node);
    } else {
        root->right = ucs_interval_tree_insert_node(root->right, new_node);
    }

    return root;
}

ucs_status_t ucs_interval_tree_insert_slow(ucs_interval_tree_t *tree,
                                           uint64_t start, uint64_t end)
{
    ucs_interval_node_t *new_node;
    uint64_t merged_start = start;
    uint64_t merged_end   = end;

    /* Remove all overlapping nodes and compute the merged range */
    tree->root = ucs_interval_tree_remove_overlapping(tree, tree->root,
                                                      &merged_start,
                                                      &merged_end);

    /* Create new node with the final merged range */
    new_node = ucs_interval_tree_node_create(tree, merged_start, merged_end);
    if (new_node == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    /* Insert the merged node into the tree */
    tree->root = ucs_interval_tree_insert_node(tree->root, new_node);
    return UCS_OK;
}
