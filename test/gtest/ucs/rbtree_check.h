/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_TEST_RBTREE_CHECK_H_
#define UCS_TEST_RBTREE_CHECK_H_

extern "C" {
#include <ucs/datastruct/rbtree.h>
}

#include "common/googletest/gtest.h"

#include <algorithm>

/*
 * Structural validation of a ucs_rbtree_t, shared by every test of a structure
 * built on it. Checks only what the core owns - colors, parent links, black
 * height and node count. Ordering and any derived subtree data belong to the
 * embedding structure; per-node data it derives can be checked through the
 * optional visitor.
 */
namespace rbtree_check {

/** Called for every node, to check whatever the embedding structure adds */
using visitor_t = void (*)(const ucs_rbtree_node_t*);


static inline size_t height(const ucs_rbtree_node_t *node)
{
    if (node == NULL) {
        return 0;
    }

    return 1 + std::max(height(node->left), height(node->right));
}

/*
 * Report a violation both as a gtest failure, so the offending node is named,
 * and through 'ok', so the caller can add its own context.
 */
#define UCS_RBTREE_CHECK(_ok, _cond, _node) \
    do { \
        if (!(_cond)) { \
            (_ok) = false; \
            ADD_FAILURE() << "node " << static_cast<const void*>(_node) \
                          << ": " << #_cond; \
        } \
    } while (0)


/* Returns the node count and sets @a black_height */
static inline size_t validate_node(const ucs_rbtree_node_t *node,
                                   size_t &black_height, bool &ok,
                                   visitor_t visit)
{
    size_t left_height = 0, right_height = 0, count;

    if (node == NULL) {
        black_height = 1;
        return 0;
    }

    /* A red node cannot have a red child */
    if (node->color == UCS_RBTREE_RED) {
        UCS_RBTREE_CHECK(ok,
                         (node->left == NULL) ||
                                 (node->left->color == UCS_RBTREE_BLACK),
                         node);
        UCS_RBTREE_CHECK(ok,
                         (node->right == NULL) ||
                                 (node->right->color == UCS_RBTREE_BLACK),
                         node);
    }

    if (node->left != NULL) {
        UCS_RBTREE_CHECK(ok, node->left->parent == node, node);
    }

    if (node->right != NULL) {
        UCS_RBTREE_CHECK(ok, node->right->parent == node, node);
    }

    if (visit != NULL) {
        visit(node);
    }

    count = 1 + validate_node(node->left, left_height, ok, visit) +
            validate_node(node->right, right_height, ok, visit);

    /* Every path to a leaf must cross the same number of black nodes */
    UCS_RBTREE_CHECK(ok, left_height == right_height, node);
    black_height = left_height + ((node->color == UCS_RBTREE_BLACK) ? 1 : 0);
    return count;
}

/**
 * Check the red-black invariants and the node count.
 *
 * @param [in]  tree            Tree to validate.
 * @param [in]  expected_count  Number of nodes the tree should hold, checked
 *                              against both the walk and the tree's own count.
 * @param [in]  visit           Optional per-node check for data the embedding
 *                              structure derives, such as an augmentation.
 *
 * @return Whether every structural invariant held.
 */
static inline bool validate(const ucs_rbtree_t *tree, size_t expected_count,
                            visitor_t visit = NULL)
{
    size_t black_height = 0;
    bool ok             = true;

    if (tree->root != NULL) {
        UCS_RBTREE_CHECK(ok, tree->root->color == UCS_RBTREE_BLACK, tree->root);
        UCS_RBTREE_CHECK(ok, tree->root->parent == NULL, tree->root);
    }

    UCS_RBTREE_CHECK(ok,
                     validate_node(tree->root, black_height, ok, visit) ==
                             expected_count,
                     tree->root);
    UCS_RBTREE_CHECK(ok, tree->num_nodes == expected_count, tree->root);
    return ok;
}

} /* namespace rbtree_check */

#endif
