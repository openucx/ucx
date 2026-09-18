/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "rbtree.h"

#include <ucs/debug/assert.h>


void ucs_rbtree_init(ucs_rbtree_t *tree)
{
    tree->root = NULL;
}

/*
 * Rotate left: 'node' becomes the left child of its right child.
 */
static void ucs_rbtree_rotate_left(ucs_rbtree_t *tree, ucs_rbtree_node_t *node)
{
    ucs_rbtree_node_t *right = node->right;

    node->right = right->left;
    if (right->left != NULL) {
        right->left->parent = node;
    }

    right->parent = node->parent;
    if (node->parent == NULL) {
        tree->root = right;
    } else if (node == node->parent->left) {
        node->parent->left = right;
    } else {
        node->parent->right = right;
    }

    right->left  = node;
    node->parent = right;
}

/* Rotate right: 'node' becomes the right child of its left child. */
static void ucs_rbtree_rotate_right(ucs_rbtree_t *tree, ucs_rbtree_node_t *node)
{
    ucs_rbtree_node_t *left = node->left;

    node->left = left->right;
    if (left->right != NULL) {
        left->right->parent = node;
    }

    left->parent = node->parent;
    if (node->parent == NULL) {
        tree->root = left;
    } else if (node == node->parent->right) {
        node->parent->right = left;
    } else {
        node->parent->left = left;
    }

    left->right  = node;
    node->parent = left;
}

static void ucs_rbtree_insert_fixup(ucs_rbtree_t *tree, ucs_rbtree_node_t *node)
{
    ucs_rbtree_node_t *parent, *grandparent, *uncle;

    while ((node->parent != NULL) && (node->parent->parent != NULL) &&
           (node->parent->color == UCS_RBTREE_RED)) {
        parent      = node->parent;
        grandparent = parent->parent;

        if (parent == grandparent->left) {
            uncle = grandparent->right;
            if ((uncle != NULL) && (uncle->color == UCS_RBTREE_RED)) {
                parent->color      = UCS_RBTREE_BLACK;
                uncle->color       = UCS_RBTREE_BLACK;
                grandparent->color = UCS_RBTREE_RED;
                node               = grandparent;
                continue;
            }

            if (node == parent->right) {
                node = parent;
                ucs_rbtree_rotate_left(tree, node);
                parent      = node->parent;
                grandparent = parent->parent;
            }

            parent->color      = UCS_RBTREE_BLACK;
            grandparent->color = UCS_RBTREE_RED;
            ucs_rbtree_rotate_right(tree, grandparent);
        } else {
            uncle = grandparent->left;
            if ((uncle != NULL) && (uncle->color == UCS_RBTREE_RED)) {
                parent->color      = UCS_RBTREE_BLACK;
                uncle->color       = UCS_RBTREE_BLACK;
                grandparent->color = UCS_RBTREE_RED;
                node               = grandparent;
                continue;
            }

            if (node == parent->left) {
                node = parent;
                ucs_rbtree_rotate_right(tree, node);
                parent      = node->parent;
                grandparent = parent->parent;
            }

            parent->color      = UCS_RBTREE_BLACK;
            grandparent->color = UCS_RBTREE_RED;
            ucs_rbtree_rotate_left(tree, grandparent);
        }
    }

    tree->root->color = UCS_RBTREE_BLACK;
}

void ucs_rbtree_insert_at(ucs_rbtree_t *tree, ucs_rbtree_node_t *parent,
                          ucs_rbtree_node_t **link, ucs_rbtree_node_t *node)
{
    ucs_assert(*link == NULL);
    ucs_assert((parent == NULL) ? (link == &tree->root) :
                                  ((link == &parent->left) ||
                                   (link == &parent->right)));

    node->parent = parent;
    node->left   = NULL;
    node->right  = NULL;
    node->color  = UCS_RBTREE_RED;
    *link        = node;

    ucs_rbtree_insert_fixup(tree, node);
}

/* Replace the subtree rooted at 'old_node' with the one rooted at 'new_node' */
static void ucs_rbtree_transplant(ucs_rbtree_t *tree,
                                  ucs_rbtree_node_t *old_node,
                                  ucs_rbtree_node_t *new_node)
{
    if (old_node->parent == NULL) {
        tree->root = new_node;
    } else if (old_node == old_node->parent->left) {
        old_node->parent->left = new_node;
    } else {
        old_node->parent->right = new_node;
    }

    if (new_node != NULL) {
        new_node->parent = old_node->parent;
    }
}

static UCS_F_ALWAYS_INLINE int
ucs_rbtree_is_black(const ucs_rbtree_node_t *node)
{
    return (node == NULL) || (node->color == UCS_RBTREE_BLACK);
}

/*
 * Restore the red-black invariants after removing a black node.
 */
static void ucs_rbtree_remove_fixup(ucs_rbtree_t *tree, ucs_rbtree_node_t *node,
                                    ucs_rbtree_node_t *parent)
{
    ucs_rbtree_node_t *sibling;

    while ((node != tree->root) && ucs_rbtree_is_black(node)) {
        if (node == parent->left) {
            sibling = parent->right;
            /* Removing a black node leaves the opposite subtree non-empty, so
             * the sibling exists */
            ucs_assert(sibling != NULL);

            if (sibling->color == UCS_RBTREE_RED) {
                sibling->color = UCS_RBTREE_BLACK;
                parent->color  = UCS_RBTREE_RED;
                ucs_rbtree_rotate_left(tree, parent);
                sibling = parent->right;
                ucs_assert(sibling != NULL);
            }

            if (ucs_rbtree_is_black(sibling->left) &&
                ucs_rbtree_is_black(sibling->right)) {
                sibling->color = UCS_RBTREE_RED;
                node           = parent;
                parent         = node->parent;
                continue;
            }

            if (ucs_rbtree_is_black(sibling->right)) {
                /* Not both children are black, so the left one is red */
                ucs_assert(sibling->left != NULL);
                sibling->left->color = UCS_RBTREE_BLACK;
                sibling->color       = UCS_RBTREE_RED;
                ucs_rbtree_rotate_right(tree, sibling);
                sibling = parent->right;
            }

            sibling->color = parent->color;
            parent->color  = UCS_RBTREE_BLACK;
            /* Either the right child was red, or the rotation above moved the
             * old sibling into that slot */
            ucs_assert(sibling->right != NULL);
            sibling->right->color = UCS_RBTREE_BLACK;
            ucs_rbtree_rotate_left(tree, parent);
        } else {
            sibling = parent->left;
            /* Removing a black node leaves the opposite subtree non-empty, so
             * the sibling exists */
            ucs_assert(sibling != NULL);

            if (sibling->color == UCS_RBTREE_RED) {
                sibling->color = UCS_RBTREE_BLACK;
                parent->color  = UCS_RBTREE_RED;
                ucs_rbtree_rotate_right(tree, parent);
                sibling = parent->left;
                ucs_assert(sibling != NULL);
            }

            if (ucs_rbtree_is_black(sibling->left) &&
                ucs_rbtree_is_black(sibling->right)) {
                sibling->color = UCS_RBTREE_RED;
                node           = parent;
                parent         = node->parent;
                continue;
            }

            if (ucs_rbtree_is_black(sibling->left)) {
                /* Not both children are black, so the right one is red */
                ucs_assert(sibling->right != NULL);
                sibling->right->color = UCS_RBTREE_BLACK;
                sibling->color        = UCS_RBTREE_RED;
                ucs_rbtree_rotate_left(tree, sibling);
                sibling = parent->left;
            }

            sibling->color = parent->color;
            parent->color  = UCS_RBTREE_BLACK;
            /* Either the left child was red, or the rotation above moved the
             * old sibling into that slot */
            ucs_assert(sibling->left != NULL);
            sibling->left->color = UCS_RBTREE_BLACK;
            ucs_rbtree_rotate_right(tree, parent);
        }

        node   = tree->root;
        parent = NULL;
    }

    if (node != NULL) {
        node->color = UCS_RBTREE_BLACK;
    }
}

void ucs_rbtree_remove(ucs_rbtree_t *tree, ucs_rbtree_node_t *node)
{
    ucs_rbtree_node_t *child, *child_parent, *successor;
    ucs_rbtree_color_t removed_color;

    removed_color = node->color;

    if (node->left == NULL) {
        child        = node->right;
        child_parent = node->parent;
        ucs_rbtree_transplant(tree, node, node->right);
    } else if (node->right == NULL) {
        child        = node->left;
        child_parent = node->parent;
        ucs_rbtree_transplant(tree, node, node->left);
    } else {
        /* Splice out the successor and put it where 'node' was. The successor
         * is relinked rather than having its key copied into 'node', so
         * caller-held node pointers stay valid. */
        successor = node->right;
        while (successor->left != NULL) {
            successor = successor->left;
        }

        removed_color = successor->color;
        child         = successor->right;

        if (successor->parent == node) {
            child_parent = successor;
        } else {
            child_parent = successor->parent;
            ucs_rbtree_transplant(tree, successor, successor->right);
            successor->right         = node->right;
            successor->right->parent = successor;
        }

        ucs_rbtree_transplant(tree, node, successor);
        successor->left         = node->left;
        successor->left->parent = successor;
        successor->color        = node->color;
    }

    if (removed_color == UCS_RBTREE_BLACK) {
        ucs_rbtree_remove_fixup(tree, child, child_parent);
    }

    node->parent = NULL;
    node->left   = NULL;
    node->right  = NULL;
}

ucs_rbtree_node_t *ucs_rbtree_first(const ucs_rbtree_t *tree)
{
    ucs_rbtree_node_t *node = tree->root;

    if (node == NULL) {
        return NULL;
    }

    while (node->left != NULL) {
        node = node->left;
    }

    return node;
}
