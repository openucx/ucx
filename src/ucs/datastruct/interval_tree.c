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

    node->left   = NULL;
    node->right  = NULL;
    node->parent = NULL;
    node->start  = start;
    node->end    = end;
    node->color  = UCS_INTERVAL_NODE_RED;

    tree->num_nodes++;
    tree->total_size += (end - start);
    return node;
}

/* Rotate left: node becomes left child of its right child. */
static void ucs_interval_tree_rotate_left(ucs_interval_tree_t *tree,
                                          ucs_interval_node_t *node)
{
    ucs_interval_node_t *right = node->right;

    if (ucs_unlikely(right == NULL)) {
        return;
    }

    node->right = right->left;
    if (right->left != NULL) {
        right->left->parent = node;
    }
    right->parent = node->parent;
    if (ucs_unlikely(node->parent == NULL)) {
        tree->root = right;
    } else if (node == node->parent->left) {
        node->parent->left = right;
    } else {
        node->parent->right = right;
    }
    right->left  = node;
    node->parent = right;
}

/* Rotate right: node becomes right child of its left child. */
static void ucs_interval_tree_rotate_right(ucs_interval_tree_t *tree,
                                           ucs_interval_node_t *node)
{
    ucs_interval_node_t *left = node->left;

    if (ucs_unlikely(left == NULL)) {
        return;
    }

    node->left = left->right;
    if (left->right != NULL) {
        left->right->parent = node;
    }
    left->parent = node->parent;
    if (ucs_unlikely(node->parent == NULL)) {
        tree->root = left;
    } else if (node == node->parent->right) {
        node->parent->right = left;
    } else {
        node->parent->left = left;
    }
    left->right  = node;
    node->parent = left;
}

static void ucs_interval_tree_insert_fixup(ucs_interval_tree_t *tree,
                                           ucs_interval_node_t *node)
{
    ucs_interval_node_t *uncle;

    while ((node->parent != NULL) &&
           (node->parent->parent != NULL) &&
           (node->parent->color == UCS_INTERVAL_NODE_RED)) {
        if (node->parent == node->parent->parent->left) {
            uncle = node->parent->parent->right;
            if (ucs_likely((uncle != NULL) &&
                           (uncle->color == UCS_INTERVAL_NODE_RED))) {
                node->parent->color         = UCS_INTERVAL_NODE_BLACK;
                uncle->color                = UCS_INTERVAL_NODE_BLACK;
                node->parent->parent->color = UCS_INTERVAL_NODE_RED;
                node                        = node->parent->parent;
            } else {
                if (node == node->parent->right) {
                    node = node->parent;
                    ucs_interval_tree_rotate_left(tree, node);
                }
                if (ucs_unlikely((node->parent == NULL) ||
                                 (node->parent->parent == NULL))) {
                    break;
                }
                node->parent->color         = UCS_INTERVAL_NODE_BLACK;
                node->parent->parent->color = UCS_INTERVAL_NODE_RED;
                ucs_interval_tree_rotate_right(tree, node->parent->parent);
            }
        } else {
            uncle = node->parent->parent->left;
            if (ucs_likely((uncle != NULL) &&
                           (uncle->color == UCS_INTERVAL_NODE_RED))) {
                node->parent->color         = UCS_INTERVAL_NODE_BLACK;
                uncle->color                = UCS_INTERVAL_NODE_BLACK;
                node->parent->parent->color = UCS_INTERVAL_NODE_RED;
                node                        = node->parent->parent;
            } else {
                if (node == node->parent->left) {
                    node = node->parent;
                    ucs_interval_tree_rotate_right(tree, node);
                }
                if (ucs_unlikely((node->parent == NULL) ||
                                 (node->parent->parent == NULL))) {
                    break;
                }
                node->parent->color         = UCS_INTERVAL_NODE_BLACK;
                node->parent->parent->color = UCS_INTERVAL_NODE_RED;
                ucs_interval_tree_rotate_left(tree, node->parent->parent);
            }
        }
    }
    tree->root->color = UCS_INTERVAL_NODE_BLACK;
}

void ucs_interval_tree_init(ucs_interval_tree_t *tree, ucs_mpool_t *mpool)
{
    tree->mpool      = mpool;
    tree->root       = NULL;
    tree->num_nodes  = 0;
    tree->total_size = 0;
}

static void ucs_interval_tree_delete_node(ucs_interval_tree_t *tree,
                                          ucs_interval_node_t *node)
{
    ucs_assertv(tree->num_nodes > 0, "tree=%p, node=%p", tree, node);
    tree->total_size -= (node->end - node->start);
    tree->num_nodes--;
    ucs_mpool_put_inline(node);
}

static void ucs_interval_tree_cleanup_recursive(ucs_interval_tree_t *tree,
                                                ucs_interval_node_t *node)
{
    if (node == NULL) {
        return;
    }

    ucs_interval_tree_cleanup_recursive(tree, node->left);
    ucs_interval_tree_cleanup_recursive(tree, node->right);
    ucs_interval_tree_delete_node(tree, node);
}

void ucs_interval_tree_cleanup(ucs_interval_tree_t *tree)
{
    ucs_interval_tree_cleanup_recursive(tree, tree->root);
}

static void ucs_interval_tree_delete_fixup(ucs_interval_tree_t *tree,
                                           ucs_interval_node_t **root_ptr,
                                           ucs_interval_node_t *parent);

/**
 * Remove node at *root_ptr (BST remove by target). Updates *root_ptr to
 * replacement and runs RB delete fixup if the removed node was black.
 */
static void ucs_interval_tree_remove_node(ucs_interval_tree_t *tree,
                                          ucs_interval_node_t **root_ptr,
                                          ucs_interval_node_t *target)
{
    ucs_interval_node_t *root = *root_ptr;
    ucs_interval_node_t *replacement;
    ucs_interval_node_t *parent;
    uint8_t removed_color;

    if (ucs_unlikely(root == NULL)) {
        return;
    }

    if (target->start < root->start) {
        ucs_interval_tree_remove_node(tree, &root->left, target);
        if (root->left != NULL) {
            root->left->parent = root;
        }
        return;
    }

    if (target->start > root->start) {
        ucs_interval_tree_remove_node(tree, &root->right, target);
        if (root->right != NULL) {
            root->right->parent = root;
        }
        return;
    }

    /* Found target at root */
    parent        = root->parent;
    removed_color = root->color;

    if (root->left == NULL) {
        replacement = root->right;
        *root_ptr   = replacement;
        if (replacement != NULL) {
            replacement->parent = parent;
        }
        ucs_interval_tree_delete_node(tree, root);
        if (removed_color == UCS_INTERVAL_NODE_BLACK) {
            ucs_interval_tree_delete_fixup(tree, root_ptr, parent);
        }
        return;
    }

    if (root->right == NULL) {
        replacement         = root->left;
        *root_ptr           = replacement;
        replacement->parent = parent;
        ucs_interval_tree_delete_node(tree, root);
        if (removed_color == UCS_INTERVAL_NODE_BLACK) {
            ucs_interval_tree_delete_fixup(tree, root_ptr, parent);
        }
        return;
    }

    /* Two children: copy successor's key and remove successor */
    {
        ucs_interval_node_t *successor = root->right;

        while (successor->left != NULL) {
            successor = successor->left;
        }
        tree->total_size -= (root->end - root->start);
        root->start       = successor->start;
        root->end         = successor->end;
        tree->total_size += (root->end - root->start);
        ucs_interval_tree_remove_node(tree, &root->right, successor);
        if (root->right != NULL) {
            root->right->parent = root;
        }
    }
}

static void ucs_interval_tree_delete_fixup(ucs_interval_tree_t *tree,
                                           ucs_interval_node_t **root_ptr,
                                           ucs_interval_node_t *parent)
{
    ucs_interval_node_t *node = *root_ptr;
    ucs_interval_node_t *sibling;

    while ((parent != NULL) &&
          ((node == NULL) || (node->color == UCS_INTERVAL_NODE_BLACK))) {
        if (root_ptr == &parent->left) {
            sibling = parent->right;
            if ((sibling != NULL) && (sibling->color == UCS_INTERVAL_NODE_RED)) {
                sibling->color = UCS_INTERVAL_NODE_BLACK;
                parent->color  = UCS_INTERVAL_NODE_RED;
                ucs_interval_tree_rotate_left(tree, parent);
                sibling = parent->right;
            }
            if (ucs_unlikely(sibling == NULL)) {
                node   = parent;
                parent = parent->parent;
                if (parent != NULL) {
                    root_ptr = (parent->left == node) ? &parent->left : 
                                                        &parent->right;
                }
                continue;
            }
            if (((sibling->left == NULL) || 
                 (sibling->left->color == UCS_INTERVAL_NODE_BLACK)) &&
                ((sibling->right == NULL) || 
                 (sibling->right->color == UCS_INTERVAL_NODE_BLACK))) {
                sibling->color = UCS_INTERVAL_NODE_RED;
                node           = parent;
                parent         = parent->parent;
                if (parent != NULL) {
                    root_ptr = (parent->left == node) ? &parent->left : 
                                                        &parent->right;
                }
                continue;
            }
            if ((sibling->right == NULL) || 
                (sibling->right->color == UCS_INTERVAL_NODE_BLACK)) {
                if (sibling->left != NULL) {
                    sibling->left->color = UCS_INTERVAL_NODE_BLACK;
                }
                sibling->color = UCS_INTERVAL_NODE_RED;
                ucs_interval_tree_rotate_right(tree, sibling);
                sibling = parent->right;
            }
            if (ucs_unlikely(sibling == NULL)) {
                break;
            }
            sibling->color = parent->color;
            parent->color  = UCS_INTERVAL_NODE_BLACK;
            if (sibling->right != NULL) {
                sibling->right->color = UCS_INTERVAL_NODE_BLACK;
            }
            ucs_interval_tree_rotate_left(tree, parent);
            break;
        } else {
            /* root_ptr == &parent->right */
            sibling = parent->left;
            if ((sibling != NULL) && (sibling->color == UCS_INTERVAL_NODE_RED)) {
                sibling->color = UCS_INTERVAL_NODE_BLACK;
                parent->color  = UCS_INTERVAL_NODE_RED;
                ucs_interval_tree_rotate_right(tree, parent);
                sibling = parent->left;
            }
            if (ucs_unlikely(sibling == NULL)) {
                node   = parent;
                parent = parent->parent;
                if (parent != NULL) {
                    root_ptr = (parent->left == node) ? &parent->left : 
                                                        &parent->right;
                }
                continue;
            }
            if (((sibling->left == NULL) || 
                 (sibling->left->color == UCS_INTERVAL_NODE_BLACK)) &&
                ((sibling->right == NULL) || 
                 (sibling->right->color == UCS_INTERVAL_NODE_BLACK))) {
                sibling->color = UCS_INTERVAL_NODE_RED;
                node           = parent;
                parent         = parent->parent;
                if (parent != NULL) {
                    root_ptr = (parent->left == node) ? &parent->left : &parent->right;
                }
                continue;
            }
            if ((sibling->left == NULL) || 
                (sibling->left->color == UCS_INTERVAL_NODE_BLACK)) {
                if (sibling->right != NULL) {
                    sibling->right->color = UCS_INTERVAL_NODE_BLACK;
                }
                sibling->color = UCS_INTERVAL_NODE_RED;
                ucs_interval_tree_rotate_left(tree, sibling);
                sibling = parent->left;
            }
            if (ucs_unlikely(sibling == NULL)) {
                break;
            }
            sibling->color = parent->color;
            parent->color  = UCS_INTERVAL_NODE_BLACK;
            if (sibling->left != NULL) {
                sibling->left->color = UCS_INTERVAL_NODE_BLACK;
            }
            ucs_interval_tree_rotate_right(tree, parent);
            break;
        }
    }
    if (node != NULL) {
        node->color = UCS_INTERVAL_NODE_BLACK;
    }
    /* Update tree->root if we were fixing the root slot */
    if ((parent == NULL) && (root_ptr == &tree->root)) {
        tree->root = node;
    }
}

/**
 * Find any single node that overlaps or is adjacent to @a range.
 * Returns NULL if none found. Does not modify the tree.
 */
static ucs_interval_node_t *
ucs_interval_tree_find_overlap(ucs_interval_node_t *node,
                               ucs_interval_tree_range_t range)
{
    ucs_interval_node_t *found;

    if (node == NULL) {
        return NULL;
    }

    if ((range.start <= (node->end + 1)) && (node->start <= (range.end + 1))) {
        return node;
    }

    found = ucs_interval_tree_find_overlap(node->left, range);
    if (found != NULL) {
        return found;
    }

    /* Right subtree: all nodes have start >= node->start, so if
     * node->start > range.end + 1, no right descendant can overlap. */
    if (node->start <= (range.end + 1)) {
        return ucs_interval_tree_find_overlap(node->right, range);
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

    while ((overlap = ucs_interval_tree_find_overlap(tree->root,
                                                     *range)) != NULL) {
        range->start = ucs_min(range->start, overlap->start);
        range->end   = ucs_max(range->end, overlap->end);
        ucs_interval_tree_remove_node(tree, &tree->root, overlap);
    }
}

/**
 * BST insert by start; sets parent when attaching. Caller must run insert_fixup.
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
        if (root->left != NULL) {
            root->left->parent = root;
        }
    } else {
        root->right = ucs_interval_tree_insert_node(root->right, new_node);
        if (root->right != NULL) {
            root->right->parent = root;
        }
    }

    return root;
}

ucs_status_t ucs_interval_tree_insert_slow(ucs_interval_tree_t *tree,
                                           ucs_interval_tree_range_t range)
{
    ucs_interval_tree_range_t merged = range;
    ucs_interval_node_t *new_node;

    ucs_interval_tree_remove_overlapping(tree, &merged);

    new_node = ucs_interval_tree_node_create(tree, merged.start, merged.end);
    if (ucs_unlikely(new_node == NULL)) {
        return UCS_ERR_NO_MEMORY;
    }

    tree->root = ucs_interval_tree_insert_node(tree->root, new_node);
    if (tree->root != NULL) {
        tree->root->parent = NULL;
    }
    ucs_interval_tree_insert_fixup(tree, new_node);
    return UCS_OK;
}

int ucs_interval_tree_pop_any(ucs_interval_tree_t *tree,
                              ucs_interval_tree_range_t *range)
{
    ucs_interval_node_t *node = tree->root;

    if (node == NULL) {
        return 0;
    }

    while (node->left != NULL) {
        node = node->left;
    }

    range->start = node->start;
    range->end   = node->end;
    ucs_interval_tree_remove_node(tree, &tree->root, node);
    return 1;
}
