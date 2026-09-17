/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_RBTREE_H_
#define UCS_RBTREE_H_

#include <ucs/sys/compiler_def.h>
#include <stddef.h>

BEGIN_C_DECLS

/**
 * Intrusive red-black tree.
 */
typedef struct ucs_rbtree_node ucs_rbtree_node_t;


typedef enum {
    UCS_RBTREE_BLACK = 0,
    UCS_RBTREE_RED   = 1
} ucs_rbtree_color_t;


struct ucs_rbtree_node {
    ucs_rbtree_node_t  *parent; /**< Parent, NULL for the root */
    ucs_rbtree_node_t  *left; /**< Left child */
    ucs_rbtree_node_t  *right; /**< Right child */
    ucs_rbtree_color_t color; /**< Node color */
};


typedef struct {
    ucs_rbtree_node_t *root; /**< Root node, NULL when empty */
} ucs_rbtree_t;


/**
 * @brief Initialize an empty tree
 *
 * @param [in]  tree  Tree to initialize.
 */
void ucs_rbtree_init(ucs_rbtree_t *tree);


/**
 * @brief Leftmost node, or NULL if the tree is empty
 */
ucs_rbtree_node_t *ucs_rbtree_first(const ucs_rbtree_t *tree);


/**
 * @brief Attach a node and restore the red-black invariants
 *
 * The caller descends to the insertion point itself, then passes the node that
 * will become the parent together with the child slot to fill:
 *
 * @code{.c}
 * ucs_rbtree_node_t *parent = NULL, **link = &tree->root;
 *
 * while (*link != NULL) {
 *     parent = *link;
 *     link   = (key < key_of(parent)) ? &parent->left : &parent->right;
 * }
 * ucs_rbtree_insert_at(tree, parent, link, node);
 * @endcode
 *
 * @param [in]  tree    Tree to insert into.
 * @param [in]  parent  Node to attach under, NULL to make @a node the root.
 * @param [in]  link    Empty child slot of @a parent, or &tree->root.
 * @param [in]  node    Node to attach. Its links and color are overwritten.
 */
void ucs_rbtree_insert_at(ucs_rbtree_t *tree, ucs_rbtree_node_t *parent,
                          ucs_rbtree_node_t **link, ucs_rbtree_node_t *node);


/**
 * @brief Remove a node. Other nodes are neither moved nor invalidated.
 *
 * @param [in]  tree  Tree to remove from.
 * @param [in]  node  Node to remove, which must be in @a tree.
 */
void ucs_rbtree_remove(ucs_rbtree_t *tree, ucs_rbtree_node_t *node);

END_C_DECLS

#endif
