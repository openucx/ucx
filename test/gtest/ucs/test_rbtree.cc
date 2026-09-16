/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include "rbtree_check.h"

#include <algorithm>
#include <map>
#include <vector>

class test_rbtree : public ucs::test {
protected:
    struct node {
        ucs_rbtree_node_t super;
        uint64_t          key;
        bool              linked;
    };

    static const unsigned NUM_NODES = 256;

    void init()
    {
        ucs::test::init();
        ucs_rbtree_init(&m_tree);
        m_nodes.resize(NUM_NODES);
        for (unsigned i = 0; i < NUM_NODES; ++i) {
            m_nodes[i].linked = false;
        }
    }

    static node *node_of(const ucs_rbtree_node_t *rb_node)
    {
        return ucs_derived_of(const_cast<ucs_rbtree_node_t*>(rb_node), node);
    }

    void insert(unsigned idx, uint64_t key)
    {
        ucs_rbtree_node_t *parent = NULL;
        ucs_rbtree_node_t **link  = &m_tree.root;

        while (*link != NULL) {
            parent = *link;
            link   = (key < node_of(parent)->key) ? &parent->left :
                                                    &parent->right;
        }

        m_nodes[idx].key    = key;
        m_nodes[idx].linked = true;
        ucs_rbtree_insert_at(&m_tree, parent, link, &m_nodes[idx].super);
        ++m_count;
    }

    void remove(unsigned idx)
    {
        ucs_rbtree_remove(&m_tree, &m_nodes[idx].super);
        m_nodes[idx].linked = false;
        --m_count;
    }

    node *find(uint64_t key) const
    {
        ucs_rbtree_node_t *rb_node = m_tree.root;

        while (rb_node != NULL) {
            node *n = node_of(rb_node);
            if (key == n->key) {
                return n;
            }
            rb_node = (key < n->key) ? rb_node->left : rb_node->right;
        }

        return NULL;
    }

    /* Search-tree ordering; the red-black invariants checked by rbtree_check */
    static void check_ordering(const ucs_rbtree_node_t *rb_node)
    {
        if (rb_node->left != NULL) {
            EXPECT_LT(node_of(rb_node->left)->key, node_of(rb_node)->key);
        }
        if (rb_node->right != NULL) {
            EXPECT_GE(node_of(rb_node->right)->key, node_of(rb_node)->key);
        }
    }

    void validate()
    {
        EXPECT_TRUE(rbtree_check::validate(&m_tree, m_count, check_ordering));
    }

    std::vector<uint64_t> in_order() const
    {
        std::vector<uint64_t> out;
        collect(m_tree.root, out);
        return out;
    }

    static void collect(const ucs_rbtree_node_t *rb_node,
                        std::vector<uint64_t> &out)
    {
        if (rb_node == NULL) {
            return;
        }
        collect(rb_node->left, out);
        out.push_back(node_of(rb_node)->key);
        collect(rb_node->right, out);
    }

    static void expect_detached(const node *n)
    {
        EXPECT_EQ(NULL, n->super.parent);
        EXPECT_EQ(NULL, n->super.left);
        EXPECT_EQ(NULL, n->super.right);
    }

    ucs_rbtree_t      m_tree;
    std::vector<node> m_nodes;
    size_t            m_count = 0;
};

UCS_TEST_F(test_rbtree, empty)
{
    EXPECT_EQ(NULL, ucs_rbtree_first(&m_tree));
    EXPECT_EQ(NULL, m_tree.root);
    validate();
}

UCS_TEST_F(test_rbtree, insert_orders_and_balances)
{
    /* Ascending keys are the worst case for an unbalanced tree */
    for (unsigned i = 0; i < 64; ++i) {
        insert(i, i * 10);
        validate();
    }

    std::vector<uint64_t> keys = in_order();
    EXPECT_TRUE(std::is_sorted(keys.begin(), keys.end()));
    EXPECT_EQ(64u, keys.size());
    EXPECT_EQ(0u, node_of(ucs_rbtree_first(&m_tree))->key);
}

UCS_TEST_F(test_rbtree, remove_leaf)
{
    insert(0, 20);
    insert(1, 10);
    insert(2, 30);

    remove(1);
    validate();
    EXPECT_EQ(NULL, find(10));
    EXPECT_EQ(std::vector<uint64_t>({20, 30}), in_order());
    expect_detached(&m_nodes[1]);
}

UCS_TEST_F(test_rbtree, remove_one_child)
{
    insert(0, 20);
    insert(1, 10);
    insert(2, 5);

    remove(1); /* 10 has a single child, 5 */
    validate();
    EXPECT_EQ(NULL, find(10));
    EXPECT_EQ(std::vector<uint64_t>({5, 20}), in_order());
    expect_detached(&m_nodes[1]);
}

/* The successor is relinked rather than having its key copied into the removed
 * node, so every surviving node object keeps its own key and the object the
 * caller named is the one detached. */
UCS_TEST_F(test_rbtree, remove_two_children_relinks_successor)
{
    insert(0, 20);
    insert(1, 10);
    insert(2, 30);
    validate();

    remove(0);
    validate();

    EXPECT_EQ(30u, m_nodes[2].key); /* successor kept its own key */
    EXPECT_EQ(10u, m_nodes[1].key);
    EXPECT_EQ(&m_nodes[2].super, m_tree.root);
    EXPECT_EQ(std::vector<uint64_t>({10, 30}), in_order());
    expect_detached(&m_nodes[0]);
}

UCS_TEST_F(test_rbtree, remove_two_children_deep_successor)
{
    insert(0, 20);
    insert(1, 10);
    insert(2, 40);
    insert(3, 30);
    insert(4, 50);
    validate();

    remove(0); /* successor is 30, whose parent is 40, not 20 */
    validate();

    EXPECT_EQ(30u, m_nodes[3].key);
    EXPECT_EQ(&m_nodes[3].super, m_tree.root);
    EXPECT_EQ(std::vector<uint64_t>({10, 30, 40, 50}), in_order());
    expect_detached(&m_nodes[0]);
}

/* A node pointer stays usable across removals of unrelated nodes. */
UCS_TEST_F(test_rbtree, node_identity_survives_other_removals)
{
    for (unsigned i = 0; i < 32; ++i) {
        insert(i, i * 10);
    }

    node *kept = &m_nodes[17];
    for (unsigned i = 0; i < 32; ++i) {
        if (i != 17) {
            remove(i);
            EXPECT_EQ(170u, kept->key);
        }
    }

    validate();
    EXPECT_EQ(&kept->super, m_tree.root);
    remove(17);
    validate();
    EXPECT_EQ(NULL, m_tree.root);
}

UCS_TEST_F(test_rbtree, first_is_leftmost)
{
    const std::vector<uint64_t> keys = {50, 20, 80, 10, 90, 5};

    for (unsigned i = 0; i < keys.size(); ++i) {
        insert(i, keys[i]);
        uint64_t smallest = *std::min_element(keys.begin(),
                                              keys.begin() + i + 1);
        EXPECT_EQ(smallest, node_of(ucs_rbtree_first(&m_tree))->key);
    }
}

/* Randomized insert/remove against a reference model, with the red-black and
 * ordering invariants re-checked as it goes. */
UCS_TEST_F(test_rbtree, random_stress)
{
    static const unsigned NUM_ITERS = 20000;
    std::map<uint64_t, unsigned> live; /* key -> index */

    for (unsigned i = 0; i < NUM_ITERS; ++i) {
        unsigned idx = ucs::rand() % NUM_NODES;

        if (m_nodes[idx].linked) {
            live.erase(m_nodes[idx].key);
            remove(idx);
        } else {
            uint64_t key;
            do {
                key = ucs::rand() % 100000;
            } while (live.find(key) != live.end());
            live[key] = idx;
            insert(idx, key);
        }

        if ((i & 15) == 0) {
            validate();
        }

        uint64_t probe = ucs::rand() % 100000;
        node *found    = find(probe);
        EXPECT_EQ(live.find(probe) != live.end(), found != NULL);
        if (found != NULL) {
            EXPECT_EQ(probe, found->key);
        }
    }

    validate();

    std::vector<uint64_t> expected;
    for (const auto &kv : live) {
        expected.push_back(kv.first);
    }
    EXPECT_EQ(expected, in_order());
}
