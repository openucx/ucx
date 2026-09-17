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
        m_nodes.assign(NUM_NODES, node());
    }

    static node *node_of(const ucs_rbtree_node_t *rb_node)
    {
        return ucs_derived_of(const_cast<ucs_rbtree_node_t*>(rb_node), node);
    }

    void insert(unsigned idx, uint64_t key)
    {
        ucs_rbtree_node_t *parent = NULL, **link = &m_tree.root;

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

    /* Insert in order, node i holding keys[i] */
    void insert_keys(const std::vector<uint64_t> &keys)
    {
        for (unsigned i = 0; i < keys.size(); ++i) {
            insert(i, keys[i]);
        }
    }

    void remove(unsigned idx)
    {
        ucs_rbtree_remove(&m_tree, &m_nodes[idx].super);
        m_nodes[idx].linked = false;
        --m_count;

        EXPECT_EQ(NULL, m_nodes[idx].super.parent);
        EXPECT_EQ(NULL, m_nodes[idx].super.left);
        EXPECT_EQ(NULL, m_nodes[idx].super.right);
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

    /* The search-tree property is that an in-order walk is sorted. */
    void validate()
    {
        EXPECT_TRUE(rbtree_check::validate(&m_tree, m_count));

        const std::vector<uint64_t> ordered = keys();
        EXPECT_TRUE(std::is_sorted(ordered.begin(), ordered.end()));
    }

    void collect(const ucs_rbtree_node_t *rb, std::vector<node*> &out) const
    {
        if (rb != NULL) {
            collect(rb->left, out);
            out.push_back(node_of(rb));
            collect(rb->right, out);
        }
    }

    std::vector<node*> in_order() const
    {
        std::vector<node*> out;
        collect(m_tree.root, out);
        return out;
    }

    std::vector<uint64_t> keys() const
    {
        std::vector<uint64_t> out;
        for (const node *n : in_order()) {
            out.push_back(n->key);
        }
        return out;
    }

    ucs_rbtree_t      m_tree;
    std::vector<node> m_nodes;
    size_t            m_count = 0;
};

UCS_TEST_F(test_rbtree, empty) {
    EXPECT_EQ(NULL, ucs_rbtree_first(&m_tree));
    EXPECT_EQ(NULL, m_tree.root);
    validate();
}

UCS_TEST_F(test_rbtree, insert_orders_and_balances) {
    for (unsigned i = 0; i < 64; ++i) {
        insert(i, i * 10);
        validate();
    }

    EXPECT_EQ(64u, keys().size());
    EXPECT_EQ(0u, node_of(ucs_rbtree_first(&m_tree))->key);
}

UCS_TEST_F(test_rbtree, remove_leaf_and_one_child) {
    insert_keys({20, 10, 30, 5}); /* 10 has a single child, 5 */

    remove(2); /* 30: leaf */
    validate();
    EXPECT_EQ(std::vector<uint64_t>({5, 10, 20}), keys());

    remove(1); /* 10: one child */
    validate();
    EXPECT_EQ(std::vector<uint64_t>({5, 20}), keys());
    EXPECT_EQ(NULL, find(10));
}

UCS_TEST_F(test_rbtree, remove_two_children_relinks_successor) {
    insert_keys({20, 10, 30});
    remove(0); /* successor is 30, the direct right child */
    validate();
    EXPECT_EQ(30u, m_nodes[2].key);
    EXPECT_EQ(&m_nodes[2].super, m_tree.root);
    EXPECT_EQ(std::vector<uint64_t>({10, 30}), keys());
}

UCS_TEST_F(test_rbtree, remove_two_children_deep_successor) {
    insert_keys({20, 10, 40, 30, 50});
    remove(0); /* successor is 30, whose parent is 40, not 20 */
    validate();
    EXPECT_EQ(30u, m_nodes[3].key);
    EXPECT_EQ(&m_nodes[3].super, m_tree.root);
    EXPECT_EQ(std::vector<uint64_t>({10, 30, 40, 50}), keys());
}

/* A node pointer stays usable across removals of unrelated nodes. */
UCS_TEST_F(test_rbtree, node_identity_survives_other_removals) {
    for (unsigned i = 0; i < 32; ++i) {
        insert(i, i * 10);
    }

    for (unsigned i = 0; i < 32; ++i) {
        if (i != 17) {
            remove(i);
            EXPECT_EQ(170u, m_nodes[17].key);
        }
    }

    EXPECT_EQ(&m_nodes[17].super, m_tree.root);
    remove(17);
    EXPECT_EQ(NULL, m_tree.root);
}

UCS_TEST_F(test_rbtree, first_is_leftmost) {
    const std::vector<uint64_t> input = {50, 20, 80, 10, 90, 5};

    for (unsigned i = 0; i < input.size(); ++i) {
        insert(i, input[i]);
        EXPECT_EQ(*std::min_element(input.begin(), input.begin() + i + 1),
                  node_of(ucs_rbtree_first(&m_tree))->key);
    }
}

/* Equal keys descend right, so duplicates keep their insertion order in the
 * in-order walk instead of displacing each other. */
UCS_TEST_F(test_rbtree, equal_keys_keep_insertion_order) {
    insert_keys({100, 100, 100, 100});
    validate();

    const std::vector<node*> ordered = in_order();
    ASSERT_EQ(4u, ordered.size());
    for (unsigned i = 0; i < 4; ++i) {
        EXPECT_EQ(&m_nodes[i], ordered[i]);
    }

    remove(1);
    validate();

    const std::vector<node*> after = in_order();
    ASSERT_EQ(3u, after.size());
    EXPECT_EQ(&m_nodes[0], after[0]);
    EXPECT_EQ(&m_nodes[2], after[1]);
    EXPECT_EQ(&m_nodes[3], after[2]);
}

/* Randomized insert/remove against a reference model, with the red-black and
 * ordering invariants re-checked as it goes. */
UCS_TEST_F(test_rbtree, random_stress) {
    static const unsigned NUM_ITERS = 20000;
    std::map<uint64_t, unsigned> live; /* key -> index */

    for (unsigned i = 0; i < NUM_ITERS; ++i) {
        const unsigned idx = ucs::rand() % NUM_NODES;

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

        const uint64_t probe = ucs::rand() % 100000;
        node *found          = find(probe);
        EXPECT_EQ(live.find(probe) != live.end(), found != NULL);
    }

    validate();

    std::vector<uint64_t> expected;
    for (const auto &kv : live) {
        expected.push_back(kv.first);
    }
    EXPECT_EQ(expected, keys());
}
