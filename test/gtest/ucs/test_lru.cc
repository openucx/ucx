/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include "ucs/datastruct/lru.h"
}

class test_lru : public ucs::test {
protected:
    virtual void init()
    {
        ucs::test::init();
        ASSERT_UCS_OK(ucs_lru_create(m_capacity, &m_lru));
    }

    virtual void cleanup()
    {
        ucs_lru_destroy(m_lru);
        ucs::test::cleanup();
    }

    void init_vector(std::vector<uint64_t> &elements, size_t capacity,
                     uint64_t init_value)
    {
        for (uint64_t i = 0; i < capacity; ++i) {
            elements.push_back(i + init_value);
        }
    }

    void
    run(const std::vector<uint64_t> &elements, std::vector<uint64_t> &expected)
    {
        for (size_t i = 0; i < m_capacity * 10; ++i) {
            ucs_lru_push(m_lru, (void*)elements[i % elements.size()]);
        }

        int elem_index = 0;
        void **item;

        std::reverse(expected.begin(), expected.end());

        ucs_lru_for_each(item, m_lru) {
            EXPECT_EQ(expected[elem_index], (uint64_t)*item);
            elem_index++;
        }

        EXPECT_EQ(expected.size(), elem_index);
        std::reverse(expected.begin(), expected.end());
    }

    static constexpr size_t m_capacity = 10;
    ucs_lru_h               m_lru      = NULL;
};

UCS_TEST_F(test_lru, full_capacity) {
    std::vector<uint64_t> elements;
    init_vector(elements, m_capacity * 2, 0);

    std::vector<uint64_t> expected(elements.begin() + m_capacity,
                                   elements.end());
    run(elements, expected);
}

UCS_TEST_F(test_lru, partial_capacity) {
    std::vector<uint64_t> elements;
    init_vector(elements, m_capacity / 2, 0);
    run(elements, elements);
}

UCS_TEST_F(test_lru, combined) {
    std::vector<uint64_t> elements1;
    init_vector(elements1, m_capacity, 0);
    run(elements1, elements1);

    std::vector<uint64_t> elements2;
    init_vector(elements2, m_capacity, m_capacity);
    run(elements2, elements2);
}

UCS_TEST_F(test_lru, reset) {
    std::vector<uint64_t> elements1;
    init_vector(elements1, m_capacity, 0);
    run(elements1, elements1);

    ucs_lru_reset(m_lru);

    std::vector<uint64_t> elements2;
    init_vector(elements2, m_capacity / 2, m_capacity);
    run(elements2, elements2);
}

UCS_TEST_F(test_lru, pop_oldest) {
    std::vector<uint64_t> elements1;
    init_vector(elements1, m_capacity, 0);
    run(elements1, elements1);

    std::vector<uint64_t> elements2;
    init_vector(elements2, m_capacity / 2, m_capacity);

    std::vector<uint64_t> expected(elements1.begin() + m_capacity / 2,
                                   elements1.end());
    expected.insert(expected.end(), elements2.begin(), elements2.end());
    run(elements2, expected);
}
