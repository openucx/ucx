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

    void run(std::vector<int> &elements)
    {
        for (size_t i = 0; i < m_capacity * 10; ++i) {
            ucs_lru_put(m_lru, &elements[i % elements.size()]);
        }

        int elem_index = 0;
        void **item;

        ucs_lru_for_each(item, m_lru) {
            EXPECT_EQ((int*)*item,
                      &elements[elements.size() - 1 - elem_index]);
            elem_index++;
        }

        size_t capacity  = m_capacity;
        int result_count = std::min(capacity, elements.size());
        EXPECT_EQ(elem_index, result_count);
    }

    static constexpr size_t m_capacity = 10;
    ucs_lru_h               m_lru      = NULL;
};

UCS_TEST_F(test_lru, full_capacity) {
    std::vector<int> elements(m_capacity * 2);
    run(elements);
}

UCS_TEST_F(test_lru, partial_capacity) {
    std::vector<int> elements(m_capacity / 2);
    run(elements);
}

UCS_TEST_F(test_lru, combined) {
    std::vector<int> elements1(m_capacity);
    run(elements1);

    std::vector<int> elements2(m_capacity);
    run(elements2);
}
