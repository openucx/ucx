/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_test.h"

class test_pd : public uct_test {
public:
    virtual void init() {
        uct_test::init();
        m_entities.push_back(new entity(GetParam()));
    }

protected:
    const entity& e() {
        return ent(0);
    }
};

UCS_TEST_P(test_pd, alloc) {
    mapped_buffer b(1 * 1024 * 1024, 64, 0, e());
}

UCT_INSTANTIATE_TEST_CASE(test_pd)
