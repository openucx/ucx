/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

extern "C" {
#include <uct/api/uct.h>
}
#include <common/test.h>
#include "uct_test.h"


class test_uct_progress : public uct_test {
public:
    virtual void init() {
        uct_test::init();
        m_entities.push_back(create_entity(0));
    }
};


UCS_TEST_P(test_uct_progress, random_enable_disable) {
    for (int i = 0; i < 100; ++i) {
        unsigned flags = 0;
        if (ucs::rand() % 2) {
            flags |= UCT_PROGRESS_SEND;
        }
        if (ucs::rand() % 2) {
            flags |= UCT_PROGRESS_RECV;
        }
        if (ucs::rand() % 2) {
            uct_iface_progress_enable(ent(0).iface(), flags);
        } else {
            uct_iface_progress_disable(ent(0).iface(), flags);
        }
        progress();
    }

}


UCT_INSTANTIATE_TEST_CASE(test_uct_progress);
