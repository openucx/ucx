/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_p2p_test.h"
extern "C" {
#include <ucs/time/time.h>
}

void uct_p2p_test::init() {
    for (unsigned i =0; i < 2; ++i) {
        m_entities.push_back(new entity(GetParam()));
    }

    m_entities[0].connect(m_entities[1]);
    m_entities[1].connect(m_entities[0]);
}

void uct_p2p_test::cleanup() {
    m_entities.clear();
}

void uct_p2p_test::short_progress_loop() {
    ucs_time_t end_time = ucs_get_time() + ucs_time_from_msec(1.0);
    while (ucs_get_time() < end_time) {
        for (ucs::ptr_vector<entity>::const_iterator iter = m_entities.begin();
             iter != m_entities.end();
             ++iter)
        {
            (*iter)->progress();
        }
    }
}

const uct_test::entity& uct_p2p_test::get_entity(unsigned index) const {
    return m_entities[index];
}

