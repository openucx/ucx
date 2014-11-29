/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_p2p_test.h"


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

const uct_test::entity& uct_p2p_test::get_entity(unsigned index) const {
    return m_entities[index];
}

