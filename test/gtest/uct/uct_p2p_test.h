/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_P2P_TEST_H_
#define UCT_P2P_TEST_H_

#include "uct_test.h"

#include <boost/ptr_container/ptr_vector.hpp>


/**
 * Point-to-point UCT test.
 */
class uct_p2p_test : public uct_test {
public:
    virtual void init();
    virtual void cleanup();

    void short_progress_loop();

    UCS_TEST_BASE_IMPL;
protected:
    const entity &get_entity(unsigned index) const;

    boost::ptr_vector<entity> m_entities;
};


#endif
