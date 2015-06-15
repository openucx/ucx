/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCP_TEST_H_
#define UCP_TEST_H_

extern "C" {
#include <ucp/api/ucp.h>
}
#include <ucs/gtest/test.h>


/**
 * UCP test
 */
class ucp_test : public ucs::test_base, public ::testing::Test {

public:
    UCS_TEST_BASE_IMPL;

protected:
    class entity {
    public:
        entity();

        void connect(const entity& other);

        void disconnect();

        ucp_ep_h ep() const;

        ucp_worker_h worker() const;

        ucp_context_h ucph() const;


    protected:
        ucs::handle<ucp_context_h> m_ucph;
        ucs::handle<ucp_worker_h>  m_worker;
        ucs::handle<ucp_ep_h>      m_ep;
    };
};


#endif
