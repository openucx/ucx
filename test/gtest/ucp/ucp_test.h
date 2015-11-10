/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
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

    class entity {
    public:
        entity(const ucp_test& test);

        void connect(const entity* other);

        void flush() const;

        void disconnect();

        ucp_ep_h ep() const;

        ucp_worker_h worker() const;

        ucp_context_h ucph() const;

        void progress();

    protected:
        ucs::handle<ucp_context_h> m_ucph;
        ucs::handle<ucp_worker_h>  m_worker;
        ucs::handle<ucp_ep_h>      m_ep;
    };

    const ucs::ptr_vector<entity>& entities() const;

protected:
    virtual void cleanup();
    entity* create_entity();
    virtual void get_params(ucp_params_t& params) const;
    void progress() const;
    void short_progress_loop() const;

    ucs::ptr_vector<entity> m_entities;
};


#endif
