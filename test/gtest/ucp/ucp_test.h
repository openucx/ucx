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
class ucp_test : public testing::TestWithParam<uct_resource_desc_t>,
                 public ucs::test_base {

public:
    UCS_TEST_BASE_IMPL;

    static std::vector<uct_resource_desc_t> enum_resources();

protected:
    class entity {
    public:
        entity();
        ~entity();

        void connect(const entity& other);

        ucp_ep_h ep() const;

        void flush() const;

    protected:
        ucp_context_h m_ucph;
        ucp_iface_h   m_iface;
        ucp_ep_h      m_ep;
    };
};


/**
 * Instantiate the parameterized test case..
 *
 * @param _test_case  Test case class, derived form ucp_test.
 */
#define UCP_INSTANTIATE_TEST_CASE(_test_case) \
    INSTANTIATE_TEST_CASE_P(ucp, _test_case, \
                            testing::ValuesIn(ucp_test::enum_resources()));

#endif /* UCP_TEST_H_ */
