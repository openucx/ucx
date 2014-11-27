/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_TEST_H_
#define UCT_TEST_H_

extern "C" {
#include <uct/api/uct.h>
}
#include <ucs/gtest/test.h>


/**
 * UCT test, parameterized on a transport/device.
 */
class uct_test : public testing::TestWithParam<uct_resource_desc_t>,
                 public ucs::test_base {
public:
    UCS_TEST_BASE_IMPL;

    static std::vector<uct_resource_desc_t> enum_resources(const std::string& tl_name);

protected:
    class entity {
    public:
        entity(const uct_resource_desc_t& resource);
        ~entity();

        void connect(const entity& other);

        uct_rkey_bundle_t mem_map(void *address, size_t length, uct_lkey_t *lkey_p) const;

        void mem_unmap(uct_lkey_t lkey, const uct_rkey_bundle_t& rkey) const;

        uct_ep_h ep() const;

        void flush() const;

    protected:
        uct_context_h m_ucth;
        uct_iface_h   m_iface;
        uct_ep_h      m_ep;
    };
};


#define UCT_TEST_TLS \
    rc_mlx5, \
    rc_verbs

/**
 * Instantiate the parameterized test case for all transports.
 *
 * @param _test_case  Test case class, derived form uct_test.
 */
#define UCT_INSTANTIATE_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_INSTANTIATE_TEST_CASE, _test_case, UCT_TEST_TLS)
#define _UCT_INSTANTIATE_TEST_CASE(_test_case, _tl_name) \
    INSTANTIATE_TEST_CASE_P(_tl_name, _test_case, \
                            testing::ValuesIn(uct_test::enum_resources(UCS_PP_QUOTE(_tl_name))));

std::ostream& operator<<(std::ostream& os, const uct_resource_desc_t& resource);

#endif
