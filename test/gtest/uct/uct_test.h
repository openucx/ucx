/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_TEST_H_
#define UCT_TEST_H_

extern "C" {
#include <uct/api/uct.h>
}
#include <ucs/gtest/test.h>
#include <vector>


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

        uct_rkey_bundle_t mem_map(void *address, size_t length, uct_lkey_t *lkey_p) const;

        void mem_unmap(uct_lkey_t lkey, const uct_rkey_bundle_t& rkey) const;

        void progress() const;

        uct_iface_h iface() const;

        const uct_iface_attr& iface_attr() const;

        void add_ep();

        uct_ep_h ep(unsigned index) const;

        void connect(unsigned index, const entity& other, unsigned other_index) const;

        void flush() const;

    private:
        entity(const entity&);

        uct_context_h         m_ucth;
        uct_iface_h           m_iface;
        std::vector<uct_ep_h> m_eps;
        uct_iface_attr_t      m_iface_attr;
    };

    class buffer {
    public:
        buffer(size_t size, size_t alignment, uint64_t seed);
        virtual ~buffer();

        static void pattern_check(void *buffer, size_t length, uint64_t seed);
        void pattern_fill(uint64_t seed);
        void pattern_check(uint64_t seed);
        void *ptr() const;
        uintptr_t addr() const;
        size_t length() const;
    private:
        static uint64_t pat(uint64_t prev);

        void             *m_buf;
        void             *m_end;
    };

    class mapped_buffer : public buffer {
    public:
        mapped_buffer(size_t size, size_t alignment, uint64_t seed, const entity& entity);
        ~mapped_buffer();

        uct_lkey_t lkey() const;
        uct_rkey_t rkey() const;
    private:
        const uct_test::entity& m_entity;
        uct_lkey_t              m_lkey;
        uct_rkey_bundle_t       m_rkey;
    };


    virtual void init();
    virtual void cleanup();

    void check_caps(uint64_t flags);
    const entity& ent(unsigned index) const;
    void progress() const;

    ucs::ptr_vector<entity> m_entities;
};


#define UCT_TEST_TLS \
    rc_mlx5, \
    rc, \
    ud_verbs, \
    ugni, \
    sysv

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
