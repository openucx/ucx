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

    uct_test();
    virtual ~uct_test();

protected:

    class entity {
    public:
        entity(const uct_resource_desc_t& resource, const uct_iface_config_t* iface_config, size_t rx_headroom);
        ~entity();

        void mem_alloc(void **address_p, size_t *length_p, size_t alignement,
                       uct_mem_h *memh_p, uct_rkey_bundle *rkey_bundle) const;

        void mem_free(void *address, uct_mem_h memh,
                      const uct_rkey_bundle_t& rkey) const;

        void progress() const;

        uct_iface_h iface() const;

        uct_worker_h worker() const;

        const uct_iface_attr& iface_attr() const;

        void add_ep();

        uct_ep_h ep(unsigned index) const;

        void connect(unsigned index, const entity& other, unsigned other_index) const;

        void flush() const;

    private:
        entity(const entity&);

        uct_context_h         m_ucth;
        uct_worker_h          m_worker;
        uct_iface_h           m_iface;
        std::vector<uct_ep_h> m_eps;
        uct_iface_attr_t      m_iface_attr;
    };

    class mapped_buffer {
    public:
        mapped_buffer(size_t size, size_t alignment, uint64_t seed, 
                      const entity& entity, size_t offset = 0);
        virtual ~mapped_buffer();

        void *ptr() const;
        uintptr_t addr() const;
        size_t length() const;
        uct_mem_h memh() const;
        uct_rkey_t rkey() const;

        void pattern_fill(uint64_t seed);
        void pattern_check(uint64_t seed);

        static void pattern_check(void *buffer, size_t length, uint64_t seed);
    private:
        static uint64_t pat(uint64_t prev);

        const uct_test::entity& m_entity;

        void                    *m_buf;
        void                    *m_buf_real;
        void                    *m_end;
        uct_mem_h               m_memh;
        uct_rkey_bundle_t       m_rkey;
    };


    virtual void init();
    virtual void cleanup();
    virtual void modify_config(const std::string& name, const std::string& value);

    void check_caps(uint64_t flags);
    const entity& ent(unsigned index) const;
    void progress() const;
    uct_test::entity* create_entity(size_t rx_headroom);

    ucs::ptr_vector<entity> m_entities;
    uct_iface_config_t      *m_iface_config;
    uct_context_h           m_dummy_ctx;
    uct_worker_h            m_dummy_worker;
};


#define UCT_TEST_TLS \
    rc_mlx5, \
    rc, \
    ud, \
    ugni, \
    sysv, \
    cuda

#define UCT_TEST_IB_TLS \
    rc_mlx5, \
    rc, \
    ud

/**
 * Instantiate the parameterized test case for all transports.
 *
 * @param _test_case  Test case class, derived from uct_test.
 */
#define UCT_INSTANTIATE_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_INSTANTIATE_TEST_CASE, _test_case, UCT_TEST_TLS)
#define _UCT_INSTANTIATE_TEST_CASE(_test_case, _tl_name) \
    INSTANTIATE_TEST_CASE_P(_tl_name, _test_case, \
                            testing::ValuesIn(uct_test::enum_resources(UCS_PP_QUOTE(_tl_name))));


/**
 * Instantiate the parameterized test case for the IB transports.
 *
 * @param _test_case  Test case class, derived from uct_test.
 */
#define UCT_INSTANTIATE_IB_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_INSTANTIATE_TEST_CASE, _test_case, UCT_TEST_IB_TLS)

std::ostream& operator<<(std::ostream& os, const uct_resource_desc_t& resource);

#endif
