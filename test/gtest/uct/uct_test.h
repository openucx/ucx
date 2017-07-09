/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2017.  ALL RIGHTS RESERVED
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_TEST_H_
#define UCT_TEST_H_

extern "C" {
#include <uct/api/uct.h>
#include <ucs/sys/sys.h>
#include <ucs/async/async.h>
}
#include <common/test.h>
#include <vector>


#define UCT_TEST_TIMEOUT_IN_SEC   10.0
#define DEFAULT_DELAY_MS           1.0
#define DEFAULT_TIMEOUT_SEC       10.0


/* Testing resource */
struct resource {
    virtual ~resource() {};
    virtual std::string name() const;
    std::string       md_name;
    cpu_set_t         local_cpus;
    std::string       tl_name;
    std::string       dev_name;
    uct_device_type_t dev_type;
};


/**
 * UCT test, parametrized on a transport/device.
 */
class uct_test : public testing::TestWithParam<const resource*>,
                 public ucs::test_base {
public:
    UCS_TEST_BASE_IMPL;

    static std::vector<const resource*> enum_resources(const std::string& tl_name,
                                                       bool loopback = false);

    uct_test();
    virtual ~uct_test();

protected:

    class entity {
    public:
        entity(const resource& resource, uct_iface_config_t *iface_config,
               uct_iface_params_t *params, uct_md_config_t *md_config);

        void mem_alloc(size_t length, uct_allocated_memory_t *mem,
                       uct_rkey_bundle *rkey_bundle, int is_nc = 0) const;

        void mem_free(const uct_allocated_memory_t *mem,
                      const uct_rkey_bundle_t& rkey) const;

        void progress() const;

        uct_md_h md() const;

        const uct_md_attr& md_attr() const;

        uct_worker_h worker() const;

        uct_iface_h iface() const;

        const uct_iface_attr& iface_attr() const;

        uct_ep_h ep(unsigned index) const;

        void create_ep(unsigned index);
        void destroy_ep(unsigned index);
        void destroy_eps();
        void connect(unsigned index, entity& other, unsigned other_index);
        void connect_to_iface(unsigned index, entity& other);
        void connect_to_ep(unsigned index, entity& other,
                           unsigned other_index);

        void flush() const;

    private:
        class async_wrapper {
        public:
            ucs_async_context_t   m_async;
            async_wrapper();
            ~async_wrapper();
            void check_miss();
        private:
            async_wrapper(const async_wrapper &);
        };
        typedef std::vector< ucs::handle<uct_ep_h> > eps_vec_t;

        entity(const entity&);

        void reserve_ep(unsigned index);

        void connect_p2p_ep(uct_ep_h from, uct_ep_h to);

        ucs::handle<uct_md_h>      m_md;
        uct_md_attr_t              m_md_attr;
        mutable async_wrapper      m_async;
        ucs::handle<uct_worker_h>  m_worker;
        ucs::handle<uct_iface_h>   m_iface;
        eps_vec_t                  m_eps;
        uct_iface_attr_t           m_iface_attr;
    };

    class mapped_buffer {
    public:
        mapped_buffer(size_t size, uint64_t seed, const entity& entity,
                      size_t offset = 0, size_t stride = 0, uct_ep_h ep = NULL);
        virtual ~mapped_buffer();

        void *ptr() const;
        uintptr_t addr() const;
        size_t length() const;
        uct_mem_h memh() const;
        uct_rkey_t rkey() const;
        const uct_iov_t* iov() const;

        void pattern_fill(uint64_t seed);
        void pattern_check(uint64_t seed);

        static size_t pack(void *dest, void *arg);
        static void pattern_fill(void *buffer, size_t length, uint64_t seed);
        static void pattern_check(const void *buffer, size_t length);
        static void pattern_check(const void *buffer, size_t length, uint64_t seed);

        size_t nc_length() const;
        uct_mem_h nc_memh() const;
        uct_rkey_t nc_rkey() const;
        ucs_status_t nc_map(uct_ep_h ep);
        void nc_unmap();

    private:
        static uint64_t pat(uint64_t prev);

        const uct_test::entity& m_entity;

        void                    *m_buf;
        void                    *m_end;
        uct_rkey_bundle_t       m_rkey;
        uct_allocated_memory_t  m_mem;
        uct_iov_t               m_iov;

        uct_md_h                m_nc_md;
        uct_mem_h               m_nc_memh;
        uct_rkey_bundle_t       m_nc_rkey;
        uct_completion_t        m_nc_comp;
    };

    template <typename T>
    static std::vector<const resource*> filter_resources(const std::vector<T>& resources,
                                                         const std::string& tl_name)
    {
        std::vector<const resource*> result;
        for (typename std::vector<T>::const_iterator iter = resources.begin();
                        iter != resources.end(); ++iter)
        {
            if (tl_name.empty() || (iter->tl_name == tl_name)) {
                result.push_back(&*iter);
            }
        }
        return result;
    }

    template <typename T>
    void wait_for_flag(volatile T *flag, double timeout = DEFAULT_TIMEOUT_SEC) const
    {
        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(timeout) * ucs::test_time_multiplier();
        while ((ucs_get_time() < deadline) && (!(*flag))) {
            short_progress_loop();
        }
    }

    template <typename T>
    void wait_for_value(volatile T *var, T value, bool progress,
                        double timeout = DEFAULT_TIMEOUT_SEC) const
    {
        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(timeout) * ucs::test_time_multiplier();
        while ((ucs_get_time() < deadline) && (*var != value)) {
            if (progress) {
                short_progress_loop();
            } else {
                twait();
            }
        }
    }

    virtual void init();
    virtual void cleanup();
    virtual void modify_config(const std::string& name, const std::string& value);
    void stats_activate();
    void stats_restore();


    void check_caps(uint64_t required_flags, uint64_t invalid_flags = 0);
    const entity& ent(unsigned index) const;
    void progress() const;
    void flush() const;
    virtual void short_progress_loop(double delay_ms = DEFAULT_DELAY_MS) const;
    virtual void twait(int delta_ms = DEFAULT_DELAY_MS) const;

    uct_test::entity* create_entity(size_t rx_headroom);
    uct_test::entity* create_entity(uct_iface_params_t &params);

    ucs::ptr_vector<entity> m_entities;
    uct_iface_config_t      *m_iface_config;
    uct_md_config_t         *m_md_config;

};

std::ostream& operator<<(std::ostream& os, const resource* resource);


#define UCT_TEST_IB_TLS \
    rc_mlx5,            \
    rc,                 \
    dc,                 \
    dc_mlx5,            \
    ud,                 \
    ud_mlx5,            \
    cm

#define UCT_TEST_NO_SELF_TLS \
    UCT_TEST_IB_TLS,         \
    ugni_rdma,               \
    ugni_udt,                \
    ugni_smsg,               \
    tcp,                     \
    mm,                      \
    cma,                     \
    knem,                    \
    cuda,                    \
    rocm

#define UCT_TEST_TLS      \
    UCT_TEST_NO_SELF_TLS, \
    self

/**
 * Instantiate the parametrized test case for all transports.
 *
 * @param _test_case  Test case class, derived from uct_test.
 */
#define UCT_INSTANTIATE_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_INSTANTIATE_TEST_CASE, _test_case, UCT_TEST_TLS)
#define _UCT_INSTANTIATE_TEST_CASE(_test_case, _tl_name) \
    INSTANTIATE_TEST_CASE_P(_tl_name, _test_case, \
                            testing::ValuesIn(_test_case::enum_resources(UCS_PP_QUOTE(_tl_name))));


/**
 * Instantiate the parametrized test case for the IB transports.
 *
 * @param _test_case  Test case class, derived from uct_test.
 */
#define UCT_INSTANTIATE_IB_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_INSTANTIATE_TEST_CASE, _test_case, UCT_TEST_IB_TLS)

/**
 * Instantiate the parametrized test case for all transports excluding SELF.
 *
 * @param _test_case  Test case class, derived from uct_test.
 */
#define UCT_INSTANTIATE_NO_SELF_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_INSTANTIATE_TEST_CASE, _test_case, UCT_TEST_NO_SELF_TLS)

std::ostream& operator<<(std::ostream& os, const uct_tl_resource_desc_t& resource);

#endif
