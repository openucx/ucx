/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2017.  ALL RIGHTS RESERVED
* Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_TEST_H_
#define UCT_TEST_H_

#include <common/test.h>

#include <poll.h>
#include <uct/api/uct.h>
#include <ucs/sys/sys.h>
#include <ucs/async/async.h>
#include <ucs/async/pipe.h>
#include <common/mem_buffer.h>
#include <common/test.h>
#include <vector>



#define DEFAULT_DELAY_MS           1.0
#define DEFAULT_TIMEOUT_SEC       10.0
#define DEFAULT_VARIANT              0

#define UCT_TEST_CALL_AND_TRY_AGAIN(_func, _res) \
    do { \
        _res = _func; \
        if (_res == UCS_ERR_NO_RESOURCE) { \
            short_progress_loop(); \
        } \
    } while (_res == UCS_ERR_NO_RESOURCE)


#define FOR_EACH_ENTITY(_iter) \
    for (ucs::ptr_vector<entity>::const_iterator _iter = m_entities.begin(); \
         _iter != m_entities.end(); ++_iter) \


/* Testing resource */
struct resource {
    virtual ~resource() {};
    virtual std::string name() const;
    uct_component_h         component;
    std::string             component_name;
    std::string             md_name;
    ucs_cpu_set_t           local_cpus;
    std::string             tl_name;
    std::string             dev_name;
    std::string             variant_name;
    uct_device_type_t       dev_type;
    ucs::sock_addr_storage  listen_sock_addr;     /* sockaddr to listen on */
    ucs::sock_addr_storage  connect_sock_addr;    /* sockaddr to connect to */
    ucs::sock_addr_storage  source_sock_addr;     /* sockaddr to connect from */
    int                     variant;

    resource();
    resource(uct_component_h component, const std::string& component_name,
             const std::string& md_name, const ucs_cpu_set_t& local_cpus,
             const std::string& tl_name, const std::string& dev_name,
             uct_device_type_t dev_type);
    resource(uct_component_h component, const uct_component_attr& cmpnt_attr,
             const uct_md_attr_t& md_attr,
             const uct_md_resource_desc_t& md_resource,
             const uct_tl_resource_desc_t& tl_resource);
};

struct resource_speed : public resource {
    double bw;

    resource_speed() : resource(), bw(0) { }
    resource_speed(uct_component_h component,
                   const uct_component_attr& cmpnt_attr,
                   const uct_worker_h& worker, const uct_md_h& md,
                   const uct_md_attr_t& md_attr,
                   const uct_md_resource_desc_t& md_resource,
                   const uct_tl_resource_desc_t& tl_resource);
};


/**
 * UCT test, without parameterization
 */
class uct_test_base : public ucs::test_base {
protected:
    struct md_resource {
        uct_component_h        cmpt;
        uct_component_attr_t   cmpt_attr;
        uct_md_resource_desc_t rsc_desc;
    };

    static std::vector<md_resource> enum_md_resources();
};


/**
 * UCT test, parametrized on a transport/device.
 */
class uct_test : public testing::TestWithParam<const resource*>,
                 public uct_test_base {
public:
    UCS_TEST_BASE_IMPL;

    /* we return a vector of pointers to allow test fixtures to extend the
     * resource structure.
     */
    static std::vector<const resource*> enum_resources(const std::string& tl_name);

    /* By default generate test variant for all tls. If variant is specific to
     * the particular transport tl_name need to be specified accordingly */
    static void generate_test_variant(int variant,
                                      const std::string &variant_name,
                                      std::vector<resource>& test_res,
                                      const std::string &tl_name="");
    uct_test();
    virtual ~uct_test();

    enum atomic_mode {
        OP32,
        OP64,
        FOP32,
        FOP64
    };

protected:

    class entity {
    public:
        typedef uct_test::atomic_mode atomic_mode;
        typedef std::vector< ucs::handle<uct_ep_h> > eps_vec_t;

        entity(const resource& resource, uct_iface_config_t *iface_config,
               uct_iface_params_t *params, uct_md_config_t *md_config);

        entity(const resource& resource, uct_md_config_t *md_config,
               uct_cm_config_t *cm_config);

        void mem_alloc_host(size_t length, uct_allocated_memory_t *mem) const;

        void mem_free_host(const uct_allocated_memory_t *mem) const;

        void mem_type_reg(uct_allocated_memory_t *mem) const;

        void mem_type_dereg(uct_allocated_memory_t *mem) const;

        void rkey_unpack(const uct_allocated_memory_t *mem,
                         uct_rkey_bundle *rkey_bundle) const;

        void rkey_release(const uct_rkey_bundle *rkey_bundle) const;

        unsigned progress() const;

        bool is_caps_supported(uint64_t required_flags);
        bool check_caps(uint64_t required_flags, uint64_t invalid_flags = 0);
        bool check_event_caps(uint64_t required_flags, uint64_t invalid_flags = 0);
        bool check_atomics(uint64_t required_ops, atomic_mode mode);

        uct_md_h md() const;

        const uct_md_attr& md_attr() const;

        uct_worker_h worker() const;

        uct_cm_h cm() const;

        const uct_cm_attr_t& cm_attr() const;

        uct_listener_h listener() const;

        uct_iface_h iface() const;

        const uct_iface_attr& iface_attr() const;

        const uct_iface_params& iface_params() const;

        uct_ep_h ep(unsigned index) const;

        eps_vec_t& eps();
        size_t num_eps() const;
        void reserve_ep(unsigned index);

        void create_ep(unsigned index);
        void destroy_ep(unsigned index);
        void revoke_ep(unsigned index);
        void destroy_eps();
        void connect(unsigned index, entity& other, unsigned other_index);
        void connect(unsigned index, entity& other, unsigned other_index,
                     const ucs::sock_addr_storage &remote_addr,
                     uct_cm_ep_priv_data_pack_callback_t pack_cb,
                     uct_cm_ep_client_connect_callback_t connect_cb,
                     uct_ep_disconnect_cb_t disconnect_cb,
                     void *user_data);
        void connect_to_iface(unsigned index, entity& other);
        void connect_to_ep(unsigned index, entity& other,
                           unsigned other_index);
        void connect_to_sockaddr(unsigned index, entity& other,
                                 const ucs::sock_addr_storage &remote_addr,
                                 uct_cm_ep_priv_data_pack_callback_t pack_cb,
                                 uct_cm_ep_client_connect_callback_t connect_cb,
                                 uct_ep_disconnect_cb_t disconnect_cb,
                                 void *user_sata);

        void listen(const ucs::sock_addr_storage &listen_addr,
                    const uct_listener_params_t &params);
        void disconnect(uct_ep_h ep);

        void flush() const;

        size_t                   max_conn_priv;

        class scoped_async_lock {
        public:
            scoped_async_lock(entity &e);
            ~scoped_async_lock();
        private:
            entity &m_entity;
        };

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

        entity(const entity&);


        void connect_p2p_ep(uct_ep_h from, uct_ep_h to);
        void cuda_mem_alloc(size_t length, uct_allocated_memory_t *mem) const;
        void cuda_mem_free(const uct_allocated_memory_t *mem) const;

        const resource              m_resource;
        ucs::handle<uct_md_h>       m_md;
        uct_md_attr_t               m_md_attr;
        mutable async_wrapper       m_async;
        ucs::handle<uct_worker_h>   m_worker;
        ucs::handle<uct_cm_h>       m_cm;
        uct_cm_attr_t               m_cm_attr;
        ucs::handle<uct_listener_h> m_listener;
        ucs::handle<uct_iface_h>    m_iface;
        eps_vec_t                   m_eps;
        uct_iface_attr_t            m_iface_attr;
        uct_iface_params_t          m_iface_params;
    };

    class mapped_buffer {
    public:
        mapped_buffer(size_t size, uint64_t seed, const entity& entity,
                      size_t offset = 0,
                      ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST);
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

    private:

        const uct_test::entity& m_entity;

        void                    *m_buf;
        void                    *m_end;
        uct_rkey_bundle_t       m_rkey;
        uct_allocated_memory_t  m_mem;
        uct_iov_t               m_iov;
    };

    class async_event_ctx {
    public:
        async_event_ctx() {
            wakeup_fd.fd      = -1;
            wakeup_fd.events  = POLLIN;
            wakeup_fd.revents = 0;
            aux_pipe_init     = false;
            memset(&aux_pipe, 0, sizeof(aux_pipe));
        }

        ~async_event_ctx() {
            if (aux_pipe_init) {
                ucs_async_pipe_destroy(&aux_pipe);
            }
        }

        void signal();
        bool wait_for_event(entity &e, double timeout_sec);

    private:
        struct pollfd    wakeup_fd;
        /* this used for UCT TLs that support async event cb
         * for event notification */
        ucs_async_pipe_t aux_pipe;
        bool             aux_pipe_init;
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

    void wait_for_bits(volatile uint64_t *flag, uint64_t mask,
                       double timeout = DEFAULT_TIMEOUT_SEC) const
    {
        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(timeout) *
                              ucs::test_time_multiplier();
        while ((ucs_get_time() < deadline) && (!ucs_test_all_flags(*flag, mask))) {
            /* Don't do short_progress_loop() to avoid extra timings */
            progress();
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
    virtual void modify_config(const std::string& name, const std::string& value,
                               modify_config_mode_t mode = FAIL_IF_NOT_EXIST);
    bool get_config(const std::string& name, std::string& value) const;
    void stats_activate();
    void stats_restore();

    virtual bool has_transport(const std::string& tl_name) const;
    virtual bool has_ud() const;
    virtual bool has_rc() const;
    virtual bool has_rc_or_dc() const;
    virtual bool has_ib() const;
    virtual bool has_mm() const;
    virtual bool has_cma() const;

    bool is_caps_supported(uint64_t required_flags);
    bool check_caps(uint64_t required_flags, uint64_t invalid_flags = 0);
    void check_caps_skip(uint64_t required_flags, uint64_t invalid_flags = 0);
    bool check_event_caps(uint64_t required_flags, uint64_t invalid_flags = 0);
    bool check_atomics(uint64_t required_ops, atomic_mode mode);
    const entity& ent(unsigned index) const;
    unsigned progress() const;
    void flush(ucs_time_t deadline = ULONG_MAX) const;
    virtual void short_progress_loop(double delay_ms = DEFAULT_DELAY_MS) const;
    virtual void twait(int delta_ms = DEFAULT_DELAY_MS) const;
    static void set_cm_resources(std::vector<resource>& all_resources);
    static bool is_interface_usable(struct ifaddrs *ifa, const char *name);
    static void set_md_sockaddr_resources(const md_resource& md_rsc, uct_md_h pm,
                                          ucs_cpu_set_t local_cpus,
                                          std::vector<resource>& all_resources);
    static void set_cm_sockaddr_resources(uct_component_h cmpt, const char *cmpt_name,
                                          ucs_cpu_set_t local_cpus,
                                          std::vector<resource>& all_resources);
    static void set_interface_rscs(uct_component_h cmpt, const char *cmpt_name,
                                   const char *md_name, ucs_cpu_set_t local_cpus,
                                   struct ifaddrs *ifa,
                                   std::vector<resource>& all_resources);
    static void init_sockaddr_rsc(resource *rsc, struct sockaddr *listen_addr,
                                  struct sockaddr *connect_addr, size_t size,
                                  bool init_src);
    uct_test::entity* create_entity(size_t rx_headroom,
                                    uct_error_handler_t err_handler = NULL,
                                    uct_tag_unexp_eager_cb_t eager_cb = NULL,
                                    uct_tag_unexp_rndv_cb_t rndv_cb = NULL,
                                    void *eager_arg = NULL,
                                    void *rndv_arg = NULL,
                                    uct_async_event_cb_t async_event_cb = NULL,
                                    void *async_event_arg = NULL);
    uct_test::entity* create_entity(uct_iface_params_t &params);
    uct_test::entity* create_entity();
    int max_connections();
    int max_connect_batch();

    void reduce_tl_send_queues();

    ucs_status_t send_am_message(entity *e, uint8_t am_id = 0, int ep_idx = 0);

    ucs::ptr_vector<entity> m_entities;
    uct_iface_config_t      *m_iface_config;
    uct_md_config_t         *m_md_config;
    uct_cm_config_t         *m_cm_config;
};

std::ostream& operator<<(std::ostream& os, const resource* resource);


class test_uct_iface_attrs : public uct_test {
public:
    typedef std::map<std::string, size_t> attr_map_t;

    void init();
    virtual attr_map_t get_num_iov() = 0;
    void basic_iov_test();

protected:
    entity *m_e;
};


#define UCT_TEST_IB_TLS \
    rc_mlx5,            \
    rc_verbs,           \
    dc_mlx5,            \
    ud_verbs,           \
    ud_mlx5,            \
    cm

#define UCT_TEST_SOCKADDR_TLS \
    sockaddr

#define UCT_TEST_NO_SELF_TLS \
    UCT_TEST_IB_TLS,         \
    ugni_rdma,               \
    ugni_udt,                \
    ugni_smsg,               \
    tcp,                     \
    posix,                   \
    sysv,                    \
    xpmem,                   \
    cma,                     \
    knem

#define UCT_TEST_CUDA_MEM_TYPE_TLS \
    cuda_copy,              \
    gdr_copy

#define UCT_TEST_ROCM_MEM_TYPE_TLS \
    rocm_copy

#define UCT_TEST_TLS      \
    UCT_TEST_NO_SELF_TLS, \
    UCT_TEST_CUDA_MEM_TYPE_TLS, \
    UCT_TEST_ROCM_MEM_TYPE_TLS, \
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

#define UCT_INSTANTIATE_SOCKADDR_TEST_CASE(_test_case) \
    UCS_PP_FOREACH(_UCT_INSTANTIATE_TEST_CASE, _test_case, UCT_TEST_SOCKADDR_TLS)

/**
 * Instantiate the parametrized test case for the RC/DC transports.
 *
 * @param _test_case  Test case class, derived from uct_test.
 */
#define UCT_INSTANTIATE_RC_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc_verbs) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc_mlx5)

#define UCT_INSTANTIATE_RC_DC_TEST_CASE(_test_case) \
    UCT_INSTANTIATE_RC_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, dc_mlx5)

std::ostream& operator<<(std::ostream& os, const uct_tl_resource_desc_t& resource);

#endif
