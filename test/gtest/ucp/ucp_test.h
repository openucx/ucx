/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCP_TEST_H_
#define UCP_TEST_H_

#define __STDC_LIMIT_MACROS
#include <ucp/api/ucp.h>
#include <ucs/time/time.h>
#include <common/mem_buffer.h>

/* ucp version compile time test */
#if (UCP_API_VERSION != UCP_VERSION(UCP_API_MAJOR,UCP_API_MINOR))
#error possible bug in UCP version
#endif

#include <common/test.h>

#include <queue>

#if _OPENMP
#include "omp.h"
#endif


namespace ucp {
extern const uint32_t MAGIC;
}


struct ucp_test_variant_value {
    int                                 value;  /* User-defined value */
    std::string                         name;   /* Variant description */
};


/* Specifies extended test parameter */
struct ucp_test_variant {
    ucp_params_t                        ctx_params;  /* UCP context parameters */
    int                                 thread_type; /* Thread mode */
    std::vector<ucp_test_variant_value> values;      /* Extended test parameters */
};


/* UCP test parameter which includes the transports to test and option to
 * define extended parameters by adding values to 'variant'
 */
struct ucp_test_param {
    std::vector<std::string>            transports;  /* Transports to test */
    ucp_test_variant                    variant;     /* Test variant */
};


class ucp_test; // forward declaration

class ucp_test_base : public ucs::test_base {
public:
    enum {
        SINGLE_THREAD = 42,
        MULTI_THREAD_CONTEXT, /* workers are single-threaded, context is mt-shared */
        MULTI_THREAD_WORKER   /* workers are multi-threaded, context is mt-single */
    };

    class entity {
        typedef std::vector<ucs::handle<ucp_ep_h, entity*> > ep_vec_t;
        typedef std::vector<std::pair<ucs::handle<ucp_worker_h>,
                                      ep_vec_t> > worker_vec_t;
        typedef std::deque<void*> close_ep_reqs_t;

    public:
        typedef enum {
            LISTEN_CB_EP,       /* User's callback accepts ucp_ep_h */
            LISTEN_CB_CONN,     /* User's callback accepts ucp_conn_request_h */
            LISTEN_CB_REJECT,   /* User's callback rejects ucp_conn_request_h */
            LISTEN_CB_CUSTOM    /* User's callback with custom test logic */
        } listen_cb_type_t;

        entity(const ucp_test_param& test_param, ucp_config_t* ucp_config,
               const ucp_worker_params_t& worker_params,
               const ucp_test_base* test_owner);

        ~entity();

        void connect(const entity* other, const ucp_ep_params_t& ep_params,
                     int ep_idx = 0, int do_set_ep = 1);

        bool verify_client_address(struct sockaddr_storage *client_address);

        void accept(int worker_index, ucp_conn_request_h conn_request);

        void* modify_ep(const ucp_ep_params_t& ep_params, int worker_idx = 0,
                       int ep_idx = 0);

        void* flush_ep_nb(int worker_index = 0, int ep_index = 0) const;

        void* flush_worker_nb(int worker_index = 0) const;

        void fence(int worker_index = 0) const;

        void *disconnect_nb(int worker_index = 0, int ep_index = 0,
                            uint32_t close_flags = 0);

        void close_ep_req_free(void *close_req);

        void close_all_eps(const ucp_test &test, int worker_idx,
                           uint32_t close_flags = 0);

        void destroy_worker(int worker_index = 0);

        ucs_status_t listen(listen_cb_type_t cb_type,
                            const struct sockaddr *saddr, socklen_t addrlen,
                            const ucp_ep_params_t& ep_params,
                            ucp_listener_conn_handler_t *custom_cb = NULL,
                            int worker_index = 0);

        ucp_ep_h ep(int worker_index = 0, int ep_index = 0) const;

        ucp_ep_h revoke_ep(int worker_index = 0, int ep_index = 0) const;

        ucp_worker_h worker(int worker_index = 0) const;

        ucp_context_h ucph() const;

        ucp_listener_h listenerh() const;

        unsigned progress(int worker_index = 0);

        ucp_mem_h mem_map(void *address, size_t length);

        void mem_unmap(ucp_mem_h memh);

        int get_num_workers() const;

        int get_num_eps(int worker_index = 0) const;

        void add_err(ucs_status_t status);

        const size_t &get_err_num_rejected() const;

        const size_t &get_err_num() const;

        const size_t &get_accept_err_num() const;

        void warn_existing_eps() const;

        double set_ib_ud_peer_timeout(double timeout_sec);

        void cleanup();

        static void ep_destructor(ucp_ep_h ep, entity *e);

        bool has_lane_with_caps(uint64_t caps) const;

        bool is_conn_reqs_queue_empty() const;

    protected:
        ucs::handle<ucp_context_h>      m_ucph;
        worker_vec_t                    m_workers;
        ucs::handle<ucp_listener_h>     m_listener;
        std::queue<ucp_conn_request_h>  m_conn_reqs;
        close_ep_reqs_t                 m_close_ep_reqs;
        size_t                          m_err_cntr;
        size_t                          m_rejected_cntr;
        size_t                          m_accept_err_cntr;
        ucs::handle<ucp_ep_params_t*>   m_server_ep_params;
        const ucp_test_base             *m_test;

    private:
        static void empty_send_completion(void *r, ucs_status_t status);
        static void accept_ep_cb(ucp_ep_h ep, void *arg);
        static void accept_conn_cb(ucp_conn_request_h conn_req, void *arg);
        static void reject_conn_cb(ucp_conn_request_h conn_req, void *arg);

        void set_ep(ucp_ep_h ep, int worker_index, int ep_index);

        static ucs_log_func_rc_t
        hide_config_warns_logger(const char *file, unsigned line,
                                 const char *function, ucs_log_level_t level,
                                 const ucs_log_component_config_t *comp_conf,
                                 const char *message, va_list ap);
    };

    static bool is_request_completed(void *req);

    static void *ep_close_nbx(ucp_ep_h ep, uint32_t flags);
};

/**
 * UCP test
 */
class ucp_test : public ucp_test_base,
                 public ::testing::TestWithParam<ucp_test_param>,
                 public ucs::entities_storage<ucp_test_base::entity> {

    friend class ucp_test_base::entity;

public:
    enum {
        DEFAULT_PARAM_VARIANT = 0
    };

    UCS_TEST_BASE_IMPL;

    ucp_test();
    virtual ~ucp_test();

    ucp_config_t* m_ucp_config;

    static std::vector<ucp_test_param>
    enum_test_params(const std::vector<ucp_test_variant>& variants,
                     const std::string& tls);

    virtual ucp_worker_params_t get_worker_params();
    virtual ucp_ep_params_t get_ep_params();

    virtual void modify_config(const std::string& name, const std::string& value,
                               modify_config_mode_t mode = FAIL_IF_NOT_EXIST);

    void disable_keepalive();

private:
    static void set_ucp_config(ucp_config_t *config, const std::string& tls);
    static bool check_tls(const std::string& tls);
    ucs_status_t request_process(void *req, int worker_index, bool wait,
                                 bool wakeup = false,
                                 const std::vector<entity*> &entities = {});

protected:
    using variant_vec_t       = std::vector<ucp_test_variant>;
    using get_variants_func_t = void (*)(variant_vec_t&);

    virtual void init();
    bool is_self() const;
    virtual void cleanup();
    virtual bool has_transport(const std::string& tl_name) const;
    bool has_any_transport(const std::vector<std::string>& tl_names) const;
    bool has_any_transport(const std::string *tls, size_t tl_size) const;
    entity* create_entity(bool add_in_front = false);
    entity* create_entity(bool add_in_front, const ucp_test_param& test_param);
    unsigned progress(const std::vector<entity*> &entities,
                      int worker_index = 0) const;
    unsigned progress(int worker_index = 0) const;
    void short_progress_loop(int worker_index = 0) const;
    void flush_ep(const entity &e, int worker_index = 0, int ep_index = 0);
    void flush_worker(const entity &e, int worker_index = 0,
                      const std::vector<entity*> &entities = {});
    void flush_workers();
    void disconnect(entity& entity);
    void check_events(const std::vector<entity*> &entities, bool wakeup,
                      int worker_index = 0);
    ucs_status_t
    request_progress(void *req, const std::vector<entity*> &entities,
                     double timeout = 10.0, int worker_index = 0);
    ucs_status_t request_wait(void *req,
                              const std::vector<entity*> &entities = {},
                              int worker_index = 0, bool wakeup = false);
    ucs_status_t requests_wait(std::vector<void*> &reqs, int worker_index = 0);
    ucs_status_t requests_wait(const std::initializer_list<void*> reqs_list,
                               int worker_index = 0);
    ucp_tag_message_h message_wait(entity& e, ucp_tag_t tag, ucp_tag_t tag_mask,
                                   ucp_tag_recv_info_t *info, int remove = 1,
                                   int worker_index = 0);
    void request_release(void *req);
    void request_cancel(entity &e, void *req);
    int wait_for_wakeup(const std::vector<entity*> &entities,
                        int poll_timeout = -1, int worker_index = 0);
    int max_connections();
    void configure_peer_failure_settings();

    static bool check_reg_mem_types(const entity& e, ucs_memory_type_t mem_type);

    // Add test variant without values, with given context params
    static ucp_test_variant&
    add_variant(std::vector<ucp_test_variant>& variants,
                const ucp_params_t& ctx_params, int thread_type = SINGLE_THREAD);

    // Add test variant without values, with given context features
    static ucp_test_variant&
    add_variant(std::vector<ucp_test_variant>& variants, uint64_t ctx_features,
                int thread_type = SINGLE_THREAD);

    // Add value to test variant
    static void
    add_variant_value(std::vector<ucp_test_variant_value>& values,
                      int value, const std::string& name);

    // Add test variant with context params and single value
    static void
    add_variant_with_value(std::vector<ucp_test_variant>& variants,
                           const ucp_params_t& ctx_params, int value,
                           const std::string& name,
                           int thread_type = SINGLE_THREAD);

    // Add test variant with context features and single value
    static void
    add_variant_with_value(std::vector<ucp_test_variant>& variants,
                           uint64_t ctx_features, int value,
                           const std::string& name,
                           int thread_type = SINGLE_THREAD);

    // Add test variants based on existing generator and additional single value
    static void
    add_variant_values(std::vector<ucp_test_variant>& variants,
                       get_variants_func_t generator, int value,
                       const std::string& name = "");

    // Add test variants based on existing generator and a bit-set of values
    static void
    add_variant_values(std::vector<ucp_test_variant>& variants,
                       get_variants_func_t generator, uint64_t bitmap,
                       const char **names);

    // Add test variants based on existing generator and enumerating all
    // supported memory types which are part of 'mem_types_mask'
    static void
    add_variant_memtypes(std::vector<ucp_test_variant>& variants,
                         get_variants_func_t generator,
                         uint64_t mem_types_mask = UINT64_MAX);

    // Return variant value at a given position
    int get_variant_value(unsigned index = 0) const;

    // Get thread mode for the test
    int get_variant_thread_type() const;

    // Return context parameters of the current test variant
    const ucp_params_t& get_variant_ctx_params() const;

    // Return maximal UCP threads in current environment, assuming each thread
    // can create 2 workers
    static unsigned mt_num_threads();

    static void err_handler_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        entity *e = reinterpret_cast<entity*>(arg);
        e->add_err(status);
    }

    template <typename T>
    void wait_for_flag(volatile T *flag, double timeout = 10.0) {
        ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(timeout);
        while ((ucs_get_time() < loop_end_limit) && (!(*flag))) {
            short_progress_loop();
        }
    }

    template <typename T>
    void wait_for_value(volatile T *var, T value, double timeout = 10.0) const
    {
        ucs_time_t deadline = ucs_get_time() +
                              ucs_time_from_sec(timeout) * ucs::test_time_multiplier();
        while ((ucs_get_time() < deadline) && (*var != value)) {
            short_progress_loop();
        }
    }

    static const ucp_datatype_t DATATYPE;
    static const ucp_datatype_t DATATYPE_IOV;

protected:
    class mapped_buffer : public mem_buffer {
    public:
        mapped_buffer(size_t size, const entity& entity, int flags = 0,
                      ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST);
        virtual ~mapped_buffer();

        ucs::handle<ucp_rkey_h> rkey(const entity& entity) const;

        ucp_mem_h memh() const;

    private:
        const entity& m_entity;
        ucp_mem_h     m_memh;
        void*         m_rkey_buffer;
    };

    ucs::ptr_vector<ucs::scoped_setenv> m_env;
};


class test_ucp_context : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants);
};


std::ostream& operator<<(std::ostream& os,
                         const std::vector<std::string>& str_vector);
std::ostream& operator<<(std::ostream& os, const ucp_test_param& test_param);

template <class T>
std::vector<ucp_test_param> enum_test_params(const std::string& tls)
{
    std::vector<ucp_test_variant> v;

    T::get_test_variants(v);
    return T::enum_test_params(v, tls);
}

/**
 * Instantiate the parameterized test case a combination of transports.
 *
 * @param _test_case   Test case class, derived from ucp_test.
 * @param _name        Instantiation name.
 * @param _tls         Transport names.
 */
#define UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, _name, _tls) \
    INSTANTIATE_TEST_SUITE_P(_name,  _test_case, \
                            testing::ValuesIn(enum_test_params<_test_case>(_tls)));


/**
 * Instantiate the parameterized test case a combination of transports with GPU
 * awareness.
 *
 * @param _test_case   Test case class, derived from ucp_test.
 * @param _name        Instantiation name.
 * @param _tls         Transport names.
 */
#define UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, _name, _tls) \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, _name, \
                                  _tls "," UCP_TEST_GPU_COPY_TLS)


/**
 * Instantiate the parameterized test case for all transport combinations.
 *
 * @param _test_case  Test case class, derived from ucp_test.
 */
#define UCP_INSTANTIATE_TEST_CASE(_test_case) \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dcx,    "dc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ud,     "ud_v") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, udx,    "ud_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, rc,     "rc_v") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, rcx,    "rc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, shm_ib, "shm,ib") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ugni,   "ugni") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, self,   "self") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, tcp,    "tcp")


/**
 * The list of GPU copy TLs
 */
#define UCP_TEST_GPU_COPY_TLS "cuda_copy,rocm_copy"


/**
 * Instantiate the parameterized test case for all transport combinations
 * with GPU memory awareness
 *
 * @param _test_case  Test case class, derived from ucp_test.
 */
#define UCP_INSTANTIATE_TEST_CASE_GPU_AWARE(_test_case) \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, dcx, \
                                            "dc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, ud, \
                                            "ud_v") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, udx, \
                                            "ud_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, rc, \
                                            "rc_v") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, rcx, \
                                            "rc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, shm_ib, \
                                            "shm,ib") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, shm_ib_ipc, \
                                            "shm,ib,cuda_ipc,rocm_ipc") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, ugni, \
                                            "ugni") \
    UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(_test_case, tcp, \
                                            "tcp")

#endif
