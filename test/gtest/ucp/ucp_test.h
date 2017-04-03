/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCP_TEST_H_
#define UCP_TEST_H_

extern "C" {
#include <ucp/api/ucp.h>
#include <ucs/time/time.h>
}
#include <common/test.h>

#if _OPENMP
#include <omp.h>
#endif


struct ucp_test_param {
    ucp_params_t              ctx_params;
    ucp_worker_params_t       worker_params;
    std::vector<std::string>  transports;
    int                       variant;
    int                       thread_type;
};

class ucp_test_base : public ucs::test_base {
public:
    enum {
        SINGLE_THREAD = 42,
        MULTI_THREAD_CONTEXT,
        MULTI_THREAD_WORKER
    };

    class entity {
    public:
        static const int default_worker = -1;
        static const int default_ep     = -1;

        entity(const ucp_test_param& test_param, int num_workers,
               ucp_config_t* ucp_config);

        ~entity();

        void connect(const entity* other);

        void flush_ep(int ep_index = default_ep) const;

        void flush_all_eps() const;

        void flush_worker(int worker_index = default_worker) const;

        void fence(int worker_index = default_worker) const;

        void disconnect(int ep_index = default_ep);

        void* disconnect_nb(int ep_index = default_ep) const;

        void destroy_worker(int worker_index = default_worker);

        ucp_ep_h ep(int ep_index = default_ep) const;

        ucp_ep_h revoke_ep(int ep_index = default_ep) const;

        ucp_worker_h worker(int worker_index = default_worker) const;

        int get_worker_index() const;

        ucp_context_h ucph() const;

        void progress(int worker_index = default_worker);

        void create_rkeys(void *rkey_buffer, std::vector<ucp_rkey_h> *rkeys);

        void destroy_rkeys(std::vector<ucp_rkey_h> *rkeys);

        void cleanup();

    private:
        ucs::handle<ucp_context_h>               m_ucph;
        std::vector<ucs::handle<ucp_worker_h> >  m_workers;
        std::vector<ucs::handle<ucp_ep_h> >      m_eps;
        int                                      m_thread_type;
   };
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
        DEFAULT_PARAM_VARIANT = 0,
        MT_NUM_THREADS        = 2
    };

    UCS_TEST_BASE_IMPL;

    ucp_test();
    virtual ~ucp_test();

    ucp_config_t* m_ucp_config;

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const ucp_worker_params_t& worker_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     const std::string& tls);

    static ucp_params_t get_ctx_params();
    static ucp_worker_params_t get_worker_params();

    static void
    generate_test_params_variant(const ucp_params_t& ctx_params,
                                 const ucp_worker_params_t& worker_params,
                                 const std::string& name,
                                 const std::string& test_case_name,
                                 const std::string& tls,
                                 int variant,
                                 std::vector<ucp_test_param>& test_params,
                                 bool multi_thread = false);

    virtual void modify_config(const std::string& name, const std::string& value);
    void stats_activate();
    void stats_restore();

protected:
    virtual void init();
    virtual void cleanup();
    entity* create_entity(bool add_in_front = false);

    void progress(int worker_index = entity::default_worker) const;
    void short_progress_loop(int worker_index = entity::default_worker) const;
    void disconnect(const entity& entity);
    void wait(void *req, int worker_index = entity::default_worker);
    static void disable_errors();
    static void restore_errors();
    int mt_num_workers() const;
    int mt_num_threads() const;

    template <typename T>
    void wait_for_flag(volatile T *flag, double timeout = 10.0,
                       int worker_index = entity::default_worker)
    {
        ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(timeout);
        while ((ucs_get_time() < loop_end_limit) && (!(*flag))) {
            short_progress_loop(worker_index);
        }
    }

private:
    static void set_ucp_config(ucp_config_t *config,
                               const ucp_test_param& test_param);
    static bool check_test_param(const std::string& name,
                                 const std::string& test_case_name,
                                 const ucp_test_param& test_param);
    static ucs_log_func_rc_t empty_log_handler(const char *file, unsigned line,
                                               const char *function, ucs_log_level_t level,
                                               const char *prefix, const char *message,
                                               va_list ap);

    static std::string m_last_err_msg;

    int m_mt_num_workers;
    int m_mt_num_threads;
};

#if _OPENMP && ENABLE_MT

#  define UCS_OMP_PARALLEL_FOR(_var) \
    _Pragma("omp parallel for") \
    for (int _var = 0; _var < mt_num_threads(); ++_var)
#  define UCS_GET_THREAD_ID \
    omp_get_thread_num()

#else

#  define UCS_OMP_PARALLEL_FOR(_var) \
    int _var = 0;
#  define UCS_GET_THREAD_ID \
    0

#endif


std::ostream& operator<<(std::ostream& os, const ucp_test_param& test_param);


/**
 * Instantiate the parameterized test case a combination of transports.
 *
 * @param _test_case   Test case class, derived from uct_test.
 * @param _name        Instantiation name.
 * @param ...          Transport names.
 */
#define UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, _name, _tls) \
    INSTANTIATE_TEST_CASE_P(_name,  _test_case, \
                            testing::ValuesIn(_test_case::enum_test_params(_test_case::get_ctx_params(), \
                                                                           _test_case::get_worker_params(), \
                                                                           #_name, \
                                                                           #_test_case, \
                                                                           _tls)));


/**
 * Instantiate the parameterized test case for all transport combinations.
 *
 * @param _test_case  Test case class, derived from uct_test.
 */
#define UCP_INSTANTIATE_TEST_CASE(_test_case)                                                    \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dc,    "\\dc")                                     \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dcx,   "\\dc_mlx5")                                \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ud,    "\\ud")                                     \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, udx,   "\\ud_mlx5")                                \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, udrc,  "\\ud,\\rc")                             \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, cmrcx, "\\cm,\\rc_mlx5")                        \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, shm,   "\\mm,\\knem,\\cma,\\xpmem,ib") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, udrcx, "\\ud_mlx5,\\rc_mlx5")                   \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ugni,  "\\ugni_smsg,\\ugni_udt,\\ugni_rdma") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, self,  "\\self") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, tcp,  "\\tcp")


#endif
