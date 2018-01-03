/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCP_TEST_H_
#define UCP_TEST_H_

#include <ucp/api/ucp.h>
#include <ucs/time/time.h>

/* ucp version compile time test */
#if (UCP_API_VERSION != UCP_VERSION(UCP_API_MAJOR,UCP_API_MINOR))
#error possible bug in UCP version
#endif

#include <common/test.h>

#define MT_TEST_NUM_THREADS       4
#define UCP_TEST_TIMEOUT_IN_SEC   10.0

struct ucp_test_param {
    ucp_params_t              ctx_params;
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
        typedef std::vector<ucs::handle<ucp_ep_h, entity *> > ep_vec_t;
        typedef std::vector<std::pair<ucs::handle<ucp_worker_h>,
                                      ep_vec_t> > worker_vec_t;

    public:
        entity(const ucp_test_param& test_param, ucp_config_t* ucp_config,
               const ucp_worker_params_t& worker_params);

        ~entity();

        void connect(const entity* other, const ucp_ep_params_t& ep_params,
                     int ep_idx = 0);

        void* modify_ep(const ucp_ep_params_t& ep_params, int worker_idx = 0,
                       int ep_idx = 0);

        void* flush_ep_nb(int worker_index = 0, int ep_index = 0) const;

        void* flush_worker_nb(int worker_index = 0) const;

        void fence(int worker_index = 0) const;

        void* disconnect_nb(int worker_index = 0, int ep_index = 0) const;

        void destroy_worker(int worker_index = 0);

        ucs_status_t listen(const struct sockaddr *saddr, socklen_t addrlen,
                            int worker_index = 0);

        ucp_ep_h ep(int worker_index = 0, int ep_index = 0) const;

        ucp_ep_h revoke_ep(int worker_index = 0, int ep_index = 0) const;

        ucp_worker_h worker(int worker_index = 0) const;

        ucp_context_h ucph() const;

        unsigned progress(int worker_index = 0);

        int get_num_workers() const;

        int get_num_eps(int worker_index = 0) const;

        void warn_existing_eps() const;

        void cleanup();

        static void ep_destructor(ucp_ep_h ep, entity *e);

    protected:
        ucs::handle<ucp_context_h>  m_ucph;
        worker_vec_t                m_workers;
        ucs::handle<ucp_listener_h> m_listener;

        int num_workers;

    private:
        static void empty_send_completion(void *r, ucs_status_t status);
        static void accept_cb(ucp_ep_h ep, void *arg);

        void set_ep(ucp_ep_h ep, int worker_index, int ep_index);
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
        DEFAULT_PARAM_VARIANT = 0
    };

    UCS_TEST_BASE_IMPL;

    ucp_test();
    virtual ~ucp_test();

    ucp_config_t* m_ucp_config;

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     const std::string& tls);

    static ucp_params_t get_ctx_params();
    virtual ucp_worker_params_t get_worker_params();
    virtual ucp_ep_params_t get_ep_params();

    static void
    generate_test_params_variant(const ucp_params_t& ctx_params,
                                 const std::string& name,
                                 const std::string& test_case_name,
                                 const std::string& tls,
                                 int variant,
                                 std::vector<ucp_test_param>& test_params,
                                 int thread_type = SINGLE_THREAD);

    virtual void modify_config(const std::string& name, const std::string& value,
                               bool optional = false);
    void stats_activate();
    void stats_restore();

protected:
    virtual void init();
    bool is_self() const;
    virtual void cleanup();
    entity* create_entity(bool add_in_front = false);
    entity* create_entity(bool add_in_front, const ucp_test_param& test_param);
    unsigned progress(int worker_index = 0) const;
    void short_progress_loop(int worker_index = 0) const;
    void flush_ep(const entity &e, int worker_index = 0, int ep_index = 0);
    void flush_worker(const entity &e, int worker_index = 0);
    void disconnect(const entity& entity);
    void wait(void *req, int worker_index = 0);
    void set_ucp_config(ucp_config_t *config);

    template <typename T>
    void wait_for_flag(volatile T *flag, double timeout = 10.0) {
        ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(timeout);
        while ((ucs_get_time() < loop_end_limit) && (!(*flag))) {
            short_progress_loop();
        }
    }

private:
    static void set_ucp_config(ucp_config_t *config,
                               const ucp_test_param& test_param);
    static bool check_test_param(const std::string& name,
                                 const std::string& test_case_name,
                                 const ucp_test_param& test_param);

protected:
    static const ucp_datatype_t DATATYPE;
    static const ucp_datatype_t DATATYPE_IOV;
};


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
                                                                           #_name, \
                                                                           #_test_case, \
                                                                           _tls)));


/**
 * Instantiate the parameterized test case for all transport combinations.
 *
 * @param _test_case  Test case class, derived from uct_test.
 */
#define UCP_INSTANTIATE_TEST_CASE(_test_case) \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dc,     "dc") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dcx,    "dc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ud,     "ud") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, udx,    "ud_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, rc,     "rc") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, rcx,    "rc_x") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, rcx_cm, "\\rc_mlx5,cm:aux") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, shm_ib, "shm,ib") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ugni,   "ugni") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, self,   "self") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, tcp,    "tcp")


#endif
