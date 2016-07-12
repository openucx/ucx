/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCP_TEST_H_
#define UCP_TEST_H_

extern "C" {
#include <ucp/api/ucp.h>
}
#include <common/test.h>


struct ucp_test_param {
    ucp_params_t              ctx_params;
    std::vector<std::string>  transports;
};


/**
 * UCP test
 */
class ucp_test : public ucs::test_base, public ::testing::TestWithParam<ucp_test_param> {

public:
    UCS_TEST_BASE_IMPL;

    ucp_test();
    virtual ~ucp_test();

    ucp_config_t* m_ucp_config;

    class entity {
    public:
        entity(const ucp_test_param& test_param, ucp_config_t* ucp_config);

        void connect(const entity* other);

        void flush_ep() const;

        void flush_worker() const;

        void fence() const;

        void disconnect();

        void destroy_worker();

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

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     ...);

    virtual void modify_config(const std::string& name, const std::string& value);

protected:
    virtual void cleanup();
    entity* create_entity();
    static ucp_params_t get_ctx_params();
    void progress() const;
    void short_progress_loop() const;
    static void disable_errors();
    static void restore_errors();

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
    ucs::ptr_vector<entity> m_entities;
};


std::ostream& operator<<(std::ostream& os, const ucp_test_param& test_param);


/**
 * Instantiate the parametrized test case a combination of transports.
 *
 * @param _test_case   Test case class, derived from uct_test.
 * @param _name        Instantiation name.
 * @param ...          Transport names.
 */
#define UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, _name, ...) \
    INSTANTIATE_TEST_CASE_P(_name,  _test_case, \
                            testing::ValuesIn(ucp_test::enum_test_params(_test_case::get_ctx_params(), \
                                                                         #_name, \
                                                                         #_test_case, \
                                                                         __VA_ARGS__, \
                                                                         NULL)));


/**
 * Instantiate the parametrized test case for all transport combinations.
 *
 * @param _test_case  Test case class, derived from uct_test.
 */
#define UCP_INSTANTIATE_TEST_CASE(_test_case)                                                    \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, dc,    "\\dc")                                     \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ud,    "\\ud")                                     \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, udx,   "\\ud_mlx5")                                \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, udrc,  "\\ud", "\\rc")                             \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, cmrcx, "\\cm", "\\rc_mlx5")                        \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, shm,   "\\mm", "\\knem", "\\cma", "\\xpmem", "ib") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, udrcx, "\\ud_mlx5", "\\rc_mlx5")                   \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, ugni,  "\\ugni_smsg", "\\ugni_udt", "\\ugni_rdma") \
    UCP_INSTANTIATE_TEST_CASE_TLS(_test_case, self,  "\\self")


#endif
