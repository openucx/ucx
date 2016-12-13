/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_helpers.h>
extern "C" {
#include <ucs/arch/atomic.h>
}


std::string ucp_test::m_last_err_msg;

std::ostream& operator<<(std::ostream& os, const ucp_test_param& test_param)
{
    std::vector<std::string>::const_iterator iter;
    const std::vector<std::string>& transports = test_param.transports;
    for (iter = transports.begin(); iter != transports.end(); ++iter) {
        if (iter != transports.begin()) {
            os << ",";
        }
        os << *iter;
    }
    return os;
}

ucp_test::ucp_test() {
    ucs_status_t status;
    status = ucp_config_read(NULL, NULL, &m_ucp_config);
    ASSERT_UCS_OK(status);
}

ucp_test::~ucp_test() {
    ucp_config_release(m_ucp_config);
}

void ucp_test::cleanup() {
    /* disconnect before destroying the entities */
    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         iter != entities().end(); ++iter)
    {
        disconnect(**iter);
    }

    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         iter != entities().end(); ++iter)
    {
        (*iter)->cleanup();
    }

    m_entities.clear();
}

void ucp_test::init() {
    test_base::init();

    const ucp_test_param &test_param = GetParam();

    create_entity();
    if ("\\self" != test_param.transports.front()) {
        create_entity();
    }
}

ucp_test_base::entity* ucp_test::create_entity(bool add_in_front) {
    entity *e = new entity(GetParam(), m_ucp_config);
    if (add_in_front) {
        m_entities.push_front(e);
    } else {
        m_entities.push_back(e);
    }
    return e;
}

ucp_params_t ucp_test::get_ctx_params() {
    ucp_params_t params;
    memset(&params, 0, sizeof(params));
    return params;
}

ucp_worker_params_t ucp_test::get_worker_params() {
    ucp_worker_params_t params;
    memset(&params, 0, sizeof(params));
    params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    params.thread_mode = UCS_THREAD_MODE_MULTI;
    return params;
}

void ucp_test::progress(int worker_index) const {
    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         iter != entities().end(); ++iter)
    {
        (*iter)->progress(worker_index);
    }
}

void ucp_test::short_progress_loop(int worker_index) const {
    for (unsigned i = 0; i < 100; ++i) {
        progress(worker_index);
        usleep(100);
    }
}

void ucp_test::wait_for_flag(volatile size_t *flag, double timeout)
{
    ucs_time_t loop_end_limit;

    loop_end_limit = ucs_get_time() + ucs_time_from_sec(timeout);

    while ((ucs_get_time() < loop_end_limit) && (!(*flag))) {
        short_progress_loop();
    }
}

void ucp_test::disconnect(const entity& entity) {
    for (int i = 0; i < entity.get_num_workers(); i++) {
        void *dreq = entity.disconnect_nb(i);
        if (!UCS_PTR_IS_PTR(dreq)) {
            ASSERT_UCS_OK(UCS_PTR_STATUS(dreq));
        }
        wait(dreq, i);
    }
}

void ucp_test::wait(void *req, int worker_index)
{
    if (req == NULL) {
        return;
    }

    ucs_status_t status;
    do {
        progress(worker_index);
        ucp_tag_recv_info info;
        status = ucp_request_test(req, &info);
    } while (status == UCS_INPROGRESS);
    ASSERT_UCS_OK(status);
    ucp_request_release(req);
}

std::vector<ucp_test_param>
ucp_test::enum_test_params(const ucp_params_t& ctx_params,
                           const ucp_worker_params_t& worker_params,
                           const std::string& name,
                           const std::string& test_case_name,
                           const std::string& tls)
{
    ucp_test_param test_param;
    std::stringstream ss(tls);

    test_param.ctx_params    = ctx_params;
    test_param.variant       = DEFAULT_PARAM_VARIANT;
    test_param.thread_type   = SINGLE_THREAD;
    test_param.worker_params = worker_params;
    
    while (ss.good()) {
        std::string tl_name;
        std::getline(ss, tl_name, ',');
        test_param.transports.push_back(tl_name);
    }

    if (check_test_param(name, test_case_name, test_param)) {
        return std::vector<ucp_test_param>(1, test_param);
    } else {
        return std::vector<ucp_test_param>();
    }
}

void ucp_test::generate_test_params_variant(const ucp_params_t& ctx_params,
                                            const ucp_worker_params_t& worker_params,
                                            const std::string& name,
                                            const std::string& test_case_name,
                                            const std::string& tls,
                                            int variant,
                                            std::vector<ucp_test_param>& test_params,
                                            int thread_type)
{
    std::vector<ucp_test_param> tmp_test_params, result;

    tmp_test_params = ucp_test::enum_test_params(ctx_params, worker_params, name,
                                                 test_case_name, tls);
    for (std::vector<ucp_test_param>::iterator iter = tmp_test_params.begin();
         iter != tmp_test_params.end(); ++iter)
    {
        iter->variant = variant;
        iter->thread_type = thread_type;
        test_params.push_back(*iter);
    }
}

void ucp_test::set_ucp_config(ucp_config_t *config,
                              const ucp_test_param& test_param)
{
    std::stringstream ss;
    ss << test_param;
    ucp_config_modify(config, "TLS", ss.str().c_str());
}

void ucp_test::modify_config(const std::string& name, const std::string& value)
{
    ucs_status_t status;

    status = ucp_config_modify(m_ucp_config, name.c_str(), value.c_str());
    if (status == UCS_ERR_NO_ELEM) {
        test_base::modify_config(name, value);
    } else if (status != UCS_OK) {
        UCS_TEST_ABORT("Couldn't modify ucp config parameter: " <<
                        name.c_str() << " to " << value.c_str() << ": " <<
                        ucs_status_string(status));
    }
}

bool ucp_test::check_test_param(const std::string& name,
                                const std::string& test_case_name,
                                const ucp_test_param& test_param)
{
    typedef std::map<std::string, bool> cache_t;
    static cache_t cache;

    if (test_param.transports.empty()) {
        return false;
    }

    cache_t::iterator iter = cache.find(name);
    if (iter != cache.end()) {
        return iter->second;
    }

    ucs::handle<ucp_config_t*> config;
    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);
    set_ucp_config(config, test_param);

    ucp_context_h ucph;
    ucs_status_t status;
    {
        disable_errors();
        status = ucp_init(&test_param.ctx_params, config, &ucph);
        restore_errors();
    }

    bool result;
    if (status == UCS_OK) {
        ucp_cleanup(ucph);
        result = true;
    } else if (status == UCS_ERR_NO_DEVICE) {
        result = false;
    } else {
        UCS_TEST_ABORT("Failed to create context (" << test_case_name << "): "
                       << ucs_status_string(status));
    }

    UCS_TEST_MESSAGE << "checking " << name << ": " << (result ? "yes" : "no");
    cache[name] = result;
    return result;
}

ucs_log_func_rc_t ucp_test::empty_log_handler(const char *file, unsigned line,
                                              const char *function, ucs_log_level_t level,
                                              const char *prefix, const char *message,
                                              va_list ap)
{
    if (level == UCS_LOG_LEVEL_ERROR) {
        std::string msg;
        msg.resize(256);
        vsnprintf(&msg[0], msg.size() - 1, message, ap);
        msg.resize(strlen(&msg[0]));
        m_last_err_msg = msg;
        level = UCS_LOG_LEVEL_DEBUG;
    }

    ucs_log_default_handler(file, line, function, level, prefix, message, ap);
    return UCS_LOG_FUNC_RC_STOP;
}

void ucp_test::disable_errors()
{
    ucs_log_push_handler((ucs_log_func_t)empty_log_handler);
}

void ucp_test::restore_errors()
{
    ucs_log_pop_handler();
}

ucp_test_base::entity::entity(const ucp_test_param& test_param, ucp_config_t* ucp_config) {
    ucp_test_param entity_param = test_param;

    num_workers = 1;
    entity_param.ctx_params.mt_workers_shared = 0;
    entity_param.worker_params.thread_mode = UCS_THREAD_MODE_MULTI;

    if (test_param.thread_type == MULTI_THREAD_CONTEXT) {
        num_workers = MT_TEST_NUM_THREADS;
        entity_param.ctx_params.mt_workers_shared = 1;
        entity_param.worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    } else if (test_param.thread_type == MULTI_THREAD_WORKER) {
        num_workers = 1;
        entity_param.ctx_params.mt_workers_shared = 0;
        entity_param.worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    }

    ucp_test::set_ucp_config(ucp_config, entity_param);

    UCS_TEST_CREATE_HANDLE(ucp_context_h, m_ucph, ucp_cleanup, ucp_init,
                           &entity_param.ctx_params, ucp_config);

    m_eps.resize(num_workers);
    m_workers.resize(num_workers);
    for (int i = 0; i < num_workers; i++) {
        UCS_TEST_CREATE_HANDLE(ucp_worker_h, m_workers.at(i), ucp_worker_destroy,
                               ucp_worker_create, m_ucph, &entity_param.worker_params);
    }
}

ucp_test_base::entity::~entity() {
    m_workers.clear();
    m_eps.clear();
}

void ucp_test_base::entity::connect(const entity* other) {
    assert(num_workers == other->get_num_workers());
    for (int i = 0; i < num_workers; i++) {
        ucs_status_t status;
        ucp_address_t *address;
        size_t address_length;
        ucp_ep_h ep;
        ucp_ep_params_t ep_params;

        status = ucp_worker_get_address(other->worker(i), &address, &address_length);
        ASSERT_UCS_OK(status);

        ucp_test::disable_errors();
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address    = address;

        status = ucp_ep_create(m_workers.at(i), &ep_params, &ep);
        ucp_test::restore_errors();

        if (status == UCS_ERR_UNREACHABLE) {
            ucp_worker_release_address(other->worker(i), address);
            UCS_TEST_SKIP_R(ucp_test::m_last_err_msg);
        }

        ASSERT_UCS_OK(status);

        m_eps.at(i).reset(ep, ucp_ep_destroy);

        ucp_worker_release_address(other->worker(i), address);
    }
}

void ucp_test_base::entity::flush_worker(int worker_index) const {
    ucs_status_t status = ucp_worker_flush(worker(worker_index));
    ASSERT_UCS_OK(status);
}

void ucp_test_base::entity::flush_ep(int ep_index) const {
    ucs_status_t status = ucp_ep_flush(ep(ep_index));
    ASSERT_UCS_OK(status);
}

void ucp_test_base::entity::fence(int worker_index) const {
    ucs_status_t status = ucp_worker_fence(worker(worker_index));
    ASSERT_UCS_OK(status);
}

void ucp_test_base::entity::disconnect(int ep_index) {
    m_eps.at(ep_index).reset();
}

void* ucp_test_base::entity::disconnect_nb(int ep_index) const {
    ucp_ep_h ep = revoke_ep(ep_index);
    if (ep == NULL) {
        return NULL;
    }
    return ucp_disconnect_nb(ep);
}

void ucp_test_base::entity::destroy_worker(int worker_index) {
    m_eps.at(worker_index).revoke();
    m_workers.at(worker_index).reset();
}

ucp_ep_h ucp_test_base::entity::ep(int ep_index) const {
    return m_eps.at(ep_index);
}

ucp_ep_h ucp_test_base::entity::revoke_ep(int ep_index) const {
    ucp_ep_h ep = m_eps.at(ep_index);
    m_eps.at(ep_index).revoke();
    return ep;
}

ucp_worker_h ucp_test_base::entity::worker(int worker_index) const {
    return m_workers.at(worker_index);
}

ucp_context_h ucp_test_base::entity::ucph() const {
    return m_ucph;
}

void ucp_test_base::entity::progress(int worker_index)
{
    ucp_worker_progress(m_workers.at(worker_index));
}

int ucp_test_base::entity::get_num_workers() const {
    return num_workers;
}

void ucp_test_base::entity::cleanup() {
    m_workers.clear();
    m_eps.clear();
}
