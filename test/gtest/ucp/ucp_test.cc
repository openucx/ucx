/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_helpers.h>
extern "C" {
#include <ucs/arch/atomic.h>
#include <ucs/stats/stats.h>
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

    if (GetParam().thread_type == MULTI_THREAD_CONTEXT) {
        m_mt_num_threads = m_mt_num_workers = MT_NUM_THREADS;
    } else if (GetParam().thread_type == MULTI_THREAD_WORKER) {
        m_mt_num_workers = 1;
        m_mt_num_threads = MT_NUM_THREADS;
    } else {
        m_mt_num_threads = m_mt_num_workers = 1;
    }
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
    entity *e = new entity(GetParam(), mt_num_workers(), m_ucp_config);
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
    params.field_mask |= UCP_PARAM_FIELD_FEATURES;
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

void ucp_test::disconnect(const entity& entity) {
    for (int i = 0; i < mt_num_workers(); i++) {
        entity.flush_worker(i);
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
                                            bool multi_thread)
{
    std::vector<ucp_test_param> tmp_test_params, result;

    tmp_test_params = ucp_test::enum_test_params(ctx_params, worker_params, name,
                                                 test_case_name, tls);
    for (std::vector<ucp_test_param>::iterator iter = tmp_test_params.begin();
         iter != tmp_test_params.end(); ++iter)
    {
        iter->variant = variant;
        if (multi_thread) {
            for (int thread_type = SINGLE_THREAD;
                 thread_type <= MULTI_THREAD_WORKER; thread_type++) {
                iter->thread_type = thread_type;
                test_params.push_back(*iter);
            }
        } else {
            iter->thread_type = SINGLE_THREAD;
            test_params.push_back(*iter);
        }
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

void ucp_test::stats_activate()
{
    ucs_stats_cleanup();
    push_config();
    modify_config("STATS_DEST",    "file:/dev/null");
    modify_config("STATS_TRIGGER", "exit");
    ucs_stats_init();
    ASSERT_TRUE(ucs_stats_is_active());
}

void ucp_test::stats_restore()
{
    ucs_stats_cleanup();
    pop_config();
    ucs_stats_init();
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
        va_list ap2;
        std::string msg;
        msg.resize(256);
        va_copy(ap2, ap);
        vsnprintf(&msg[0], msg.size() - 1, message, ap2);
        va_end(ap2);
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

int ucp_test::mt_num_workers() const {
    return m_mt_num_workers;
}

int ucp_test::mt_num_threads() const {
    return m_mt_num_threads;
}

ucp_test_base::entity::entity(const ucp_test_param& test_param,
                              int num_workers, ucp_config_t* ucp_config) :
    m_thread_type(test_param.thread_type)
{
    ucp_test_param entity_param = test_param;

    if (m_thread_type == MULTI_THREAD_CONTEXT) {
        entity_param.ctx_params.mt_workers_shared = 1;
        entity_param.worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    } else if (m_thread_type == MULTI_THREAD_WORKER) {
        entity_param.ctx_params.mt_workers_shared = 0;
        entity_param.worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    } else {
        entity_param.ctx_params.mt_workers_shared = 0;
        entity_param.worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    }

    entity_param.ctx_params.field_mask    |= UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    entity_param.worker_params.field_mask |= UCP_WORKER_PARAM_FIELD_THREAD_MODE;

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
    for (size_t i = 0; i < m_workers.size(); i++) {
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

        status = ucp_ep_create(worker(i), &ep_params, &ep);
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
    if (worker(worker_index) == NULL) {
        return;
    }
    ucs_status_t status = ucp_worker_flush(worker(worker_index));
    ASSERT_UCS_OK(status);
}

void ucp_test_base::entity::flush_ep(int ep_index) const {
    ucs_status_t status = ucp_ep_flush(ep(ep_index));
    ASSERT_UCS_OK(status);
}

void ucp_test_base::entity::flush_all_eps() const {
    for (std::vector<ucs::handle<ucp_ep_h> >::const_iterator iter = m_eps.begin();
         iter != m_eps.end(); ++iter)
    {
        ucs_status_t status = ucp_ep_flush(*iter);
        ASSERT_UCS_OK(status);
    }
}

void ucp_test_base::entity::fence(int worker_index) const {
    ucs_status_t status = ucp_worker_fence(worker(worker_index));
    ASSERT_UCS_OK(status);
}

void ucp_test_base::entity::disconnect(int ep_index) {
    if (ep_index == default_ep) {
        ep_index = get_worker_index();
    }
    flush_ep(ep_index);
    m_eps.at(ep_index).reset();
}

void* ucp_test_base::entity::disconnect_nb(int ep_index) const {
    ucp_ep_h ep = revoke_ep((ep_index == default_ep) ? get_worker_index() : ep_index);
    if (ep == NULL) {
        return NULL;
    }
    return ucp_disconnect_nb(ep);
}

void ucp_test_base::entity::destroy_worker(int worker_index) {
    if (worker_index == default_worker) {
        worker_index = get_worker_index();
    }
    m_eps.at(worker_index).revoke();
    m_workers.at(worker_index).reset();
}

ucp_ep_h ucp_test_base::entity::ep(int ep_index) const {
    return m_eps.at((ep_index == default_ep) ? get_worker_index() : ep_index);
}

ucp_ep_h ucp_test_base::entity::revoke_ep(int ep_index) const {
    if (ep_index == default_ep) {
        ep_index = get_worker_index();
    }
    ucp_ep_h ep = m_eps.at(ep_index);
    m_eps.at(ep_index).revoke();
    return ep;
}

ucp_worker_h ucp_test_base::entity::worker(int worker_index) const {
    return m_workers.at((worker_index == default_worker) ? get_worker_index() :
                        worker_index);
}

int ucp_test_base::entity::get_worker_index() const {
    return (m_thread_type == MULTI_THREAD_CONTEXT) ? UCS_GET_THREAD_ID : 0;
}

ucp_context_h ucp_test_base::entity::ucph() const {
    return m_ucph;
}

void ucp_test_base::entity::progress(int worker_index)
{
    ucp_worker_progress(worker(worker_index));
}

void ucp_test_base::entity::create_rkeys(void *rkey_buffer, std::vector<ucp_rkey_h> *rkeys) {
    ucs_status_t status;
    for (size_t i = 0; i < m_eps.size(); i++) {
        ucp_rkey_h rkey;
        status = ucp_ep_rkey_unpack(ep(i), rkey_buffer, &rkey);
        ASSERT_UCS_OK(status);
        (*rkeys).push_back(rkey);
    }
}

void ucp_test_base::entity::destroy_rkeys(std::vector<ucp_rkey_h> *rkeys){
    for (size_t i = 0; i < m_eps.size(); i++) {
        ucp_rkey_destroy((*rkeys)[i]);
    }
}

void ucp_test_base::entity::cleanup() {
    m_workers.clear();
    m_eps.clear();
}
