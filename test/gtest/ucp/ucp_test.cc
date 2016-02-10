/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_helpers.h>
extern "C" {
#include <ucs/arch/atomic.h>
}


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

const ucs::ptr_vector<ucp_test::entity>& ucp_test::entities() const {
    return m_entities;
}

void ucp_test::cleanup() {
    /* disconnect before destroying the entities */
    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         iter != entities().end(); ++iter)
    {
        (*iter)->disconnect();
    }
    m_entities.clear();
}

ucp_test::entity* ucp_test::create_entity() {
    entity *e = new entity(GetParam());
    m_entities.push_back(e);
    return e;
}

ucp_params_t ucp_test::get_ctx_params() {
    ucp_params_t params = { 0, 0, NULL, NULL };
    return params;
}

void ucp_test::progress() const {
    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         iter != entities().end(); ++iter)
    {
        (*iter)->progress();
    }
}

void ucp_test::short_progress_loop() const {
    for (unsigned i = 0; i < 100; ++i) {
        progress();
        usleep(100);
    }
}

std::vector<ucp_test_param>
ucp_test::enum_test_params(const ucp_params_t& ctx_params,
                           const std::string& name,
                           const std::string& test_case_name,
                           ...)
{
    ucp_test_param test_param;
    const char *tl_name;
    va_list ap;

    test_param.ctx_params = ctx_params;

    va_start(ap, test_case_name);
    tl_name = va_arg(ap, const char *);
    while (tl_name != NULL) {
        test_param.transports.push_back(tl_name);
        tl_name = va_arg(ap, const char *);
    }
    va_end(ap);

    if (check_test_param(name, test_case_name, test_param)) {
        return std::vector<ucp_test_param>(1, test_param);
    } else {
        return std::vector<ucp_test_param>();
    }
}

void ucp_test::set_ucp_config(ucp_config_t *config,
                              const ucp_test_param& test_param)
{
    std::stringstream ss;
    ss << test_param;
    ucp_config_modify(config, "TLS", ss.str().c_str());
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

ucs_log_func_rc_t ucp_test::empty_log_handler(...)
{
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

ucp_test::entity::entity(const ucp_test_param& test_param) {
    ucs::handle<ucp_config_t*> config;

    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);
    set_ucp_config(config, test_param);

    UCS_TEST_CREATE_HANDLE(ucp_context_h, m_ucph, ucp_cleanup, ucp_init,
                           &test_param.ctx_params, config);

    UCS_TEST_CREATE_HANDLE(ucp_worker_h, m_worker, ucp_worker_destroy,
                           ucp_worker_create, m_ucph, UCS_THREAD_MODE_MULTI);
}

void ucp_test::entity::connect(const ucp_test::entity* other) {
    ucs_status_t status;
    ucp_address_t *address;
    size_t address_length;

    status = ucp_worker_get_address(other->worker(), &address, &address_length);
    ASSERT_UCS_OK(status);

    ucp_ep_h ep;
    status = ucp_ep_create(m_worker, address, &ep);
    if (status == UCS_ERR_UNREACHABLE) {
        ucp_worker_release_address(other->worker(), address);
        UCS_TEST_SKIP_R("could not find a valid transport");
    }

    ASSERT_UCS_OK(status);
    m_ep.reset(ep, ucp_ep_destroy);

    ucp_worker_release_address(other->worker(), address);
}

void ucp_test::entity::flush() const {
    ucs_status_t status = ucp_worker_flush(worker());
    ASSERT_UCS_OK(status);
}

void ucp_test::entity::disconnect() {
    m_ep.reset();
}

ucp_ep_h ucp_test::entity::ep() const {
    return m_ep;
}

ucp_worker_h ucp_test::entity::worker() const {
    return m_worker;
}

ucp_context_h ucp_test::entity::ucph() const {
    return m_ucph;
}

void ucp_test::entity::progress()
{
    ucp_worker_progress(m_worker);
}

