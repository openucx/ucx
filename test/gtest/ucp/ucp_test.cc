/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_helpers.h>
#include <ucs/arch/atomic.h>
#include <ucs/stats/stats.h>


namespace ucp {
const uint32_t MAGIC = 0xd7d7d7d7U;
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

const ucp_datatype_t ucp_test::DATATYPE     = ucp_dt_make_contig(1);
const ucp_datatype_t ucp_test::DATATYPE_IOV = ucp_dt_make_iov();

ucp_test::ucp_test() {
    ucs_status_t status;
    status = ucp_config_read(NULL, NULL, &m_ucp_config);
    ASSERT_UCS_OK(status);
}

ucp_test::~ucp_test() {

    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         iter != entities().end(); ++iter)
    {
        (*iter)->warn_existing_eps();
    }
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

    create_entity();
    if (!is_self()) {
        create_entity();
    }
}

bool ucp_test::is_self() const {
    return "self" == GetParam().transports.front();
}

ucp_test_base::entity* ucp_test::create_entity(bool add_in_front) {
    return create_entity(add_in_front, GetParam());
}

ucp_test_base::entity* ucp_test::create_entity(bool add_in_front,
                                               const ucp_test_param &test_param) {
    entity *e = new entity(test_param, m_ucp_config, get_worker_params());
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

ucp_ep_params_t ucp_test::get_ep_params() {
    ucp_ep_params_t params;
    memset(&params, 0, sizeof(params));
    return params;
}

unsigned ucp_test::progress(int worker_index) const {
    unsigned count = 0;
    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         iter != entities().end(); ++iter)
    {
        count += (*iter)->progress(worker_index);
        sched_yield();
    }
    return count;
}

void ucp_test::short_progress_loop(int worker_index) const {
    for (unsigned i = 0; i < 100; ++i) {
        progress(worker_index);
        usleep(100);
    }
}

void ucp_test::flush_ep(const entity &e, int worker_index, int ep_index)
{
    void *request = e.flush_ep_nb(worker_index, ep_index);
    wait(request, worker_index);
}

void ucp_test::flush_worker(const entity &e, int worker_index)
{
    void *request = e.flush_worker_nb(worker_index);
    wait(request, worker_index);
}

void* ucp_test::disconnect(const entity& entity) {
    for (int i = 0; i < entity.get_num_workers(); i++) {
        flush_worker(entity, i);
        for (int j = 0; j < entity.get_num_eps(i); j++) {
            void *dreq = entity.disconnect_nb(i, j);
            if (!UCS_PTR_IS_PTR(dreq)) {
                ASSERT_UCS_OK(UCS_PTR_STATUS(dreq));
            }
            wait(dreq, i);
        }
    }
    return NULL;
}

void ucp_test::wait(void *req, int worker_index)
{
    if (req == NULL) {
        return;
    }

    EXPECT_TRUE(UCS_PTR_IS_PTR(req)) << "error: "
                                     << ucs_status_string(UCS_PTR_STATUS(req));

    ucs_status_t status;
    do {
        progress(worker_index);
        status = ucp_request_check_status(req);
    } while (status == UCS_INPROGRESS);

    if (status != UCS_OK) {
        /* UCS errors are suppressed in case of error handling tests */
        ucs_error("request %p completed with error %s", req,
                  ucs_status_string(status));
    }

    ucp_request_release(req);
}

void ucp_test::set_ucp_config(ucp_config_t *config) {
    set_ucp_config(config, GetParam());
}

std::vector<ucp_test_param>
ucp_test::enum_test_params(const ucp_params_t& ctx_params,
                           const std::string& name,
                           const std::string& test_case_name,
                           const std::string& tls)
{
    ucp_test_param test_param;
    std::stringstream ss(tls);

    test_param.ctx_params    = ctx_params;
    test_param.variant       = DEFAULT_PARAM_VARIANT;
    test_param.thread_type   = SINGLE_THREAD;

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
                                            const std::string& name,
                                            const std::string& test_case_name,
                                            const std::string& tls,
                                            int variant,
                                            std::vector<ucp_test_param>& test_params,
                                            int thread_type)
{
    std::vector<ucp_test_param> tmp_test_params;

    tmp_test_params = ucp_test::enum_test_params(ctx_params,name,
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
    /* prevent configuration warnings in the UCP testing */
    ucp_config_modify(config, "WARN_INVALID_CONFIG", "no");
}

void ucp_test::modify_config(const std::string& name, const std::string& value,
                             bool optional)
{
    ucs_status_t status;

    status = ucp_config_modify(m_ucp_config, name.c_str(), value.c_str());
    if (status == UCS_ERR_NO_ELEM) {
        test_base::modify_config(name, value, optional);
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
        hide_errors();
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

ucp_test_base::entity::entity(const ucp_test_param& test_param,
                              ucp_config_t* ucp_config,
                              const ucp_worker_params_t& worker_params)
{
    ucp_test_param entity_param = test_param;
    ucp_worker_params_t local_worker_params = worker_params;

    if (test_param.thread_type == MULTI_THREAD_CONTEXT) {
        num_workers = MT_TEST_NUM_THREADS;
        entity_param.ctx_params.mt_workers_shared = 1;
        local_worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    } else if (test_param.thread_type == MULTI_THREAD_WORKER) {
        num_workers = 1;
        entity_param.ctx_params.mt_workers_shared = 0;
        local_worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    } else {
        num_workers = 1;
        entity_param.ctx_params.mt_workers_shared = 0;
        local_worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    }

    entity_param.ctx_params.field_mask |= UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    local_worker_params.field_mask     |= UCP_WORKER_PARAM_FIELD_THREAD_MODE;

    ucp_test::set_ucp_config(ucp_config, entity_param);

    UCS_TEST_CREATE_HANDLE(ucp_context_h, m_ucph, ucp_cleanup, ucp_init,
                           &entity_param.ctx_params, ucp_config);

    m_workers.resize(num_workers);
    for (int i = 0; i < num_workers; i++) {
        UCS_TEST_CREATE_HANDLE(ucp_worker_h, m_workers[i].first,
                               ucp_worker_destroy, ucp_worker_create, m_ucph,
                               &local_worker_params);
    }
}

ucp_test_base::entity::~entity() {
    cleanup();
}

void ucp_test_base::entity::connect(const entity* other,
                                    const ucp_ep_params_t& ep_params,
                                    int ep_idx, int do_set_ep) {
    assert(num_workers == other->get_num_workers());
    for (unsigned i = 0; i < unsigned(num_workers); i++) {
        ucs_status_t status;
        ucp_address_t *address;
        size_t address_length;
        ucp_ep_h ep;
        ucp_ep_params_t local_ep_params = ep_params;

        status = ucp_worker_get_address(other->worker(i), &address, &address_length);
        ASSERT_UCS_OK(status);

        ucp_test::hide_errors();
        local_ep_params.field_mask |= UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        local_ep_params.address     = address;

        status = ucp_ep_create(m_workers[i].first, &local_ep_params, &ep);
        ucp_test::restore_errors();

        if (status == UCS_ERR_UNREACHABLE) {
            ucp_worker_release_address(other->worker(i), address);
            UCS_TEST_SKIP_R(m_errors.empty() ? "" : m_errors.back());
        }

        ASSERT_UCS_OK(status);

        if (do_set_ep) {
            set_ep(ep, i, ep_idx);
        }

        ucp_worker_release_address(other->worker(i), address);
    }
}

void* ucp_test_base::entity::modify_ep(const ucp_ep_params_t& ep_params,
                                      int worker_idx, int ep_idx) {
    return ucp_ep_modify_nb(ep(worker_idx, ep_idx), &ep_params);
}


void ucp_test_base::entity::set_ep(ucp_ep_h ep, int worker_index, int ep_index)
{
    if (ep_index < get_num_eps(worker_index)) {
        m_workers[worker_index].second[ep_index].reset(ep, ep_destructor, this);
    } else {
        m_workers[worker_index].second.push_back(
                        ucs::handle<ucp_ep_h, entity *>(ep, ucp_ep_destroy));
    }
}

void ucp_test_base::entity::empty_send_completion(void *r, ucs_status_t status) {
}

void ucp_test_base::entity::accept_cb(ucp_ep_h ep, void *arg) {
    entity *self = reinterpret_cast<entity*>(arg);
    int worker_index = 0; /* TODO pass worker index in arg */
    self->set_ep(ep, worker_index, self->get_num_eps(worker_index));
}

void* ucp_test_base::entity::flush_ep_nb(int worker_index, int ep_index) const {
    return ucp_ep_flush_nb(ep(worker_index, ep_index), 0, empty_send_completion);
}

void* ucp_test_base::entity::flush_worker_nb(int worker_index) const {
    if (worker(worker_index) == NULL) {
        return NULL;
    }
    return ucp_worker_flush_nb(worker(worker_index), 0, empty_send_completion);
}

void ucp_test_base::entity::fence(int worker_index) const {
    ucs_status_t status = ucp_worker_fence(worker(worker_index));
    ASSERT_UCS_OK(status);
}

void* ucp_test_base::entity::disconnect_nb(int worker_index, int ep_index) const {
    ucp_ep_h ep = revoke_ep(worker_index, ep_index);
    if (ep == NULL) {
        return NULL;
    }
    return ucp_disconnect_nb(ep);
}

void ucp_test_base::entity::destroy_worker(int worker_index) {
    for (size_t i = 0; i < m_workers[worker_index].second.size(); ++i) {
        m_workers[worker_index].second[i].revoke();
    }
    m_workers[worker_index].first.reset();
}

ucp_ep_h ucp_test_base::entity::ep(int worker_index, int ep_index) const {
    if (size_t(worker_index) < m_workers.size()) {
        if (size_t(ep_index) < m_workers[worker_index].second.size()) {
            return m_workers[worker_index].second[ep_index];
        }
    }
    return NULL;
}

ucp_ep_h ucp_test_base::entity::revoke_ep(int worker_index, int ep_index) const {
    ucp_ep_h ucp_ep = ep(worker_index, ep_index);

    if (ucp_ep) {
        m_workers[worker_index].second[ep_index].revoke();
    }

    return ucp_ep;
}

ucs_status_t ucp_test_base::entity::listen(const struct sockaddr* saddr,
                                           socklen_t addrlen, int worker_index)
{
    ucp_listener_params_t params;
    ucp_listener_h listener;

    params.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
    params.sockaddr.addr      = saddr;
    params.sockaddr.addrlen   = addrlen;
    params.accept_handler.cb  = accept_cb;
    params.accept_handler.arg = reinterpret_cast<void*>(this);

    wrap_errors();
    ucs_status_t status = ucp_listener_create(worker(worker_index), &params, &listener);
    restore_errors();
    if (status == UCS_OK) {
        m_listener.reset(listener, ucp_listener_destroy);
    } else {
        /* throw error if status is not (UCS_OK or UCS_ERR_UNREACHABLE).
         * UCS_ERR_INVALID_PARAM may also return but then the test should fail */
        EXPECT_EQ(UCS_ERR_UNREACHABLE, status);
    }
    return status;
}

ucp_worker_h ucp_test_base::entity::worker(int worker_index) const {
    ucs_assert(size_t(worker_index) < m_workers.size());
    return m_workers[worker_index].first;
}

ucp_context_h ucp_test_base::entity::ucph() const {
    return m_ucph;
}

unsigned ucp_test_base::entity::progress(int worker_index)
{
    ucp_worker_h ucp_worker = worker(worker_index);
    return ucp_worker ? ucp_worker_progress(ucp_worker) : 0;
}

int ucp_test_base::entity::get_num_workers() const {
    assert(m_workers.size() == size_t(num_workers));
    return num_workers;
}

int ucp_test_base::entity::get_num_eps(int worker_index) const {
    return m_workers[worker_index].second.size();
}

void ucp_test_base::entity::warn_existing_eps() const {
    for (size_t worker_index = 0; worker_index < m_workers.size(); ++worker_index) {
        for (size_t ep_index = 0; ep_index < m_workers[worker_index].second.size();
             ++ep_index) {
            ADD_FAILURE() << "ep(" << worker_index << "," << ep_index <<
                             ")=" << m_workers[worker_index].second[ep_index].get() <<
                             " was not destroyed during test cleanup()";
        }
    }
}

void ucp_test_base::entity::cleanup() {
    m_listener.reset();
    m_workers.clear();
}

void ucp_test_base::entity::ep_destructor(ucp_ep_h ep, entity *e)
{
    ucs_status_ptr_t req = ucp_disconnect_nb(ep);
    if (!UCS_PTR_IS_PTR(req)) {
        return;
    }

    ucs_status_t        status;
    ucp_tag_recv_info_t info;
    do {
        e->progress();
        status = ucp_request_test(req, &info);
    } while (status == UCS_INPROGRESS);
    EXPECT_EQ(UCS_OK, status);
    ucp_request_release(req);
}
