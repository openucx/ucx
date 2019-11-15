/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include <common/test_helpers.h>

extern "C" {
#include <ucp/core/ucp_worker.h>
#if HAVE_IB
#include <uct/ib/ud/base/ud_iface.h>
#endif
#include <ucs/arch/atomic.h>
#include <ucs/stats/stats.h>
}

#include <queue>

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

ucp_test::ucp_test() : m_err_handler_count(0) {
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

static bool check_transport(const std::string check_tl_name,
                            const std::vector<std::string>& tl_names) {
    return (std::find(tl_names.begin(), tl_names.end(),
                      check_tl_name) != tl_names.end());
}

bool ucp_test::has_transport(const std::string& tl_name) const {
    return check_transport(tl_name, GetParam().transports);
}

bool ucp_test::has_only_transports(const std::vector<std::string>& tl_names) const {
    const std::vector<std::string>& transports = GetParam().transports;
    size_t other_tls_count                     = 0;
    std::vector<std::string>::const_iterator iter;

    for(iter = transports.begin(); iter != transports.end(); ++iter) {
        if (!check_transport(*iter, tl_names)) {
            other_tls_count++;
        }
    }

    return !other_tls_count;
}

bool ucp_test::is_self() const {
    return "self" == GetParam().transports.front();
}

ucp_test_base::entity* ucp_test::create_entity(bool add_in_front) {
    return create_entity(add_in_front, GetParam());
}

ucp_test_base::entity*
ucp_test::create_entity(bool add_in_front, const ucp_test_param &test_param) {
    entity *e = new entity(test_param, m_ucp_config, get_worker_params(), this);
    if (add_in_front) {
        m_entities.push_front(e);
    } else {
        m_entities.push_back(e);
    }
    return e;
}

ucp_test::entity* ucp_test::get_entity_by_ep(ucp_ep_h ep) {
    ucs::ptr_vector<entity>::const_iterator e_it;
    for (e_it = entities().begin(); e_it != entities().end(); ++e_it) {
        for (int w_idx = 0; w_idx < (*e_it)->get_num_workers(); ++w_idx) {
            for (int ep_idx = 0; ep_idx < (*e_it)->get_num_eps(w_idx); ++ep_idx) {
                if (ep == (*e_it)->ep(w_idx, ep_idx)) {
                    return *e_it;
                }
            }
        }
    }
    return NULL;
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

void ucp_test::disconnect(const entity& entity) {
    for (int i = 0; i < entity.get_num_workers(); i++) {
        if (m_err_handler_count == 0) {
            flush_worker(entity, i);
        }

        for (int j = 0; j < entity.get_num_eps(i); j++) {
            void *dreq = entity.disconnect_nb(i, j, m_err_handler_count == 0 ?
                                                    UCP_EP_CLOSE_MODE_FLUSH :
                                                    UCP_EP_CLOSE_MODE_FORCE);
            if (!UCS_PTR_IS_PTR(dreq)) {
                ASSERT_UCS_OK(UCS_PTR_STATUS(dreq));
            }
            wait(dreq, i);
        }
    }
}

void ucp_test::wait(void *req, int worker_index)
{
    if (req == NULL) {
        return;
    }

    if (UCS_PTR_IS_ERR(req)) {
        ucs_error("operation returned error: %s",
                  ucs_status_string(UCS_PTR_STATUS(req)));
        return;
    }

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

int ucp_test::max_connections() {
    if (has_transport("tcp")) {
        return ucs::max_tcp_connections();
    } else {
        return std::numeric_limits<int>::max();
    }
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
        scoped_log_handler slh(hide_errors_logger);
        status = ucp_init(&test_param.ctx_params, config, &ucph);
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
                              const ucp_worker_params_t& worker_params,
                              const ucp_test_base *test_owner)
    : m_test_owner(test_owner), m_rejected_cntr(0)
{
    ucp_test_param entity_param = test_param;
    ucp_worker_params_t local_worker_params = worker_params;
    int num_workers;

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

    {
        scoped_log_handler slh(hide_errors_logger);
        UCS_TEST_CREATE_HANDLE(ucp_context_h, m_ucph, ucp_cleanup, ucp_init,
                               &entity_param.ctx_params, ucp_config);
    }

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
    assert(get_num_workers() == other->get_num_workers());
    for (unsigned i = 0; i < unsigned(get_num_workers()); i++) {
        ucs_status_t status;
        ucp_address_t *address;
        size_t address_length;
        ucp_ep_h ep;

        status = ucp_worker_get_address(other->worker(i), &address, &address_length);
        ASSERT_UCS_OK(status);

        {
            scoped_log_handler slh(hide_errors_logger);

            ucp_ep_params_t local_ep_params = ep_params;
            local_ep_params.field_mask |= UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
            local_ep_params.address     = address;

            status = ucp_ep_create(m_workers[i].first, &local_ep_params, &ep);
        }

        if (status == UCS_ERR_UNREACHABLE) {
            ucp_worker_release_address(other->worker(i), address);
            UCS_TEST_SKIP_R(m_errors.empty() ? "Unreachable" : m_errors.back());
        }

        ASSERT_UCS_OK(status, << " (" << m_errors.back() << ")");

        if (do_set_ep) {
            set_ep(ep, i, ep_idx);
        }

        ucp_worker_release_address(other->worker(i), address);
    }
}

ucp_ep_h ucp_test_base::entity::accept(ucp_worker_h worker,
                                       ucp_conn_request_h conn_request,
                                       const void *ep_user_data)
{
    ucp_ep_h        ep;
    ucp_ep_params_t ep_params;
    ep_params.field_mask   = UCP_EP_PARAM_FIELD_USER_DATA |
                             UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.user_data    = (void *)ep_user_data;
    ep_params.conn_request = conn_request;

    ucs_status_t status    = ucp_ep_create(worker, &ep_params, &ep);
    if (status == UCS_ERR_UNREACHABLE) {
        UCS_TEST_SKIP_R("Skipping due an unreachable destination (unsupported "
                        "feature or no supported transport to send partial "
                        "worker address)");
    }
    ASSERT_UCS_OK(status);
    return ep;
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

void ucp_test_base::entity::accept_ep_cb(ucp_ep_h ep, void *arg) {
    entity *self = reinterpret_cast<entity*>(arg);
    int worker_index = 0; /* TODO pass worker index in arg */
    self->set_ep(ep, worker_index, self->get_num_eps(worker_index));
}

void ucp_test_base::entity::accept_conn_cb(ucp_conn_request_h conn_req, void* arg)
{
    entity *self = reinterpret_cast<entity*>(arg);
    self->m_conn_reqs.push(conn_req);
}

void ucp_test_base::entity::reject_conn_cb(ucp_conn_request_h conn_req, void* arg)
{
    entity *self = reinterpret_cast<entity*>(arg);
    ucp_listener_reject(self->m_listener, conn_req);
    self->m_rejected_cntr++;
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

void* ucp_test_base::entity::disconnect_nb(int worker_index, int ep_index,
                                           enum ucp_ep_close_mode mode) const {
    ucp_ep_h ep = revoke_ep(worker_index, ep_index);
    if (ep == NULL) {
        return NULL;
    }
    return ucp_ep_close_nb(ep, mode);
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

ucs_status_t ucp_test_base::entity::listen(listen_cb_type_t cb_type,
                                           const struct sockaddr* saddr,
                                           socklen_t addrlen, int worker_index)
{
    ucp_listener_params_t params;
    ucp_listener_h        listener;

    params.field_mask             = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR;
    params.sockaddr.addr          = saddr;
    params.sockaddr.addrlen       = addrlen;

    switch (cb_type) {
    case LISTEN_CB_EP:
        params.field_mask        |= UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
        params.accept_handler.cb  = accept_ep_cb;
        params.accept_handler.arg = reinterpret_cast<void*>(this);
        break;
    case LISTEN_CB_CONN:
        params.field_mask        |= UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
        params.conn_handler.cb    = accept_conn_cb;
        params.conn_handler.arg   = reinterpret_cast<void*>(this);
        break;
    case LISTEN_CB_REJECT:
        params.field_mask        |= UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
        params.conn_handler.cb    = reject_conn_cb;
        params.conn_handler.arg   = reinterpret_cast<void*>(this);
        break;
    default:
        UCS_TEST_ABORT("invalid test parameter");
    }

    ucs_status_t status;
    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        status = ucp_listener_create(worker(worker_index), &params, &listener);
    }

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
    if (worker_index < get_num_workers()) {
        return m_workers[worker_index].first;
    } else {
        return NULL;
    }
}

ucp_context_h ucp_test_base::entity::ucph() const {
    return m_ucph;
}

ucp_listener_h ucp_test_base::entity::listenerh() const {
    return m_listener;
}

unsigned ucp_test_base::entity::progress(int worker_index)
{
    ucp_worker_h ucp_worker = worker(worker_index);

    if (ucp_worker == NULL) {
        return 0;
    }

    unsigned progress_count = 0;
    if (!m_conn_reqs.empty()) {
        ucp_conn_request_h conn_req = m_conn_reqs.back();
        m_conn_reqs.pop();
        ucp_ep_h ep = accept(ucp_worker, conn_req, m_test_owner);
        set_ep(ep, worker_index, std::numeric_limits<int>::max());
        ++progress_count;
    }

    return progress_count + ucp_worker_progress(ucp_worker);
}

int ucp_test_base::entity::get_num_workers() const {
    return m_workers.size();
}

int ucp_test_base::entity::get_num_eps(int worker_index) const {
    return m_workers[worker_index].second.size();
}

size_t ucp_test_base::entity::get_rejected_cntr() const {
    return m_rejected_cntr;
}

void ucp_test_base::entity::inc_rejected_cntr() {
    ++m_rejected_cntr;
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

double ucp_test_base::entity::set_ib_ud_timeout(double timeout_sec)
{
    double prev_timeout_sec = 0.;
#if HAVE_IB
    for (ucp_rsc_index_t rsc_index = 0;
         rsc_index < ucph()->num_tls; ++rsc_index) {
        ucp_worker_iface_t *wiface = ucp_worker_iface(worker(), rsc_index);
        // check if the iface is ud transport
        if (wiface->iface->ops.iface_flush == uct_ud_iface_flush) {
            uct_ud_iface_t *iface =
                ucs_derived_of(wiface->iface, uct_ud_iface_t);

            uct_ud_enter(iface);
            if (!prev_timeout_sec) {
                prev_timeout_sec = ucs_time_to_sec(iface->config.peer_timeout);
            }

            iface->config.peer_timeout = ucs_time_from_sec(timeout_sec);
            uct_ud_leave(iface);
        }
    }
#endif
    return prev_timeout_sec;
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

ucp_test::mapped_buffer::mapped_buffer(size_t size, const entity& entity,
                                       int flags, ucs_memory_type_t mem_type) :
    mem_buffer(size, mem_type), m_entity(entity), m_memh(NULL),
    m_rkey_buffer(NULL)
{
    ucs_status_t status;

    if (flags & (UCP_MEM_MAP_ALLOCATE|UCP_MEM_MAP_FIXED)) {
        UCS_TEST_ABORT("mapped_buffer does not support allocation by UCP");
    }

    ucp_mem_map_params_t params;
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.flags      = flags;
    params.address    = ptr();
    params.length     = size;

    status = ucp_mem_map(m_entity.ucph(), &params, &m_memh);
    ASSERT_UCS_OK(status);

    size_t rkey_buffer_size;
    status = ucp_rkey_pack(m_entity.ucph(), m_memh, &m_rkey_buffer,
                           &rkey_buffer_size);
    ASSERT_UCS_OK(status);
}

ucp_test::mapped_buffer::~mapped_buffer()
{
    ucp_rkey_buffer_release(m_rkey_buffer);
    ucs_status_t status = ucp_mem_unmap(m_entity.ucph(), m_memh);
    EXPECT_UCS_OK(status);
}

ucs::handle<ucp_rkey_h> ucp_test::mapped_buffer::rkey(const entity& entity) const
{
    ucp_rkey_h rkey;

    ucs_status_t status = ucp_ep_rkey_unpack(entity.ep(), m_rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);
    return ucs::handle<ucp_rkey_h>(rkey, ucp_rkey_destroy);
}

ucp_mem_h ucp_test::mapped_buffer::memh() const
{
    return m_memh;
}
