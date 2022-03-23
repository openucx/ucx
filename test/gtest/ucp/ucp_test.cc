/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucp_test.h"
#include <common/test_helpers.h>
#include <ifaddrs.h>
#include <sys/poll.h>

extern "C" {
#include <ucp/core/ucp_worker.inl>
#include <ucp/core/ucp_ep.inl>
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

std::ostream& operator<<(std::ostream& os,
                         const std::vector<std::string>& str_vector)
{
    for (std::vector<std::string>::const_iterator iter = str_vector.begin();
         iter != str_vector.end(); ++iter) {
        if (iter != str_vector.begin()) {
            os << ",";
        }
        os << *iter;
    }

    return os;
}

std::ostream& operator<<(std::ostream& os, const ucp_test_param& test_param)
{
    os << test_param.transports;

    for (size_t i = 0; i < test_param.variant.values.size(); ++i) {
        const std::string& name = test_param.variant.values[i].name;
        if (!name.empty()) {
            os << "/" << name;
        }
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
        try {
            // suppress Coverity warning about throwing an exception from the
            // function which is marked as noexcept
            (*iter)->warn_existing_eps();
        } catch (const std::exception &e) {
            UCS_TEST_MESSAGE << "got \"" << e.what() << "\" exception when"
                    << " checking existing endpoints";
        }
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

bool ucp_test::has_any_transport(const std::vector<std::string>& tl_names) const {
    const std::vector<std::string>& all_tl_names = GetParam().transports;

    return std::find_first_of(all_tl_names.begin(), all_tl_names.end(),
                              tl_names.begin(),     tl_names.end()) !=
           all_tl_names.end();
}

bool ucp_test::has_any_transport(const std::string *tls, size_t tl_size) const {
    const std::vector<std::string> tl_names(tls, tls + tl_size);
    return has_any_transport(tl_names);
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

unsigned ucp_test::progress(const std::vector<entity*> &entities,
                            int worker_index) const
{
    unsigned count = 0;
    for (auto e : entities) {
        count += e->progress(worker_index);
        sched_yield();
    }

    return count;
}

unsigned ucp_test::progress(int worker_index) const {
    return progress(entities(), worker_index);
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
    request_wait(request, worker_index);
}

void ucp_test::flush_worker(const entity &e, int worker_index)
{
    void *request = e.flush_worker_nb(worker_index);
    request_wait(request, worker_index);
}

void ucp_test::flush_workers()
{
    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         iter != entities().end(); ++iter) {
        const entity &e = **iter;
        for (int i = 0; i < e.get_num_workers(); i++) {
            flush_worker(e, i);
        }
    }
}

void ucp_test::disconnect(entity& e) {
    bool has_failed_entity = false;
    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         !has_failed_entity && (iter != entities().end()); ++iter) {
        has_failed_entity = ((*iter)->get_err_num() > 0);
    }

    for (int i = 0; i < e.get_num_workers(); i++) {
        enum ucp_ep_close_mode close_mode;

        if (has_failed_entity) {
            close_mode = UCP_EP_CLOSE_MODE_FORCE;
        } else {
            flush_worker(e, i);
            close_mode = UCP_EP_CLOSE_MODE_FLUSH;
        }

        e.close_all_eps(*this, i, close_mode);
    }
}

ucp_tag_message_h ucp_test::message_wait(entity& e, ucp_tag_t tag,
                                         ucp_tag_t tag_mask,
                                         ucp_tag_recv_info_t *info, int remove,
                                         int worker_index)
{
    ucs_time_t deadline = ucs::get_deadline();
    ucp_tag_message_h message;
    do {
        progress(worker_index);
        message = ucp_tag_probe_nb(e.worker(worker_index), tag, tag_mask,
                                   remove, info);
    } while ((message == NULL) && (ucs_get_time() < deadline));

    return message;
}

void ucp_test::check_events(const std::vector<entity*> &entities, bool wakeup,
                            int worker_index)
{
    if (progress(entities, worker_index)) {
        return;
    }

    if (wakeup) {
        int ret = wait_for_wakeup(entities, -1, worker_index);
        EXPECT_GE(ret, 1);
    }
}

ucs_status_t
ucp_test::request_progress(void *req, const std::vector<entity*> &entities,
                           double timeout, int worker_index)
{
    ucs_time_t loop_end_limit = ucs_get_time() + ucs_time_from_sec(timeout);
    ucs_status_t status;
    do {
        progress(entities, worker_index);
        status = ucp_request_check_status(req);
    } while ((status == UCS_INPROGRESS) &&
             (ucs_get_time() < loop_end_limit));

    return status;
}

ucs_status_t ucp_test::request_process(void *req, int worker_index, bool wait,
                                       bool wakeup)
{
    if (req == NULL) {
        return UCS_OK;
    }

    if (UCS_PTR_IS_ERR(req)) {
        ucs_error("operation returned error: %s",
                  ucs_status_string(UCS_PTR_STATUS(req)));
        return UCS_PTR_STATUS(req);
    }

    ucs_time_t deadline = ucs::get_deadline();
    while (wait &&
           (ucp_request_check_status(req) == UCS_INPROGRESS) &&
           (ucs_get_time() < deadline)) {
        check_events(entities(), wakeup, worker_index);
    }

    ucs_status_t status = ucp_request_check_status(req);
    if (status == UCS_INPROGRESS) {
        if (wait) {
            ucs_error("request %p did not complete on time", req);
        }
    } else if (status != UCS_OK) {
        /* UCS errors are suppressed in case of error handling tests */
        ucs_error("request %p completed with error %s", req,
                  ucs_status_string(status));
    }

    ucp_request_release(req);
    return status;
}

ucs_status_t ucp_test::request_wait(void *req, int worker_index, bool wakeup)
{
    return request_process(req, worker_index, true, wakeup);
}

ucs_status_t ucp_test::requests_wait(std::vector<void*> &reqs,
                                     int worker_index)
{
    ucs_status_t ret_status = UCS_OK;

    while (!reqs.empty()) {
        ucs_status_t status = request_wait(reqs.back(), worker_index);
        if (ret_status == UCS_OK) {
            // Save the first failure
            ret_status = status;
        }
        reqs.pop_back();
    }

    return ret_status;
}

ucs_status_t
ucp_test::requests_wait(const std::initializer_list<void*> reqs_list,
                        int worker_index)
{
    std::vector<void*> reqs(reqs_list);
    return requests_wait(reqs, worker_index);
}

void ucp_test::request_release(void *req)
{
    request_process(req, 0, false);
}

void ucp_test::request_cancel(entity &e, void *req)
{
    if (UCS_PTR_IS_PTR(req)) {
        ucp_request_cancel(e.worker(), req);
        ucp_request_free(req);
    }
}

int ucp_test::wait_for_wakeup(const std::vector<entity*> &entities,
                              int poll_timeout, int worker_index)
{
    int total_ret = 0, ret;
    std::vector<int> efds;

    for (auto e : entities) {
        ucp_worker_h worker = e->worker(worker_index);
        int efd;

        ASSERT_UCS_OK(ucp_worker_get_efd(worker, &efd));
        efds.push_back(efd);

        ucs_status_t status = ucp_worker_arm(worker);
        if (status == UCS_ERR_BUSY) {
            ++total_ret;
        } else {
            ASSERT_UCS_OK(status);
        }
    }

    if (total_ret > 0) {
        return total_ret;
    }

    std::vector<struct pollfd> pfd;
    for (int fd : efds) {
        pfd.push_back({ fd, POLLIN });
    }

    do {
        ret = poll(&pfd[0], efds.size(), poll_timeout);
        if (ret > 0) {
            total_ret += ret;
        }
    } while ((ret < 0) && (errno == EINTR));

    if (ret < 0) {
        UCS_TEST_MESSAGE << "poll() failed: " << strerror(errno);
    }

    return total_ret;
}

int ucp_test::max_connections() {
    if (has_transport("tcp")) {
        return ucs::max_tcp_connections();
    } else {
        return std::numeric_limits<int>::max();
    }
}

void ucp_test::configure_peer_failure_settings()
{
    /* Set small TL timeouts to reduce testing time */
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_TIMEOUT",     "10ms"));
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_RNR_TIMEOUT", "10ms"));
    m_env.push_back(new ucs::scoped_setenv("UCX_RC_RETRY_COUNT", "2"));
}

void ucp_test::set_ucp_config(ucp_config_t *config, const std::string& tls)
{
    ucs_status_t status;

    status = ucp_config_modify(config, "TLS", tls.c_str());
    if (status != UCS_OK) {
        UCS_TEST_ABORT("Failed to set UCX transports");
    }
    status = ucp_config_modify(config, "WARN_INVALID_CONFIG", "n");
    if (status != UCS_OK) {
        UCS_TEST_ABORT("Failed to set UCX to ignore invalid configuration");
    }
}

ucp_test_variant& ucp_test::add_variant(std::vector<ucp_test_variant>& variants,
                                        const ucp_params_t& ctx_params,
                                        int thread_type) {
    variants.push_back(ucp_test_variant());
    ucp_test_variant& variant = variants.back();
    variant.ctx_params        = ctx_params;
    variant.thread_type       = thread_type;
    return variant;
}

ucp_test_variant& ucp_test::add_variant(std::vector<ucp_test_variant>& variants,
                                        uint64_t ctx_features, int thread_type)
{
    ucp_params_t ctx_params = {};
    ctx_params.field_mask   = UCP_PARAM_FIELD_FEATURES;
    ctx_params.features     = ctx_features;
    return add_variant(variants, ctx_params, thread_type);
}

int ucp_test::get_variant_value(unsigned index) const {
    if (GetParam().variant.values.size() <= index) {
        return DEFAULT_PARAM_VARIANT;
    }
    return GetParam().variant.values.at(index).value;
}

const ucp_params_t& ucp_test::get_variant_ctx_params() const {
    return GetParam().variant.ctx_params;
}

int ucp_test::get_variant_thread_type() const {
    return GetParam().variant.thread_type;
}

void ucp_test::add_variant_value(std::vector<ucp_test_variant_value>& values,
                                 int value, const std::string& name)
{
    ucp_test_variant_value entry = {value, name};
    values.push_back(entry);
}

void ucp_test::add_variant_with_value(std::vector<ucp_test_variant>& variants,
                                      uint64_t ctx_features, int value,
                                      const std::string& name, int thread_type)
{
    add_variant_value(add_variant(variants, ctx_features, thread_type).values,
                      value, name);
}

void ucp_test::add_variant_with_value(std::vector<ucp_test_variant>& variants,
                                      const ucp_params_t& ctx_params, int value,
                                      const std::string& name, int thread_type)
{
    add_variant_value(add_variant(variants, ctx_params, thread_type).values,
                      value, name);
}

void ucp_test::add_variant_values(std::vector<ucp_test_variant>& variants,
                                  get_variants_func_t generator, int value,
                                  const std::string& name)
{
    std::vector<ucp_test_variant> tmp_variants;
    generator(tmp_variants);
    for (std::vector<ucp_test_variant>::const_iterator iter = tmp_variants.begin();
         iter != tmp_variants.end(); ++iter) {
        variants.push_back(*iter);
        add_variant_value(variants.back().values, value, name);
    }
}

void ucp_test::add_variant_values(std::vector<ucp_test_variant>& variants,
                                  get_variants_func_t generator, uint64_t bitmap,
                                  const char **names)
{
    int value;
    ucs_for_each_bit(value, bitmap) {
        add_variant_values(variants, generator, value, names[value]);
    }
}

void ucp_test::add_variant_memtypes(std::vector<ucp_test_variant>& variants,
                                    get_variants_func_t generator,
                                    uint64_t mem_types_mask)
{
    for (auto mem_type : mem_buffer::supported_mem_types()) {
        if (UCS_BIT(mem_type) & mem_types_mask) {
            add_variant_values(variants, generator, mem_type,
                               ucs_memory_type_names[mem_type]);
        }
    }
}

std::vector<ucp_test_param>
ucp_test::enum_test_params(const std::vector<ucp_test_variant>& variants,
                           const std::string& tls) {
    std::vector<ucp_test_param> result;

    if (!check_tls(tls)) {
        goto out;
    }

    for (std::vector<ucp_test_variant>::const_iterator iter = variants.begin();
         iter != variants.end(); ++iter) {

        result.push_back(ucp_test_param());
        result.back().variant = *iter;

        /* split transports to a vector */
        std::stringstream ss(tls);
        while (ss.good()) {
            std::string tl_name;
            std::getline(ss, tl_name, ',');
            result.back().transports.push_back(tl_name);
        }
    }

out:
    return result;
}

void ucp_test::modify_config(const std::string& name, const std::string& value,
                             modify_config_mode_t mode)
{
    ucs_status_t status;

    if (mode == IGNORE_IF_NOT_EXIST) {
        (void)ucp_config_modify(m_ucp_config, name.c_str(), value.c_str());
    } else {
        status = ucp_config_modify_internal(m_ucp_config, name.c_str(),
                                            value.c_str());
        if (status == UCS_ERR_NO_ELEM) {
            test_base::modify_config(name, value, mode);
        } else if (status != UCS_OK) {
            UCS_TEST_ABORT("Couldn't modify ucp config parameter: "
                           << name.c_str() << " to " << value.c_str() << ": "
                           << ucs_status_string(status));
        }
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

bool ucp_test::check_tls(const std::string& tls)
{
    typedef std::map<std::string, bool> cache_t;
    static cache_t cache;

    cache_t::iterator iter = cache.find(tls);
    if (iter != cache.end()) {
        return iter->second;
    }

    ucs::handle<ucp_config_t*> config;
    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);
    set_ucp_config(config, tls);

    ucp_context_h ucph;
    ucs_status_t status;
    {
        scoped_log_handler slh(hide_errors_logger);
        ucp_params_t ctx_params = {};
        ctx_params.field_mask   = UCP_PARAM_FIELD_FEATURES;
        ctx_params.features     = UCP_FEATURE_TAG |
                                  UCP_FEATURE_RMA |
                                  UCP_FEATURE_STREAM |
                                  UCP_FEATURE_AM |
                                  UCP_FEATURE_AMO32 |
                                  UCP_FEATURE_AMO64;
        status = ucp_init(&ctx_params, config, &ucph);
    }
    if (status == UCS_OK) {
        ucp_cleanup(ucph);
    } else if (status == UCS_ERR_NO_DEVICE) {
        UCS_TEST_MESSAGE << tls << " is not available";
    } else {
        UCS_TEST_ABORT("Failed to create context (TLS=" << tls << "): "
                       << ucs_status_string(status));
    }

    return cache[tls] = (status == UCS_OK);
}

unsigned ucp_test::mt_num_threads()
{
#if _OPENMP && ENABLE_MT
    /* Assume each thread can create two workers (sender and receiver entity),
       and each worker can open up to 64 files */
    return std::min(omp_get_max_threads(), ucs_sys_max_open_files() / (64 * 2));
#else
    return 1;
#endif
}

ucp_test_base::entity::entity(const ucp_test_param& test_param,
                              ucp_config_t* ucp_config,
                              const ucp_worker_params_t& worker_params,
                              const ucp_test_base *test_owner) :
        m_err_cntr(0), m_rejected_cntr(0), m_accept_err_cntr(0),
        m_test(test_owner)
{
    const int thread_type                   = test_param.variant.thread_type;
    ucp_params_t local_ctx_params           = test_param.variant.ctx_params;
    ucp_worker_params_t local_worker_params = worker_params;
    int num_workers;

    if (thread_type == MULTI_THREAD_CONTEXT) {
        /* Test multi-threading on context level, so create multiple workers
           which share the context */
        num_workers                        = ucp_test::mt_num_threads();
        local_ctx_params.field_mask       |= UCP_PARAM_FIELD_MT_WORKERS_SHARED;
        local_ctx_params.mt_workers_shared = 1;
    } else {
        /* Test multi-threading on worker level, so create a single worker */
        num_workers = 1;
    }

    /* Set thread mode according to variant.thread_type, unless it's already set
       in worker_params */
    if (!(worker_params.field_mask & UCP_WORKER_PARAM_FIELD_THREAD_MODE) &&
        (thread_type == MULTI_THREAD_WORKER)) {
        local_worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
        local_worker_params.field_mask |= UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    }

    /* Set transports configuration */
    std::stringstream ss;
    ss << test_param.transports;
    ucp_test::set_ucp_config(ucp_config, ss.str());

    {
        scoped_log_handler slh(hide_errors_logger);
        UCS_TEST_CREATE_HANDLE_IF_SUPPORTED(ucp_context_h, m_ucph, ucp_cleanup,
                                            ucp_init, &local_ctx_params,
                                            ucp_config);
    }

    m_workers.resize(num_workers);
    for (int i = 0; i < num_workers; i++) {
        /* We could have "invalid configuration" errors only when used
           ucp_config_modify(), in which case we wanted to ignore them. */
        scoped_log_handler slh(hide_config_warns_logger);
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

/*
 * Checks if the client's address matches any IP address on the server's side.
 */
bool ucp_test_base::entity::verify_client_address(struct sockaddr_storage
                                                  *client_address)
{
    struct ifaddrs* ifaddrs;

    if (getifaddrs(&ifaddrs) != 0) {
        return false;
    }

    for (struct ifaddrs *ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
        if (ucs_netif_flags_is_active(ifa->ifa_flags) &&
            ucs::is_inet_addr(ifa->ifa_addr))
        {
            if (!ucs_sockaddr_ip_cmp((const struct sockaddr*)client_address,
                                     ifa->ifa_addr)) {
                freeifaddrs(ifaddrs);
                return true;
            }
        }
    }

    freeifaddrs(ifaddrs);
    return false;
}

void ucp_test_base::entity::accept(int worker_index,
                                   ucp_conn_request_h conn_request)
{
    ucp_worker_h ucp_worker   = worker(worker_index);
    ucp_ep_params_t ep_params = *m_server_ep_params;
    ucp_conn_request_attr_t attr;
    ucs_status_t status;
    ucp_ep_h ep;

    attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    status = ucp_conn_request_query(conn_request, &attr);
    EXPECT_TRUE((status == UCS_OK) || (status == UCS_ERR_UNSUPPORTED));
    if (status == UCS_OK) {
        EXPECT_TRUE(verify_client_address(&attr.client_address));
    }

    ep_params.field_mask  |= UCP_EP_PARAM_FIELD_CONN_REQUEST |
                             UCP_EP_PARAM_FIELD_USER_DATA;
    ep_params.user_data    = reinterpret_cast<void*>(this);
    ep_params.conn_request = conn_request;

    status = ucp_ep_create(ucp_worker, &ep_params, &ep);
    if (status == UCS_ERR_UNREACHABLE) {
        UCS_TEST_SKIP_R("Skipping due an unreachable destination (unsupported "
                        "feature or no supported transport to send partial "
                        "worker address)");
    } else if (status != UCS_OK) {
        ++m_accept_err_cntr;
        return;
    }

    set_ep(ep, worker_index, std::numeric_limits<int>::max());
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

ucs_log_func_rc_t ucp_test_base::entity::hide_config_warns_logger(
        const char *file, unsigned line, const char *function,
        ucs_log_level_t level, const ucs_log_component_config_t *comp_conf,
        const char *message, va_list ap)
{
    if (strstr(message, "invalid configuration") == NULL) {
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    return common_logger(UCS_LOG_LEVEL_WARN, false, m_warnings,
                         std::numeric_limits<size_t>::max(), file, line,
                         function, level, comp_conf, message, ap);
}

void ucp_test_base::entity::empty_send_completion(void *r, ucs_status_t status) {
}

void ucp_test_base::entity::accept_ep_cb(ucp_ep_h ep, void *arg) {
    entity *self = reinterpret_cast<entity*>(arg);
    int worker_index = 0; /* TODO pass worker index in arg */

    /* take error handler from test fixture and add user data */
    ucp_ep_params_t ep_params = *self->m_server_ep_params;
    ep_params.field_mask &= UCP_EP_PARAM_FIELD_ERR_HANDLER;
    ep_params.field_mask |= UCP_EP_PARAM_FIELD_USER_DATA;
    ep_params.user_data   = reinterpret_cast<void*>(self);

    void *req = ucp_ep_modify_nb(ep, &ep_params);
    ASSERT_UCS_PTR_OK(req); /* don't expect this operation to block */

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

void *ucp_test_base::entity::disconnect_nb(int worker_index, int ep_index,
                                           enum ucp_ep_close_mode mode) {
    ucp_ep_h ep = revoke_ep(worker_index, ep_index);
    if (ep == NULL) {
        return NULL;
    }

    void *req = ucp_ep_close_nb(ep, mode);
    if (UCS_PTR_IS_PTR(req)) {
        m_close_ep_reqs.push_back(req);
        return req;
    }

    /* close request can be completed with any status depends on peer state */
    return NULL;
}

void ucp_test_base::entity::close_ep_req_free(void *close_req) {
    if (close_req == NULL) {
        return;
    }

    ucs_status_t status = UCS_PTR_IS_ERR(close_req) ? UCS_PTR_STATUS(close_req) :
                          ucp_request_check_status(close_req);
    ASSERT_NE(UCS_INPROGRESS, status) << "free not completed EP close request: "
                                      << close_req;
    if (status != UCS_OK) {
        UCS_TEST_MESSAGE << "ucp_ep_close_nb completed with status "
                         << ucs_status_string(status);
    }

    m_close_ep_reqs.erase(std::find(m_close_ep_reqs.begin(),
                                    m_close_ep_reqs.end(), close_req));
    ucp_request_free(close_req);
}

void ucp_test_base::entity::close_all_eps(const ucp_test &test, int worker_idx,
                                          enum ucp_ep_close_mode mode) {
    for (int j = 0; j < get_num_eps(worker_idx); j++) {
        disconnect_nb(worker_idx, j, mode);
    }

    ucs_time_t deadline = ucs::get_deadline();
    while (!m_close_ep_reqs.empty() && (ucs_get_time() < deadline)) {
        void *req = m_close_ep_reqs.front();
        while (!is_request_completed(req)) {
            test.progress(worker_idx);
        }

        close_ep_req_free(req);
    }

    EXPECT_TRUE(m_close_ep_reqs.empty()) << m_close_ep_reqs.size()
                                         << " endpoints were not closed";
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
                                           socklen_t addrlen,
                                           const ucp_ep_params_t& ep_params,
                                           ucp_listener_conn_handler_t* custom_cb,
                                           int worker_index)
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
    case LISTEN_CB_CUSTOM:
        params.field_mask        |= UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
        params.conn_handler.cb    = custom_cb->cb;
        params.conn_handler.arg   = custom_cb->arg;
        break;
    default:
        UCS_TEST_ABORT("invalid test parameter");
    }

    m_server_ep_params.reset(new ucp_ep_params_t(ep_params),
                             ucs::deleter<ucp_ep_params_t>);

    ucs_status_t status;
    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        status = ucp_listener_create(worker(worker_index), &params, &listener);
    }

    if (status == UCS_OK) {
        m_listener.reset(listener, ucp_listener_destroy);
    } else {
        /* throw error if status is not (UCS_OK or UCS_ERR_UNREACHABLE or
         * UCS_ERR_BUSY).
         * UCS_ERR_INVALID_PARAM may also return but then the test should fail */
        EXPECT_TRUE((status == UCS_ERR_UNREACHABLE) ||
                    (status == UCS_ERR_BUSY)) << ucs_status_string(status);
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
        accept(worker_index, conn_req);
        ++progress_count;
    }

    return progress_count + ucp_worker_progress(ucp_worker);
}

ucp_mem_h ucp_test_base::entity::mem_map(void *address, size_t length)
{
    ucp_mem_map_params_t params;

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address    = address;
    params.length     = length;

    ucp_mem_h memh;
    ucs_status_t status = ucp_mem_map(ucph(), &params, &memh);
    ASSERT_UCS_OK(status);

    return memh;
}

void ucp_test_base::entity::mem_unmap(ucp_mem_h memh)
{
    ucs_status_t status = ucp_mem_unmap(ucph(), memh);
    ASSERT_UCS_OK(status);
}

int ucp_test_base::entity::get_num_workers() const
{
    return m_workers.size();
}

int ucp_test_base::entity::get_num_eps(int worker_index) const {
    return m_workers[worker_index].second.size();
}

void ucp_test_base::entity::add_err(ucs_status_t status) {
    switch (status) {
    case UCS_ERR_REJECTED:
        ++m_rejected_cntr;
        /* fall through */
    default:
        ++m_err_cntr;
    }

    EXPECT_EQ(1ul, m_err_cntr) << "error callback is called more than once";
}

const size_t &ucp_test_base::entity::get_err_num_rejected() const {
    return m_rejected_cntr;
}

const size_t &ucp_test_base::entity::get_err_num() const {
    return m_err_cntr;
}

const size_t &ucp_test_base::entity::get_accept_err_num() const {
    return m_accept_err_cntr;
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

double ucp_test_base::entity::set_ib_ud_peer_timeout(double timeout_sec)
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
        const ucp_test *test = dynamic_cast<const ucp_test*>(e->m_test);
        ASSERT_TRUE(test != NULL);

        test->progress();
        status = ucp_request_test(req, &info);
    } while (status == UCS_INPROGRESS);
    EXPECT_EQ(UCS_OK, status);
    ucp_request_release(req);
}

bool ucp_test_base::entity::has_lane_with_caps(uint64_t caps) const
{
    ucp_ep_h ep         = this->ep();
    ucp_worker_h worker = this->worker();
    ucp_lane_index_t lane;
    uct_iface_attr_t *iface_attr;

    for (lane = 0; lane < ucp_ep_config(ep)->key.num_lanes; lane++) {
        iface_attr = ucp_worker_iface_get_attr(worker,
                                               ucp_ep_config(ep)->key.lanes[lane].rsc_index);
        if (ucs_test_all_flags(iface_attr->cap.flags, caps)) {
            return true;
        }
    }

    return false;
}

bool ucp_test_base::entity::is_conn_reqs_queue_empty() const
{
    return m_conn_reqs.empty();
}

bool ucp_test_base::is_request_completed(void *request) {
    return (request == NULL) ||
           (ucp_request_check_status(request) != UCS_INPROGRESS);
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
    if (status != UCS_OK) {
        ucs_warn("failed to unmap memh=%p: %s", m_memh,
                 ucs_status_string(status));
    }
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

void test_ucp_context::get_test_variants(std::vector<ucp_test_variant> &variants)
{
    add_variant(variants, UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP);
}

void ucp_test::disable_keepalive()
{
    modify_config("KEEPALIVE_INTERVAL", "inf");
}

bool ucp_test::check_reg_mem_types(const entity& e, ucs_memory_type_t mem_type) {
    for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(e.ep()); lane++) {
        const uct_md_attr_t* attr = ucp_ep_md_attr(e.ep(), lane);
        if (attr->cap.reg_mem_types & UCS_BIT(mem_type)) {
            return true;
        }
    }

    return false;
}
