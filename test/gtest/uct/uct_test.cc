/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_test.h"
#include "uct/api/uct_def.h"

#include <ucs/stats/stats.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/string.h>
#include <common/test_helpers.h>
#include <algorithm>
#include <malloc.h>
#include <ifaddrs.h>


std::string resource::name() const {
    std::stringstream ss;
    ss << tl_name << "/" << dev_name;
    return ss.str();
}

resource::resource() : component(NULL), md_name(""), tl_name(""), dev_name(""),
                       dev_type(UCT_DEVICE_TYPE_LAST)
{
    CPU_ZERO(&local_cpus);
}

resource::resource(uct_component_h component, const std::string& md_name,
                   const ucs_cpu_set_t& local_cpus, const std::string& tl_name,
                   const std::string& dev_name, uct_device_type_t dev_type) :
                   component(component), md_name(md_name), local_cpus(local_cpus),
                   tl_name(tl_name), dev_name(dev_name), dev_type(dev_type)
{
}

resource::resource(uct_component_h component, const uct_md_attr_t& md_attr,
                   const uct_md_resource_desc_t& md_resource,
                   const uct_tl_resource_desc_t& tl_resource) :
                   component(component),
                   md_name(md_resource.md_name),
                   local_cpus(md_attr.local_cpus),
                   tl_name(tl_resource.tl_name),
                   dev_name(tl_resource.dev_name),
                   dev_type(tl_resource.dev_type)
{
}

resource_speed::resource_speed(uct_component_h component, const uct_worker_h& worker,
                               const uct_md_h& md, const uct_md_attr_t& md_attr,
                               const uct_md_resource_desc_t& md_resource,
                               const uct_tl_resource_desc_t& tl_resource) :
                               resource(component, md_attr, md_resource,
                                        tl_resource) {
    ucs_status_t status;
    uct_iface_params_t iface_params = { 0 };
    uct_iface_config_t *iface_config;
    uct_iface_attr_t iface_attr;
    uct_iface_h iface;

    status = uct_md_iface_config_read(md, tl_name.c_str(), NULL,
                                      NULL, &iface_config);
    ASSERT_UCS_OK(status);

    iface_params.field_mask           = UCT_IFACE_PARAM_FIELD_OPEN_MODE |
                                        UCT_IFACE_PARAM_FIELD_DEVICE;
    iface_params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
    iface_params.mode.device.tl_name  = tl_name.c_str();
    iface_params.mode.device.dev_name = dev_name.c_str();

    status = uct_iface_open(md, worker, &iface_params, iface_config, &iface);
    ASSERT_UCS_OK(status);

    status = uct_iface_query(iface, &iface_attr);
    ASSERT_UCS_OK(status);

    bw = ucs_max(iface_attr.bandwidth.dedicated, iface_attr.bandwidth.shared);

    uct_iface_close(iface);
    uct_config_release(iface_config);
}

std::vector<uct_test_base::md_resource> uct_test_base::enum_md_resources() {

    static std::vector<uct_test::md_resource> all_md_resources;

    if (all_md_resources.empty()) {
        uct_component_h *uct_components;
        unsigned num_components;
        ucs_status_t status;

        status = uct_query_components(&uct_components, &num_components);
        ASSERT_UCS_OK(status);

        /* for RAII */
        ucs::handle<uct_component_h*> cmpt_list(uct_components,
                                                uct_release_component_list);

        for (unsigned cmpt_index = 0; cmpt_index < num_components; ++cmpt_index) {
            uct_component_attr_t component_attr = {0};

            component_attr.field_mask = UCT_COMPONENT_ATTR_FIELD_NAME |
                                        UCT_COMPONENT_ATTR_FIELD_MD_RESOURCE_COUNT |
                                        UCT_COMPONENT_ATTR_FIELD_FLAGS;
            /* coverity[var_deref_model] */
            status = uct_component_query(uct_components[cmpt_index], &component_attr);
            ASSERT_UCS_OK(status);

            /* Save attributes before asking for MD resource list */
            md_resource md_rsc;
            md_rsc.cmpt      = uct_components[cmpt_index];
            md_rsc.cmpt_attr = component_attr;

            std::vector<uct_md_resource_desc_t> md_resources;
            uct_component_attr_t component_attr_resouces = {0};
            md_resources.resize(md_rsc.cmpt_attr.md_resource_count);
            component_attr_resouces.field_mask   = UCT_COMPONENT_ATTR_FIELD_MD_RESOURCES;
            component_attr_resouces.md_resources = &md_resources[0];
            status = uct_component_query(uct_components[cmpt_index],
                                         &component_attr_resouces);
            ASSERT_UCS_OK(status);

            for (unsigned md_index = 0;
                 md_index < md_rsc.cmpt_attr.md_resource_count; ++md_index) {
                md_rsc.rsc_desc = md_resources[md_index];
                all_md_resources.push_back(md_rsc);
            }
        }
    }

    return all_md_resources;
}

uct_test::uct_test() {
    ucs_status_t status;
    uct_md_attr_t pd_attr;
    uct_md_h pd;

    status = uct_md_config_read(GetParam()->component, NULL, NULL, &m_md_config);
    ASSERT_UCS_OK(status);

    status = uct_md_open(GetParam()->component, GetParam()->md_name.c_str(),
                         m_md_config, &pd);
    ASSERT_UCS_OK(status);

    status = uct_md_query(pd, &pd_attr);
    ASSERT_UCS_OK(status);

    if (pd_attr.cap.flags & UCT_MD_FLAG_SOCKADDR) {
        status = uct_md_iface_config_read(pd, NULL, NULL, NULL, &m_iface_config);
    } else {
        status = uct_md_iface_config_read(pd, GetParam()->tl_name.c_str(), NULL,
                                          NULL, &m_iface_config);
    }

    ASSERT_UCS_OK(status);
    uct_md_close(pd);
}

uct_test::~uct_test() {
    uct_config_release(m_iface_config);
    uct_config_release(m_md_config);
}

void uct_test::init_sockaddr_rsc(resource *rsc, struct sockaddr *listen_addr,
                                 struct sockaddr *connect_addr, size_t size)
{
    rsc->listen_sock_addr.set_sock_addr(*listen_addr, size);
    rsc->connect_sock_addr.set_sock_addr(*connect_addr, size);
}

void uct_test::set_interface_rscs(const md_resource& md_rsc,
                                  ucs_cpu_set_t local_cpus, struct ifaddrs *ifa,
                                  std::vector<resource>& all_resources)
{
    int i;

    /* Create two resources on the same interface. the first one will have the
     * ip of the interface and the second one will have INADDR_ANY */
    for (i = 0; i < 2; i++) {
        resource rsc(md_rsc.cmpt, std::string(md_rsc.rsc_desc.md_name),
                     local_cpus, "sockaddr", std::string(ifa->ifa_name),
                     UCT_DEVICE_TYPE_NET);

        if (i == 0) {
            /* first rsc */
            if (ifa->ifa_addr->sa_family == AF_INET) {
                uct_test::init_sockaddr_rsc(&rsc, ifa->ifa_addr, ifa->ifa_addr,
                                            sizeof(struct sockaddr_in));
            } else if (ifa->ifa_addr->sa_family == AF_INET6) {
                uct_test::init_sockaddr_rsc(&rsc, ifa->ifa_addr, ifa->ifa_addr,
                                            sizeof(struct sockaddr_in6));
            } else {
                UCS_TEST_ABORT("Unknown sa_family " << ifa->ifa_addr->sa_family);
            }
            all_resources.push_back(rsc);
        } else {
            /* second rsc */
            if (ifa->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in sin;
                memset(&sin, 0, sizeof(struct sockaddr_in));
                sin.sin_family      = AF_INET;
                sin.sin_addr.s_addr = INADDR_ANY;
                uct_test::init_sockaddr_rsc(&rsc, (struct sockaddr*)&sin,
                                            ifa->ifa_addr, sizeof(struct sockaddr_in));
            } else if (ifa->ifa_addr->sa_family == AF_INET6) {
                struct sockaddr_in6 sin;
                memset(&sin, 0, sizeof(struct sockaddr_in6));
                sin.sin6_family     = AF_INET6;
                sin.sin6_addr       = in6addr_any;
                uct_test::init_sockaddr_rsc(&rsc, (struct sockaddr*)&sin,
                                            ifa->ifa_addr, sizeof(struct sockaddr_in6));
            } else {
                UCS_TEST_ABORT("Unknown sa_family " << ifa->ifa_addr->sa_family);
            }
            all_resources.push_back(rsc);
        }
    }
}

void uct_test::set_sockaddr_resources(const md_resource& md_rsc, uct_md_h md,
                                      ucs_cpu_set_t local_cpus,
                                      std::vector<resource>& all_resources) {

    struct ifaddrs *ifaddr, *ifa;
    ucs_sock_addr_t sock_addr;

    EXPECT_TRUE(getifaddrs(&ifaddr) != -1);

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        sock_addr.addr = ifa->ifa_addr;

        if (!ucs_netif_flags_is_active(ifa->ifa_flags)) {
            continue;
        }

        /* If rdmacm is tested, make sure that this is an IPoIB or RoCE interface */
        if (!strcmp(md_rsc.rsc_desc.md_name, "rdmacm") &&
            !ucs::is_rdmacm_netdev(ifa->ifa_name)) {
            continue;
        }

        if (uct_md_is_sockaddr_accessible(md, &sock_addr, UCT_SOCKADDR_ACC_LOCAL) &&
            uct_md_is_sockaddr_accessible(md, &sock_addr, UCT_SOCKADDR_ACC_REMOTE))
        {
            uct_test::set_interface_rscs(md_rsc, local_cpus, ifa, all_resources);
        }
    }

    freeifaddrs(ifaddr);
}

std::vector<const resource*> uct_test::enum_resources(const std::string& tl_name)
{
    static bool tcp_fastest_dev = (getenv("GTEST_UCT_TCP_FASTEST_DEV") != NULL);
    static std::vector<resource> all_resources;

    if (all_resources.empty()) {
        ucs_async_context_t *async;
        uct_worker_h worker;
        ucs_status_t status;

        status = ucs_async_context_create(UCS_ASYNC_MODE_THREAD_SPINLOCK, &async);
        ASSERT_UCS_OK(status);

        status = uct_worker_create(async, UCS_THREAD_MODE_SINGLE, &worker);
        ASSERT_UCS_OK(status);

        std::vector<md_resource> md_resources = enum_md_resources();

        for (std::vector<md_resource>::iterator iter = md_resources.begin();
             iter != md_resources.end(); ++iter) {
            uct_md_h md;
            uct_md_config_t *md_config;
            status = uct_md_config_read(iter->cmpt, NULL, NULL, &md_config);
            ASSERT_UCS_OK(status);

            {
                scoped_log_handler slh(hide_errors_logger);
                status = uct_md_open(iter->cmpt, iter->rsc_desc.md_name,
                                     md_config, &md);
            }
            uct_config_release(md_config);
            if (status != UCS_OK) {
                continue;
            }

            uct_md_attr_t md_attr;
            status = uct_md_query(md, &md_attr);
            ASSERT_UCS_OK(status);

            uct_tl_resource_desc_t *tl_resources;
            unsigned num_tl_resources;
            status = uct_md_query_tl_resources(md, &tl_resources, &num_tl_resources);
            ASSERT_UCS_OK(status);

            resource_speed tcp_fastest_rsc;

            for (unsigned j = 0; j < num_tl_resources; ++j) {
                if (tcp_fastest_dev && (std::string("tcp") == tl_resources[j].tl_name)) {
                    resource_speed rsc(iter->cmpt, worker, md, md_attr,
                                       iter->rsc_desc, tl_resources[j]);
                    if (!tcp_fastest_rsc.bw || (rsc.bw > tcp_fastest_rsc.bw)) {
                        tcp_fastest_rsc = rsc;
                    }
                } else {
                    resource rsc(iter->cmpt, md_attr, iter->rsc_desc,
                                 tl_resources[j]);
                    all_resources.push_back(rsc);
                }
            }

            if (tcp_fastest_dev && tcp_fastest_rsc.bw) {
                all_resources.push_back(tcp_fastest_rsc);
            }

            if (md_attr.cap.flags & UCT_MD_FLAG_SOCKADDR) {
                uct_test::set_sockaddr_resources(*iter, md, md_attr.local_cpus,
                                                 all_resources);
            }

            uct_release_tl_resource_list(tl_resources);
            uct_md_close(md);
        }

        uct_worker_destroy(worker);
        ucs_async_context_destroy(async);
    }

    return filter_resources(all_resources, tl_name);
}

void uct_test::init() {
}

void uct_test::cleanup() {
    FOR_EACH_ENTITY(iter) {
        (*iter)->destroy_eps();
    }
    m_entities.clear();
}

bool uct_test::is_caps_supported(uint64_t required_flags) {
    bool ret = true;

    FOR_EACH_ENTITY(iter) {
        ret &= (*iter)->is_caps_supported(required_flags);
    }

    return ret;
}

bool uct_test::check_caps(uint64_t required_flags, uint64_t invalid_flags) {
    FOR_EACH_ENTITY(iter) {
        if (!(*iter)->check_caps(required_flags, invalid_flags)) {
            return false;
        }
    }
    return true;
}

void uct_test::check_caps_skip(uint64_t required_flags, uint64_t invalid_flags) {
    if (!check_caps(required_flags, invalid_flags)) {
        UCS_TEST_SKIP_R("unsupported");
    }
}

bool uct_test::check_atomics(uint64_t required_ops, atomic_mode mode) {
    FOR_EACH_ENTITY(iter) {
        if (!(*iter)->check_atomics(required_ops, mode)) {
            return false;
        }
    }
    return true;
}

void uct_test::modify_config(const std::string& name, const std::string& value,
                             bool optional) {
    ucs_status_t status;

    status = uct_config_modify(m_iface_config, name.c_str(), value.c_str());
    if (status == UCS_ERR_NO_ELEM) {
        status = uct_config_modify(m_md_config, name.c_str(), value.c_str());
        if (status == UCS_ERR_NO_ELEM) {
            test_base::modify_config(name, value, optional);
        } else if (status != UCS_OK) {
            UCS_TEST_ABORT("Couldn't modify pd config parameter: " << name.c_str() <<
                           " to " << value.c_str() << ": " << ucs_status_string(status));
        }

    } else if (status != UCS_OK) {
        UCS_TEST_ABORT("Couldn't modify iface config parameter: " << name.c_str() <<
                       " to " << value.c_str() << ": " << ucs_status_string(status));
    }
}

bool uct_test::get_config(const std::string& name, std::string& value) const
{
    ucs_status_t status;
    const size_t max = 1024;

    value.resize(max);
    status = uct_config_get(m_iface_config, name.c_str(),
                            const_cast<char *>(value.c_str()), max);

    if (status == UCS_ERR_NO_ELEM) {
        status = uct_config_get(m_md_config, name.c_str(),
                                const_cast<char *>(value.c_str()), max);
    }

    return (status == UCS_OK);
}

bool uct_test::has_transport(const std::string& tl_name) const {
    return (GetParam()->tl_name == tl_name);
}

bool uct_test::has_ud() const {
    return (has_transport("ud_verbs") || has_transport("ud_mlx5"));
}

bool uct_test::has_rc() const {
    return (has_transport("rc_verbs") || has_transport("rc_mlx5"));
}

bool uct_test::has_rc_or_dc() const {
    return (has_rc() || has_transport("dc_mlx5"));
}

bool uct_test::has_ib() const {
    return (has_rc_or_dc() || has_ud());
}

void uct_test::stats_activate()
{
    ucs_stats_cleanup();
    push_config();
    modify_config("STATS_DEST",    "file:/dev/null");
    modify_config("STATS_TRIGGER", "exit");
    ucs_stats_init();
    ASSERT_TRUE(ucs_stats_is_active());
}

void uct_test::stats_restore()
{
    ucs_stats_cleanup();
    pop_config();
    ucs_stats_init();
}

uct_test::entity* uct_test::create_entity(size_t rx_headroom,
                                          uct_error_handler_t err_handler) {
    uct_iface_params_t iface_params;

    iface_params.field_mask        = UCT_IFACE_PARAM_FIELD_RX_HEADROOM     |
                                     UCT_IFACE_PARAM_FIELD_OPEN_MODE       |
                                     UCT_IFACE_PARAM_FIELD_ERR_HANDLER     |
                                     UCT_IFACE_PARAM_FIELD_ERR_HANDLER_ARG |
                                     UCT_IFACE_PARAM_FIELD_ERR_HANDLER_FLAGS;
    iface_params.rx_headroom       = rx_headroom;
    iface_params.open_mode         = UCT_IFACE_OPEN_MODE_DEVICE;
    iface_params.err_handler       = err_handler;
    iface_params.err_handler_arg   = this;
    iface_params.err_handler_flags = 0;
    entity *new_ent = new entity(*GetParam(), m_iface_config, &iface_params,
                                 m_md_config);
    return new_ent;
}

uct_test::entity* uct_test::create_entity(uct_iface_params_t &params) {
    entity *new_ent = new entity(*GetParam(), m_iface_config, &params,
                                 m_md_config);
    return new_ent;
}

uct_test::entity* uct_test::create_entity() {
    return new entity(*GetParam(), m_md_config);
}

const uct_test::entity& uct_test::ent(unsigned index) const {
    return m_entities.at(index);
}

unsigned uct_test::progress() const {
    unsigned count = 0;
    FOR_EACH_ENTITY(iter) {
        count += (*iter)->progress();
    }
    return count;
}

void uct_test::flush(ucs_time_t deadline) const {

    bool flushed;
    do {
        flushed = true;
        FOR_EACH_ENTITY(iter) {
            (*iter)->progress();
            ucs_status_t status = uct_iface_flush((*iter)->iface(), 0, NULL);
            if ((status == UCS_ERR_NO_RESOURCE) || (status == UCS_INPROGRESS)) {
                flushed = false;
            } else {
                ASSERT_UCS_OK(status);
            }
        }
    } while (!flushed && (ucs_get_time() < deadline));

    EXPECT_TRUE(flushed) << "Timed out";
}

void uct_test::short_progress_loop(double delay_ms) const {
    ucs_time_t end_time = ucs_get_time() + ucs_time_from_msec(delay_ms * ucs::test_time_multiplier());
    while (ucs_get_time() < end_time) {
        progress();
    }
}

void uct_test::twait(int delta_ms) const {
    ucs_time_t now, t1, t2;
    int left;

    now = ucs_get_time();
    left = delta_ms;
    do {
        t1 = ucs_get_time();
        usleep(1000 * left);
        t2 = ucs_get_time();
        left -= (int)ucs_time_to_msec(t2-t1);
    } while (now + ucs_time_from_msec(delta_ms) > ucs_get_time());
}

int uct_test::max_connections()
{
    if (has_transport("tcp")) {
        return ucs::max_tcp_connections();
    } else {
        return std::numeric_limits<int>::max();
    }
}

int uct_test::max_connect_batch()
{
    if (has_transport("tcp")) {
        /* TCP connection listener is limited by Accept queue */
        return ucs_socket_max_conn();
    } else {
        return std::numeric_limits<int>::max();
    }
}

const std::string uct_test::entity::server_priv_data = "Server private data";
std::string uct_test::entity::client_priv_data = "";

uct_test::entity::entity(const resource& resource, uct_iface_config_t *iface_config,
                         uct_iface_params_t *params, uct_md_config_t *md_config) :
    m_resource(resource)
{
    ucs_status_t status;

    if (params->open_mode == UCT_IFACE_OPEN_MODE_DEVICE) {
        params->field_mask          |= UCT_IFACE_PARAM_FIELD_DEVICE;
        params->mode.device.tl_name  = resource.tl_name.c_str();
        params->mode.device.dev_name = resource.dev_name.c_str();
    }

    params->field_mask |= UCT_IFACE_PARAM_FIELD_STATS_ROOT |
                          UCT_IFACE_PARAM_FIELD_CPU_MASK;
    params->stats_root  = ucs_stats_get_root();
    UCS_CPU_ZERO(&params->cpu_mask);

    UCS_TEST_CREATE_HANDLE(uct_worker_h, m_worker, uct_worker_destroy,
                           uct_worker_create, &m_async.m_async,
                           UCS_THREAD_MODE_SINGLE);

    UCS_TEST_CREATE_HANDLE(uct_md_h, m_md, uct_md_close, uct_md_open,
                           resource.component, resource.md_name.c_str(),
                           md_config);

    status = uct_md_query(m_md, &m_md_attr);
    ASSERT_UCS_OK(status);

    for (;;) {
        {
            scoped_log_handler slh(wrap_errors_logger);
            status = UCS_TEST_TRY_CREATE_HANDLE(uct_iface_h, m_iface,
                                                uct_iface_close, uct_iface_open,
                                                m_md, m_worker, params,
                                                iface_config);
            if (status == UCS_OK) {
                break;
            }
        }
        EXPECT_EQ(UCS_ERR_BUSY, status);
        if (params->open_mode != UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER) {
            UCS_TEST_ABORT("any mode different from UCT_IFACE_OPEN_MODE_SOCKADDR_SERVER must go with status UCS_OK");
        }

        const struct sockaddr* c_ifa_addr =
            params->mode.sockaddr.listen_sockaddr.addr;
        struct sockaddr* ifa_addr = const_cast<struct sockaddr*>(c_ifa_addr);
        if (ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in *addr =
                reinterpret_cast<struct sockaddr_in *>(ifa_addr);
            addr->sin_port = ucs::get_port();
        } else {
            struct sockaddr_in6 *addr =
                reinterpret_cast<struct sockaddr_in6 *>(ifa_addr);
            addr->sin6_port = ucs::get_port();
        }
    }

    status = uct_iface_query(m_iface, &m_iface_attr);
    ASSERT_UCS_OK(status);

    uct_iface_progress_enable(m_iface, UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    m_iface_params = *params;

    memset(&m_cm_attr, 0, sizeof(m_cm_attr));
    max_conn_priv = 0;
}

uct_test::entity::entity(const resource& resource, uct_md_config_t *md_config) {
    memset(&m_iface_attr,   0, sizeof(m_iface_attr));
    memset(&m_iface_params, 0, sizeof(m_iface_params));

    UCS_TEST_CREATE_HANDLE(uct_worker_h, m_worker, uct_worker_destroy,
                           uct_worker_create, &m_async.m_async,
                           UCS_THREAD_MODE_SINGLE);

    UCS_TEST_CREATE_HANDLE(uct_md_h, m_md, uct_md_close,
                           uct_md_open, resource.component,
                           resource.md_name.c_str(), md_config);

    ucs_status_t status = uct_md_query(m_md, &m_md_attr);
    ASSERT_UCS_OK(status);

    UCS_TEST_CREATE_HANDLE_IF_SUPPORTED(uct_cm_h, m_cm, uct_cm_close,
                                        uct_cm_open, resource.component, m_worker);

    m_cm_attr.field_mask = UCT_CM_ATTR_FIELD_MAX_CONN_PRIV;
    status = uct_cm_query(m_cm, &m_cm_attr);
    ASSERT_UCS_OK(status);

    max_conn_priv = 0;
}

void uct_test::entity::mem_alloc_host(size_t length,
                                      uct_allocated_memory_t *mem) const {

    ucs_status_t status;

    if (md_attr().cap.flags & (UCT_MD_FLAG_ALLOC|UCT_MD_FLAG_REG)) {
        status = uct_iface_mem_alloc(m_iface, length, UCT_MD_MEM_ACCESS_ALL,
                                     "uct_test", mem);
        ASSERT_UCS_OK(status);
    } else {
        uct_alloc_method_t method = UCT_ALLOC_METHOD_MMAP;
        status = uct_mem_alloc(NULL, length, UCT_MD_MEM_ACCESS_ALL, &method, 1,
                               NULL, 0, "uct_test", mem);
        ASSERT_UCS_OK(status);
        ucs_assert(mem->memh == UCT_MEM_HANDLE_NULL);
    }
    ucs_assert(mem->mem_type == UCS_MEMORY_TYPE_HOST);
}

void uct_test::entity::mem_free_host(const uct_allocated_memory_t *mem) const {
    if (mem->method != UCT_ALLOC_METHOD_LAST) {
        uct_iface_mem_free(mem);
    }
}

void uct_test::entity::mem_type_reg(uct_allocated_memory_t *mem) const {
    if (md_attr().cap.reg_mem_types & UCS_BIT(mem->mem_type)) {
        ucs_status_t status = uct_md_mem_reg(m_md, mem->address, mem->length,
                                             UCT_MD_MEM_ACCESS_ALL, &mem->memh);
        ASSERT_UCS_OK(status);
        mem->md = m_md;
    }
}

void uct_test::entity::mem_type_dereg(uct_allocated_memory_t *mem) const {
    if ((mem->memh != UCT_MEM_HANDLE_NULL) &&
        (md_attr().cap.reg_mem_types & UCS_BIT(mem->mem_type))) {
        ucs_status_t status = uct_md_mem_dereg(m_md, mem->memh);
        ASSERT_UCS_OK(status);
        mem->memh = UCT_MEM_HANDLE_NULL;
        mem->md   = NULL;
    }
}

void uct_test::entity::rkey_unpack(const uct_allocated_memory_t *mem,
                                   uct_rkey_bundle *rkey_bundle) const
{
    if ((mem->memh != UCT_MEM_HANDLE_NULL) &&
        (md_attr().cap.flags & UCT_MD_FLAG_NEED_RKEY)) {

        void *rkey_buffer = malloc(md_attr().rkey_packed_size);
        if (rkey_buffer == NULL) {
            UCS_TEST_ABORT("Failed to allocate rkey buffer");
        }

        ucs_status_t status = uct_md_mkey_pack(m_md, mem->memh, rkey_buffer);
        ASSERT_UCS_OK(status);

        status = uct_rkey_unpack(m_resource.component, rkey_buffer,
                                 rkey_bundle);
        ASSERT_UCS_OK(status);

        free(rkey_buffer);
    } else {
        rkey_bundle->handle = NULL;
        rkey_bundle->rkey   = UCT_INVALID_RKEY;
    }
}

void uct_test::entity::rkey_release(const uct_rkey_bundle *rkey_bundle) const
{
    if (rkey_bundle->rkey != UCT_INVALID_RKEY) {
        ucs_status_t status = uct_rkey_release(m_resource.component, rkey_bundle);
        ASSERT_UCS_OK(status);
    }
}

unsigned uct_test::entity::progress() const {
    unsigned count = uct_worker_progress(m_worker);
    m_async.check_miss();
    return count;
}

bool uct_test::entity::is_caps_supported(uint64_t required_flags) {
    uint64_t iface_flags = iface_attr().cap.flags;
    return ucs_test_all_flags(iface_flags, required_flags);
}

bool uct_test::entity::check_caps(uint64_t required_flags,
                                  uint64_t invalid_flags)
{
    uint64_t iface_flags = iface_attr().cap.flags;
    return (ucs_test_all_flags(iface_flags, required_flags) &&
            !(iface_flags & invalid_flags));
}

bool uct_test::entity::check_atomics(uint64_t required_ops, atomic_mode mode)
{
    uint64_t amo;

    switch (mode) {
    case OP32:
        amo = iface_attr().cap.atomic32.op_flags;
        break;
    case OP64:
        amo = iface_attr().cap.atomic64.op_flags;
        break;
    case FOP32:
        amo = iface_attr().cap.atomic32.fop_flags;
        break;
    case FOP64:
        amo = iface_attr().cap.atomic64.fop_flags;
        break;
    default:
        UCS_TEST_ABORT("Incorrect atomic mode: " << mode);
    }

    return ucs_test_all_flags(amo, required_ops);
}

uct_md_h uct_test::entity::md() const {
    return m_md;
}

const uct_md_attr& uct_test::entity::md_attr() const {
    return m_md_attr;
}

uct_worker_h uct_test::entity::worker() const {
    return m_worker;
}

uct_cm_h uct_test::entity::cm() const {
    return m_cm;
}

const uct_cm_attr_t& uct_test::entity::cm_attr() const {
    return m_cm_attr;
}

uct_listener_h uct_test::entity::listener() const {
    return m_listener;
}

uct_iface_h uct_test::entity::iface() const {
    return m_iface;
}

const uct_iface_attr& uct_test::entity::iface_attr() const {
    return m_iface_attr;
}

const uct_iface_params& uct_test::entity::iface_params() const {
    return m_iface_params;
}

uct_ep_h uct_test::entity::ep(unsigned index) const {
    return m_eps.at(index);
}

size_t uct_test::entity::num_eps() const {
    return m_eps.size();
}

void uct_test::entity::reserve_ep(unsigned index) {
    if (index >= m_eps.size()) {
        m_eps.resize(index + 1);
    }
}

void uct_test::entity::connect_p2p_ep(uct_ep_h from, uct_ep_h to)
{
    uct_iface_attr_t iface_attr;
    uct_device_addr_t *dev_addr;
    uct_ep_addr_t *ep_addr;
    ucs_status_t status;

    status = uct_iface_query(to->iface, &iface_attr);
    ASSERT_UCS_OK(status);

    dev_addr = (uct_device_addr_t*)malloc(iface_attr.device_addr_len);
    ep_addr  = (uct_ep_addr_t*)malloc(iface_attr.ep_addr_len);

    status = uct_iface_get_device_address(to->iface, dev_addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_get_address(to, ep_addr);
    ASSERT_UCS_OK(status);

    status = uct_ep_connect_to_ep(from, dev_addr, ep_addr);
    ASSERT_UCS_OK(status);

    free(ep_addr);
    free(dev_addr);
}

void uct_test::entity::create_ep(unsigned index) {
    uct_ep_h ep = NULL;
    uct_ep_params_t ep_params;
    ucs_status_t status;

    reserve_ep(index);

    if (m_eps[index]) {
        UCS_TEST_ABORT("ep[" << index << "] already exists");
    }

    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = m_iface;
    status = uct_ep_create(&ep_params, &ep);
    ASSERT_UCS_OK(status);
    m_eps[index].reset(ep, uct_ep_destroy);
}

void uct_test::entity::destroy_ep(unsigned index) {
    if (!m_eps[index]) {
        UCS_TEST_ABORT("ep[" << index << "] does not exist");
    }

    m_eps[index].reset();
}

void uct_test::entity::destroy_eps() {
    for (unsigned index = 0; index < m_eps.size(); ++index) {
        if (!m_eps[index]) {
            continue;
        }
        m_eps[index].reset();
    }
}

size_t uct_test::entity::priv_data_do_pack(void *priv_data)
{
    size_t priv_data_len;

    client_priv_data = "Client private data";
    priv_data_len = 1 + client_priv_data.length();

    memcpy(priv_data, client_priv_data.c_str(), priv_data_len);
    return priv_data_len;
}

ssize_t uct_test::entity::server_priv_data_cb(void *arg, const char *dev_name,
                                              void *priv_data)
{
    const size_t priv_data_len = server_priv_data.length() + 1;

    memcpy(priv_data, server_priv_data.c_str(), priv_data_len);
    return priv_data_len;
}

void
uct_test::entity::connect_to_sockaddr(unsigned index, entity& other,
                                      const ucs::sock_addr_storage &remote_addr,
                                      uct_sockaddr_priv_pack_callback_t pack_cb,
                                      uct_ep_client_connect_cb_t connect_cb,
                                      uct_ep_disconnect_cb_t disconnect_cb,
                                      void *user_data)
{
    ucs_sock_addr_t ucs_remote_addr = remote_addr.to_ucs_sock_addr();
    uct_ep_params_t params;
    uct_ep_h ep;
    ucs_status_t status;

    reserve_ep(index);
    if (m_eps[index]) {
        return; /* Already connected */
    }

    /* Connect to the server */
    if (m_cm) {
        params.field_mask = UCT_EP_PARAM_FIELD_CM                     |
                            UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB    |
                            UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB |
                            UCT_EP_PARAM_FIELD_USER_DATA;
        params.cm                         = m_cm;
        params.sockaddr_connect_cb.client = connect_cb;
        params.disconnect_cb              = disconnect_cb;
    } else {
        params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
        params.iface      = m_iface;
    }

    params.field_mask       |= UCT_EP_PARAM_FIELD_USER_DATA         |
                               UCT_EP_PARAM_FIELD_SOCKADDR          |
                               UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS |
                               UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB;
    params.user_data         = user_data;
    params.sockaddr          = &ucs_remote_addr;
    params.sockaddr_cb_flags = UCT_CB_FLAG_ASYNC;
    params.sockaddr_pack_cb  = pack_cb;
    status = uct_ep_create(&params, &ep);
    ASSERT_UCS_OK(status);

    m_eps[index].reset(ep, uct_ep_destroy);
}

void uct_test::entity::connect_to_ep(unsigned index, entity& other,
                                     unsigned other_index)
{
    ucs_status_t status;
    uct_ep_h ep, remote_ep;
    uct_ep_params_t ep_params;

    reserve_ep(index);
    if (m_eps[index]) {
        return; /* Already connected */
    }

    other.reserve_ep(other_index);
    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    ep_params.iface      = other.m_iface;
    status               = uct_ep_create(&ep_params, &remote_ep);
    ASSERT_UCS_OK(status);
    other.m_eps[other_index].reset(remote_ep, uct_ep_destroy);

    if (&other == this) {
        connect_p2p_ep(remote_ep, remote_ep);
    } else {
        ep_params.iface     = m_iface;
        ucs_status_t status = uct_ep_create(&ep_params, &ep);
        ASSERT_UCS_OK(status);

        connect_p2p_ep(ep, remote_ep);
        connect_p2p_ep(remote_ep, ep);

        m_eps[index].reset(ep, uct_ep_destroy);
    }
}

void uct_test::entity::connect_to_iface(unsigned index, entity& other) {
    uct_device_addr_t *dev_addr;
    uct_iface_addr_t *iface_addr;
    uct_ep_params_t ep_params;
    ucs_status_t status;
    uct_ep_h ep;

    reserve_ep(index);
    if (m_eps[index]) {
        return; /* Already connected */
    }

    dev_addr   = (uct_device_addr_t*)malloc(other.iface_attr().device_addr_len);
    iface_addr = (uct_iface_addr_t*) malloc(other.iface_attr().iface_addr_len);

    status = uct_iface_get_device_address(other.iface(), dev_addr);
    ASSERT_UCS_OK(status);

    status = uct_iface_get_address(other.iface(), iface_addr);
    ASSERT_UCS_OK(status);

    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE    |
                           UCT_EP_PARAM_FIELD_DEV_ADDR |
                           UCT_EP_PARAM_FIELD_IFACE_ADDR;
    ep_params.iface      = iface();
    ep_params.dev_addr   = dev_addr;
    ep_params.iface_addr = iface_addr;

    status = uct_ep_create(&ep_params, &ep);
    ASSERT_UCS_OK(status);

    m_eps[index].reset(ep, uct_ep_destroy);

    free(iface_addr);
    free(dev_addr);
}

void uct_test::entity::connect(unsigned index, entity& other,
                               unsigned other_index,
                               const ucs::sock_addr_storage &remote_addr,
                               uct_sockaddr_priv_pack_callback_t pack_cb,
                               uct_ep_client_connect_cb_t connect_cb,
                               uct_ep_disconnect_cb_t disconnect_cb,
                               void *user_data)
{
    if (m_cm ||
        iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR) {
        connect_to_sockaddr(index, other, remote_addr, pack_cb, connect_cb,
                            disconnect_cb, user_data);
    } else {
        UCS_TEST_SKIP_R("cannot connect");
    }
}

void uct_test::entity::connect(unsigned index, entity& other, unsigned other_index)
{
    if (iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        connect_to_ep(index, other, other_index);
    } else if (iface_attr().cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
        connect_to_iface(index, other);
    } else {
        UCS_TEST_SKIP_R("cannot connect");
    }
}

void uct_test::entity::accept(uct_cm_h cm, uct_conn_request_h conn_request,
                              uct_ep_server_connect_cb_t connect_cb,
                              uct_ep_disconnect_cb_t disconnect_cb,
                              void *user_data)
{
    uct_ep_params_t ep_params;
    ucs_status_t status;
    uct_ep_h ep;

    ASSERT_TRUE(m_listener);
    reserve_ep(m_eps.size());

    ep_params.field_mask = UCT_EP_PARAM_FIELD_CM                     |
                           UCT_EP_PARAM_FIELD_CONN_REQUEST           |
                           UCT_EP_PARAM_FIELD_USER_DATA              |
                           UCT_EP_PARAM_FIELD_SOCKADDR_CONNECT_CB    |
                           UCT_EP_PARAM_FIELD_SOCKADDR_DISCONNECT_CB |
                           UCT_EP_PARAM_FIELD_SOCKADDR_CB_FLAGS      |
                           UCT_EP_PARAM_FIELD_SOCKADDR_PACK_CB;

    ep_params.cm                         = cm;
    ep_params.conn_request               = conn_request;
    ep_params.sockaddr_cb_flags          = UCT_CB_FLAG_ASYNC;
    ep_params.sockaddr_pack_cb           = server_priv_data_cb;
    ep_params.sockaddr_connect_cb.server = connect_cb;
    ep_params.disconnect_cb              = disconnect_cb;
    ep_params.user_data                  = user_data;

    status = uct_ep_create(&ep_params, &ep);
    ASSERT_UCS_OK(status);
    m_eps.back().reset(ep, uct_ep_destroy);
}

void uct_test::entity::listen(const ucs::sock_addr_storage &listen_addr,
                              const uct_listener_params_t &params)
{
    ucs_status_t status;

    for (;;) {
        {
            scoped_log_handler slh(wrap_errors_logger);
            status = UCS_TEST_TRY_CREATE_HANDLE(uct_listener_h, m_listener,
                                                uct_listener_destroy,
                                                uct_listener_create, m_cm,
                                                listen_addr.get_sock_addr_ptr(),
                                                listen_addr.get_addr_size(),
                                                &params);
            if (status == UCS_OK) {
                break;
            }
        }
        EXPECT_EQ(UCS_ERR_BUSY, status);

        const struct sockaddr* c_ifa_addr = listen_addr.get_sock_addr_ptr();
        struct sockaddr* ifa_addr = const_cast<struct sockaddr*>(c_ifa_addr);
        if (ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in *addr =
                            reinterpret_cast<struct sockaddr_in *>(ifa_addr);
            addr->sin_port = ucs::get_port();
        } else {
            struct sockaddr_in6 *addr =
                            reinterpret_cast<struct sockaddr_in6 *>(ifa_addr);
            addr->sin6_port = ucs::get_port();
        }
    }
}

void uct_test::entity::disconnect(uct_ep_h ep) {
    ASSERT_UCS_OK(uct_ep_disconnect(ep, 0));
}

void uct_test::entity::flush() const {
    ucs_status_t status;
    do {
        progress();
        status = uct_iface_flush(m_iface, 0, NULL);
    } while (status == UCS_INPROGRESS);
    ASSERT_UCS_OK(status);
}

std::ostream& operator<<(std::ostream& os, const uct_tl_resource_desc_t& resource) {
    return os << resource.tl_name << "/" << resource.dev_name;
}

uct_test::mapped_buffer::mapped_buffer(size_t size, uint64_t seed,
                                       const entity& entity, size_t offset,
                                       ucs_memory_type_t mem_type) :
    m_entity(entity)
{
    if (size > 0)  {
        size_t alloc_size = size + offset;
        if (mem_type == UCS_MEMORY_TYPE_HOST) {
            m_entity.mem_alloc_host(alloc_size, &m_mem);
        } else {
            m_mem.method   = UCT_ALLOC_METHOD_LAST;
            m_mem.address  = mem_buffer::allocate(alloc_size, mem_type);
            m_mem.length   = alloc_size;
            m_mem.mem_type = mem_type;
            m_mem.memh     = UCT_MEM_HANDLE_NULL;
            m_mem.md       = NULL;
            m_entity.mem_type_reg(&m_mem);
        }
        m_buf = (char*)m_mem.address + offset;
        m_end = (char*)m_buf         + size;
        pattern_fill(seed);
    } else {
        m_mem.method  = UCT_ALLOC_METHOD_LAST;
        m_mem.address = NULL;
        m_mem.md      = NULL;
        m_mem.memh    = UCT_MEM_HANDLE_NULL;
        m_mem.mem_type= UCS_MEMORY_TYPE_HOST;
        m_mem.length  = 0;
        m_buf         = NULL;
        m_end         = NULL;
        m_rkey.rkey   = UCT_INVALID_RKEY;
        m_rkey.handle = NULL;
    }
    m_iov.buffer = ptr();
    m_iov.length = length();
    m_iov.count  = 1;
    m_iov.stride = 0;
    m_iov.memh   = memh();

    m_entity.rkey_unpack(&m_mem, &m_rkey);
    m_rkey.type  = NULL;
}

uct_test::mapped_buffer::~mapped_buffer() {
    m_entity.rkey_release(&m_rkey);
    if (m_mem.mem_type == UCS_MEMORY_TYPE_HOST) {
        m_entity.mem_free_host(&m_mem);
    } else {
        ucs_assert(m_mem.method == UCT_ALLOC_METHOD_LAST);
        m_entity.mem_type_dereg(&m_mem);
        mem_buffer::release(m_mem.address, m_mem.mem_type);
    }
}

void uct_test::mapped_buffer::pattern_fill(uint64_t seed) {
    mem_buffer::pattern_fill(ptr(), length(), seed, m_mem.mem_type);
}

void uct_test::mapped_buffer::pattern_check(uint64_t seed) {
    mem_buffer::pattern_check(ptr(), length(), seed, m_mem.mem_type);
}

void *uct_test::mapped_buffer::ptr() const {
    return m_buf;
}

uintptr_t uct_test::mapped_buffer::addr() const {
    return (uintptr_t)m_buf;
}

size_t uct_test::mapped_buffer::length() const {
    return (char*)m_end - (char*)m_buf;
}

uct_mem_h uct_test::mapped_buffer::memh() const {
    return m_mem.memh;
}

uct_rkey_t uct_test::mapped_buffer::rkey() const {
    return m_rkey.rkey;
}

const uct_iov_t*  uct_test::mapped_buffer::iov() const {
    return &m_iov;
}

size_t uct_test::mapped_buffer::pack(void *dest, void *arg) {
    const mapped_buffer* buf = (const mapped_buffer*)arg;
    mem_buffer::copy_from(dest, buf->ptr(), buf->length(), buf->m_mem.mem_type);
    return buf->length();
}

std::ostream& operator<<(std::ostream& os, const resource* resource) {
    return os << resource->name();
}

uct_test::entity::async_wrapper::async_wrapper()
{
    ucs_status_t status;

    /* Initialize context */
    status = ucs_async_context_init(&m_async, UCS_ASYNC_THREAD_LOCK_TYPE);
    if (UCS_OK != status) {
        fprintf(stderr, "Failed to init async context.\n");fflush(stderr);
    }
    ASSERT_UCS_OK(status);
}

uct_test::entity::async_wrapper::~async_wrapper()
{
    ucs_async_context_cleanup(&m_async);
}

void uct_test::entity::async_wrapper::check_miss()
{
    ucs_async_check_miss(&m_async);
}

ucs_status_t uct_test::send_am_message(entity *e, uint8_t am_id, int ep_idx)
{
    ssize_t res;

    if (is_caps_supported(UCT_IFACE_FLAG_AM_SHORT)) {
        return uct_ep_am_short(e->ep(ep_idx), am_id, 0, NULL, 0);
    } else {
        res = uct_ep_am_bcopy(e->ep(ep_idx), am_id,
                              (uct_pack_callback_t)ucs_empty_function_return_zero_int64,
                              NULL, 0);
        return (ucs_status_t)(res >= 0 ? UCS_OK : res);
    }
}
