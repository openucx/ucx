/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <uct/uct_test.h>

#include <set>

extern "C" {
#include <uct/api/uct.h>
#include <uct/tcp/tcp.h>
#include <ucs/sys/netlink.h>
#include <ucs/sys/sys.h>
}

class test_uct_tcp : public uct_test {
public:
    void init() {
        if (RUNNING_ON_VALGRIND) {
            modify_config("TCP_TX_SEG_SIZE", "1kb");
            modify_config("TCP_RX_SEG_SIZE", "1kb");
        }

        uct_test::init();
        m_ent = uct_test::create_entity(0);
        m_entities.push_back(m_ent);
        m_tcp_iface = (uct_tcp_iface*)m_ent->iface();
    }

    static int get_maxseg(int fd) {
        int maxseg    = 0;
        socklen_t len = sizeof(maxseg);

        EXPECT_EQ(0, getsockopt(fd, IPPROTO_TCP, TCP_MAXSEG, &maxseg, &len))
                << strerror(errno);
        return maxseg;
    }

    size_t get_accepted_conn_num(entity& ent) {
        size_t num = 0;
        uct_tcp_ep_t *ep;

        // go through all EPs on iface with lock held, since EPs are created
        // and inserted to the EP list from async thread during accepting a
        // connection
        UCS_ASYNC_BLOCK(m_tcp_iface->super.worker->async);
        ucs_list_for_each(ep, &m_tcp_iface->ep_list, list) {
            num += (ep->conn_state == UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER);
        }
        UCS_ASYNC_UNBLOCK(m_tcp_iface->super.worker->async);

        return num;
    }

    ucs_status_t post_recv(int fd, bool nb = false) {
        uint8_t msg;
        size_t msg_size = sizeof(msg);
        ucs_status_t status;

        scoped_log_handler slh(wrap_errors_logger);
        if (nb) {
            status = ucs_socket_recv_nb(fd, &msg, 0, &msg_size);
        } else {
            status = ucs_socket_recv(fd, &msg, msg_size);
        }

        return status;
    }

    void post_send(int fd, const std::vector<char> &buf) {
        scoped_log_handler slh(wrap_errors_logger);
        ucs_status_t status = ucs_socket_send(fd, &buf[0], buf.size());
        // send can be OK or fail when a connection was closed by a peer
        // before all data were sent
        ASSERT_TRUE((status == UCS_OK) ||
                    (status == UCS_ERR_IO_ERROR));
    }

    void detect_conn_reset(int fd) {
        // Try to receive something on this socket fd - it has to be failed
        ucs_status_t status = post_recv(fd);
        ASSERT_TRUE(status == UCS_ERR_CONNECTION_RESET);
        EXPECT_EQ(0, ucs_socket_is_connected(fd));
    }

    void test_listener_flood(entity& test_entity, size_t max_conn,
                             size_t msg_size) {
        std::vector<int> fds;
        std::vector<char> buf;

        if (msg_size > 0) {
            buf.resize(msg_size + sizeof(uct_tcp_am_hdr_t));
            std::fill(buf.begin(), buf.end(), 0);
            init_data(&buf[0], buf.size());
        }

        setup_conns_to_entity(test_entity, max_conn, fds);

        size_t handled = 0;
        for (std::vector<int>::const_iterator iter = fds.begin();
             iter != fds.end(); ++iter) {
            size_t sent_length = 0;
            do {
                if (msg_size > 0) {
                    post_send(*iter, buf);
                    sent_length += buf.size();
                } else {
                    close(*iter);
                }

                // If it was sent >= the length of the magic number or sending
                // is not required by the current test, wait until connection
                // is destroyed. Otherwise, need to send more data
                if ((msg_size == 0) || (sent_length >= sizeof(uint64_t))) {
                    handled++;

                    while (get_accepted_conn_num(test_entity) != (max_conn - handled)) {
                        sched_yield();
                        progress();
                    }
                } else {
                    // Peers still have to be connected
                    ucs_status_t status = post_recv(*iter, true);
                    EXPECT_TRUE((status == UCS_OK) ||
                                (status == UCS_ERR_NO_PROGRESS));
                    EXPECT_EQ(1, ucs_socket_is_connected(*iter));
                }
            } while ((msg_size != 0) && (sent_length < sizeof(uint64_t)));
        }

        // give a chance to close all connections
        while (!ucs_list_is_empty(&m_tcp_iface->ep_list)) {
            sched_yield();
            progress();
        }

        // TCP has to reject all connections and forget EPs that were
        // created after accept():
        // - EP list has to be empty
        EXPECT_EQ(1, ucs_list_is_empty(&m_tcp_iface->ep_list));
        // - all connections have to be destroyed (if wasn't closed
        //   yet by the clients)
        if (msg_size > 0) {
            // if we sent data during the test, close socket fd here
            while (!fds.empty()) {
                int fd = fds.back();
                fds.pop_back();
                detect_conn_reset(fd);
                close(fd);
            }
        }
    }

    void setup_conns_to_entity(entity& to, size_t max_conn,
                               std::vector<int> &fds) {
        for (size_t i = 0; i < max_conn; i++) {
            int fd = setup_conn_to_entity(to, i + 1lu);
            fds.push_back(fd);

            // give a chance to finish all connections
            while (get_accepted_conn_num(to) != (i + 1lu)) {
                sched_yield();
                progress();
            }

            EXPECT_EQ(1, ucs_socket_is_connected(fd));
        }
    }

private:
    void init_data(void *buf, size_t msg_size) {
        uct_tcp_am_hdr_t *tcp_am_hdr;
        ASSERT_TRUE(msg_size >= sizeof(*tcp_am_hdr));
        tcp_am_hdr         = static_cast<uct_tcp_am_hdr_t*>(buf);
        tcp_am_hdr->am_id  = std::numeric_limits<uint8_t>::max();
        tcp_am_hdr->length = msg_size;
    }

    int connect_to_entity(entity& to) {
        uct_device_addr_t *dev_addr;
        uct_iface_addr_t *iface_addr;
        ucs_status_t status;

        if (to.tcp_vrf_drops_loopback()) {
            UCS_TEST_SKIP_R("same-host traffic in VRF is dropped");
        }

        dev_addr   = (uct_device_addr_t*)malloc(to.iface_attr().device_addr_len);
        iface_addr = (uct_iface_addr_t*)malloc(to.iface_attr().iface_addr_len);

        status = uct_iface_get_device_address(to.iface(), dev_addr);
        ASSERT_UCS_OK(status);

        status = uct_iface_get_address(to.iface(), iface_addr);
        ASSERT_UCS_OK(status);

        struct sockaddr_storage dest_addr;
        uct_tcp_ep_set_dest_addr(dev_addr, iface_addr,
                                 (struct sockaddr*)&dest_addr);

        int fd;
        EXPECT_TRUE((dest_addr.ss_family == AF_INET) ||
                    (dest_addr.ss_family == AF_INET6));
        status = ucs_socket_create(dest_addr.ss_family, SOCK_STREAM, 0, &fd);
        ASSERT_UCS_OK(status);

        // Addresses in a VRF are reachable only from sockets bound to the VRF
        const char *vrf_name = uct_tcp_iface_vrf_name(
                (uct_tcp_iface_t*)to.iface());
        if (vrf_name != NULL) {
            status = ucs_socket_bind_to_device(fd, vrf_name);
            ASSERT_UCS_OK(status);
        }

        status = ucs_socket_connect(fd, (struct sockaddr*)&dest_addr);
        ASSERT_UCS_OK(status);

        status = ucs_sys_fcntl_modfl(fd, O_NONBLOCK, 0);
        ASSERT_UCS_OK(status);

        free(iface_addr);
        free(dev_addr);

        return fd;
    }

    int setup_conn_to_entity(entity &to, size_t sn = 1) {
        int fd = -1;

        do {
            if (fd != -1) {
                close(fd);
            }

            fd = connect_to_entity(to);
            EXPECT_NE(-1, fd);

            // give a chance to finish the connection
            while (get_accepted_conn_num(to) != sn) {
                sched_yield();
                progress();

                ucs_status_t status = post_recv(fd, true);
                if ((status != UCS_OK) &&
                    (status != UCS_ERR_NO_PROGRESS)) {
                    break;
                }
            }
        } while (!ucs_socket_is_connected(fd));

        EXPECT_EQ(1, ucs_socket_is_connected(fd));

        return fd;
    }

protected:
    ucs_status_t open_md(const char *vrf_enable, uct_md_h *md_p)
    {
        uct_md_config_t *md_config;
        ucs_status_t status;

        status = uct_md_config_read(GetParam()->component, NULL, NULL,
                                    &md_config);
        if (status != UCS_OK) {
            return status;
        }

        status = uct_config_modify(md_config, "TCP_VRF_ENABLE", vrf_enable);
        if (status == UCS_OK) {
            status = uct_md_open(GetParam()->component,
                                 GetParam()->md_name.c_str(), md_config, md_p);
        }

        uct_config_release(md_config);
        return status;
    }

    void query_devices(const char *vrf_enable, std::set<std::string> &devices)
    {
        uct_tl_resource_desc_t *resources;
        unsigned num_resources;
        uct_md_h md;

        ASSERT_UCS_OK(open_md(vrf_enable, &md));

        ucs_status_t status = uct_md_query_tl_resources(md, &resources,
                                                        &num_resources);
        if (status == UCS_OK) {
            for (unsigned i = 0; i < num_resources; ++i) {
                devices.insert(resources[i].dev_name);
            }
            uct_release_tl_resource_list(resources);
        }
        uct_md_close(md);
        ASSERT_UCS_OK(status);
    }

    static int vrf_master_index(const std::string &dev_name)
    {
        return ucs_netlink_get_vrf_master_index(
                if_nametoindex(dev_name.c_str()));
    }

    uct_tcp_iface *m_tcp_iface;
    entity        *m_ent;
};

UCS_TEST_P(test_uct_tcp, listener_flood_connect_and_send_large) {
    const size_t max_conn =
        ucs_min(static_cast<size_t>(max_connections()), 128lu) /
        ucs::test_time_multiplier();
    const size_t msg_size = m_tcp_iface->config.rx_seg_size * 4;
    test_listener_flood(*m_ent, max_conn, msg_size);
}

UCS_TEST_P(test_uct_tcp, listener_flood_connect_and_send_small) {
    const size_t max_conn =
        ucs_min(static_cast<size_t>(max_connections()), 128lu) /
        ucs::test_time_multiplier();
    // It should be less than length of the expected magic number by TCP
    const size_t msg_size = 1;
    test_listener_flood(*m_ent, max_conn, msg_size);
}

UCS_TEST_P(test_uct_tcp, listener_flood_connect_and_close) {
    const size_t max_conn =
        ucs_min(static_cast<size_t>(max_connections()), 128lu) /
        ucs::test_time_multiplier();
    test_listener_flood(*m_ent, max_conn, 0);
}

UCS_TEST_P(test_uct_tcp, reject_oversized_am)
{
    constexpr uint64_t magic_number = UCT_TCP_MAGIC_NUMBER;
    const uct_tcp_am_hdr_t hdr      = {
        0, std::numeric_limits<uint32_t>::max()
    };
    std::vector<char> buf(sizeof(magic_number) + sizeof(hdr));
    std::vector<int> fds;

    memcpy(&buf[0], &magic_number, sizeof(magic_number));
    memcpy(&buf[sizeof(magic_number)], &hdr, sizeof(hdr));

    setup_conns_to_entity(*m_ent, 1, fds);
    post_send(fds.front(), buf);

    wait_for_cond([this]() {
                      return ucs_list_is_empty(&m_tcp_iface->ep_list);
                  },
                  [this]() { progress(); }, 1.0);
    EXPECT_TRUE(ucs_list_is_empty(&m_tcp_iface->ep_list));

    close(fds.front());
}

UCS_TEST_P(test_uct_tcp, check_addr_len)
{
    uct_iface_attr_t iface_attr;

    ucs_status_t status = uct_iface_query(m_ent->iface(), &iface_attr);
    ASSERT_UCS_OK(status);

    UCS_TEST_MESSAGE << m_ent->md()->component->name;
    if (!GetParam()->dev_name.compare("lo")) {
        EXPECT_EQ(sizeof(uct_tcp_device_addr_t) +
                          sizeof(uct_iface_local_addr_ns_t),
                  iface_attr.device_addr_len);
    } else {
        struct sockaddr *saddr = reinterpret_cast<struct sockaddr*>(
                                         &m_tcp_iface->config.ifaddr);
        size_t in_addr_len;
        status = ucs_sockaddr_inet_addr_sizeof(saddr, &in_addr_len);
        ASSERT_UCS_OK(status);

        EXPECT_EQ(sizeof(uct_tcp_device_addr_t) + in_addr_len,
                  iface_attr.device_addr_len);
    }

    EXPECT_EQ(sizeof(uct_tcp_iface_addr_t), iface_attr.iface_addr_len);
    EXPECT_EQ(sizeof(uct_tcp_ep_addr_t), iface_attr.ep_addr_len);
}

UCS_TEST_P(test_uct_tcp, vrf_enable)
{
    std::set<std::string> devices, vrf_devices, yes_devices;

    query_devices("n", devices);
    query_devices("try", vrf_devices);

    // With "y", a device whose sockets cannot be bound to its VRF fails the
    // tcp device query instead of being skipped. uct_md_query_tl_resources()
    // skips a transport whose query fails, so it returns UCS_OK and no devices.
    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        query_devices("y", yes_devices);
    }
    if (m_errors.empty()) {
        EXPECT_EQ(vrf_devices, yes_devices);
    } else {
        EXPECT_TRUE(yes_devices.empty());
    }

    // Devices in a VRF are not reported when VRF_ENABLE is "n"
    for (const auto &dev_name : devices) {
        EXPECT_EQ(-1, vrf_master_index(dev_name)) << dev_name;
        EXPECT_EQ(1u, vrf_devices.count(dev_name)) << dev_name;
    }

    for (const auto &dev_name : vrf_devices) {
        if (devices.count(dev_name) == 0) {
            EXPECT_NE(-1, vrf_master_index(dev_name)) << dev_name;
            UCS_TEST_MESSAGE << dev_name << " is reported only with VRF_ENABLE";
        }
    }
}

UCS_TEST_P(test_uct_tcp, vrf_disabled_iface_open)
{
    uct_iface_params_t params = {};
    ucs_status_t status;
    uct_iface_h iface;
    uct_md_h md;

    if (uct_tcp_iface_vrf_name(m_tcp_iface) == NULL) {
        UCS_TEST_SKIP_R("the device is not in a VRF");
    }

    ASSERT_UCS_OK(open_md("n", &md));

    params.field_mask           = UCT_IFACE_PARAM_FIELD_OPEN_MODE |
                                  UCT_IFACE_PARAM_FIELD_DEVICE;
    params.open_mode            = UCT_IFACE_OPEN_MODE_DEVICE;
    params.mode.device.tl_name  = GetParam()->tl_name.c_str();
    params.mode.device.dev_name = GetParam()->dev_name.c_str();

    // A device in a VRF cannot be opened when VRF_ENABLE is "n"
    {
        scoped_log_handler wrap_err(wrap_errors_logger);
        status = uct_iface_open(md, m_ent->worker(), &params, m_iface_config,
                                &iface);
    }
    EXPECT_EQ(UCS_ERR_UNSUPPORTED, status);

    if (status == UCS_OK) {
        uct_iface_close(iface);
    }
    uct_md_close(md);
}

UCS_TEST_P(test_uct_tcp, vrf_bind_to_device)
{
    const char *iface_vrf_name   = uct_tcp_iface_vrf_name(m_tcp_iface);
    int vrf_index                = vrf_master_index(m_tcp_iface->if_name);
    char bound_if_name[IFNAMSIZ] = {};
    socklen_t len                = sizeof(bound_if_name);
    char vrf_name[IFNAMSIZ]      = {};

    if (vrf_index != -1) {
        ASSERT_NE(nullptr, if_indextoname(vrf_index, vrf_name));
    }
    EXPECT_STREQ(vrf_name, (iface_vrf_name != NULL) ? iface_vrf_name : "");

    // The listener must be bound to the VRF device, if any
    ASSERT_EQ(0, getsockopt(m_tcp_iface->listen_fd, SOL_SOCKET,
                            SO_BINDTODEVICE, bound_if_name, &len));
    EXPECT_STREQ(vrf_name, bound_if_name);
}

UCS_TEST_P(test_uct_tcp, vrf_ipv6_local_reachable)
{
    const char *vrf_name = uct_tcp_iface_vrf_name(m_tcp_iface);
    std::vector<char> dev_addr(m_ent->iface_attr().device_addr_len);
    std::vector<char> iface_addr(m_ent->iface_attr().iface_addr_len);
    uct_iface_is_reachable_params_t params;
    long vrf_ipv6_disabled = 0;
    bool local_reachable;
    char info_str[256];

    // Same-host traffic in a VRF is looped back through the VRF device, which
    // drops IPv6 packets if IPv6 is disabled on it. The sysctl file is missing
    // if IPv6 is disabled in the kernel.
    if (vrf_name != NULL) {
        (void)ucs_read_file_number(&vrf_ipv6_disabled, 1,
                                   "/proc/sys/net/ipv6/conf/%s/disable_ipv6",
                                   vrf_name);
    }
    EXPECT_EQ(!vrf_ipv6_disabled,
              !!m_tcp_iface->config.vrf_ipv6_local_reachable);

    ASSERT_UCS_OK(uct_iface_get_device_address(
            m_ent->iface(), (uct_device_addr_t*)&dev_addr[0]));
    ASSERT_UCS_OK(uct_iface_get_address(m_ent->iface(),
                                        (uct_iface_addr_t*)&iface_addr[0]));

    params.field_mask         = UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR |
                                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR |
                                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING |
                                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH |
                                UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR_LENGTH |
                                UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR_LENGTH;
    params.device_addr        = (uct_device_addr_t*)&dev_addr[0];
    params.device_addr_length = dev_addr.size();
    params.iface_addr         = (uct_iface_addr_t*)&iface_addr[0];
    params.iface_addr_length  = iface_addr.size();
    params.info_string        = info_str;
    params.info_string_length = sizeof(info_str);

    // The own IPv6 address is unreachable if same-host IPv6 traffic in the VRF
    // is dropped
    local_reachable = !m_ent->tcp_vrf_drops_loopback();
    EXPECT_EQ(local_reachable,
              !!uct_iface_is_reachable_v2(m_ent->iface(), &params));
    if (!local_reachable) {
        EXPECT_NE(std::string::npos,
                  std::string(info_str).find("is a local address of VRF"))
                << info_str;
    }
}


UCS_TEST_P(test_uct_tcp, maxseg_auto)
{
    // Room for the IPv6 and TCP headers with options
    static const int hdrs_len = 120;
    static const int limit    = 1000;
    std::vector<int> fds;
    struct ifreq ifr;

    ASSERT_UCS_OK(ucs_netif_ioctl(GetParam()->dev_name.c_str(), SIOCGIFMTU,
                                  &ifr));
    if (ifr.ifr_mtu - hdrs_len <= limit) {
        UCS_TEST_SKIP_R("device MTU " + ucs::to_string(ifr.ifr_mtu) +
                        " is too small");
    }

    // No TCP_MAXSEG is set by default, so the segment size follows the MTU
    setup_conns_to_entity(*m_ent, 1, fds);
    EXPECT_GT(get_maxseg(fds.front()), limit);
    close(fds.front());
}

_UCT_INSTANTIATE_TEST_CASE(test_uct_tcp, tcp)


class test_uct_tcp_conn_nb : public test_uct_tcp {
public:
    void init() {
        // Do not wait for a connection, which can be dropped in a VRF
        modify_config("TCP_CONN_NB", "y");
        test_uct_tcp::init();
    }
};

UCS_TEST_P(test_uct_tcp_conn_nb, vrf_ep_bind_to_device)
{
    const char *vrf_name         = uct_tcp_iface_vrf_name(m_tcp_iface);
    std::vector<char> dev_addr(m_ent->iface_attr().device_addr_len);
    std::vector<char> iface_addr(m_ent->iface_attr().iface_addr_len);
    char bound_if_name[IFNAMSIZ] = {};
    socklen_t len                = sizeof(bound_if_name);
    uct_ep_params_t ep_params;
    uct_ep_h ep;

    ASSERT_UCS_OK(uct_iface_get_device_address(
            m_ent->iface(), (uct_device_addr_t*)&dev_addr[0]));
    ASSERT_UCS_OK(uct_iface_get_address(m_ent->iface(),
                                        (uct_iface_addr_t*)&iface_addr[0]));

    ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE |
                           UCT_EP_PARAM_FIELD_DEV_ADDR |
                           UCT_EP_PARAM_FIELD_IFACE_ADDR;
    ep_params.iface      = m_ent->iface();
    ep_params.dev_addr   = (uct_device_addr_t*)&dev_addr[0];
    ep_params.iface_addr = (uct_iface_addr_t*)&iface_addr[0];
    ASSERT_UCS_OK(uct_ep_create(&ep_params, &ep));

    // The connecting socket must be bound to the VRF device, if any
    EXPECT_EQ(0, getsockopt(ucs_derived_of(ep, uct_tcp_ep_t)->fd, SOL_SOCKET,
                            SO_BINDTODEVICE, bound_if_name, &len));
    EXPECT_STREQ((vrf_name != NULL) ? vrf_name : "", bound_if_name);

    uct_ep_destroy(ep);
}

_UCT_INSTANTIATE_TEST_CASE(test_uct_tcp_conn_nb, tcp)


class test_uct_tcp_maxseg : public test_uct_tcp {
public:
    void init() {
        modify_config("TCP_MAXSEG", ucs::to_string(MAXSEG));
        test_uct_tcp::init();
    }

protected:
    class scoped_fd {
    public:
        scoped_fd() : m_fd(-1) {}
        scoped_fd(const scoped_fd&) = delete;
        scoped_fd &operator=(const scoped_fd&) = delete;

        ~scoped_fd() {
            if (m_fd >= 0) {
                close(m_fd);
            }
        }

        int get() const {
            return m_fd;
        }

        int *ptr() {
            return &m_fd;
        }

    private:
        int m_fd;
    };

    static constexpr int MAXSEG = 1000;
};

constexpr int test_uct_tcp_maxseg::MAXSEG;

UCS_TEST_P(test_uct_tcp_maxseg, set_sockopt)
{
    struct sockaddr_in sin = {};
    socklen_t addr_len     = sizeof(sin);

    {
        scoped_fd listen_fd, fd;

        // Older kernels report the segment size only for a connected socket
        sin.sin_family      = AF_INET;
        sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        ASSERT_UCS_OK(ucs_socket_create(AF_INET, SOCK_STREAM, 0,
                                        listen_fd.ptr()));
        ASSERT_EQ(0, bind(listen_fd.get(), (struct sockaddr*)&sin,
                          sizeof(sin)));
        ASSERT_EQ(0, listen(listen_fd.get(), 1));
        ASSERT_EQ(0, getsockname(listen_fd.get(), (struct sockaddr*)&sin,
                                 &addr_len));

        // A connecting socket limits the segment size of its connection
        ASSERT_UCS_OK(ucs_socket_create(AF_INET, SOCK_STREAM, 0, fd.ptr()));
        ASSERT_UCS_OK(uct_tcp_iface_set_sockopt(m_tcp_iface, fd.get(), 0));
        ASSERT_UCS_OK(ucs_socket_connect(fd.get(), (struct sockaddr*)&sin));
        EXPECT_LE(get_maxseg(fd.get()), MAXSEG);
    }

    // The listener advertises the segment size to a peer, which does not
    // limit it itself, and the accepted socket gets it from the listener
    std::vector<int> fds;
    setup_conns_to_entity(*m_ent, 1, fds);
    EXPECT_LE(get_maxseg(fds.front()), MAXSEG);
    {
        uct_tcp_ep_t *ep;

        UCS_ASYNC_BLOCK(m_tcp_iface->super.worker->async);
        EXPECT_FALSE(ucs_list_is_empty(&m_tcp_iface->ep_list));
        ucs_list_for_each(ep, &m_tcp_iface->ep_list, list) {
            EXPECT_LE(get_maxseg(ep->fd), MAXSEG);
        }
        UCS_ASYNC_UNBLOCK(m_tcp_iface->super.worker->async);
    }
    close(fds.front());
}

_UCT_INSTANTIATE_TEST_CASE(test_uct_tcp_maxseg, tcp)


class test_uct_tcp_base : public ucs::test {
};

UCS_TEST_F(test_uct_tcp_base, check_maxseg)
{
    EXPECT_UCS_OK(ucs_tcp_base_check_maxseg(UCS_MEMUNITS_AUTO));
    EXPECT_UCS_OK(ucs_tcp_base_check_maxseg(88));
    EXPECT_UCS_OK(ucs_tcp_base_check_maxseg(32767));

    scoped_log_handler wrap_err(wrap_errors_logger);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, ucs_tcp_base_check_maxseg(87));
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, ucs_tcp_base_check_maxseg(32768));
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucs_tcp_base_check_maxseg(UCS_MEMUNITS_INF));
}
