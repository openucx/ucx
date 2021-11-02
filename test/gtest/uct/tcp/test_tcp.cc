/**
 * Copyright (C) Mellanox Technologies Ltd. 2020. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <uct/uct_test.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/tcp/tcp.h>
}

class test_uct_tcp : public uct_test {
public:
    void init() {
        if (RUNNING_ON_VALGRIND) {
            modify_config("TX_SEG_SIZE", "1kb");
            modify_config("RX_SEG_SIZE", "1kb");
        }

        uct_test::init();
        m_ent = uct_test::create_entity(0);
        m_entities.push_back(m_ent);
        m_tcp_iface = (uct_tcp_iface*)m_ent->iface();
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
            status = ucs_socket_recv_nb(fd, &msg, &msg_size);
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
        status = ucs_socket_create(dest_addr.ss_family, SOCK_STREAM, &fd);
        ASSERT_UCS_OK(status);

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


_UCT_INSTANTIATE_TEST_CASE(test_uct_tcp, tcp)
