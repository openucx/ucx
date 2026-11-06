/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/netlink.h>
}

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <libgen.h>
#include <limits.h>
#include <linux/rtnetlink.h>
#include <unistd.h>
#include <vector>


class test_netlink : public ucs::test {
protected:
    static std::string sysfs_master_name(const char *if_name)
    {
        std::string path = std::string("/sys/class/net/") + if_name +
                           "/master";
        char target[PATH_MAX];
        ssize_t len;

        len = readlink(path.c_str(), target, sizeof(target) - 1);
        if (len < 0) {
            return "";
        }

        target[len] = '\0';
        return basename(target);
    }

    static void append_nlmsg(std::vector<uint8_t> &stream, unsigned short type,
                             const void *hdr, size_t hdr_size)
    {
        size_t msg_len      = NLMSG_LENGTH(hdr_size);
        size_t offset       = stream.size();
        struct nlmsghdr nlh = {static_cast<uint32_t>(msg_len), type,
                               NLM_F_MULTI, 0, 0};

        stream.resize(offset + NLMSG_ALIGN(msg_len), 0);
        memcpy(&stream[offset], &nlh, sizeof(nlh));
        memcpy(&stream[offset + NLMSG_HDRLEN], hdr, hdr_size);
    }

    static ucs_status_t count_msg_cb(const struct nlmsghdr*, void *arg)
    {
        ++(*static_cast<int*>(arg));
        return UCS_INPROGRESS;
    }
};

UCS_TEST_F(test_netlink, get_vrf_master_index) {
    struct if_nameindex *ifs = if_nameindex();
    unsigned num_vrf_slaves  = 0;
    unsigned num_vrfs        = 0;
    char master_name[IFNAMSIZ];

    ASSERT_NE(nullptr, ifs);

    for (struct if_nameindex *it = ifs; it->if_index != 0; ++it) {
        int master_index = ucs_netlink_get_vrf_master_index(it->if_index);
        if (master_index == -1) {
            continue;
        }

        ASSERT_NE(nullptr, if_indextoname(master_index, master_name))
                << it->if_name;

        /* A VRF device is the master of itself */
        if (master_index == (int)it->if_index) {
            ++num_vrfs;
            continue;
        }

        /* A VRF slave must be linked to its master in sysfs */
        EXPECT_EQ(sysfs_master_name(it->if_name), master_name) << it->if_name;
        EXPECT_EQ(master_index, ucs_netlink_get_vrf_master_index(master_index))
                << it->if_name;
        ++num_vrf_slaves;
    }

    if_freenameindex(ifs);

    EXPECT_EQ(-1, ucs_netlink_get_vrf_master_index(if_nametoindex("lo")));
    UCS_TEST_MESSAGE << num_vrfs << " VRF devices and " << num_vrf_slaves
                     << " VRF slave interfaces found";
}

UCS_TEST_F(test_netlink, is_local_addr) {
    unsigned lo_index      = if_nametoindex("lo");
    unsigned num_vrf_addrs = 0;
    struct sockaddr_in sin = {};
    struct ifaddrs *ifaddrs, *ifa;
    int local_index;

    ASSERT_NE(0u, lo_index);
    ASSERT_EQ(0, getifaddrs(&ifaddrs));

    for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
        if ((ifa->ifa_addr == NULL) ||
            ((ifa->ifa_addr->sa_family != AF_INET) &&
             (ifa->ifa_addr->sa_family != AF_INET6))) {
            continue;
        }

        /* Tentative addresses and addresses of down interfaces have no local
         * route */
        local_index = ucs_netlink_get_local_route_ndev_index(ifa->ifa_addr);
        if (local_index < 0) {
            continue;
        }

        /* An address is local in the routing domain of its local route */
        EXPECT_EQ(1, ucs_netlink_is_local_addr(local_index, ifa->ifa_addr))
                << ifa->ifa_name;
        if (ucs_netlink_get_vrf_master_index(local_index) == -1) {
            continue;
        }

        /* VRF addresses are not local outside of the VRF */
        EXPECT_EQ(0, ucs_netlink_is_local_addr(lo_index, ifa->ifa_addr))
                << ifa->ifa_name;
        ++num_vrf_addrs;
    }

    freeifaddrs(ifaddrs);

    sin.sin_family      = AF_INET;
    sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    EXPECT_EQ(1, ucs_netlink_is_local_addr(lo_index, (struct sockaddr*)&sin));

    /* TEST-NET-2 address is not assigned to any host */
    ASSERT_EQ(1, inet_pton(AF_INET, "198.51.100.1", &sin.sin_addr));
    EXPECT_EQ(0, ucs_netlink_is_local_addr(lo_index, (struct sockaddr*)&sin));

    UCS_TEST_MESSAGE << num_vrf_addrs << " VRF addresses found";
}

UCS_TEST_F(test_netlink, best_route_to_local_addr) {
    struct if_nameindex *ifs = if_nameindex();
    unsigned num_checks      = 0;
    struct ifaddrs *ifaddrs, *ifa;
    struct if_nameindex *it;
    int local_index;

    ASSERT_NE(nullptr, ifs);
    ASSERT_EQ(0, getifaddrs(&ifaddrs));

    for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
        if ((ifa->ifa_addr == NULL) ||
            ((ifa->ifa_addr->sa_family != AF_INET) &&
             (ifa->ifa_addr->sa_family != AF_INET6))) {
            continue;
        }

        local_index = ucs_netlink_get_local_route_ndev_index(ifa->ifa_addr);
        if (local_index < 0) {
            continue;
        }

        /* A local address is reached through the interface of its local
         * route */
        EXPECT_EQ(1, ucs_netlink_route_matches(local_index, ifa->ifa_addr,
                                               UCS_NETLINK_ROUTE_CHECK_BEST))
                << ifa->ifa_name;

        /* Other routing domains route it out of the host */
        for (it = ifs; it->if_index != 0; ++it) {
            if (ucs_netlink_get_vrf_master_index(it->if_index) ==
                ucs_netlink_get_vrf_master_index(local_index)) {
                continue;
            }

            EXPECT_EQ(0, ucs_netlink_route_matches(
                                 it->if_index, ifa->ifa_addr,
                                 UCS_NETLINK_ROUTE_CHECK_BEST))
                    << ifa->ifa_name << " from " << it->if_name;
            ++num_checks;
        }
    }

    freeifaddrs(ifaddrs);
    if_freenameindex(ifs);
    UCS_TEST_MESSAGE << num_checks << " routes from other domains checked";
}

UCS_TEST_F(test_netlink, parse_msg) {
    auto fail_cb = [](const struct nlmsghdr*, void *arg) -> ucs_status_t {
        ++(*static_cast<int*>(arg));
        return UCS_ERR_NO_MEMORY;
    };
    const int done_ok     = 0;
    const int done_error  = -EINVAL;
    struct ifinfomsg ifi  = {};
    struct nlmsgerr nlerr = {-ENODEV};
    std::vector<uint8_t> dump, reply;
    int count, msg_done;
    size_t done_offset;
    struct nlmsghdr *nlh;

    /* A datagram in the middle of a dump */
    append_nlmsg(dump, RTM_NEWLINK, &ifi, sizeof(ifi));
    append_nlmsg(dump, RTM_NEWLINK, &ifi, sizeof(ifi));
    count = 0;
    ASSERT_UCS_OK(ucs_netlink_parse_msg(dump.data(), dump.size(), count_msg_cb,
                                        &count, UCS_LOG_LEVEL_ERROR,
                                        &msg_done));
    EXPECT_EQ(2, count);
    EXPECT_EQ(0, msg_done);

    /* NLMSG_DONE may follow the last messages of a dump in the same datagram */
    done_offset = dump.size();
    append_nlmsg(dump, NLMSG_DONE, &done_ok, sizeof(done_ok));
    count = 0;
    ASSERT_UCS_OK(ucs_netlink_parse_msg(dump.data(), dump.size(), count_msg_cb,
                                        &count, UCS_LOG_LEVEL_ERROR,
                                        &msg_done));
    EXPECT_EQ(2, count);
    EXPECT_EQ(1, msg_done);

    /* A parser error stops the parsing */
    count = 0;
    EXPECT_EQ(UCS_ERR_NO_MEMORY,
              ucs_netlink_parse_msg(dump.data(), dump.size(), fail_cb, &count,
                                    UCS_LOG_LEVEL_ERROR, &msg_done));
    EXPECT_EQ(1, count);

    /* The kernel marks the messages of an inconsistent dump, including
     * NLMSG_DONE, with NLM_F_DUMP_INTR */
    for (size_t offset : {size_t(0), done_offset}) {
        nlh = reinterpret_cast<struct nlmsghdr*>(&dump[offset]);
        nlh->nlmsg_flags |= NLM_F_DUMP_INTR;
        count = 0;
        EXPECT_EQ(UCS_ERR_BUSY,
                  ucs_netlink_parse_msg(dump.data(), dump.size(), count_msg_cb,
                                        &count, UCS_LOG_LEVEL_ERROR,
                                        &msg_done));
        nlh->nlmsg_flags &= ~NLM_F_DUMP_INTR;
        EXPECT_EQ(2, count);
        EXPECT_EQ(1, msg_done);
    }

    scoped_log_handler slh(hide_errors_logger);

    /* An error reply to a request is logged with the requested log level */
    append_nlmsg(reply, NLMSG_ERROR, &nlerr, sizeof(nlerr));
    count = 0;
    EXPECT_EQ(UCS_ERR_IO_ERROR,
              ucs_netlink_parse_msg(reply.data(), reply.size(), count_msg_cb,
                                    &count, UCS_LOG_LEVEL_DEBUG, &msg_done));
    EXPECT_EQ(0, count);
    EXPECT_EQ(0, msg_done);
    EXPECT_TRUE(m_errors.empty());

    /* A dump which failed in the kernel ends with an error in NLMSG_DONE */
    dump.resize(done_offset);
    append_nlmsg(dump, NLMSG_DONE, &done_error, sizeof(done_error));
    count = 0;
    EXPECT_EQ(UCS_ERR_IO_ERROR,
              ucs_netlink_parse_msg(dump.data(), dump.size(), count_msg_cb,
                                    &count, UCS_LOG_LEVEL_ERROR, &msg_done));
    EXPECT_EQ(2, count);
    EXPECT_EQ(1, msg_done);
    EXPECT_EQ(1u, m_errors.size());
}

UCS_TEST_F(test_netlink, send_request) {
    struct ifinfomsg ifi = {};
    int count            = 0;

    /* Every network namespace has a loopback device */
    ASSERT_UCS_OK_OR_BUSY(
            ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETLINK, NLM_F_DUMP,
                                     &ifi, sizeof(ifi), count_msg_cb, &count,
                                     UCS_LOG_LEVEL_ERROR));
    EXPECT_GE(count, 1);

    /* The kernel replies with ENODEV to a query of a missing device */
    ifi.ifi_index = INT_MAX;
    count         = 0;
    EXPECT_EQ(UCS_ERR_IO_ERROR,
              ucs_netlink_send_request(NETLINK_ROUTE, RTM_GETLINK, 0, &ifi,
                                       sizeof(ifi), count_msg_cb, &count,
                                       UCS_LOG_LEVEL_DEBUG));
    EXPECT_EQ(0, count);
}
