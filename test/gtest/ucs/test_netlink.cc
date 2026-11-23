/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/config/global_opts.h>
#include <ucs/sys/netlink.h>
}

#include <algorithm>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <net/route.h>
#include <libgen.h>
#include <limits.h>
#include <linux/rtnetlink.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <thread>
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

UCS_TEST_F(test_netlink, route_get) {
    unsigned lo_index      = if_nametoindex("lo");
    unsigned num_addrs     = 0;
    struct sockaddr_in sin = {};
    struct ifaddrs *ifaddrs, *ifa;
    ucs_netlink_route_t route;
    int local_index, vrf_index;
    ucs_status_t status;

    ASSERT_NE(0u, lo_index);

    sin.sin_family      = AF_INET;
    sin.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    status = ucs_netlink_route_get((struct sockaddr*)&sin, NULL, 0, &route);
    if (status == UCS_ERR_UNSUPPORTED) {
        UCS_TEST_SKIP_R("kernel does not support route lookup by fib match");
    }

    ASSERT_UCS_OK(status);
    EXPECT_EQ(RTN_LOCAL, route.type);
    ASSERT_EQ(1, route.num_oifs);
    EXPECT_EQ(static_cast<int>(lo_index), route.oifs[0]);

    ASSERT_UCS_OK(ucs_netlink_route_get((struct sockaddr*)&sin,
                                        (struct sockaddr*)&sin, lo_index,
                                        &route));
    EXPECT_EQ(RTN_LOCAL, route.type);

    /* TEST-NET-2 address has no route through the loopback device */
    ASSERT_EQ(1, inet_pton(AF_INET, "198.51.100.1", &sin.sin_addr));
    for (int i = 0; i < 2; ++i) {
        /* The second lookup returns the cached result */
        EXPECT_EQ(UCS_ERR_UNREACHABLE,
                  ucs_netlink_route_get((struct sockaddr*)&sin, NULL, lo_index,
                                        &route));
    }

    ASSERT_EQ(0, getifaddrs(&ifaddrs));

    for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
        if ((ifa->ifa_addr == NULL) ||
            ((ifa->ifa_addr->sa_family != AF_INET) &&
             (ifa->ifa_addr->sa_family != AF_INET6))) {
            continue;
        }

        /* A link-local address is ambiguous without an interface */
        if ((ifa->ifa_addr->sa_family == AF_INET6) &&
            IN6_IS_ADDR_LINKLOCAL(
                    &((struct sockaddr_in6*)ifa->ifa_addr)->sin6_addr)) {
            continue;
        }

        local_index = ucs_netlink_get_local_route_ndev_index(ifa->ifa_addr);
        if (local_index < 0) {
            continue;
        }

        /* The kernel selects the host route of a local address in the
         * routing domain of the address */
        vrf_index = ucs_netlink_get_vrf_master_index(local_index);
        ASSERT_UCS_OK(ucs_netlink_route_get(ifa->ifa_addr, NULL,
                                            std::max(vrf_index, 0), &route),
                      << ifa->ifa_name);
        EXPECT_EQ(RTN_LOCAL, route.type) << ifa->ifa_name;
        EXPECT_EQ((ifa->ifa_addr->sa_family == AF_INET) ? 32 : 128,
                  route.prefix_len)
                << ifa->ifa_name;
        ++num_addrs;
    }

    freeifaddrs(ifaddrs);
    UCS_TEST_MESSAGE << num_addrs << " local addresses checked";
}

UCS_TEST_F(test_netlink, kernel_route_matches) {
    unsigned lo_index      = if_nametoindex("lo");
    unsigned num_defaults  = 0;
    struct sockaddr_in sin = {};
    struct if_nameindex *ifs, *ifp;
    ucs_netlink_route_t route;
    uint8_t best_prefix_len;
    ucs_status_t status;

    ASSERT_NE(0u, lo_index);

    sin.sin_family = AF_INET;
    ASSERT_EQ(1, inet_pton(AF_INET, "198.51.100.1", &sin.sin_addr));
    status = ucs_netlink_kernel_route_matches(lo_index, (struct sockaddr*)&sin,
                                              NULL, &best_prefix_len);
    if (status == UCS_ERR_UNSUPPORTED) {
        UCS_TEST_SKIP_R("kernel does not support route lookup by fib match");
    }

    /* TEST-NET-2 address has no route through the loopback device */
    EXPECT_EQ(UCS_ERR_UNREACHABLE, status);

    /* No interface has a more specific route than the default to TEST-NET-2.
     * A lookup through a VRF device selects a route through its slaves. */
    ifs = if_nameindex();
    ASSERT_NE(nullptr, ifs);
    for (ifp = ifs; ifp->if_index != 0; ++ifp) {
        if ((ucs_netlink_route_get((struct sockaddr*)&sin, NULL, ifp->if_index,
                                   &route) == UCS_OK) &&
            (route.prefix_len == 0) &&
            (std::find(route.oifs, route.oifs + route.num_oifs,
                       static_cast<int>(ifp->if_index)) !=
             route.oifs + route.num_oifs)) {
            EXPECT_EQ(UCS_OK, ucs_netlink_kernel_route_matches(
                                      ifp->if_index, (struct sockaddr*)&sin,
                                      NULL, &best_prefix_len))
                    << ifp->if_name;
            EXPECT_EQ(0, best_prefix_len) << ifp->if_name;
            ++num_defaults;
        }
    }

    if_freenameindex(ifs);
    UCS_TEST_MESSAGE << num_defaults << " default routes checked";
}

class test_netlink_route_cache : private ucs::clear_dontcopy_regions,
                                 public test_netlink {
protected:
    /* Exit code of the child process when the test cannot run */
    static const int CHILD_SKIP = 77;

    static int child_error(const std::string &msg)
    {
        UCS_TEST_MESSAGE << msg;
        return 1;
    }

    static std::string status_str(ucs_status_t status)
    {
        return ucs_status_string(status);
    }

    void run_in_child(int (*check)(), const std::string &skip_reason)
    {
        int status;
        pid_t pid;

        pid = fork();
        ASSERT_NE(-1, pid) << strerror(errno);
        if (pid == 0) {
            _exit(check());
        }

        ASSERT_EQ(pid, waitpid(pid, &status, 0));
        ASSERT_TRUE(WIFEXITED(status)) << ucs::exit_status_info(status);
        if (WEXITSTATUS(status) == CHILD_SKIP) {
            UCS_TEST_SKIP_R(skip_reason);
        }

        EXPECT_EQ(0, WEXITSTATUS(status));
    }

    static ucs_status_t route_get(const char *ttl, const char *addr,
                                  unsigned if_index, ucs_netlink_route_t *route)
    {
        struct sockaddr_in sin = {};
        ucs_status_t status;

        status = ucs_global_opts_set_value("NETLINK_ROUTE_CACHE_TTL", ttl);
        if (status != UCS_OK) {
            return status;
        }

        sin.sin_family = AF_INET;
        inet_pton(AF_INET, addr, &sin.sin_addr);
        return ucs_netlink_route_get((struct sockaddr*)&sin, NULL, if_index,
                                     route);
    }

    static bool set_lo_up(int fd)
    {
        struct ifreq ifr = {};

        strncpy(ifr.ifr_name, "lo", sizeof(ifr.ifr_name) - 1);
        if (ioctl(fd, SIOCGIFFLAGS, &ifr) != 0) {
            return false;
        }

        ifr.ifr_flags |= IFF_UP;
        return ioctl(fd, SIOCSIFFLAGS, &ifr) == 0;
    }

    /* Add or delete the TEST-NET-2 route through the loopback device */
    static bool change_test_net_2_route(int fd, unsigned long request)
    {
        struct rtentry rt = {};
        struct sockaddr_in *sin;

        sin                  = reinterpret_cast<sockaddr_in*>(&rt.rt_dst);
        sin->sin_family      = AF_INET;
        sin->sin_addr.s_addr = htonl(0xc6336400); /* 198.51.100.0 */
        sin                  = reinterpret_cast<sockaddr_in*>(&rt.rt_genmask);
        sin->sin_family      = AF_INET;
        sin->sin_addr.s_addr = htonl(0xffffff00);
        rt.rt_flags          = RTF_UP;
        rt.rt_dev            = const_cast<char*>("lo");
        return ioctl(fd, request, &rt) == 0;
    }

    static int check_route_change(int fd)
    {
        /* Lookups of 198.51.100.1 by other tests can be in the cache copied
         * from the parent process */
        const char *addr  = "198.51.100.2";
        unsigned lo_index = if_nametoindex("lo");
        ucs_netlink_route_t route;
        ucs_status_t status;

        status = route_get("inf", addr, lo_index, &route);
        if (status == UCS_ERR_UNSUPPORTED) {
            return CHILD_SKIP;
        } else if (status != UCS_ERR_UNREACHABLE) {
            return child_error("lookup without a route: " + status_str(status));
        }

        if (!change_test_net_2_route(fd, SIOCADDRT)) {
            return child_error(std::string("failed to add a route: ") +
                               strerror(errno));
        }

        status = route_get("inf", addr, lo_index, &route);
        if (status != UCS_ERR_UNREACHABLE) {
            return child_error("cached lookup: " + status_str(status));
        }

        usleep(20000);
        status = route_get("10ms", addr, lo_index, &route);
        if ((status != UCS_OK) || (route.prefix_len != 24)) {
            return child_error("lookup after adding a route: " +
                               status_str(status));
        }

        if (!change_test_net_2_route(fd, SIOCDELRT)) {
            return child_error(std::string("failed to delete a route: ") +
                               strerror(errno));
        }

        status = route_get("inf", addr, lo_index, &route);
        if (status != UCS_OK) {
            return child_error("cached lookup of a route: " +
                               status_str(status));
        }

        status = route_get("0", addr, lo_index, &route);
        if (status != UCS_ERR_UNREACHABLE) {
            return child_error("lookup without the cache: " +
                               status_str(status));
        }

        usleep(20000);
        status = route_get("10ms", addr, lo_index, &route);
        if (status != UCS_ERR_UNREACHABLE) {
            return child_error("lookup after deleting a route: " +
                               status_str(status));
        }

        return 0;
    }

    static int route_change_child()
    {
        int fd, ret;

        /* The process gets the capabilities of the new user namespace, which
         * owns the new network namespace */
        if (unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0) {
            return CHILD_SKIP;
        }

        fd = socket(AF_INET, SOCK_DGRAM, 0);
        if (fd < 0) {
            return child_error(std::string("failed to create a socket: ") +
                               strerror(errno));
        }

        /* The loopback device of a new network namespace is down */
        ret = set_lo_up(fd) ? check_route_change(fd) : CHILD_SKIP;
        close(fd);
        return ret;
    }

    /* A lookup which reaches the kernel fails to create a netlink socket, and
     * logs an error */
    static int check_lookups_without_files(unsigned lo_index)
    {
        ucs_netlink_route_t route;
        ucs_status_t status;
        size_t num_errors;

        num_errors = m_errors.size();
        status     = route_get("inf", "127.0.0.1", lo_index, &route);
        if ((status != UCS_OK) || (m_errors.size() != num_errors)) {
            return child_error("cached lookup: " + status_str(status));
        }

        status = route_get("0", "127.0.0.1", lo_index, &route);
        if ((status != UCS_ERR_NO_RESOURCE) ||
            (m_errors.size() == num_errors)) {
            return child_error("lookup without the cache: " +
                               status_str(status));
        }

        num_errors = m_errors.size();
        usleep(20000);
        status = route_get("10ms", "127.0.0.1", lo_index, &route);
        if ((status != UCS_OK) || (m_errors.size() == num_errors)) {
            return child_error("failed lookup of an expired route: " +
                               status_str(status));
        }

        status = route_get("inf", "127.0.0.2", lo_index, &route);
        if (status != UCS_ERR_NO_RESOURCE) {
            return child_error("failed lookup of a new destination: " +
                               status_str(status));
        }

        return 0;
    }

    static int failed_lookup_child()
    {
        /* Getting the interface index requires a socket */
        unsigned lo_index = if_nametoindex("lo");
        scoped_log_handler slh(hide_errors_logger);
        ucs_netlink_route_t route;
        ucs_status_t status;
        int ret;

        status = route_get("inf", "127.0.0.1", lo_index, &route);
        if (status == UCS_ERR_UNSUPPORTED) {
            return CHILD_SKIP;
        } else if (status != UCS_OK) {
            return child_error("lookup of a local address: " +
                               status_str(status));
        }

        {
            ucs::rlimit_setter no_files(RLIMIT_NOFILE, 0);

            ret = (no_files.limit() == 0) ?
                          check_lookups_without_files(lo_index) :
                          child_error("failed to limit open files");
        }

        if (ret != 0) {
            return ret;
        }

        status = route_get("inf", "127.0.0.2", lo_index, &route);
        if (status != UCS_OK) {
            return child_error("lookup after a failed lookup: " +
                               status_str(status));
        }

        return 0;
    }

    static int lookup_local_addrs(unsigned lo_index, int num_lookups)
    {
        struct sockaddr_in sin = {};
        int num_failures       = 0;
        ucs_netlink_route_t route;
        ucs_status_t status;

        sin.sin_family = AF_INET;
        for (int i = 0; i < num_lookups; ++i) {
            /* 127.1.0.0/26, on the local route of the loopback */
            sin.sin_addr.s_addr = htonl(0x7f010000 + (i % 64));
            status = ucs_netlink_route_get((struct sockaddr*)&sin, NULL,
                                           lo_index, &route);
            if ((status != UCS_OK) || (route.type != RTN_LOCAL)) {
                ++num_failures;
            }
        }

        return num_failures;
    }

    /* Cached results expire at once, so threads look up the same destinations
     * in the kernel at the same time, and add them to the cache */
    static int concurrent_lookups_child()
    {
        const int num_threads = 8;
        const int num_lookups = 200;
        unsigned lo_index     = if_nametoindex("lo");
        std::atomic<int> num_failures(0);
        std::vector<std::thread> threads;
        ucs_netlink_route_t route;
        ucs_status_t status;

        status = route_get("1us", "127.1.0.0", lo_index, &route);
        if (status == UCS_ERR_UNSUPPORTED) {
            return CHILD_SKIP;
        } else if (status != UCS_OK) {
            return child_error("lookup of a local address: " +
                               status_str(status));
        }

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&]() {
                num_failures += lookup_local_addrs(lo_index, num_lookups);
            });
        }

        for (std::thread &thread : threads) {
            thread.join();
        }

        if (num_failures != 0) {
            return child_error(std::to_string(num_failures.load()) +
                               " failed lookups");
        }

        return 0;
    }
};

UCS_TEST_F(test_netlink_route_cache, route_change) {
    run_in_child(route_change_child,
                 "cannot change routes in a new network namespace");
}

UCS_TEST_F(test_netlink_route_cache, failed_lookup) {
    run_in_child(failed_lookup_child,
                 "kernel does not support route lookup by fib match");
}

UCS_TEST_F(test_netlink_route_cache, concurrent_lookups) {
    run_in_child(concurrent_lookups_child,
                 "kernel does not support route lookup by fib match");
}

UCS_TEST_F(test_netlink, is_host_addr) {
    struct sockaddr_in sin = {};
    struct ifaddrs *ifaddrs, *ifa;

    ASSERT_EQ(0, getifaddrs(&ifaddrs));

    for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
        if ((ifa->ifa_addr != NULL) &&
            ((ifa->ifa_addr->sa_family == AF_INET) ||
             (ifa->ifa_addr->sa_family == AF_INET6))) {
            EXPECT_EQ(1, ucs_netlink_is_host_addr(ifa->ifa_addr))
                    << ifa->ifa_name;
        }
    }

    freeifaddrs(ifaddrs);

    /* TEST-NET-2 address is not assigned to any host */
    sin.sin_family = AF_INET;
    ASSERT_EQ(1, inet_pton(AF_INET, "198.51.100.1", &sin.sin_addr));
    EXPECT_EQ(0, ucs_netlink_is_host_addr((struct sockaddr*)&sin));
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

    auto parse_error = [&](int error, unsigned short type,
                           unsigned short flags) {
        nlerr.error           = error;
        nlerr.msg.nlmsg_type  = type;
        nlerr.msg.nlmsg_flags = flags;
        reply.clear();
        append_nlmsg(reply, NLMSG_ERROR, &nlerr, sizeof(nlerr));
        return ucs_netlink_parse_msg(reply.data(), reply.size(), count_msg_cb,
                                     &count, UCS_LOG_LEVEL_DEBUG, &msg_done);
    };

    /* A route lookup fails when the kernel has no route */
    EXPECT_EQ(UCS_ERR_UNREACHABLE, parse_error(-EHOSTUNREACH, RTM_GETROUTE, 0));
    EXPECT_EQ(UCS_ERR_UNREACHABLE, parse_error(-ENETUNREACH, RTM_GETROUTE, 0));

    /* A route lookup which selects a prohibit route fails with an error which
     * other requests return for other failures */
    EXPECT_EQ(UCS_ERR_UNREACHABLE, parse_error(-EACCES, RTM_GETROUTE, 0));
    EXPECT_EQ(UCS_ERR_IO_ERROR, parse_error(-EACCES, RTM_GETROUTE, NLM_F_DUMP));
    EXPECT_EQ(UCS_ERR_IO_ERROR, parse_error(-EACCES, RTM_GETLINK, 0));

    /* The kernel also rejects invalid route lookups with EINVAL, as it does
     * lookups which select a blackhole route */
    EXPECT_EQ(UCS_ERR_IO_ERROR, parse_error(-EINVAL, RTM_GETROUTE, 0));

    EXPECT_EQ(UCS_ERR_NO_MEMORY, parse_error(-ENOBUFS, RTM_GETROUTE, 0));
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
