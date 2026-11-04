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
#include <unistd.h>


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
};

UCS_TEST_F(test_netlink, get_vrf_master_index) {
    struct if_nameindex *ifs = if_nameindex();
    unsigned num_vrf_slaves  = 0;
    char master_name[IFNAMSIZ];

    ASSERT_NE(nullptr, ifs);

    for (struct if_nameindex *it = ifs; it->if_index != 0; ++it) {
        int master_index = ucs_netlink_get_vrf_master_index(it->if_index);
        if (master_index == -1) {
            continue;
        }

        /* A VRF slave must be linked to its master in sysfs */
        ASSERT_NE(nullptr, if_indextoname(master_index, master_name))
                << it->if_name;
        EXPECT_EQ(sysfs_master_name(it->if_name), master_name) << it->if_name;
        ++num_vrf_slaves;
    }

    if_freenameindex(ifs);

    EXPECT_EQ(-1, ucs_netlink_get_vrf_master_index(if_nametoindex("lo")));
    UCS_TEST_MESSAGE << num_vrf_slaves << " VRF slave interfaces found";
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
