/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

#include <ucs/sys/netlink_int.h>

#include <arpa/inet.h>
#include <linux/rtnetlink.h>
#include <vector>

class test_netlink : public ucs::test {
protected:
    struct route_desc {
        int         if_index;
        const char *dest;
        uint8_t     subnet_prefix_len;
        uint8_t     route_type;
    };

    struct route_case {
        const char             *name;
        int                     family;
        const char             *remote;
        std::vector<route_desc> routes;
        int                     if_index;
        int                     best_result;
        int                     relaxed_result;
    };

    /* Holds a route table for the lifetime of a single check */
    class route_table {
    public:
        route_table()
        {
            ucs_netlink_route_table_init(&m_table);
        }

        ~route_table()
        {
            ucs_netlink_route_table_cleanup(&m_table);
        }

        ucs_netlink_route_table_t *get()
        {
            return &m_table;
        }

    private:
        ucs_netlink_route_table_t m_table;
    };

    static void init_sockaddr(struct sockaddr_storage *saddr, int family,
                              const char *addr_str)
    {
        void *in_addr;

        memset(saddr, 0, sizeof(*saddr));
        saddr->ss_family = family;
        if (family == AF_INET) {
            in_addr = &reinterpret_cast<struct sockaddr_in*>(saddr)->sin_addr;
        } else {
            in_addr = &reinterpret_cast<struct sockaddr_in6*>(saddr)->sin6_addr;
        }

        ASSERT_EQ(1, inet_pton(family, addr_str, in_addr));
    }

    void add_routes(ucs_netlink_route_table_t *table, int family,
                    const std::vector<route_desc> &routes)
    {
        struct sockaddr_storage dest;

        for (const auto &route : routes) {
            init_sockaddr(&dest, family, route.dest);
            ASSERT_UCS_OK(ucs_netlink_route_table_add(
                            table, route.if_index,
                            reinterpret_cast<const struct sockaddr*>(&dest),
                            route.subnet_prefix_len, route.route_type));
        }
    }

    void check_route_matches(const route_case &test_case,
                             ucs_netlink_route_check_t route_check,
                             int expected_result)
    {
        struct sockaddr_storage remote;
        route_table table;

        add_routes(table.get(), test_case.family, test_case.routes);
        init_sockaddr(&remote, test_case.family, test_case.remote);
        EXPECT_EQ(expected_result,
                  ucs_netlink_route_matches_in_table(
                          table.get(), test_case.if_index,
                          reinterpret_cast<const struct sockaddr*>(&remote),
                          route_check));
    }
};

UCS_TEST_F(test_netlink, route_matches)
{
    const std::vector<route_case> cases = {
        {"no route", AF_INET, "10.1.2.3", {}, 1, 0, 0},
        {"more specific route on another interface", AF_INET, "10.1.2.3",
         {{1, "10.1.0.0", 16}, {2, "10.1.2.0", 24}}, 1, 0, 1},
        {"tied best routes", AF_INET, "10.1.2.3",
         {{1, "10.1.2.0", 24}, {2, "10.1.2.0", 24}}, 1, 1, 1},
        {"default route only", AF_INET, "10.1.2.3",
         {{1, "0.0.0.0", 0}}, 1, 1, 1},
        {"default route with specific route on another interface", AF_INET,
         "10.1.2.3", {{1, "0.0.0.0", 0}, {2, "10.1.2.0", 24}}, 1, 0, 0},
        {"longest route on candidate interface", AF_INET, "10.1.2.3",
         {{1, "10.0.0.0", 8}, {1, "10.1.2.0", 24}, {2, "10.1.0.0", 16}},
         1, 1, 1},
        {"no matching route on candidate interface", AF_INET, "10.1.2.3",
         {{1, "192.168.0.0", 16}, {2, "10.1.2.0", 24}}, 1, 0, 0},
        {"ipv6 more specific route on another interface", AF_INET6,
         "2001:db8:1:2::3",
         {{1, "2001:db8:1::", 48}, {2, "2001:db8:1:2::", 64}}, 1, 0, 1},
        {"ipv6 longest route on candidate interface", AF_INET6,
         "2001:db8:1:2::3",
         {{1, "2001:db8:1:2::", 64}, {2, "2001:db8:1::", 48}}, 1, 1, 1}
    };

    for (const auto &test_case : cases) {
        SCOPED_TRACE(test_case.name);
        check_route_matches(test_case, UCS_NETLINK_ROUTE_CHECK_BEST,
                            test_case.best_result);
        check_route_matches(test_case, UCS_NETLINK_ROUTE_CHECK_RELAXED,
                            test_case.relaxed_result);
    }
}

UCS_TEST_F(test_netlink, local_route_ndev_index)
{
    const std::vector<route_desc> routes = {
        {1, "10.1.0.0", 16, RTN_LOCAL},
        {2, "10.1.2.0", 24, RTN_UNICAST},
        {3, "10.1.2.0", 24, RTN_LOCAL}
    };
    struct sockaddr_storage remote;
    route_table table;

    add_routes(table.get(), AF_INET, routes);

    /* Only local routes are considered, so the unicast route on interface 2 is
       ignored and the longest local route wins */
    init_sockaddr(&remote, AF_INET, "10.1.2.3");
    EXPECT_EQ(3, ucs_netlink_local_route_ndev_index_in_table(
                         table.get(),
                         reinterpret_cast<const struct sockaddr*>(&remote)));

    init_sockaddr(&remote, AF_INET, "192.168.1.1");
    EXPECT_EQ(-1, ucs_netlink_local_route_ndev_index_in_table(
                          table.get(),
                          reinterpret_cast<const struct sockaddr*>(&remote)));
}
