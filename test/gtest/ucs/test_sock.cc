/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/sock.h>
#include <ucs/debug/assert.h>
}

#include <sys/un.h>


static std::string socket_err_exp_str;

class test_socket : public ucs::test {
public:
protected:

    static ucs_log_func_rc_t
    socket_error_handler(const char *file, unsigned line, const char *function,
                         ucs_log_level_t level,
                         const ucs_log_component_config_t *comp_conf,
                         const char *message, va_list ap)
    {
        // Ignore errors that invalid input parameters as it is expected
        if (level == UCS_LOG_LEVEL_ERROR) {
            std::string err_str = format_message(message, ap);

            if (err_str.find(socket_err_exp_str) != std::string::npos) {
                UCS_TEST_MESSAGE << err_str;
                return UCS_LOG_FUNC_RC_STOP;
            }
        }

        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    static void check_ip_port(const char *ip_port_str, uint16_t default_port,
                              ucs_status_t exp_status, const std::string &exp_str)
    {
        struct sockaddr_storage ss = {0};
        struct sockaddr *saddr     = (struct sockaddr*)&ss;
        ucs_status_t status;

        status = ucs_sock_ipportstr_to_sockaddr(ip_port_str, default_port, &ss);
        EXPECT_EQ(exp_status, status);
        if (UCS_OK == status) {
            char test_str[1024];
            ucs_sockaddr_str(saddr, test_str, sizeof(test_str));
            EXPECT_EQ(exp_str, test_str);
        }
    }

    void verify_subnet_match(const char *ip_addr1,
                             const char *ip_addr2,
                             unsigned longest_common_prefix) const
    {
        struct sockaddr_storage saddr1, saddr2;
        int matched;

        ASSERT_UCS_OK(ucs_sock_ipstr_to_sockaddr(ip_addr1, &saddr1));
        ASSERT_UCS_OK(ucs_sock_ipstr_to_sockaddr(ip_addr2, &saddr2));

        size_t addr_size;
        ASSERT_UCS_OK(ucs_sockaddr_inet_addr_sizeof((struct sockaddr *)&saddr1,
                                                    &addr_size));
        size_t addr_size_bits = addr_size * CHAR_BIT;

        /* Add extra iterations to check prefix truncation */
        constexpr size_t extra_iters = 10;

        for (size_t prefix = 0; prefix <= addr_size_bits + extra_iters; ++prefix) {
            matched = ucs_sockaddr_is_same_subnet((struct sockaddr*)&saddr1,
                                                  (struct sockaddr*)&saddr2,
                                                  prefix);
            EXPECT_EQ(std::min(prefix, addr_size_bits) <= longest_common_prefix,
                      matched)
                    << "failed to match subnet: ip1=" << ip_addr1
                    << " ip2=" << ip_addr2 << " prefix=" << prefix
                    << " max_common=" << longest_common_prefix
                    << " (matched=" << matched << ")";
        }
    }

    void verify_subnet_error(sa_family_t af1, sa_family_t af2, unsigned prefix) const
    {
        struct sockaddr_storage storage1 = {0}, storage2 = {0};
        struct sockaddr *sa1             = (struct sockaddr *)&storage1;
        struct sockaddr *sa2             = (struct sockaddr *)&storage2;

        sa1->sa_family = af1;
        sa2->sa_family = af2;
        EXPECT_FALSE(ucs_sockaddr_is_same_subnet(sa1, sa2, prefix));
    }
};

UCS_TEST_F(test_socket, sockaddr_sizeof) {
    struct sockaddr_storage ss = {0};
    struct sockaddr *saddr     = (struct sockaddr*)&ss;
    size_t size;


    /* Check with IPv4 */
    {
        size             = 0;
        saddr->sa_family = AF_INET;
        EXPECT_UCS_OK(ucs_sockaddr_sizeof(saddr, &size));
        EXPECT_EQ(sizeof(struct sockaddr_in), size);
    }

    /* Check with IPv6 */
    {
        size             = 0;
        saddr->sa_family = AF_INET6;
        EXPECT_UCS_OK(ucs_sockaddr_sizeof(saddr, &size));
        EXPECT_EQ(sizeof(struct sockaddr_in6), size);
    }

    /* Check with wrong address family */
    {
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        size             = 0;
        saddr->sa_family = AF_UNIX;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, ucs_sockaddr_sizeof(saddr, &size));
        /* Check that doesn't touch provided memory in error case */
        EXPECT_EQ(0ULL, size);
    }
}

UCS_TEST_F(test_socket, sockaddr_inet_addr_sizeof) {
    struct sockaddr_storage ss = {0};
    struct sockaddr *saddr     = (struct sockaddr*)&ss;
    size_t size;

    /* Check with IPv4 */
    {
        size             = 0;
        saddr->sa_family = AF_INET;
        EXPECT_UCS_OK(ucs_sockaddr_inet_addr_sizeof(saddr, &size));
        EXPECT_EQ(UCS_IPV4_ADDR_LEN, size);
    }

    /* Check with IPv6 */
    {
        size             = 0;
        saddr->sa_family = AF_INET6;
        EXPECT_UCS_OK(ucs_sockaddr_inet_addr_sizeof(saddr, &size));
        EXPECT_EQ(UCS_IPV6_ADDR_LEN, size);
    }

    /* Check with wrong address family */
    {
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        size             = 0;
        saddr->sa_family = AF_UNIX;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  ucs_sockaddr_inet_addr_sizeof(saddr, &size));
        /* Check that doesn't touch provided memory in error case */
        EXPECT_EQ(0ULL, size);
    }
}

UCS_TEST_F(test_socket, sockaddr_get_port) {
    const uint16_t sin_port    = 5555;
    uint16_t port              = 0;
    struct sockaddr_storage ss = {0};
    struct sockaddr *saddr     = (struct sockaddr*)&ss;

    /* Check with IPv4 */
    {
        port                                   = 0;
        saddr->sa_family                       = AF_INET;
        ((struct sockaddr_in*)saddr)->sin_port = htons(sin_port);
        EXPECT_UCS_OK(ucs_sockaddr_get_port(saddr, &port));
        EXPECT_EQ(sin_port, port);
    }

    /* Check with IPv6 */
    {
        port                                     = 0;
        saddr->sa_family                         = AF_INET6;
        ((struct sockaddr_in6*)saddr)->sin6_port = htons(sin_port);
        EXPECT_UCS_OK(ucs_sockaddr_get_port(saddr, &port));
        EXPECT_EQ(sin_port, port);
    }

    /* Check with wrong address family */
    {
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        port             = sin_port;
        saddr->sa_family = AF_UNIX;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, ucs_sockaddr_get_port(saddr, &port));
        /* Check that doesn't touch provided memory in error case */
        EXPECT_EQ(sin_port, port);
    }
}

UCS_TEST_F(test_socket, sockaddr_get_inet_addr) {
    struct sockaddr_storage ss = {0};
    struct sockaddr *saddr     = (struct sockaddr*)&ss;
    struct sockaddr_in *saddr_in;
    struct sockaddr_in6 *saddr_in6;
    struct in_addr sin_addr;
    struct in6_addr sin6_addr;

    /* Check with wrong IPv4 */
    {
        saddr->sa_family                              = AF_INET;
        sin_addr.s_addr                               = htonl(INADDR_ANY);
        ((struct sockaddr_in*)saddr)->sin_addr.s_addr = htonl(INADDR_ANY);
        EXPECT_EQ(&((struct sockaddr_in*)saddr)->sin_addr,
                  ucs_sockaddr_get_inet_addr(saddr));
        saddr_in = (struct sockaddr_in*)saddr;
        EXPECT_EQ(0, memcmp(&saddr_in->sin_addr, &sin_addr, sizeof(sin_addr)));
    }

    /* Check with wrong IPv6 */
    {
        saddr->sa_family                         = AF_INET6;
        sin6_addr                                = in6addr_any;
        ((struct sockaddr_in6*)saddr)->sin6_addr = in6addr_any;
        EXPECT_EQ(&((struct sockaddr_in6*)saddr)->sin6_addr,
                  ucs_sockaddr_get_inet_addr(saddr));
        saddr_in6 = (struct sockaddr_in6*)saddr;
        EXPECT_EQ(0,
                  memcmp(&saddr_in6->sin6_addr, &sin6_addr, sizeof(sin6_addr)));
    }

    /* Check with wrong address family */
    {
        saddr->sa_family   = AF_UNIX;
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        EXPECT_EQ(NULL, ucs_sockaddr_get_inet_addr(saddr));
    }
}

UCS_TEST_F(test_socket, sockaddr_set_port) {
    const uint16_t sin_port     = 5555;
    const uint16_t sin_port_net = htons(sin_port);
    struct sockaddr_storage ss  = {0};
    struct sockaddr *saddr      = (struct sockaddr*)&ss;

    /* Check with IPv4 */
    {
        saddr->sa_family = AF_INET;
        EXPECT_UCS_OK(ucs_sockaddr_set_port(saddr, sin_port));
        EXPECT_EQ(sin_port_net, ((struct sockaddr_in*)saddr)->sin_port);
    }

    /* Check with IPv6 */
    {
        saddr->sa_family = AF_INET6;
        EXPECT_UCS_OK(ucs_sockaddr_set_port(saddr, sin_port));
        EXPECT_EQ(sin_port_net, ((struct sockaddr_in6*)saddr)->sin6_port);
    }

    /* Check with wrong address family */
    {
        saddr->sa_family   = AF_UNIX;
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  ucs_sockaddr_set_port(saddr, sin_port));
    }
}

UCS_TEST_F(test_socket, sockaddr_set_inet_addr) {
    struct sockaddr_storage ss = {0};
    struct sockaddr *saddr     = (struct sockaddr*)&ss;
    struct in_addr sin_addr;
    struct in6_addr sin6_addr;

    sin_addr.s_addr = htonl(INADDR_ANY);
    sin6_addr       = in6addr_any;

    /* Check with IPv4 */
    {
        saddr->sa_family = AF_INET;

        EXPECT_UCS_OK(ucs_sockaddr_set_inet_addr(saddr, &sin_addr));
        EXPECT_EQ(0, memcmp(&((struct sockaddr_in*)saddr)->sin_addr, &sin_addr,
                            sizeof(sin_addr)));
    }

    /* Check with IPv6 */
    {
        saddr->sa_family = AF_INET6;

        EXPECT_UCS_OK(ucs_sockaddr_set_inet_addr(saddr, &sin6_addr));
        EXPECT_EQ(0, memcmp(&((struct sockaddr_in6*)saddr)->sin6_addr,
                            &sin6_addr, sizeof(sin6_addr)));
    }

    /* Check with wrong address family */
    {
        saddr->sa_family   = AF_UNIX;
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  ucs_sockaddr_set_inet_addr(saddr, NULL));
    }
}

UCS_TEST_F(test_socket, sockaddr_is_inaddr) {
    struct sockaddr_storage ss = {0};
    struct sockaddr *saddr     = (struct sockaddr*)&ss;
    struct in_addr sin_addr_loopback, sin_addr_any;
    struct in6_addr sin6_addr_loopback, sin6_addr_any;

    sin_addr_any.s_addr = htonl(INADDR_ANY);
    sin6_addr_any       = in6addr_any;

    sin_addr_loopback.s_addr = htonl(INADDR_LOOPBACK);
    sin6_addr_loopback       = in6addr_loopback;

    /* Check with IPv4 */
    {
        saddr->sa_family = AF_INET;

        /* ANY is specified to address */
        ASSERT_UCS_OK(ucs_sockaddr_set_inet_addr(saddr, &sin_addr_any));
        EXPECT_TRUE(ucs_sockaddr_is_inaddr_any(saddr));
        EXPECT_FALSE(ucs_sockaddr_is_inaddr_loopback(saddr));

        /* LOOPBACK is specified to address */
        ASSERT_UCS_OK(ucs_sockaddr_set_inet_addr(saddr, &sin_addr_loopback));
        EXPECT_FALSE(ucs_sockaddr_is_inaddr_any(saddr));
        EXPECT_TRUE(ucs_sockaddr_is_inaddr_loopback(saddr));
    }

    /* Check with IPv6 */
    {
        saddr->sa_family = AF_INET6;

        /* ANY is specified to address */
        ASSERT_UCS_OK(ucs_sockaddr_set_inet_addr(saddr, &sin6_addr_any));
        EXPECT_TRUE(ucs_sockaddr_is_inaddr_any(saddr));
        EXPECT_FALSE(ucs_sockaddr_is_inaddr_loopback(saddr));

        /* LOOPBACK is specified to address */
        ASSERT_UCS_OK(ucs_sockaddr_set_inet_addr(saddr, &sin6_addr_loopback));
        EXPECT_FALSE(ucs_sockaddr_is_inaddr_any(saddr));
        EXPECT_TRUE(ucs_sockaddr_is_inaddr_loopback(saddr));
    }

    /* Check with wrong address family */
    {
        saddr->sa_family   = AF_UNIX;
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        EXPECT_FALSE(ucs_sockaddr_is_inaddr_any(saddr));
    }
}

UCS_TEST_F(test_socket, str_sockaddr_str) {
    const uint16_t port        = 65534;
    const uint32_t scope_id    = 1;
    const char *ipv4_addr      = "192.168.122.157";
    const char *ipv6_addr      = "fe80::218:e7ff:fe16:fb97";
    struct sockaddr_storage ss = {0};
    struct sockaddr *saddr     = (struct sockaddr*)&ss;
    char ipv4_addr_out[128], ipv6_addr_out[128], *str, test_str[1024];
    ucs_status_t status;

    sprintf(ipv4_addr_out, "%s:%d", ipv4_addr, port);
    sprintf(ipv6_addr_out, "%s%%%d:%d", ipv6_addr, scope_id, port);

    /* Check `str_len` with IPv4 address */
    {
        saddr->sa_family                       = AF_INET;
        ((struct sockaddr_in*)saddr)->sin_port = htons(port);
        status = ucs_sock_ipstr_to_sockaddr(ipv4_addr, &ss);
        ASSERT_EQ(UCS_OK, status);

        /* Check with short `str_len` to fit IP address only */
        str = (char*)ucs_sockaddr_str(saddr, test_str, INET_ADDRSTRLEN);
        EXPECT_EQ(str, test_str);
        EXPECT_EQ(0, strncmp(test_str, ipv4_addr_out,
                             INET_ADDRSTRLEN - 1));

        /* Check with big enough `str_len` */
        str = (char*)ucs_sockaddr_str(saddr, test_str, 1024);
        EXPECT_EQ(str, test_str);
        EXPECT_EQ(0, strcmp(test_str, ipv4_addr_out));
    }

    /* Check `str_len` with IPv6 address */
    {
        saddr->sa_family                         = AF_INET6;
        ((struct sockaddr_in6*)saddr)->sin6_port     = htons(port);
        ((struct sockaddr_in6*)saddr)->sin6_scope_id = scope_id;
        status = ucs_sock_ipstr_to_sockaddr(ipv6_addr, &ss);
        ASSERT_EQ(UCS_OK, status);

        /* Check with short `str_len` to fit IP address only */
        str = (char*)ucs_sockaddr_str(saddr, test_str, INET6_ADDRSTRLEN);
        EXPECT_EQ(str, test_str);
        EXPECT_EQ(0, strncmp(test_str, ipv6_addr_out, INET6_ADDRSTRLEN - 1));

        /* Check with big enough `str_len` */
        str = (char*)ucs_sockaddr_str(saddr, test_str, 1024);
        EXPECT_TRUE(str == test_str);
        EXPECT_EQ(0, strcmp(test_str, ipv6_addr_out));
    }

    /* Check with wrong sa_family */
    {
        saddr->sa_family = AF_UNIX;

        /* with big enough string */
        {
            str = (char*)ucs_sockaddr_str(saddr, test_str, 1024);
            EXPECT_EQ(test_str, str);
            EXPECT_EQ(0, strcmp(str, "<invalid address family>"));
        }

        /* without string */
        {
            str = (char*)ucs_sockaddr_str(saddr, NULL, 0);
            EXPECT_EQ(NULL, str);
        }
    }

    /* Check NULL sockaddr */
    {
        /* with big enough string */
        {
            str = (char*)ucs_sockaddr_str(NULL, test_str, 1024);
            EXPECT_EQ(test_str, str);
            EXPECT_EQ(0, strcmp(str, "<null>"));
        }

        /* without string */
        {
            str = (char*)ucs_sockaddr_str(NULL, NULL, 0);
            EXPECT_EQ(NULL, str);
        }
    }
}

UCS_TEST_F(test_socket, sock_ipportstr_to_sockaddr) {
    check_ip_port("1.2.3.4", 1234, UCS_OK, "1.2.3.4:1234");
    check_ip_port("1.2.3.4:4567", 1234, UCS_OK, "1.2.3.4:4567");
    check_ip_port("192.168.100.100:60000", 1234, UCS_OK, "192.168.100.100:60000");

    /* Check wrong IPv4:port addresses */
    {
        socket_err_exp_str = "invalid address";
        scoped_log_handler log_handler(socket_error_handler);

        check_ip_port("", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("randomstring", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port(":9999", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("1.2.3.", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("1.1.1.256:9999", 0, UCS_ERR_INVALID_ADDR, "");
    }
    {
        socket_err_exp_str = "invalid port";
        scoped_log_handler log_handler(socket_error_handler);

        check_ip_port("1.1.1.1:-1", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("1.1.1.1:65536", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("1.1.1.1:1111111", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("1.1.1.1:randomstring", 0, UCS_ERR_INVALID_ADDR, "");
    }

    /* TODO: for IPv6 addresses string representation "IPv6%0:port" produced by
     * ucs_sockaddr_str does not conform RFC format: "[IPv6]:port" */

    check_ip_port("::", 1234, UCS_OK, "::%0:1234");
    check_ip_port("::1", 1234, UCS_OK, "::1%0:1234");
    check_ip_port("[::1]:9999", 1234, UCS_OK, "::1%0:9999");
    check_ip_port("::ffff:127.0.0.1", 1234, UCS_OK, "::ffff:127.0.0.1%0:1234");
    check_ip_port("[::ffff:127.0.0.1]:9999", 0, UCS_OK, "::ffff:127.0.0.1%0:9999");
    check_ip_port("1234:1234:1234:1234:1234:1234:1234:1234", 4444, UCS_OK,
                  "1234:1234:1234:1234:1234:1234:1234:1234%0:4444");
    check_ip_port("[1234:1234:1234:1234:1234:1234:1234:1234]:7777", 4444, UCS_OK,
                  "1234:1234:1234:1234:1234:1234:1234:1234%0:7777");

    /* Check wrong IPv6:port addresses */
    {
        socket_err_exp_str = "invalid address";
        scoped_log_handler log_handler(socket_error_handler);

        check_ip_port("", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port(":::", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("::fffg", 0, UCS_ERR_INVALID_ADDR, "");
    }
    {
        socket_err_exp_str = "invalid port";
        scoped_log_handler log_handler(socket_error_handler);

        check_ip_port("[::1]:-1", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("[::1]:65536", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("[::1]:1111111", 0, UCS_ERR_INVALID_ADDR, "");
        check_ip_port("[::1]:randomstring", 0, UCS_ERR_INVALID_ADDR, "");
    }
}

UCS_TEST_F(test_socket, port_from_string) {
    uint16_t port = 0;

    EXPECT_EQ(UCS_OK, ucs_sock_port_from_string("0", &port));
    EXPECT_EQ(0, port);
    EXPECT_EQ(UCS_OK, ucs_sock_port_from_string("1234", &port));
    EXPECT_EQ(1234, port);
    EXPECT_EQ(UCS_OK, ucs_sock_port_from_string("65535", &port));
    EXPECT_EQ(65535, port);

    {
        socket_err_exp_str = "invalid port";
        scoped_log_handler log_handler(socket_error_handler);

        EXPECT_EQ(UCS_ERR_INVALID_ADDR, ucs_sock_port_from_string("", &port));
        EXPECT_EQ(UCS_ERR_INVALID_ADDR, ucs_sock_port_from_string("-1", &port));
        EXPECT_EQ(UCS_ERR_INVALID_ADDR, ucs_sock_port_from_string("str", &port));
        EXPECT_EQ(UCS_ERR_INVALID_ADDR, ucs_sock_port_from_string("65536", &port));
    }
}

UCS_TEST_F(test_socket, socket_setopt) {
    socklen_t optlen;
    int optname;
    int optval;
    int level;
    ucs_status_t status;
    int fd;

    optlen = sizeof(optval);

    status = ucs_socket_create(AF_INET, SOCK_STREAM, &fd);
    EXPECT_UCS_OK(status);
    EXPECT_GE(fd, 0);

    /* with acceptable parameters */
    {
        level   = SOL_SOCKET;
        optname = SO_REUSEADDR;
        optval  = 1;

        status = ucs_socket_setopt(fd, level, optname, &optval, optlen);
        EXPECT_UCS_OK(status);
    }

    /* with bad parameters */
    {
        level   = IPPROTO_TCP;
        optname = SO_REUSEADDR;
        optval  = 1;

        socket_err_exp_str = "failed to set " + ucs::to_string(optname) + " option for " +
                             ucs::to_string(level) + " level on fd " + ucs::to_string(fd) +
                             + ": " + strerror(EINVAL);
        scoped_log_handler log_handler(socket_error_handler);
        status = ucs_socket_setopt(fd, level, optname, &optval, optlen);
        EXPECT_EQ(status, UCS_ERR_IO_ERROR);
    }

    close(fd);
}

static void sockaddr_cmp_test(int sa_family, const char *ip_addr1,
                              const char *ip_addr2, unsigned port1,
                              unsigned port2, struct sockaddr *sa1,
                              struct sockaddr *sa2)
{
    int cmp_res1, cmp_res2;
    ucs_status_t status;

    sa1->sa_family = sa_family;
    sa2->sa_family = sa_family;

    inet_pton(sa_family, ip_addr1,
              const_cast<void*>(ucs_sockaddr_get_inet_addr(sa1)));
    inet_pton(sa_family, ip_addr2,
              const_cast<void*>(ucs_sockaddr_get_inet_addr(sa2)));

    status = ucs_sockaddr_set_port(sa1, port1);
    ASSERT_UCS_OK(status);
    status = ucs_sockaddr_set_port(sa2, port2);
    ASSERT_UCS_OK(status);

    const void *addr1 = ucs_sockaddr_get_inet_addr(sa1);
    const void *addr2 = ucs_sockaddr_get_inet_addr(sa2);

    ASSERT_TRUE(addr1 != NULL);
    ASSERT_TRUE(addr2 != NULL);

    size_t addr_size = ((sa_family == AF_INET) ?
                        sizeof(UCS_SOCKET_INET_ADDR(sa1)) :
                        sizeof(UCS_SOCKET_INET6_ADDR(sa1)));

    // `sa1` vs `sa2`
    {
        int addr_cmp_res = memcmp(addr1, addr2, addr_size);
        int port_cmp_res =
            (port1 == port2) ? 0 : ((port1 < port2) ? -1 : 1);
        int expected_cmp_res =
            addr_cmp_res ? addr_cmp_res : port_cmp_res;

        cmp_res1 = ucs_sockaddr_cmp(sa1, sa2, &status);
        EXPECT_UCS_OK(status);
        EXPECT_EQ(expected_cmp_res, cmp_res1);

        // Call w/o `status` provided
        cmp_res2 = ucs_sockaddr_cmp(sa1, sa2, &status);
        EXPECT_EQ(cmp_res1, cmp_res2);
    }

    // `sa2` vs `sa1`
    {
        int addr_cmp_res = memcmp(addr2, addr1, addr_size);
        int port_cmp_res =
            (port2 == port1) ? 0 : ((port2 < port1) ? -1 : 1);
        int expected_cmp_res =
            addr_cmp_res ? addr_cmp_res : port_cmp_res;

        cmp_res1 = ucs_sockaddr_cmp(sa2, sa1, &status);
        EXPECT_UCS_OK(status);
        EXPECT_EQ(expected_cmp_res, cmp_res1);

        // Call w/o `status` provided
        cmp_res2 = ucs_sockaddr_cmp(sa2, sa1, &status);
        EXPECT_EQ(cmp_res1, cmp_res2);
    }
}

UCS_TEST_F(test_socket, sockaddr_cmp) {
    const unsigned port1        = 65534;
    const unsigned port2        = 65533;
    const char *ipv4_addr1      = "192.168.122.157";
    const char *ipv4_addr2      = "192.168.123.157";
    const char *ipv6_addr1      = "fe80::218:e7ff:fe16:fb97";
    const char *ipv6_addr2      = "fe80::219:e7ff:fe16:fb97";
    struct sockaddr_storage ss1 = {0};
    struct sockaddr_storage ss2 = {0};
    struct sockaddr *saddr1     = (struct sockaddr*)&ss1;
    struct sockaddr *saddr2     = (struct sockaddr*)&ss2;

    // Same addresses; same ports
    sockaddr_cmp_test(AF_INET, ipv4_addr1, ipv4_addr1, port1, port1, saddr1,
                      saddr2);
    sockaddr_cmp_test(AF_INET6, ipv6_addr1, ipv6_addr1, port1, port1, saddr1,
                      saddr2);

    // Same addresses; different ports
    sockaddr_cmp_test(AF_INET, ipv4_addr1, ipv4_addr1, port1, port2, saddr1,
                      saddr2);
    sockaddr_cmp_test(AF_INET6, ipv6_addr1, ipv6_addr1, port1, port2, saddr1,
                      saddr2);

    // Different addresses; same ports
    sockaddr_cmp_test(AF_INET, ipv4_addr1, ipv4_addr2, port1, port1, saddr1,
                      saddr2);
    sockaddr_cmp_test(AF_INET6, ipv6_addr1, ipv6_addr2, port1, port1, saddr1,
                      saddr2);

    // Different addresses; different ports
    sockaddr_cmp_test(AF_INET, ipv4_addr1, ipv4_addr2, port1, port2, saddr1,
                      saddr2);
    sockaddr_cmp_test(AF_INET6, ipv6_addr1, ipv6_addr2, port1, port2, saddr1,
                      saddr2);
}

static void sockaddr_cmp_err_test(const struct sockaddr *sa1,
                                  const struct sockaddr *sa2)
{
    ucs_status_t status;
    int result;

    result = ucs_sockaddr_cmp((const struct sockaddr*)sa1,
                              (const struct sockaddr*)sa2,
                              &status);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    EXPECT_TRUE(result > 0);

    // Call w/o `status` provided
    result = ucs_sockaddr_cmp((const struct sockaddr*)sa1,
                              (const struct sockaddr*)sa2,
                              NULL);
    EXPECT_TRUE(result > 0);
}

UCS_TEST_F(test_socket, sockaddr_cmp_err) {
    // Check with wrong sa_family
    struct sockaddr_un sa_un = {};
    struct sockaddr_in sa_in = {};

    sa_un.sun_family = AF_UNIX;
    sa_in.sin_family = AF_INET;

    socket_err_exp_str = "unknown address family: ";
    scoped_log_handler log_handler(socket_error_handler);

    sockaddr_cmp_err_test((const struct sockaddr*)&sa_un,
                          (const struct sockaddr*)&sa_un);

    sockaddr_cmp_err_test((const struct sockaddr*)&sa_in,
                          (const struct sockaddr*)&sa_un);

    sockaddr_cmp_err_test((const struct sockaddr*)&sa_un,
                          (const struct sockaddr*)&sa_in);
}

UCS_TEST_F(test_socket, subnet_match) {
    /* IPv4 non byte-aligned */
    verify_subnet_match("192.168.122.28", "192.168.122.13", 27);
    verify_subnet_match("192.168.3.28", "192.168.7.13", 21);
    verify_subnet_match("192.45.3.28", "192.14.7.13", 10);

    /* IPv4 byte-aligned */
    verify_subnet_match("192.168.173.157", "10.33.44.100", 0);
    verify_subnet_match("192.168.173.157", "192.33.44.100", 8);
    verify_subnet_match("192.168.173.157", "192.168.44.100", 16);
    verify_subnet_match("192.168.173.157", "192.168.173.100", 24);
    verify_subnet_match("192.168.173.157", "192.168.173.157", 32);

    /* IPv6 non byte-aligned */
    verify_subnet_match("fe80::218:e7ff:fe16:fb97", "fe80::219:e7ff:fe16:fb97",
                        79);
    verify_subnet_match("fe80::218:e7ff:fef8:fb97", "fe80::218:e7ff:feff:fb97",
                        109);

    /* IPv6 byte-aligned */
    verify_subnet_match("fe80::218:e7ff:fe16:fb97", "fe80::218:e77f:fe16:fb97",
                        88);
    verify_subnet_match("fe80::218:e7ff:fe16:fb97", "fe80::218:e7ff:fe16:fb97",
                        128);
}

UCS_TEST_F(test_socket, subnet_match_failure) {
    scoped_log_handler log_handler(socket_error_handler);

    /* Unknown address family */
    socket_err_exp_str = "unknown address family: ";
    verify_subnet_error(AF_UNIX, AF_INET, 10);
    verify_subnet_error(AF_INET, AF_UNIX, 10);

    /* Different address families */
    verify_subnet_error(AF_INET, AF_INET6, 10);
}

static void sockaddr_get_ipstr_check(const struct sockaddr *sockaddr,
                                     ucs_status_t expected_ret,
                                     const char *expected_str = NULL)
{
    const size_t max_size = 1024;
    char str[max_size];

    EXPECT_EQ(expected_ret, ucs_sockaddr_get_ipstr(sockaddr, str, max_size));
    if (expected_str != NULL) {
        EXPECT_STREQ(expected_str, str);
    }
}

static void sockaddr_get_ipstr_check_ip(void *sockaddr, const char *ip_str)
{
    EXPECT_UCS_OK(
            ucs_sock_ipstr_to_sockaddr(ip_str,
                                       (struct sockaddr_storage*)sockaddr));
    sockaddr_get_ipstr_check((const struct sockaddr*)sockaddr, UCS_OK, ip_str);
}

UCS_TEST_F(test_socket, sockaddr_get_ipstr) {
    struct sockaddr_storage ss;
    struct sockaddr *saddr = (struct sockaddr*)&ss;

    /* Check ipv4 */
    saddr->sa_family = AF_INET;
    sockaddr_get_ipstr_check_ip(saddr, "192.168.122.157");

    /* Check ipv6 */
    saddr->sa_family = AF_INET6;
    sockaddr_get_ipstr_check_ip(saddr, "fe80::218:e7ff:fe16:fb97");

    /* Check invalid sa_family */
    socket_err_exp_str = "unknown address family:";
    scoped_log_handler log_handler(socket_error_handler);

    saddr->sa_family = AF_UNIX;
    sockaddr_get_ipstr_check(saddr, UCS_ERR_INVALID_PARAM);
}
