/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/sock.h>
}

#include <sys/un.h>

static std::string socket_err_exp_str;

class test_socket : public ucs::test {
public:
protected:

    static ucs_log_func_rc_t
    socket_error_handler(const char *file, unsigned line, const char *function,
                         ucs_log_level_t level, const char *message, va_list ap)
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
};

UCS_TEST_F(test_socket, sockaddr_sizeof) {
    struct sockaddr_in sa_in   = {
        .sin_family            = AF_INET,
    };
    struct sockaddr_in6 sa_in6 = {
        .sin6_family           = AF_INET6,
    };
    struct sockaddr_un sa_un = {
        .sun_family            = AF_UNIX,
    };
    size_t size;

    /* Check with wrong IPv4 */
    {
        size = 0;
        EXPECT_EQ(UCS_OK, ucs_sockaddr_sizeof((const struct sockaddr*)&sa_in, &size));
        EXPECT_EQ(sizeof(struct sockaddr_in), size);
    }

    /* Check with wrong IPv6 */
    {
        size = 0;
        EXPECT_EQ(UCS_OK, ucs_sockaddr_sizeof((const struct sockaddr*)&sa_in6, &size));
        EXPECT_EQ(sizeof(struct sockaddr_in6), size);
    }

    /* Check with wrong address family */
    {
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        size = 0;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  ucs_sockaddr_sizeof((const struct sockaddr*)&sa_un, &size));
        /* Check that doesn't touch provided memory in error case */
        EXPECT_EQ(0ULL, size);
    }
}

UCS_TEST_F(test_socket, sockaddr_get_port) {
    const unsigned sin_port    = 5555;
    struct sockaddr_in sa_in   = {
        .sin_family            = AF_INET,
        .sin_port              = htons(sin_port),
    };
    struct sockaddr_in6 sa_in6 = {
        .sin6_family           = AF_INET6,
        .sin6_port             = htons(sin_port),
    };
    struct sockaddr_un sa_un = {
        .sun_family            = AF_UNIX,
    };
    unsigned port = 0;

    /* Check with wrong IPv4 */
    {
        port = 0;
        EXPECT_EQ(UCS_OK, ucs_sockaddr_get_port((const struct sockaddr*)&sa_in, &port));
        EXPECT_EQ(sin_port, port);
    }

    /* Check with wrong IPv6 */
    {
        port = 0;
        EXPECT_EQ(UCS_OK, ucs_sockaddr_get_port((const struct sockaddr*)&sa_in6, &port));
        EXPECT_EQ(sin_port, port);
    }

    /* Check with wrong address family */
    {
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        port = sin_port;
        EXPECT_EQ(UCS_ERR_INVALID_PARAM,
                  ucs_sockaddr_get_port((const struct sockaddr*)&sa_un, &port));
        /* Check that doesn't touch provided memory in error case */
        EXPECT_EQ(sin_port, port);
    }
}

UCS_TEST_F(test_socket, sockaddr_get_inet_addr) {
    struct sockaddr_in sa_in   = {
        .sin_family            = AF_INET,
    };
    struct sockaddr_in6 sa_in6 = {
        .sin6_family           = AF_INET6,
    };
    struct sockaddr_un sa_un   = {
        .sun_family            = AF_UNIX,
    };
    struct in_addr sin_addr;
    struct in6_addr sin6_addr;

    sin_addr.s_addr = sa_in.sin_addr.s_addr = htonl(INADDR_ANY);
    sin6_addr       = sa_in6.sin6_addr      = in6addr_any;

    /* Check with wrong IPv4 */
    {
        EXPECT_EQ(&sa_in.sin_addr,
                  ucs_sockaddr_get_inet_addr((const struct sockaddr*)&sa_in));
        EXPECT_EQ(0, memcmp(&sa_in.sin_addr, &sin_addr,
                            sizeof(sa_in.sin_addr)));
    }

    /* Check with wrong IPv6 */
    {
        EXPECT_EQ(&sa_in6.sin6_addr,
                  ucs_sockaddr_get_inet_addr((const struct sockaddr*)&sa_in6));
        EXPECT_EQ(0, memcmp(&sa_in6.sin6_addr, &sin6_addr,
                            sizeof(sa_in6.sin6_addr)));
    }

    /* Check with wrong address family */
    {
        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        EXPECT_EQ(NULL, ucs_sockaddr_get_inet_addr((const struct sockaddr*)&sa_un));
    }
}

UCS_TEST_F(test_socket, sockaddr_str) {
    const unsigned port        = 65534;
    const char *ipv4_addr      = "192.168.122.157";
    const char *ipv6_addr      = "fe80::218:e7ff:fe16:fb97";
    struct sockaddr_in sa_in   = {
        .sin_family            = AF_INET,
        .sin_port              = htons(port),
    };
    struct sockaddr_in6 sa_in6 = {
        .sin6_family           = AF_INET6,
        .sin6_port             = htons(port),
    };
    char ipv4_addr_out[128], ipv6_addr_out[128], *str, test_str[1024];

    sprintf(ipv4_addr_out, "%s:%d", ipv4_addr, port);
    sprintf(ipv6_addr_out, "%s:%d", ipv6_addr, port);

    inet_pton(AF_INET, ipv4_addr, &(sa_in.sin_addr));
    inet_pton(AF_INET6, ipv6_addr, &(sa_in6.sin6_addr));

    /* Check with short `str_len` to fit IP address only */
    {
        str = (char*)ucs_sockaddr_str((const struct sockaddr*)&sa_in,
                                      test_str, INET_ADDRSTRLEN);
        EXPECT_EQ(str, test_str);
        EXPECT_EQ(0, strncmp(test_str, ipv4_addr_out,
                             INET_ADDRSTRLEN - 1));

        str = (char*)ucs_sockaddr_str((const struct sockaddr*)&sa_in6,
                                      test_str, INET6_ADDRSTRLEN);
        EXPECT_EQ(str, test_str);
        EXPECT_EQ(0, strncmp(test_str, ipv6_addr_out,
                             INET6_ADDRSTRLEN - 1));
    }

    /* Check with big enough `str_len` */
    {
        str = (char*)ucs_sockaddr_str((const struct sockaddr*)&sa_in,
                                      test_str, 1024);
        EXPECT_EQ(str, test_str);
        EXPECT_EQ(0, strcmp(test_str, ipv4_addr_out));

        str = (char*)ucs_sockaddr_str((const struct sockaddr*)&sa_in6,
                                      test_str, 1024);
        EXPECT_TRUE(str == test_str);
        EXPECT_EQ(0, strcmp(test_str, ipv6_addr_out));
    }

    /* Check with wrong sa_family */
    {
        struct sockaddr_un sa_un = {
            .sun_family          = AF_UNIX,
        };

        /* with big enough string */
        {
            str = (char*)ucs_sockaddr_str((const struct sockaddr*)&sa_un,
                                          test_str, 1024);
            EXPECT_EQ(test_str, str);
            EXPECT_EQ(0, strcmp(str, "<invalid address family>"));
        }

        /* without string */
        {
            str = (char*)ucs_sockaddr_str((const struct sockaddr*)&sa_un,
                                          NULL, 0);
            EXPECT_EQ(NULL, str);
        }
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

UCS_TEST_F(test_socket, sockaddr_addr_v4_cmp) {
    const unsigned port1       = 65534;
    const unsigned port2       = 65533;
    const char *ipv4_addr1     = "192.168.122.157";
    const char *ipv4_addr2     = "192.168.123.157";
    struct sockaddr_in sa_in_1 = {
        .sin_family            = AF_INET,
    };
    struct sockaddr_in sa_in_2 = {
        .sin_family            = AF_INET,
    };
    ucs_status_t status;
    int result;
    
    // Different ports shouldn't have any effect
    sa_in_1.sin_port = htons(port1);
    sa_in_2.sin_port = htons(port2);

    // Same addresses
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_1));
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_2));
    status = ucs_sockaddr_addr_cmp((const struct sockaddr*)&sa_in_1,
                                   (const struct sockaddr*)&sa_in_2,
                                   &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(0, result);

    // Different addresses
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_1));
    inet_pton(AF_INET, ipv4_addr2, &UCS_SOCKET_INET_ADDR(&sa_in_2));
    status = ucs_sockaddr_addr_cmp((const struct sockaddr*)&sa_in_1,
                                   (const struct sockaddr*)&sa_in_2,
                                   &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET_ADDR(&sa_in_1),
                             &UCS_SOCKET_INET_ADDR(&sa_in_2),
                             sizeof(UCS_SOCKET_INET_ADDR(&sa_in_1))));
}

UCS_TEST_F(test_socket, sockaddr_port_v4_cmp) {
    const unsigned port1       = 65534;
    const unsigned port2       = 65533;
    const char *ipv4_addr1     = "192.168.122.157";
    const char *ipv4_addr2     = "192.168.123.157";
    struct sockaddr_in sa_in_1 = {
        .sin_family            = AF_INET,
    };
    struct sockaddr_in sa_in_2 = {
        .sin_family            = AF_INET,
    };
    ucs_status_t status;
    int result;

    // Different addresses shouldn't have any effect
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_1));
    inet_pton(AF_INET, ipv4_addr2, &UCS_SOCKET_INET_ADDR(&sa_in_2));

    // Same ports
    sa_in_1.sin_port = htons(port1);
    sa_in_2.sin_port = htons(port1);
    status = ucs_sockaddr_port_cmp((const struct sockaddr*)&sa_in_1,
                                   (const struct sockaddr*)&sa_in_2,
                                   &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(0, result);

    // Different ports
    sa_in_1.sin_port = htons(port1);
    sa_in_2.sin_port = htons(port2);
    status = ucs_sockaddr_port_cmp((const struct sockaddr*)&sa_in_1,
                                   (const struct sockaddr*)&sa_in_2,
                                   &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET_PORT(&sa_in_1),
                             &UCS_SOCKET_INET_PORT(&sa_in_2),
                             sizeof(UCS_SOCKET_INET_PORT(&sa_in_1))));
}

UCS_TEST_F(test_socket, sockaddr_v4_cmp) {    
    const unsigned port1       = 65534;
    const unsigned port2       = 65533;
    const char *ipv4_addr1     = "192.168.122.157";
    const char *ipv4_addr2     = "192.168.123.157";
    struct sockaddr_in sa_in_1 = {
        .sin_family            = AF_INET,
    };
    struct sockaddr_in sa_in_2 = {
        .sin_family            = AF_INET,
    };
    ucs_status_t status;
    int result;

    // Same addresses; same ports
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_1));
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_2));
    sa_in_1.sin_port = htons(port1);
    sa_in_2.sin_port = htons(port1);
    status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_in_1,
                              (const struct sockaddr*)&sa_in_2,
                              &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(0, result);

    // Same addresses; different ports
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_1));
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_2));
    sa_in_1.sin_port = htons(port1);
    sa_in_2.sin_port = htons(port2);
    status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_in_1,
                              (const struct sockaddr*)&sa_in_2,
                              &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET_PORT(&sa_in_1),
                             &UCS_SOCKET_INET_PORT(&sa_in_2),
                             sizeof(UCS_SOCKET_INET_PORT(&sa_in_1))));

    // Different addresses; same ports
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_1));
    inet_pton(AF_INET, ipv4_addr2, &UCS_SOCKET_INET_ADDR(&sa_in_2));
    sa_in_1.sin_port = htons(port1);
    sa_in_2.sin_port = htons(port1);
    status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_in_1,
                              (const struct sockaddr*)&sa_in_2,
                              &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET_ADDR(&sa_in_1),
                             &UCS_SOCKET_INET_ADDR(&sa_in_2),
                             sizeof(UCS_SOCKET_INET_ADDR(&sa_in_1))));

    // Different addresses; different ports
    inet_pton(AF_INET, ipv4_addr1, &UCS_SOCKET_INET_ADDR(&sa_in_1));
    inet_pton(AF_INET, ipv4_addr2, &UCS_SOCKET_INET_ADDR(&sa_in_2));
    sa_in_1.sin_port = htons(port1);
    sa_in_2.sin_port = htons(port2);
    status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_in_1,
                              (const struct sockaddr*)&sa_in_2,
                              &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET_ADDR(&sa_in_1),
                             &UCS_SOCKET_INET_ADDR(&sa_in_2),
                             sizeof(UCS_SOCKET_INET_ADDR(&sa_in_1))));
}

UCS_TEST_F(test_socket, sockaddr_addr_v6_cmp) {
    const unsigned port1         = 65534;
    const unsigned port2         = 65533;
    const char *ipv6_addr1       = "fe80::218:e7ff:fe16:fb97";
    const char *ipv6_addr2       = "fe80::219:e7ff:fe16:fb97";
    struct sockaddr_in6 sa_in6_1 = {
        .sin6_family             = AF_INET6,
    };
    struct sockaddr_in6 sa_in6_2 = {
        .sin6_family             = AF_INET6,
    };
    ucs_status_t status;
    int result;

    // Different ports shouldn't have any effect
    sa_in6_1.sin6_port = htons(port1);
    sa_in6_2.sin6_port = htons(port2);

    // Same addresses
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_1));
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_2));
    status = ucs_sockaddr_addr_cmp((const struct sockaddr*)&sa_in6_1,
                                   (const struct sockaddr*)&sa_in6_2,
                                   &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(0, result);

    // Different addresses
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_1));
    inet_pton(AF_INET6, ipv6_addr2, &UCS_SOCKET_INET6_ADDR(&sa_in6_2));
    status = ucs_sockaddr_addr_cmp((const struct sockaddr*)&sa_in6_1,
                                   (const struct sockaddr*)&sa_in6_2,
                                   &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET6_ADDR(&sa_in6_1),
                             &UCS_SOCKET_INET6_ADDR(&sa_in6_2),
                             sizeof(UCS_SOCKET_INET6_ADDR(&sa_in6_1))));
}

UCS_TEST_F(test_socket, sockaddr_port_v6_cmp) {
    const unsigned port1         = 65534;
    const unsigned port2         = 65533;
    const char *ipv6_addr1       = "fe80::218:e7ff:fe16:fb97";
    const char *ipv6_addr2       = "fe80::219:e7ff:fe16:fb97";
    struct sockaddr_in6 sa_in6_1 = {
        .sin6_family             = AF_INET6,
    };
    struct sockaddr_in6 sa_in6_2 = {
        .sin6_family             = AF_INET6,
    };
    ucs_status_t status;
    int result;

    // Different addresses shouldn't have any effect
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_1));
    inet_pton(AF_INET6, ipv6_addr2, &UCS_SOCKET_INET6_ADDR(&sa_in6_2));

    // Same ports
    sa_in6_1.sin6_port = htons(port1);
    sa_in6_2.sin6_port = htons(port1);
    status = ucs_sockaddr_port_cmp((const struct sockaddr*)&sa_in6_1,
                                   (const struct sockaddr*)&sa_in6_2,
                                   &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(0, result);

    // Different ports
    sa_in6_1.sin6_port = htons(port1);
    sa_in6_2.sin6_port = htons(port2);
    status = ucs_sockaddr_port_cmp((const struct sockaddr*)&sa_in6_1,
                                   (const struct sockaddr*)&sa_in6_2,
                                   &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET6_PORT(&sa_in6_1),
                             &UCS_SOCKET_INET6_PORT(&sa_in6_2),
                             sizeof(UCS_SOCKET_INET6_PORT(&sa_in6_1))));
}

UCS_TEST_F(test_socket, sockaddr_v6_cmp) {
    const unsigned port1         = 65534;
    const unsigned port2         = 65533;
    const char *ipv6_addr1       = "fe80::218:e7ff:fe16:fb97";
    const char *ipv6_addr2       = "fe80::219:e7ff:fe16:fb97";
    struct sockaddr_in6 sa_in6_1 = {
        .sin6_family             = AF_INET6,
    };
    struct sockaddr_in6 sa_in6_2 = {
        .sin6_family             = AF_INET6,
    };
    ucs_status_t status;
    int result;

    // Same addresses; same ports
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_1));
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_2));
    sa_in6_1.sin6_port = htons(port1);
    sa_in6_2.sin6_port = htons(port1);
    status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_in6_1,
                              (const struct sockaddr*)&sa_in6_2,
                              &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(0, result);

    // Same addresses; different ports
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_1));
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_2));
    sa_in6_1.sin6_port = htons(port1);
    sa_in6_2.sin6_port = htons(port2);
    status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_in6_1,
                              (const struct sockaddr*)&sa_in6_2,
                              &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET6_PORT(&sa_in6_1),
                             &UCS_SOCKET_INET6_PORT(&sa_in6_2),
                             sizeof(UCS_SOCKET_INET6_PORT(&sa_in6_1))));

    // Different addresses; same ports
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_1));
    inet_pton(AF_INET6, ipv6_addr2, &UCS_SOCKET_INET6_ADDR(&sa_in6_2));
    sa_in6_1.sin6_port = htons(port1);
    sa_in6_2.sin6_port = htons(port1);
    status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_in6_1,
                              (const struct sockaddr*)&sa_in6_2,
                              &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET6_ADDR(&sa_in6_1),
                             &UCS_SOCKET_INET6_ADDR(&sa_in6_2),
                             sizeof(UCS_SOCKET_INET6_ADDR(&sa_in6_1))));

    // Different addresses; different ports
    inet_pton(AF_INET6, ipv6_addr1, &UCS_SOCKET_INET6_ADDR(&sa_in6_1));
    inet_pton(AF_INET6, ipv6_addr2, &UCS_SOCKET_INET6_ADDR(&sa_in6_2));
    sa_in6_1.sin6_port = htons(port1);
    sa_in6_2.sin6_port = htons(port2);
    status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_in6_1,
                              (const struct sockaddr*)&sa_in6_2,
                              &result);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(result, memcmp(&UCS_SOCKET_INET6_ADDR(&sa_in6_1),
                             &UCS_SOCKET_INET6_ADDR(&sa_in6_2),
                             sizeof(UCS_SOCKET_INET6_ADDR(&sa_in6_1))));
}

UCS_TEST_F(test_socket, sockaddr_cmp_err) {
    const unsigned port        = 65534;
    const char *ipv4_addr      = "192.168.122.157";
    const char *ipv6_addr      = "fe80::218:e7ff:fe16:fb97";
    struct sockaddr_in sa_in   = {
        .sin_family            = AF_INET,
        .sin_port              = htons(port),
    };
    struct sockaddr_in6 sa_in6 = {
        .sin6_family           = AF_INET6,
        .sin6_port             = htons(port),
    };
    ucs_status_t status;
    int result;

    inet_pton(AF_INET, ipv4_addr, &UCS_SOCKET_INET6_ADDR(&sa_in));
    inet_pton(AF_INET6, ipv6_addr, &UCS_SOCKET_INET6_ADDR(&sa_in6));

    // Check with wrong sa_family
    {
        struct sockaddr_un sa_un = {
            .sun_family          = AF_UNIX,
        };

        socket_err_exp_str = "unknown address family:";
        scoped_log_handler log_handler(socket_error_handler);

        // When fails, shouldn't touch the `result` argument
        result = 100;

        status = ucs_sockaddr_addr_cmp((const struct sockaddr*)&sa_un,
                                       (const struct sockaddr*)&sa_un,
                                       &result);
        EXPECT_EQ(status, UCS_ERR_INVALID_PARAM);
        EXPECT_EQ(100, result);

        status = ucs_sockaddr_port_cmp((const struct sockaddr*)&sa_un,
                                       (const struct sockaddr*)&sa_un,
                                       &result);
        EXPECT_EQ(status, UCS_ERR_INVALID_PARAM);
        EXPECT_EQ(100, result);

        status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_un,
                                  (const struct sockaddr*)&sa_un,
                                  &result);
        EXPECT_EQ(status, UCS_ERR_INVALID_PARAM);
        EXPECT_EQ(100, result);
    }

    // Check with different sa_family's
    {
        socket_err_exp_str = "unable to compare socket addresses with " \
            "different address families:";
        scoped_log_handler log_handler(socket_error_handler);

        // When fails, shouldn't touch the `result` argument
        result = 100;

        status = ucs_sockaddr_addr_cmp((const struct sockaddr*)&sa_in,
                                       (const struct sockaddr*)&sa_in6,
                                       &result);
        EXPECT_EQ(status, UCS_ERR_INVALID_PARAM);
        EXPECT_EQ(100, result);

        // When fails, shouldn't touch the `result` argument
        result = 100;

        status = ucs_sockaddr_port_cmp((const struct sockaddr*)&sa_in,
                                       (const struct sockaddr*)&sa_in6,
                                       &result);
        EXPECT_EQ(status, UCS_ERR_INVALID_PARAM);
        EXPECT_EQ(100, result);

        // When fails, shouldn't touch the `result` argument
        result = 100;

        status = ucs_sockaddr_cmp((const struct sockaddr*)&sa_in,
                                  (const struct sockaddr*)&sa_in6,
                                  &result);
        EXPECT_EQ(status, UCS_ERR_INVALID_PARAM);
        EXPECT_EQ(100, result);
    }
}

UCS_TEST_F(test_socket, sockaddr_v4_copy) {
    const unsigned port      = 65534;
    const char *ipv4_addr    = "192.168.122.157";
    struct sockaddr_in sa_in = {
        .sin_family          = AF_INET,
        .sin_port            = htons(port),
    };
    struct sockaddr_in sa_in_res = { 0 };
    ucs_status_t status;
    size_t length;

    inet_pton(AF_INET, ipv4_addr, &UCS_SOCKET_INET6_ADDR(&sa_in));

    status = ucs_sockaddr_sizeof((const struct sockaddr*)&sa_in, &length);
    EXPECT_EQ(UCS_OK, status);

    status = ucs_sockaddr_copy((struct sockaddr*)&sa_in_res,
                               (const struct sockaddr*)&sa_in);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(0, memcmp(&sa_in, &sa_in_res, length));
}

UCS_TEST_F(test_socket, sockaddr_v6_copy) {
    const unsigned port        = 65534;
    const char *ipv6_addr      = "fe80::218:e7ff:fe16:fb97";
    struct sockaddr_in6 sa_in6 = {
        .sin6_family           = AF_INET6,
        .sin6_port             = htons(port),
    };
    struct sockaddr_in6 sa_in6_res = { 0 };
    ucs_status_t status;
    size_t length;

    inet_pton(AF_INET6, ipv6_addr, &UCS_SOCKET_INET6_ADDR(&sa_in6));

    status = ucs_sockaddr_sizeof((const struct sockaddr*)&sa_in6, &length);
    EXPECT_EQ(UCS_OK, status);

    status = ucs_sockaddr_copy((struct sockaddr*)&sa_in6_res,
                               (const struct sockaddr*)&sa_in6);
    EXPECT_EQ(UCS_OK, status);
    EXPECT_EQ(0, memcmp(&sa_in6, &sa_in6_res, length));
}

UCS_TEST_F(test_socket, sockaddr_copy_err) {
    struct sockaddr_un sa_un       = {
        .sun_family                = AF_UNIX,
    };
    struct sockaddr_un sa_un_res   = { 0 };
    struct sockaddr_un sa_un_check = { 0 };
    ucs_status_t status;

    socket_err_exp_str = "unknown address family:";
    scoped_log_handler log_handler(socket_error_handler);
    status = ucs_sockaddr_copy((struct sockaddr*)&sa_un_res,
                               (const struct sockaddr*)&sa_un);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    // When fails, shouldn't touch the `to` argument
    EXPECT_EQ(0, memcmp(&sa_un_check, &sa_un_res, sizeof(sa_un)));
}
