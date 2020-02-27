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
};

UCS_TEST_F(test_socket, sockaddr_sizeof) {
    struct sockaddr_in sa_in;
    struct sockaddr_in6 sa_in6;
    struct sockaddr_un sa_un;
    size_t size;

    sa_in.sin_family   = AF_INET;
    sa_in6.sin6_family = AF_INET6;
    sa_un.sun_family   = AF_UNIX;

    /* Check with wrong IPv4 */
    {
        size = 0;
        EXPECT_UCS_OK(ucs_sockaddr_sizeof((const struct sockaddr*)&sa_in, &size));
        EXPECT_EQ(sizeof(struct sockaddr_in), size);
    }

    /* Check with wrong IPv6 */
    {
        size = 0;
        EXPECT_UCS_OK(ucs_sockaddr_sizeof((const struct sockaddr*)&sa_in6, &size));
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
    const uint16_t sin_port    = 5555;
    struct sockaddr_in sa_in;
    struct sockaddr_in6 sa_in6;
    struct sockaddr_un sa_un;
    uint16_t port = 0;

    sa_in.sin_family   = AF_INET;
    sa_in.sin_port     = htons(sin_port);
    sa_in6.sin6_family = AF_INET6;
    sa_in6.sin6_port   = htons(sin_port);
    sa_un.sun_family   = AF_UNIX;

    /* Check with wrong IPv4 */
    {
        port = 0;
        EXPECT_UCS_OK(ucs_sockaddr_get_port((const struct sockaddr*)&sa_in, &port));
        EXPECT_EQ(sin_port, port);
    }

    /* Check with wrong IPv6 */
    {
        port = 0;
        EXPECT_UCS_OK(ucs_sockaddr_get_port((const struct sockaddr*)&sa_in6, &port));
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
    struct sockaddr_in sa_in;
    struct sockaddr_in6 sa_in6;
    struct sockaddr_un sa_un;
    struct in_addr sin_addr;
    struct in6_addr sin6_addr;

    sa_in.sin_family   = AF_INET;
    sa_in6.sin6_family = AF_INET6;
    sa_un.sun_family   = AF_UNIX;

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
    const uint16_t port        = 65534;
    const char *ipv4_addr      = "192.168.122.157";
    const char *ipv6_addr      = "fe80::218:e7ff:fe16:fb97";
    struct sockaddr_in sa_in;
    struct sockaddr_in6 sa_in6;
    char ipv4_addr_out[128], ipv6_addr_out[128], *str, test_str[1024];

    sa_in.sin_family   = AF_INET;
    sa_in.sin_port     = htons(port);
    sa_in6.sin6_family = AF_INET6;
    sa_in6.sin6_port   = htons(port);

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
        struct sockaddr_un sa_un;
        sa_un.sun_family = AF_UNIX;

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
    const unsigned port1         = 65534;
    const unsigned port2         = 65533;
    const char *ipv4_addr1       = "192.168.122.157";
    const char *ipv4_addr2       = "192.168.123.157";
    const char *ipv6_addr1       = "fe80::218:e7ff:fe16:fb97";
    const char *ipv6_addr2       = "fe80::219:e7ff:fe16:fb97";
    struct sockaddr_in sa_in_1   = { 0 };
    struct sockaddr_in sa_in_2   = { 0 };
    struct sockaddr_in6 sa_in6_1 = { 0 };
    struct sockaddr_in6 sa_in6_2 = { 0 };

    // Same addresses; same ports
    sockaddr_cmp_test(AF_INET, ipv4_addr1, ipv4_addr1,
                      port1, port1,
                      (struct sockaddr*)&sa_in_1,
                      (struct sockaddr*)&sa_in_2);
    sockaddr_cmp_test(AF_INET6, ipv6_addr1, ipv6_addr1,
                      port1, port1,
                      (struct sockaddr*)&sa_in6_1,
                      (struct sockaddr*)&sa_in6_2);

    // Same addresses; different ports
    sockaddr_cmp_test(AF_INET, ipv4_addr1, ipv4_addr1,
                      port1, port2,
                      (struct sockaddr*)&sa_in_1,
                      (struct sockaddr*)&sa_in_2);
    sockaddr_cmp_test(AF_INET6, ipv6_addr1, ipv6_addr1,
                      port1, port2,
                      (struct sockaddr*)&sa_in6_1,
                      (struct sockaddr*)&sa_in6_2);

    // Different addresses; same ports
    sockaddr_cmp_test(AF_INET, ipv4_addr1, ipv4_addr2,
                      port1, port1,
                      (struct sockaddr*)&sa_in_1,
                      (struct sockaddr*)&sa_in_2);
    sockaddr_cmp_test(AF_INET6, ipv6_addr1, ipv6_addr2,
                      port1, port1,
                      (struct sockaddr*)&sa_in6_1,
                      (struct sockaddr*)&sa_in6_2);

    // Different addresses; different ports
    sockaddr_cmp_test(AF_INET, ipv4_addr1, ipv4_addr2,
                      port1, port2,
                      (struct sockaddr*)&sa_in_1,
                      (struct sockaddr*)&sa_in_2);
    sockaddr_cmp_test(AF_INET6, ipv6_addr1, ipv6_addr2,
                      port1, port2,
                      (struct sockaddr*)&sa_in6_1,
                      (struct sockaddr*)&sa_in6_2);
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
    struct sockaddr_un sa_un;
    struct sockaddr_in sa_in;

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
