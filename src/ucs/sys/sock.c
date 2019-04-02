/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/math.h>

#include <unistd.h>
#include <errno.h>
#include <string.h>


ucs_status_t ucs_netif_ioctl(const char *if_name, unsigned long request,
                             struct ifreq *if_req)
{
    ucs_status_t status;
    int fd, ret;

    ucs_strncpy_zero(if_req->ifr_name, if_name, sizeof(if_req->ifr_name));

    status = ucs_socket_create(AF_INET, SOCK_STREAM, &fd);
    if (status != UCS_OK) {
        goto out;
    }

    ret = ioctl(fd, request, if_req);
    if (ret < 0) {
        ucs_debug("ioctl(req=%lu, ifr_name=%s) failed: %m", request, if_name);
        status = UCS_ERR_IO_ERROR;
        goto out_close_fd;
    }

    status = UCS_OK;

out_close_fd:
    close(fd);
out:
    return status;
}

int ucs_netif_is_active(const char *if_name)
{
    ucs_status_t status;
    struct ifreq ifr;

    status = ucs_netif_ioctl(if_name, SIOCGIFADDR, &ifr);
    if (status != UCS_OK) {
        return 0;
    }

    status = ucs_netif_ioctl(if_name, SIOCGIFFLAGS, &ifr);
    if (status != UCS_OK) {
        return 0;
    }

    return (ifr.ifr_flags & IFF_UP) && (ifr.ifr_flags & IFF_RUNNING) &&
           !(ifr.ifr_flags & IFF_LOOPBACK);
}

ucs_status_t ucs_socket_create(int domain, int type, int *fd_p)
{
    int fd = socket(domain, type, 0);
    if (fd < 0) {
        ucs_error("socket create failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    *fd_p = fd;
    return UCS_OK;
}

ucs_status_t ucs_socket_setopt(int fd, int level, int optname,
                               const void *optval, socklen_t optlen)
{
    int ret = setsockopt(fd, level, optname, optval, optlen);
    if (ret < 0) {
        ucs_error("failed to set %d option for %d level on fd %d: %m",
                  optname, level, fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t ucs_socket_connect(int fd, const struct sockaddr *dest_addr)
{
    char str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    size_t addr_size;
    int ret;

    status = ucs_sockaddr_sizeof(dest_addr, &addr_size);
    if (status != UCS_OK) {
        return status;
    }

    do {
        ret = connect(fd, dest_addr, addr_size);
        if (ret < 0) {
            if (errno == EINPROGRESS) {
                status = UCS_INPROGRESS;
                goto out;
            }

            if (errno == EISCONN) {
                status = UCS_ERR_ALREADY_EXISTS;
                goto out;
            }

            if (errno != EINTR) {
                ucs_error("connect(fd=%d, dest_addr=%s) failed: %m", fd,
                          ucs_sockaddr_str(dest_addr, str, UCS_SOCKADDR_STRING_LEN));
                return UCS_ERR_UNREACHABLE;
            }
        }
    } while ((ret < 0) && (errno == EINTR));

out:
    ucs_debug("connect(fd=%d, dest_addr=%s): %m", fd,
              ucs_sockaddr_str(dest_addr, str, UCS_SOCKADDR_STRING_LEN));
    return status;
}

ucs_status_t ucs_socket_connect_nb_get_status(int fd)
{
    socklen_t conn_status_sz;
    int ret, conn_status;

    conn_status_sz = sizeof(conn_status);

    ret = getsockopt(fd, SOL_SOCKET, SO_ERROR,
                     &conn_status, &conn_status_sz);
    if (ret < 0) {
        ucs_error("getsockopt(fd=%d) failed to get SOL_SOCKET(SO_ERROR): %m", fd);
        return UCS_ERR_IO_ERROR;
    }

    if ((conn_status == EINPROGRESS) || (conn_status == EWOULDBLOCK)) {
        return UCS_INPROGRESS;
    }

    if (conn_status != 0) {
        ucs_error("SOL_SOCKET(SO_ERROR) status on fd %d: %s", fd, strerror(conn_status));
        return UCS_ERR_UNREACHABLE;
    }

    return UCS_OK;
}

ucs_status_t ucs_sockaddr_sizeof(const struct sockaddr *addr, size_t *size_p)
{
    switch (addr->sa_family) {
    case AF_INET:
        *size_p = sizeof(struct sockaddr_in);
        return UCS_OK;
    case AF_INET6:
        *size_p = sizeof(struct sockaddr_in6);
        return UCS_OK;
    default:
        ucs_error("unknown address family: %d", addr->sa_family);
        return UCS_ERR_INVALID_PARAM;
    }
}

ucs_status_t ucs_sockaddr_get_port(const struct sockaddr *addr, unsigned *port_p)
{
    switch (addr->sa_family) {
    case AF_INET:
        *port_p = ntohs(UCS_SOCKET_INET_PORT(addr));
        return UCS_OK;
    case AF_INET6:
        *port_p = ntohs(UCS_SOCKET_INET6_PORT(addr));
        return UCS_OK;
    default:
        ucs_error("unknown address family: %d", addr->sa_family);
        return UCS_ERR_INVALID_PARAM;
    }
}

ucs_status_t ucs_sockaddr_set_port(struct sockaddr *addr, unsigned port)
{
    switch (addr->sa_family) {
    case AF_INET:
        UCS_SOCKET_INET_PORT(addr) = htons(port);
        return UCS_OK;
    case AF_INET6:
        UCS_SOCKET_INET6_PORT(addr) = htons(port);
        return UCS_OK;
    default:
        ucs_error("unknown address family: %d", addr->sa_family);
        return UCS_ERR_INVALID_PARAM;
    }
}

const void *ucs_sockaddr_get_inet_addr(const struct sockaddr *addr)
{
    switch (addr->sa_family) {
    case AF_INET:
        return &UCS_SOCKET_INET_ADDR(addr);
    case AF_INET6:
        return &UCS_SOCKET_INET6_ADDR(addr);
    default:
        ucs_error("unknown address family: %d", addr->sa_family);
        return NULL;
    }
}

const char* ucs_sockaddr_str(const struct sockaddr *sock_addr,
                             char *str, size_t max_size)
{
    unsigned port;
    size_t str_len;

    if ((sock_addr->sa_family != AF_INET) && (sock_addr->sa_family != AF_INET6)) {
        ucs_strncpy_zero(str, "<invalid address family>", max_size);
        return str;
    }

    if (!inet_ntop(sock_addr->sa_family, ucs_sockaddr_get_inet_addr(sock_addr),
                   str, max_size)) {
        ucs_strncpy_zero(str, "<failed to convert sockaddr to string>", max_size);
        return str;
    }

    if (ucs_sockaddr_get_port(sock_addr, &port) != UCS_OK) {
        ucs_strncpy_zero(str, "<unable to get port>", max_size);
        return str;
    }

    str_len = strlen(str);

    ucs_snprintf_zero(str + str_len, max_size - str_len, ":%d", port);

    return str;
}

ucs_status_t ucs_sockaddr_addr_cmp(const struct sockaddr *sa1,
                                   const struct sockaddr *sa2,
                                   int *result_p)
{
    if (sa1->sa_family != sa2->sa_family) {
        ucs_error("unable to compare socket addresses with "
                  "different address families: %d vs %d",
                  sa1->sa_family, sa2->sa_family);
        return UCS_ERR_INVALID_PARAM;
    }

    switch (sa1->sa_family) {
    case AF_INET:
        *result_p = memcmp(&UCS_SOCKET_INET_ADDR(sa1),
                           &UCS_SOCKET_INET_ADDR(sa2),
                           sizeof(UCS_SOCKET_INET_ADDR(sa1)));
        return UCS_OK;
    case AF_INET6:
        *result_p = memcmp(&UCS_SOCKET_INET6_ADDR(sa1),
                           &UCS_SOCKET_INET6_ADDR(sa2),
                           sizeof(UCS_SOCKET_INET6_ADDR(sa1)));
        return UCS_OK;
    default:
        ucs_error("unknown address family: %d", sa1->sa_family);
        return UCS_ERR_INVALID_PARAM;
    }
}

ucs_status_t ucs_sockaddr_port_cmp(const struct sockaddr *sa1,
                                   const struct sockaddr *sa2,
                                   int *result_p)
{
    if (sa1->sa_family != sa2->sa_family) {
        ucs_error("unable to compare socket addresses with "
                  "different address families: %d vs %d",
                  sa1->sa_family, sa2->sa_family);
        return UCS_ERR_INVALID_PARAM;
    }

    switch (sa1->sa_family) {
    case AF_INET:
        *result_p = memcmp(&UCS_SOCKET_INET_PORT(sa1),
                           &UCS_SOCKET_INET_PORT(sa2),
                           sizeof(UCS_SOCKET_INET_PORT(sa1)));
        return UCS_OK;
    case AF_INET6:
        *result_p = memcmp(&UCS_SOCKET_INET6_PORT(sa1),
                           &UCS_SOCKET_INET6_PORT(sa2),
                           sizeof(UCS_SOCKET_INET6_PORT(sa1)));
        return UCS_OK;
    default:
        ucs_error("unknown address family: %d", sa1->sa_family);
        return UCS_ERR_INVALID_PARAM;
    }
}

ucs_status_t ucs_sockaddr_cmp(const struct sockaddr *sa1,
                              const struct sockaddr *sa2,
                              int *result_p)
{
    ucs_status_t status;
    int result;

    if (sa1->sa_family != sa2->sa_family) {
        ucs_error("unable to compare socket addresses with "
                  "different address families: %d vs %d",
                  sa1->sa_family, sa2->sa_family);
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucs_sockaddr_addr_cmp(sa1, sa2, &result);
    if (status != UCS_OK) {
        return status;
    }

    if (result) {
        goto out;
    }

    status = ucs_sockaddr_port_cmp(sa1, sa2, &result);
    if (status != UCS_OK) {
        return status;
    }

out:
    *result_p = result;
    return status;
}

ucs_status_t ucs_sockaddr_copy(struct sockaddr *to, const struct sockaddr *from)
{
    ucs_status_t status;
    size_t length;

    status = ucs_sockaddr_sizeof(from, &length);
    if (status != UCS_OK) {
        return status;
    }

    memcpy(to, from, length);
    return UCS_OK;
}
