/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/iovec.h>
#include <ucs/sys/iovec.inl>

#include <sys/types.h>
#include <sys/socket.h>
#include <ifaddrs.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>


#define UCS_NETIF_BOND_AD_NUM_PORTS_FMT  "/sys/class/net/%s/bonding/ad_num_ports"
#define UCS_SOCKET_MAX_CONN_PATH         "/proc/sys/net/core/somaxconn"
/* The port space of IPv6 is shared with IPv4 */
#define UCX_PROCESS_IP_PORT_RANGE        "/proc/sys/net/ipv4/ip_local_port_range"


typedef ssize_t (*ucs_socket_io_func_t)(int fd, void *data,
                                        size_t size, int flags);

typedef ssize_t (*ucs_socket_iov_func_t)(int fd, const struct msghdr *msg,
                                         int flags);


static void ucs_socket_print_error_info(int sys_errno)
{
    if (sys_errno == EMFILE) {
        ucs_error("the maximal number of files that could be opened "
                  "simultaneously was reached, try to increase the limit "
                  "by setting the max open files limit (ulimit -n) to "
                  "a greater value (current: %d)",
                  ucs_sys_max_open_files());
    }
}

void ucs_close_fd(int *fd_p)
{
    if (*fd_p == -1) {
        return;
    }

    if (close(*fd_p) < 0) {
        ucs_warn("failed to close fd %d: %m", *fd_p);
        return;
    }

    *fd_p = -1;
}

int ucs_netif_flags_is_active(unsigned int flags)
{
    return (flags & IFF_UP) && (flags & IFF_RUNNING) && !(flags & IFF_LOOPBACK);
}

ucs_status_t ucs_netif_ioctl(const char *if_name, unsigned long request,
                             struct ifreq *if_req)
{
    ucs_status_t status;
    int fd = -1, ret;

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
    ucs_close_fd(&fd);
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

    return ucs_netif_flags_is_active(ifr.ifr_flags);
}

unsigned ucs_netif_bond_ad_num_ports(const char *bond_name)
{
    ucs_status_t status;
    long ad_num_ports;

    status = ucs_read_file_number(&ad_num_ports, 1,
                                  UCS_NETIF_BOND_AD_NUM_PORTS_FMT, bond_name);
    if ((status != UCS_OK) || (ad_num_ports <= 0) ||
        (ad_num_ports > UINT_MAX)) {
        ucs_diag("failed to read from " UCS_NETIF_BOND_AD_NUM_PORTS_FMT ": %m, "
                 "assuming 802.3ad bonding is disabled", bond_name);
        return 1;
    }

    return (unsigned)ad_num_ports;
}

ucs_status_t ucs_socket_create(int domain, int type, int *fd_p)
{
    int fd = socket(domain, type, 0);
    if (fd < 0) {
        ucs_error("socket create failed: %m");
        ucs_socket_print_error_info(errno);
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

ucs_status_t ucs_socket_getopt(int fd, int level, int optname,
                               void *optval, socklen_t optlen)
{
    socklen_t len = optlen;
    int ret;

    ret = getsockopt(fd, level, optname, optval, &len);
    if (ret < 0) {
        ucs_error("failed to get %d option for %d level on fd %d: %m",
                  optname, level, fd);
        return UCS_ERR_IO_ERROR;
    }

    if (len != optlen) {
        ucs_error("returned length of option (%d) is not the same as provided (%d)",
                  len, optlen);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

const char *ucs_socket_getname_str(int fd, char *str, size_t max_size)
{
    struct sockaddr_storage sock_addr = {0}; /* Suppress Clang false-positive */
    socklen_t addr_size;
    int ret;

    addr_size = sizeof(sock_addr);
    ret       = getsockname(fd, (struct sockaddr*)&sock_addr,
                            &addr_size);
    if (ret < 0) {
        ucs_debug("getsockname(fd=%d) failed: %m", fd);
        ucs_strncpy_safe(str, "-", max_size);
        return str;
    }

    return ucs_sockaddr_str((const struct sockaddr*)&sock_addr,
                            str, max_size);
}

static ucs_status_t ucs_socket_check_errno(int io_errno)
{
    if ((io_errno == EAGAIN) || (io_errno == EWOULDBLOCK) || (io_errno == EINTR)) {
        /* IO operation or connection establishment procedure was interrupted
         * or would block and need to try again */
        return UCS_ERR_NO_PROGRESS;
    }

    if (io_errno == ECONNRESET) {
        /* Connection reset by peer */
        return UCS_ERR_CONNECTION_RESET;
    } else if (io_errno == ECONNREFUSED) {
        /* A remote host refused to allow the network connection */
        return UCS_ERR_REJECTED;
    } else if (io_errno == ETIMEDOUT) {
        /* Connection establishment procedure timed out */
        return UCS_ERR_TIMED_OUT;
    } else if (io_errno == EPIPE) {
        /* The local end has been shut down */
        return UCS_ERR_CONNECTION_RESET;
    }

    return UCS_ERR_IO_ERROR;
}

ucs_status_t ucs_socket_connect(int fd, const struct sockaddr *dest_addr)
{
    char dest_str[UCS_SOCKADDR_STRING_LEN];
    char src_str[UCS_SOCKADDR_STRING_LEN];
    ucs_status_t status;
    size_t dest_addr_size;
    int UCS_V_UNUSED conn_errno;
    int ret;

    status = ucs_sockaddr_sizeof(dest_addr, &dest_addr_size);
    if (status != UCS_OK) {
        return status;
    }

    do {
        ret = connect(fd, dest_addr, dest_addr_size);
        if (ret < 0) {
            /* Save errno to separate variable to not override it
             * when calling getsockname() below */
            conn_errno = errno;

            if (errno == EINPROGRESS) {
                status = UCS_INPROGRESS;
                break;
            }

            if (errno == EISCONN) {
                status = UCS_ERR_ALREADY_EXISTS;
                break;
            }

            if (errno != EINTR) {
                ucs_error("connect(fd=%d, dest_addr=%s) failed: %m", fd,
                          ucs_sockaddr_str(dest_addr, dest_str,
                                           UCS_SOCKADDR_STRING_LEN));
                return UCS_ERR_UNREACHABLE;
            }
        } else {
            conn_errno = 0;
        }
    } while ((ret < 0) && (errno == EINTR));

    ucs_debug("connect(fd=%d, src_addr=%s dest_addr=%s): %s", fd,
              ucs_socket_getname_str(fd, src_str, UCS_SOCKADDR_STRING_LEN),
              ucs_sockaddr_str(dest_addr, dest_str, UCS_SOCKADDR_STRING_LEN),
              strerror(conn_errno));

    return status;
}

ucs_status_t ucs_socket_accept(int fd, struct sockaddr *addr, socklen_t *length_ptr,
                               int *accept_fd)
{
    ucs_status_t status;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];

    *accept_fd = accept(fd, addr, length_ptr);
    if (*accept_fd < 0) {
        status = ucs_socket_check_errno(errno);
        if (status == UCS_ERR_NO_PROGRESS) {
            return status;
        }

        ucs_error("accept() failed (client addr %s): %m",
                  ucs_sockaddr_str(addr, ip_port_str, UCS_SOCKADDR_STRING_LEN));

        ucs_socket_print_error_info(errno);

        return status;
    }

    return UCS_OK;
}

ucs_status_t ucs_socket_getpeername(int fd, struct sockaddr_storage *peer_addr,
                                    socklen_t *peer_addr_len)
{
    int ret;

    *peer_addr_len = sizeof(*peer_addr);
    ret            = getpeername(fd, (struct sockaddr*)peer_addr,
                                 peer_addr_len);
    if (ret < 0) {
        if ((errno != ENOTCONN) && (errno != ECONNRESET)) {
            ucs_error("getpeername(fd=%d) failed: %m", fd);
            return UCS_ERR_IO_ERROR;
        }

        return UCS_ERR_NOT_CONNECTED;
    }

    return UCS_OK;
}

int ucs_socket_is_connected(int fd)
{
    struct sockaddr_storage peer_addr = {0}; /* Suppress Clang false-positive */
    char peer_str[UCS_SOCKADDR_STRING_LEN];
    char local_str[UCS_SOCKADDR_STRING_LEN];
    socklen_t peer_addr_len;
    ucs_status_t status;

    status = ucs_socket_getpeername(fd, &peer_addr, &peer_addr_len);
    if (status != UCS_OK) {
        return 0;
    }

    ucs_debug("[%s]<->[%s] is a connected pair",
              ucs_socket_getname_str(fd, local_str, UCS_SOCKADDR_STRING_LEN),
              ucs_sockaddr_str((const struct sockaddr*)&peer_addr, peer_str,
                               UCS_SOCKADDR_STRING_LEN));

    return 1;
}

ucs_status_t ucs_socket_set_buffer_size(int fd, size_t sockopt_sndbuf,
                                        size_t sockopt_rcvbuf)
{
    ucs_status_t status;

    if (sockopt_sndbuf != UCS_MEMUNITS_AUTO) {
        status = ucs_socket_setopt(fd, SOL_SOCKET, SO_SNDBUF,
                                   (const void*)&sockopt_sndbuf, sizeof(int));
        if (status != UCS_OK) {
            return status;
        }
    }

    if (sockopt_rcvbuf != UCS_MEMUNITS_AUTO) {
        status = ucs_socket_setopt(fd, SOL_SOCKET, SO_RCVBUF,
                                   (const void*)&sockopt_rcvbuf, sizeof(int));
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

ucs_status_t ucs_socket_server_init(const struct sockaddr *saddr, socklen_t socklen,
                                    int backlog, int silent_err_in_use,
                                    int allow_addr_inuse, int *listen_fd)
{
    int so_reuse_optval = 1;
    char ip_port_str[UCS_SOCKADDR_STRING_LEN];
    ucs_log_level_t bind_log_level;
    ucs_status_t status;
    int ret, fd;

    /* Create the server socket for accepting incoming connections */
    status = ucs_socket_create(saddr->sa_family, SOCK_STREAM, &fd);
    if (status != UCS_OK) {
        goto err;
    }

    /* Set the fd to non-blocking mode (so that accept() won't be blocking) */
    status = ucs_sys_fcntl_modfl(fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_close_socket;
    }

    if (allow_addr_inuse) {
        status = ucs_socket_setopt(fd, SOL_SOCKET, SO_REUSEADDR,
                                   &so_reuse_optval, sizeof(so_reuse_optval));
        if (status != UCS_OK) {
            goto err_close_socket;
        }
    }

    ret = bind(fd, saddr, socklen);
    if (ret < 0) {
        if ((errno == EADDRINUSE) && silent_err_in_use) {
            bind_log_level = UCS_LOG_LEVEL_DEBUG;
        } else {
            bind_log_level = UCS_LOG_LEVEL_ERROR;
        }
        status = (errno == EADDRINUSE) ? UCS_ERR_BUSY : UCS_ERR_IO_ERROR;
        ucs_log(bind_log_level, "bind(fd=%d addr=%s) failed: %m",
                fd, ucs_sockaddr_str(saddr, ip_port_str, sizeof(ip_port_str)));
        goto err_close_socket;
    }

    if (listen(fd, backlog) < 0) {
        ucs_error("listen(fd=%d addr=%s backlog=%d) failed: %m",
                  fd, ucs_sockaddr_str(saddr, ip_port_str, sizeof(ip_port_str)),
                  backlog);
        status = UCS_ERR_IO_ERROR;
        goto err_close_socket;
    }

    *listen_fd = fd;
    return UCS_OK;

err_close_socket:
    ucs_close_fd(&fd);
err:
    return status;
}

int ucs_socket_max_conn()
{
    static long somaxconn_val = 0;

    if (somaxconn_val ||
        (ucs_read_file_number(&somaxconn_val, 1,
                              UCS_SOCKET_MAX_CONN_PATH) == UCS_OK)) {
        ucs_assert(somaxconn_val <= INT_MAX);
        return somaxconn_val;
    } else {
        ucs_warn("unable to read somaxconn value from %s file",
                 UCS_SOCKET_MAX_CONN_PATH);
        somaxconn_val = SOMAXCONN;
        return somaxconn_val;
    }
}

static ucs_status_t
ucs_socket_handle_io_error(int fd, const char *name, ssize_t io_retval, int io_errno)
{
    ucs_status_t status;

    if (io_retval == 0) {
        /* 0 can be returned only by recv() system call as an error if
         * the connection was dropped by peer */
        ucs_assert(!strcmp(name, "recv"));
        ucs_trace("fd %d is closed", fd);
        status = UCS_ERR_NOT_CONNECTED; /* Connection closed by peer */
    } else {
        ucs_debug("%s(%d) failed: %s", name, fd, strerror(io_errno));
        status = ucs_socket_check_errno(io_errno);
    }

    return status;
}

/**
 * Handle the IO operation.
 *
 * @param [in]  fd         The socket fd.
 * @param [in]  data       The pointer to user's data or pointer to the array of
 *                         iov elements.
 * @param [in]  count      The length of user's data or the number of elemnts in
 *                         the array of iov.
 * @param [out] length_p   Pointer to the result length of user's data that was
 *                         sent/received.
 * @param [in]  is_iov     Flag that specifies type of the operation (1 if vector
 *                         operation).
 * @param [in]  io_retval  The result of the IO operation.
 * @param [in]  io_errno   IO operation errno.
 *
 * @return if the IO operation was successful - UCS_OK, otherwise - error status.
 */
static inline ucs_status_t
ucs_socket_handle_io(int fd, const void *data, size_t count,
                     size_t *length_p, int is_iov, int io_retval,
                     int io_errno, const char *name)
{
    /* The IO operation is considered as successful if: */
    if (ucs_likely(io_retval > 0)) {
        /* - the return value > 0 */
        *length_p = io_retval;
        return UCS_OK;
    }

    if ((io_retval == 0) &&
        ((count == 0) ||
         (is_iov && (ucs_iovec_total_length((const struct iovec*)data,
                                            count) == 0)))) {
        /* - the return value == 0 and the user's data length == 0
         *   (the number of the iov array buffers == 0 or the total
         *   length of the iov array buffers == 0) */
        *length_p = 0;
        return UCS_OK;
    }

    *length_p = 0;
    return ucs_socket_handle_io_error(fd, name, io_retval, io_errno);
}

static inline ucs_status_t
ucs_socket_do_io_nb(int fd, void *data, size_t *length_p,
                    ucs_socket_io_func_t io_func, const char *name)
{
    ssize_t ret = io_func(fd, data, *length_p, MSG_NOSIGNAL);
    return ucs_socket_handle_io(fd, data, *length_p, length_p, 0,
                                ret, errno, name);
}

static inline ucs_status_t
ucs_socket_do_io_b(int fd, void *data, size_t length,
                   ucs_socket_io_func_t io_func, const char *name)
{
    size_t done_cnt = 0, cur_cnt = length;
    ucs_status_t status;

    do {
        status = ucs_socket_do_io_nb(fd, data, &cur_cnt, io_func, name);
        done_cnt += cur_cnt;
        ucs_assert(done_cnt <= length);
        cur_cnt = length - done_cnt;
    } while ((done_cnt < length) &&
             ((status == UCS_OK) || (status == UCS_ERR_NO_PROGRESS)));

    return status;
}

static inline ucs_status_t
ucs_socket_do_iov_nb(int fd, struct iovec *iov, size_t iov_cnt, size_t *length_p,
                     ucs_socket_iov_func_t iov_func, const char *name)
{
    struct msghdr msg = {
        .msg_iov    = iov,
        .msg_iovlen = iov_cnt
    };
    ssize_t ret;

    ret = iov_func(fd, &msg, MSG_NOSIGNAL);
    return ucs_socket_handle_io(fd, iov, iov_cnt, length_p, 1, ret, errno, name);
}

ucs_status_t ucs_socket_send_nb(int fd, const void *data, size_t *length_p)
{
    return ucs_socket_do_io_nb(fd, (void*)data, length_p,
                               (ucs_socket_io_func_t)send, "send");
}

/* recv is declared as 'always_inline' on some platforms, it leads to
 * compilation warning. wrap it into static function */
static ssize_t ucs_socket_recv_io(int fd, void *data, size_t size, int flags)
{
    return recv(fd, data, size, flags);
}

ucs_status_t ucs_socket_recv_nb(int fd, void *data, size_t *length_p)
{
    return ucs_socket_do_io_nb(fd, data, length_p, ucs_socket_recv_io, "recv");
}

ucs_status_t ucs_socket_send(int fd, const void *data, size_t length)
{
    return ucs_socket_do_io_b(fd, (void*)data, length,
                              (ucs_socket_io_func_t)send, "send");
}

ucs_status_t ucs_socket_recv(int fd, void *data, size_t length)
{
    return ucs_socket_do_io_b(fd, data, length, ucs_socket_recv_io, "recv");
}

ucs_status_t
ucs_socket_sendv_nb(int fd, struct iovec *iov, size_t iov_cnt, size_t *length_p)
{
    return ucs_socket_do_iov_nb(fd, iov, iov_cnt, length_p, sendmsg, "sendv");
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

ucs_status_t ucs_sockaddr_get_port(const struct sockaddr *addr, uint16_t *port_p)
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

ucs_status_t ucs_sockaddr_set_port(struct sockaddr *addr, uint16_t port)
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

int ucs_sockaddr_is_known_af(const struct sockaddr *sa)
{
    return ((sa->sa_family == AF_INET) ||
            (sa->sa_family == AF_INET6));
}

const char* ucs_sockaddr_str(const struct sockaddr *sock_addr,
                             char *str, size_t max_size)
{
    uint16_t port;
    size_t str_len;

    if (!ucs_sockaddr_is_known_af(sock_addr)) {
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

ucs_status_t ucs_sock_ipstr_to_sockaddr(const char *ip_str,
                                        struct sockaddr_storage *sa_storage)
{
    struct sockaddr_in* sa_in;
    struct sockaddr_in6* sa_in6;
    int ret;

    /* try IPv4 */
    sa_in             = (struct sockaddr_in*)sa_storage;
    sa_in->sin_family = AF_INET;
    ret = inet_pton(AF_INET, ip_str, &sa_in->sin_addr);
    if (ret == 1) {
        return UCS_OK;
    }

    /* try IPv6 */
    sa_in6              = (struct sockaddr_in6*)sa_storage;
    sa_in6->sin6_family = AF_INET6;
    ret = inet_pton(AF_INET6, ip_str, &sa_in6->sin6_addr);
    if (ret == 1) {
        return UCS_OK;
    }

    ucs_error("invalid address %s", ip_str);
    return UCS_ERR_INVALID_ADDR;
}

int ucs_sockaddr_cmp(const struct sockaddr *sa1,
                     const struct sockaddr *sa2,
                     ucs_status_t *status_p)
{
    int result          = 1;
    uint16_t port1      = 0, port2 = 0;
    ucs_status_t status = UCS_OK;

    if (!ucs_sockaddr_is_known_af(sa1) ||
        !ucs_sockaddr_is_known_af(sa2)) {
        ucs_error("unknown address family: %d",
                  !ucs_sockaddr_is_known_af(sa1) ?
                  sa1->sa_family : sa2->sa_family);
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    if (sa1->sa_family != sa2->sa_family) {
        result = (int)sa1->sa_family - (int)sa2->sa_family;
        goto out;
    }

    switch (sa1->sa_family) {
    case AF_INET:
        result = memcmp(&UCS_SOCKET_INET_ADDR(sa1),
                        &UCS_SOCKET_INET_ADDR(sa2),
                        sizeof(UCS_SOCKET_INET_ADDR(sa1)));
        port1 = ntohs(UCS_SOCKET_INET_PORT(sa1));
        port2 = ntohs(UCS_SOCKET_INET_PORT(sa2));
        break;
    case AF_INET6:
        result = memcmp(&UCS_SOCKET_INET6_ADDR(sa1),
                        &UCS_SOCKET_INET6_ADDR(sa2),
                        sizeof(UCS_SOCKET_INET6_ADDR(sa1)));
        port1 = ntohs(UCS_SOCKET_INET6_PORT(sa1));
        port2 = ntohs(UCS_SOCKET_INET6_PORT(sa2));
        break;
    }

    if (!result && (port1 != port2)) {
        result = (int)port1 - (int)port2;
    }

out:
    if (status_p) {
        *status_p = status;
    }
    return result;
}

int ucs_sockaddr_ip_cmp(const struct sockaddr *sa1, const struct sockaddr *sa2)
{
    if (!ucs_sockaddr_is_known_af(sa1) || !ucs_sockaddr_is_known_af(sa2)) {
        ucs_error("unknown address family: %d",
                  !ucs_sockaddr_is_known_af(sa1) ? sa1->sa_family : sa2->sa_family);
        return -1;
    }

    return memcmp(ucs_sockaddr_get_inet_addr(sa1),
                  ucs_sockaddr_get_inet_addr(sa2),
                  (sa1->sa_family == AF_INET) ?
                  UCS_IPV4_ADDR_LEN : UCS_IPV6_ADDR_LEN);
}

int ucs_sockaddr_is_inaddr_any(struct sockaddr *addr)
{
    switch (addr->sa_family) {
    case AF_INET:
        return UCS_SOCKET_INET_ADDR(addr).s_addr == INADDR_ANY;
    case AF_INET6:
        return !memcmp(&(UCS_SOCKET_INET6_ADDR(addr)), &in6addr_any,
                       sizeof(UCS_SOCKET_INET6_ADDR(addr)));
    default:
        ucs_debug("invalid address family: %d", addr->sa_family);
        return 0;
    }
}

ucs_status_t ucs_sockaddr_copy(struct sockaddr *dst_addr,
                               const struct sockaddr *src_addr)
{
    ucs_status_t status;
    size_t size;

    status = ucs_sockaddr_sizeof(src_addr, &size);
    if (status != UCS_OK) {
        return status;
    }

    memcpy(dst_addr, src_addr, size);
    return UCS_OK;
}

ucs_status_t ucs_sockaddr_get_ifname(int fd, char *ifname_str, size_t max_strlen)
{
    ucs_status_t status = UCS_ERR_NO_DEVICE;
    struct ifaddrs *ifa;
    struct ifaddrs* ifaddrs;
    struct sockaddr *sa;
    struct sockaddr *my_addr;
    socklen_t sockaddr_len;
    char str_local_addr[UCS_SOCKADDR_STRING_LEN];

    sockaddr_len = sizeof(struct sockaddr_storage);
    my_addr      = ucs_alloca(sockaddr_len);

    if (getsockname(fd, my_addr, &sockaddr_len)) {
        ucs_warn("getsockname error: %m");
        return UCS_ERR_IO_ERROR;
    }

    /* port number is not important, so we assign zero because sockaddr
     * structures returned by getifaddrs have ports assigned to zero */
    if (UCS_OK != ucs_sockaddr_set_port(my_addr, 0)) {
        ucs_warn("sockcm doesn't support unknown address family");
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_debug("check ifname for socket on %s", 
              ucs_sockaddr_str(my_addr, str_local_addr, UCS_SOCKADDR_STRING_LEN));

    if (getifaddrs(&ifaddrs)) {
        ucs_warn("getifaddrs error: %m");
        return UCS_ERR_IO_ERROR;
    }

    for (ifa = ifaddrs; ifa != NULL; ifa = ifa->ifa_next) {
        sa = (struct sockaddr*) ifa->ifa_addr;

        if (sa == NULL) {
            ucs_debug("NULL ifaddr encountered with ifa_name: %s", ifa->ifa_name);
            continue;
        }

        if (((sa->sa_family == AF_INET) ||(sa->sa_family == AF_INET6)) && 
            (!ucs_sockaddr_cmp(sa, my_addr, NULL))) {
            ucs_debug("matching ip found iface on %s", ifa->ifa_name);
            ucs_strncpy_safe(ifname_str, ifa->ifa_name, max_strlen);
            status = UCS_OK;
            break;
        }
    }

    freeifaddrs(ifaddrs);

    return status;
}

const char *ucs_sockaddr_address_family_str(sa_family_t af)
{
    switch (af) {
    case AF_INET:
        return "IPv4";
    case AF_INET6:
        return "IPv6";
    default:
        return "not IPv4 or IPv6";
    }
}

ucs_status_t ucs_sockaddr_get_ip_local_port_range(ucs_range_spec_t *port_range)
{
    char ip_local_port_range[32];
    char *endptr;
    ssize_t nread;

    nread = ucs_read_file_str(ip_local_port_range, sizeof(ip_local_port_range),
                              1, UCX_PROCESS_IP_PORT_RANGE);
    if (nread < 0) {
        ucs_diag("failed to read " UCX_PROCESS_IP_PORT_RANGE);
        return UCS_ERR_IO_ERROR;
    }

    port_range->first = strtol(ip_local_port_range, &endptr, 10);
    if ((port_range->first <= 0) || (*endptr == '\0')) {
        return UCS_ERR_IO_ERROR;
    }

    port_range->last = strtol(endptr, &endptr, 10);
    if (port_range->last <= 0) {
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}
