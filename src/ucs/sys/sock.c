/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <dirent.h>

#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/uio.h>
/* need this to get IOV_MAX on some platforms. */
#ifndef __need_IOV_MAX
#define __need_IOV_MAX
#endif
#include <limits.h>


#define UCS_SOCKET_MAX_CONN_PATH "/proc/sys/net/core/somaxconn"


typedef ssize_t (*ucs_socket_io_func_t)(int fd, void *data,
                                        size_t size, int flags);

typedef ssize_t (*ucs_socket_iov_func_t)(int fd, const struct msghdr *msg,
                                         int flags);


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
        ucs_error("SOL_SOCKET(SO_ERROR) status on fd %d: %s",
                  fd, strerror(conn_status));
        return UCS_ERR_UNREACHABLE;
    }

    return UCS_OK;
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

int ucs_socket_max_iov()
{
    static int max_iov = -1;

#ifdef _SC_IOV_MAX
    if (max_iov != -1) {
        return max_iov;
    }

    max_iov = sysconf(_SC_IOV_MAX);
    if (max_iov != -1) {
        return max_iov;
    }
    /* if unable to get value from sysconf(),
     * use a predefined value */
#endif

#if defined(IOV_MAX)
    max_iov = IOV_MAX;
#elif defined(UIO_MAXIOV)
    max_iov = UIO_MAXIOV;
#else
    /* The value is used as a fallback when system value is not available.
     * The latest kernels define it as 1024 */
    max_iov = 1024;
#endif

    return max_iov;
}

static ucs_status_t
ucs_socket_handle_io_error(int fd, const char *name, ssize_t io_retval, int io_errno,
                           ucs_socket_io_err_cb_t err_cb, void *err_cb_arg)
{
    if (io_retval == 0) {
        ucs_trace("fd %d is closed", fd);
        return UCS_ERR_CANCELED; /* Connection closed */
    }

    if ((io_errno == EINTR) || (io_errno == EAGAIN) || (io_errno == EWOULDBLOCK)) {
        return UCS_ERR_NO_PROGRESS;
    }

    if ((err_cb != NULL) && (err_cb(err_cb_arg, io_errno) != UCS_OK)) {
        ucs_error("%s(fd=%d) failed: %m", name, fd);
    }

    return UCS_ERR_IO_ERROR;
}

static inline ucs_status_t
ucs_socket_do_io_nb(int fd, void *data, size_t *length_p,
                    ucs_socket_io_func_t io_func, const char *name,
                    ucs_socket_io_err_cb_t err_cb, void *err_cb_arg)
{
    ssize_t ret;

    ucs_assert(*length_p > 0);

    ret = io_func(fd, data, *length_p, MSG_NOSIGNAL);
    if (ucs_likely(ret > 0)) {
        *length_p = ret;
        return UCS_OK;
    }

    *length_p = 0;
    return ucs_socket_handle_io_error(fd, name, ret, errno,
                                      err_cb, err_cb_arg);
}

static inline ucs_status_t
ucs_socket_do_io_b(int fd, void *data, size_t length,
                   ucs_socket_io_func_t io_func, const char *name,
                   ucs_socket_io_err_cb_t err_cb, void *err_cb_arg)
{
    size_t done_cnt = 0, cur_cnt = length;
    ucs_status_t status;

    do {
        status = ucs_socket_do_io_nb(fd, data, &cur_cnt, io_func,
                                     name, err_cb, err_cb);
        if (ucs_likely(status == UCS_OK)) {
            done_cnt += cur_cnt;
            ucs_assert(done_cnt <= length);
            cur_cnt = length - done_cnt;
        } else if (status != UCS_ERR_NO_PROGRESS) {
            return status;
        }
    } while (done_cnt < length);

    return UCS_OK;
}

static inline ucs_status_t
ucs_socket_do_iov_nb(int fd, struct iovec *iov, size_t iov_cnt, size_t *length_p,
                     ucs_socket_iov_func_t iov_func, const char *name,
                     ucs_socket_io_err_cb_t err_cb, void *err_cb_arg)
{
    struct msghdr msg = {
        .msg_iov    = iov,
        .msg_iovlen = iov_cnt
    };
    ssize_t ret;

    ucs_assert(iov_cnt > 0);

    ret = iov_func(fd, &msg, MSG_NOSIGNAL);
    if (ucs_likely(ret > 0)) {
        *length_p = ret;
        return UCS_OK;
    }

    *length_p = 0;
    return ucs_socket_handle_io_error(fd, name, ret, errno,
                                      err_cb, err_cb_arg);
}

ucs_status_t ucs_socket_send_nb(int fd, const void *data, size_t *length_p,
                                ucs_socket_io_err_cb_t err_cb,
                                void *err_cb_arg)
{
    return ucs_socket_do_io_nb(fd, (void*)data, length_p,
                               (ucs_socket_io_func_t)send,
                               "send", err_cb, err_cb_arg);
}

ucs_status_t ucs_socket_recv_nb(int fd, void *data, size_t *length_p,
                                ucs_socket_io_err_cb_t err_cb,
                                void *err_cb_arg)
{
    return ucs_socket_do_io_nb(fd, data, length_p, recv,
                               "recv", err_cb, err_cb_arg);
}

ucs_status_t ucs_socket_send(int fd, const void *data, size_t length,
                             ucs_socket_io_err_cb_t err_cb,
                             void *err_cb_arg)
{
    return ucs_socket_do_io_b(fd, (void*)data, length,
                              (ucs_socket_io_func_t)send,
                              "send", err_cb, err_cb_arg);
}

ucs_status_t ucs_socket_recv(int fd, void *data, size_t length,
                             ucs_socket_io_err_cb_t err_cb,
                             void *err_cb_arg)
{
    return ucs_socket_do_io_b(fd, data, length, recv,
                              "recv", err_cb, err_cb_arg);
}

ucs_status_t
ucs_socket_sendv_nb(int fd, struct iovec *iov, size_t iov_cnt, size_t *length_p,
                    ucs_socket_io_err_cb_t err_cb, void *err_cb_arg)
{
    return ucs_socket_do_iov_nb(fd, iov, iov_cnt, length_p, sendmsg,
                                "sendv", err_cb, err_cb_arg);
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

int ucs_sockaddr_is_inaddr_any(struct sockaddr *addr)
{
    struct sockaddr_in6 *addr_in6;
    struct sockaddr_in *addr_in;

    switch (addr->sa_family) {
    case AF_INET:
        addr_in = (struct sockaddr_in *)addr;
        return addr_in->sin_addr.s_addr == INADDR_ANY;
    case AF_INET6:
        addr_in6 = (struct sockaddr_in6 *)addr;
        return !memcmp(&addr_in6->sin6_addr, &in6addr_any, sizeof(addr_in6->sin6_addr));
    default:
        ucs_debug("Invalid address family: %d", addr->sa_family);
    }

    return 0;
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

static unsigned ucs_sockaddr_is_known_af(const struct sockaddr *sa)
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

int ucs_sockaddr_cmp(const struct sockaddr *sa1,
                     const struct sockaddr *sa2,
                     ucs_status_t *status_p)
{
    int result          = 1;
    ucs_status_t status = UCS_OK;
    uint16_t port1, port2;

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

ucs_status_t ucs_sockaddr_get_dev_names(unsigned *num_resources_p, 
                                        char **dev_names_pp, 
                                        size_t dev_name_len)
{
    char *dev_names, *tmp, *dev_name;
    static const char *netdev_dir = "/sys/class/net";
    ucs_status_t status = UCS_OK;
    unsigned num_resources;
    struct dirent *entry;
    DIR *dir;

    dir = opendir(netdev_dir);
    if (dir == NULL) {
        ucs_error("opendir(%s) failed: %m", netdev_dir);
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    dev_names     = NULL;
    num_resources = 0;
    for (;;) {
        errno = 0;
        entry = readdir(dir);
        if (entry == NULL) {
            if (errno != 0) {
                ucs_error("readdir(%s) failed: %m", netdev_dir);
                ucs_free(dev_names);
                status = UCS_ERR_IO_ERROR;
                goto out_closedir;
            }
            break; /* no more items */
        }

        if (!ucs_netif_is_active(entry->d_name)) {
            continue;
        }

        tmp = ucs_realloc(dev_names, dev_name_len * (num_resources + 1),
                          "dev_names");
        if (tmp == NULL) {
            ucs_free(dev_names);
            status = UCS_ERR_NO_MEMORY;
            goto out_closedir;
        }
        dev_names = tmp;

        /* UCT_DEVICE_NAME_MAX */
        dev_name = (char *)dev_names + (num_resources * dev_name_len);
        num_resources++;
        ucs_snprintf_zero(dev_name, dev_name_len, "%s", entry->d_name);
    }
    *num_resources_p = num_resources;
    *dev_names_pp    = dev_names;

out_closedir:
    closedir(dir);
out:
    return status;
}
