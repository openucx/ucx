/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "fd_exchange.h"

#include <ucs/async/async.h>
#include <ucs/datastruct/khash.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/sock.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/time/time.h>
#include <ucs/type/spinlock.h>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <string.h>
#include <unistd.h>


/* Private server directory under the user-provided directory, the server
 * socket in it, and the reply socket of an importer next to the server */
#define UCS_FD_EXCHANGE_DIR_FMT      "%s/ucx-%08x"
#define UCS_FD_EXCHANGE_SERVER_FMT   "%s/s"
#define UCS_FD_EXCHANGE_REPLY_FMT    "%.*s/r%08x"

/* Longest path created under the user-provided directory, and the part of it
 * which an importer adds to the server directory */
#define UCS_FD_EXCHANGE_PATH_SUFFIX  "/ucx-XXXXXXXX/rXXXXXXXX"
#define UCS_FD_EXCHANGE_REPLY_SUFFIX "/rXXXXXXXX"
#define UCS_FD_EXCHANGE_PATH_MAX     ucs_field_sizeof(struct sockaddr_un, \
                                                      sun_path)

/* Attempts to pick an unused random name */
#define UCS_FD_EXCHANGE_MAX_ATTEMPTS 8


KHASH_MAP_INIT_INT64(ucs_fd_exchange_fds, int);


struct ucs_fd_exchange_server {
    int                          sock_fd;
    ucs_spinlock_t               lock; /* Protects fds from the async thread */
    khash_t(ucs_fd_exchange_fds) fds;  /* Served descriptors by key */
    char                         dir[UCS_FD_EXCHANGE_PATH_MAX];
    char                         path[UCS_FD_EXCHANGE_PATH_MAX];
};


static ucs_status_t
ucs_fd_exchange_sockaddr_init(const char *path, struct sockaddr_un *addr)
{
    if (strlen(path) >= sizeof(addr->sun_path)) {
        ucs_error("unix socket path '%s' is longer than %zu characters", path,
                  sizeof(addr->sun_path) - 1);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    memset(addr, 0, sizeof(*addr));
    addr->sun_family = AF_UNIX;
    ucs_strncpy_zero(addr->sun_path, path, sizeof(addr->sun_path));
    return UCS_OK;
}

/* Parse the control messages of a received datagram: return the credentials
 * of the sender if present, and take the first descriptor it carried into
 * *fd_p, closing any other one, so that unexpected descriptors do not leak */
static const struct ucred *
ucs_fd_exchange_msg_parse(struct msghdr *msgh, int *fd_p)
{
    const struct ucred *cred = NULL;
    struct cmsghdr *cmsg;
    size_t i, num_fds;
    int fd;

    *fd_p = -1;
    for (cmsg = CMSG_FIRSTHDR(msgh); cmsg != NULL;
         cmsg = CMSG_NXTHDR(msgh, cmsg)) {
        if (cmsg->cmsg_level != SOL_SOCKET) {
            continue;
        }

        if ((cmsg->cmsg_type == SCM_CREDENTIALS) &&
            (cmsg->cmsg_len == CMSG_LEN(sizeof(*cred)))) {
            cred = (const struct ucred*)CMSG_DATA(cmsg);
        } else if (cmsg->cmsg_type == SCM_RIGHTS) {
            num_fds = (cmsg->cmsg_len - CMSG_LEN(0)) / sizeof(fd);
            for (i = 0; i < num_fds; ++i) {
                memcpy(&fd, CMSG_DATA(cmsg) + (i * sizeof(fd)), sizeof(fd));
                if (*fd_p < 0) {
                    *fd_p = fd;
                } else {
                    close(fd);
                }
            }
        }
    }

    return cred;
}

/* Like ptrace, only processes of the same user may take descriptors */
static int ucs_fd_exchange_is_same_user(const struct ucred *cred)
{
    return (cred != NULL) && (cred->uid == getuid()) &&
           (cred->gid == getgid());
}

static int
ucs_fd_exchange_server_dup_fd(ucs_fd_exchange_server_t *server, uint64_t key)
{
    int fd = -1;
    khiter_t iter;

    ucs_spin_lock(&server->lock);
    iter = kh_get(ucs_fd_exchange_fds, &server->fds, key);
    if (iter != kh_end(&server->fds)) {
        /* Duplicate under the lock, so a concurrent remove cannot close the
         * descriptor and let its number be reused before it is sent */
        fd = fcntl(kh_val(&server->fds, iter), F_DUPFD_CLOEXEC, 0);
        if (fd < 0) {
            ucs_warn("fcntl(%d, F_DUPFD_CLOEXEC) failed: %m",
                     kh_val(&server->fds, iter));
        }
    }
    ucs_spin_unlock(&server->lock);

    return fd;
}

static void ucs_fd_exchange_server_reply(ucs_fd_exchange_server_t *server,
                                         struct sockaddr_un *client_addr,
                                         socklen_t client_addr_len,
                                         uint8_t reply, int fd)
{
    char cbuf[CMSG_SPACE(sizeof(fd))] UCS_V_ALIGNED(sizeof(size_t));
    struct cmsghdr *cmsg;
    struct msghdr msgh;
    struct iovec iov;

    iov.iov_base     = &reply;
    iov.iov_len      = sizeof(reply);
    memset(&msgh, 0, sizeof(msgh));
    msgh.msg_name    = client_addr;
    msgh.msg_namelen = client_addr_len;
    msgh.msg_iov     = &iov;
    msgh.msg_iovlen  = 1;

    if (fd >= 0) {
        memset(cbuf, 0, sizeof(cbuf));
        msgh.msg_control    = cbuf;
        msgh.msg_controllen = sizeof(cbuf);
        cmsg                = CMSG_FIRSTHDR(&msgh);
        cmsg->cmsg_level    = SOL_SOCKET;
        cmsg->cmsg_type     = SCM_RIGHTS;
        cmsg->cmsg_len      = CMSG_LEN(sizeof(fd));
        memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
    }

    /* An importer which gave up has already removed its reply socket. Its
     * address is not necessarily NUL-terminated. */
    if (sendmsg(server->sock_fd, &msgh, MSG_DONTWAIT | MSG_NOSIGNAL) < 0) {
        ucs_debug("fd exchange %s: sendmsg(%.*s) failed: %m", server->path,
                  (int)(client_addr_len -
                        offsetof(struct sockaddr_un, sun_path)),
                  client_addr->sun_path);
    }
}

static void ucs_fd_exchange_server_handler(int sock_fd,
                                           ucs_event_set_types_t events,
                                           void *arg)
{
    ucs_fd_exchange_server_t *server = arg;
    char cbuf[CMSG_SPACE(sizeof(struct ucred))] UCS_V_ALIGNED(sizeof(size_t));
    struct sockaddr_un client_addr;
    ucs_fd_exchange_request_t req;
    const struct ucred *cred;
    struct msghdr msgh;
    struct iovec iov;
    ssize_t nrecvd;
    uint8_t reply;
    int fd;

    for (;;) {
        iov.iov_base        = &req;
        iov.iov_len         = sizeof(req);
        memset(&msgh, 0, sizeof(msgh));
        msgh.msg_name       = &client_addr;
        msgh.msg_namelen    = sizeof(client_addr);
        msgh.msg_iov        = &iov;
        msgh.msg_iovlen     = 1;
        msgh.msg_control    = cbuf;
        msgh.msg_controllen = sizeof(cbuf);

        nrecvd = recvmsg(server->sock_fd, &msgh,
                         MSG_DONTWAIT | MSG_CMSG_CLOEXEC);
        if (nrecvd < 0) {
            if (errno == EINTR) {
                continue;
            } else if ((errno != EAGAIN) && (errno != EWOULDBLOCK)) {
                ucs_warn("fd exchange %s: recvmsg() failed: %m", server->path);
            }

            return;
        }

        /* The server does not expect descriptors: the control buffer has room
         * for the credentials only, so the kernel drops the descriptors of a
         * datagram which carries them, and any which still arrive are closed */
        cred = ucs_fd_exchange_msg_parse(&msgh, &fd);
        if (fd >= 0) {
            close(fd);
        }

        /* Without the magic number, the sender's address or the same user,
         * this is not a request this server may answer */
        if ((nrecvd < offsetof(ucs_fd_exchange_request_t, key)) ||
            (req.magic != UCS_FD_EXCHANGE_MAGIC) ||
            (msgh.msg_flags & MSG_CTRUNC) ||
            (msgh.msg_namelen <= offsetof(struct sockaddr_un, sun_path)) ||
            !ucs_fd_exchange_is_same_user(cred)) {
            ucs_debug("fd exchange %s: dropping an invalid request",
                      server->path);
            continue;
        }

        /* A datagram of another size was sent by another protocol version,
         * which is told so instead of waiting for a reply until its timeout */
        if ((req.version != UCS_FD_EXCHANGE_VERSION) ||
            (nrecvd != sizeof(req)) || (msgh.msg_flags & MSG_TRUNC)) {
            ucs_debug("fd exchange %s: unsupported request version %u "
                      "size %zd", server->path, req.version, nrecvd);
            ucs_fd_exchange_server_reply(server, &client_addr,
                                         msgh.msg_namelen,
                                         UCS_FD_EXCHANGE_REPLY_UNSUPPORTED, -1);
            continue;
        }

        fd    = ucs_fd_exchange_server_dup_fd(server, req.key);
        reply = (fd >= 0) ? UCS_FD_EXCHANGE_REPLY_FOUND :
                            UCS_FD_EXCHANGE_REPLY_NOT_FOUND;
        ucs_fd_exchange_server_reply(server, &client_addr, msgh.msg_namelen,
                                     reply, fd);
        if (fd >= 0) {
            close(fd);
        }
    }
}

static ucs_status_t
ucs_fd_exchange_server_unlink_cb(const struct dirent *entry, void *arg)
{
    const char *dir = arg;
    char path[UCS_FD_EXCHANGE_PATH_MAX];

    if ((strcmp(entry->d_name, ".") == 0) ||
        (strcmp(entry->d_name, "..") == 0)) {
        return UCS_OK;
    }

    ucs_snprintf_safe(path, sizeof(path), "%s/%s", dir, entry->d_name);
    if (unlink(path) < 0) {
        ucs_warn("unlink(%s) failed: %m", path);
    }

    return UCS_OK;
}

ucs_status_t ucs_fd_exchange_server_create(const char *dir,
                                           ucs_fd_exchange_server_t **server_p)
{
    int optval = 1;
    ucs_fd_exchange_server_t *server;
    struct sockaddr_un addr;
    ucs_status_t status;
    unsigned attempt;

    if (dir[0] != '/') {
        ucs_error("fd exchange directory '%s' is not an absolute path", dir);
        return UCS_ERR_INVALID_PARAM;
    }

    if ((strlen(dir) + sizeof(UCS_FD_EXCHANGE_PATH_SUFFIX)) >
        UCS_FD_EXCHANGE_PATH_MAX) {
        ucs_error("fd exchange directory '%s' is longer than %zu characters",
                  dir, UCS_FD_EXCHANGE_PATH_MAX -
                       sizeof(UCS_FD_EXCHANGE_PATH_SUFFIX));
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    server = ucs_calloc(1, sizeof(*server), "ucs_fd_exchange_server");
    if (server == NULL) {
        ucs_error("failed to allocate fd exchange server");
        return UCS_ERR_NO_MEMORY;
    }

    kh_init_inplace(ucs_fd_exchange_fds, &server->fds);
    ucs_spinlock_init(&server->lock, 0);

    /* The private directory lets only the same user reach the socket and
     * create reply sockets next to it */
    for (attempt = 0;; ++attempt) {
        ucs_snprintf_safe(server->dir, sizeof(server->dir),
                          UCS_FD_EXCHANGE_DIR_FMT, dir,
                          (uint32_t)ucs_generate_uuid(attempt));
        if (mkdir(server->dir, S_IRWXU) == 0) {
            break;
        }

        if ((errno != EEXIST) || (attempt == UCS_FD_EXCHANGE_MAX_ATTEMPTS)) {
            ucs_error("mkdir(%s) failed: %m", server->dir);
            status = UCS_ERR_IO_ERROR;
            goto err_free;
        }
    }

    ucs_snprintf_safe(server->path, sizeof(server->path),
                      UCS_FD_EXCHANGE_SERVER_FMT, server->dir);
    status = ucs_fd_exchange_sockaddr_init(server->path, &addr);
    if (status != UCS_OK) {
        goto err_rmdir;
    }

    status = ucs_socket_create(AF_UNIX,
                               SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0,
                               &server->sock_fd);
    if (status != UCS_OK) {
        goto err_rmdir;
    }

    if (bind(server->sock_fd, (const struct sockaddr*)&addr, sizeof(addr)) <
        0) {
        ucs_error("bind(%s) failed: %m", server->path);
        status = UCS_ERR_IO_ERROR;
        goto err_close;
    }

    /* Get the credentials of every request */
    status = ucs_socket_setopt(server->sock_fd, SOL_SOCKET, SO_PASSCRED,
                               &optval, sizeof(optval));
    if (status != UCS_OK) {
        goto err_unlink;
    }

    /* Serve from the async thread, so importers do not depend on the progress
     * of this process */
    status = ucs_async_set_event_handler(UCS_ASYNC_THREAD_LOCK_TYPE,
                                         server->sock_fd,
                                         UCS_EVENT_SET_EVREAD,
                                         ucs_fd_exchange_server_handler,
                                         server, NULL);
    if (status != UCS_OK) {
        goto err_unlink;
    }

    ucs_debug("created fd exchange server %s", server->path);
    *server_p = server;
    return UCS_OK;

err_unlink:
    unlink(server->path);
err_close:
    close(server->sock_fd);
err_rmdir:
    rmdir(server->dir);
err_free:
    ucs_spinlock_destroy(&server->lock);
    kh_destroy_inplace(ucs_fd_exchange_fds, &server->fds);
    ucs_free(server);
    return status;
}

void ucs_fd_exchange_server_destroy(ucs_fd_exchange_server_t *server)
{
    ucs_assertv(kh_size(&server->fds) == 0, "server %s has %u descriptors",
                server->path, kh_size(&server->fds));

    ucs_async_remove_handler(server->sock_fd, 1);
    close(server->sock_fd);

    /* Also remove reply sockets left by importers which were killed */
    ucs_sys_readdir(server->dir, ucs_fd_exchange_server_unlink_cb,
                    server->dir);
    if (rmdir(server->dir) < 0) {
        ucs_warn("rmdir(%s) failed: %m", server->dir);
    }

    ucs_spinlock_destroy(&server->lock);
    kh_destroy_inplace(ucs_fd_exchange_fds, &server->fds);
    ucs_free(server);
}

const char *
ucs_fd_exchange_server_path(const ucs_fd_exchange_server_t *server)
{
    return server->path;
}

ucs_status_t ucs_fd_exchange_server_add_fd(ucs_fd_exchange_server_t *server,
                                           uint64_t key, int fd)
{
    ucs_status_t status;
    khiter_t iter;
    int ret;

    ucs_spin_lock(&server->lock);
    iter = kh_put(ucs_fd_exchange_fds, &server->fds, key, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_error("fd exchange %s: failed to add key 0x%" PRIx64, server->path,
                  key);
        status = UCS_ERR_NO_MEMORY;
    } else if (ret == UCS_KH_PUT_KEY_PRESENT) {
        ucs_error("fd exchange %s: key 0x%" PRIx64 " already exists",
                  server->path, key);
        status = UCS_ERR_ALREADY_EXISTS;
    } else {
        kh_val(&server->fds, iter) = fd;
        status                     = UCS_OK;
    }
    ucs_spin_unlock(&server->lock);

    return status;
}

void ucs_fd_exchange_server_remove_fd(ucs_fd_exchange_server_t *server,
                                      uint64_t key)
{
    khiter_t iter;

    ucs_spin_lock(&server->lock);
    iter = kh_get(ucs_fd_exchange_fds, &server->fds, key);
    if (iter != kh_end(&server->fds)) {
        kh_del(ucs_fd_exchange_fds, &server->fds, iter);
    } else {
        ucs_warn("fd exchange %s: key 0x%" PRIx64 " does not exist",
                 server->path, key);
    }
    ucs_spin_unlock(&server->lock);
}

/* Make the next send or receive on the socket give up at the deadline */
static ucs_status_t ucs_fd_exchange_set_timeout(int sock_fd, int optname,
                                                ucs_time_t deadline)
{
    struct timeval tv = {0, 0};
    ucs_time_t now;

    /* A zero timeout waits forever */
    if (deadline != UCS_TIME_INFINITY) {
        now = ucs_get_time();
        if (now >= deadline) {
            return UCS_ERR_TIMED_OUT;
        }

        ucs_sec_to_timeval(ucs_time_to_sec(deadline - now), &tv);
        if ((tv.tv_sec == 0) && (tv.tv_usec == 0)) {
            tv.tv_usec = 1;
        }
    }

    return ucs_socket_setopt(sock_fd, SOL_SOCKET, optname, &tv, sizeof(tv));
}

/* Send the request, waiting for room in the server queue until the deadline */
static ucs_status_t
ucs_fd_exchange_send_request(int sock_fd, const struct sockaddr_un *server_addr,
                             const ucs_fd_exchange_request_t *req,
                             ucs_time_t deadline)
{
    ucs_status_t status;
    ssize_t ret;

    for (;;) {
        status = ucs_fd_exchange_set_timeout(sock_fd, SO_SNDTIMEO, deadline);
        if (status != UCS_OK) {
            return status;
        }

        ret = sendto(sock_fd, req, sizeof(*req), MSG_NOSIGNAL,
                     (const struct sockaddr*)server_addr, sizeof(*server_addr));
        if (ret == sizeof(*req)) {
            return UCS_OK;
        } else if ((ret < 0) && (errno == EINTR)) {
            continue;
        }

        if (ret >= 0) {
            status = UCS_ERR_IO_ERROR;
        } else if ((errno == EAGAIN) || (errno == EWOULDBLOCK)) {
            status = UCS_ERR_TIMED_OUT;
        } else if ((errno == ENOENT) || (errno == ECONNREFUSED)) {
            /* The server socket is gone, so its process is as well */
            status = UCS_ERR_UNREACHABLE;
        } else {
            status = UCS_ERR_IO_ERROR;
        }

        ucs_debug("sendto(%s) returned %zd: %m", server_addr->sun_path, ret);
        return status;
    }
}

/* Receive the reply, and the descriptor it carries into *fd_p */
static ucs_status_t ucs_fd_exchange_recv_reply(int sock_fd, const char *path,
                                               ucs_time_t deadline, int *fd_p)
{
    char cbuf[CMSG_SPACE(sizeof(*fd_p))] UCS_V_ALIGNED(sizeof(size_t));
    struct msghdr msgh;
    struct iovec iov;
    ucs_status_t status;
    ssize_t nrecvd;
    uint8_t reply;
    int fd;

    for (;;) {
        status = ucs_fd_exchange_set_timeout(sock_fd, SO_RCVTIMEO, deadline);
        if (status != UCS_OK) {
            return status;
        }

        iov.iov_base        = &reply;
        iov.iov_len         = sizeof(reply);
        memset(&msgh, 0, sizeof(msgh));
        msgh.msg_iov        = &iov;
        msgh.msg_iovlen     = 1;
        msgh.msg_control    = cbuf;
        msgh.msg_controllen = sizeof(cbuf);

        nrecvd = recvmsg(sock_fd, &msgh, MSG_CMSG_CLOEXEC);
        if (nrecvd >= 0) {
            break;
        } else if (errno == EINTR) {
            continue;
        }

        status = ((errno == EAGAIN) || (errno == EWOULDBLOCK)) ?
                 UCS_ERR_TIMED_OUT : UCS_ERR_IO_ERROR;
        ucs_debug("recvmsg() from %s failed: %m", path);
        return status;
    }

    ucs_fd_exchange_msg_parse(&msgh, &fd);
    if ((nrecvd != sizeof(reply)) || (msgh.msg_flags & MSG_TRUNC)) {
        ucs_debug("fd exchange %s: invalid reply of %zd bytes", path, nrecvd);
        status = UCS_ERR_IO_ERROR;
    } else if (reply == UCS_FD_EXCHANGE_REPLY_FOUND) {
        if (fd >= 0) {
            *fd_p  = fd;
            return UCS_OK;
        }

        ucs_debug("fd exchange %s: reply without a descriptor", path);
        status = UCS_ERR_IO_ERROR;
    } else if (reply == UCS_FD_EXCHANGE_REPLY_NOT_FOUND) {
        status = UCS_ERR_NO_ELEM;
    } else if (reply == UCS_FD_EXCHANGE_REPLY_UNSUPPORTED) {
        ucs_debug("fd exchange %s: protocol version %u is not supported", path,
                  UCS_FD_EXCHANGE_VERSION);
        status = UCS_ERR_UNSUPPORTED;
    } else {
        ucs_debug("fd exchange %s: invalid reply %u", path, reply);
        status = UCS_ERR_IO_ERROR;
    }

    /* An invalid reply may still carry a descriptor */
    if (fd >= 0) {
        close(fd);
    }

    return status;
}

ucs_status_t ucs_fd_exchange_import(const char *path, uint64_t key,
                                    ucs_time_t timeout, int *fd_p)
{
    ucs_fd_exchange_request_t req = {
        .magic   = UCS_FD_EXCHANGE_MAGIC,
        .version = UCS_FD_EXCHANGE_VERSION,
        .key     = key
    };
    ucs_time_t now                = ucs_get_time();
    char reply_path[UCS_FD_EXCHANGE_PATH_MAX];
    struct sockaddr_un server_addr, reply_addr;
    ucs_time_t deadline;
    ucs_status_t status;
    unsigned attempt;
    size_t dir_len;
    int sock_fd;

    /* The path comes from the peer, so validate it like the server directory */
    if (path[0] != '/') {
        ucs_error("fd exchange socket path '%s' is not absolute", path);
        return UCS_ERR_INVALID_PARAM;
    }

    dir_len = strrchr(path, '/') - path;
    if ((dir_len + sizeof(UCS_FD_EXCHANGE_REPLY_SUFFIX)) >
        UCS_FD_EXCHANGE_PATH_MAX) {
        ucs_error("fd exchange socket path '%s' leaves no room for a reply "
                  "socket next to it", path);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    status = ucs_fd_exchange_sockaddr_init(path, &server_addr);
    if (status != UCS_OK) {
        return status;
    }

    /* Saturate, so that UCS_TIME_INFINITY stays infinite */
    deadline = (timeout > (UCS_TIME_INFINITY - now)) ? UCS_TIME_INFINITY :
                                                        (now + timeout);

    status = ucs_socket_create(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0,
                               &sock_fd);
    if (status != UCS_OK) {
        return status;
    }

    /* The reply socket is bound in the server directory, since an address in
     * the abstract namespace would not be visible from another network
     * namespace */
    for (attempt = 0;; ++attempt) {
        ucs_snprintf_safe(reply_path, sizeof(reply_path),
                          UCS_FD_EXCHANGE_REPLY_FMT, (int)dir_len, path,
                          (uint32_t)ucs_generate_uuid(attempt));
        status = ucs_fd_exchange_sockaddr_init(reply_path, &reply_addr);
        if (status != UCS_OK) {
            goto out_close;
        }

        if (bind(sock_fd, (const struct sockaddr*)&reply_addr,
                 sizeof(reply_addr)) == 0) {
            break;
        }

        if ((errno != EADDRINUSE) ||
            (attempt == UCS_FD_EXCHANGE_MAX_ATTEMPTS)) {
            /* No server directory means the server is gone */
            status = (errno == ENOENT) ? UCS_ERR_UNREACHABLE :
                                         UCS_ERR_IO_ERROR;
            ucs_debug("bind(%s) failed: %m", reply_path);
            goto out_close;
        }
    }

    status = ucs_fd_exchange_send_request(sock_fd, &server_addr, &req,
                                          deadline);
    if (status != UCS_OK) {
        goto out_unlink;
    }

    status = ucs_fd_exchange_recv_reply(sock_fd, path, deadline, fd_p);

out_unlink:
    unlink(reply_path);
out_close:
    close(sock_fd);
    return status;
}
