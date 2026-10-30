/*
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_ipc_fd.h"

#include <ucs/async/async.h>
#include <ucs/datastruct/list.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/sys.h>

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/un.h>
#include <unistd.h>


/* Bounds how long a peer can stall the async thread or an importer */
#define UCT_ROCM_IPC_FD_SOCK_TIMEOUT_SEC 5


typedef struct {
    uint64_t fd_id;
} uct_rocm_ipc_fd_request_t;


typedef struct {
    int32_t status; /* 0 if a file descriptor is attached */
} uct_rocm_ipc_fd_reply_t;


typedef struct {
    ucs_list_link_t list;
    uint64_t        fd_id;
    int             fd;
    uintptr_t       address;
    size_t          length;
    unsigned        refcount;
} uct_rocm_ipc_fd_entry_t;


static struct {
    pthread_mutex_t lock;
    ucs_list_link_t entries;
    int             listen_fd;
    pid_t           pid;       /* Process the server was started in */
    uint64_t        server_id;
    uint64_t        next_fd_id;
} uct_rocm_ipc_fd_server = {
    .lock       = PTHREAD_MUTEX_INITIALIZER,
    .entries    = UCS_LIST_INITIALIZER(&uct_rocm_ipc_fd_server.entries,
                                       &uct_rocm_ipc_fd_server.entries),
    .listen_fd  = -1,
    .pid        = -1,
    .server_id  = 0,
    .next_fd_id = 1
};


static socklen_t uct_rocm_ipc_fd_sock_addr(pid_t pid, uint64_t server_id,
                                           struct sockaddr_un *addr)
{
    int len;

    /* Abstract namespace: the name starts with a NUL byte */
    memset(addr, 0, sizeof(*addr));
    addr->sun_family = AF_UNIX;
    len = snprintf(addr->sun_path + 1, sizeof(addr->sun_path) - 1,
                   "ucx-rocm-ipc-%d-%" PRIx64, (int)pid, server_id);
    return offsetof(struct sockaddr_un, sun_path) + 1 + len;
}


static void uct_rocm_ipc_fd_set_timeouts(int sock)
{
    struct timeval tv = {.tv_sec = UCT_ROCM_IPC_FD_SOCK_TIMEOUT_SEC};

    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
}


/* Called with the server lock held */
static int uct_rocm_ipc_fd_lookup(uint64_t fd_id)
{
    uct_rocm_ipc_fd_entry_t *entry;

    ucs_list_for_each(entry, &uct_rocm_ipc_fd_server.entries, list) {
        if (entry->fd_id == fd_id) {
            return entry->fd;
        }
    }

    return -1;
}


static void uct_rocm_ipc_fd_serve(int conn)
{
    char cbuf[CMSG_SPACE(sizeof(int))] UCS_V_ALIGNED(sizeof(size_t));
    uct_rocm_ipc_fd_request_t request;
    uct_rocm_ipc_fd_reply_t reply;
    struct cmsghdr *cmsgp;
    struct ucred cred;
    socklen_t cred_len = sizeof(cred);
    struct msghdr msgh;
    struct iovec iov;
    ssize_t ret;
    int fd;

    uct_rocm_ipc_fd_set_timeouts(conn);

    /* Only serve processes of the same user, as pidfd_getfd would require */
    if ((getsockopt(conn, SOL_SOCKET, SO_PEERCRED, &cred, &cred_len) != 0) ||
        (cred.uid != geteuid())) {
        ucs_debug("rocm_ipc fd server: rejecting peer");
        return;
    }

    ret = recv(conn, &request, sizeof(request), MSG_WAITALL);
    if (ret != sizeof(request)) {
        ucs_debug("rocm_ipc fd server: failed to receive request: %m");
        return;
    }

    memset(&msgh, 0, sizeof(msgh));
    memset(cbuf, 0, sizeof(cbuf));
    iov.iov_base    = &reply;
    iov.iov_len     = sizeof(reply);
    msgh.msg_iov    = &iov;
    msgh.msg_iovlen = 1;

    /* Send under the lock so the fd cannot be closed by unexport meanwhile */
    pthread_mutex_lock(&uct_rocm_ipc_fd_server.lock);
    fd = uct_rocm_ipc_fd_lookup(request.fd_id);
    if (fd >= 0) {
        reply.status        = 0;
        msgh.msg_control    = cbuf;
        msgh.msg_controllen = sizeof(cbuf);
        cmsgp               = CMSG_FIRSTHDR(&msgh);
        cmsgp->cmsg_level   = SOL_SOCKET;
        cmsgp->cmsg_type    = SCM_RIGHTS;
        cmsgp->cmsg_len     = CMSG_LEN(sizeof(fd));
        memcpy(CMSG_DATA(cmsgp), &fd, sizeof(fd));
    } else {
        reply.status = -1;
    }

    do {
        ret = sendmsg(conn, &msgh, MSG_NOSIGNAL);
    } while ((ret < 0) && (errno == EINTR));
    pthread_mutex_unlock(&uct_rocm_ipc_fd_server.lock);

    if (ret != sizeof(reply)) {
        ucs_debug("rocm_ipc fd server: failed to send reply: %m");
    }
}


static void uct_rocm_ipc_fd_server_cb(int listen_fd,
                                      ucs_event_set_types_t events, void *arg)
{
    int conn;

    for (;;) {
        conn = accept4(listen_fd, NULL, NULL, SOCK_CLOEXEC);
        if (conn < 0) {
            if (errno == EINTR) {
                continue;
            }

            if ((errno != EAGAIN) && (errno != EWOULDBLOCK)) {
                ucs_debug("rocm_ipc fd server: accept failed: %m");
            }
            return;
        }

        uct_rocm_ipc_fd_serve(conn);
        close(conn);
    }
}


/* Called with the server lock held */
static ucs_status_t uct_rocm_ipc_fd_server_start(void)
{
    struct sockaddr_un addr;
    ucs_status_t status;
    socklen_t addr_len;
    pid_t pid = getpid();
    int sock;

    if ((uct_rocm_ipc_fd_server.listen_fd >= 0) &&
        (uct_rocm_ipc_fd_server.pid == pid)) {
        return UCS_OK;
    }

    /* A forked child does not inherit the parent's async thread, and peers
     * address servers by pid, so it needs a server of its own */
    sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0);
    if (sock < 0) {
        ucs_error("rocm_ipc fd server: socket() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    uct_rocm_ipc_fd_server.server_id = ucs_generate_uuid((uintptr_t)&addr);
    addr_len = uct_rocm_ipc_fd_sock_addr(pid, uct_rocm_ipc_fd_server.server_id,
                                         &addr);
    if ((bind(sock, (struct sockaddr*)&addr, addr_len) != 0) ||
        (listen(sock, SOMAXCONN) != 0)) {
        ucs_error("rocm_ipc fd server: failed to listen on %s: %m",
                  addr.sun_path + 1);
        status = UCS_ERR_IO_ERROR;
        goto err_close;
    }

    status = ucs_async_set_event_handler(UCS_ASYNC_MODE_THREAD_SPINLOCK, sock,
                                         UCS_EVENT_SET_EVREAD,
                                         uct_rocm_ipc_fd_server_cb, NULL, NULL);
    if (status != UCS_OK) {
        ucs_error("rocm_ipc fd server: failed to set event handler: %s",
                  ucs_status_string(status));
        goto err_close;
    }

    uct_rocm_ipc_fd_server.listen_fd = sock;
    uct_rocm_ipc_fd_server.pid       = pid;
    ucs_debug("rocm_ipc fd server listening on %s", addr.sun_path + 1);
    return UCS_OK;

err_close:
    close(sock);
    return status;
}


ucs_status_t uct_rocm_ipc_fd_export(int fd, uintptr_t address, size_t length,
                                    uint64_t *server_id_p, uint64_t *fd_id_p)
{
    uct_rocm_ipc_fd_entry_t *entry;
    ucs_status_t status;

    pthread_mutex_lock(&uct_rocm_ipc_fd_server.lock);
    status = uct_rocm_ipc_fd_server_start();
    if (status != UCS_OK) {
        goto out_unlock;
    }

    ucs_list_for_each(entry, &uct_rocm_ipc_fd_server.entries, list) {
        if ((entry->address == address) && (entry->length == length)) {
            ++entry->refcount;
            close(fd);
            goto out_found;
        }
    }

    entry = ucs_malloc(sizeof(*entry), "uct_rocm_ipc_fd_entry_t");
    if (entry == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_unlock;
    }

    entry->fd       = fd;
    entry->fd_id    = uct_rocm_ipc_fd_server.next_fd_id++;
    entry->address  = address;
    entry->length   = length;
    entry->refcount = 1;
    ucs_list_add_tail(&uct_rocm_ipc_fd_server.entries, &entry->list);

out_found:
    *server_id_p = uct_rocm_ipc_fd_server.server_id;
    *fd_id_p     = entry->fd_id;
out_unlock:
    pthread_mutex_unlock(&uct_rocm_ipc_fd_server.lock);
    return status;
}


void uct_rocm_ipc_fd_unexport(uint64_t fd_id)
{
    uct_rocm_ipc_fd_entry_t *entry;

    pthread_mutex_lock(&uct_rocm_ipc_fd_server.lock);
    ucs_list_for_each(entry, &uct_rocm_ipc_fd_server.entries, list) {
        if (entry->fd_id == fd_id) {
            if (--entry->refcount == 0) {
                ucs_list_del(&entry->list);
                close(entry->fd);
                ucs_free(entry);
            }
            break;
        }
    }
    pthread_mutex_unlock(&uct_rocm_ipc_fd_server.lock);
}


ucs_status_t uct_rocm_ipc_fd_import(pid_t pid, uint64_t server_id,
                                    uint64_t fd_id, int *fd_p)
{
    char cbuf[CMSG_SPACE(sizeof(int))] UCS_V_ALIGNED(sizeof(size_t));
    uct_rocm_ipc_fd_request_t request = {.fd_id = fd_id};
    uct_rocm_ipc_fd_reply_t reply;
    struct sockaddr_un addr;
    struct cmsghdr *cmsgp;
    ucs_status_t status;
    struct msghdr msgh;
    socklen_t addr_len;
    struct iovec iov;
    ssize_t ret;
    int sock;

    sock = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (sock < 0) {
        ucs_error("rocm_ipc fd import: socket() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    uct_rocm_ipc_fd_set_timeouts(sock);
    addr_len = uct_rocm_ipc_fd_sock_addr(pid, server_id, &addr);
    if (connect(sock, (struct sockaddr*)&addr, addr_len) != 0) {
        /* e.g. the peer is in another network namespace */
        ucs_debug("rocm_ipc fd import: cannot connect to %s: %m",
                  addr.sun_path + 1);
        status = UCS_ERR_UNREACHABLE;
        goto out;
    }

    if (send(sock, &request, sizeof(request), MSG_NOSIGNAL) !=
        sizeof(request)) {
        ucs_debug("rocm_ipc fd import: failed to send request: %m");
        status = UCS_ERR_UNREACHABLE;
        goto out;
    }

    memset(&msgh, 0, sizeof(msgh));
    memset(cbuf, 0, sizeof(cbuf));
    iov.iov_base        = &reply;
    iov.iov_len         = sizeof(reply);
    msgh.msg_iov        = &iov;
    msgh.msg_iovlen     = 1;
    msgh.msg_control    = cbuf;
    msgh.msg_controllen = sizeof(cbuf);

    do {
        ret = recvmsg(sock, &msgh, MSG_WAITALL | MSG_CMSG_CLOEXEC);
    } while ((ret < 0) && (errno == EINTR));

    cmsgp = CMSG_FIRSTHDR(&msgh);
    if ((ret != sizeof(reply)) || (reply.status != 0) || (cmsgp == NULL) ||
        (cmsgp->cmsg_level != SOL_SOCKET) || (cmsgp->cmsg_type != SCM_RIGHTS) ||
        (cmsgp->cmsg_len != CMSG_LEN(sizeof(int)))) {
        ucs_debug("rocm_ipc fd import: no fd %" PRIu64 " from pid %d", fd_id,
                  (int)pid);
        status = UCS_ERR_UNREACHABLE;
        goto out;
    }

    memcpy(fd_p, CMSG_DATA(cmsgp), sizeof(*fd_p));
    status = UCS_OK;

out:
    close(sock);
    return status;
}

/* Stop serving when the library is unloaded: remove the async handler before
 * closing the socket it watches, and close the exported descriptors */
UCS_STATIC_CLEANUP
{
    uct_rocm_ipc_fd_entry_t *entry, *tmp;

    if ((uct_rocm_ipc_fd_server.listen_fd >= 0) &&
        (uct_rocm_ipc_fd_server.pid == getpid())) {
        ucs_async_remove_handler(uct_rocm_ipc_fd_server.listen_fd, 1);
        close(uct_rocm_ipc_fd_server.listen_fd);
    }
    uct_rocm_ipc_fd_server.listen_fd = -1;

    ucs_list_for_each_safe(entry, tmp, &uct_rocm_ipc_fd_server.entries, list) {
        ucs_list_del(&entry->list);
        close(entry->fd);
        ucs_free(entry);
    }
}
