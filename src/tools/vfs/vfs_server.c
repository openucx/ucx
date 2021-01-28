/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "vfs_daemon.h"

#include <ucs/datastruct/khash.h>
#include <sys/poll.h>
#include <signal.h>


#define VFS_MAX_FDS 1024

typedef enum {
    VFS_FD_STATE_LISTENING,
    VFS_FD_STATE_ACCEPTED,
    VFS_FD_STATE_MOUNTED,
    VFS_FD_STATE_FD_SENT,
    VFS_FD_STATE_CLOSED
} vfs_socket_state_t;

typedef struct {
    vfs_socket_state_t state;
    pid_t              pid;
    int                fuse_fd;
} vfs_serever_fd_state_t;

typedef struct {
    vfs_serever_fd_state_t fd_state[VFS_MAX_FDS];
    struct pollfd          poll_fds[VFS_MAX_FDS];
    int                    nfds;
    int                    stop;
} vfs_server_context_t;

static vfs_server_context_t vfs_server_context;

static const char *vfs_server_fd_state_names[] = {
    [VFS_FD_STATE_LISTENING] = "LISTENING",
    [VFS_FD_STATE_ACCEPTED]  = "ACCEPTED",
    [VFS_FD_STATE_MOUNTED]   = "MOUNTED",
    [VFS_FD_STATE_FD_SENT]   = "FD_SENT",
    [VFS_FD_STATE_CLOSED]    = "CLOSED"
};

static void vfs_server_log_context(int events)
{
    vfs_serever_fd_state_t *fd_state;
    char log_message[1024];
    struct pollfd *pfd;
    char *p, *endp;
    int idx;

    if (g_opts.verbose < 2) {
        return;
    }

    p    = log_message;
    endp = log_message + sizeof(log_message);

    for (idx = 0; idx < vfs_server_context.nfds; ++idx) {
        pfd      = &vfs_server_context.poll_fds[idx];
        fd_state = &vfs_server_context.fd_state[idx];
        snprintf(p, endp - p, "[%d]{%c %d%s%s %d} ", idx,
                 vfs_server_fd_state_names[fd_state->state][0],
                 vfs_server_context.poll_fds[idx].fd,
                 (events && (pfd->revents & POLLIN)) ? "i" : "",
                 (events && (pfd->revents & POLLOUT)) ? "o" : "",
                 fd_state->pid);
        p += strlen(p);
    }

    *(p - 1) = '\0';

    vfs_log("%s", log_message);
}

static int vfs_server_poll_events()
{
    int ret;

    vfs_server_log_context(0);

    ret = poll(vfs_server_context.poll_fds, vfs_server_context.nfds, -1);
    if (ret < 0) {
        ret = -errno;
        if (errno != EINTR) {
            vfs_error("poll(nfds=%d) failed: %m", vfs_server_context.nfds)
        }
        return ret;
    }

    vfs_server_log_context(1);
    return 0;
}

static void vfs_server_close_fd(int fd)
{
    int ret = close(fd);
    if (ret < 0) {
        vfs_error("failed to close fd %d: %m", fd);
    }
}

static void vfs_server_log_fd(int idx, const char *message)
{
    vfs_serever_fd_state_t *fd_state = &vfs_server_context.fd_state[idx];
    struct pollfd *pfd               = &vfs_server_context.poll_fds[idx];

    vfs_log("%s fd[%d]=%d %s, pid: %d fuse_fd: %d", message, idx, pfd->fd,
            vfs_server_fd_state_names[fd_state->state], fd_state->fuse_fd,
            fd_state->pid);
}

static int vfs_server_add_fd(int fd, vfs_socket_state_t state)
{
    int idx, ret;

    ret = fcntl(fd, F_GETFL);
    if (ret < 0) {
        vfs_error("fcntl(%d, F_GETFL) failed: %m", fd);
        return -errno;
    }

    ret = fcntl(fd, F_SETFL, ret | O_NONBLOCK);
    if (ret < 0) {
        vfs_error("fcntl(%d, F_SETFL) failed: %m", fd);
        return -errno;
    }

    idx                                      = vfs_server_context.nfds++;
    vfs_server_context.fd_state[idx].state   = state;
    vfs_server_context.fd_state[idx].pid     = -1;
    vfs_server_context.fd_state[idx].fuse_fd = -1;
    vfs_server_context.poll_fds[idx].events  = POLLIN;
    vfs_server_context.poll_fds[idx].fd      = fd;
    vfs_server_context.poll_fds[idx].revents = 0;

    vfs_server_log_fd(idx, "added");
    return 0;
}

static void vfs_server_remove_fd(int idx)
{
    vfs_server_log_fd(idx, "removing");

    switch (vfs_server_context.fd_state[idx].state) {
    case VFS_FD_STATE_FD_SENT:
    case VFS_FD_STATE_MOUNTED:
        vfs_server_close_fd(vfs_server_context.fd_state[idx].fuse_fd);
        vfs_unmount(vfs_server_context.fd_state[idx].pid);
        /* Fall through */
    case VFS_FD_STATE_ACCEPTED:
        vfs_server_close_fd(vfs_server_context.poll_fds[idx].fd);
        /* Fall through */
    default:
        break;
    }

    vfs_server_context.fd_state[idx].state   = VFS_FD_STATE_CLOSED;
    vfs_server_context.fd_state[idx].pid     = -1;
    vfs_server_context.fd_state[idx].fuse_fd = -1;
    vfs_server_context.poll_fds[idx].events  = 0;
    vfs_server_context.poll_fds[idx].fd      = -1;
    vfs_server_context.poll_fds[idx].revents = 0;
}

static void vfs_server_remove_all_fds()
{
    while (vfs_server_context.nfds > 0) {
        vfs_server_remove_fd(--vfs_server_context.nfds);
    }
}

static void vfs_server_accept(int listen_fd)
{
    int ret, connfd;

    connfd = accept(listen_fd, NULL, NULL);
    if (connfd < 0) {
        vfs_error("accept(listen_fd=%d) failed: %m", listen_fd);
        return;
    }

    ret = ucs_vfs_sock_setopt_passcred(connfd);
    if (ret < 0) {
        close(connfd);
        return;
    }

    vfs_server_add_fd(connfd, VFS_FD_STATE_ACCEPTED);
}

static void vfs_server_mount(int idx, pid_t pid)
{
    int fuse_fd;

    if (pid < 0) {
        vfs_error("received invalid pid: %d", pid);
        vfs_server_close_fd(idx);
        return;
    }

    fuse_fd = vfs_mount(pid);
    if (fuse_fd < 0) {
        vfs_server_close_fd(idx);
        return;
    }

    vfs_server_context.fd_state[idx].state   = VFS_FD_STATE_MOUNTED;
    vfs_server_context.fd_state[idx].pid     = pid;
    vfs_server_context.fd_state[idx].fuse_fd = fuse_fd;
    vfs_server_context.poll_fds[idx].events |= POLLOUT;
}

static void vfs_server_recv(int idx)
{
    ucs_vfs_sock_message_t vfs_msg_in;
    char message[64];
    int ret;

    ret = ucs_vfs_sock_recv(vfs_server_context.poll_fds[idx].fd, &vfs_msg_in);
    if (ret < 0) {
        vfs_error("failed to receive a message: %d (%s)", ret, strerror(-ret));
        vfs_server_remove_fd(idx);
        return;
    }

    snprintf(message, sizeof(message), "got action '%s' on",
             vfs_action_names[vfs_msg_in.action]);
    vfs_server_log_fd(idx, message);

    switch (vfs_msg_in.action) {
    case UCS_VFS_SOCK_ACTION_STOP:
        vfs_server_context.stop = 1;
        break;
    case UCS_VFS_SOCK_ACTION_MOUNT:
        vfs_server_mount(idx, vfs_msg_in.pid);
        break;
    case UCS_VFS_SOCK_ACTION_NOP:
        vfs_server_remove_fd(idx);
        break;
    default:
        vfs_error("ignoring invalid action %d", vfs_msg_in.action);
        vfs_server_remove_fd(idx);
        break;
    }
}

static void vfs_server_handle_pollin(int idx)
{
    switch (vfs_server_context.fd_state[idx].state) {
    case VFS_FD_STATE_LISTENING:
        vfs_server_accept(vfs_server_context.poll_fds[idx].fd);
        break;
    case VFS_FD_STATE_ACCEPTED:
        vfs_server_recv(idx);
        break;
    case VFS_FD_STATE_FD_SENT:
        vfs_server_remove_fd(idx);
        break;
    default:
        vfs_server_log_fd(idx, "unexpected POLLIN event on");
        vfs_server_remove_fd(idx);
        break;
    }
}

static void vfs_server_handle_pollout(int idx)
{
    ucs_vfs_sock_message_t vfs_msg_out;
    int ret;

    if (vfs_server_context.fd_state[idx].state != VFS_FD_STATE_MOUNTED) {
        vfs_server_log_fd(idx, "unexpected POLLOUT event on");
        vfs_server_remove_fd(idx);
        return;
    }

    /* Send reply with file descriptor from fuse mount */
    vfs_msg_out.action = UCS_VFS_SOCK_ACTION_MOUNT_REPLY;
    vfs_msg_out.fd     = vfs_server_context.fd_state[idx].fuse_fd;
    ret = ucs_vfs_sock_send(vfs_server_context.poll_fds[idx].fd, &vfs_msg_out);
    if (ret < 0) {
        vfs_error("failed to send a message: %d", ret);
        vfs_server_remove_fd(idx);
        return;
    }

    vfs_server_log_fd(idx, "sent fuse_fd on");
    vfs_server_context.fd_state[idx].state   = VFS_FD_STATE_FD_SENT;
    vfs_server_context.poll_fds[idx].events &= ~POLLOUT;
}

static void vfs_server_copy_fd_state(int dest_idx, int src_idx)
{
    if (dest_idx != src_idx) {
        vfs_server_context.fd_state[dest_idx] =
                vfs_server_context.fd_state[src_idx];
        vfs_server_context.poll_fds[dest_idx] =
                vfs_server_context.poll_fds[src_idx];
    }
}

static void vfs_server_sighandler(int signo)
{
    vfs_server_context.stop = 1;
}

static void vfs_server_set_sighandler()
{
    struct sigaction sigact;

    sigact.sa_handler = vfs_server_sighandler;
    sigact.sa_flags   = 0;
    sigemptyset(&sigact.sa_mask);

    sigaction(SIGINT, &sigact, NULL);
    sigaction(SIGHUP, &sigact, NULL);
    sigaction(SIGTERM, &sigact, NULL);
}

int vfs_server_loop(int listen_fd)
{
    int idx, valid_idx;
    int ret;

    vfs_server_context.nfds = 0;
    vfs_server_context.stop = 0;

    vfs_server_set_sighandler();

    vfs_server_add_fd(listen_fd, VFS_FD_STATE_LISTENING);

    while (!vfs_server_context.stop) {
        ret = vfs_server_poll_events();
        if (ret < 0) {
            if (ret == -EINTR) {
                continue;
            } else {
                return ret;
            }
        }

        valid_idx = 0;
        for (idx = 0; idx < vfs_server_context.nfds; ++idx) {
            if (vfs_server_context.poll_fds[idx].events == 0) {
                vfs_server_copy_fd_state(valid_idx++, idx);
                continue;
            }

            if (vfs_server_context.poll_fds[idx].revents & POLLIN) {
                vfs_server_handle_pollin(idx);
            }
            if (vfs_server_context.poll_fds[idx].revents & POLLOUT) {
                vfs_server_handle_pollout(idx);
            }

            if (vfs_server_context.fd_state[idx].state != VFS_FD_STATE_CLOSED) {
                vfs_server_copy_fd_state(valid_idx++, idx);
            }
        }

        vfs_server_context.nfds = valid_idx;
    }

    vfs_server_remove_all_fds();

    return 0;
}
