/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "vfs_sock.h"

#include <ucs/sys/compiler_def.h>
#include <sys/socket.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <pwd.h>


typedef struct {
    uint8_t action;
} UCS_S_PACKED ucs_vfs_msg_t;


int ucs_vfs_sock_get_address(struct sockaddr_un *un_addr)
{
    struct passwd *pw;
    int ret;

    pw = getpwuid(geteuid());
    if (pw == NULL) {
        return -errno;
    }

    memset(un_addr, 0, sizeof(*un_addr));
    un_addr->sun_family = AF_UNIX;
    snprintf(un_addr->sun_path, sizeof(un_addr->sun_path) - 1,
             "/tmp/ucx-vfs-%s.sock", pw->pw_name);
    return 0;
}

int ucs_vfs_sock_setopt_passcred(int sockfd)
{
    int optval, ret;

    optval = 1;
    ret = setsockopt(sockfd, SOL_SOCKET, SO_PASSCRED, &optval, sizeof(optval));
    if (ret < 0) {
        return -errno;
    }

    return 0;
}

static int ucs_vfs_sock_retval(ssize_t ret, size_t expected)
{
    if (ret == expected) {
        return 0;
    } else if (ret < 0) {
        return -errno;
    } else {
        return -EIO;
    }
}

int ucs_vfs_sock_send(int sockfd, const ucs_vfs_sock_message_t *vfs_msg)
{
    char cbuf[CMSG_SPACE(sizeof(*vfs_msg))] UCS_V_ALIGNED(sizeof(size_t));
    struct cmsghdr *cmsgp;
    struct msghdr msgh;
    ucs_vfs_msg_t msg;
    struct iovec iov;
    ssize_t nsent;

    memset(cbuf, 0, sizeof(cbuf));
    memset(&msgh, 0, sizeof(msgh));
    msg.action      = vfs_msg->action;
    iov.iov_base    = &msg;
    iov.iov_len     = sizeof(msg);
    msgh.msg_iov    = &iov;
    msgh.msg_iovlen = 1;

    if (vfs_msg->action == UCS_VFS_SOCK_ACTION_MOUNT_REPLY) {
        /* send file descriptor */
        msgh.msg_control    = cbuf;
        msgh.msg_controllen = sizeof(cbuf);
        cmsgp               = CMSG_FIRSTHDR(&msgh);
        cmsgp->cmsg_level   = SOL_SOCKET;
        cmsgp->cmsg_len     = CMSG_LEN(sizeof(vfs_msg->fd));
        cmsgp->cmsg_type    = SCM_RIGHTS;
        memcpy(CMSG_DATA(cmsgp), &vfs_msg->fd, sizeof(vfs_msg->fd));
    }

    do {
        nsent = sendmsg(sockfd, &msgh, 0);
    } while ((nsent < 0) && (errno == EINTR));
    return ucs_vfs_sock_retval(nsent, iov.iov_len);
}

int ucs_vfs_sock_recv(int sockfd, ucs_vfs_sock_message_t *vfs_msg)
{
    char cbuf[CMSG_SPACE(sizeof(*vfs_msg))] UCS_V_ALIGNED(sizeof(size_t));
    const struct ucred *cred;
    struct cmsghdr *cmsgp;
    struct msghdr msgh;
    ucs_vfs_msg_t msg;
    struct iovec iov;
    ssize_t nrecvd;

    /* initialize to invalid values */
    vfs_msg->action = UCS_VFS_SOCK_ACTION_LAST;
    vfs_msg->fd     = -1;
    vfs_msg->pid    = -1;

    memset(cbuf, 0, sizeof(cbuf));
    memset(&msgh, 0, sizeof(msgh));
    iov.iov_base        = &msg;
    iov.iov_len         = sizeof(msg);
    msgh.msg_iov        = &iov;
    msgh.msg_iovlen     = 1;
    msgh.msg_control    = cbuf;
    msgh.msg_controllen = sizeof(cbuf);

    do {
        nrecvd = recvmsg(sockfd, &msgh, MSG_WAITALL);
    } while ((nrecvd < 0) && (errno == EINTR));
    if (nrecvd != iov.iov_len) {
        assert(nrecvd < iov.iov_len);
        return ucs_vfs_sock_retval(nrecvd, iov.iov_len);
    }

    vfs_msg->action = msg.action;

    cmsgp = CMSG_FIRSTHDR(&msgh);
    if ((cmsgp == NULL) || (cmsgp->cmsg_level != SOL_SOCKET)) {
        return -EINVAL;
    }

    if (msg.action == UCS_VFS_SOCK_ACTION_MOUNT_REPLY) {
        /* expect file descriptor */
        if ((cmsgp->cmsg_type != SCM_RIGHTS) ||
            (cmsgp->cmsg_len != CMSG_LEN(sizeof(vfs_msg->fd)))) {
            return -EINVAL;
        }

        memcpy(&vfs_msg->fd, CMSG_DATA(cmsgp), sizeof(vfs_msg->fd));
    } else {
        /* expect credentials */
        if ((cmsgp->cmsg_type != SCM_CREDENTIALS) ||
            (cmsgp->cmsg_len != CMSG_LEN(sizeof(*cred)))) {
            return -EINVAL;
        }

        cred = (const struct ucred*)CMSG_DATA(cmsgp);
        if ((cred->uid != getuid()) || (cred->gid != getgid())) {
            return -EPERM;
        }

        if (msg.action == UCS_VFS_SOCK_ACTION_MOUNT) {
            vfs_msg->pid = cred->pid;
        }
    }

    return 0;
}
