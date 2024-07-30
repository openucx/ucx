/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "vfs_sock.h"

#include <ucs/debug/log_def.h>
#include <ucs/sys/compiler_def.h>
#include <sys/socket.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <pwd.h>
#include <sys/file.h>
#include <sys/shm.h>


#define UCS_VFS_SHM_KEY   13579
#define UCS_VFS_FILE_LOCK "/tmp/ucx-vfs.lock"


/**
 * VFS shared data structure
 */
typedef struct ucs_vfs_shared_data {
    pthread_mutex_t mutex;
    pthread_cond_t  cond;
    ucs_vfs_info_t  info;
} ucs_vfs_shared_data_t;


typedef struct {
    uint8_t action;
} UCS_S_PACKED ucs_vfs_msg_t;


void ucs_vfs_sock_get_address(struct sockaddr_un *un_addr)
{
    struct passwd *pw;
    uid_t euid;

    memset(un_addr, 0, sizeof(*un_addr));
    un_addr->sun_family = AF_UNIX;

    euid = geteuid();
    pw   = getpwuid(euid);
    if (pw != NULL) {
        /* By name */
        snprintf(un_addr->sun_path, sizeof(un_addr->sun_path) - 1,
                 "/tmp/ucx-vfs-%s.sock", pw->pw_name);
    } else {
        /* By number */
        snprintf(un_addr->sun_path, sizeof(un_addr->sun_path) - 1,
                 "/tmp/ucx-vfs-%u.sock", euid);
    }
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

int ucs_vfs_data_destroy_shared(ucs_vfs_data_t *data)
{
    int destroyed = 0;
    struct shmid_ds stat;
    int ret;

    ucs_debug("destroy VFS shared data %p", data);
    if (data->shared != UCS_VFS_UNDEFINED) {
        ret = shmdt(data->shared);
        if (ret != 0) {
            ucs_warn("shmdt(%p) failed: %m", data->shared);
        }
    }

    if (data->shmid != -1) {
        /* Check how many existing attachments exist from other processes */
        ret = shmctl(data->shmid, IPC_STAT, &stat);
        if (ret != 0) {
            ucs_warn("shmctl(%d, IPC_STAT) failed: %m", data->shmid);
        } else if (stat.shm_nattch == 0) {
            /* Last attachment, remove shared memory */
            ucs_debug("destroy VFS shared memory %p", data);
            ret = shmctl(data->shmid, IPC_RMID, NULL);
            if (ret != 0) {
                ucs_warn("shmctl(%d, IPC_RMID) failed: %m", data->shmid);
            } else {
                destroyed = 1;
            }
        }
    }

    return destroyed;
}

static ucs_status_t ucs_vfs_data_init_shared(ucs_vfs_data_t *data)
{
    int init_needed = 0;
    pthread_mutexattr_t mutex_attr;
    pthread_condattr_t cond_attr;
    ucs_status_t status;
    int ret;

    /* Try to create shared memory, succeed only if it's not created yet by
     * another process */
    data->shmid = shmget(UCS_VFS_SHM_KEY, sizeof(ucs_vfs_shared_data_t),
                         IPC_CREAT | IPC_EXCL | 0666);
    if (data->shmid == -1) {
        /* Already exists, try to open it */
        if (errno == EEXIST) {
            data->shmid = shmget(UCS_VFS_SHM_KEY, sizeof(ucs_vfs_shared_data_t),
                                 0666);
        }

        if (data->shmid == -1) {
            ucs_error("shmget() failed: %m");
            return UCS_ERR_NO_MEMORY;
        }
    } else {
        init_needed = 1;
    }

    /* Attach shared memory */
    data->shared = (ucs_vfs_shared_data_t*)shmat(data->shmid, NULL, 0);
    if (data->shared == UCS_VFS_UNDEFINED) {
        ucs_error("shmat(data=%d) failed: %m", data->shmid);
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

#define CHECK_INIT(_init) \
    ret = _init; \
    if (ret != 0) { \
        ucs_error(#_init " failed: %m"); \
        status = UCS_ERR_INVALID_PARAM; \
        goto err; \
    }

    if (init_needed) {
        /* This block is executed by the first process that created shared
         * memory region, either client or daemon */
        CHECK_INIT(pthread_mutexattr_init(&mutex_attr));
        CHECK_INIT(pthread_mutexattr_setpshared(&mutex_attr,
                                                PTHREAD_PROCESS_SHARED));
        CHECK_INIT(pthread_mutex_init(&data->shared->mutex, &mutex_attr));

        CHECK_INIT(pthread_condattr_init(&cond_attr));
        CHECK_INIT(pthread_condattr_setpshared(&cond_attr,
                                               PTHREAD_PROCESS_SHARED));
        CHECK_INIT(pthread_cond_init(&data->shared->cond, &cond_attr));

        ucs_debug("created VFS shared data %p", data);
    } else {
        ucs_debug("loaded VFS shared data %p", data);
    }

    return UCS_OK;

err:
    ucs_vfs_data_destroy_shared(data);
    return status;
}

ucs_status_t ucs_vfs_data_init(ucs_vfs_data_t *data)
{
    ucs_status_t status;
    int ret;

    data->stop = 0;

    /* File lock to avoid data races during shared memory creation phase:
     * It might happen that one process creates shared memory, and the second
     * one attaches to it, and we must guarantee that all initialization is done
     * by the time second process starts using it */
    data->flock = open(UCS_VFS_FILE_LOCK, O_RDWR | O_CREAT | O_CLOEXEC, 0777);
    if (data->flock == -1) {
        ucs_error("open(%s) failed: %m", UCS_VFS_FILE_LOCK);
        return UCS_ERR_IO_ERROR;
    }

    ret = flock(data->flock, LOCK_EX);
    if (ret != 0) {
        ucs_error("flock(%d, LOCK_EX) failed: %m", data->flock);
        return UCS_ERR_IO_ERROR;
    }

    status = ucs_vfs_data_init_shared(data);

    flock(data->flock, LOCK_UN);
    return status;
}

int ucs_vfs_data_destroy(ucs_vfs_data_t *data)
{
    int ret;

    flock(data->flock, LOCK_EX);
    ret = ucs_vfs_data_destroy_shared(data);
    flock(data->flock, LOCK_UN);
    close(data->flock);
    return ret;
}

void ucs_vfs_data_get(ucs_vfs_data_t *data, ucs_vfs_info_t *info)
{
    pthread_mutex_lock(&data->shared->mutex);
    *info = data->shared->info;
    pthread_mutex_unlock(&data->shared->mutex);
}

void ucs_vfs_data_notify(ucs_vfs_data_t *data, const ucs_vfs_info_t *info)
{
    ucs_debug("notify VFS shared data %p", data);
    pthread_mutex_lock(&data->shared->mutex);
    data->shared->info = *info;
    pthread_cond_broadcast(&data->shared->cond);
    pthread_mutex_unlock(&data->shared->mutex);
}

ucs_status_t ucs_vfs_data_wait(ucs_vfs_data_t *data, ucs_vfs_info_t *info)
{
    ucs_debug("wait on VFS shared data %p", data);
    pthread_mutex_lock(&data->shared->mutex);
    /* Wait until interrupted or info sequence is greater than the old value */
    while ((!data->stop) && (data->shared->info.sequence <= info->sequence)) {
        pthread_cond_wait(&data->shared->cond, &data->shared->mutex);
    }

    *info = data->shared->info;
    pthread_mutex_unlock(&data->shared->mutex);
    return data->stop ? UCS_ERR_CANCELED : UCS_OK;
}

void ucs_vfs_data_interrupt(ucs_vfs_data_t *data)
{
    ucs_debug("interrupt VFS shared data %p", data);
    pthread_mutex_lock(&data->shared->mutex);
    data->stop = 1;
    pthread_cond_broadcast(&data->shared->cond);
    pthread_mutex_unlock(&data->shared->mutex);
}
