/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_VFS_SOCK_H_
#define UCS_VFS_SOCK_H_

#include <ucs/config/types.h>
#include <sys/types.h>
#include <sys/un.h>
#include <stdint.h>

/* This header file defines socket operations for communicating between UCS
 * library and VFS daemon */

#define UCX_VFS_SOCK_DEFAULT_PATH "/run/user/%i/ucx/vfs.sock"


/**
 * VFS socket message type
 */
typedef enum {
    UCS_VFS_SOCK_ACTION_STOP,        /* daemon is asked to stop */
    UCS_VFS_SOCK_ACTION_MOUNT,       /* daemon is asked to mount a file system */
    UCS_VFS_SOCK_ACTION_MOUNT_REPLY, /* daemon sends back FUSE file descriptor */
    UCS_VFS_SOCK_ACTION_NOP,         /* no-operation, used to test connection */
    UCS_VFS_SOCK_ACTION_LAST
} ucs_vfs_sock_action_t;


/**
 * Parameters structure for sending/receiving a message over VFS socket
 */
typedef struct {
    ucs_vfs_sock_action_t action;

    /* If action==MOUNT_REPLY: in/out parameter, holds FUSE file descriptor.
     * Otherwise: unused
     */
    int                   fd;

    /* If action==MOUNT: out parameter, holds the pid of sender process.
     * Otherwise: unused
     */
    pid_t                 pid;
} ucs_vfs_sock_message_t;


/**
 * Return the Unix-domain socket address of the VFS daemon.
 *
 * @param [out] un_addr  Filled with socket address.
 */
void ucs_vfs_sock_get_address(struct sockaddr_un *un_addr);


/**
 * Find directory in the given path, and create it if it does not exist.
 *
 * @param [in]   sock_path Path to the socket file.
 * @param [in]   log_level Log level to use in case of failure.
 * @return 0 on success, or the negative value of errno in case of failure.
 */
int ucs_vfs_sock_mkdir(const char *sock_path, ucs_log_level_t log_level);


/**
 * Enable receiving credentials of the remote process for every message.
 * Typically used by the VFS daemon to verify sender identity.
 *
 * @param [in] fd       Enable SO_PASSCRED on this socket.
 *
 * @return 0 on success, or the negative value of errno in case of failure.
 */
int ucs_vfs_sock_setopt_passcred(int sockfd);


/**
 * Send a message on the VFS socket.
 *
 * @param [in] fd       Socket file descriptor to send the message on.
 * @param [in] vfs_msg  Message to send.
 *
 * @return 0 on success, or the negative value of errno in case of failure.
 */
int ucs_vfs_sock_send(int sockfd, const ucs_vfs_sock_message_t *vfs_msg);


/**
 * Receive a message on the VFS socket.
 *
 * @param [in] fd        Socket file descriptor to receive the message on.
 * @param [out] vfs_msg  Filled with details of the received message.
 *
 * @return 0 on success, or the negative value of errno in case of failure.
 */
int ucs_vfs_sock_recv(int sockfd, ucs_vfs_sock_message_t *vfs_msg);

#endif
