/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_VFS_SOCK_H_
#define UCS_VFS_SOCK_H_

#include <ucs/time/time.h>
#include <sys/types.h>
#include <sys/un.h>
#include <stdint.h>

/* This header file defines socket operations for communicating between UCS
 * library and VFS daemon */

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


/* VFS daemon information stored in the shared memory */
typedef struct {
    pid_t      pid;
    uint64_t   sequence;
    ucs_time_t start_time;
} ucs_vfs_info_t;


/* Handler structure to operate VFS shared memory */
typedef struct {
    struct ucs_vfs_shared_data *shared;
    int                        flock;
    int                        shmid;
    int                        stop;
} ucs_vfs_data_t;


#define UCS_VFS_UNDEFINED ((void *)-1)
#define UCS_VFS_DATA_INIT {UCS_VFS_UNDEFINED, -1, -1, 0}


/**
 * Initialize VFS shared memory handler.
 * @param [in] data  VFS shared memory handler.
 * @return UCS_OK on success, or an error code on failure.
 */
ucs_status_t ucs_vfs_data_init(ucs_vfs_data_t *data);


/**
 * Destroy VFS shared memory handler.
 * @param [in] data  VFS shared memory handler.
 */
void ucs_vfs_data_destroy(ucs_vfs_data_t *data);


/**
 * Get VFS daemon information.
 * @param [in]  data  VFS shared memory handler.
 * @param [out] info  Filled with VFS daemon information.
 */
void ucs_vfs_data_get(ucs_vfs_data_t *data, ucs_vfs_info_t *info);


/**
 * Update VFS daemon information and notify all the waiting processes.
 * @param [in] data  VFS shared memory handler.
 * @param [in] info  New VFS daemon information.
 */
void ucs_vfs_data_notify(ucs_vfs_data_t *data, const ucs_vfs_info_t *info);


/**
 * Wait for update of VFS daemon information to change.
 * This operation is blocking and may be interrupted by ucs_vfs_data_interrupt.
 * @param [in]     data  VFS shared memory handler.
 * @param [in/out] info  Input with the current VFS daemon information,
 *                       output with the new information.
 * @return UCS_OK on success,
 *         UCS_ERR_CANCELED on interrupt from ucs_vfs_data_interrupt
 */
ucs_status_t ucs_vfs_data_wait(ucs_vfs_data_t *data, ucs_vfs_info_t *info);


/**
 * Interrupt the blocking ucs_vfs_data_wait operation.
 * @param [in] data  VFS shared memory handler.
 */
void ucs_vfs_data_interrupt(ucs_vfs_data_t *data);

#endif
