/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#ifndef VFS_DAEMON_H_
#define VFS_DAEMON_H_

#include <ucs/vfs/sock/vfs_sock.h>
#include <ucs/sys/compiler_def.h>
#include <sys/socket.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <fuse.h>


#define VFS_DEFAULT_MOUNTPOINT_DIR "/tmp/ucx"
#define VFS_FUSE_MOUNT_PROG        "fusermount3"


enum {
    VFS_DAEMON_ACTION_START = UCS_VFS_SOCK_ACTION_NOP
};


#define vfs_error ucs_error
#define vfs_log   ucs_debug


typedef struct {
    int        action;
    int        foreground;
    int        verbose;
    const char *mountpoint_dir;
    const char *mount_opts;
} vfs_opts_t;


extern vfs_opts_t g_opts;
extern const char *vfs_action_names[];

int vfs_mount(int pid);

int vfs_unmount(int pid);

int vfs_server_loop(int listen_fd);

#endif
