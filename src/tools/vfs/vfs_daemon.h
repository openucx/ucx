/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
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


#define vfs_error(_fmt, ...) \
    { \
        fprintf(stderr, "Error: " _fmt "\n", ##__VA_ARGS__); \
    }


#define vfs_log(_fmt, ...) \
    { \
        if (g_opts.verbose) { \
            fprintf(stderr, "Debug: " _fmt "\n", ##__VA_ARGS__); \
        } \
    }


typedef struct {
    int        action;
    int        foreground;
    int        verbose;
    const char *mountpoint_dir;
    const char *mount_opts;
    const char *sock_path;
} vfs_opts_t;


extern vfs_opts_t g_opts;
extern const char *vfs_action_names[];

int vfs_mount(int pid);

int vfs_unmount(int pid);

int vfs_server_loop(int listen_fd);

#endif
