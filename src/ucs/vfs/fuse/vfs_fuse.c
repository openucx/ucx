/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/vfs/sock/vfs_sock.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <ucs/sys/compiler.h>
#include <pthread.h>


static struct {
    pthread_t thread_id;
} ucs_vfs_fuse_context = {
    .thread_id = -1,
};

static void *ucs_vfs_fuse_thread_func(void *arg)
{
    return NULL;
}

static void ucs_fuse_thread_stop()
{
    pthread_join(ucs_vfs_fuse_context.thread_id, NULL);
}

UCS_STATIC_INIT
{
    pthread_create(&ucs_vfs_fuse_context.thread_id, NULL,
                   ucs_vfs_fuse_thread_func, NULL);
}

UCS_STATIC_CLEANUP
{
    if (ucs_vfs_fuse_context.thread_id != -1) {
        ucs_fuse_thread_stop();
    }
}
