/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "spinlock.h"

#include <ucs/debug/log.h>
#include <string.h>


ucs_status_t ucs_spinlock_init(ucs_spinlock_t *lock)
{
    int ret;

    ret = pthread_spin_init(&lock->lock, 0);
    if (ret != 0) {
        return UCS_ERR_IO_ERROR;
    }

    lock->count = 0;
    lock->owner = 0xfffffffful;
    return UCS_OK;
}

void ucs_spinlock_destroy(ucs_spinlock_t *lock)
{
    int ret;

    if (lock->count != 0) {
        ucs_warn("destroying spinlock %p with use count %d (owner: 0x%lx)",
                 lock, lock->count, lock->owner);
    }

    ret = pthread_spin_destroy(&lock->lock);
    if (ret != 0) {
        ucs_warn("failed to destroy spinlock %p: %s", lock, strerror(ret));
    }
}
