/**
* Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "spinlock.h"

#include <ucs/debug/log.h>


void ucs_spinlock_destroy(ucs_spinlock_t *lock)
{
    int ret;

    ret = pthread_spin_destroy(&lock->lock);
    if (ret != 0) {
        ucs_warn("pthread_spin_destroy() failed: %d", ret);
    }
}

void ucs_recursive_spinlock_destroy(ucs_recursive_spinlock_t *lock)
{
    if (lock->count != 0) {
        ucs_warn("ucs_recursive_spinlock_destroy() failed: busy");
        return;
    }

    ucs_spinlock_destroy(&lock->super);
}
