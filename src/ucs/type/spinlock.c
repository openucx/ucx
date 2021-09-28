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

int ucs_spinlock_is_held(ucs_spinlock_t *lock)
{
    if (!ucs_spin_try_lock(lock)) {
        return 1; /* If can't lock, it is already locked */
    }

    ucs_spin_unlock(lock);
    return 0;
}

int ucs_recursive_spinlock_is_held(const ucs_recursive_spinlock_t *lock)
{
    return ucs_recursive_spin_is_owner(lock, pthread_self());
}
