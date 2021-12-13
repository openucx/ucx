/*
* Copyright (c) 2021 Nvidia Corporation. All Rights Reserved.
*
* See file LICENSE for terms.
*/

#ifndef UCS_PPC64_LOCK_H
#define UCS_PPC64_LOCK_H

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

static UCS_F_ALWAYS_INLINE int ucs_spin_try_lock(ucs_spinlock_t *lock)
{
    int res;

    res = ucs_atomic_cswap32(&lock->lock, UCS_SPINLOCK_FREE,
                             UCS_SPINLOCK_BUSY) == UCS_SPINLOCK_FREE;

    /* barrier on locked spin */
    if (res) {
        asm volatile ("isync " ::: "memory");
    }

    return res;
}


static UCS_F_ALWAYS_INLINE void ucs_spin_lock(ucs_spinlock_t *lock)
{
    while (!ucs_spin_try_lock(lock)) {
        while (lock->lock != UCS_SPINLOCK_FREE) {
            /* spin */
            asm volatile ("lwsync \n" \
                          "isync  \n" \
                          ::: "memory");
        }
    }
}


static UCS_F_ALWAYS_INLINE void ucs_spin_unlock(ucs_spinlock_t *lock)
{
    static unsigned int lockfree = UCS_SPINLOCK_FREE;

    asm volatile("lwsync" : : : "memory");

    __atomic_store(&lock->lock, &lockfree, __ATOMIC_RELAXED);
}

END_C_DECLS


#endif /* UCS_PPC64_LOCK_H */

