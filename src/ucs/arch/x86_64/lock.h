/*
* Copyright (c) 2021 Nvidia Corporation. All Rights Reserved.
*
* See file LICENSE for terms.
*/

#ifndef UCS_X86_64_LOCK_H
#define UCS_X86_64_LOCK_H

#include <ucs/sys/compiler_def.h>
#include <ucs/arch/lock.h>

BEGIN_C_DECLS

static UCS_F_ALWAYS_INLINE int ucs_spin_try_lock(ucs_spinlock_t *lock)
{
    return ucs_atomic_cswap32(&lock->lock, UCS_SPINLOCK_FREE,
                              UCS_SPINLOCK_BUSY) == UCS_SPINLOCK_FREE;
}


static UCS_F_ALWAYS_INLINE void ucs_spin_lock(ucs_spinlock_t *lock)
{
    while (!ucs_spin_try_lock(lock)) {
        while (lock->lock != UCS_SPINLOCK_FREE) {
            /* spin */
            asm volatile("pause" ::: "memory");
        }
    }
}


static UCS_F_ALWAYS_INLINE void ucs_spin_unlock(ucs_spinlock_t *lock)
{
    static unsigned int lockfree = UCS_SPINLOCK_FREE;

    __atomic_store(&lock->lock, &lockfree, __ATOMIC_RELAXED);
}

END_C_DECLS

#endif /* UCS_X86_64_LOCK_H */
