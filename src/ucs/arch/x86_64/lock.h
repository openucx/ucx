/*
* Copyright (c) 2021 Nvidia Corporation. All Rights Reserved.
*
* See file LICENSE for terms.
*/

#ifndef UCS_X86_64_LOCK_H
#define UCS_X86_64_LOCK_H

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

#define ucs_spin_try_lock_barrier()

#define ucs_spin_lock_pause() \
    asm volatile("pause" ::: "memory")

#define ucs_spin_unlock_barrier()
#define ucs_spin_unlock_event()

END_C_DECLS

#endif /* UCS_X86_64_LOCK_H */
