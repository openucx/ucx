/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#if defined(__x86_64__)

.globl ucs_arch_is_mem_accessible
.type  ucs_arch_is_mem_accessible,@function
ucs_arch_is_mem_accessible:
    movb  0(%rdi), %dil
    movl  $1, %eax
    ret
.global ucs_arch_is_mem_accessible_restore
ucs_arch_is_mem_accessible_restore:
    xor   %eax, %eax
    ret

#endif
