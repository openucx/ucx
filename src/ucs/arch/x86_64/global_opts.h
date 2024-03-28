/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2023. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_X86_64_GLOBAL_OPTS_H_
#define UCS_X86_64_GLOBAL_OPTS_H_

#include <stddef.h>

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

#if ENABLE_NT_BUFFER_TRANSFER
#define UCS_ARCH_GLOBAL_OPTS_INITALIZER { \
    .builtin_memcpy_min     = UCS_MEMUNITS_AUTO, \
    .builtin_memcpy_max     = UCS_MEMUNITS_AUTO, \
    .nt_buffer_transfer_min = UCS_MEMUNITS_AUTO, \
    .nt_dest_threshold      = UCS_MEMUNITS_AUTO  \
}
#else
#define UCS_ARCH_GLOBAL_OPTS_INITALIZER { \
    .builtin_memcpy_min     = UCS_MEMUNITS_AUTO, \
    .builtin_memcpy_max     = UCS_MEMUNITS_AUTO  \
}
#endif

/* built-in memcpy & nt-buffer-transfer config */
typedef struct ucs_arch_global_opts {
    size_t builtin_memcpy_min;
    size_t builtin_memcpy_max;
#if ENABLE_NT_BUFFER_TRANSFER
    size_t nt_buffer_transfer_min;
    size_t nt_dest_threshold;
#endif
} ucs_arch_global_opts_t;

END_C_DECLS

#endif
