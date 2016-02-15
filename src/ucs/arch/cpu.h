/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_CPU_H
#define UCS_ARCH_CPU_H

#include <ucs/sys/math.h>


/* CPU models */
typedef enum ucs_cpu_model {
    UCS_CPU_MODEL_UNKNOWN,
    UCS_CPU_MODEL_INTEL_IVYBRIDGE,
    UCS_CPU_MODEL_INTEL_SANDYBRIDGE,
    UCS_CPU_MODEL_INTEL_NEHALEM,
    UCS_CPU_MODEL_INTEL_WESTMERE,
    UCS_CPU_MODEL_LAST
} ucs_cpu_model_t;

/* CPU flags */
typedef enum ucs_cpu_flag {
    UCS_CPU_FLAG_UNKNOWN    = (-1),
    UCS_CPU_FLAG_CMOV       = UCS_BIT(0),
    UCS_CPU_FLAG_MMX        = UCS_BIT(1),
    UCS_CPU_FLAG_MMX2       = UCS_BIT(2),
    UCS_CPU_FLAG_SSE        = UCS_BIT(3),
    UCS_CPU_FLAG_SSE2       = UCS_BIT(4),
    UCS_CPU_FLAG_SSE3       = UCS_BIT(5),
    UCS_CPU_FLAG_SSSE3      = UCS_BIT(6),
    UCS_CPU_FLAG_SSE41      = UCS_BIT(7),
    UCS_CPU_FLAG_SSE42      = UCS_BIT(8),
    UCS_CPU_FLAG_AVX        = UCS_BIT(9),
    UCS_CPU_FLAG_AVX2       = UCS_BIT(10)
} ucs_cpu_flag_t;

/* System constants */
#define UCS_SYS_POINTER_SIZE       (sizeof(void*))
#define UCS_SYS_PARAGRAPH_SIZE     16


#if defined(__x86_64__)
#  include "x86_64/cpu.h"
#elif defined(__powerpc64__)
#  include "ppc64/cpu.h"
#elif defined(__aarch64__)
#  include "aarch64/cpu.h"
#else
#  error "Unsupported architecture"
#endif

#endif
