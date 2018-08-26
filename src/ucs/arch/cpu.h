/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_CPU_H
#define UCS_ARCH_CPU_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler_def.h>


/* CPU models */
typedef enum ucs_cpu_model {
    UCS_CPU_MODEL_UNKNOWN,
    UCS_CPU_MODEL_INTEL_IVYBRIDGE,
    UCS_CPU_MODEL_INTEL_SANDYBRIDGE,
    UCS_CPU_MODEL_INTEL_NEHALEM,
    UCS_CPU_MODEL_INTEL_WESTMERE,
    UCS_CPU_MODEL_INTEL_HASWELL,
    UCS_CPU_MODEL_INTEL_BROADWELL,
    UCS_CPU_MODEL_INTEL_SKYLAKE,
    UCS_CPU_MODEL_ARM_AARCH64,
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
#define UCS_SYS_PCI_MAX_PAYLOAD    512


#if defined(__x86_64__)
#  include "x86_64/cpu.h"
#elif defined(__powerpc64__)
#  include "ppc64/cpu.h"
#elif defined(__aarch64__)
#  include "aarch64/cpu.h"
#else
#  error "Unsupported architecture"
#endif

#if defined(HAVE_CACHE_LINE_SIZE)
#define UCS_SYS_CACHE_LINE_SIZE    HAVE_CACHE_LINE_SIZE
#else
#define UCS_SYS_CACHE_LINE_SIZE    UCS_ARCH_CACHE_LINE_SIZE
#endif

/**
 * Clear processor data and instruction caches, intended for
 * self-modifying code.
 *
 * @start start of region to clear cache, including address
 * @end   end of region to clear cache, excluding address
 */
static inline void ucs_clear_cache(void *start, void *end)
{
#if HAVE___CLEAR_CACHE
    /* do not allow global declaration of compiler intrinsic */
    void __clear_cache(void* beg, void* end);

    __clear_cache(start, end);
#else
    ucs_arch_clear_cache(start, end);
#endif
}
#endif
