/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
* Copyright (C) Shanghai Zhaoxin Semiconductor Co., Ltd. 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_CPU_H
#define UCS_ARCH_CPU_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler_def.h>
#include <stddef.h>

BEGIN_C_DECLS

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
    UCS_CPU_MODEL_AMD_NAPLES,
    UCS_CPU_MODEL_AMD_ROME,
    UCS_CPU_MODEL_AMD_MILAN,
    UCS_CPU_MODEL_ZHAOXIN_ZHANGJIANG,
    UCS_CPU_MODEL_ZHAOXIN_WUDAOKOU,
    UCS_CPU_MODEL_ZHAOXIN_LUJIAZUI,
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


/* CPU vendors */
typedef enum ucs_cpu_vendor {
    UCS_CPU_VENDOR_UNKNOWN,
    UCS_CPU_VENDOR_INTEL,
    UCS_CPU_VENDOR_AMD,
    UCS_CPU_VENDOR_GENERIC_ARM,
    UCS_CPU_VENDOR_GENERIC_PPC,
    UCS_CPU_VENDOR_FUJITSU_ARM,
    UCS_CPU_VENDOR_ZHAOXIN,
    UCS_CPU_VENDOR_LAST
} ucs_cpu_vendor_t;


/* CPU cache types */
typedef enum ucs_cpu_cache_type {
    UCS_CPU_CACHE_L1d, /**< L1 data cache */
    UCS_CPU_CACHE_L1i, /**< L1 instruction cache */
    UCS_CPU_CACHE_L2,  /**< L2 cache */
    UCS_CPU_CACHE_L3,  /**< L3 cache */
    UCS_CPU_CACHE_LAST
} ucs_cpu_cache_type_t;


/* Built-in memcpy settings */
typedef struct ucs_cpu_builtin_memcpy {
    size_t min;
    size_t max;
} ucs_cpu_builtin_memcpy_t;


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

/* Array of default built-in memcpy settings for different CPU architectures */
extern const ucs_cpu_builtin_memcpy_t ucs_cpu_builtin_memcpy[UCS_CPU_VENDOR_LAST];

#if HAVE___CLEAR_CACHE
/* libc routine declaration */
void __clear_cache(void* beg, void* end);
#endif

/**
 * Get size of CPU cache.
 *
 * @param type  Cache type.
 * @param value Filled with the cache size.
 *
 * @return Cache size value or 0 if cache is not supported or can't be read.
 */
size_t ucs_cpu_get_cache_size(ucs_cpu_cache_type_t type);


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
    __clear_cache(start, end);
#else
    ucs_arch_clear_cache(start, end);
#endif
}

/**
 * Get memory copy bandwidth.
 *
 * @return Memory copy bandwidth estimation based on CPU used.
 */
double ucs_cpu_get_memcpy_bw();


static inline int ucs_cpu_prefer_relaxed_order()
{
    ucs_cpu_vendor_t cpu_vendor = ucs_arch_get_cpu_vendor();
    ucs_cpu_model_t cpu_model   = ucs_arch_get_cpu_model();

    return (cpu_vendor == UCS_CPU_VENDOR_FUJITSU_ARM) ||
           ((cpu_vendor == UCS_CPU_VENDOR_AMD) &&
            ((cpu_model == UCS_CPU_MODEL_AMD_NAPLES) ||
             (cpu_model == UCS_CPU_MODEL_AMD_ROME) ||
             (cpu_model == UCS_CPU_MODEL_AMD_MILAN)));
}


END_C_DECLS

#endif
