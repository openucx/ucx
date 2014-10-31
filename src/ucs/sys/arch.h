/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_ARCH_H
#define UCS_ARCH_H


/* CPU models */
typedef enum ucs_cpu_model {
    UCS_CPU_MODEL_UNKNOWN,
    UCS_CPU_MODEL_INTEL_IVYBRIDGE,
    UCS_CPU_MODEL_INTEL_SANDYBRIDGE,
    UCS_CPU_MODEL_INTEL_NEHALEM,
    UCS_CPU_MODEL_INTEL_WESTMERE,
    UCS_CPU_MODEL_LAST
} ucs_cpu_model_t;


/* System constants */
#define UCS_SYS_POINTER_SIZE       (sizeof(void*))
#define UCS_SYS_PARAGRAPH_SIZE     16


/* Forward declaration */
double ucs_get_cpuinfo_clock_freq(const char *mhz_header);


#if defined(__x86_64__)
#  include "asm/asm-x86_64.h"
#elif defined(__powerpc64__)
#  include "asm/asm-ppc64.h"
#elif defined(__aarch64__)
#  include "asm/asm-aarch64.h"
#else
#  error "Unsupported architecture"
#endif

/*
 * Define atomic functions
 */
UCS_DEFINE_ATOMIC_ADD(8,  b);
UCS_DEFINE_ATOMIC_ADD(16, w);
UCS_DEFINE_ATOMIC_ADD(32, l);
UCS_DEFINE_ATOMIC_ADD(64, q);

UCS_DEFINE_ATOMIC_FADD(8,  b);
UCS_DEFINE_ATOMIC_FADD(16, w);
UCS_DEFINE_ATOMIC_FADD(32, l);
UCS_DEFINE_ATOMIC_FADD(64, q);

UCS_DEFINE_ATOMIC_SWAP(8,  b);
UCS_DEFINE_ATOMIC_SWAP(16, w);
UCS_DEFINE_ATOMIC_SWAP(32, l);
UCS_DEFINE_ATOMIC_SWAP(64, q);

UCS_DEFINE_ATOMIC_CSWAP(8,  b);
UCS_DEFINE_ATOMIC_CSWAP(16, w);
UCS_DEFINE_ATOMIC_CSWAP(32, l);
UCS_DEFINE_ATOMIC_CSWAP(64, q);


#endif
