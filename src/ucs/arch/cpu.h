/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_CPU_H
#define UCS_ARCH_CPU_H


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
