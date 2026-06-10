/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#if defined(__aarch64__)

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/arch/cpu.h>
#include <stdio.h>

#if HAVE_SYS_AUXV_H
#  include <sys/auxv.h>
#endif

/* Older userspace headers may not expose Arm64 HWCAP3 definitions. */
#ifndef AT_HWCAP3
#  define AT_HWCAP3 29
#endif

#ifndef HWCAP3_LS64
#  define HWCAP3_LS64 (1UL << 3)
#endif

static void ucs_aarch64_cpuid_from_proc(ucs_aarch64_cpuid_t *cpuid)
{
    char buf[256];
    int value;
    FILE* f;

    cpuid->implementer  = -1;
    cpuid->architecture = -1;
    cpuid->variant      = -1;
    cpuid->part         = -1;
    cpuid->revision     = -1;

    f = fopen("/proc/cpuinfo","r");
    if (!f) {
        return;
    }

    while (fgets(buf, sizeof(buf), f)) {
        if (sscanf(buf, "CPU implementer : 0x%x", &value) == 1) {
            cpuid->implementer  = value;
        } else if (sscanf(buf, "CPU architecture : %d", &value) == 1) {
            cpuid->architecture = value;
        } else if (sscanf(buf, "CPU variant : 0x%x", &value) == 1) {
            cpuid->variant      = value;
        } else if (sscanf(buf, "CPU part : 0x%x", &value) == 1) {
            cpuid->part         = value;
        } else if (sscanf(buf, "CPU revision : %d", &value) == 1) {
            cpuid->revision     = value;
        }

        if ((cpuid->implementer != -1) && (cpuid->architecture != -1) &&
            (cpuid->variant != -1) && (cpuid->part != -1) && (cpuid->revision != -1)) {
            break;
        }
    }

    fclose(f);
}

void ucs_aarch64_cpuid(ucs_aarch64_cpuid_t *cpuid)
{
    static ucs_aarch64_cpuid_t cached_cpuid;
    static int initialized = 0;

    if (!initialized) {
        ucs_aarch64_cpuid_from_proc(&cached_cpuid);
        ucs_memory_cpu_store_fence();
        initialized = 1;
    }

    ucs_memory_cpu_load_fence();
    *cpuid = cached_cpuid;
}

ucs_cpu_flag_t ucs_arch_get_cpu_flag()
{
    static ucs_cpu_flag_t cpu_flag;
    static int initialized = 0;

    if (!initialized) {
        ucs_cpu_flag_t result = 0;

#if HAVE_SYS_AUXV_H && HAVE_DECL_GETAUXVAL
        if (getauxval(AT_HWCAP3) & HWCAP3_LS64) {
            result |= UCS_CPU_FLAG_ST64B;
        }
#endif

        cpu_flag = result;
        ucs_memory_cpu_store_fence();
        initialized = 1;
    }

    ucs_memory_cpu_load_fence();
    return cpu_flag;
}

#endif
