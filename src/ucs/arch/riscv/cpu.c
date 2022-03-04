/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#if defined(__riscv)

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/arch/cpu.h>
#include <stdio.h>

static void ucs_rv64_cpuid_from_proc(ucs_rv64_cpuid_t *cpuid)
{
    int count = 0;
    char buf[256];
    char value[256] = {0};
    FILE* f;

    f = fopen("/proc/cpuinfo","r");
    if (!f) {
        return;
    }

    while (fgets(buf, sizeof(buf), f)) {
        if (sscanf(buf, "uarch           : %s", value) == 1) {
                memcpy(cpuid->uarch, value, 256); //printf("%s\n", value);
                ++count;
        }
        else if (sscanf(buf, "isa           : %s", value) == 1) {
                memcpy(cpuid->isa, value, 256); //printf("%s\n", value);
                ++count;
        } else if (sscanf(buf, "mmu           : %s", value) == 1) {
                memcpy(cpuid->mmu, value, 256); //printf("%s\n", value);
                ++count;
        }

        if (count == 3) {
            break;
        }
    }

    fclose(f);
}

void ucs_rv64_cpuid(ucs_rv64_cpuid_t *cpuid)
{
    static ucs_rv64_cpuid_t cached_cpuid;
    static int initialized = 0;

    if (!initialized) {
        ucs_rv64_cpuid_from_proc(&cached_cpuid);
        ucs_memory_cpu_store_fence();
        initialized = 1;
    }

    ucs_memory_cpu_load_fence();
    *cpuid = cached_cpuid;
}

ucs_cpu_vendor_t ucs_arch_get_cpu_vendor()
{
    ucs_rv64_cpuid_t cpuid;
    ucs_rv64_cpuid(&cpuid);

    return UCS_CPU_VENDOR_GENERIC_RV64G;
}

#endif
