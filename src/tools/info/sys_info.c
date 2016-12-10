/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucx_info.h"

#include <ucs/sys/sys.h>
#include <ucs/time/time.h>
#include <sys/mman.h>
#include <string.h>


static const char* cpu_model_names[] = {
    [UCS_CPU_MODEL_UNKNOWN]           = "unknown",
    [UCS_CPU_MODEL_INTEL_IVYBRIDGE]   = "IvyBridge",
    [UCS_CPU_MODEL_INTEL_SANDYBRIDGE] = "SandyBridge",
    [UCS_CPU_MODEL_INTEL_NEHALEM]     = "Nehalem",
    [UCS_CPU_MODEL_INTEL_WESTMERE]    = "Westmere"
};

static double measure_memcpy_bandwidth(size_t size)
{
    ucs_time_t start_time, end_time;
    void *src, *dst;
    double result = 0.0;
    int iter;

    src = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (src == MAP_FAILED) {
        goto out;
    }

    dst = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (dst == MAP_FAILED) {
        goto out_unmap_src;
    }

    memset(dst, 0, size);
    memset(src, 0, size);
    memcpy(dst, src, size);

    iter = 0;
    start_time = ucs_get_time();
    do {
        memcpy(dst, src, size);
        end_time = ucs_get_time();
        ++iter;
    } while (end_time < start_time + ucs_time_from_sec(0.5));

    result = size * iter / ucs_time_to_sec(end_time - start_time);

    munmap(dst, size);
out_unmap_src:
    munmap(src, size);
out:
    return result;
}

void print_sys_info()
{
    size_t size;

    printf("# Timer frequency: %.3f MHz\n", ucs_get_cpu_clocks_per_sec() / 1e6);
    printf("# CPU model: %s\n", cpu_model_names[ucs_arch_get_cpu_model()]);

    printf("# Memcpy bandwidth:\n");
    for (size = 4096; size <= 256 * UCS_MBYTE; size *= 2) {
        printf("#     %10zu bytes: %.3f MB/s\n", size,
               measure_memcpy_bandwidth(size) / UCS_MBYTE);
    }
}
