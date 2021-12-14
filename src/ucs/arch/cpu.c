/**
* Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
* Copyright (C) Shanghai Zhaoxin Semiconductor Co., Ltd. 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/arch/cpu.h>
#include <ucs/arch/generic/cpu.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/sys/stubs.h>
#include <ucs/type/init_once.h>

#define UCS_CPU_CACHE_FILE_FMT   "/sys/devices/system/cpu/cpu%d/cache/index%d/%s"
#define UCS_CPU_CACHE_LEVEL_FILE "level"
#define UCS_CPU_CACHE_TYPE_FILE  "type"
#define UCS_CPU_CACHE_SIZE_FILE  "size"

/* cache size array. index - cache type (ucs_cpu_cache_type_t), value - cache value,
 * 0 means cache is not supported */
static size_t ucs_cpu_cache_size[UCS_CPU_CACHE_LAST] = {0};

static ucs_init_once_t ucs_cache_read_once = UCS_INIT_ONCE_INITIALIZER;

/* cache datatypes */
struct { /* sysfs entries for system cache sizes */
    int         level;
    const char *type;
} const ucs_cpu_cache_sysfs_name[] = {
    [UCS_CPU_CACHE_L1d] = {.level = 1, .type = "Data"},
    [UCS_CPU_CACHE_L1i] = {.level = 1, .type = "Instruction"},
    [UCS_CPU_CACHE_L2]  = {.level = 2, .type = "Unified"},
    [UCS_CPU_CACHE_L3]  = {.level = 3, .type = "Unified"}
};

const ucs_cpu_builtin_memcpy_t ucs_cpu_builtin_memcpy[UCS_CPU_VENDOR_LAST] = {
    [UCS_CPU_VENDOR_UNKNOWN] = {
        .min = UCS_MEMUNITS_INF,
        .max = UCS_MEMUNITS_INF
    },
    [UCS_CPU_VENDOR_INTEL] = {
        .min = 1 * UCS_KBYTE,
        .max = 8 * UCS_MBYTE
    },
    /* TODO: investigate why `rep movsb` is slow for shared buffers
     * on some AMD configurations */
    [UCS_CPU_VENDOR_AMD] = {
        .min = UCS_MEMUNITS_INF,
        .max = UCS_MEMUNITS_INF
    },
    [UCS_CPU_VENDOR_GENERIC_ARM] = {
        .min = UCS_MEMUNITS_INF,
        .max = UCS_MEMUNITS_INF
    },
    [UCS_CPU_VENDOR_GENERIC_PPC] = {
        .min = UCS_MEMUNITS_INF,
        .max = UCS_MEMUNITS_INF
    },
    [UCS_CPU_VENDOR_FUJITSU_ARM] = {
        .min = UCS_MEMUNITS_INF,
        .max = UCS_MEMUNITS_INF
    },
    [UCS_CPU_VENDOR_ZHAOXIN] = {
        .min = UCS_MEMUNITS_INF,
        .max = UCS_MEMUNITS_INF
    }
};

const size_t ucs_cpu_est_bcopy_bw[UCS_CPU_VENDOR_LAST] = {
    [UCS_CPU_VENDOR_UNKNOWN]     = 5800 * UCS_MBYTE,
    [UCS_CPU_VENDOR_INTEL]       = 5800 * UCS_MBYTE,
    [UCS_CPU_VENDOR_AMD]         = 5008 * UCS_MBYTE,
    [UCS_CPU_VENDOR_GENERIC_ARM] = 5800 * UCS_MBYTE,
    [UCS_CPU_VENDOR_GENERIC_PPC] = 5800 * UCS_MBYTE,
    [UCS_CPU_VENDOR_FUJITSU_ARM] = 12000 * UCS_MBYTE
};

static void ucs_sysfs_get_cache_size()
{
    char type_str[32];  /* Data/Instruction/Unified */
    char size_str[32];  /* memunits */
    int cache_index;
    int cpu;
    long level;
    ssize_t file_size;
    ucs_cpu_cache_type_t cache_type;
    ucs_status_t status;

    cpu = ucs_get_first_cpu();

    for (cache_index = 0;; cache_index++) {
        file_size = ucs_read_file_str(type_str, sizeof(type_str), 1,
                                      UCS_CPU_CACHE_FILE_FMT, cpu,
                                      cache_index, UCS_CPU_CACHE_TYPE_FILE);
        if (file_size < 0) {
            return; /* no more files */
        }

        ucs_strtrim(type_str);
        status = ucs_read_file_number(&level, 1, UCS_CPU_CACHE_FILE_FMT,
                                      cpu, cache_index, UCS_CPU_CACHE_LEVEL_FILE);
        if (status != UCS_OK) {
            return; /* no more files */
        }

        /* ok, we found valid directory, let's try to read cache size */
        file_size = ucs_read_file_str(size_str, sizeof(size_str), 1, UCS_CPU_CACHE_FILE_FMT,
                                      cpu, cache_index, UCS_CPU_CACHE_SIZE_FILE);
        if (file_size < 0) {
            return; /* no more files */
        }

        /* now lookup for cache size entry */
        for (cache_type = UCS_CPU_CACHE_L1d; cache_type < UCS_CPU_CACHE_LAST; cache_type++) {
            if ((ucs_cpu_cache_sysfs_name[cache_type].level == level) &&
                !strcasecmp(ucs_cpu_cache_sysfs_name[cache_type].type, type_str)) {
                if (ucs_cpu_cache_size[cache_type] != 0) {
                    break;
                }

                status = ucs_str_to_memunits(ucs_strtrim(size_str),
                                             &ucs_cpu_cache_size[cache_type]);
                if (status != UCS_OK) {
                    ucs_cpu_cache_size[cache_type] = 0; /* reset cache value */
                }
            }
        }
    }
}

size_t ucs_cpu_get_cache_size(ucs_cpu_cache_type_t type)
{
    ucs_status_t status;

    if (type >= UCS_CPU_CACHE_LAST) {
        return 0;
    }

    UCS_INIT_ONCE(&ucs_cache_read_once) {
        UCS_STATIC_ASSERT(ucs_static_array_size(ucs_cpu_cache_size) ==
                          UCS_CPU_CACHE_LAST);
        /* try first CPU-specific algorithm */
        status = ucs_arch_get_cache_size(ucs_cpu_cache_size);
        if (status != UCS_OK) {
            /* read rest of caches from sysfs */
            ucs_sysfs_get_cache_size();
        }
    }

    return ucs_cpu_cache_size[type];
}

double ucs_cpu_get_memcpy_bw()
{
    return ucs_cpu_est_bcopy_bw[ucs_arch_get_cpu_vendor()];
}
