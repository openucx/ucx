/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
* Copyright (C) Shanghai Zhaoxin Semiconductor Co., Ltd. 2020. ALL RIGHTS RESERVED.
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
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

#define UCS_CPU_CACHE_FILE_FMT   UCS_SYS_FS_CPUS_PATH "/cpu%d/cache/index%d/%s"
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
    },
    [UCS_CPU_VENDOR_GENERIC_RV64G] = {
        .min = UCS_MEMUNITS_INF,
        .max = UCS_MEMUNITS_INF
    },
    [UCS_CPU_VENDOR_NVIDIA] = {
        .min = UCS_MEMUNITS_INF,
        .max = UCS_MEMUNITS_INF
    }
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

const char *ucs_cpu_vendor_name()
{
    static const char *cpu_vendor_names[] = {
        [UCS_CPU_VENDOR_UNKNOWN]       = UCS_VALUE_UNKNOWN_STR,
        [UCS_CPU_VENDOR_INTEL]         = "Intel",
        [UCS_CPU_VENDOR_AMD]           = "AMD",
        [UCS_CPU_VENDOR_GENERIC_ARM]   = "Generic ARM",
        [UCS_CPU_VENDOR_GENERIC_PPC]   = "Generic PPC",
        [UCS_CPU_VENDOR_GENERIC_RV64G] = "Generic RV64G",
        [UCS_CPU_VENDOR_FUJITSU_ARM]   = "Fujitsu ARM",
        [UCS_CPU_VENDOR_ZHAOXIN]       = "Zhaoxin",
        [UCS_CPU_VENDOR_NVIDIA]        = "Nvidia"
    };

    return cpu_vendor_names[ucs_arch_get_cpu_vendor()];
}

const char *ucs_cpu_model_name()
{
    static const char *cpu_model_names[] = {
        [UCS_CPU_MODEL_UNKNOWN]            = UCS_VALUE_UNKNOWN_STR,
        [UCS_CPU_MODEL_INTEL_IVYBRIDGE]    = "IvyBridge",
        [UCS_CPU_MODEL_INTEL_SANDYBRIDGE]  = "SandyBridge",
        [UCS_CPU_MODEL_INTEL_NEHALEM]      = "Nehalem",
        [UCS_CPU_MODEL_INTEL_WESTMERE]     = "Westmere",
        [UCS_CPU_MODEL_INTEL_HASWELL]      = "Haswell",
        [UCS_CPU_MODEL_INTEL_BROADWELL]    = "Broadwell",
        [UCS_CPU_MODEL_INTEL_SKYLAKE]      = "Skylake",
        [UCS_CPU_MODEL_INTEL_ICELAKE]      = "Icelake",
        [UCS_CPU_MODEL_ARM_AARCH64]        = "ARM 64-bit",
        [UCS_CPU_MODEL_AMD_NAPLES]         = "Naples",
        [UCS_CPU_MODEL_AMD_ROME]           = "Rome",
        [UCS_CPU_MODEL_AMD_MILAN]          = "Milan",
        [UCS_CPU_MODEL_AMD_GENOA]          = "Genoa",
        [UCS_CPU_MODEL_ZHAOXIN_ZHANGJIANG] = "Zhangjiang",
        [UCS_CPU_MODEL_ZHAOXIN_WUDAOKOU]   = "Wudaokou",
        [UCS_CPU_MODEL_ZHAOXIN_LUJIAZUI]   = "Lujiazui",
        [UCS_CPU_MODEL_RV64G]              = "RV64G",
        [UCS_CPU_MODEL_NVIDIA_GRACE]       = "Grace"
    };

    return cpu_model_names[ucs_arch_get_cpu_model()];
}
