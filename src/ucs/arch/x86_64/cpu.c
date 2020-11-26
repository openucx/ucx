/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
* Copyright (C) Shanghai Zhaoxin Semiconductor Co., Ltd. 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#if defined(__x86_64__)

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>

#define X86_CPUID_GENUINEINTEL    "GenuntelineI" /* GenuineIntel in magic notation */
#define X86_CPUID_AUTHENTICAMD    "AuthcAMDenti" /* AuthenticAMD in magic notation */
#define X86_CPUID_CENTAURHAULS    "CentaulsaurH" /* CentaurHauls in magic notation */
#define X86_CPUID_SHANGHAI        "  Shai  angh" /* Shanghai in magic notation */
#define X86_CPUID_GET_MODEL       0x00000001u
#define X86_CPUID_GET_BASE_VALUE  0x00000000u
#define X86_CPUID_GET_EXTD_VALUE  0x00000007u
#define X86_CPUID_GET_MAX_VALUE   0x80000000u
#define X86_CPUID_INVARIANT_TSC   0x80000007u
#define X86_CPUID_GET_CACHE_INFO  0x00000002u
#define X86_CPUID_GET_LEAF4_INFO  0x00000004u

#define X86_CPU_CACHE_RESERVED    0x80000000
#define X86_CPU_CACHE_TAG_L1_ONLY 0x40
#define X86_CPU_CACHE_TAG_LEAF4   0xff

#if defined (__SSE4_1__)
#define _mm_load(a)    _mm_stream_load_si128((__m128i *) (a))
#define _mm_store(a,v) _mm_storeu_si128((__m128i *) (a), (v))
#endif


typedef enum ucs_x86_cpu_cache_type {
    X86_CPU_CACHE_TYPE_DATA        = 1,
    X86_CPU_CACHE_TYPE_INSTRUCTION = 2,
    X86_CPU_CACHE_TYPE_UNIFIED     = 3
} ucs_x86_cpu_cache_type_t;

/* CPU version */
typedef union ucs_x86_cpu_version {
    struct {
        unsigned stepping   : 4;
        unsigned model      : 4;
        unsigned family     : 4;
        unsigned type       : 2;
        unsigned unused     : 2;
        unsigned ext_model  : 4;
        unsigned ext_family : 8;
    };
    uint32_t reg;
} UCS_S_PACKED ucs_x86_cpu_version_t;

/* cache datatypes */
typedef struct ucs_x86_cpu_cache_info {
    unsigned                 level;
    ucs_x86_cpu_cache_type_t type;
} UCS_S_PACKED ucs_x86_cpu_cache_info_t;

typedef union ucs_x86_cache_line_reg_info {
    uint32_t reg;
    struct {
        unsigned size          : 12;
        unsigned partitions    : 10;
        unsigned associativity : 10;
    };
    struct {
        unsigned type          : 5;
        unsigned level         : 3;
    };
} UCS_S_PACKED ucs_x86_cache_line_reg_info_t;

typedef union ucs_x86_cpu_registers {
    struct {
        union {
            uint32_t     eax;
            uint8_t      max_iter; /* leaf 2 - max iterations */
        };
        union {
            struct {
                uint32_t ebx;
                uint32_t ecx;
                uint32_t edx;
            };
            char         id[sizeof(uint32_t) * 3]; /* leaf 0 - CPU ID */
        };
    };
    union {
        uint32_t         value;
        uint8_t          tag[sizeof(uint32_t)];
    }                    reg[4]; /* leaf 2 tags */
} UCS_S_PACKED ucs_x86_cpu_registers;

typedef struct ucs_x86_cpu_cache_size_codes {
    ucs_cpu_cache_type_t type;
    size_t               size;
} ucs_x86_cpu_cache_size_codes_t;


ucs_ternary_auto_value_t ucs_arch_x86_enable_rdtsc = UCS_TRY;

static const ucs_x86_cpu_cache_info_t x86_cpu_cache[] = {
    [UCS_CPU_CACHE_L1d] = {.level = 1, .type = X86_CPU_CACHE_TYPE_DATA},
    [UCS_CPU_CACHE_L1i] = {.level = 1, .type = X86_CPU_CACHE_TYPE_INSTRUCTION},
    [UCS_CPU_CACHE_L2]  = {.level = 2, .type = X86_CPU_CACHE_TYPE_UNIFIED},
    [UCS_CPU_CACHE_L3]  = {.level = 3, .type = X86_CPU_CACHE_TYPE_UNIFIED}
};

static const ucs_x86_cpu_cache_size_codes_t ucs_x86_cpu_cache_size_codes[] = {
    [0x06] = {.type = UCS_CPU_CACHE_L1i, .size =     8192 },
    [0x08] = {.type = UCS_CPU_CACHE_L1i, .size =    16384 },
    [0x09] = {.type = UCS_CPU_CACHE_L1i, .size =    32768 },
    [0x0a] = {.type = UCS_CPU_CACHE_L1d, .size =     8192 },
    [0x0c] = {.type = UCS_CPU_CACHE_L1d, .size =    16384 },
    [0x0d] = {.type = UCS_CPU_CACHE_L1d, .size =    16384 },
    [0x0e] = {.type = UCS_CPU_CACHE_L1d, .size =    24576 },
    [0x21] = {.type = UCS_CPU_CACHE_L2,  .size =   262144 },
    [0x22] = {.type = UCS_CPU_CACHE_L3,  .size =   524288 },
    [0x23] = {.type = UCS_CPU_CACHE_L3,  .size =  1048576 },
    [0x25] = {.type = UCS_CPU_CACHE_L3,  .size =  2097152 },
    [0x29] = {.type = UCS_CPU_CACHE_L3,  .size =  4194304 },
    [0x2c] = {.type = UCS_CPU_CACHE_L1d, .size =    32768 },
    [0x30] = {.type = UCS_CPU_CACHE_L1i, .size =    32768 },
    [0x39] = {.type = UCS_CPU_CACHE_L2,  .size =   131072 },
    [0x3a] = {.type = UCS_CPU_CACHE_L2,  .size =   196608 },
    [0x3b] = {.type = UCS_CPU_CACHE_L2,  .size =   131072 },
    [0x3c] = {.type = UCS_CPU_CACHE_L2,  .size =   262144 },
    [0x3d] = {.type = UCS_CPU_CACHE_L2,  .size =   393216 },
    [0x3e] = {.type = UCS_CPU_CACHE_L2,  .size =   524288 },
    [0x3f] = {.type = UCS_CPU_CACHE_L2,  .size =   262144 },
    [0x41] = {.type = UCS_CPU_CACHE_L2,  .size =   131072 },
    [0x42] = {.type = UCS_CPU_CACHE_L2,  .size =   262144 },
    [0x43] = {.type = UCS_CPU_CACHE_L2,  .size =   524288 },
    [0x44] = {.type = UCS_CPU_CACHE_L2,  .size =  1048576 },
    [0x45] = {.type = UCS_CPU_CACHE_L2,  .size =  2097152 },
    [0x46] = {.type = UCS_CPU_CACHE_L3,  .size =  4194304 },
    [0x47] = {.type = UCS_CPU_CACHE_L3,  .size =  8388608 },
    [0x48] = {.type = UCS_CPU_CACHE_L2,  .size =  3145728 },
    [0x49] = {.type = UCS_CPU_CACHE_L2,  .size =  4194304 },
    [0x4a] = {.type = UCS_CPU_CACHE_L3,  .size =  6291456 },
    [0x4b] = {.type = UCS_CPU_CACHE_L3,  .size =  8388608 },
    [0x4c] = {.type = UCS_CPU_CACHE_L3,  .size = 12582912 },
    [0x4d] = {.type = UCS_CPU_CACHE_L3,  .size = 16777216 },
    [0x4e] = {.type = UCS_CPU_CACHE_L2,  .size =  6291456 },
    [0x60] = {.type = UCS_CPU_CACHE_L1d, .size =    16384 },
    [0x66] = {.type = UCS_CPU_CACHE_L1d, .size =     8192 },
    [0x67] = {.type = UCS_CPU_CACHE_L1d, .size =    16384 },
    [0x68] = {.type = UCS_CPU_CACHE_L1d, .size =    32768 },
    [0x78] = {.type = UCS_CPU_CACHE_L2,  .size =  1048576 },
    [0x79] = {.type = UCS_CPU_CACHE_L2,  .size =   131072 },
    [0x7a] = {.type = UCS_CPU_CACHE_L2,  .size =   262144 },
    [0x7b] = {.type = UCS_CPU_CACHE_L2,  .size =   524288 },
    [0x7c] = {.type = UCS_CPU_CACHE_L2,  .size =  1048576 },
    [0x7d] = {.type = UCS_CPU_CACHE_L2,  .size =  2097152 },
    [0x7f] = {.type = UCS_CPU_CACHE_L2,  .size =   524288 },
    [0x80] = {.type = UCS_CPU_CACHE_L2,  .size =   524288 },
    [0x82] = {.type = UCS_CPU_CACHE_L2,  .size =   262144 },
    [0x83] = {.type = UCS_CPU_CACHE_L2,  .size =   524288 },
    [0x84] = {.type = UCS_CPU_CACHE_L2,  .size =  1048576 },
    [0x85] = {.type = UCS_CPU_CACHE_L2,  .size =  2097152 },
    [0x86] = {.type = UCS_CPU_CACHE_L2,  .size =   524288 },
    [0x87] = {.type = UCS_CPU_CACHE_L2,  .size =  1048576 },
    [0xd0] = {.type = UCS_CPU_CACHE_L3,  .size =   524288 },
    [0xd1] = {.type = UCS_CPU_CACHE_L3,  .size =  1048576 },
    [0xd2] = {.type = UCS_CPU_CACHE_L3,  .size =  2097152 },
    [0xd6] = {.type = UCS_CPU_CACHE_L3,  .size =  1048576 },
    [0xd7] = {.type = UCS_CPU_CACHE_L3,  .size =  2097152 },
    [0xd8] = {.type = UCS_CPU_CACHE_L3,  .size =  4194304 },
    [0xdc] = {.type = UCS_CPU_CACHE_L3,  .size =  2097152 },
    [0xdd] = {.type = UCS_CPU_CACHE_L3,  .size =  4194304 },
    [0xde] = {.type = UCS_CPU_CACHE_L3,  .size =  8388608 },
    [0xe2] = {.type = UCS_CPU_CACHE_L3,  .size =  2097152 },
    [0xe3] = {.type = UCS_CPU_CACHE_L3,  .size =  4194304 },
    [0xe4] = {.type = UCS_CPU_CACHE_L3,  .size =  8388608 },
    [0xea] = {.type = UCS_CPU_CACHE_L3,  .size = 12582912 },
    [0xeb] = {.type = UCS_CPU_CACHE_L3,  .size = 18874368 },
    [0xec] = {.type = UCS_CPU_CACHE_L3,  .size = 25165824 }
};


static UCS_F_NOOPTIMIZE inline void ucs_x86_cpuid(uint32_t level,
                                                  uint32_t *a, uint32_t *b,
                                                  uint32_t *c, uint32_t *d)
{
    asm volatile ("cpuid\n\t"
                  : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
                  : "0"(level));
}

static UCS_F_NOOPTIMIZE inline void ucs_x86_cpuid_ecx(uint32_t level, uint32_t ecx,
                                                      uint32_t *a, uint32_t *b,
                                                      uint32_t *c, uint32_t *d)
{
    asm volatile("cpuid"
                 : "=a"(*a), "=b"(*b), "=c"(*c), "=d"(*d)
                 : "0"(level), "2"(ecx));
}

/* This allows the CPU detection to work with assemblers not supporting
 * the xgetbv mnemonic.  These include clang and some BSD versions.
 */
#define ucs_x86_xgetbv(_index, _eax, _edx) \
    asm volatile (".byte 0x0f, 0x01, 0xd0" : "=a"(_eax), "=d"(_edx) : "c" (_index))

static int ucs_x86_invariant_tsc()
{
    uint32_t _eax, _ebx, _ecx, _edx;

    ucs_x86_cpuid(X86_CPUID_GET_MAX_VALUE, &_eax, &_ebx, &_ecx, &_edx);
    if (_eax <= X86_CPUID_INVARIANT_TSC) {
        goto warn;
    }

    ucs_x86_cpuid(X86_CPUID_INVARIANT_TSC, &_eax, &_ebx, &_ecx, &_edx);
    if (!(_edx & UCS_BIT(8))) {
        goto warn;
    }

    return 1;

warn:
    ucs_debug("CPU does not support invariant TSC, using fallback timer");

    return 0;
}

double ucs_x86_tsc_freq_from_cpu_model()
{
    char buf[256];
    char model[256];
    char *rate;
    char newline[2];
    double ghz, max_ghz;
    FILE* f;
    int rc;
    int warn;

    f = fopen("/proc/cpuinfo","r");
    if (!f) {
        return -1;
    }

    warn = 0;
    max_ghz = 0.0;
    while (fgets(buf, sizeof(buf), f)) {
        rc = sscanf(buf, "model name : %s", model);
        if (rc != 1) {
            continue;
        }

        rate = strrchr(buf, '@');
        if (rate == NULL) {
            continue;
        }

        rc = sscanf(rate, "@ %lfGHz%[\n]", &ghz, newline);
        if (rc != 2) {
            continue;
        }

        max_ghz = ucs_max(max_ghz, ghz);
        if (max_ghz != ghz) {
            warn = 1;
            break;
        }
    }
    fclose(f);

    if (warn) {
        ucs_debug("Conflicting CPU frequencies detected, using fallback timer");
        return -1;
    }

    return max_ghz * 1e9;
}

double ucs_x86_init_tsc_freq()
{
    double result;

    if (!ucs_x86_invariant_tsc()) {
        goto err_disable_rdtsc;
    }

    ucs_arch_x86_enable_rdtsc = UCS_YES;

    result = ucs_x86_tsc_freq_from_cpu_model();
    if (result <= 0.0) {
        result = ucs_get_cpuinfo_clock_freq("cpu MHz", 1e6);
    }

    if (result > 0.0) {
        return result;
    }

err_disable_rdtsc:
    ucs_arch_x86_enable_rdtsc = UCS_NO;
    return -1;
}

double ucs_arch_get_clocks_per_sec()
{
    double freq;

    /* Init rdtsc state ucs_arch_x86_enable_rdtsc */
    freq = ucs_x86_init_tsc_freq();
    if (ucs_arch_x86_enable_rdtsc == UCS_YES) {
        /* using rdtsc */
        return freq;
    }

    return ucs_arch_generic_get_clocks_per_sec();
}

ucs_cpu_model_t ucs_arch_get_cpu_model()
{
    ucs_x86_cpu_version_t version;
    uint32_t _ebx, _ecx, _edx;
    uint32_t model, family;

    /* Get CPU model/family */
    ucs_x86_cpuid(X86_CPUID_GET_MODEL, ucs_unaligned_ptr(&version.reg), &_ebx, &_ecx, &_edx);

    model  = version.model;
    family = version.family;

    /* Adjust family/model */
    if (family == 0xf) {
        family += version.ext_family;
    }
    if ((family == 0x6) || (family == 0x7) || (family == 0xf) || (family == 0x17)) {
        model = (version.ext_model << 4) | model;
    }

    if (ucs_arch_get_cpu_vendor() == UCS_CPU_VENDOR_ZHAOXIN) {
        if (family == 0x06) {
            switch (model) {
            case 0x0f:
                return UCS_CPU_MODEL_ZHAOXIN_ZHANGJIANG;
            }
        }

        if (family == 0x07) {
            switch (model) {
            case 0x1b:
                return UCS_CPU_MODEL_ZHAOXIN_WUDAOKOU;
            case 0x3b:
                return UCS_CPU_MODEL_ZHAOXIN_LUJIAZUI;
            }
        }
    } else {
        /* Check known CPUs */
        if (family == 0x06) {
            switch (model) {
            case 0x3a:
            case 0x3e:
                return UCS_CPU_MODEL_INTEL_IVYBRIDGE;
            case 0x2a:
            case 0x2d:
                return UCS_CPU_MODEL_INTEL_SANDYBRIDGE;
            case 0x1a:
            case 0x1e:
            case 0x1f:
            case 0x2e:
                return UCS_CPU_MODEL_INTEL_NEHALEM;
            case 0x25:
            case 0x2c:
            case 0x2f:
                return UCS_CPU_MODEL_INTEL_WESTMERE;
            case 0x3c:
            case 0x3f:
            case 0x45:
            case 0x46:
                return UCS_CPU_MODEL_INTEL_HASWELL;
            case 0x3d:
            case 0x47:
            case 0x4f:
            case 0x56:
                return UCS_CPU_MODEL_INTEL_BROADWELL;
            case 0x5e:
            case 0x4e:
            case 0x55:
                return UCS_CPU_MODEL_INTEL_SKYLAKE;
            }
        }

        if (family == 0x17) {
            switch (model) {
            case 0x29:
                return UCS_CPU_MODEL_AMD_NAPLES;
            case 0x31:
                return UCS_CPU_MODEL_AMD_ROME;
            }
        }
    }

    return UCS_CPU_MODEL_UNKNOWN;
}


int ucs_arch_get_cpu_flag()
{
    static int cpu_flag = UCS_CPU_FLAG_UNKNOWN;

    if (UCS_CPU_FLAG_UNKNOWN == cpu_flag) {
        uint32_t result = 0;
        uint32_t base_value;
        uint32_t _eax, _ebx, _ecx, _edx;

        ucs_x86_cpuid(X86_CPUID_GET_BASE_VALUE, &_eax, &_ebx, &_ecx, &_edx);
        base_value = _eax;

        if (base_value >= 1) {
            ucs_x86_cpuid(X86_CPUID_GET_MODEL, &_eax, &_ebx, &_ecx, &_edx);
            if (_edx & (1 << 15)) {
                result |= UCS_CPU_FLAG_CMOV;
            }
            if (_edx & (1 << 23)) {
                result |= UCS_CPU_FLAG_MMX;
            }
            if (_edx & (1 << 25)) {
                result |= UCS_CPU_FLAG_MMX2;
            }
            if (_edx & (1 << 25)) {
                result |= UCS_CPU_FLAG_SSE;
            }
            if (_edx & (1 << 26)) {
                result |= UCS_CPU_FLAG_SSE2;
            }
            if (_ecx & 1) {
                result |= UCS_CPU_FLAG_SSE3;
            }
            if (_ecx & (1 << 9)) {
                result |= UCS_CPU_FLAG_SSSE3;
            }
            if (_ecx & (1 << 19)) {
                result |= UCS_CPU_FLAG_SSE41;
            }
            if (_ecx & (1 << 20)) {
                result |= UCS_CPU_FLAG_SSE42;
            }
            if ((_ecx & 0x18000000) == 0x18000000) {
                ucs_x86_xgetbv(0, _eax, _edx);
                if ((_eax & 0x6) == 0x6) {
                    result |= UCS_CPU_FLAG_AVX;
                }
            }
        }
        if (base_value >= 7) {
            ucs_x86_cpuid(X86_CPUID_GET_EXTD_VALUE, &_eax, &_ebx, &_ecx, &_edx);
            if ((result & UCS_CPU_FLAG_AVX) && (_ebx & (1 << 5))) {
                result |= UCS_CPU_FLAG_AVX2;
            }
        }
        cpu_flag = result;
    }

    return cpu_flag;
}

ucs_cpu_vendor_t ucs_arch_get_cpu_vendor()
{
    ucs_x86_cpu_registers reg;

    ucs_x86_cpuid(X86_CPUID_GET_BASE_VALUE,
                  ucs_unaligned_ptr(&reg.eax), ucs_unaligned_ptr(&reg.ebx),
                  ucs_unaligned_ptr(&reg.ecx), ucs_unaligned_ptr(&reg.edx));
    if (!memcmp(reg.id, X86_CPUID_GENUINEINTEL, sizeof(X86_CPUID_GENUINEINTEL) - 1)) {
        return UCS_CPU_VENDOR_INTEL;
    } else if (!memcmp(reg.id, X86_CPUID_AUTHENTICAMD, sizeof(X86_CPUID_AUTHENTICAMD) - 1)) {
        return UCS_CPU_VENDOR_AMD;
    } else if (!memcmp(reg.id, X86_CPUID_CENTAURHAULS, sizeof(X86_CPUID_CENTAURHAULS) - 1) ||
               !memcmp(reg.id, X86_CPUID_SHANGHAI, sizeof(X86_CPUID_SHANGHAI) - 1)) {
        return UCS_CPU_VENDOR_ZHAOXIN;
    }

    return UCS_CPU_VENDOR_UNKNOWN;
}

#if ENABLE_BUILTIN_MEMCPY
static size_t ucs_cpu_memcpy_thresh(size_t user_val, size_t auto_val)
{
    if (user_val != UCS_MEMUNITS_AUTO) {
        return user_val;
    }

    if (((ucs_arch_get_cpu_vendor() == UCS_CPU_VENDOR_INTEL) &&
         (ucs_arch_get_cpu_model() >= UCS_CPU_MODEL_INTEL_HASWELL)) ||
        (ucs_arch_get_cpu_vendor() == UCS_CPU_VENDOR_AMD) ||
        (ucs_arch_get_cpu_vendor() == UCS_CPU_VENDOR_ZHAOXIN)) {
        return auto_val;
    } else {
        return UCS_MEMUNITS_INF;
    }
}
#endif

void ucs_cpu_init()
{
#if ENABLE_BUILTIN_MEMCPY
    ucs_global_opts.arch.builtin_memcpy_min =
        ucs_cpu_memcpy_thresh(ucs_global_opts.arch.builtin_memcpy_min,
                              ucs_cpu_builtin_memcpy[ucs_arch_get_cpu_vendor()].min);
    ucs_global_opts.arch.builtin_memcpy_max =
        ucs_cpu_memcpy_thresh(ucs_global_opts.arch.builtin_memcpy_max,
                              ucs_cpu_builtin_memcpy[ucs_arch_get_cpu_vendor()].max);
#endif
}

ucs_status_t ucs_arch_get_cache_size(size_t *cache_sizes)
{
    ucs_x86_cache_line_reg_info_t cache_info;
    ucs_x86_cache_line_reg_info_t line_info;
    ucs_x86_cpu_registers reg;
    uint32_t sets;
    uint32_t i, t, r, l4;
    uint32_t max_iter;
    size_t c;
    int level1_only; /* level 1 cache only supported */
    int tag;
    int cache_count;
    ucs_cpu_cache_type_t type;

    /* Get CPU ID and vendor - it will reset cache iteration sequence */
    if (ucs_arch_get_cpu_vendor() != UCS_CPU_VENDOR_INTEL) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_x86_cpuid(X86_CPUID_GET_BASE_VALUE,
                  ucs_unaligned_ptr(&reg.eax), ucs_unaligned_ptr(&reg.ebx),
                  ucs_unaligned_ptr(&reg.ecx), ucs_unaligned_ptr(&reg.edx));
    if (reg.eax < X86_CPUID_GET_CACHE_INFO) {
        return UCS_ERR_UNSUPPORTED;
    }

    level1_only = 0;
    cache_count = 0;

    for (i = 0, max_iter = 1; i < max_iter; i++) {
        ucs_x86_cpuid(X86_CPUID_GET_CACHE_INFO,
                      ucs_unaligned_ptr(&reg.eax), ucs_unaligned_ptr(&reg.ebx),
                      ucs_unaligned_ptr(&reg.ecx), ucs_unaligned_ptr(&reg.edx));

        if (i == 0) { /* on first iteration get max iteration number */
            max_iter     = reg.max_iter;
            reg.max_iter = 0; /* mask iteration register from processing */
        }

        for (r = 0; r < ucs_static_array_size(reg.reg); r++) {
            if (ucs_test_all_flags(reg.reg[r].value, X86_CPU_CACHE_RESERVED)) {
                continue;
            }

            for (t = 0; (t < ucs_static_array_size(reg.reg[r].tag)) &&
                        (reg.reg[r].tag[t] != 0);
                 t++) {
                tag = reg.reg[r].tag[t];

                switch(tag) {
                case X86_CPU_CACHE_TAG_L1_ONLY:
                    level1_only = 1;
                    break;
                case X86_CPU_CACHE_TAG_LEAF4:
                    for (l4 = 0; cache_count < UCS_CPU_CACHE_LAST; l4++) {
                        ucs_x86_cpuid_ecx(X86_CPUID_GET_LEAF4_INFO, l4,
                                          ucs_unaligned_ptr(&cache_info.reg),
                                          ucs_unaligned_ptr(&line_info.reg),
                                          &sets, ucs_unaligned_ptr(&reg.edx));

                        if (cache_info.type == 0) {
                            /* we are done - nothing found, go to next register */
                            break;
                        }

                        for (c = 0; c < UCS_CPU_CACHE_LAST; c++) {
                            if ((cache_info.level == x86_cpu_cache[c].level) &&
                                (cache_info.type  == x86_cpu_cache[c].type)) {
                                /* found it */
                                /* cache entry is not updated yet */
                                /* and cache level is 1 or all levels are supported */
                                if (!((cache_sizes[c] == 0) &&
                                      ((x86_cpu_cache[c].level == 1) || !level1_only))) {
                                    break;
                                }

                                cache_sizes[c] = (line_info.associativity + 1) *
                                                 (line_info.partitions + 1)    *
                                                 (line_info.size + 1)          *
                                                 (sets + 1);
                                cache_count++;
                            }
                        }
                    }
                    return cache_count == UCS_CPU_CACHE_LAST ? UCS_OK : UCS_ERR_UNSUPPORTED;
                default:
                    if ((tag >= ucs_static_array_size(ucs_x86_cpu_cache_size_codes)) ||
                        (ucs_x86_cpu_cache_size_codes[tag].size != 0)) {
                        break; /* tag is out of table or in empty entry */
                    }

                    type = ucs_x86_cpu_cache_size_codes[tag].type;
                    if (cache_sizes[type] != 0) { /* cache is filled already */
                        break;
                    }

                    cache_sizes[type] = ucs_x86_cpu_cache_size_codes[tag].size;
                    cache_count++;
                    break; 
                }
            }
        }
    }

    return cache_count == UCS_CPU_CACHE_LAST ? UCS_OK : UCS_ERR_UNSUPPORTED;
}

void ucs_x86_memcpy_sse_movntdqa(void *dst, const void *src, size_t len)
{
#if defined (__SSE4_1__)
    /* Copy unaligned portion of src */
    if ((uintptr_t)src & 15) {
        uintptr_t aligned  = (uintptr_t)src & ~15;
        uintptr_t misalign = (uintptr_t)src & 15;
        uintptr_t copy     = ucs_min(len, 16 - misalign);

        __m128i tmp = _mm_load(aligned);
        memcpy(dst, UCS_PTR_BYTE_OFFSET(&tmp, misalign), copy);

        src = UCS_PTR_BYTE_OFFSET(src, copy);
        dst = UCS_PTR_BYTE_OFFSET(dst, copy);
        len -= copy;
    }

    /* Copy 64 bytes at a time */
    while (len >= 64) {
        __m128i *S = (__m128i *)src;
        __m128i *D = (__m128i *)dst;
        __m128i tmp[4];

        tmp[0] = _mm_load(S + 0);
        tmp[1] = _mm_load(S + 1);
        tmp[2] = _mm_load(S + 2);
        tmp[3] = _mm_load(S + 3);

        _mm_store(D + 0, tmp[0]);
        _mm_store(D + 1, tmp[1]);
        _mm_store(D + 2, tmp[2]);
        _mm_store(D + 3, tmp[3]);

        src = UCS_PTR_BYTE_OFFSET(src, 64);
        dst = UCS_PTR_BYTE_OFFSET(dst, 64);
        len -= 64;
    }

    /* Copy 16 bytes at a time */
    while (len >= 16) {
        _mm_store(dst, _mm_load(src));

        src = UCS_PTR_BYTE_OFFSET(src, 16);
        dst = UCS_PTR_BYTE_OFFSET(dst, 16);
        len -= 16;
    }

    /* Copy any remaining bytes */
    if (len) {
        __m128i tmp = _mm_load(src);
        memcpy(dst, &tmp, len);
    }
#else
    memcpy(dst, src, len);
#endif
}

#endif
