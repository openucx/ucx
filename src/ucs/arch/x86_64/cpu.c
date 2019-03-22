/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#if defined(__x86_64__)

#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>

#define X86_CPUID_GET_MODEL       0x00000001u
#define X86_CPUID_GET_BASE_VALUE  0x00000000u
#define X86_CPUID_GET_EXTD_VALUE  0x00000007u
#define X86_CPUID_GET_MAX_VALUE   0x80000000u
#define X86_CPUID_INVARIANT_TSC   0x80000007u


ucs_ternary_value_t ucs_arch_x86_enable_rdtsc = UCS_TRY;

static UCS_F_NOOPTIMIZE inline void ucs_x86_cpuid(uint32_t level,
                                                uint32_t *a, uint32_t *b,
                                                uint32_t *c, uint32_t *d)
{
  asm volatile ("cpuid\n\t"
                  : "=a" (*a), "=b" (*b), "=c" (*c), "=d" (*d)
                  : "0" (level));
}

/* This allows the CPU detection to work with assemblers not supporting
 * the xgetbv mnemonic.  These include clang and some BSD versions.
 */
#define ucs_x86_xgetbv(_index, _eax, _edx) \
	asm volatile (".byte 0x0f, 0x01, 0xd0" : "=a"(_eax), "=d"(_edx) : "c" (_index))

static int ucs_x86_invariant_tsc(void)
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

double ucs_x86_tsc_freq_from_cpu_model(void);
double ucs_x86_tsc_freq_from_cpu_model(void)
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

double ucs_x86_init_tsc_freq(void)
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

double ucs_arch_get_clocks_per_sec(void)
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

ucs_cpu_model_t ucs_arch_get_cpu_model(void)
{
    uint32_t _eax, _ebx, _ecx, _edx;
    uint32_t model, family;
    uint32_t ext_model, ext_family;

    /* Get CPU model/family */
    ucs_x86_cpuid(X86_CPUID_GET_MODEL, &_eax, &_ebx, &_ecx, &_edx);

    model      = (_eax >> 4)  & UCS_MASK(8  - 4 );
    family     = (_eax >> 8)  & UCS_MASK(12 - 8 );
    ext_model  = (_eax >> 16) & UCS_MASK(20 - 16);
    ext_family = (_eax >> 20) & UCS_MASK(28 - 20);

    /* Adjust family/model */
    if (family == 0xf) {
        family += ext_family;
    }
    if (family == 0x6 || family == 0xf) {
        model = (ext_model << 4) | model;
    }

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

    return UCS_CPU_MODEL_UNKNOWN;
}


int ucs_arch_get_cpu_flag(void)
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

#endif
