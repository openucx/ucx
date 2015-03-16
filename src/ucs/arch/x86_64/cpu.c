/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#if defined(__x86_64__)

#include <ucs/arch/arch.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>

#define X86_CPUID_GET_MODEL       0x00000001u
#define X86_CPUID_GET_MAX_VALUE   0x80000000u
#define X86_CPUID_INVARIANT_TSC   0x80000007u


static inline void ucs_x86_cpuid(uint32_t level, uint32_t *a, uint32_t *b,
                                 uint32_t *c, uint32_t *d)
{
  asm volatile ("cpuid\n\t"
                  : "=a" (*a), "=b" (*b), "=c" (*c), "=d" (*d)
                  : "0" (level));
}

static void ucs_x86_check_invariant_tsc()
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

    return;
warn:
    ucs_warn("CPU does not support invariant TSC, time may be unstable");
}

static double ucs_x86_tsc_freq_from_cpu_model()
{
    char buf[256];
    char model[256];
    char *rate;
    char newline;
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

        rc = sscanf(rate, "@ %lfGHz%[\n]", &ghz, &newline);
        if (rc != 2) {
            continue;
        }

        max_ghz = ucs_max(ghz, ghz);
        if (max_ghz != ghz) {
            warn = 1;
        }
    }
    fclose(f);

    if (warn) {
        ucs_warn("Conflicting CPU frequencies detected, using: %.2f MHz",
                 max_ghz * 1e3);
    }
    return max_ghz * 1e9;
}

double ucs_arch_get_clocks_per_sec()
{
    double result;

    ucs_x86_check_invariant_tsc();

    /* First, try to find the information in the model name string (will work on Intel) */
    result = ucs_x86_tsc_freq_from_cpu_model();
    if (result > 0) {
        return result;
    }

    /* Read clock speed from cpuinfo */
    return ucs_get_cpuinfo_clock_freq("cpu MHz");
}

ucs_cpu_model_t ucs_arch_get_cpu_model()
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
       }
    }

    return UCS_CPU_MODEL_UNKNOWN;
}

#endif
