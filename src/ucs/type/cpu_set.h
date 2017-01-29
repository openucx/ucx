/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TYPE_CPU_SET_H
#define UCS_TYPE_CPU_SET_H

#include <stdint.h>
#include <stddef.h>

/* Type for array elements in 'ucs_cpu_set_t'.  */
typedef unsigned long int ucs_cpu_mask_t;


/* Size definition for CPU sets.  */
#define UCS_CPU_SETSIZE  1024
#define UCS_NCPUBITS     (8 * sizeof(ucs_cpu_mask_t))

#define UCS_CPUELT(_cpu)  ((_cpu) / UCS_NCPUBITS)
#define UCS_CPUMASK(_cpu) ((ucs_cpu_mask_t) 1 << ((_cpu) % UCS_NCPUBITS))


/* Data structure to describe CPU mask.  */
typedef struct {
    ucs_cpu_mask_t ucs_bits[UCS_CPU_SETSIZE / UCS_NCPUBITS];
} ucs_cpu_set_t;


#define UCS_CPU_ZERO(_cpusetp) \
    do { \
        int _i; \
        for ( _i = 0; _i <  (int)(UCS_CPU_SETSIZE / UCS_NCPUBITS); ++_i) { \
            ((_cpusetp)->ucs_bits)[_i] = 0; \
        } \
    } while (0)

#define UCS_CPU_SET(_cpu, _cpusetp) \
    do { \
        size_t _cpu2 = (_cpu); \
        if (_cpu2 < (8 * sizeof (ucs_cpu_set_t))) { \
            (((ucs_cpu_mask_t *)((_cpusetp)->ucs_bits))[UCS_CPUELT(_cpu2)] |= \
                                      UCS_CPUMASK(_cpu2)); \
        } \
    } while (0)

#define UCS_CPU_CLR(_cpu, _cpusetp) \
    do { \
        size_t _cpu2 = (_cpu); \
        if (_cpu2 < (8 * sizeof(ucs_cpu_set_t))) { \
            (((ucs_cpu_mask_t *) ((_cpusetp)->ucs_bits))[UCS_CPUELT(_cpu2)] &= \
                                     ~UCS_CPUMASK(_cpu2)); \
        } \
    } while (0)

static inline int ucs_cpu_is_set(int cpu, const ucs_cpu_set_t *cpusetp)
{
    if (cpu < (int)(8 * sizeof(ucs_cpu_set_t))) {
        const ucs_cpu_mask_t *mask = cpusetp->ucs_bits;
        return ((mask[UCS_CPUELT(cpu)] & UCS_CPUMASK(cpu)) != 0);
    }
    return 0;
}

static inline int ucs_cpu_set_find_lcs(const ucs_cpu_set_t * cpu_mask)
{
    int i;
    for (i = 0; i < UCS_CPU_SETSIZE; ++i) {
        if (ucs_cpu_is_set(i, cpu_mask)) {
            return i;
        }
    }
    return 0;
}

#endif
