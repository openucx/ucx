/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_NUMA_H_
#define UCS_NUMA_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/debug/memtrack.h>

#if HAVE_NUMA
#include <numaif.h>
#include <numa.h>

#if HAVE_STRUCT_BITMASK
#  define numa_nodemask_p(_nm)            ((_nm)->maskp)
#  define numa_nodemask_size(_nm)         ((_nm)->size)
#  define numa_get_thread_node_mask(_nmp) \
        { \
            numa_free_nodemask(*(_nmp)); \
            *(_nmp) = numa_get_run_node_mask(); \
        }
#else
#  define numa_allocate_nodemask()        ucs_malloc(sizeof(nodemask_t), "nodemask")
#  define numa_free_nodemask(_nm)         ucs_free(_nm)
#  define numa_nodemask_p(_nm)            ((_nm)->maskp.n)
#  define numa_nodemask_size(_nm)         ((size_t)NUMA_NUM_NODES)
#  define numa_bitmask_clearall(_nm)      nodemask_zero(&(_nm)->maskp)
#  define numa_bitmask_setbit(_nm, _n)    nodemask_set(&(_nm)->maskp, _n)
#  define numa_get_thread_node_mask(_nmp) \
        { \
            (*(_nmp))->maskp = numa_get_run_node_mask(); \
        }

struct bitmask {
    nodemask_t maskp;
};
#endif /* HAVE_STRUCT_BITMASK */

#endif /* HAVE_NUMA */


#define UCS_NUMA_MIN_DISTANCE    10


typedef enum {
    UCS_NUMA_POLICY_DEFAULT,
    UCS_NUMA_POLICY_BIND,
    UCS_NUMA_POLICY_PREFERRED,
    UCS_NUMA_POLICY_LAST
} ucs_numa_policy_t;


extern const char *ucs_numa_policy_names[];


int ucs_numa_node_of_cpu(int cpu);


#endif
