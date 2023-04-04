/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_NUMA_H_
#define UCS_NUMA_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <stdint.h>

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


typedef int ucs_numa_distance_t;


typedef uint16_t ucs_numa_node_t;


extern const char *ucs_numa_policy_names[];


void ucs_numa_init();


void ucs_numa_cleanup();


int ucs_numa_node_of_cpu(int cpu);


/**
 * @return The number of CPU cores in the system.
 */
unsigned ucs_numa_num_configured_cpus();


/**
 * @return The number of memory nodes in the system.
 */
unsigned ucs_numa_num_configured_nodes();


/**
 * @param [in]  cpu CPU to query.
 *
 * @return The NUMA node that the cpu belongs to.
 */
ucs_numa_node_t ucs_numa_node_of_cpu_v2(int cpu);


/**
 * @param [in]  dev_path sysfs path of the device.
 *
 * @return The node that the device belongs to.
 */
ucs_numa_node_t ucs_numa_node_of_device(const char *dev_path);


/**
 * Reports the distance between two nodes according to the machine topology.
 * 
 * @param [in]  node1 first NUMA node.
 * @param [in]  node2 second NUMA node.
 *
 * @return NUMA distance between the two NUMA nodes.
 */
ucs_numa_distance_t
ucs_numa_distance(ucs_numa_node_t node1, ucs_numa_node_t node2);

#endif
