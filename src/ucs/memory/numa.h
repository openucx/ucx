/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_NUMA_H_
#define UCS_NUMA_H_

#include <ucs/sys/compiler_def.h>
#include <stdint.h>

BEGIN_C_DECLS

#define UCS_NUMA_NODE_DEFAULT    0
#define UCS_NUMA_NODE_UNDEFINED -1

typedef int ucs_numa_distance_t;


typedef int16_t ucs_numa_node_t;


extern const char *ucs_numa_policy_names[];


void ucs_numa_init();


void ucs_numa_cleanup();


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
ucs_numa_node_t ucs_numa_node_of_cpu(int cpu);


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

END_C_DECLS

#endif
