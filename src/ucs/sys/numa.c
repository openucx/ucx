/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "numa.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <stdint.h>
#include <sched.h>


const char *ucs_numa_policy_names[] = {
    [UCS_NUMA_POLICY_DEFAULT]   = "default",
    [UCS_NUMA_POLICY_PREFERRED] = "preferred",
    [UCS_NUMA_POLICY_BIND]      = "bind",
    [UCS_NUMA_POLICY_LAST]      = NULL,
};

#if HAVE_NUMA


static void ucs_numa_populate_cpumap(int16_t cpu_numa_nodes[])
{
    struct bitmask *cpumask;
    int numa_node, cpu;
    int ret;

    cpumask = numa_allocate_cpumask();

    for (numa_node = 0; numa_node <= numa_max_node(); ++numa_node) {
        if (!numa_bitmask_isbitset(numa_all_nodes_ptr, numa_node)) {
            continue;
        }

        ret = numa_node_to_cpus(numa_node, cpumask);
        if (ret == -1) {
            ucs_warn("failed to get CPUs for NUMA node %d: %m", numa_node);
            continue;
        }

        for (cpu = 0; cpu < numa_num_configured_cpus(); ++cpu) {
            if (numa_bitmask_isbitset(cpumask, cpu)) {
                cpu_numa_nodes[cpu] = numa_node + 1;
            }
        }
    }

    numa_free_cpumask(cpumask);
}


int ucs_numa_node_of_cpu(int cpu)
{
    /* we can initialize statically only to the value 0, so the NUMA node
     * numbers will be stored as 1..N instead of 0..N-1 */
    static int16_t cpu_numa_nodes[__CPU_SETSIZE] = {0};

    UCS_STATIC_ASSERT(NUMA_NUM_NODES <= INT16_MAX);
    ucs_assert(cpu < __CPU_SETSIZE);

    if (cpu_numa_nodes[cpu] == 0) {
        ucs_numa_populate_cpumap(cpu_numa_nodes);
    }
    return cpu_numa_nodes[cpu] - 1;
}

#endif
