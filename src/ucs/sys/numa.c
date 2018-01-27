/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "numa.h"

const char *ucs_numa_policy_names[] = {
    [UCS_NUMA_POLICY_DEFAULT]   = "default",
    [UCS_NUMA_POLICY_PREFERRED] = "preferred",
    [UCS_NUMA_POLICY_BIND]      = "bind",
    [UCS_NUMA_POLICY_LAST]      = NULL,
};

