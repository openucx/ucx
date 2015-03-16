/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucx_info.h"

#include <ucs/sys/sys.h>
#include <ucs/time/time.h>

static const char* cpu_model_names[] = {
    [UCS_CPU_MODEL_UNKNOWN]           = "unknown",
    [UCS_CPU_MODEL_INTEL_IVYBRIDGE]   = "IvyBridge",
    [UCS_CPU_MODEL_INTEL_SANDYBRIDGE] = "SandyBridge",
    [UCS_CPU_MODEL_INTEL_NEHALEM]     = "Nehalem",
    [UCS_CPU_MODEL_INTEL_WESTMERE]    = "Westmere"
};

void print_sys_info()
{
    printf("# CPU model: %s\n", cpu_model_names[ucs_arch_get_cpu_model()]);
    printf("# Timer frequency: %.3f MHz\n", ucs_get_cpu_clocks_per_sec() / 1e6);
}

