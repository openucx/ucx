/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler_def.h>
#include <ucs/time/time.h>
#include <ucs/debug/log.h>


double ucs_get_cpu_clocks_per_sec()
{
    static double clocks_per_sec = -42.0;

    ucs_compiler_fence();

    if (ucs_unlikely(clocks_per_sec <= 0.0)) {
        clocks_per_sec = ucs_arch_get_clocks_per_sec();
        ucs_debug("arch clock frequency: %.2f Hz", clocks_per_sec);
    }
    return clocks_per_sec;
}
