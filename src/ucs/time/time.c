/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/time/time.h>
#include <ucs/debug/log.h>


static double clocks_per_sec = -42.0;

void ucs_init_cpu_clocks_per_sec()
{
    clocks_per_sec = ucs_arch_get_clocks_per_sec();
    ucs_debug("arch clock frequency: %.2f Hz", clocks_per_sec);
}

double ucs_get_cpu_clocks_per_sec()
{
    return clocks_per_sec;
}
