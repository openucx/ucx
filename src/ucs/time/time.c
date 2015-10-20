/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/time/time.h>
#include <ucs/debug/log.h>


double ucs_get_cpu_clocks_per_sec()
{
    static double clocks_per_sec = 0.0;
    static int initialized = 0;

    if (!initialized) {
        clocks_per_sec = ucs_arch_get_clocks_per_sec();
        ucs_debug("measured arch clock speed: %.2f Hz", clocks_per_sec);
        initialized = 1;
    }
    return clocks_per_sec;
}
