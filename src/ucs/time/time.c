/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/time/time.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/math.h>


#define UCS_SECONDS_PER_DAY  (60.0 * 60.0 * 24.0)


double UCS_F_CTOR ucs_get_cpu_clocks_per_sec()
{
    static double clocks_per_sec = 0.0;
    static int initialized = 0;

    if (!initialized) {
        clocks_per_sec = ucs_arch_get_clocks_per_sec();
        initialized = 1;
    }
    return clocks_per_sec;
}

/*
 * Calculate how much we should shift the 64-bit time value, so that it will fit
 * a 32-bit integer, and cover an interval of at least one day.
 */
unsigned UCS_F_CTOR ucs_get_short_time_shift()
{
    static unsigned time_shift = 0;
    static int initialized = 0;
    double ts;

    if (!initialized) {
        ts = ucs_log2(UCS_SECONDS_PER_DAY * ucs_get_cpu_clocks_per_sec()) - 31.0;
        ucs_assert(ts >= 0.0);
        ucs_assert(ts < 32.0);
        time_shift = (int)ts;
        initialized = 1;
    }

    return time_shift;
}
