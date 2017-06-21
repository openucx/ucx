/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/sys/sys.h>
#ifdef HAVE_SYS_PLATFORM_PPC_H
#  include <sys/platform/ppc.h>
#endif


#if defined(__powerpc64__)

double ucs_arch_get_clocks_per_sec()
{
#if HAVE_DECL___PPC_GET_TIMEBASE_FREQ
    return __ppc_get_timebase_freq();
#else
    return ucs_get_cpuinfo_clock_freq("timebase", 1.0);
#endif
}

#endif
