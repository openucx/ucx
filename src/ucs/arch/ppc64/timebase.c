/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * SPDX-License-Identifier: BSD-3-Clause
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

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
