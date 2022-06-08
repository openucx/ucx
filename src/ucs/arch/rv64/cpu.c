/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#if defined(__riscv)

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/arch/cpu.h>

ucs_cpu_vendor_t ucs_arch_get_cpu_vendor()
{
    return UCS_CPU_VENDOR_GENERIC_RV64G;
}

#endif
