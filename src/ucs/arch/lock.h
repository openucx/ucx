/*
* Copyright (c) 2021 Nvidia Corporation. All Rights Reserved.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_LOCK_H
#define UCS_ARCH_LOCK_H

#include <ucs/sys/compiler_def.h>

#if defined(__x86_64__)
#  include "x86_64/lock.h"
#elif defined(__powerpc64__)
#  include "ppc64/lock.h"
#elif defined(__aarch64__)
#  include "aarch64/lock.h"
#else
#  error "Unsupported architecture"
#endif

#endif /* UCS_ARCH_LOCK_H */
