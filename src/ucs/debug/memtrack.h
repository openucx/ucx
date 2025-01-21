/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
 * SPDX-License-Identifier: BSD-3-Clause
* See file LICENSE for terms.
*/

#ifndef UCS_MEMTRACK_H_
#define UCS_MEMTRACK_H_

#include <ucs/sys/compiler_def.h>
#include <stddef.h>


BEGIN_C_DECLS

/** @file memtrack.h */


/**
 * Track custom allocation. Need to be called after custom allocation returns.
 */
void ucs_memtrack_allocated(void *ptr, size_t size, const char *name);


/**
 * Track release of custom allocation. Need to be called before actually
 * releasing the memory.
 */
void ucs_memtrack_releasing(void *ptr);

END_C_DECLS

#endif /* UCS_MEMTRACK_H_ */
