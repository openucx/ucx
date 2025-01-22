/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2016-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_ALLOCATOR_H_
#define UCM_ALLOCATOR_H_

#if HAVE_UCM_PTMALLOC286
#include <ucm/ptmalloc286/malloc-2.8.6.h>
#else
#error "No memory allocator is defined"
#endif

#endif /* UCM_ALLOCATOR_H_ */
