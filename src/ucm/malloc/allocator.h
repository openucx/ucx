/**
 * Copyright (C) Mellanox Technologies Ltd. 2016.  ALL RIGHTS RESERVED.
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
