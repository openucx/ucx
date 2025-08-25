/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_KERNELS_H_
#define TEST_KERNELS_H_

namespace cuda {
/* -1 if error, 0 if memory is equal */
int memcmp(const void *s1, const void *s2, size_t size);
};

#endif
