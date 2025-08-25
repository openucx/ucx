/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_UCP_DEVICE_LIB_H_
#define TEST_UCP_DEVICE_LIB_H_

/* -1 if error, 0 if memory is equal */
int cuda_memcmp(const void *s1, const void *s2, size_t size);

#endif
