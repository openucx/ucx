/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_UCP_DEVICE_LIB_H_
#define TEST_UCP_DEVICE_LIB_H_

void launch_cuda_memcmp(const void* a, const void* b,
                        int* result, size_t size);

#endif
