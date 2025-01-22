/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

#include "cuda_context.h"
#include "test_helpers.h"


cuda_context::cuda_context()
{
    if (cuInit(0) != CUDA_SUCCESS) {
        UCS_TEST_SKIP_R("can't init cuda device");
    }

    if (cuDeviceGet(&m_device, 0) != CUDA_SUCCESS) {
        UCS_TEST_SKIP_R("can't get cuda device");
    }

    if (cuCtxCreate(&m_context, 0, m_device) != CUDA_SUCCESS) {
        UCS_TEST_SKIP_R("can't create cuda context");
    }
}

cuda_context::~cuda_context()
{
    EXPECT_EQ(CUDA_SUCCESS, cuCtxDestroy(m_context));
}

CUdevice cuda_context::cuda_device() const
{
    return m_device;
}
