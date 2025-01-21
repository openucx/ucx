/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

#include <cuda.h>


class cuda_context {
public:
    cuda_context();
    ~cuda_context();

    CUdevice cuda_device() const;

private:
    CUdevice  m_device{CU_DEVICE_INVALID};
    CUcontext m_context{NULL};
};
