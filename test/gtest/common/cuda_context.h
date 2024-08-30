/*
 * Copyright (C) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
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
