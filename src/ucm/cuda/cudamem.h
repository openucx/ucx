/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_CUDAMEM_H_
#define UCM_CUDAMEM_H_

#include <ucm/api/ucm.h>
#include <cuda.h>
#include <cuda_runtime.h>

ucs_status_t ucm_cudamem_install();

cudaError_t ucm_override_cudaFree(void *addr);
cudaError_t ucm_orig_cudaFree(void *address);
cudaError_t ucm_cudaFree(void *address);

#endif
