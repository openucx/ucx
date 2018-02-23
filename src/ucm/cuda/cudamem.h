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

/*cudaFree*/
cudaError_t ucm_override_cudaFree(void *devPtr);
cudaError_t ucm_orig_cudaFree(void *devPtr);
cudaError_t ucm_cudaFree(void *devPtr);

/*cudaFreeHost*/
cudaError_t ucm_override_cudaFreeHost(void *ptr);
cudaError_t ucm_orig_cudaFreeHost(void *ptr);
cudaError_t ucm_cudaFreeHost(void *ptr);

/*cudaMalloc*/
cudaError_t ucm_override_cudaMalloc(void **devPtr, size_t size);
cudaError_t ucm_orig_cudaMalloc(void **devPtr, size_t size);
cudaError_t ucm_cudaMalloc(void **devPtr, size_t size);

/*cudaMallocPitch*/
cudaError_t ucm_override_cudaMallocPitch(void **devPtr, size_t *pitch,
                                         size_t width, size_t height);
cudaError_t ucm_orig_cudaMallocPitch(void **devPtr, size_t *pitch,
                                     size_t width, size_t height);
cudaError_t ucm_cudaMallocPitch(void **devPtr, size_t *pitch,
                                size_t width, size_t height);

/*cudaHostGetDevicePointer*/
cudaError_t ucm_override_cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                                  unsigned int flags);
cudaError_t ucm_orig_cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                              unsigned int flags);
cudaError_t ucm_cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);


/*cudaHostUnregister*/
cudaError_t ucm_override_cudaHostUnregister(void *ptr);
cudaError_t ucm_orig_cudaHostUnregister(void *ptr);
cudaError_t ucm_cudaHostUnregister(void *ptr);
#endif
