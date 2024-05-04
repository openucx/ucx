/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_CUDAMEM_H_
#define UCM_CUDAMEM_H_

#include <cuda_runtime.h>
#include <cuda.h>


CUresult ucm_cuMemAlloc(CUdeviceptr *dptr, size_t size);
CUresult ucm_cuMemAlloc_v2(CUdeviceptr *dptr, size_t size);
CUresult ucm_cuMemAllocManaged(CUdeviceptr *dptr, size_t size, unsigned int flags);
CUresult ucm_cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                             size_t WidthInBytes, size_t Height,
                             unsigned int ElementSizeBytes);
CUresult ucm_cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                                size_t WidthInBytes, size_t Height,
                                unsigned int ElementSizeBytes);
CUresult ucm_cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                      CUmemGenericAllocationHandle handle,
                      unsigned long long flags);
#if CUDA_VERSION >= 11020
CUresult ucm_cuMemAllocAsync(CUdeviceptr *dptr, size_t size, CUstream hStream);
CUresult ucm_cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t size,
                                     CUmemoryPool pool, CUstream hStream);
#endif
CUresult ucm_cuMemFree(CUdeviceptr dptr);
CUresult ucm_cuMemFree_v2(CUdeviceptr dptr);
CUresult ucm_cuMemFreeHost(void *p);
CUresult ucm_cuMemFreeHost_v2(void *p);
CUresult ucm_cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult ucm_cuMemFreeAsync(CUdeviceptr dptr, CUstream hStream);

cudaError_t ucm_cudaFree(void *devPtr);
cudaError_t ucm_cudaFreeHost(void *ptr);
cudaError_t ucm_cudaFreeAsync(void *devPtr, cudaStream_t hStream);
cudaError_t ucm_cudaMalloc(void **devPtr, size_t size);
cudaError_t ucm_cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
cudaError_t ucm_cudaMallocPitch(void **devPtr, size_t *pitch,
                                size_t width, size_t height);
#if CUDA_VERSION >= 11020
cudaError_t ucm_cudaMallocAsync(void **devPtr, size_t size,
                                cudaStream_t hStream);
cudaError_t ucm_cudaMallocFromPoolAsync(void **devPtr, size_t size,
                                        cudaMemPool_t memPool,
                                        cudaStream_t hStream);
#endif

#endif
