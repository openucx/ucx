/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_CUDAMEM_H_
#define UCM_CUDAMEM_H_

#include <ucm/api/ucm.h>
#include <cuda_runtime.h>
#include <cuda.h>


/*cuMemFree */
CUresult ucm_override_cuMemFree(CUdeviceptr dptr);
CUresult ucm_orig_cuMemFree(CUdeviceptr dptr);
CUresult ucm_cuMemFree(CUdeviceptr dptr);

/*cuMemFreeHost */
CUresult ucm_override_cuMemFreeHost(void *p);
CUresult ucm_orig_cuMemFreeHost(void *p);
CUresult ucm_cuMemFreeHost(void *p);

/*cuMemAlloc*/
CUresult ucm_override_cuMemAlloc(CUdeviceptr *dptr, size_t size);
CUresult ucm_orig_cuMemAlloc(CUdeviceptr *dptr, size_t size);
CUresult ucm_cuMemAlloc(CUdeviceptr *dptr, size_t size);

/*cuMemAllocManaged*/
CUresult ucm_override_cuMemAllocManaged(CUdeviceptr *dptr, size_t size,
                                        unsigned int flags);
CUresult ucm_orig_cuMemAllocManaged(CUdeviceptr *dptr, size_t size, unsigned int flags);
CUresult ucm_cuMemAllocManaged(CUdeviceptr *dptr, size_t size, unsigned int flags);

/*cuMemAllocPitch*/
CUresult ucm_override_cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                                      size_t WidthInBytes, size_t Height,
                                      unsigned int ElementSizeBytes);
CUresult ucm_orig_cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                                  size_t WidthInBytes, size_t Height,
                                  unsigned int ElementSizeBytes);
CUresult ucm_cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                             size_t WidthInBytes, size_t Height,
                             unsigned int ElementSizeBytes);

/*cuMemHostGetDevicePointer*/
CUresult ucm_override_cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                                unsigned int Flags);
CUresult ucm_orig_cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p,
                                            unsigned int Flags);
CUresult ucm_cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags);

/*cuMemHostUnregister */
CUresult ucm_override_cuMemHostUnregister(void *p);
CUresult ucm_orig_cuMemHostUnregister(void *p);
CUresult ucm_cuMemHostUnregister(void *p);

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

/*cudaMallocManaged*/
cudaError_t ucm_override_cudaMallocManaged(void **devPtr, size_t size,
                                           unsigned int flags);
cudaError_t ucm_orig_cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
cudaError_t ucm_cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);

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
