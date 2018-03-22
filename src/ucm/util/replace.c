/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/type/component.h>

#if HAVE_CUDA
#include "ucm/cuda/cudamem.h"
#endif


#define MAP_FAILED ((void*)-1)

pthread_mutex_t ucm_reloc_get_orig_lock = PTHREAD_MUTEX_INITIALIZER;

pthread_t volatile ucm_reloc_get_orig_thread = PTHREAD_T_NULL;

UCM_DEFINE_REPLACE_FUNC(mmap,   void*, MAP_FAILED, void*, size_t, int, int, int, off_t)
UCM_DEFINE_REPLACE_FUNC(munmap, int,   -1,         void*, size_t)
UCM_DEFINE_REPLACE_FUNC(mremap, void*, MAP_FAILED, void*, size_t, size_t, int)
UCM_DEFINE_REPLACE_FUNC(shmat,  void*, MAP_FAILED, int, const void*, int)
UCM_DEFINE_REPLACE_FUNC(shmdt,  int,   -1,         const void*)
UCM_DEFINE_REPLACE_FUNC(sbrk,   void*, MAP_FAILED, intptr_t)

#if ENABLE_SYMBOL_OVERRIDE
UCM_OVERRIDE_FUNC(mmap, void)
UCM_OVERRIDE_FUNC(munmap, void)
UCM_OVERRIDE_FUNC(mremap, void)
UCM_OVERRIDE_FUNC(shmat, void)
UCM_OVERRIDE_FUNC(shmdt, void)
UCM_OVERRIDE_FUNC(sbrk, void)
#endif

#if HAVE_CUDA

UCM_DEFINE_REPLACE_FUNC(cuMemFree, CUresult,-1, CUdeviceptr)
UCM_DEFINE_REPLACE_FUNC(cuMemFreeHost, CUresult, -1, void *)
UCM_DEFINE_REPLACE_FUNC(cuMemAlloc, CUresult, -1, CUdeviceptr *, size_t)
UCM_DEFINE_REPLACE_FUNC(cuMemAllocPitch, CUresult, -1, CUdeviceptr *, size_t *,
                        size_t, size_t, unsigned int)
UCM_DEFINE_REPLACE_FUNC(cuMemHostGetDevicePointer, CUresult, -1, CUdeviceptr *,
                        void *, unsigned int)
UCM_DEFINE_REPLACE_FUNC(cuMemHostUnregister, CUresult, -1, void *)
UCM_DEFINE_REPLACE_FUNC(cudaFree, cudaError_t, -1, void*)
UCM_DEFINE_REPLACE_FUNC(cudaFreeHost, cudaError_t, -1, void*)
UCM_DEFINE_REPLACE_FUNC(cudaMalloc, cudaError_t, -1, void**, size_t)
UCM_DEFINE_REPLACE_FUNC(cudaMallocPitch, cudaError_t, -1, void**, size_t *,
                        size_t, size_t)
UCM_DEFINE_REPLACE_FUNC(cudaHostGetDevicePointer, cudaError_t, -1, void**,
                        void *, unsigned int)
UCM_DEFINE_REPLACE_FUNC(cudaHostUnregister, cudaError_t, -1, void*)

#if ENABLE_SYMBOL_OVERRIDE
UCM_OVERRIDE_FUNC(cuMemFree,                 CUresult)
UCM_OVERRIDE_FUNC(cuMemFreeHost,             CUresult)
UCM_OVERRIDE_FUNC(cuMemAlloc,                CUresult)
UCM_OVERRIDE_FUNC(cuMemAllocPitch,           CUresult)
UCM_OVERRIDE_FUNC(cuMemHostGetDevicePointer, CUresult)
UCM_OVERRIDE_FUNC(cuMemHostUnregister,       CUresult)
UCM_OVERRIDE_FUNC(cudaFree,                  cudaError_t)
UCM_OVERRIDE_FUNC(cudaFreeHost,              cudaError_t)
UCM_OVERRIDE_FUNC(cudaMalloc,                cudaError_t)
UCM_OVERRIDE_FUNC(cudaMallocPitch,           cudaError_t)
UCM_OVERRIDE_FUNC(cudaHostGetDevicePointer,  cudaError_t)
UCM_OVERRIDE_FUNC(cudaHostUnregister,        cudaError_t)
#endif

#endif

void UCS_F_CTOR ucm_reloc_get_orig_lock_initializer()
{
    pthread_mutexattr_t mutattr;
    pthread_mutexattr_init(&mutattr);
    pthread_mutexattr_settype(&mutattr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&ucm_reloc_get_orig_lock, &mutattr);
    pthread_mutexattr_destroy(&mutattr);
}
