/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <errno.h>
#include <unistd.h>
#include <sys/syscall.h>

#include <ucm/event/event.h>
#include <ucm/util/log.h>
#include <ucm/util/reloc.h>
#include <ucm/util/replace.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/type/component.h>
#include <ucm/bistro/bistro.h>

#if HAVE_CUDA
#include "ucm/cuda/cudamem.h"
#endif


#define MAP_FAILED ((void*)-1)

pthread_mutex_t ucm_reloc_get_orig_lock = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
pthread_t volatile ucm_reloc_get_orig_thread = -1;

UCM_DEFINE_REPLACE_FUNC(mmap,    void*, MAP_FAILED, void*, size_t, int, int, int, off_t)
UCM_DEFINE_REPLACE_FUNC(munmap,  int,   -1,         void*, size_t)
UCM_DEFINE_REPLACE_FUNC(mremap,  void*, MAP_FAILED, void*, size_t, size_t, int)
UCM_DEFINE_REPLACE_FUNC(shmat,   void*, MAP_FAILED, int, const void*, int)
UCM_DEFINE_REPLACE_FUNC(shmdt,   int,   -1,         const void*)
UCM_DEFINE_REPLACE_FUNC(sbrk,    void*, MAP_FAILED, intptr_t)
UCM_DEFINE_REPLACE_FUNC(brk,     int,   -1,         void*)
UCM_DEFINE_REPLACE_FUNC(madvise, int,   -1,         void*, size_t, int)

#if HAVE_DECL_SYS_MMAP
UCM_DEFINE_SELECT_FUNC(mmap, void*, MAP_FAILED, SYS_mmap, void*, size_t, int, int, int, off_t)
#else
UCM_DEFINE_DLSYM_FUNC(mmap, void*, MAP_FAILED, void*, size_t, int, int, int, off_t)
#endif

#if HAVE_DECL_SYS_MUNMAP
UCM_DEFINE_SELECT_FUNC(munmap, int, -1, SYS_munmap, void*, size_t)
#else
UCM_DEFINE_DLSYM_FUNC(munmap, int, -1, void*, size_t)
#endif

#if HAVE_DECL_SYS_MREMAP
UCM_DEFINE_SELECT_FUNC(mremap, void*, MAP_FAILED, SYS_mremap, void*, size_t, size_t, int)
#else
UCM_DEFINE_DLSYM_FUNC(mremap, void*, MAP_FAILED, void*, size_t, size_t, int)
#endif

#if HAVE_DECL_SYS_SHMAT
UCM_DEFINE_SELECT_FUNC(shmat, void*, MAP_FAILED, SYS_shmat, int, const void*, int)
#else
UCM_DEFINE_DLSYM_FUNC(shmat, void*, MAP_FAILED, int, const void*, int)
#endif

#if HAVE_DECL_SYS_SHMDT
UCM_DEFINE_SELECT_FUNC(shmdt, int, -1, SYS_shmdt, const void*)
#else
UCM_DEFINE_DLSYM_FUNC(shmdt, int, -1, const void*)
#endif

#ifdef HAVE_DECL_SYS_MADVISE
UCM_DEFINE_SELECT_FUNC(madvise, int, -1, SYS_madvise, void*, size_t, int)
#else
UCM_DEFINE_DLSYM_FUNC(madvise, int, -1, void*, size_t, int)
#endif


#if HAVE_CUDA

UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemFree, CUresult,-1, CUdeviceptr)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemFreeHost, CUresult, -1, void *)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemAlloc, CUresult, -1, CUdeviceptr *, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemAllocManaged, CUresult, -1, CUdeviceptr *,
                             size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemAllocPitch, CUresult, -1, CUdeviceptr *, size_t *,
                             size_t, size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemHostGetDevicePointer, CUresult, -1, CUdeviceptr *,
                             void *, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cuMemHostUnregister, CUresult, -1, void *)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaFree, cudaError_t, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaFreeHost, cudaError_t, -1, void*)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaMalloc, cudaError_t, -1, void**, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaMallocManaged, cudaError_t, -1, void**, size_t, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaMallocPitch, cudaError_t, -1, void**, size_t *,
                             size_t, size_t)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaHostGetDevicePointer, cudaError_t, -1, void**,
                             void *, unsigned int)
UCM_DEFINE_REPLACE_DLSYM_FUNC(cudaHostUnregister, cudaError_t, -1, void*)

#if ENABLE_SYMBOL_OVERRIDE
UCM_OVERRIDE_FUNC(cuMemFree,                 CUresult)
UCM_OVERRIDE_FUNC(cuMemFreeHost,             CUresult)
UCM_OVERRIDE_FUNC(cuMemAlloc,                CUresult)
UCM_OVERRIDE_FUNC(cuMemAllocManaged,         CUresult)
UCM_OVERRIDE_FUNC(cuMemAllocPitch,           CUresult)
UCM_OVERRIDE_FUNC(cuMemHostGetDevicePointer, CUresult)
UCM_OVERRIDE_FUNC(cuMemHostUnregister,       CUresult)
UCM_OVERRIDE_FUNC(cudaFree,                  cudaError_t)
UCM_OVERRIDE_FUNC(cudaFreeHost,              cudaError_t)
UCM_OVERRIDE_FUNC(cudaMalloc,                cudaError_t)
UCM_OVERRIDE_FUNC(cudaMallocManaged,         cudaError_t)
UCM_OVERRIDE_FUNC(cudaMallocPitch,           cudaError_t)
UCM_OVERRIDE_FUNC(cudaHostGetDevicePointer,  cudaError_t)
UCM_OVERRIDE_FUNC(cudaHostUnregister,        cudaError_t)
#endif

#endif

#if HAVE___CURBRK
extern void *__curbrk;
#endif

#if HAVE_DECL_SYS_BRK
_UCM_DEFINE_DLSYM_FUNC(brk, ucm_orig_dlsym_brk, ucm_override_brk, int, -1, void*)

void *ucm_brk_syscall(void *addr)
{
    return
#if HAVE___CURBRK
        __curbrk =
#endif
        (void*)syscall(SYS_brk, addr);
}

int ucm_orig_brk(void *addr)
{
    void *new_addr;

    new_addr = ucm_brk_syscall(addr);

    if (new_addr < addr) {
        errno = ENOMEM;
        return -1;
    } else {
        return 0;
    }
}

_UCM_DEFINE_DLSYM_FUNC(sbrk, ucm_orig_dlsym_sbrk, ucm_override_sbrk,
                       void*, MAP_FAILED, intptr_t)

void *ucm_orig_sbrk(intptr_t increment)
{
    void *prev;

    if (ucm_global_opts.mmap_hook_mode == UCM_MMAP_HOOK_RELOC) {
        return ucm_orig_dlsym_sbrk(increment);
    } else {
        prev = ucm_brk_syscall(0);
        return ucm_orig_brk(prev + increment) ? (void*)-1 : prev;
    }
}

#else

void *ucm_brk_syscall(void *addr)
{
    return (void*)-1;
}

static int ucm_override_brk(void *addr)
{
    return -1;
}

UCM_DEFINE_DLSYM_FUNC(brk, int, -1, void*)
UCM_DEFINE_DLSYM_FUNC(sbrk, void*, MAP_FAILED, intptr_t)

#endif
