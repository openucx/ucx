/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "event.h"

#include <ucm/api/ucm.h>
#include <ucm/mmap/mmap.h>
#include <ucm/malloc/malloc_hook.h>
#if HAVE_CUDA
#include <ucm/cuda/cudamem.h>
#endif
#include <ucm/util/log.h>
#include <ucm/util/sys.h>
#include <ucs/arch/cpu.h>
#include <ucs/datastruct/khash.h>
#include <ucs/type/component.h>

#include <sys/mman.h>
#include <pthread.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>


static pthread_spinlock_t ucm_kh_lock;
#define ucm_ptr_hash(_ptr)  kh_int64_hash_func((uintptr_t)(_ptr))
KHASH_INIT(ucm_ptr_size, const void*, size_t, 1, ucm_ptr_hash, kh_int64_hash_equal)

static pthread_rwlock_t ucm_event_lock = PTHREAD_RWLOCK_INITIALIZER;
static ucs_list_link_t ucm_event_handlers;
static int ucm_external_events = 0;
static khash_t(ucm_ptr_size) ucm_shmat_ptrs;

static size_t ucm_shm_size(int shmid)
{
    struct shmid_ds ds;
    int ret;

    ret = shmctl(shmid, IPC_STAT, &ds);
    if (ret < 0) {
        return 0;
    }

    return ds.shm_segsz;
}

static void ucm_event_call_orig(ucm_event_type_t event_type, ucm_event_t *event,
                                void *arg)
{
    switch (event_type) {
    case UCM_EVENT_MMAP:
        if (event->mmap.result == MAP_FAILED) {
            event->mmap.result = ucm_orig_mmap(event->mmap.address,
                                               event->mmap.size,
                                               event->mmap.prot,
                                               event->mmap.flags,
                                               event->mmap.fd,
                                               event->mmap.offset);
        }
        break;
    case UCM_EVENT_MUNMAP:
        if (event->munmap.result == -1) {
            event->munmap.result = ucm_orig_munmap(event->munmap.address,
                                                   event->munmap.size);
        }
        break;
    case UCM_EVENT_MREMAP:
        if (event->mremap.result == MAP_FAILED) {
            event->mremap.result = ucm_orig_mremap(event->mremap.address,
                                                   event->mremap.old_size,
                                                   event->mremap.new_size,
                                                   event->mremap.flags);
        }
        break;
    case UCM_EVENT_SHMAT:
        if (event->shmat.result == MAP_FAILED) {
            event->shmat.result = ucm_orig_shmat(event->shmat.shmid,
                                                 event->shmat.shmaddr,
                                                 event->shmat.shmflg);
        }
        break;
    case UCM_EVENT_SHMDT:
        if (event->shmdt.result == -1) {
            event->shmdt.result = ucm_orig_shmdt(event->shmdt.shmaddr);
        }
        break;
    case UCM_EVENT_SBRK:
        if (event->sbrk.result == MAP_FAILED) {
            event->sbrk.result = ucm_orig_sbrk(event->sbrk.increment);
        }
        break;
    case UCM_EVENT_MADVISE:
        if (event->madvise.result == -1) {
            event->madvise.result = ucm_orig_madvise(event->madvise.addr,
                                                     event->madvise.length,
                                                     event->madvise.advice);
        }
        break;
    default:
        ucm_warn("Got unknown event %d", event_type);
        break;
    }
}

/*
 * Add a handler which calls the original implementation, and declare the callback
 * list so that initially it will be the single element on that list.
 */
static ucm_event_handler_t ucm_event_orig_handler = {
    .list     = UCS_LIST_INITIALIZER(&ucm_event_handlers, &ucm_event_handlers),
    .events   = UCM_EVENT_MMAP | UCM_EVENT_MUNMAP | UCM_EVENT_MREMAP |
                UCM_EVENT_SHMAT | UCM_EVENT_SHMDT | UCM_EVENT_SBRK |
                UCM_EVENT_MADVISE,      /* All events */
    .priority = 0,                      /* Between negative and positive handlers */
    .cb       = ucm_event_call_orig
};
static ucs_list_link_t ucm_event_handlers =
                UCS_LIST_INITIALIZER(&ucm_event_orig_handler.list,
                                     &ucm_event_orig_handler.list);


static void ucm_event_dispatch(ucm_event_type_t event_type, ucm_event_t *event)
{
    ucm_event_handler_t *handler;

    ucs_list_for_each(handler, &ucm_event_handlers, list) {
        if (handler->events & event_type) {
            handler->cb(event_type, event, handler->arg);
        }
    }
}

#define ucm_event_lock(_lock_func) \
    { \
        int ret; \
        do { \
            ret = _lock_func(&ucm_event_lock); \
        } while (ret == EAGAIN); \
        if (ret != 0) { \
            ucm_fatal("%s() failed: %s", #_lock_func, strerror(ret)); \
        } \
    }

static void ucm_event_enter()
{
    ucm_event_lock(pthread_rwlock_rdlock);
}

static void ucm_event_enter_exclusive()
{
    ucm_event_lock(pthread_rwlock_wrlock);
}

static void ucm_event_leave()
{
    pthread_rwlock_unlock(&ucm_event_lock);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_vm_mmap(void *addr, size_t length)
{
    ucm_event_t event;

    ucm_trace("vm_map addr=%p length=%zu", addr, length);

    event.vm_mapped.address = addr;
    event.vm_mapped.size    = length;
    ucm_event_dispatch(UCM_EVENT_VM_MAPPED, &event);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_vm_munmap(void *addr, size_t length)
{
    ucm_event_t event;

    ucm_trace("vm_unmap addr=%p length=%zu", addr, length);

    event.vm_unmapped.address = addr;
    event.vm_unmapped.size    = length;
    ucm_event_dispatch(UCM_EVENT_VM_UNMAPPED, &event);
}

void *ucm_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
{
    ucm_event_t event;

    ucm_trace("ucm_mmap(addr=%p length=%lu prot=0x%x flags=0x%x fd=%d offset=%ld)",
              addr, length, prot, flags, fd, offset);

    ucm_event_enter();

    if ((flags & MAP_FIXED) && (addr != NULL)) {
        ucm_dispatch_vm_munmap(addr, length);
    }

    event.mmap.result  = MAP_FAILED;
    event.mmap.address = addr;
    event.mmap.size    = length;
    event.mmap.prot    = prot;
    event.mmap.flags   = flags;
    event.mmap.fd      = fd;
    event.mmap.offset  = offset;
    ucm_event_dispatch(UCM_EVENT_MMAP, &event);

    if (event.mmap.result != MAP_FAILED) {
        /* Use original length */
        ucm_dispatch_vm_mmap(event.mmap.result, length);
    }

    ucm_event_leave();

    return event.mmap.result;
}

int ucm_munmap(void *addr, size_t length)
{
    ucm_event_t event;

    ucm_event_enter();

    ucm_trace("ucm_munmap(addr=%p length=%lu)", addr, length);

    ucm_dispatch_vm_munmap(addr, length);

    event.munmap.result  = -1;
    event.munmap.address = addr;
    event.munmap.size    = length;
    ucm_event_dispatch(UCM_EVENT_MUNMAP, &event);

    ucm_event_leave();

    return event.munmap.result;
}

void ucm_vm_mmap(void *addr, size_t length)
{
    ucm_event_enter();

    ucm_trace("ucm_vm_mmap(addr=%p length=%lu)", addr, length);
    ucm_dispatch_vm_mmap(addr, length);

    ucm_event_leave();
}

void ucm_vm_munmap(void *addr, size_t length)
{
    ucm_event_enter();

    ucm_trace("ucm_vm_munmap(addr=%p length=%lu)", addr, length);
    ucm_dispatch_vm_munmap(addr, length);

    ucm_event_leave();
}

void *ucm_mremap(void *old_address, size_t old_size, size_t new_size, int flags)
{
    ucm_event_t event;

    ucm_event_enter();

    ucm_trace("ucm_mremap(old_address=%p old_size=%lu new_size=%ld flags=0x%x)",
              old_address, old_size, new_size, flags);

    ucm_dispatch_vm_munmap(old_address, old_size);

    event.mremap.result   = MAP_FAILED;
    event.mremap.address  = old_address;
    event.mremap.old_size = old_size;
    event.mremap.new_size = new_size;
    event.mremap.flags    = flags;
    ucm_event_dispatch(UCM_EVENT_MREMAP, &event);

    if (event.mremap.result != MAP_FAILED) {
        /* Use original new_size */
        ucm_dispatch_vm_mmap(event.mremap.result, new_size);
    }

    ucm_event_leave();

    return event.mremap.result;
}

void *ucm_shmat(int shmid, const void *shmaddr, int shmflg)
{
    uintptr_t attach_addr;
    ucm_event_t event;
    khiter_t iter;
    size_t size;
    int result;

    ucm_event_enter();

    ucm_trace("ucm_shmat(shmid=%d shmaddr=%p shmflg=0x%x)",
              shmid, shmaddr, shmflg);

    size = ucm_shm_size(shmid);

    if ((shmflg & SHM_REMAP) && (shmaddr != NULL)) {
        attach_addr = (uintptr_t)shmaddr;
        if (shmflg & SHM_RND) {
            attach_addr -= attach_addr % SHMLBA;
        }
        ucm_dispatch_vm_munmap((void*)shmaddr, size);
    }

    event.shmat.result  = MAP_FAILED;
    event.shmat.shmid   = shmid;
    event.shmat.shmaddr = shmaddr;
    event.shmat.shmflg  = shmflg;
    ucm_event_dispatch(UCM_EVENT_SHMAT, &event);

    pthread_spin_lock(&ucm_kh_lock);
    if (event.shmat.result != MAP_FAILED) {
        iter = kh_put(ucm_ptr_size, &ucm_shmat_ptrs, event.mmap.result, &result);
        if (result != -1) {
            kh_value(&ucm_shmat_ptrs, iter) = size;
        }
        pthread_spin_unlock(&ucm_kh_lock);
        ucm_dispatch_vm_mmap(event.shmat.result, size);
    } else {
        pthread_spin_unlock(&ucm_kh_lock);
    }

    ucm_event_leave();

    return event.shmat.result;
}

int ucm_shmdt(const void *shmaddr)
{
    ucm_event_t event;
    khiter_t iter;
    size_t size;

    ucm_event_enter();

    ucm_debug("ucm_shmdt(shmaddr=%p)", shmaddr);

    pthread_spin_lock(&ucm_kh_lock);
    iter = kh_get(ucm_ptr_size, &ucm_shmat_ptrs, shmaddr);
    if (iter != kh_end(&ucm_shmat_ptrs)) {
        size = kh_value(&ucm_shmat_ptrs, iter);
        kh_del(ucm_ptr_size, &ucm_shmat_ptrs, iter);
    } else {
        size = ucm_get_shm_seg_size(shmaddr);
    }
    pthread_spin_unlock(&ucm_kh_lock);

    ucm_dispatch_vm_munmap((void*)shmaddr, size);

    event.shmdt.result  = -1;
    event.shmdt.shmaddr = shmaddr;
    ucm_event_dispatch(UCM_EVENT_SHMDT, &event);

    ucm_event_leave();

    return event.shmdt.result;
}

void *ucm_sbrk(intptr_t increment)
{
    ucm_event_t event;

    ucm_event_enter();

    ucm_trace("ucm_sbrk(increment=%+ld)", increment);

    if (increment < 0) {
        ucm_dispatch_vm_munmap(ucm_orig_sbrk(0) + increment, -increment);
    }

    event.sbrk.result    = MAP_FAILED;
    event.sbrk.increment = increment;
    ucm_event_dispatch(UCM_EVENT_SBRK, &event);

    if ((increment > 0) && (event.sbrk.result != MAP_FAILED)) {
        ucm_dispatch_vm_mmap(ucm_orig_sbrk(0) - increment, increment);
    }

    ucm_event_leave();

    return event.sbrk.result;
}

int ucm_brk(void *addr)
{
    void *old_addr     = ucm_orig_sbrk(0);
    intptr_t increment = (intptr_t)addr - (intptr_t)old_addr;
    ucm_event_t event;

    ucm_event_enter();

    ucm_trace("ucm_brk(addr=%p)", addr);

    if (increment < 0) {
        ucm_dispatch_vm_munmap(old_addr, -increment);
    }

    event.sbrk.result    = (void*)-1;
    event.sbrk.increment = increment;
    ucm_event_dispatch(UCM_EVENT_SBRK, &event);

    if ((increment > 0) && (event.sbrk.result != MAP_FAILED)) {
        ucm_dispatch_vm_mmap(old_addr, increment);
    }

    ucm_event_leave();

    return event.sbrk.result == MAP_FAILED ? -1 : 0;
}

int ucm_madvise(void *addr, size_t length, int advice)
{
    ucm_event_t event;

    ucm_event_enter();

    ucm_trace("ucm_madvise(addr=%p length=%zu advice=%d)", addr, length, advice);

    /* madvise(MADV_DONTNEED) and madvise(MADV_FREE) are releasing pages */
    if ((advice == MADV_DONTNEED)
#if HAVE_DECL_MADV_REMOVE
        || (advice == MADV_REMOVE)
#endif
#if HAVE_DECL_POSIX_MADV_DONTNEED
        || (advice == POSIX_MADV_DONTNEED)
#endif
#if HAVE_DECL_MADV_FREE
        || (advice == MADV_FREE)
#endif
       ) {
        ucm_dispatch_vm_munmap(addr, length);
    }

    event.madvise.result = -1;
    event.madvise.addr   = addr;
    event.madvise.length = length;
    event.madvise.advice = advice;
    ucm_event_dispatch(UCM_EVENT_MADVISE, &event);

    ucm_event_leave();

    return event.madvise.result;
}


#if HAVE_CUDA
static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_alloc(void *addr, size_t length, ucm_mem_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_ALLOC, &event);
}

static UCS_F_ALWAYS_INLINE void
ucm_dispatch_mem_type_free(void *addr, size_t length, ucm_mem_type_t mem_type)
{
    ucm_event_t event;

    event.mem_type.address  = addr;
    event.mem_type.size     = length;
    event.mem_type.mem_type = mem_type;
    ucm_event_dispatch(UCM_EVENT_MEM_TYPE_FREE, &event);
}

static void ucm_cudafree_dispatch_events(void *dptr)
{
    CUresult ret;
    CUdeviceptr pbase;
    size_t psize;

    if (dptr == NULL) {
        return;
    }

    ret = cuMemGetAddressRange(&pbase, &psize, (CUdeviceptr) dptr);
    if (ret != CUDA_SUCCESS) {
        ucm_warn("cuMemGetAddressRange(devPtr=%p) failed", (void *)dptr);
        psize = 1; /* set minimum length */
    }
    ucs_assert(dptr == (void *)pbase);

    ucm_dispatch_mem_type_free((void *)dptr, psize, UCM_MEM_TYPE_CUDA);
}

CUresult ucm_cuMemFree(CUdeviceptr dptr)
{
    CUresult ret;

    ucm_event_enter();

    ucm_trace("ucm_cuMemFree(dptr=%p)",(void *)dptr);

    ucm_cudafree_dispatch_events((void *)dptr);

    ret = ucm_orig_cuMemFree(dptr);

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemFreeHost(void *p)
{
    CUresult ret;

    ucm_event_enter();

    ucm_trace("ucm_cuMemFreeHost(ptr=%p)", p);

    ucm_dispatch_vm_munmap(p, 0);

    ret = ucm_orig_cuMemFreeHost(p);

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemAlloc(CUdeviceptr *dptr, size_t size)
{
    CUresult ret;

    ucm_event_enter();

    ret = ucm_orig_cuMemAlloc(dptr, size);
    if (ret == CUDA_SUCCESS) {
        ucm_trace("ucm_cuMemAlloc(dptr=%p size:%lu)",(void *)*dptr, size);
        ucm_dispatch_mem_type_alloc((void *)*dptr, size, UCM_MEM_TYPE_CUDA);
    }

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemAllocManaged(CUdeviceptr *dptr, size_t size, unsigned int flags)
{
    CUresult ret;

    ucm_event_enter();

    ret = ucm_orig_cuMemAllocManaged(dptr, size, flags);
    if (ret == CUDA_SUCCESS) {
        ucm_trace("ucm_cuMemAllocManaged(dptr=%p size:%lu, flags:%d)",
                  (void *)*dptr, size, flags);
        ucm_dispatch_mem_type_alloc((void *)*dptr, size, UCM_MEM_TYPE_CUDA_MANAGED);
    }

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                             size_t WidthInBytes, size_t Height,
                             unsigned int ElementSizeBytes)
{
    CUresult ret;

    ucm_event_enter();

    ret = ucm_orig_cuMemAllocPitch(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
    if (ret == CUDA_SUCCESS) {
        ucm_trace("ucm_cuMemAllocPitch(dptr=%p size:%lu)",(void *)*dptr,
                  (WidthInBytes * Height));
        ucm_dispatch_mem_type_alloc((void *)*dptr, WidthInBytes * Height,
                                    UCM_MEM_TYPE_CUDA);
    }

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags)
{
    CUresult ret;

    ucm_event_enter();

    ret = ucm_orig_cuMemHostGetDevicePointer(pdptr, p, Flags);
    if (ret == CUDA_SUCCESS) {
        ucm_trace("ucm_cuMemHostGetDevicePointer(pdptr=%p p=%p)",(void *)*pdptr, p);
    }

    ucm_event_leave();
    return ret;
}

CUresult ucm_cuMemHostUnregister(void *p)
{
    CUresult ret;

    ucm_event_enter();

    ucm_trace("ucm_cuMemHostUnregister(ptr=%p)", p);

    ret = ucm_orig_cuMemHostUnregister(p);

    ucm_event_leave();
    return ret;
}

cudaError_t ucm_cudaFree(void *devPtr)
{
    cudaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_cudaFree(devPtr=%p)", devPtr);

    ucm_cudafree_dispatch_events((void *)devPtr);

    ret = ucm_orig_cudaFree(devPtr);

    ucm_event_leave();

    return ret;
}

cudaError_t ucm_cudaFreeHost(void *ptr)
{
    cudaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_cudaFreeHost(ptr=%p)", ptr);

    ucm_dispatch_vm_munmap(ptr, 0);

    ret = ucm_orig_cudaFreeHost(ptr);

    ucm_event_leave();
    return ret;
}

cudaError_t ucm_cudaMalloc(void **devPtr, size_t size)
{
    cudaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_cudaMalloc(devPtr, size);
    if (ret == cudaSuccess) {
        ucm_trace("ucm_cudaMalloc(devPtr=%p size:%lu)", *devPtr, size);
        ucm_dispatch_mem_type_alloc(*devPtr, size, UCM_MEM_TYPE_CUDA);
    }

    ucm_event_leave();

    return ret;
}

cudaError_t ucm_cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    cudaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_cudaMallocManaged(devPtr, size, flags);
    if (ret == cudaSuccess) {
        ucm_trace("ucm_cudaMallocManaged(devPtr=%p size:%lu flags:%d)",
                  *devPtr, size, flags);
        ucm_dispatch_mem_type_alloc(*devPtr, size, UCM_MEM_TYPE_CUDA_MANAGED);
    }

    ucm_event_leave();

    return ret;
}

cudaError_t ucm_cudaMallocPitch(void **devPtr, size_t *pitch,
                                size_t width, size_t height)
{
    cudaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_cudaMallocPitch(devPtr, pitch, width, height);
    if (ret == cudaSuccess) {
        ucm_trace("ucm_cudaMallocPitch(devPtr=%p size:%lu)",*devPtr, (width * height));
        ucm_dispatch_mem_type_alloc(*devPtr, (width * height), UCM_MEM_TYPE_CUDA);
    }

    ucm_event_leave();
    return ret;
}

cudaError_t ucm_cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    cudaError_t ret;

    ucm_event_enter();

    ret = ucm_orig_cudaHostGetDevicePointer(pDevice, pHost, flags);
    if (ret == cudaSuccess) {
        ucm_trace("ucm_cuMemHostGetDevicePointer(pDevice=%p pHost=%p)", pDevice, pHost);
    }

    ucm_event_leave();
    return ret;
}

cudaError_t ucm_cudaHostUnregister(void *ptr)
{
    cudaError_t ret;

    ucm_event_enter();

    ucm_trace("ucm_cudaHostUnregister(ptr=%p)", ptr);

    ret = ucm_orig_cudaHostUnregister(ptr);

    ucm_event_leave();
    return ret;
}

#endif

void ucm_event_handler_add(ucm_event_handler_t *handler)
{
    ucm_event_handler_t *elem;

    ucm_event_enter_exclusive();
    ucs_list_for_each(elem, &ucm_event_handlers, list) {
        if (handler->priority < elem->priority) {
            ucs_list_insert_before(&elem->list, &handler->list);
            ucm_event_leave();
            return;
        }
    }

    ucs_list_add_tail(&ucm_event_handlers, &handler->list);
    ucm_event_leave();
}

void ucm_event_handler_remove(ucm_event_handler_t *handler)
{
    ucm_event_enter_exclusive();
    ucs_list_del(&handler->list);
    ucm_event_leave();
}

static ucs_status_t ucm_event_install(int events)
{
    int native_events, malloc_events;
    ucs_status_t status;

    /* Replace aggregate events with the native events which make them */
    native_events = events & ~(UCM_EVENT_VM_MAPPED | UCM_EVENT_VM_UNMAPPED |
                               UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE);
    if (events & UCM_EVENT_VM_MAPPED) {
        native_events |= UCM_EVENT_MMAP | UCM_EVENT_MREMAP |
                         UCM_EVENT_SHMAT | UCM_EVENT_SBRK;
    }
    if (events & UCM_EVENT_VM_UNMAPPED) {
        native_events |= UCM_EVENT_MMAP | UCM_EVENT_MUNMAP | UCM_EVENT_MREMAP |
                         UCM_EVENT_SHMDT | UCM_EVENT_SHMAT |
                         UCM_EVENT_SBRK | UCM_EVENT_MADVISE;
    }

    /* TODO lock */
    status = ucm_mmap_install(native_events);
    if (status != UCS_OK) {
        ucm_debug("failed to install mmap events");
        goto out_unlock;
    }

    ucm_debug("mmap hooks are ready");

    malloc_events = events & ~(UCM_EVENT_MEM_TYPE_ALLOC |
                               UCM_EVENT_MEM_TYPE_FREE);
    status = ucm_malloc_install(malloc_events);
    if (status != UCS_OK) {
        ucm_debug("failed to install malloc events");
        goto out_unlock;
    }

    ucm_debug("malloc hooks are ready");

#if HAVE_CUDA
    if (events & (UCM_EVENT_MEM_TYPE_ALLOC | UCM_EVENT_MEM_TYPE_FREE)) {
        status = ucm_cudamem_install();
        if (status != UCS_OK) {
            ucm_debug("failed to install cudamem events");
            goto out_unlock;
        }
        ucm_debug("cudaFree hooks are ready");
    }
#endif

    status = UCS_OK;

out_unlock:
    return status;

}

ucs_status_t ucm_set_event_handler(int events, int priority,
                                   ucm_event_callback_t cb, void *arg)
{
    ucm_event_handler_t *handler;
    ucs_status_t status;

    if (!ucm_global_opts.enable_events) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (!(events & UCM_EVENT_FLAG_NO_INSTALL) && (events & ~ucm_external_events)) {
        status = ucm_event_install(events & ~ucm_external_events);
        if (status != UCS_OK) {
            return status;
        }
    }

    handler = malloc(sizeof(*handler));
    if (handler == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    handler->events   = events;
    handler->priority = priority;
    handler->cb       = cb;
    handler->arg      = arg;

    ucm_event_handler_add(handler);

    ucm_debug("added user handler (func=%p arg=%p) for events=0x%x prio=%d", cb,
              arg, events, priority);
    return UCS_OK;
}

void ucm_set_external_event(int events)
{
    ucm_event_enter_exclusive();
    ucm_external_events |= events;
    ucm_event_leave();
}

void ucm_unset_external_event(int events)
{
    ucm_event_enter_exclusive();
    ucm_external_events &= ~events;
    ucm_event_leave();
}

void ucm_unset_event_handler(int events, ucm_event_callback_t cb, void *arg)
{
    ucm_event_handler_t *elem, *tmp;
    UCS_LIST_HEAD(gc_list);

    ucm_event_enter_exclusive();
    ucs_list_for_each_safe(elem, tmp, &ucm_event_handlers, list) {
        if ((cb == elem->cb) && (arg == elem->arg)) {
            elem->events &= ~events;
            if (elem->events == 0) {
                ucs_list_del(&elem->list);
                ucs_list_add_tail(&gc_list, &elem->list);
            }
        }
    }
    ucm_event_leave();

    /* Do not release memory while we hold event lock - may deadlock */
    while (!ucs_list_is_empty(&gc_list)) {
        elem = ucs_list_extract_head(&gc_list, ucm_event_handler_t, list);
        free(elem);
    }
}

UCS_STATIC_INIT {
    pthread_spin_init(&ucm_kh_lock, PTHREAD_PROCESS_PRIVATE);
    kh_init_inplace(ucm_ptr_size, &ucm_shmat_ptrs);
}

UCS_STATIC_CLEANUP {
    kh_destroy_inplace(ucm_ptr_size, &ucm_shmat_ptrs);
    pthread_spin_destroy(&ucm_kh_lock);
}
