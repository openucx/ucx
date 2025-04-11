/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucs/debug/log_def.h>

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda.h>
#include <pthread.h>

#define SIZE_S 8
#define SIZE_M 4 * 1024 * 1024
#define SIZE_L 16 * 1024 * 1024

#define PRINT_ROOT(fmt, ...) \
    do { \
        int _rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank); \
        if (_rank == 0) { \
            fprintf(stdout, fmt, ##__VA_ARGS__); \
        } \
    } while (0)

#define array_size(_array) (sizeof(_array) / sizeof((_array)[0]))

#define CUDA_CALL(_func) \
    do { \
        CUresult __err = (_func); \
        if (__err != CUDA_SUCCESS) { \
            const char *__err_str; \
            cuGetErrorString(__err, &__err_str); \
            fprintf(stderr, "test_mpi_cuda.c:%-3u %s failed: %d (%s)\n", \
                    __LINE__, #_func, __err, __err_str); \
            exit(__err); \
        } \
    } while (0)

typedef struct {
    CUdeviceptr ptr;
    size_t      size;
    void        *obj;
} alloc_mem_t;

typedef struct {
    const char *name;
    alloc_mem_t (*alloc)(size_t);
    void (*free)(alloc_mem_t*);
} allocator_t;

typedef struct {
    const char     *description;
    const unsigned min_gpus;
    const int      mthread_support;
    const int (*func)(const allocator_t*, const allocator_t*, size_t, CUdevice);
} test_t;

typedef struct {
    CUdeviceptr d_send;
    CUdeviceptr d_recv;
    size_t size;
} thread_arg_t;

const size_t sizes[] = {SIZE_S, SIZE_M, SIZE_L};

static unsigned total_ucx_errors_and_warnings = 0;

ucs_log_func_rc_t
ucx_errors_and_warnings_counter(const char *file, unsigned line,
                                const char *function, ucs_log_level_t level,
                                const ucs_log_component_config_t *comp_conf,
                                const char *format, va_list ap)
{
    if ((level == UCS_LOG_LEVEL_ERROR) || (level == UCS_LOG_LEVEL_WARN)) {
        ++total_ucx_errors_and_warnings;
    }

    return UCS_LOG_FUNC_RC_CONTINUE;
}

static int
mpi_send_recv(int sender, CUdeviceptr d_send, CUdeviceptr d_recv, size_t size)
{
    int rank, receiver, res;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    receiver = (sender + 1) % 2;

    if (rank == sender) {
        res = MPI_Send((void*)d_send, size, MPI_BYTE, receiver, 0,
                       MPI_COMM_WORLD);
    } else {
        res = MPI_Recv((void*)d_recv, size, MPI_BYTE, sender, 0, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE);
    }

    return (res == MPI_SUCCESS) ? 0 : -1;
}

static int mpi_pingpong(CUdeviceptr d_send, CUdeviceptr d_recv, size_t size)
{
    int res;

    res  = mpi_send_recv(0, d_send, d_recv, size);
    res += mpi_send_recv(1, d_send, d_recv, size);

    return res;
}

static void *mpi_pingpong_thread(void *arg)
{
    thread_arg_t *thread_arg = arg;
    int *res                 = malloc(sizeof(int));

    *res = mpi_pingpong(thread_arg->d_send, thread_arg->d_recv,
                        thread_arg->size);

    return res;
}

static void retain_and_push_primary_context(CUdevice cu_dev)
{
    CUcontext cu_ctx;

    CUDA_CALL(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    CUDA_CALL(cuCtxPushCurrent(cu_ctx));
}

static void pop_and_release_primary_context(CUdevice cu_dev)
{
    CUDA_CALL(cuCtxPopCurrent(NULL));
    CUDA_CALL(cuDevicePrimaryCtxRelease(cu_dev));
}

static alloc_mem_t alloc_mem_alloc(size_t size)
{
    alloc_mem_t alloc_mem = {};

    CUDA_CALL(cuMemAlloc(&alloc_mem.ptr, size));

    return alloc_mem;
}

static void free_mem_alloc(alloc_mem_t *alloc_mem)
{
    CUDA_CALL(cuMemFree(alloc_mem->ptr));
}

static alloc_mem_t alloc_mem_alloc_managed(size_t size)
{
    alloc_mem_t alloc_mem = {};

    CUDA_CALL(cuMemAllocManaged(&alloc_mem.ptr, size, CU_MEM_ATTACH_GLOBAL));

    return alloc_mem;
}

static alloc_mem_t alloc_vmm_type(size_t size, unsigned handle_type)
{
    CUmemAllocationProp prop    = {};
    size_t granularity          = 0;
    CUmemAccessDesc access_desc = {};
    alloc_mem_t alloc_mem       = {};
    CUdevice cu_dev;
    CUmemGenericAllocationHandle handle;
    CUresult result;
    CUdeviceptr ptr;

    CUDA_CALL(cuCtxGetDevice(&cu_dev));

    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = cu_dev;
    if (handle_type != 0) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)handle_type;
    }

    prop.allocFlags.gpuDirectRDMACapable = 1;

    CUDA_CALL(cuMemGetAllocationGranularity(&granularity, &prop,
                                            CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    if ((size % granularity) != 0) {
        size = ((size / granularity) + 1) * granularity;
    }

    result = cuMemCreate(&handle, size, &prop, 0);
    if (result != CUDA_SUCCESS) {
        /**
         * сuMemCreate fails with an error if cannot create the CUDA memory
         * handle with CU_MEM_HANDLE_TYPE_FABRIC handle type
         */
        alloc_mem.ptr = 0;
        goto out;
    }

    CUDA_CALL(cuMemAddressReserve(&ptr, size, 0, 0, 0));
    CUDA_CALL(cuMemMap(ptr, size, 0, handle, 0));

    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id   = cu_dev;
    access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_CALL(cuMemSetAccess(ptr, size, &access_desc, 1));

    alloc_mem.ptr  = ptr;
    alloc_mem.size = size;
    alloc_mem.obj  = (void*)handle;

out:
    return alloc_mem;
}

static alloc_mem_t alloc_vmm(size_t size)
{
    return alloc_vmm_type(size, 0);
}

static void free_vmm(alloc_mem_t *alloc_mem)
{
    CUDA_CALL(cuMemUnmap(alloc_mem->ptr, alloc_mem->size));
    CUDA_CALL(cuMemAddressFree(alloc_mem->ptr, alloc_mem->size));
    CUDA_CALL(cuMemRelease((CUmemGenericAllocationHandle)alloc_mem->obj));
}

#if HAVE_CUDA_FABRIC
static alloc_mem_t alloc_vmm_fabric(size_t size)
{
    return alloc_vmm_type(size, CU_MEM_HANDLE_TYPE_FABRIC);
}

static alloc_mem_t alloc_mempool(size_t size)
{
    CUmemPoolProps pool_props = {};
    alloc_mem_t alloc_mem     = {};
    CUdevice cu_dev;
    CUmemoryPool mpool;
    CUmemAccessDesc map_desc;
    CUdeviceptr ptr;

    CUDA_CALL(cuCtxGetDevice(&cu_dev));

    pool_props.allocType     = CU_MEM_ALLOCATION_TYPE_PINNED;
    pool_props.location.id   = (int)cu_dev;
    pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pool_props.handleTypes   = CU_MEM_HANDLE_TYPE_FABRIC;
    pool_props.maxSize       = size;
    CUDA_CALL(cuMemPoolCreate(&mpool, &pool_props));

    map_desc.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    map_desc.location = pool_props.location;
    CUDA_CALL(cuMemPoolSetAccess(mpool, &map_desc, 1));
    CUDA_CALL(cuMemAllocFromPoolAsync(&ptr, size, mpool, NULL));
    CUDA_CALL(cuStreamSynchronize(NULL));

    alloc_mem.ptr = ptr;
    alloc_mem.obj = (void*)mpool;

    return alloc_mem;
}

void free_mempool(alloc_mem_t *alloc_mem)
{
    CUDA_CALL(cuMemFree(alloc_mem->ptr));
    CUDA_CALL(cuMemPoolDestroy((CUmemoryPool)alloc_mem->obj));
}
#endif

allocator_t allocators[] = {{"cuMemAlloc", alloc_mem_alloc, free_mem_alloc},
                            {"cuMemAllocManaged", alloc_mem_alloc_managed,
                             free_mem_alloc},
                            {"VMM", alloc_vmm, free_vmm},
#if HAVE_CUDA_FABRIC
                            {"mempool", alloc_mempool, free_mempool},
                            {"VMM_Fabric", alloc_vmm_fabric, free_vmm}
#endif
};

static int test_alloc_prim_send_prim(const allocator_t *allocator_send,
                                     const allocator_t *allocator_recv,
                                     size_t size, CUdevice cu_dev)
{
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    int res;

    retain_and_push_primary_context(cu_dev);

    alloc_mem_send = allocator_send->alloc(size);
    alloc_mem_recv = allocator_recv->alloc(size);

    res = mpi_pingpong(alloc_mem_send.ptr, alloc_mem_recv.ptr, size);

    allocator_send->free(&alloc_mem_send);
    allocator_recv->free(&alloc_mem_recv);

    pop_and_release_primary_context(cu_dev);

    return res;
}

static int test_alloc_prim_send_no(const allocator_t *allocator_send,
                                   const allocator_t *allocator_recv,
                                   size_t size, CUdevice cu_dev)
{
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    CUcontext primary_ctx;
    int res;

    retain_and_push_primary_context(cu_dev);

    alloc_mem_send = allocator_send->alloc(size);
    alloc_mem_recv = allocator_recv->alloc(size);

    CUDA_CALL(cuCtxPopCurrent(&primary_ctx));

    res = mpi_pingpong(alloc_mem_send.ptr, alloc_mem_recv.ptr, size);

    CUDA_CALL(cuCtxPushCurrent(primary_ctx));

    allocator_send->free(&alloc_mem_send);
    allocator_recv->free(&alloc_mem_recv);

    pop_and_release_primary_context(cu_dev);

    return res;
}

static int test_alloc_prim_send_thread(const allocator_t *allocator_send,
                                       const allocator_t *allocator_recv,
                                       size_t size, CUdevice cu_dev)
{
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    pthread_t thread;
    thread_arg_t thread_arg;
    void *thread_retval;
    int res;

    retain_and_push_primary_context(cu_dev);

    alloc_mem_send = allocator_send->alloc(size);
    alloc_mem_recv = allocator_recv->alloc(size);

    thread_arg.d_send = alloc_mem_send.ptr;
    thread_arg.d_recv = alloc_mem_recv.ptr;
    thread_arg.size   = size;
    pthread_create(&thread, NULL, mpi_pingpong_thread, &thread_arg);
    pthread_join(thread, &thread_retval);
    res = *(int*)thread_retval;
    free(thread_retval);

    allocator_send->free(&alloc_mem_send);
    allocator_recv->free(&alloc_mem_recv);

    pop_and_release_primary_context(cu_dev);

    return res;
}

static int test_alloc_prim_send_user(const allocator_t *allocator_send,
                                     const allocator_t *allocator_recv,
                                     size_t size, CUdevice cu_dev)
{
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    CUcontext primary_ctx, user_ctx;
    int res;

    retain_and_push_primary_context(cu_dev);

    alloc_mem_send = allocator_send->alloc(size);
    alloc_mem_recv = allocator_recv->alloc(size);

    CUDA_CALL(cuCtxPopCurrent(&primary_ctx));
    CUDA_CALL(cuCtxCreate(&user_ctx, 0, cu_dev));

    res = mpi_pingpong(alloc_mem_send.ptr, alloc_mem_recv.ptr, size);

    CUDA_CALL(cuCtxDestroy(user_ctx));
    CUDA_CALL(cuCtxPushCurrent(primary_ctx));

    allocator_send->free(&alloc_mem_send);
    allocator_recv->free(&alloc_mem_recv);

    pop_and_release_primary_context(cu_dev);

    return res;
}

static int test_alloc_user_send_prim(const allocator_t *allocator_send,
                                     const allocator_t *allocator_recv,
                                     size_t size, CUdevice cu_dev)
{
    CUcontext user_ctx;
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    int res;

    CUDA_CALL(cuCtxCreate(&user_ctx, 0, cu_dev));

    alloc_mem_send = allocator_send->alloc(size);
    alloc_mem_recv = allocator_recv->alloc(size);

    retain_and_push_primary_context(cu_dev);
    res = mpi_pingpong(alloc_mem_send.ptr, alloc_mem_recv.ptr, size);
    pop_and_release_primary_context(cu_dev);

    allocator_send->free(&alloc_mem_send);
    allocator_recv->free(&alloc_mem_recv);

    CUDA_CALL(cuCtxDestroy(user_ctx));

    return res;
}

static int test_alloc_prim_send_others_prim(const allocator_t *allocator_send,
                                            const allocator_t *allocator_recv,
                                            size_t size, CUdevice cu_dev)
{
    int res = 0, dev_count, dev_idx;
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    CUdevice cu_dev_other;

    retain_and_push_primary_context(cu_dev);

    alloc_mem_send = allocator_send->alloc(size);
    alloc_mem_recv = allocator_recv->alloc(size);

    CUDA_CALL(cuDeviceGetCount(&dev_count));

    /* Send on contexts of all devices */
    for (dev_idx = 1; dev_idx < dev_count; ++dev_idx) {
        CUDA_CALL(cuDeviceGet(&cu_dev_other, dev_idx));
        retain_and_push_primary_context(cu_dev_other);

        res += mpi_pingpong(alloc_mem_send.ptr, alloc_mem_recv.ptr, size);

        pop_and_release_primary_context(cu_dev_other);
    }

    allocator_send->free(&alloc_mem_send);
    allocator_recv->free(&alloc_mem_recv);

    pop_and_release_primary_context(cu_dev);

    return res;
}

const test_t tests[] = {
    {"MPI pingpong, memory allocation and communication are performed with the "
     "same primary context set",
     1, 0, test_alloc_prim_send_prim},
    {"MPI pingpong, memory is allocated with the primary context set, "
     "communication is done with no context set",
     1, 0,test_alloc_prim_send_no},
    {"MPI pingpong, memory is allocated with the primary context set, "
     "communication is done from a thread with no context set",
     1, 1, test_alloc_prim_send_thread},
    {"MPI pingpong, memory is allocated with the primary context set, "
     "communication is done with the user context set",
     1, 0, test_alloc_prim_send_user},
    {"MPI pingpong, memory is allocated with the user context set, "
     "communication is done with the primary context set",
     1, 0,test_alloc_user_send_prim},
    {"MPI pingpong, memory is allocated with the primary context of GPU 0, "
     "communication is done with the primary contexts of all other GPUs",
     2, 0,test_alloc_prim_send_others_prim}
};

static int check_allocator(const allocator_t *allocator, CUdevice cu_dev)
{
    alloc_mem_t alloc_mem;

    retain_and_push_primary_context(cu_dev);
    alloc_mem = allocator->alloc(1);
    if (alloc_mem.ptr == 0) {
        return 0;
    }

    allocator->free(&alloc_mem);
    pop_and_release_primary_context(cu_dev);

    return 1;
}

int main(int argc, char **argv)
{
    int provided;
    int comm_size;
    int rank;
    int dev_count, dev_idx;
    CUdevice cu_dev;
    CUcontext cu_ctx;
    int i, j, k, l;
    const test_t *test;
    const allocator_t *allocator_send, *allocator_recv;
    int res;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 2) {
        PRINT_ROOT("This test requires 2 processes, not %d\n", comm_size);
        goto out;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    CUDA_CALL(cuInit(0));
    CUDA_CALL(cuDeviceGetCount(&dev_count));
    if (dev_count < 1) {
        PRINT_ROOT("This test requires at least 1 GPU\n");
        goto out;
    }

    for (dev_idx = dev_count - 1; dev_idx > -1; --dev_idx) {
        CUDA_CALL(cuDeviceGet(&cu_dev, dev_idx));
        CUDA_CALL(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    }

    if ((rank == 1) && (dev_count > 1)) {
        CUDA_CALL(cuDeviceGet(&cu_dev, 1));
    }

    ucs_log_push_handler(ucx_errors_and_warnings_counter);

    for (i = 0; i < array_size(tests); ++i) {
        test = tests + i;
        PRINT_ROOT("TEST[%d]: %s\n", i, test->description);

        if (dev_count < test->min_gpus) {
            PRINT_ROOT("TEST[%d]: SKIP (min %d GPUs needed)\n", i,
                       test->min_gpus);
            continue;
        }

        if (test->mthread_support && (provided != MPI_THREAD_MULTIPLE)) {
            PRINT_ROOT("TEST[%d]: SKIP (multi-thread is not provided)\n", i);
            continue;
        }

        res                           = 0;
        total_ucx_errors_and_warnings = 0;
        for (j = 0; j < array_size(allocators); ++j) {
            allocator_send = allocators + j;
            if (!check_allocator(allocator_send, cu_dev)) {
                PRINT_ROOT("\tSKIP allocator %s\n", allocator_send->name);
                continue;
            }

            for (k = 0; k < array_size(allocators); ++k) {
                allocator_recv = allocators + k;
                if (!check_allocator(allocator_recv, cu_dev)) {
                    PRINT_ROOT("\tSKIP allocator %s\n", allocator_recv->name);
                    continue;
                }

                for (l = 0; l < array_size(sizes); ++l) {
                    PRINT_ROOT("\tTesting allocators: %s for the send buffer, "
                               "%s for the receive buffer, message size %zi\n",
                               allocator_send->name, allocator_recv->name,
                               sizes[l]);
                    res += test->func(allocator_send, allocator_recv, sizes[l],
                                      cu_dev);
                    PRINT_ROOT("\t------------------------------------\n");
                }
            }
        }

        res += total_ucx_errors_and_warnings;

        if (rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, 0,
                       MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&res, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        PRINT_ROOT("TEST[%d]: %s\n", i, (res ? "FAIL" : "PASS"));
    }

    ucs_log_pop_handler();

    for (dev_idx = 0; dev_idx < dev_count; ++dev_idx) {
        CUDA_CALL(cuDeviceGet(&cu_dev, dev_idx));
        CUDA_CALL(cuDevicePrimaryCtxRelease(cu_dev));
    }

out:
    MPI_Finalize();

    return 0;
}
