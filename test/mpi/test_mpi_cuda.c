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

#define MPI_COMM_SIZE 2
#define MIN_DEV_COUNT 1

#define SIZE_S 8
#define SIZE_M (4 * 1024 * 1024)
#define SIZE_L (16 * 1024 * 1024)

#define TEST_NAME_FMT      "TEST[%d:%-17s:%-17s:*       ]"
#define TEST_NAME_SIZE_FMT "TEST[%d:%-17s:%-17s:%-8zi]"

#define PRINT_ROOT(fmt, ...) \
    do { \
        int _rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank); \
        if (_rank == 0) { \
            fprintf(stdout, fmt, ##__VA_ARGS__); \
        } \
    } while (0)

#define CUDA_CHECK(_func) \
    do { \
        CUresult __err = (_func); \
        if (__err != CUDA_SUCCESS) { \
            const char *__err_str; \
            cuGetErrorString(__err, &__err_str); \
            fprintf(stderr, "%s failed: %d (%s)\n", #_func, __err, __err_str); \
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
    int               rank;
    const allocator_t *allocator_send;
    const allocator_t *allocator_recv;
    size_t            size;
    CUdevice          cu_dev;
} test_params_t;

typedef struct {
    const char     *description;
    const unsigned min_gpus;
    const int      mthread_support;
    int (*func)(const test_params_t*);
} test_t;

typedef struct {
    int         rank;
    CUdeviceptr send_ptr;
    CUdeviceptr recv_ptr;
    size_t      size;
} thread_arg_t;

static alloc_mem_t alloc_mem_alloc(size_t);
static void free_mem_alloc(alloc_mem_t*);
static alloc_mem_t alloc_mem_alloc_managed(size_t);
static alloc_mem_t alloc_vmm(size_t);
static void free_vmm(alloc_mem_t*);
#if HAVE_CUDA_FABRIC
static alloc_mem_t alloc_mempool(size_t);
static void free_mempool(alloc_mem_t*);
static alloc_mem_t alloc_vmm_fabric(size_t);
#endif

static int test_alloc_prim_send_prim(const test_params_t*);
static int test_alloc_prim_send_no(const test_params_t*);
static int test_alloc_prim_send_thread(const test_params_t*);
static int test_alloc_prim_send_user(const test_params_t*);
static int test_alloc_user_send_prim(const test_params_t*);
static int test_alloc_prim_send_other_prim(const test_params_t*);

const size_t sizes[] = {SIZE_S, SIZE_M, SIZE_L};

allocator_t allocators[] = {{"cuMemAlloc", alloc_mem_alloc, free_mem_alloc},
                            {"cuMemAllocManaged", alloc_mem_alloc_managed,
                             free_mem_alloc},
                            {"VMM", alloc_vmm, free_vmm},
#if HAVE_CUDA_FABRIC
                            {"mempool", alloc_mempool, free_mempool},
                            {"VMM_Fabric", alloc_vmm_fabric, free_vmm}
#endif
};

const test_t tests[] = {
    {"MPI pingpong, memory allocation and communication are performed with the "
     "same primary context set",
     MIN_DEV_COUNT, 0, test_alloc_prim_send_prim},
    {"MPI pingpong, memory is allocated with the primary context set, "
     "communication is done with no context set",
     MIN_DEV_COUNT, 0, test_alloc_prim_send_no},
    {"MPI pingpong, memory is allocated with the primary context set, "
     "communication is done from a thread with no context set",
     MIN_DEV_COUNT, 1, test_alloc_prim_send_thread},
    {"MPI pingpong, memory is allocated with the primary context set, "
     "communication is done with the user context set",
     MIN_DEV_COUNT, 0, test_alloc_prim_send_user},
    {"MPI pingpong, memory is allocated with the user context set, "
     "communication is done with the primary context set",
     MIN_DEV_COUNT, 0, test_alloc_user_send_prim},
    {"MPI pingpong, memory is allocated with the primary context of one GPU, "
     "communication is done with the primary contexts of another GPU",
     2, 0, test_alloc_prim_send_other_prim}
};

static unsigned total_ucx_errors_and_warnings = 0;

void *gold_data;

static void create_gold_data(size_t size)
{
    int i;

    gold_data = malloc(size);
    if (gold_data == NULL) {
        fprintf(stderr, "failed to create gold data\n");
        exit(-1);
    }

    for (i = 0; i < size; ++i) {
        ((uint8_t*)gold_data)[i] = rand() % 256;
    }
}

static ucs_log_func_rc_t
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

static void mpi_send_recv(int rank, int sender, CUdeviceptr d_send,
                          CUdeviceptr d_recv, size_t size)
{
    int receiver = (sender + 1) % 2;

    if (rank == sender) {
        MPI_Send((void*)d_send, size, MPI_BYTE, receiver, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv((void*)d_recv, size, MPI_BYTE, sender, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

static void
mpi_pingpong(int rank, CUdeviceptr d_send, CUdeviceptr d_recv, size_t size)
{
    mpi_send_recv(rank, 0, d_send, d_recv, size);
    mpi_send_recv(rank, 1, d_send, d_recv, size);
}

static void *mpi_pingpong_thread(void *arg)
{
    thread_arg_t *thread_arg = arg;

    mpi_pingpong(thread_arg->rank, thread_arg->send_ptr, thread_arg->recv_ptr,
                 thread_arg->size);

    return NULL;
}

static void retain_and_push_primary_context(CUdevice cu_dev)
{
    CUcontext cu_ctx;

    CUDA_CHECK(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    CUDA_CHECK(cuCtxPushCurrent(cu_ctx));
}

static void pop_and_release_primary_context(CUdevice cu_dev)
{
    CUDA_CHECK(cuCtxPopCurrent(NULL));
    CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_dev));
}

static alloc_mem_t alloc_mem_alloc(size_t size)
{
    alloc_mem_t alloc_mem = {};

    CUDA_CHECK(cuMemAlloc(&alloc_mem.ptr, size));

    return alloc_mem;
}

static void free_mem_alloc(alloc_mem_t *alloc_mem)
{
    CUDA_CHECK(cuMemFree(alloc_mem->ptr));
}

static alloc_mem_t alloc_mem_alloc_managed(size_t size)
{
    alloc_mem_t alloc_mem = {};

    CUDA_CHECK(cuMemAllocManaged(&alloc_mem.ptr, size, CU_MEM_ATTACH_GLOBAL));

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

    CUDA_CHECK(cuCtxGetDevice(&cu_dev));

    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = cu_dev;
    if (handle_type != 0) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)handle_type;
    }

    prop.allocFlags.gpuDirectRDMACapable = 1;

    CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop,
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

    CUDA_CHECK(cuMemAddressReserve(&ptr, size, 0, 0, 0));
    CUDA_CHECK(cuMemMap(ptr, size, 0, handle, 0));

    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id   = cu_dev;
    access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_CHECK(cuMemSetAccess(ptr, size, &access_desc, 1));

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
    CUDA_CHECK(cuMemUnmap(alloc_mem->ptr, alloc_mem->size));
    CUDA_CHECK(cuMemAddressFree(alloc_mem->ptr, alloc_mem->size));
    CUDA_CHECK(cuMemRelease((CUmemGenericAllocationHandle)alloc_mem->obj));
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

    CUDA_CHECK(cuCtxGetDevice(&cu_dev));

    pool_props.allocType     = CU_MEM_ALLOCATION_TYPE_PINNED;
    pool_props.location.id   = (int)cu_dev;
    pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pool_props.handleTypes   = CU_MEM_HANDLE_TYPE_FABRIC;
    pool_props.maxSize       = size;
    CUDA_CHECK(cuMemPoolCreate(&mpool, &pool_props));

    map_desc.flags    = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    map_desc.location = pool_props.location;
    CUDA_CHECK(cuMemPoolSetAccess(mpool, &map_desc, 1));
    CUDA_CHECK(cuMemAllocFromPoolAsync(&ptr, size, mpool, NULL));
    CUDA_CHECK(cuStreamSynchronize(NULL));

    alloc_mem.ptr = ptr;
    alloc_mem.obj = (void*)mpool;

    return alloc_mem;
}

void free_mempool(alloc_mem_t *alloc_mem)
{
    CUDA_CHECK(cuMemFree(alloc_mem->ptr));
    CUDA_CHECK(cuMemPoolDestroy((CUmemoryPool)alloc_mem->obj));
}
#endif

static void check_no_current_context()
{
    CUcontext cu_ctx;

    CUDA_CHECK(cuCtxGetCurrent(&cu_ctx));
    if (cu_ctx != NULL) {
        fprintf(stderr, "cu_ctx is not NULL\n");
        exit(-1);
    }
}

void cuda_memcpy(void *dest, const void *src, size_t n)
{
    CUDA_CHECK(cuMemcpy((CUdeviceptr)dest, (CUdeviceptr)src, n));
    CUDA_CHECK(cuCtxSynchronize());
}

static void alloc_mem(const test_params_t *params, alloc_mem_t *alloc_mem_send,
                      alloc_mem_t *alloc_mem_recv)
{
    *alloc_mem_send = params->allocator_send->alloc(params->size);
    *alloc_mem_recv = params->allocator_recv->alloc(params->size);
    cuda_memcpy((void*)alloc_mem_send->ptr, gold_data, params->size);
    CUDA_CHECK(cuMemsetD8(alloc_mem_recv->ptr, 0, params->size));
}

static int check_result(CUdeviceptr dptr, size_t size)
{
    void *h_buffer;
    int ret;

    h_buffer = malloc(size);
    if (h_buffer == NULL) {
        fprintf(stderr, "failed to allocate host memory\n");
        return -1;
    }

    cuda_memcpy(h_buffer, (void*)dptr, size);
    ret = memcmp(h_buffer, gold_data, size);
    free(h_buffer);
    return ret;
}

static int check_and_free_mem(const test_params_t *params,
                              alloc_mem_t *alloc_mem_send,
                              alloc_mem_t *alloc_mem_recv)
{
    int ret;

    ret = check_result(alloc_mem_recv->ptr, params->size);
    params->allocator_send->free(alloc_mem_send);
    params->allocator_recv->free(alloc_mem_recv);
    return ret;
}

static int test_alloc_prim_send_prim(const test_params_t *params)
{
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    int res;

    retain_and_push_primary_context(params->cu_dev);

    alloc_mem(params, &alloc_mem_send, &alloc_mem_recv);

    mpi_pingpong(params->rank, alloc_mem_send.ptr, alloc_mem_recv.ptr,
                 params->size);

    res = check_and_free_mem(params, &alloc_mem_send, &alloc_mem_recv);

    pop_and_release_primary_context(params->cu_dev);

    return res;
}

static int test_alloc_prim_send_no(const test_params_t *params)
{
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    CUcontext primary_ctx;
    int res;

    retain_and_push_primary_context(params->cu_dev);

    alloc_mem(params, &alloc_mem_send, &alloc_mem_recv);

    CUDA_CHECK(cuCtxPopCurrent(&primary_ctx));
    check_no_current_context();

    mpi_pingpong(params->rank, alloc_mem_send.ptr, alloc_mem_recv.ptr,
                 params->size);

    CUDA_CHECK(cuCtxPushCurrent(primary_ctx));

    res = check_and_free_mem(params, &alloc_mem_send, &alloc_mem_recv);

    pop_and_release_primary_context(params->cu_dev);

    return res;
}

static int test_alloc_prim_send_thread(const test_params_t *params)
{
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    pthread_t thread;
    thread_arg_t thread_arg;
    int res;

    retain_and_push_primary_context(params->cu_dev);

    alloc_mem(params, &alloc_mem_send, &alloc_mem_recv);

    thread_arg.rank     = params->rank;
    thread_arg.send_ptr = alloc_mem_send.ptr;
    thread_arg.recv_ptr = alloc_mem_recv.ptr;
    thread_arg.size     = params->size;
    pthread_create(&thread, NULL, mpi_pingpong_thread, &thread_arg);
    pthread_join(thread, NULL);

    res = check_and_free_mem(params, &alloc_mem_send, &alloc_mem_recv);

    pop_and_release_primary_context(params->cu_dev);

    return res;
}

static int test_alloc_prim_send_user(const test_params_t *params)
{
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    CUcontext primary_ctx, user_ctx;
    int res;
#if CUDA_VERSION >= 12050
    CUctxCreateParams ctx_create_params = {};
#endif

    retain_and_push_primary_context(params->cu_dev);

    alloc_mem(params, &alloc_mem_send, &alloc_mem_recv);

    CUDA_CHECK(cuCtxPopCurrent(&primary_ctx));
#if CUDA_VERSION >= 12050
    CUDA_CHECK(cuCtxCreate_v4(&user_ctx, &ctx_create_params, 0, params->cu_dev));
#else
    CUDA_CHECK(cuCtxCreate(&user_ctx, 0, params->cu_dev));
#endif

    mpi_pingpong(params->rank, alloc_mem_send.ptr, alloc_mem_recv.ptr,
                 params->size);

    CUDA_CHECK(cuCtxDestroy(user_ctx));
    CUDA_CHECK(cuCtxPushCurrent(primary_ctx));

    res = check_and_free_mem(params, &alloc_mem_send, &alloc_mem_recv);

    pop_and_release_primary_context(params->cu_dev);

    return res;
}

static int test_alloc_user_send_prim(const test_params_t *params)
{
    CUcontext user_ctx;
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    int res;

#if CUDA_VERSION >= 12050
    CUctxCreateParams ctx_create_params = {};
    CUDA_CHECK(cuCtxCreate_v4(&user_ctx, &ctx_create_params, 0, params->cu_dev));
#else
    CUDA_CHECK(cuCtxCreate(&user_ctx, 0, params->cu_dev));
#endif

    alloc_mem(params, &alloc_mem_send, &alloc_mem_recv);

    retain_and_push_primary_context(params->cu_dev);
    mpi_pingpong(params->rank, alloc_mem_send.ptr, alloc_mem_recv.ptr,
                 params->size);
    pop_and_release_primary_context(params->cu_dev);

    res = check_and_free_mem(params, &alloc_mem_send, &alloc_mem_recv);

    CUDA_CHECK(cuCtxDestroy(user_ctx));

    return res;
}

static int test_alloc_prim_send_other_prim(const test_params_t *params)
{
    int res = 0;
    int dev_count;
    alloc_mem_t alloc_mem_send, alloc_mem_recv;
    CUdevice cu_dev_other;

    CUDA_CHECK(cuDeviceGetCount(&dev_count));

    retain_and_push_primary_context(params->cu_dev);

    alloc_mem(params, &alloc_mem_send, &alloc_mem_recv);

    CUDA_CHECK(cuDeviceGet(&cu_dev_other, (params->rank + 1) % dev_count));
    retain_and_push_primary_context(cu_dev_other);

    mpi_pingpong(params->rank, alloc_mem_send.ptr, alloc_mem_recv.ptr,
                 params->size);

    pop_and_release_primary_context(cu_dev_other);

    res = check_and_free_mem(params, &alloc_mem_send, &alloc_mem_recv);

    pop_and_release_primary_context(params->cu_dev);

    return res;
}

static int check_allocator(const allocator_t *allocator, CUdevice cu_dev)
{
    alloc_mem_t alloc_mem;
    int ret;

    retain_and_push_primary_context(cu_dev);
    alloc_mem = allocator->alloc(1);
    if (alloc_mem.ptr == 0) {
        ret = 0;
        goto pop_and_release;
    }

    allocator->free(&alloc_mem);
    ret = 1;

pop_and_release:
    pop_and_release_primary_context(cu_dev);
    return ret;
}

static void run_test_for_all_sizes(test_params_t *params, int test_idx)
{
    int size_idx;
    int res;

    for (size_idx = 0; size_idx < ucs_static_array_size(sizes); ++size_idx) {
        total_ucx_errors_and_warnings = 0;

        params->size = sizes[size_idx];
        res          = tests[test_idx].func(params);

        res += total_ucx_errors_and_warnings;

        if (params->rank == 0) {
            MPI_Reduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, 0,
                       MPI_COMM_WORLD);
        } else {
            MPI_Reduce(&res, NULL, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        PRINT_ROOT(TEST_NAME_SIZE_FMT ": %s\n", test_idx,
                   params->allocator_send->name, params->allocator_recv->name,
                   params->size, (res ? "FAIL" : "PASS"));
    }
}

int main(int argc, char **argv)
{
#if ENABLE_MT
    int required = MPI_THREAD_MULTIPLE;
#else
    int required = MPI_THREAD_SINGLE;
#endif
    int provided;
    int comm_size;
    int dev_count, dev_idx;
    CUdevice cu_dev;
    CUcontext cu_ctx;
    int test_idx, allocator_send_idx, allocator_recv_idx;
    test_params_t test_params;
    const test_t *test;

    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != MPI_COMM_SIZE) {
        PRINT_ROOT("This test requires %d processes, not %d\n", MPI_COMM_SIZE,
                   comm_size);
        goto out;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &test_params.rank);

    CUDA_CHECK(cuInit(0));
    CUDA_CHECK(cuDeviceGetCount(&dev_count));
    if (dev_count < MIN_DEV_COUNT) {
        PRINT_ROOT("This test requires at least %d GPU\n", MIN_DEV_COUNT);
        goto out;
    }

    for (dev_idx = dev_count - 1; dev_idx > -1; --dev_idx) {
        CUDA_CHECK(cuDeviceGet(&cu_dev, dev_idx));
        CUDA_CHECK(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    }

    if ((test_params.rank == 1) && (dev_count > 1)) {
        CUDA_CHECK(cuDeviceGet(&cu_dev, 1));
    }

    create_gold_data(SIZE_L);
    test_params.cu_dev = cu_dev;

    ucs_log_push_handler(ucx_errors_and_warnings_counter);

    for (test_idx = 0; test_idx < ucs_static_array_size(tests); ++test_idx) {
        test = tests + test_idx;
        PRINT_ROOT("\nTEST[%d]: %s\n", test_idx, test->description);

        if (dev_count < test->min_gpus) {
            PRINT_ROOT(TEST_NAME_FMT ": SKIP (min %d GPUs needed)\n", test_idx,
                       "*", "*", test->min_gpus);
            continue;
        }

        if (test->mthread_support && (provided != MPI_THREAD_MULTIPLE)) {
            PRINT_ROOT(TEST_NAME_FMT ": SKIP (multi-thread is not provided)\n",
                       test_idx, "*", "*");
            continue;
        }

        for (allocator_send_idx = 0;
             allocator_send_idx < ucs_static_array_size(allocators);
             ++allocator_send_idx) {
            test_params.allocator_send = allocators + allocator_send_idx;
            if (!check_allocator(test_params.allocator_send,
                                 test_params.cu_dev)) {
                PRINT_ROOT(TEST_NAME_FMT ": SKIP (not supported)\n", test_idx,
                           test_params.allocator_send->name, "*");
                continue;
            }

            for (allocator_recv_idx = 0;
                 allocator_recv_idx < ucs_static_array_size(allocators);
                 ++allocator_recv_idx) {
                test_params.allocator_recv = allocators + allocator_recv_idx;
                if (!check_allocator(test_params.allocator_recv,
                                     test_params.cu_dev)) {
                    PRINT_ROOT(TEST_NAME_FMT ": SKIP (not supported)\n",
                               test_idx, test_params.allocator_send->name,
                               test_params.allocator_recv->name);
                    continue;
                }

                run_test_for_all_sizes(&test_params, test_idx);
            }
        }
    }

    ucs_log_pop_handler();

    free(gold_data);

    for (dev_idx = 0; dev_idx < dev_count; ++dev_idx) {
        CUDA_CHECK(cuDeviceGet(&cu_dev, dev_idx));
        CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_dev));
    }

out:
    MPI_Finalize();

    return 0;
}
