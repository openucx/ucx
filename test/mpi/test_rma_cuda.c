/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucp/api/ucp.h>
#include <ucs/sys/string.h>

#include <mpi.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#define SIZE_S 8
#define SIZE_M 4 * 1024 * 1024
#define SIZE_L 16 * 1024 * 1024

#define MPI_TAG_WORKER_ADDRESS_LENGTH 777
#define MPI_TAG_WORKER_ADDRESS        778
#define MPI_TAG_ADDRESS               779
#define MPI_TAG_PACKED_RKEY_SIZE      780
#define MPI_TAG_PACKED_RKEY           781
#define MPI_TAG_ACK                   782

#define PRINT_ROOT(fmt, ...) \
    do { \
        int _rank; \
        MPI_Comm_rank(MPI_COMM_WORLD, &_rank); \
        if (_rank == 0) { \
            fprintf(stdout, fmt, ##__VA_ARGS__); \
        } \
    } while (0)

#define LIB_CALL(_err_t, _func, _success_code, _lib_error_string) \
    do { \
        _err_t _err = (_func); \
        if (_err != _success_code) { \
            fprintf(stderr, "test_rma_cuda.c:%-3u %s failed: %d (%s)\n", \
                    __LINE__, #_func, _err, \
                    _lib_error_string(_err)); \
            exit(_err); \
        } \
    } while (0)

#define CUDA_ERROR_STRING(_err) \
    ({ \
        const char *_err_str; \
        do { \
            cuGetErrorString(_err, &_err_str); \
        } while (0); \
        _err_str; \
    })

#define CUDA_CALL(_func) \
    LIB_CALL(CUresult, _func, CUDA_SUCCESS, CUDA_ERROR_STRING)

#define UCX_CALL(_func) \
    LIB_CALL(ucs_status_t, _func, UCS_OK, ucs_status_string)

#define array_size(_array) (sizeof(_array) / sizeof((_array)[0]))

typedef enum {
    OP_TYPE_GET,
    OP_TYPE_PUT
} op_type_t;

typedef struct {
    const char *name;
    op_type_t  type;
} operation_t;

typedef struct {
    void   *ptr;
    size_t size;
    void   *obj;
} alloc_mem_t;

typedef struct {
    const char *name;
    alloc_mem_t (*alloc)(size_t);
    void (*free)(alloc_mem_t*);
} allocator_t;

typedef struct {
    const char     *name;
    const unsigned min_gpus;
    const int      mthread_support;
    int (*func)(int, op_type_t, CUdevice, const allocator_t*, size_t,
                const void*, ucp_context_h, ucp_worker_h, ucp_ep_h);
} test_t;

typedef struct {
    int           rank;
    op_type_t     op_type;
    ucp_context_h ucp_context;
    ucp_worker_h  ucp_worker;
    ucp_ep_h      ucp_ep;
    void          *buffer;
    size_t        size;
} thread_arg_t;

const size_t sizes[] = {SIZE_S, SIZE_M, SIZE_L};

char test_name[32] = "";
char operation_name[32] = "";
char allocator_0_name[32] = "";
char allocator_1_name[32] = "";

static void parse_allocator_params(const char *opt_arg)
{
    const char *delim   = ",";
    char *token         = strtok((char*)opt_arg, delim);

    ucs_strncpy_safe(allocator_0_name, token, sizeof(allocator_1_name));

    token = strtok(NULL, delim);
    if (NULL == token) {
        token = allocator_0_name;
    }

    ucs_strncpy_safe(allocator_1_name, token, sizeof(allocator_1_name));
}

static void parse_opts(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "t:o:a:")) != -1) {
        switch (c) {
            case 't':
                ucs_strncpy_safe(test_name, optarg, sizeof(test_name));
                break;
            case 'o':
                ucs_strncpy_safe(operation_name, optarg, sizeof(operation_name));
                break;
            case 'a':
                parse_allocator_params(optarg);
                break;
            default:
                fprintf(stderr, "invalid option: %c\n", c);
                exit(1);
        }
    }
}

static ucp_context_h create_ucp_context()
{
    ucp_params_t ucp_params;
    ucp_config_t *ucp_config;
    ucp_context_h ucp_context;

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_NAME;
    ucp_params.features   = UCP_FEATURE_RMA | UCP_FEATURE_WAKEUP;
    ucp_params.name       = "test_rma_cuda_ctx";
    UCX_CALL(ucp_config_read(NULL, NULL, &ucp_config));
    UCX_CALL(ucp_init(&ucp_params, ucp_config, &ucp_context));
    return ucp_context;
}

static ucp_worker_h create_ucp_worker(ucp_context_h ucp_context)
{
    ucp_worker_params_t ucp_worker_params;
    ucp_worker_h ucp_worker;

    ucp_worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE |
                                    UCP_WORKER_PARAM_FIELD_NAME;
    ucp_worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    ucp_worker_params.name        = "test_rma_cuda_worker";
    UCX_CALL(ucp_worker_create(ucp_context, &ucp_worker_params, &ucp_worker));
    return ucp_worker;
}

static void send_worker_address(int receiver_rank, ucp_worker_h ucp_worker)
{
    ucp_worker_attr_t ucp_worker_attr;
    
    ucp_worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
    UCX_CALL(ucp_worker_query(ucp_worker, &ucp_worker_attr));

    MPI_Send(&ucp_worker_attr.address_length,
              sizeof(ucp_worker_attr.address_length), MPI_BYTE, receiver_rank,
              MPI_TAG_WORKER_ADDRESS_LENGTH, MPI_COMM_WORLD);

    MPI_Send(ucp_worker_attr.address, ucp_worker_attr.address_length,
             MPI_BYTE, receiver_rank, MPI_TAG_WORKER_ADDRESS, MPI_COMM_WORLD);

    ucp_worker_release_address(ucp_worker, ucp_worker_attr.address);
}

static void* recv_worker_address(int sender_rank)
{
    size_t ucp_worker_address_length;
    void *ucp_worker_address;

    MPI_Recv(&ucp_worker_address_length, sizeof(ucp_worker_address_length),
             MPI_BYTE, sender_rank, MPI_TAG_WORKER_ADDRESS_LENGTH,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    ucp_worker_address = malloc(ucp_worker_address_length);
    MPI_Recv(ucp_worker_address, ucp_worker_address_length, MPI_BYTE,
             sender_rank, MPI_TAG_WORKER_ADDRESS, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    return ucp_worker_address;
}

static ucp_ep_h create_ucp_ep(ucp_worker_h ucp_worker)
{
    int rank;
    void *ucp_worker_address;
    ucp_ep_params_t ucp_ep_params;
    ucp_ep_h ucp_ep;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        send_worker_address(1, ucp_worker);
        ucp_worker_address = recv_worker_address(1);
    } else {
        ucp_worker_address = recv_worker_address(0);
        send_worker_address(0, ucp_worker);
    }

    ucp_ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ucp_ep_params.address    = (ucp_address_t*)ucp_worker_address;

    UCX_CALL(ucp_ep_create(ucp_worker, &ucp_ep_params, &ucp_ep));
    free(ucp_worker_address);
    return ucp_ep;
}

static void *create_gold_data(size_t size)
{
    void *gold_data;
    int i;

    gold_data = malloc(size);
    if (gold_data == NULL) {
        fprintf(stderr,"failed to create gold data\n");
        exit(-1);
    }

    for (i = 0; i < size - 1; ++i) {
        ((char*)gold_data)[i] = 97 + i % 26;
    }

    ((char*)gold_data)[size - 1] = '\0';
    return gold_data;
}

static int is_initiator(int rank, op_type_t op_type)
{
    return (rank == 0 && op_type == OP_TYPE_GET) ||
           (rank == 1 && op_type == OP_TYPE_PUT);
}

static void retain_and_push_primary_context(CUdevice cu_dev)
{
    CUcontext cu_ctx;

    CUDA_CALL(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    CUDA_CALL(cuCtxPushCurrent(cu_ctx));
}

void cuda_memcpy(void *dest, const void *src, size_t n)
{
    CUDA_CALL(cuMemcpy((CUdeviceptr)dest, (CUdeviceptr)src, n));
    CUDA_CALL(cuCtxSynchronize());
}

static void request_wait(ucp_worker_h ucp_worker, void *request)
{
    ucs_status_t ucs_status;

    if (UCS_PTR_IS_PTR(request)) {
        do {
            ucp_worker_progress(ucp_worker);
            ucs_status = ucp_request_check_status(request);
        } while (ucs_status == UCS_INPROGRESS);

        ucp_request_free(request);
    } else {
        ucs_status = UCS_PTR_STATUS(request);
    }

    if (ucs_status != UCS_OK) {
        fprintf(stderr, "failed to wait for request: %s\n",
                ucs_status_string(ucs_status));
    }
}

void initiate_rma(int peer_rank, op_type_t op_type, ucp_worker_h ucp_worker,
                  ucp_ep_h ucp_ep, void *buffer, size_t size)
{
    ucp_request_param_t ucp_request_param = {0};
    int ack                               = 0;
    uint64_t remote_address;
    size_t ucp_packed_rkey_size;
    void *ucp_packed_rkey;
    ucp_rkey_h ucp_rkey;

    MPI_Recv(&remote_address, sizeof(remote_address), MPI_BYTE, peer_rank,
             MPI_TAG_ADDRESS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Recv(&ucp_packed_rkey_size, sizeof(ucp_packed_rkey_size),
             MPI_BYTE, peer_rank, MPI_TAG_PACKED_RKEY_SIZE, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    ucp_packed_rkey = malloc(ucp_packed_rkey_size);
    MPI_Recv(ucp_packed_rkey, ucp_packed_rkey_size, MPI_BYTE, peer_rank,
             MPI_TAG_PACKED_RKEY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    UCX_CALL(ucp_ep_rkey_unpack(ucp_ep, ucp_packed_rkey, &ucp_rkey));
    free(ucp_packed_rkey);

    if (op_type == OP_TYPE_GET) {
        request_wait(ucp_worker,
                     ucp_get_nbx(ucp_ep, buffer, size, remote_address,
                                 ucp_rkey, &ucp_request_param));
    } else {
        request_wait(ucp_worker,
                     ucp_put_nbx(ucp_ep, buffer, size, remote_address,
                                 ucp_rkey, &ucp_request_param));

        request_wait(ucp_worker, ucp_ep_flush_nbx(ucp_ep, &ucp_request_param));
    }

    MPI_Ssend(&ack, 1, MPI_INT, peer_rank, MPI_TAG_ACK, MPI_COMM_WORLD);

    ucp_rkey_destroy(ucp_rkey);
}

static ucp_mem_h create_ucp_mem(void *buffer, size_t size,
                                ucp_context_h ucp_context)
{
    ucp_mem_map_params_t ucp_mem_map_params;
    ucp_mem_h ucp_mem;

    ucp_mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                    UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    ucp_mem_map_params.address    = buffer;
    ucp_mem_map_params.length     = size;
    UCX_CALL(ucp_mem_map(ucp_context, &ucp_mem_map_params, &ucp_mem));
    return ucp_mem;
}

void complete_rma(int peer_rank, op_type_t op_type, ucp_context_h ucp_context,
                  ucp_worker_h ucp_worker, void *buffer, size_t size)
{
    uint64_t local_address = (uint64_t)buffer;
    ucp_memh_buffer_release_params_t ucp_memh_buffer_release_params = {};
    int request_complete = 0;
    ucp_mem_h ucp_mem;
    void *ucp_packed_rkey;
    size_t ucp_packed_rkey_size;
    int ack;
    MPI_Request request;

    MPI_Send(&local_address, sizeof(local_address), MPI_BYTE, peer_rank,
             MPI_TAG_ADDRESS, MPI_COMM_WORLD);

    ucp_mem = create_ucp_mem(buffer, size, ucp_context);
    UCX_CALL(ucp_rkey_pack(ucp_context, ucp_mem, &ucp_packed_rkey,
                           &ucp_packed_rkey_size));

    MPI_Send(&ucp_packed_rkey_size, sizeof(ucp_packed_rkey_size), MPI_BYTE,
             peer_rank, MPI_TAG_PACKED_RKEY_SIZE, MPI_COMM_WORLD);

    MPI_Send(ucp_packed_rkey, ucp_packed_rkey_size, MPI_BYTE, peer_rank,
             MPI_TAG_PACKED_RKEY, MPI_COMM_WORLD);

    ucp_memh_buffer_release(ucp_packed_rkey, &ucp_memh_buffer_release_params);

    MPI_Irecv(&ack, 1, MPI_INT, peer_rank, MPI_TAG_ACK, MPI_COMM_WORLD,
              &request);

    while (!request_complete) {
           ucp_worker_progress(ucp_worker);
           MPI_Test(&request, &request_complete, MPI_STATUS_IGNORE);
    }

    ucp_mem_unmap(ucp_context, ucp_mem);
}

void do_rma(int rank, op_type_t op_type, ucp_context_h ucp_context,
            ucp_worker_h ucp_worker, ucp_ep_h ucp_ep, void *buffer, size_t size)
{
    int peer_rank = (rank + 1) % 2;

    if (is_initiator(rank, op_type)) {
        initiate_rma(peer_rank, op_type, ucp_worker, ucp_ep, buffer, size);
    } else {
        complete_rma(peer_rank, op_type, ucp_context, ucp_worker, buffer, size);
    }
}

static int
check_result(const void *buffer, const void *gold_data, size_t size)
{
    void *h_buffer;
    int ret;

    h_buffer = malloc(size);
    if (h_buffer == NULL) {
        fprintf(stderr, "failed to allocate host memory\n");
        return -1;
    }

    cuda_memcpy(h_buffer, buffer, size);
    ret = memcmp(h_buffer, gold_data, size);
    free(h_buffer);
    return ret;
}

static void pop_and_release_primary_context(CUdevice cu_dev)
{
    CUDA_CALL(cuCtxPopCurrent(NULL));
    CUDA_CALL(cuDevicePrimaryCtxRelease(cu_dev));
}

static int
test_alloc_prim_send_prim(int rank, op_type_t op_type, CUdevice cu_dev,
                          const allocator_t *allocator, size_t size,
                          const void *gold_data, ucp_context_h ucp_context,
                          ucp_worker_h ucp_worker, ucp_ep_h ucp_ep)
{
    alloc_mem_t alloc_mem;
    int ret;

    retain_and_push_primary_context(cu_dev);

    alloc_mem = allocator->alloc(size);
    if (rank == 1) {
        cuda_memcpy((void*)alloc_mem.ptr, gold_data, size);
    }

    do_rma(rank, op_type, ucp_context, ucp_worker, ucp_ep, alloc_mem.ptr, size);

    if (rank == 0) {
        ret = check_result(alloc_mem.ptr, gold_data, size);
    } else {
        ret = 0;
    }

    allocator->free(&alloc_mem);

    pop_and_release_primary_context(cu_dev);

    return ret;
}

static int
test_alloc_prim_send_no(int rank, op_type_t op_type, CUdevice cu_dev,
                        const allocator_t *allocator, size_t size,
                        const void *gold_data, ucp_context_h ucp_context,
                        ucp_worker_h ucp_worker, ucp_ep_h ucp_ep)
{
    alloc_mem_t alloc_mem;
    CUcontext primary_ctx;
    int ret;

    retain_and_push_primary_context(cu_dev);

    alloc_mem = allocator->alloc(size);
    if (rank == 1) {
        cuda_memcpy(alloc_mem.ptr, gold_data, size);
    }

    CUDA_CALL(cuCtxPopCurrent(&primary_ctx));

    do_rma(rank, op_type, ucp_context, ucp_worker, ucp_ep, alloc_mem.ptr, size);

    CUDA_CALL(cuCtxPushCurrent(primary_ctx));

    if (rank == 0) {
        ret = check_result(alloc_mem.ptr, gold_data, size);
    } else {
        ret = 0;
    }

    allocator->free(&alloc_mem);

    pop_and_release_primary_context(cu_dev);

    return ret;
}

static void *do_rma_thread(void *arg)
{
    thread_arg_t *thread_arg = arg;

    do_rma(thread_arg->rank, thread_arg->op_type, thread_arg->ucp_context,
           thread_arg->ucp_worker, thread_arg->ucp_ep, thread_arg->buffer,
           thread_arg->size);

    return NULL;
}

static int
test_alloc_prim_send_thread(int rank, op_type_t op_type, CUdevice cu_dev,
                            const allocator_t *allocator, size_t size,
                            const void *gold_data, ucp_context_h ucp_context,
                            ucp_worker_h ucp_worker, ucp_ep_h ucp_ep)
{
    alloc_mem_t alloc_mem;
    pthread_t thread;
    thread_arg_t thread_arg;
    int ret;

    retain_and_push_primary_context(cu_dev);

    alloc_mem = allocator->alloc(size);
    if (rank == 1) {
        cuda_memcpy(alloc_mem.ptr, gold_data, size);
    }

    thread_arg.rank        = rank;
    thread_arg.op_type     = op_type;
    thread_arg.ucp_context = ucp_context;
    thread_arg.ucp_worker  = ucp_worker;
    thread_arg.ucp_ep      = ucp_ep;
    thread_arg.buffer      = alloc_mem.ptr;
    thread_arg.size        = size;

    pthread_create(&thread, NULL, do_rma_thread, &thread_arg);
    pthread_join(thread, NULL);

    if (rank == 0) {
        ret = check_result(alloc_mem.ptr, gold_data, size);
    } else {
        ret = 0;
    }

    allocator->free(&alloc_mem);

    pop_and_release_primary_context(cu_dev);

    return ret;
}

static int test_alloc_prim_send_others_prim(int rank, op_type_t op_type, CUdevice cu_dev,
                            const allocator_t *allocator, size_t size,
                            const void *gold_data, ucp_context_h ucp_context,
                            ucp_worker_h ucp_worker, ucp_ep_h ucp_ep)
{
    int ret = 0;
    int dev_count, i;
    CUdevice cu_dev_alloc, cu_dev_op;
    alloc_mem_t alloc_mem;

    CUDA_CALL(cuDeviceGetCount(&dev_count));
    for (i = 0; i < dev_count; ++i) {
        CUDA_CALL(cuDeviceGet(&cu_dev_alloc, (i + rank) % dev_count));
        retain_and_push_primary_context(cu_dev_alloc);

        alloc_mem = allocator->alloc(size);
        if (rank == 1) {
            cuda_memcpy(alloc_mem.ptr, gold_data, size);
        }

        CUDA_CALL(cuDeviceGet(&cu_dev_op, (i + 1 + rank) % dev_count));
        retain_and_push_primary_context(cu_dev_op);

        do_rma(rank, op_type, ucp_context, ucp_worker, ucp_ep, alloc_mem.ptr,
               size);

        pop_and_release_primary_context(cu_dev_op);

        if (rank == 0) {
            ret += check_result(alloc_mem.ptr, gold_data, size);
        }

        allocator->free(&alloc_mem);

        pop_and_release_primary_context(cu_dev_alloc);
    }

    return ret;
}

const test_t tests[] = {
    {"alloc_prim_send_prim", 1, 0, test_alloc_prim_send_prim},
    {"alloc_prim_send_no", 1, 0, test_alloc_prim_send_no},
    {"alloc_prim_send_thread", 1, 1, test_alloc_prim_send_thread},
    {"alloc_prim_send_others_prim", 2, 0, test_alloc_prim_send_others_prim}
};

static alloc_mem_t alloc_mem_alloc(size_t size)
{
    alloc_mem_t alloc_mem = {};

    CUDA_CALL(cuMemAlloc((CUdeviceptr*)&alloc_mem.ptr, size));

    return alloc_mem;
}

static void free_mem_alloc(alloc_mem_t *alloc_mem)
{
    CUDA_CALL(cuMemFree((CUdeviceptr)alloc_mem->ptr));
}

static alloc_mem_t alloc_mem_alloc_managed(size_t size)
{
    alloc_mem_t alloc_mem = {};

    CUDA_CALL(cuMemAllocManaged((CUdeviceptr*)&alloc_mem.ptr, size,
                                CU_MEM_ATTACH_GLOBAL));

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

    alloc_mem.ptr  = (void*)ptr;
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
    CUDA_CALL(cuMemUnmap((CUdeviceptr)alloc_mem->ptr, alloc_mem->size));
    CUDA_CALL(cuMemAddressFree((CUdeviceptr)alloc_mem->ptr, alloc_mem->size));
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

    alloc_mem.ptr = (void*)ptr;
    alloc_mem.obj = (void*)mpool;

    return alloc_mem;
}

void free_mempool(alloc_mem_t *alloc_mem)
{
    CUDA_CALL(cuMemFree((CUdeviceptr)alloc_mem->ptr));
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

const operation_t operations[] = {
    {"get", OP_TYPE_GET},
    {"put", OP_TYPE_PUT}
};

static void destroy_ucp_ep(ucp_ep_h ucp_ep)
{
    ucp_request_param_t ucp_request_param;

    ucp_request_param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    ucp_request_param.flags        = UCP_EP_CLOSE_FLAG_FORCE;
    ucp_ep_close_nbx(ucp_ep, &ucp_request_param);
}

int main(int argc, char **argv)
{
#if ENABLE_MT
    int required = MPI_THREAD_MULTIPLE;
#else
    int required = MPI_THREAD_SINGLE;
#endif
    int provided;
    int comm_size, rank;
    int dev_count, dev_idx;
    CUdevice cu_dev;
    CUcontext cu_ctx;
    ucp_context_h ucp_context;
    ucp_worker_h ucp_worker;
    ucp_ep_h ucp_ep;
    void *gold_data;
    int i, j, k, l, m;
    const test_t *test;
    const allocator_t *allocator_0;
    const allocator_t *allocator_1;
    const operation_t *operation;
    size_t size;
    int ret;

    MPI_Init_thread(&argc, &argv, required, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 2) {
        fprintf(stderr, "this program must be run with exactly 2 processes\n");
        goto mpi_finalize;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    CUDA_CALL(cuInit(0));
    CUDA_CALL(cuDeviceGetCount(&dev_count));
    if (dev_count < 1) {
        PRINT_ROOT("This test requires at least 1 GPU\n");
        goto mpi_finalize;
    }

    parse_opts(argc, argv);

    for (dev_idx = dev_count - 1; dev_idx > -1; --dev_idx) {
        CUDA_CALL(cuDeviceGet(&cu_dev, dev_idx));
        CUDA_CALL(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
    }

    if ((rank == 1) && (dev_count > 1)) {
        CUDA_CALL(cuDeviceGet(&cu_dev, 1));
    }

    ucp_context = create_ucp_context();
    ucp_worker  = create_ucp_worker(ucp_context);
    ucp_ep      = create_ucp_ep(ucp_worker);
    gold_data   = create_gold_data(SIZE_L);

    for (i = 0; i < array_size(tests); ++i) {
        test = tests + i;
        if ((test_name[0] != '\0') &&
            strcmp(test_name, test->name) != 0) {
            continue;
        }

        if (dev_count < test->min_gpus) {
            PRINT_ROOT("TEST[%s]: SKIP (min %d GPUs needed)\n", test->name,
                       test->min_gpus);
            continue;
        }

        if (test->mthread_support && (provided != MPI_THREAD_MULTIPLE)) {
            PRINT_ROOT("TEST[%s]: SKIP (multi-thread is not provided)\n",
                       test->name);
            continue;
        }

        for (j = 0; j < array_size(allocators); ++j) {
            allocator_0 = allocators + j;
            if ((allocator_0_name[0] != '\0') &&
                strcmp(allocator_0_name, allocator_0->name) != 0) {
                continue;
            }

            if (!check_allocator(allocator_0, cu_dev)) {
                PRINT_ROOT("TEST[%s:%s]: SKIP (%s not supported)\n",
                           test->name, allocator_0->name, allocator_0->name);
                continue;
            }

            for (k = 0; k < array_size(allocators); ++k) {
                allocator_1 = allocators + k;
                if ((allocator_1_name[0] != '\0') &&
                    strcmp(allocator_1_name, allocator_1->name) != 0) {
                    continue;
                }

                if (!check_allocator(allocator_1, cu_dev)) {
                    PRINT_ROOT("TEST[%s:%s:%s]: SKIP (%s not supported)\n",
                               test->name, allocator_0->name, allocator_1->name,
                               allocator_1->name);
                    continue;
                }

                for (l = 0; l < array_size(operations); ++l) {
                    operation = operations + l;
                    if ((operation_name[0] != '\0') &&
                        strcmp(operation_name, operation->name) != 0) {
                        continue;
                    }

                    for (m = 0; m < array_size(sizes); ++m) {
                        size = sizes[m];

                        ret = test->func(rank, operation->type, cu_dev,
                                         (rank == 0) ? allocator_0 :
                                                       allocator_1,
                                         size, gold_data,
                                         ucp_context, ucp_worker, ucp_ep);

                        if (rank == 0) {
                            MPI_Reduce(MPI_IN_PLACE, &ret, 1, MPI_INT, MPI_SUM,
                                       0, MPI_COMM_WORLD);
                        } else {
                            MPI_Reduce(&ret, NULL, 1, MPI_INT, MPI_SUM, 0,
                                       MPI_COMM_WORLD);
                        }

                        PRINT_ROOT("TEST[%s:%s:%s:%s:%zi] %s\n", test->name,
                                   allocator_0->name, allocator_1->name,
                                   operation->name, size,
                                   ret ? "FAIL" : "PASS");
                    }
                }
            }
        }
    }

    free(gold_data);
    destroy_ucp_ep(ucp_ep);
    ucp_worker_destroy(ucp_worker);
    ucp_cleanup(ucp_context);
    for (dev_idx = 0; dev_idx < dev_count; ++dev_idx) {
        cuDeviceGet(&cu_dev, dev_idx);
        cuDevicePrimaryCtxRelease(cu_dev);
    }
mpi_finalize:
    MPI_Finalize();
    return 0;
}
