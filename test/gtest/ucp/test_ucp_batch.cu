/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_ep.inl>
}

#include <cuda.h>

#include <ucp/api/cuda/ucp_dev.cuh>

class test_ucp_batch_base : public ucp_test {
public:

    struct mem_chunk {
        ucp_context_h           context;
        ucp_mem_h               memh;
        std::vector<ucp_rkey_h> rkeys;
        void                    *address;
        size_t                  length;

        uint64_t rva() const {
            return reinterpret_cast<uint64_t>(address);
        }

        mem_chunk(ucp_context_h, void *addr, size_t size);
        ~mem_chunk();
        ucp_rkey_h unpack(ucp_ep_h, ucp_md_map_t md_map = 0);
    };

    static void
    get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        add_variant_with_value(variants, UCP_FEATURE_RMA | UCP_FEATURE_AMO64, 0, "");
    }

    virtual void init() {
        ucs::skip_on_address_sanitizer();
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }
    }
};

test_ucp_batch_base::mem_chunk::mem_chunk(ucp_context_h ctx,
                                    void *addr,
                                    size_t size) : context(ctx),
    address(addr), length(size)
{
    ucp_mem_map_params_t params;
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.flags      = 0;
    params.address    = addr;
    params.length     = size;
    auto status       = ucp_mem_map(ctx, &params, &memh);
    ASSERT_UCS_OK(status);
}

test_ucp_batch_base::mem_chunk::~mem_chunk()
{
    for (auto &rkey : rkeys) {
        ucp_rkey_destroy(rkey);
    }

    EXPECT_UCS_OK(ucp_mem_unmap(context, memh));
}

ucp_rkey_h test_ucp_batch_base::mem_chunk::unpack(ucp_ep_h ep, ucp_md_map_t md_map)
{
    ucp_rkey_h rkey;
    void *rkey_buffer;
    size_t rkey_size;

    ASSERT_UCS_OK(ucp_rkey_pack(context, memh, &rkey_buffer, &rkey_size));
    if (md_map == 0) {
        ASSERT_UCS_OK(ucp_ep_rkey_unpack(ep, rkey_buffer, &rkey));
    } else {
        // Different MD map means different config index on proto v2
        ASSERT_UCS_OK(ucp_ep_rkey_unpack_internal(
                        ep, rkey_buffer, rkey_size, md_map, 0,
                        UCS_SYS_DEVICE_ID_UNKNOWN, &rkey));
    }

    ucp_rkey_buffer_release(rkey_buffer);
    rkeys.push_back(rkey);
    return rkey;
}

__global__ void compare_kernel(const void* a, const void* b,
                               bool* result, size_t size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        if (reinterpret_cast<const uint8_t*>(a)[i]
            != reinterpret_cast<const uint8_t*>(b)[i]) {
            result[0] = false;
        }
    }
}

template <typename T, typename... Args>
std::unique_ptr<T> my_make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

__global__ void execute_ucp_batch_kernel(ucp_batch_h batch, uint64_t flags, uint64_t signal_inc);

class test_ucp_batch : public test_ucp_batch_base {
public:
    using buffers    = std::vector<std::unique_ptr<mem_buffer>>;
    using mem_chunks = std::vector<std::unique_ptr<mem_chunk>>;
    using rma_iov    = std::unique_ptr<ucp_rma_iov_t[]>;

    buffers alloc_chunks(size_t size, int count,
                         ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_CUDA,
                         bool zero = false);

    mem_chunks create_regs(buffers& chunks, entity& e) {
        mem_chunks mc;

        for (auto i = 0; i < chunks.size(); ++i) {
            mc.emplace_back(new mem_chunk(e.ucph(),
                                          chunks[i]->ptr(),
                                          chunks[i]->size()));
        }

        return mc;
    }

    rma_iov create_rma_iov(ucp_ep_h ep,
                           mem_chunks& local_chunks,
                           mem_chunks& remote_chunks) {
        rma_iov iovs(new ucp_rma_iov_t[local_chunks.size()]);

        for (auto i = 0; i < local_chunks.size(); ++i) {
            iovs[i].opcode    = UCP_RMA_PUT;
            iovs[i].local_va  = local_chunks[i]->address;
            iovs[i].remote_va = remote_chunks[i]->rva();
            iovs[i].length    = local_chunks[i]->length;
            iovs[i].rkey      = remote_chunks[i]->unpack(ep);
            iovs[i].memh      = local_chunks[i]->memh;
        }

        return iovs;
    }

    // Allocations
    buffers chunks_tx;
    buffers chunks_rx;
    buffers signal;
    buffers host;

    // Registrations of allocations
    mem_chunks local_reg;
    mem_chunks remote_reg;
    mem_chunks signal_reg;
    mem_chunks host_reg;

    ucp_ep_prepare_batch_param_t prepare_param;

    // New added UCP API
    rma_iov   rma_list;

    ucp_rkey_h signal_rkey;

    size_t chunk_size  = 4096;
    size_t signal_size = 1024;
    size_t chunk_count = 32;
    int    init_val    = 1;

    static bool cuda_is_same(const void *a, const void *b, size_t size)
    {
        bool*d_result;
        bool h_result = true;

        cudaMalloc(&d_result, sizeof(*d_result));
        cudaMemcpy(d_result, &h_result, sizeof(h_result), cudaMemcpyHostToDevice);
        compare_kernel<<<16, 64>>>(a, b, d_result, size);
        cudaDeviceSynchronize();
        cudaMemcpy(&h_result, d_result, sizeof(h_result), cudaMemcpyDeviceToHost);
        cudaFree(d_result);
        return h_result;
    }

    static uint64_t cuda_value_get(const void* value) {
        uint64_t tmp;
        // TODO - replace with async op so we can safely poll value from cuda memory
        cudaMemcpy(&tmp, value, sizeof(tmp), cudaMemcpyDeviceToHost);
        return tmp;
    }

    void wait_for_signal(uint64_t expected) {
        wait_for_cond(
                [&]() {
                    return cuda_value_get(signal_reg[0]->address) == expected;
                },
                []() {usleep(100);}
            );
        ASSERT_EQ(expected, cuda_value_get(signal_reg[0]->address));
    }
    
    void wait_for_cuda(cudaStream_t &stream, cudaError_t expected = cudaSuccess) {
        cudaError_t status;
        wait_for_cond(
                [&]() {
                    status = cudaStreamQuery(stream);
                    return status == expected;
                },
                []() {usleep(100);}
            );
        ASSERT_EQ(expected, status);
    }

    bool batch_is_identical(rma_iov& list) {
        for (int i = 0; i < chunk_count; i++) {
            auto rma_iov = &list.get()[i];
            if (!cuda_is_same(rma_iov->local_va,
                              reinterpret_cast<void*>(rma_iov->remote_va),
                              rma_iov->length)) {
                return false;
            }
        }
        return true;
    }

    void test_batch_export_unexport(rma_iov* rma_list,
                                    size_t chunk_count,
                                    uint64_t signal_va,
                                    ucp_rkey_h signal_rkey)
    {
        ucs_status_ptr_t req;
        ucp_batch_h batch;
        ucs_status_t status;
        cudaStream_t stream;

        req = ucp_ep_rma_batch_prepare(sender().ep(),
                                       rma_list == NULL ? NULL : rma_list->get(),
                                       chunk_count, signal_va,
                                       signal_rkey, &prepare_param);
        ASSERT_TRUE(UCS_PTR_IS_PTR(req));

        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        status = ucp_ep_rma_batch_export(req, &batch);
        if (!has_transport("gdaki")) {
            ASSERT_UCS_STATUS_EQ(UCS_ERR_NOT_IMPLEMENTED, status);
            ucp_request_release(req);
            return;
        }

        status = ucp_ep_rma_batch_release(req, batch);
        EXPECT_EQ(UCS_OK, status);
        status = ucp_ep_rma_batch_release(req, batch);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

        // Return the request
        ucp_request_release(req);
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }

    void test_batch_export_transfer_unexport(rma_iov* rma_list,
                                             size_t chunk_count,
                                             uint64_t signal_va,
                                             ucp_rkey_h signal_rkey,
                                             bool no_delay = true)
    {
        const uint64_t signal_inc = 1;
        uint64_t expected_signal  = signal_inc;
        const bool signal_only    = (rma_list == NULL);
        uint64_t flags = UCP_DEV_BATCH_FLAG_DEFAULT;
        ucs_status_ptr_t req;
        ucp_batch_h batch;
        ucs_status_t status;
        cudaStream_t stream;

        if (signal_va == 0) {
            flags &= ~UCP_DEV_BATCH_FLAG_ATOMIC;
            expected_signal = 0;
        }

        if (signal_only) {
            flags &= ~UCP_DEV_BATCH_FLAG_RMA_IOV;
        }

        if (!no_delay) {
            flags &= ~UCP_DEV_BATCH_FLAG_NODELAY;
        }

        if ((flags & (UCP_DEV_BATCH_FLAG_ATOMIC | UCP_DEV_BATCH_FLAG_RMA_IOV)) == 0) {
            UCS_TEST_ABORT("Invalid params - Try to prepare an empty batch");
        }

        cudaMemset(signal_reg[0]->address, 0, sizeof(uint64_t));
        req = ucp_ep_rma_batch_prepare(sender().ep(),
                                       rma_list == NULL ? NULL : rma_list->get(),
                                       chunk_count, signal_va,
                                       signal_rkey, &prepare_param);
        ASSERT_TRUE(UCS_PTR_IS_PTR(req));

        ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

        status = ucp_ep_rma_batch_export(req, &batch);
        if (!has_transport("gdaki")) {
            ASSERT_UCS_STATUS_EQ(UCS_ERR_NOT_IMPLEMENTED, status);
            ucp_request_release(req);
            return;
        }

        // Prerequisites
        ASSERT_UCS_OK(status);
        if (!signal_only) {
            ASSERT_FALSE(batch_is_identical(*rma_list));
        }
        ASSERT_EQ(0, cuda_value_get(signal_reg[0]->address));

        // Take a warp for now, busy loop and check content
        execute_ucp_batch_kernel<<<1, 32, 0, stream>>>(
                batch, flags, signal_inc);
        wait_for_cuda(stream);
        if (no_delay) {
            wait_for_signal(expected_signal);
            if (!signal_only) {
                ASSERT_TRUE(batch_is_identical(*rma_list));
            }
        } else {
            wait_for_signal(0);
        }

        flags |= UCP_DEV_BATCH_FLAG_NODELAY;
        // Reuse the batch
        execute_ucp_batch_kernel<<<1, 32, 0, stream>>>(
                batch, flags, signal_inc);
        if (signal_va != 0) {
            expected_signal++;
        }

        wait_for_cuda(stream);
        wait_for_signal(expected_signal);

        status = ucp_ep_rma_batch_release(req, batch);
        EXPECT_EQ(UCS_OK, status);
        status = ucp_ep_rma_batch_release(req, batch);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

        // Return the request
        ucp_request_release(req);
        ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
    }

    virtual void init() {
        test_ucp_batch_base::init();

        memset(&prepare_param, 0, sizeof(prepare_param));

        chunks_tx  = alloc_chunks(chunk_size, chunk_count); // TODO: Add GPU id
        chunks_rx  = alloc_chunks(chunk_size, chunk_count); // for all
        signal     = alloc_chunks(signal_size, 1, UCS_MEMORY_TYPE_CUDA, true);
        host       = alloc_chunks(signal_size, 1, UCS_MEMORY_TYPE_HOST);

        local_reg  = create_regs(chunks_tx, sender());
        remote_reg = create_regs(chunks_rx, receiver());
        signal_reg = create_regs(signal, receiver());
        host_reg   = create_regs(host, receiver());

        rma_list   = create_rma_iov(sender().ep(), local_reg, remote_reg);

        signal_rkey = signal_reg[0]->unpack(sender().ep());
        if (signal_rkey->md_map == 0) {
            UCS_TEST_SKIP_R("No MD for signal rkey");
            return;
        }

        /*
         * Conclude wireup and work on real endpoints, put_batch will return
         * no resource, and retry will be needed otherwise.
         */
        for (int count = 200; count > 0; count--) {
            progress();
        }
    }

    virtual void cleanup() {
        local_reg.clear();
        remote_reg.clear();
        signal_reg.clear();
        host_reg.clear();
        test_ucp_batch_base::cleanup();
    }
};

std::vector<std::unique_ptr<mem_buffer>>
test_ucp_batch::alloc_chunks(size_t size, int count, ucs_memory_type_t mem_type,
                             bool zero)
{
    std::vector<std::unique_ptr<mem_buffer>> chunks;

    while (count-- > 0) {
        chunks.emplace_back(my_make_unique<mem_buffer>(size, mem_type));
        chunks.back()->memset(zero? 0 : init_val++);
    }

    return chunks;
}

UCS_TEST_P(test_ucp_batch, prepare_empty_returns_err)
{
    ucs_status_ptr_t req;
    req = ucp_ep_rma_batch_prepare(sender().ep(), rma_list.get(), 0, 0,
                                   signal_reg[0]->unpack(sender().ep()),
                                   &prepare_param);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, UCS_PTR_STATUS(req));
}

UCS_TEST_P(test_ucp_batch, prepare_iov_returns_err)
{
    ucs_status_ptr_t req;
    req = ucp_ep_rma_batch_prepare(sender().ep(), NULL, 1, signal_reg[0]->rva(),
                                   signal_reg[0]->unpack(sender().ep()),
                                   &prepare_param);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, UCS_PTR_STATUS(req));
}

UCS_TEST_P(test_ucp_batch, prepare_signal_returns_err)
{
    ucs_status_ptr_t req;
    req = ucp_ep_rma_batch_prepare(sender().ep(), rma_list.get(),
                                   chunk_count, signal_reg[0]->rva(),
                                   NULL, &prepare_param);
    ASSERT_UCS_STATUS_EQ(UCS_ERR_INVALID_PARAM, UCS_PTR_STATUS(req));
}

UCS_TEST_P(test_ucp_batch, prepare_returns_req)
{
    ucs_status_ptr_t req;

    req = ucp_ep_rma_batch_prepare(sender().ep(),
                                   rma_list.get(), chunk_count,
                                   signal_reg[0]->rva(),
                                   signal_reg[0]->unpack(sender().ep()),
                                   &prepare_param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(req));
    ucp_request_release(req);
}

UCS_TEST_P(test_ucp_batch, prepare_iov_returns_req)
{
    ucs_status_ptr_t req;

    req = ucp_ep_rma_batch_prepare(sender().ep(),
                                   rma_list.get(), chunk_count,
                                   0, NULL, &prepare_param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(req));
    ucp_request_release(req);
}

UCS_TEST_P(test_ucp_batch, prepare_signal_returns_req)
{
    ucs_status_ptr_t req;

    req = ucp_ep_rma_batch_prepare(sender().ep(), NULL, 0, signal_reg[0]->rva(),
                                   signal_reg[0]->unpack(sender().ep()),
                                   &prepare_param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(req));
    ucp_request_release(req);
}

UCS_TEST_P(test_ucp_batch, export_wrong_req)
{
    ucs_status_t status;
    ucp_batch_h batch;
    uint64_t buffer;
    uint64_t value = 1;
    ucp_request_param_t param;

    param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                         UCP_OP_ATTR_FIELD_FLAGS |
                         UCP_OP_ATTR_FIELD_REPLY_BUFFER;
    param.flags        = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    param.reply_buffer = &buffer;
    param.datatype     = ucp_dt_make_contig(sizeof(value));

    auto req = ucp_atomic_op_nbx(sender().ep(), UCP_ATOMIC_OP_ADD, &value,
                                 1, host_reg[0]->rva(),
                                 host_reg[0]->unpack(sender().ep()), &param);

    status = ucp_ep_rma_batch_export(req, &batch);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    ASSERT_UCS_OK(requests_wait({req}));
}

UCS_TEST_P(test_ucp_batch, unexport_without_export)
{
    ucs_status_ptr_t req;
    ucs_status_t status;
    ucp_batch_h batch = NULL;

    req = ucp_ep_rma_batch_prepare(sender().ep(),
                                   rma_list.get(), chunk_count,
                                   signal_reg[0]->rva(),
                                   signal_rkey,
                                   &prepare_param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(req));
    status = ucp_ep_rma_batch_release(req, batch);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    ucp_request_release(req);
}

__global__ void execute_ucp_batch_kernel(ucp_batch_h batch, uint64_t flags, uint64_t signal_inc)
{
    __shared__ ucp_dev_request_t request;
    ucs_status_t status;

    ucp_dev_batch_execute(batch, flags, signal_inc,
                          &request);

    if (flags & UCP_DEV_BATCH_FLAG_NODELAY) {
        status = ucp_dev_request_progress(&request);
        if (status != UCS_OK) {
            printf("Failed to progress request %d\n", status);
            return;
        }
    }
}

UCS_TEST_P(test_ucp_batch, prepare_export_unexport)
{
    test_batch_export_unexport(&rma_list, chunk_count, signal_reg[0]->rva(), signal_rkey);
}

UCS_TEST_P(test_ucp_batch, prepare_iov_export_unexport)
{
    test_batch_export_unexport(&rma_list, chunk_count, 0, NULL);
    test_batch_export_unexport(&rma_list, chunk_count, 0, signal_rkey);
}

UCS_TEST_P(test_ucp_batch, prepare_signal_export_unexport)
{
    test_batch_export_unexport(NULL, 0, signal_reg[0]->rva(), signal_rkey);
    test_batch_export_unexport(&rma_list, 0, signal_reg[0]->rva(), signal_rkey);
}

UCS_TEST_P(test_ucp_batch, prepare_export_transfer_unexport)
{
    test_batch_export_transfer_unexport(&rma_list, chunk_count, signal_reg[0]->rva(), signal_rkey);
}

UCS_TEST_P(test_ucp_batch, prepare_iov_export_transfer_unexport)
{
    test_batch_export_transfer_unexport(&rma_list, chunk_count, 0, NULL);
}

UCS_TEST_P(test_ucp_batch, prepare_signal_export_transfer_unexport)
{
    test_batch_export_transfer_unexport(NULL, 0, signal_reg[0]->rva(), signal_rkey);
}

UCS_TEST_P(test_ucp_batch, no_delay_export_transfer_unexport)
{
    test_batch_export_transfer_unexport(&rma_list, chunk_count, signal_reg[0]->rva(), signal_rkey, false);
}


// TODO - Extends partial batch tests.
template<int indcnt, int per_warp>
__global__ void execute_ucp_batch_part_kernel(ucp_batch_h batch, uint64_t flags, uint64_t signal_inc, size_t iovcnt, size_t length)
{
    const int num_warps = indcnt / per_warp;
    __shared__ ucp_dev_request_t requests[num_warps];
    int warp_id = threadIdx.x / WARP_THREADS;
    int lane_id = threadIdx.x % WARP_THREADS;
    ucs_status_t status;
    int indices[per_warp];
    size_t sizes[per_warp];
    size_t src_offs[per_warp];
    size_t dst_offs[per_warp];

    if (lane_id < per_warp) {
        indices[lane_id] = (warp_id * per_warp + lane_id) / (indcnt/ iovcnt);
        sizes[lane_id] = length / indcnt;
        dst_offs[lane_id] = length / indcnt * (lane_id % (indcnt / iovcnt));
        src_offs[lane_id] = length / indcnt * (lane_id % (indcnt / iovcnt));
    }

    __syncwarp();
    ucp_dev_batch_execute_part<UCP_DEV_SCALE_WARP>(
            batch, flags, signal_inc, per_warp, indices, src_offs, dst_offs,
            sizes, requests + warp_id);
    status = ucp_dev_request_progress<UCP_DEV_SCALE_WARP>(requests + warp_id);
    if (status != UCS_OK) {
        printf("Failed to progress request %d\n", status);
        return;
    }
}

UCS_TEST_P(test_ucp_batch, part_warp)
{
    const uint64_t signal_inc = 1;
    const int chunk_count = 32;
    const int per_warp = 8;
    const int slices = 2;
    const int indcnt = chunk_count * slices;
    const int num_warps = indcnt / per_warp;
    const int num_threads = num_warps * WARP_THREADS;
    ucs_status_ptr_t req;
    ucp_batch_h batch;
    ucs_status_t status;
    cudaStream_t stream;

    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    req = ucp_ep_rma_batch_prepare(sender().ep(),
                                   rma_list.get(), chunk_count,
                                   signal_reg[0]->rva(),
                                   signal_rkey,
                                   &prepare_param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(req));

    status = ucp_ep_rma_batch_export(req, &batch);
    if (!has_transport("gdaki")) {
        ASSERT_UCS_STATUS_EQ(UCS_ERR_NOT_IMPLEMENTED, status);
        ucp_request_release(req);
        return;
    }

    // Prerequisites
    ASSERT_UCS_OK(status);
    ASSERT_FALSE(batch_is_identical(rma_list));
    ASSERT_EQ(0, cuda_value_get(signal_reg[0]->address));

    // Take a warp for now, busy loop and check content
    execute_ucp_batch_part_kernel<indcnt, per_warp><<<1, num_threads, 0, stream>>>(batch,
            UCP_DEV_BATCH_FLAG_DEFAULT, signal_inc, chunk_count, chunk_count * chunk_size);
    wait_for_cuda(stream);
    wait_for_signal(num_warps);
    ASSERT_TRUE(batch_is_identical(rma_list));

    // Reuse the batch
    execute_ucp_batch_part_kernel<indcnt, per_warp><<<1, num_threads, 0, stream>>>(batch,
            UCP_DEV_BATCH_FLAG_DEFAULT, signal_inc, chunk_count, chunk_count * chunk_size);
    wait_for_cuda(stream);
    ASSERT_EQ(2 * num_warps, cuda_value_get(signal_reg[0]->address));

    status = ucp_ep_rma_batch_release(req, batch);
    EXPECT_EQ(UCS_OK, status);
    status = ucp_ep_rma_batch_release(req, batch);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    // Return the request
    ucp_request_release(req);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

template<int indcnt, int per_warp>
__global__ void execute_no_req_kernel(ucp_batch_h batch, uint64_t flags, uint64_t signal_inc, size_t iovcnt, size_t length)
{
    int warp_id = threadIdx.x / WARP_THREADS;
    int lane_id = threadIdx.x % WARP_THREADS;
    int indices[per_warp];
    size_t sizes[per_warp];
    size_t src_offs[per_warp];
    size_t dst_offs[per_warp];

    if (lane_id < per_warp) {
        indices[lane_id] = (warp_id * per_warp + lane_id) / (indcnt/ iovcnt);
        sizes[lane_id] = length / indcnt;
        dst_offs[lane_id] = length / indcnt * (lane_id % (indcnt / iovcnt));
        src_offs[lane_id] = length / indcnt * (lane_id % (indcnt / iovcnt));
    }

    __syncwarp();
    ucp_dev_batch_execute_part<UCP_DEV_SCALE_WARP>(
            batch, flags, signal_inc, per_warp, indices, src_offs, dst_offs,
            sizes, NULL);
}

UCS_TEST_P(test_ucp_batch, no_req)
{
    const uint64_t signal_inc = 1;
    const int chunk_count = 32;
    const int per_warp = 8;
    const int slices = 2;
    const int indcnt = chunk_count * slices;
    const int num_warps = indcnt / per_warp;
    const int num_threads = num_warps * WARP_THREADS;
    ucs_status_ptr_t req;
    ucp_batch_h batch;
    ucs_status_t status;
    cudaStream_t stream;

    ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));

    req = ucp_ep_rma_batch_prepare(sender().ep(),
                                   rma_list.get(), chunk_count,
                                   signal_reg[0]->rva(),
                                   signal_rkey,
                                   &prepare_param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(req));

    status = ucp_ep_rma_batch_export(req, &batch);
    if (!has_transport("gdaki")) {
        ASSERT_UCS_STATUS_EQ(UCS_ERR_NOT_IMPLEMENTED, status);
        ucp_request_release(req);
        return;
    }

    ASSERT_UCS_OK(status);
    ASSERT_FALSE(batch_is_identical(rma_list));
    ASSERT_EQ(0, cuda_value_get(signal_reg[0]->address));

    const auto flags = UCP_DEV_BATCH_FLAG_DEFAULT & ~UCP_DEV_BATCH_FLAG_COMP;
    execute_no_req_kernel<indcnt, per_warp><<<1, num_threads, 0, stream>>>(batch,
        flags, signal_inc, chunk_count, chunk_count * chunk_size);
    wait_for_cuda(stream);
    while (cuda_value_get(signal_reg[0]->address) != num_warps);
    ASSERT_TRUE(batch_is_identical(rma_list));

    status = ucp_ep_rma_batch_release(req, batch);
    EXPECT_EQ(UCS_OK, status);

    ucp_request_release(req);
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_batch, rc_v, "rc_v")
UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_batch, gdaki, "gdaki,rc_v")
