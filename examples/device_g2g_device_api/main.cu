/*
 * Minimal GPU-to-GPU example using UCX device-side API from a CUDA kernel.
 * Host sets up MPI + UCP, exchanges remote addresses and rkeys, creates a
 * ucp_device_mem_list, and launches a kernel that performs a device-side PUT.
 */

#include <mpi.h>
#include <cuda_runtime.h>
// #include <sm_60_atomic_functions.h>

#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_impl.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>

// Simple CUDA check
#define CUDA_CHECK(cmd) do { \
    cudaError_t _e = (cmd); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define MPI_CHECK(cmd) do { \
    int _e = (cmd); \
    if (_e != MPI_SUCCESS) { \
        fprintf(stderr, "MPI error %s:%d code=%d\n", __FILE__, __LINE__, _e); \
        MPI_Abort(MPI_COMM_WORLD, _e); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define UCP_CHECK(sts, msg) do { \
    if ((sts) != UCS_OK) { \
        fprintf(stderr, "UCX error %s:%d: %s failed: %d\n", __FILE__, __LINE__, (msg), (sts)); \
        MPI_Abort(MPI_COMM_WORLD, (sts)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel params
typedef struct {
    unsigned                              num_threads;
    unsigned                              num_blocks;
    ucs_device_level_t                    level;
    bool                                  with_request;
    const ucp_device_mem_list_handle_h   *mem_lists;  // device pointer to array of handles
    unsigned                              num_lists;  // equals world_size
} kernel_params_t;

// A single PUT operation descriptor for device-side API
typedef struct {
    unsigned     list_handle_index; // index into params.mem_lists (per-destination)
    unsigned     element_index;     // element inside that mem list (always 0 here)
    const void  *address;           // local source address on device
    uint64_t     remote_address;    // remote device address at peer
    size_t       length;            // bytes to transfer
} put_op_t;

template <ucs_device_level_t level>
__global__ void do_alltoallv_kernel(kernel_params_t params,
                                     const put_op_t *ops,
                                     unsigned num_ops,
                                     ucs_status_t *status_out)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_ops) return;

    ucp_device_request_t req_obj;
    ucp_device_request_t *req = &req_obj;

    const put_op_t &op = ops[tid];
    ucs_status_t st = ucp_device_put_single<level>(params.mem_lists[op.list_handle_index],
                                                   op.element_index,
                                                   op.address,
                                                   op.remote_address,
                                                   op.length,
                                                   UCT_DEVICE_FLAG_NODELAY,
                                                   req);
    if (st != UCS_OK) {
        (void)atomicCAS((int*)status_out, (int)UCS_OK, (int)st);
    }
}

static void init_ucp(ucp_context_h &ucp_ctx, ucp_worker_h &worker)
{
    ucp_params_t ucp_params;
    memset(&ucp_params, 0, sizeof(ucp_params));
    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_RMA | UCP_FEATURE_DEVICE;
    UCP_CHECK(ucp_init(&ucp_params, nullptr, &ucp_ctx), "ucp_init");

    ucp_worker_params_t worker_params;
    memset(&worker_params, 0, sizeof(worker_params));
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    UCP_CHECK(ucp_worker_create(ucp_ctx, &worker_params, &worker), "ucp_worker_create");
}

static void create_all_endpoints(ucp_worker_h worker, int rank, int size, std::vector<ucp_ep_h> &eps)
{
    // Gather all worker addresses
    ucp_address_t *my_addr; size_t my_addr_len;
    UCP_CHECK(ucp_worker_get_address(worker, &my_addr, &my_addr_len), "ucp_worker_get_address");

    std::vector<size_t> addr_lens(size);
    MPI_CHECK(MPI_Allgather(&my_addr_len, 1, MPI_UNSIGNED_LONG, addr_lens.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD));

    // Convert size_t to int for MPI_Allgatherv
    std::vector<int> addr_lens_i(addr_lens.begin(), addr_lens.end());

    size_t total = 0; for (auto l : addr_lens) total += l;
    std::vector<uint8_t> all_addr(total);

    std::vector<int> displs(size, 0);
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + addr_lens_i[i-1];

    MPI_CHECK(MPI_Allgatherv(my_addr, my_addr_len, MPI_BYTE,
                             all_addr.data(), addr_lens_i.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD));

    eps.assign(size, nullptr);
    for (int peer = 0; peer < size; ++peer) {
        if (peer == rank) continue;
        const uint8_t *peer_addr_ptr = all_addr.data() + displs[peer];
        ucp_ep_params_t ep_params; memset(&ep_params, 0, sizeof(ep_params));
        ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
        ep_params.address    = (const ucp_address_t*)peer_addr_ptr;
        UCP_CHECK(ucp_ep_create(worker, &ep_params, &eps[peer]), "ucp_ep_create");
    }

    ucp_worker_release_address(worker, my_addr);
}

int main(int argc, char **argv)
{
    // MPI Init
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank = 0, world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2) {
        if (rank == 0) fprintf(stderr, "Run with at least 2 ranks.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Get local rank
    int local_rank = 0, local_size = 0;
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_size(local_comm, &local_size));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    // CUDA Init
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (ndev < local_size) {
        fprintf(stderr, "Not enough CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dev = local_rank;
    CUDA_CHECK(cudaSetDevice(dev));

    // Init UCP
    ucp_context_h ucp_ctx = nullptr; ucp_worker_h worker = nullptr;
    init_ucp(ucp_ctx, worker);

    // Create endpoints to all peers
    std::vector<ucp_ep_h> eps; eps.reserve(world_size);
    create_all_endpoints(worker, rank, world_size, eps);

    // Set sendcounts/displs (with variable sizes)
    size_t base_len = 1 << 20;
    std::vector<int> sendcounts(world_size, 0), senddispls(world_size, 0);
    for (int dst = 0; dst < world_size; ++dst) {
        size_t len = base_len + (((rank * world_size + dst) % 4) * 256);
        assert(len <= INT_MAX);
        sendcounts[dst] = len;
    }
    for (int i = 1; i < world_size; ++i) senddispls[i] = senddispls[i-1] + sendcounts[i-1];
    size_t total_send = senddispls.back() + sendcounts.back();

    // Compute recvcounts via alltoall
    std::vector<int> recvcounts(world_size, 0), recvdispls(world_size, 0);
    MPI_CHECK(MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int i = 1; i < world_size; ++i) recvdispls[i] = recvdispls[i-1] + recvcounts[i-1];
    size_t total_recv = recvdispls.back() + recvcounts.back();

    // Allocate CUDA send/recv buffers
    void *send_buf = nullptr; void *recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&send_buf, total_send));
    CUDA_CHECK(cudaMalloc(&recv_buf, total_recv));

    // Fill send segments with a dst-varying byte; clear recv
    for (int dst = 0; dst < world_size; ++dst) {
        void *seg = (void*)((uintptr_t)send_buf + senddispls[dst]);
        unsigned char dst_byte = (unsigned char)(0x10 + ((rank + dst) & 0xEF));
        CUDA_CHECK(cudaMemset(seg, dst_byte, sendcounts[dst]));
    }
    CUDA_CHECK(cudaMemset(recv_buf, 0x00, total_recv));

    // Register recv memory and pack rkey
    ucp_mem_map_params_t mmap_params; memset(&mmap_params, 0, sizeof(mmap_params));
    mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_params.address    = recv_buf;
    mmap_params.length     = total_recv;
    mmap_params.memory_type= UCS_MEMORY_TYPE_CUDA;

    ucp_mem_h recv_memh = nullptr;
    UCP_CHECK(ucp_mem_map(ucp_ctx, &mmap_params, &recv_memh), "ucp_mem_map(recv)");

    void *rkey_buf = nullptr; size_t rkey_size = 0;
    UCP_CHECK(ucp_rkey_pack(ucp_ctx, recv_memh, &rkey_buf, &rkey_size), "ucp_rkey_pack");

    // Share base remote address and rkey to all peers (variable sizes)
    uint64_t my_remote_addr = (uint64_t)recv_buf;
    struct { uint64_t addr; uint32_t size; } header { my_remote_addr, (uint32_t)rkey_size };

    std::vector<uint8_t> my_blob(sizeof(header) + rkey_size);
    memcpy(my_blob.data(), &header, sizeof(header));
    memcpy(my_blob.data() + sizeof(header), rkey_buf, rkey_size);

    std::vector<int> blob_sizes(world_size, 0), blob_displs(world_size, 0);
    MPI_CHECK(MPI_Allgather(&(header.size), 1, MPI_INT, blob_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int i = 0; i < world_size; ++i) blob_sizes[i] += sizeof(header);
    for (int i = 1; i < world_size; ++i) blob_displs[i] = blob_displs[i-1] + blob_sizes[i-1];
    int total_blob = 0; for (int s : blob_sizes) total_blob += s;
    std::vector<uint8_t> all_blob(total_blob);
    MPI_CHECK(MPI_Allgatherv(my_blob.data(), my_blob.size(), MPI_BYTE,
                             all_blob.data(), blob_sizes.data(), blob_displs.data(), MPI_BYTE, MPI_COMM_WORLD));

    // Unpack each peer's rkey and record remote base address
    std::vector<ucp_rkey_h> peer_rkeys(world_size, nullptr);
    std::vector<uint64_t>   peer_bases(world_size, 0);
    for (int p = 0; p < world_size; ++p) {
        const uint8_t *ptr = all_blob.data() + blob_displs[p];
        struct { uint64_t addr; uint32_t size; } ph;
        memcpy(&ph, ptr, sizeof(ph));
        peer_bases[p] = ph.addr;
        const void *prkey = ptr + sizeof(ph);
        if (p != rank) {
            UCP_CHECK(ucp_ep_rkey_unpack(eps[p], prkey, &peer_rkeys[p]), "ucp_ep_rkey_unpack");
        }
    }

    // Gather each rank's recvdispls so senders can compute remote offsets
    std::vector<int> all_recvdispls(world_size * world_size, 0);
    MPI_CHECK(MPI_Allgather(recvdispls.data(), world_size, MPI_INT,
                            all_recvdispls.data(), world_size, MPI_INT, MPI_COMM_WORLD));

    // Map local send buffer for device-side API
    ucp_mem_h send_memh = nullptr;
    ucp_mem_map_params_t mmap_send; memset(&mmap_send, 0, sizeof(mmap_send));
    mmap_send.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_send.address    = send_buf;
    mmap_send.length     = total_send;
    mmap_send.memory_type= UCS_MEMORY_TYPE_CUDA;
    UCP_CHECK(ucp_mem_map(ucp_ctx, &mmap_send, &send_memh), "ucp_mem_map(send)");

    // Create one device mem list per peer (excluding self), each with a single element
    std::vector<ucp_device_mem_list_handle_h> mem_lists(world_size, nullptr);
    std::vector<unsigned> element_index(world_size, 0);
    for (int p = 0; p < world_size; ++p) {
        if (p == rank) continue;
        ucp_device_mem_list_elem_t elem;
        elem.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH | UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY;
        elem.memh       = send_memh;
        elem.rkey       = peer_rkeys[p];

        ucp_device_mem_list_params_t ml_params;
        memset(&ml_params, 0, sizeof(ml_params));
        ml_params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                                 UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                                 UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
        ml_params.element_size = sizeof(elem);
        ml_params.num_elements = 1;
        ml_params.elements     = &elem;

        ucs_status_t st;
        do {
            st = ucp_device_mem_list_create(eps[p], &ml_params, &mem_lists[p]);
            if (st == UCS_ERR_NOT_CONNECTED) {
                ucp_worker_progress(worker);
            }
        } while (st == UCS_ERR_NOT_CONNECTED);
        UCP_CHECK(st, "ucp_device_mem_list_create(per-peer)");
        element_index[p] = 0; // single element
    }

    // Build PUT operations: for each peer, compute local and remote offsets
    std::vector<put_op_t> ops; ops.reserve(world_size);
    for (int p = 0; p < world_size; ++p) {
        if (p == rank) continue;
        put_op_t op;
        op.list_handle_index = p;
        op.element_index     = element_index[p];
        op.address        = (const void*)((uintptr_t)send_buf + senddispls[p]);
        size_t remote_off = all_recvdispls[p * world_size + rank];
        op.remote_address = peer_bases[p] + remote_off;
        op.length         = sendcounts[p];
        ops.push_back(op);
    }

    // Upload ops to device
    put_op_t *d_ops = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ops, ops.size() * sizeof(put_op_t)));
    CUDA_CHECK(cudaMemcpy(d_ops, ops.data(), ops.size() * sizeof(put_op_t), cudaMemcpyHostToDevice));

    // Prepare kernel params
    kernel_params_t kparams = {};
    const unsigned threads_per_block = 128;
    const unsigned num_ops = static_cast<unsigned>(ops.size());
    const unsigned num_blocks = (num_ops + threads_per_block - 1) / threads_per_block;
    kparams.num_threads  = threads_per_block;
    kparams.num_blocks   = num_blocks;
    kparams.level        = UCS_DEVICE_LEVEL_THREAD;
    kparams.with_request = false;
    // Upload mem list handle array to device
    ucp_device_mem_list_handle_h *d_mem_lists = nullptr;
    CUDA_CHECK(cudaMalloc(&d_mem_lists, sizeof(ucp_device_mem_list_handle_h) * world_size));
    CUDA_CHECK(cudaMemcpy(d_mem_lists, mem_lists.data(), sizeof(ucp_device_mem_list_handle_h) * world_size, cudaMemcpyHostToDevice));
    kparams.mem_lists  = d_mem_lists;
    kparams.num_lists  = world_size;

    // Launch kernel
    ucs_status_t *d_status = nullptr; ucs_status_t h_status = UCS_OK;
    CUDA_CHECK(cudaMalloc(&d_status, sizeof(*d_status)));
    CUDA_CHECK(cudaMemcpy(d_status, &h_status, sizeof(h_status), cudaMemcpyHostToDevice));

    do_alltoallv_kernel<UCS_DEVICE_LEVEL_THREAD><<<kparams.num_blocks, kparams.num_threads>>>(kparams, d_ops, num_ops, d_status);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_status, d_status, sizeof(h_status), cudaMemcpyDeviceToHost));
    if (h_status != UCS_OK) {
        fprintf(stderr, "Rank %d kernel failed: %d\n", rank, h_status);
        MPI_Abort(MPI_COMM_WORLD, h_status);
    }
    // TODO: Is call to flush necessary?
    UCP_CHECK(ucp_worker_flush(worker), "ucp_worker_flush");

    // Handle self-copy on host to emulate alltoallv semantics for self
    void *dst = (void*)((uintptr_t)recv_buf + recvdispls[rank]);
    const void *src = (const void*)((uintptr_t)send_buf + senddispls[rank]);
    CUDA_CHECK(cudaMemcpy(dst, src, sendcounts[rank], cudaMemcpyDeviceToDevice));

    // Validate
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int s = 0; s < world_size; ++s) {
        size_t len = recvcounts[s];
        std::vector<uint8_t> host_check(len, 0);
        const void *seg = (const void*)((uintptr_t)recv_buf + recvdispls[s]);
        CUDA_CHECK(cudaMemcpy(host_check.data(), seg, len, cudaMemcpyDeviceToHost));
        unsigned char expected = (unsigned char)(0x10 + ((s + rank) & 0xEF));
        for (size_t i = 0; i < len; ++i) {
            if (host_check[i] != expected) {
                fprintf(stderr, "Rank %d validation failed for segment from %d (expected 0x%02x)\n", rank, s, expected);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_ops));
    CUDA_CHECK(cudaFree(d_mem_lists));
    CUDA_CHECK(cudaFree(d_status));
    for (int p = 0; p < world_size; ++p) {
        if (p == rank) continue;
        if (mem_lists[p]) ucp_device_mem_list_release(mem_lists[p]);
    }
    for (int p = 0; p < world_size; ++p) {
        if (peer_rkeys[p]) ucp_rkey_destroy(peer_rkeys[p]);
        if (p != rank && eps[p]) ucp_ep_destroy(eps[p]);
    }
    ucp_rkey_buffer_release(rkey_buf);
    UCP_CHECK(ucp_mem_unmap(ucp_ctx, send_memh), "ucp_mem_unmap(send)");
    UCP_CHECK(ucp_mem_unmap(ucp_ctx, recv_memh), "ucp_mem_unmap(recv)");
    ucp_worker_destroy(worker);
    ucp_cleanup(ucp_ctx);

    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(recv_buf));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_Finalize();

    return 0;
}



