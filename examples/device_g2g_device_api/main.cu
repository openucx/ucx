/*
 * Minimal GPU-to-GPU example using UCX device-side API from a CUDA kernel.
 * Host sets up MPI + UCP, exchanges remote addresses and rkeys, creates a
 * ucp_device_mem_list, and launches a kernel that performs a device-side PUT.
 */

#include <mpi.h>
#include <cuda_runtime.h>
#include <cuda.h>
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
        fprintf(stderr, "UCX error %s:%d: %s failed: %d\n", __FILE__, __LINE__, (msg), (int)(sts)); \
        MPI_Abort(MPI_COMM_WORLD, (int)(sts)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Kernel params similar to UCX tests
typedef struct {
    unsigned                     num_threads;
    unsigned                     num_blocks;
    ucs_device_level_t           level;
    bool                         with_request;
    ucp_device_mem_list_handle_h mem_list;
    struct {
        unsigned   mem_list_index;
        const void *address;
        uint64_t   remote_address;
        size_t     length;
    } single;
} kernel_params_t;

template <ucs_device_level_t level>
__global__ void do_put_kernel(kernel_params_t params, ucs_status_t *status_out)
{
    ucp_device_request_t req_obj;
    ucp_device_request_t *req = &req_obj;

    if (threadIdx.x == 0) {
        *status_out = ucp_device_put_single<level>(params.mem_list,
                                                   params.single.mem_list_index,
                                                   params.single.address,
                                                   params.single.remote_address,
                                                   params.single.length,
                                                   UCT_DEVICE_FLAG_NODELAY,
                                                   req);
    }
    __syncthreads();
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

static void create_endpoint_ring(ucp_worker_h worker, int rank, int size, ucp_ep_h &ep)
{
    // Simple ring: each rank connects to (rank+1)%size and receives from (rank-1+size)%size
    int dest = (rank + 1) % size;

    // Exchange worker addresses
    ucp_address_t *my_addr; size_t my_addr_len;
    UCP_CHECK(ucp_worker_get_address(worker, &my_addr, &my_addr_len), "ucp_worker_get_address");

    std::vector<size_t> addr_lens(size);
    MPI_CHECK(MPI_Allgather(&my_addr_len, 1, MPI_UNSIGNED_LONG, addr_lens.data(), 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD));
    // Convert counts to int as required by MPI_Allgatherv
    std::vector<int> addr_lens_i(size);
    for (int i = 0; i < size; ++i) addr_lens_i[i] = (int)addr_lens[i];
    std::vector<uint8_t> all_addr;
    size_t total = 0; for (auto l : addr_lens) total += l; all_addr.resize(total);
    std::vector<int> displs(size, 0);
    for (int i = 1; i < size; ++i) displs[i] = displs[i-1] + addr_lens_i[i-1];
    MPI_CHECK(MPI_Allgatherv(my_addr, (int)my_addr_len, MPI_BYTE, all_addr.data(), addr_lens_i.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD));

    const uint8_t *dest_addr_ptr = all_addr.data() + displs[dest];
    size_t dest_addr_len = addr_lens[dest];

    ucp_ep_params_t ep_params; memset(&ep_params, 0, sizeof(ep_params));
    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = (const ucp_address_t*)dest_addr_ptr;
    UCP_CHECK(ucp_ep_create(worker, &ep_params, &ep), "ucp_ep_create");

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
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    // CUDA Init (use Driver API first to control primary context flags)
    CUresult cu_st = cuInit(0);
    if (cu_st != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA driver init failed on rank %d: %d\n", rank, (int)cu_st);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int ndev = 0;
    cu_st = cuDeviceGetCount(&ndev);
    if (cu_st != CUDA_SUCCESS || ndev == 0) {
        fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (ndev < local_size) {
        fprintf(stderr, "Not enough CUDA devices available on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int dev = local_rank;
    CUdevice cu_dev;
    cu_st = cuDeviceGet(&cu_dev, dev);
    if (cu_st != CUDA_SUCCESS) {
        fprintf(stderr, "cuDeviceGet failed on rank %d: %d\n", rank, (int)cu_st);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // Set primary context flags before runtime attaches
    unsigned int ctx_flags = CU_CTX_SCHED_BLOCKING_SYNC; // conservative
    cu_st = cuDevicePrimaryCtxSetFlags(cu_dev, ctx_flags);
    if (cu_st != CUDA_SUCCESS) {
        fprintf(stderr, "cuDevicePrimaryCtxSetFlags failed on rank %d: %d\n", rank, (int)cu_st);
        // Not fatal; continue
    }
    CUDA_CHECK(cudaSetDevice(dev));

    size_t length = 1 << 20; // 1 MB
    if (const char *env = getenv("MSG_LEN")) {
        length = strtoull(env, nullptr, 0);
    }

    // Allocate CUDA buffers
    void *send_buf = nullptr; void *recv_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&send_buf, length));
    CUDA_CHECK(cudaMalloc(&recv_buf, length));
    // Fill send buffer with rank-specific byte, recv with distinct byte
    unsigned char send_byte = (unsigned char)(0x10 + (rank & 0xEF));
    CUDA_CHECK(cudaMemset(send_buf, send_byte, length));
    CUDA_CHECK(cudaMemset(recv_buf, 0x00, length));

    // Init UCP
    ucp_context_h ucp_ctx = nullptr; ucp_worker_h worker = nullptr;
    init_ucp(ucp_ctx, worker);

    // Create EP ring
    ucp_ep_h ep = nullptr;
    create_endpoint_ring(worker, rank, world_size, ep);

    // Register memory we want others to PUT into (recv_buf) and pack rkey
    ucp_mem_map_params_t mmap_params; memset(&mmap_params, 0, sizeof(mmap_params));
    mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_params.address    = recv_buf;
    mmap_params.length     = length;
    mmap_params.memory_type= UCS_MEMORY_TYPE_CUDA;

    ucp_mem_h recv_memh = nullptr;
    UCP_CHECK(ucp_mem_map(ucp_ctx, &mmap_params, &recv_memh), "ucp_mem_map");

    void *rkey_buf = nullptr; size_t rkey_size = 0;
    UCP_CHECK(ucp_rkey_pack(ucp_ctx, recv_memh, &rkey_buf, &rkey_size), "ucp_rkey_pack");

    // Exchange remote address and rkey with peer
    uint64_t my_remote_addr = (uint64_t)recv_buf;
    struct { uint64_t addr; uint32_t size; } header { my_remote_addr, (uint32_t)rkey_size };

    std::vector<uint8_t> my_blob(sizeof(header) + rkey_size);
    memcpy(my_blob.data(), &header, sizeof(header));
    memcpy(my_blob.data() + sizeof(header), rkey_buf, rkey_size);

    std::vector<int> sizes(world_size, 0), displs(world_size, 0);
    MPI_CHECK(MPI_Allgather(&(header.size), 1, MPI_INT, sizes.data(), 1, MPI_INT, MPI_COMM_WORLD));
    for (int i = 0; i < world_size; ++i) sizes[i] += (int)sizeof(header);
    for (int i = 1; i < world_size; ++i) displs[i] = displs[i-1] + sizes[i-1];
    int total_blob = 0; for (int s : sizes) total_blob += s;
    std::vector<uint8_t> all_blob(total_blob);
    MPI_CHECK(MPI_Allgatherv(my_blob.data(), (int)my_blob.size(), MPI_BYTE,
                             all_blob.data(), sizes.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD));

    // We send to next rank, so we need next's rkey/addr
    int peer = (rank + 1) % world_size;
    const uint8_t *p = all_blob.data() + displs[peer];
    struct { uint64_t addr; uint32_t size; } peer_header;
    memcpy(&peer_header, p, sizeof(peer_header));
    const void *peer_rkey_buf = p + sizeof(peer_header);

    ucp_rkey_h peer_rkey = nullptr;
    UCP_CHECK(ucp_ep_rkey_unpack(ep, peer_rkey_buf, &peer_rkey), "ucp_ep_rkey_unpack");

    // Create device mem list for our local source memory handle used on device
    // For single PUT, mem_list element binds the request's memory registration
    // We map our local send buffer for device-side API usage
    ucp_mem_h send_memh = nullptr;
    ucp_mem_map_params_t mmap_send; memset(&mmap_send, 0, sizeof(mmap_send));
    mmap_send.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH | UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    mmap_send.address    = send_buf;
    mmap_send.length     = length;
    mmap_send.memory_type= UCS_MEMORY_TYPE_CUDA;
    UCP_CHECK(ucp_mem_map(ucp_ctx, &mmap_send, &send_memh), "ucp_mem_map(send)");

    ucp_device_mem_list_elem_t elem;
    elem.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH | UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY;
    elem.memh       = send_memh;
    elem.rkey       = peer_rkey;

    ucp_device_mem_list_params_t ml_params;
    memset(&ml_params, 0, sizeof(ml_params));
    ml_params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                             UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                             UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
    ml_params.element_size = sizeof(elem);
    ml_params.num_elements = 1;
    ml_params.elements     = &elem;

    ucp_device_mem_list_handle_h mem_list = nullptr;
    // EP might not be connected immediately; retry if needed
    ucs_status_t st;
    do {
        st = ucp_device_mem_list_create(ep, &ml_params, &mem_list);
        if (st == UCS_ERR_NOT_CONNECTED) {
            ucp_worker_progress(worker);
        }
    } while (st == UCS_ERR_NOT_CONNECTED);
    UCP_CHECK(st, "ucp_device_mem_list_create");

    // Prepare kernel params for PUT: send from our send_buf into next's remote addr
    kernel_params_t kparams = {};
    kparams.num_threads                 = 1;
    kparams.num_blocks                  = 1;
    kparams.level                       = UCS_DEVICE_LEVEL_THREAD;
    kparams.with_request                = false;
    kparams.mem_list                    = mem_list;
    kparams.single.mem_list_index       = 0;
    kparams.single.address              = send_buf;
    kparams.single.remote_address       = peer_header.addr; // peer recv buffer
    kparams.single.length               = length;

    // Launch kernel
    ucs_status_t *d_status = nullptr; ucs_status_t h_status = UCS_OK;
    CUDA_CHECK(cudaMalloc(&d_status, sizeof(*d_status)));
    CUDA_CHECK(cudaMemcpy(d_status, &h_status, sizeof(h_status), cudaMemcpyHostToDevice));

    // Use THREAD level to avoid requiring warp-wide participation
    do_put_kernel<UCS_DEVICE_LEVEL_THREAD><<<kparams.num_blocks, kparams.num_threads>>>(kparams, d_status);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&h_status, d_status, sizeof(h_status), cudaMemcpyDeviceToHost));
    if (h_status != UCS_OK) {
        fprintf(stderr, "Rank %d kernel failed: %d\n", rank, (int)h_status);
        MPI_Abort(MPI_COMM_WORLD, (int)h_status);
    }

    // Validate: recv_buf should contain previous rank's byte
    unsigned char expected = (unsigned char)(0x10 + (((rank + world_size - 1) % world_size) & 0xEF));
    std::vector<uint8_t> host_check(4096, 0);
    CUDA_CHECK(cudaMemcpy(host_check.data(), recv_buf, host_check.size(), cudaMemcpyDeviceToHost));
    bool is_correct = true; for (auto b : host_check) if (b != expected) { is_correct = false; break; }
    if (!is_correct) {
        fprintf(stderr, "Rank %d validation failed (expected 0x%02x).\n", rank, expected);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_status));
    ucp_device_mem_list_release(mem_list);
    ucp_rkey_destroy(peer_rkey);
    ucp_rkey_buffer_release(rkey_buf);
    ucp_mem_unmap(ucp_ctx, send_memh);
    ucp_mem_unmap(ucp_ctx, recv_memh);
    ucp_ep_destroy(ep);
    ucp_worker_destroy(worker);
    ucp_cleanup(ucp_ctx);

    CUDA_CHECK(cudaFree(send_buf));
    CUDA_CHECK(cudaFree(recv_buf));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    MPI_Finalize();
    return 0;
}



