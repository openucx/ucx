/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <string>

extern "C" {
#include <uct/api/uct.h>
}

#include "uct_test.h"

extern "C" {
#include <uct/cuda/gdaki/gdaki_ep.h>
#include <uct/cuda/cuda_copy/cuda_copy_md.h>
}

#include <uct/api/cuda/uct_dev.cuh>

class test_gdaki : public uct_test {
protected:
    void init() {
        CUresult res_drv;
        int cuda_id;
        ucs_status_t status;

        uct_test::init();

        cuda_id = std::stoi(GetParam()->dev_name.substr(UCT_DEV_CUDA_NAME_LEN));
        res_drv = cuDeviceGet(&m_cuda_dev, cuda_id);
        if (res_drv != CUDA_SUCCESS) {
            ucs_error("cuDeviceGet returned %d.", res_drv);
            return;
        }

        status = uct_cuda_copy_push_ctx(m_cuda_dev, 1, UCS_LOG_LEVEL_ERROR);
        if (status != UCS_OK) {
            return;
        }

        m_receiver = uct_test::create_entity(0);
        m_entities.push_back(m_receiver);

        m_sender = uct_test::create_entity(0);
        m_entities.push_back(m_sender);

        m_sender->connect(0, *m_receiver, 0);
    }

    void cleanup() {
        uct_cuda_copy_pop_alloc_ctx(m_cuda_dev);
        uct_test::cleanup();
    }

    entity * m_sender;
    entity * m_receiver;

private:
    CUdevice m_cuda_dev;
};

__global__ void execute_batch_kernel(uct_dev_ep_h ep, uct_batch_h handle,
                                     uint64_t flags,
                                     ucs_status_t *status_p)
{
    __shared__ uct_dev_completion_t comp;

    comp.count          = 1;
    ucs_status_t status = uct_dev_batch_execute(handle,
                                                flags, 1,
                                                &comp);
    if (status != UCS_OK) {
        *status_p = status;
        return;
    }

    while (comp.count != 0) {
        uct_dev_ep_progress(ep);
    }
    *status_p = UCS_OK;
}

UCS_TEST_P(test_gdaki, basic)
{
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    size_t length = 1024;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer status(sizeof(ucs_status_t), 0, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    uint64_t signal_val;

    size_t iovcnt = 32;
    uct_rma_iov_t iov[iovcnt];
    size_t i;

    for (i = 0; i < iovcnt; i++) {
        iov[i].length    = length / iovcnt;
        iov[i].local_va  = (void*)((uintptr_t)sendbuf.ptr() + length / iovcnt * i);
        iov[i].memh      = sendbuf.memh();
        iov[i].remote_va = (uintptr_t)recvbuf.ptr() + length / iovcnt * i;
        iov[i].rkey      = recvbuf.rkey();
    }

    uct_dev_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_export_dev(m_sender->ep(0), &dev_ep));

    uct_batch_h batch;
    ASSERT_UCS_OK(uct_ep_batch_prepare(m_sender->ep(0), iov, iovcnt, (uint64_t)signal.ptr(), signal.rkey(), &batch));

    for (size_t i = 0; i < 100; i++) {
        signal_val = i;
        EXPECT_TRUE(mem_buffer::compare(&signal_val, signal.ptr(), sizeof(signal_val),
                                        UCS_MEMORY_TYPE_CUDA));

        status.memset(1);

        execute_batch_kernel<<<1, iovcnt/2>>>(dev_ep, batch, UCT_DEV_BATCH_FLAG_DEFAULT, (ucs_status_t *)status.ptr());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            ucs_error("kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            ucs_error("kernel execution failed: %s\n", cudaGetErrorString(err));
            return;
        }

        ucs_status_t status_ok = UCS_OK;
        EXPECT_TRUE(mem_buffer::compare(&status_ok, status.ptr(), sizeof(status_ok),
                                        UCS_MEMORY_TYPE_CUDA));

        recvbuf.pattern_check(SEED1);
        recvbuf.pattern_fill(SEED2);
    }

    uct_ep_batch_release(m_sender->ep(0), batch);
}

__global__ void resources_kernel(uct_dev_ep_h ep, uct_batch_h handle, size_t batch_size)
{
    __shared__ uct_dev_completion_t comp;
    ssize_t avail = 1024;

    for (size_t i = 0; i < 2000; i++) {
        avail -= batch_size + 1;
        if (avail > 0) {
            assert(uct_dev_batch_execute(handle, UCT_DEV_BATCH_FLAG_DEFAULT,
                                         1, &comp) == UCS_OK);
        } else {
            assert(uct_dev_batch_execute(handle, UCT_DEV_BATCH_FLAG_DEFAULT,
                                         1, &comp) == UCS_ERR_NO_RESOURCE);
            avail += batch_size + 1;
            while (avail < 512) {
                ucs_status_t status = uct_dev_ep_progress(ep);
                if (status == UCS_OK) {
                    avail += batch_size + 1;
                }
            }
        }
    }
}

UCS_TEST_P(test_gdaki, resources, "IB_TX_QUEUE_LEN=256")
{
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    size_t length = 4096;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer signal(sizeof(uint64_t), 0, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    size_t iovcnt = 64;
    uct_rma_iov_t iov[iovcnt];
    size_t i;

    for (i = 0; i < iovcnt; i++) {
        iov[i].length    = length / iovcnt;
        iov[i].local_va  = (void*)((uintptr_t)sendbuf.ptr() + length / iovcnt * i);
        iov[i].memh      = sendbuf.memh();
        iov[i].remote_va = (uintptr_t)recvbuf.ptr() + length / iovcnt * i;
        iov[i].rkey      = recvbuf.rkey();
    }

    uct_dev_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_export_dev(m_sender->ep(0), &dev_ep));

    uct_batch_h batch;
    ASSERT_UCS_OK(uct_ep_batch_prepare(m_sender->ep(0), iov, iovcnt, (uint64_t)signal.ptr(), signal.rkey(), &batch));

    resources_kernel<<<1, iovcnt>>>(dev_ep, batch, iovcnt);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        ucs_error("kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        ucs_error("kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }

    uct_ep_batch_release(m_sender->ep(0), batch);
}

template<size_t indcnt>
__global__ void execute_batch_kernel_part(uct_dev_ep_h ep, uct_batch_h handle,
                                          uint64_t flags,
                                          size_t iovcnt, size_t length)
{
    __shared__ uct_dev_completion_t comp;
    __shared__ int indices[indcnt];
    __shared__ size_t sizes[indcnt];
    __shared__ size_t src_offs[indcnt];
    __shared__ size_t dst_offs[indcnt];

    for (int i = threadIdx.x; i < indcnt; i += blockDim.x) {
        indices[i] = i / ( indcnt / iovcnt );
        sizes[i] = length / indcnt;
        dst_offs[i] = length / indcnt * (i % (indcnt / iovcnt));
        src_offs[i] = length / indcnt * (i % (indcnt / iovcnt));
    }

    __syncthreads();

    comp.count = 1;
    comp.status = UCS_OK;
    assert(!UCS_STATUS_IS_ERR(
            uct_dev_batch_execute_part(handle, flags, 4,
                                       indcnt, indices, src_offs, dst_offs,
                                       sizes, &comp)));
    while (comp.count != 0) {
        assert(!UCS_STATUS_IS_ERR(uct_dev_ep_progress(ep)));
    }
    assert(comp.status == UCS_OK);
}

UCS_TEST_P(test_gdaki, offset)
{
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    size_t length = 1024;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer signal(8, 0, *m_receiver, 0, UCS_MEMORY_TYPE_HOST);
    uint64_t signal_val;

    const size_t iovcnt = 32;
    const size_t indcnt = 64;
    uct_rma_iov_t iov[iovcnt];
    size_t i;

    for (i = 0; i < iovcnt; i++) {
        iov[i].length    = length / iovcnt;
        iov[i].local_va  = (void*)((uintptr_t)sendbuf.ptr() + length / iovcnt * i);
        iov[i].memh      = sendbuf.memh();
        iov[i].remote_va = (uintptr_t)recvbuf.ptr() + length / iovcnt * i;
        iov[i].rkey      = recvbuf.rkey();
    }

    uct_dev_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_export_dev(m_sender->ep(0), &dev_ep));

    uct_batch_h batch;
    ASSERT_UCS_OK(uct_ep_batch_prepare(m_sender->ep(0), iov, iovcnt, (uint64_t)signal.ptr(), signal.rkey(), &batch));

    for (size_t i = 0; i < 100; i++) {
        signal_val = i * 4;
        EXPECT_TRUE(mem_buffer::compare(&signal_val, signal.ptr(), 8, UCS_MEMORY_TYPE_CUDA));

        execute_batch_kernel_part<indcnt><<<1, iovcnt/2>>>(dev_ep, batch, UCT_DEV_BATCH_FLAG_DEFAULT, iovcnt, length);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            ucs_error("kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            ucs_error("kernel execution failed: %s\n", cudaGetErrorString(err));
            return;
        }

        recvbuf.pattern_check(SEED1);
        recvbuf.pattern_fill(SEED2);
    }

    uct_ep_batch_release(m_sender->ep(0), batch);
}

template <int num_warps> __global__ void
execute_warp(uct_dev_ep_h ep, uct_batch_h handle, size_t length, int num_iters, size_t indcnt) {
    __shared__ uct_dev_completion_t comp[num_warps];
    int indices[num_warps];
    size_t sizes[num_warps];
    size_t src_offs[num_warps];
    size_t dst_offs[num_warps];
    int warp_id = threadIdx.x / WARP_THREADS;
    int lane_id = threadIdx.x % WARP_THREADS;
    int per_warp = warp_id + 1;
    ucs_status_t status;

    if (lane_id < per_warp) {
        indices[lane_id] = (1 + warp_id) * (warp_id) / 2 + lane_id;
        sizes[lane_id] = length / indcnt / num_iters;
        dst_offs[lane_id] = 0;
        src_offs[lane_id] = 0;
    }

    __syncwarp();
    comp[warp_id].status = UCS_OK;
    comp[warp_id].count = num_iters;
    for (int i = 0; i < num_iters; i++) {
        do {
            assert(!UCS_STATUS_IS_ERR(uct_dev_ep_progress<UCT_DEV_SCALE_WARP>(ep)));
            status = uct_dev_batch_execute_part<UCT_DEV_SCALE_WARP>(
                    handle, UCT_DEV_BATCH_FLAG_DEFAULT, 4,
                    per_warp, indices, src_offs,
                    dst_offs, sizes, comp + warp_id);
        } while (status != UCS_OK);
        src_offs[lane_id] += sizes[lane_id];
        dst_offs[lane_id] += sizes[lane_id];
    }
    while (comp[warp_id].count != 0) {
        assert(!UCS_STATUS_IS_ERR(uct_dev_ep_progress<UCT_DEV_SCALE_WARP>(ep)));
    }
    assert(comp[warp_id].status == UCS_OK);
}

UCS_TEST_P(test_gdaki, warp)
{
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    const int num_warps = 8;
    const int iovcnt = (num_warps + 1) * num_warps / 2;
    const int num_iters = 16;
    size_t length = iovcnt * num_iters * 64;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer signal(8, 0, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    uint64_t signal_val = 0;
    uct_rma_iov_t iov[iovcnt];
    size_t i;

    for (i = 0; i < iovcnt; i++) {
        iov[i].length    = length / iovcnt;
        iov[i].local_va  = (void*)((uintptr_t)sendbuf.ptr() + length / iovcnt * i);
        iov[i].memh      = sendbuf.memh();
        iov[i].remote_va = (uintptr_t)recvbuf.ptr() + length / iovcnt * i;
        iov[i].rkey      = recvbuf.rkey();
    }

    uct_dev_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_export_dev(m_sender->ep(0), &dev_ep));

    uct_batch_h batch;
    ASSERT_UCS_OK(uct_ep_batch_prepare(m_sender->ep(0), iov, iovcnt, (uint64_t)signal.ptr(), signal.rkey(), &batch));

    for (size_t i = 0; i < 100; i++) {
        execute_warp<num_warps><<<1, num_warps * WARP_THREADS>>>(dev_ep, batch, length, num_iters, iovcnt);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            ucs_error("kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            ucs_error("kernel execution failed: %s\n", cudaGetErrorString(err));
            return;
        }

        signal_val += 4 * num_warps * num_iters;
        while (!mem_buffer::compare(&signal_val, signal.ptr(), sizeof(signal_val),
                                    UCS_MEMORY_TYPE_CUDA))
            ;
        recvbuf.pattern_check(SEED1);
        recvbuf.pattern_fill(SEED2);
    }

    uct_ep_batch_release(m_sender->ep(0), batch);
}

template <int num_warps> __global__ void
execute_thread(void *epsp, void *handlesp, size_t length, int num_iters, size_t indcnt) {
    uct_dev_ep_h *eps = (uct_dev_ep_h *)epsp;
    uct_batch_h *handles = (uct_batch_h *)handlesp;
    int indices[num_warps];
    size_t sizes[num_warps];
    size_t src_offs[num_warps];
    size_t dst_offs[num_warps];
    int warp_id = threadIdx.x / WARP_THREADS;
    int lane_id = threadIdx.x % WARP_THREADS;
    int per_warp = warp_id + 1;
    ucs_status_t status;

    if (lane_id < per_warp) {
        indices[lane_id] = (1 + warp_id) * (warp_id) / 2 + lane_id;
        sizes[lane_id] = length / indcnt / num_iters;
        dst_offs[lane_id] = 0;
        src_offs[lane_id] = 0;

        for (int i = 0; i < num_iters; i++) {
            do {
                assert(!UCS_STATUS_IS_ERR(uct_dev_ep_progress<UCT_DEV_SCALE_THREAD>(eps[lane_id])));
                status = uct_dev_batch_execute_part<UCT_DEV_SCALE_THREAD>(
                        handles[lane_id], UCT_DEV_BATCH_FLAG_DEFAULT & ~UCT_DEV_BATCH_FLAG_COMP, 4,
                        1, indices + lane_id, src_offs + lane_id,
                        dst_offs + lane_id, sizes + lane_id, NULL);
            } while (status != UCS_OK);
            src_offs[lane_id] += sizes[lane_id];
            dst_offs[lane_id] += sizes[lane_id];
        }
    }
}

UCS_TEST_P(test_gdaki, thread)
{
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    const int num_warps = 8;
    const int iovcnt = (num_warps + 1) * num_warps / 2;
    const int num_iters = 16;
    size_t length = iovcnt * num_iters * 64;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer signal(8, 0, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer gpu_eps(num_warps * sizeof(void *), 0, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer gpu_batches(num_warps * sizeof(void *), 0, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    uint64_t signal_val = 0;
    uct_rma_iov_t iov[iovcnt];
    size_t i;
    uct_dev_ep_h dev_ep[num_warps];
    uct_batch_h batch[num_warps];

    for (i = 0; i < iovcnt; i++) {
        iov[i].length    = length / iovcnt;
        iov[i].local_va  = (void*)((uintptr_t)sendbuf.ptr() + length / iovcnt * i);
        iov[i].memh      = sendbuf.memh();
        iov[i].remote_va = (uintptr_t)recvbuf.ptr() + length / iovcnt * i;
        iov[i].rkey      = recvbuf.rkey();
    }

    for (i = 0; i < num_warps; i++) {
        if (i) m_sender->connect(i, *m_receiver, i);
        ASSERT_UCS_OK(uct_ep_export_dev(m_sender->ep(i), &dev_ep[i]));
        ASSERT_UCS_OK(uct_ep_batch_prepare(m_sender->ep(i), iov, iovcnt, (uint64_t)signal.ptr(), signal.rkey(), &batch[i]));
    }

    mem_buffer::copy_to(gpu_eps.ptr(), dev_ep, sizeof(dev_ep), UCS_MEMORY_TYPE_CUDA);
    mem_buffer::copy_to(gpu_batches.ptr(), batch, sizeof(batch), UCS_MEMORY_TYPE_CUDA);

    for (size_t i = 0; i < 100; i++) {
        execute_thread<num_warps><<<1, num_warps * WARP_THREADS>>>(gpu_eps.ptr(), gpu_batches.ptr(), length, num_iters, iovcnt);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            ucs_error("kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            ucs_error("kernel execution failed: %s\n", cudaGetErrorString(err));
            return;
        }

        signal_val += 4 * iovcnt * num_iters;
        while (!mem_buffer::compare(&signal_val, signal.ptr(), sizeof(signal_val),
                                    UCS_MEMORY_TYPE_CUDA))
            ;

        recvbuf.pattern_check(SEED1);
        recvbuf.pattern_fill(SEED2);
    }

    for (i = 0; i < num_warps; i++) {
        uct_ep_batch_release(m_sender->ep(i), batch[i]);
    }
}

static void hexdump(const char *pfx, void *buff, size_t len)
{
    unsigned char *p = (unsigned char*)buff, *end = (unsigned char*)buff + len;
    unsigned int curr, prev                       = -1U;
    unsigned char c;
    char out[256], *outp;

    while (p < end) {
        outp = out;
        curr = 0;
        for (c = 0; c < 16; c++) {
            curr += p[c];
        }
        if (!curr && curr == prev) {
            goto skip;
        }
        prev  = curr;
        outp += sprintf(outp, "%4s %#lx+%04lx: ", pfx, (intptr_t)buff,
                        p - (unsigned char*)buff);
        for (c = 0; c < 16; c++) {
            outp += sprintf(outp, p + c < end ? "%02x " : "   ", p[c]);
        }
        for (c = 0; c < 16; c++) {
            outp += sprintf(outp, p + c < end ? "%c" : " ",
                            p[c] >= 32 && p[c] < 128 ? p[c] : '.');
        }
        printf("%s\n", out);
skip:
        p += 16;
    }
}

UCS_TEST_P(test_gdaki, cpu)
{
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    size_t length = 1024;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);

    uct_rma_iov_t iov;
    iov.length    = length;
    iov.local_va  = sendbuf.ptr();
    iov.memh      = sendbuf.memh();
    iov.remote_va = (uintptr_t)recvbuf.ptr();
    iov.rkey      = recvbuf.rkey();

    uct_batch_h batch_gpu;
    ASSERT_UCS_OK(uct_ep_batch_prepare(m_sender->ep(0), &iov, 1, 0, 0, &batch_gpu));

    struct doca_gpu_dev_verbs_qp qp;
    uct_gdaki_dev_ep_t ep;
    struct {
        uct_gdaki_batch_t header;
        uct_gdaki_batch_elem_t list[1];
    } batch;
    cudaMemcpy(&batch, batch_gpu, sizeof(batch), cudaMemcpyDefault);
    cudaMemcpy(&ep, batch.header.ep, sizeof(ep), cudaMemcpyDefault);
    cudaMemcpy(&qp, ep.qp, sizeof(qp), cudaMemcpyDefault);

    char buf[64];
    void *wqe    = (void*)((uintptr_t)qp.sq_wqe_daddr + ((qp.sq_wqe_pi & qp.sq_wqe_mask) << DOCA_GPUNETIO_MLX5_WQE_SQ_SHIFT));
    struct mlx5_wqe_ctrl_seg *ctrl    = (mlx5_wqe_ctrl_seg*)buf;
    struct mlx5_wqe_raddr_seg *rdma   = (mlx5_wqe_raddr_seg*)(ctrl + 1);
    struct mlx5_wqe_data_seg *data = (mlx5_wqe_data_seg*)(rdma + 1);
    int ds = 3;
    int bb = 1;

    mlx5dv_set_ctrl_seg(ctrl, qp.sq_wqe_pi, MLX5_OPCODE_RDMA_WRITE, 0,
                        qp.sq_num, MLX5_WQE_CTRL_CQ_UPDATE, ds, 0, 0);
    rdma->raddr      = htobe64(batch.list[0].dst);
    rdma->rkey       = batch.list[0].rkey;
    data->byte_count = htobe32(batch.list[0].size);
    data->addr       = htobe64(batch.list[0].src);
    data->lkey       = batch.list[0].lkey;
    cudaMemcpy(wqe, buf, 0x30, cudaMemcpyDefault);
    hexdump("WQE", ctrl, 0x30);

    qp.sq_wqe_pi += bb;
    uint32_t dbrec = htobe32(qp.sq_wqe_pi & 0xffff);
    cudaMemcpy(qp.sq_dbrec, &dbrec, 4, cudaMemcpyDefault);

    *(volatile uint64_t *)qp.sq_db = *(volatile uint64_t*)ctrl;

    struct mlx5_cqe64 *cqe;
    cqe = (struct mlx5_cqe64 *)((uintptr_t)qp.cq_sq.cqe_daddr + qp.cq_sq.cqe_ci * qp.cq_sq.cqe_size);
    struct mlx5_cqe64 cpu_cqe;
    uint64_t tmo = 1000;
    do {
        cudaMemcpy(&cpu_cqe, cqe, 64, cudaMemcpyDefault);
    } while (((qp.cq_sq.cqe_ci / qp.cq_sq.cqe_num) ^ cpu_cqe.op_own) & 1 == 1 && tmo-- != 0);

    hexdump("CQE", &cpu_cqe, sizeof(*cqe));
    uct_ep_batch_release(m_sender->ep(0), batch_gpu);
}



template<uct_dev_scale_t scale>
__global__ void execute_single_kernel(uct_dev_ep_h ep, uct_batch_h handle,
                                     uint64_t flags, size_t length,
                                     ucs_status_t *status_p)
{
    ucs_status_t status = uct_dev_batch_execute_single<scale>(
            handle, flags, 0, 0, length, NULL);
    if (status != UCS_OK) {
        *status_p = status;
        return;
    }
    *status_p = UCS_OK;
}

UCS_TEST_P(test_gdaki, single)
{
    static const uint64_t SEED1 = 0x1111111111111111lu;
    static const uint64_t SEED2 = 0x2222222222222222lu;
    size_t length = 1024;
    mapped_buffer sendbuf(length, SEED1, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer recvbuf(length, SEED2, *m_receiver, 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer status(sizeof(ucs_status_t), 0, *m_sender, 0, UCS_MEMORY_TYPE_CUDA);

    uct_rma_iov_t iov;
    iov.length    = length;
    iov.local_va  = sendbuf.ptr();
    iov.memh      = sendbuf.memh();
    iov.remote_va = (uintptr_t)recvbuf.ptr();
    iov.rkey      = recvbuf.rkey();

    uct_dev_ep_h dev_ep;
    ASSERT_UCS_OK(uct_ep_export_dev(m_sender->ep(0), &dev_ep));

    uct_batch_h batch;
    ASSERT_UCS_OK(uct_ep_batch_prepare(m_sender->ep(0), &iov, 1, 0, 0, &batch));

    for (size_t i = 0; i < 100; i++) {
        status.memset(1);
        if (i & 1) {
            execute_single_kernel<UCT_DEV_SCALE_WARP><<<1, 32>>>(dev_ep, batch, UCT_DEV_BATCH_FLAG_DEFAULT, length, (ucs_status_t *)status.ptr());
        } else {
            execute_single_kernel<UCT_DEV_SCALE_THREAD><<<1, 32>>>(dev_ep, batch, UCT_DEV_BATCH_FLAG_DEFAULT, length, (ucs_status_t *)status.ptr());
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            ucs_error("kernel launch failed: %s\n", cudaGetErrorString(err));
            return;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            ucs_error("kernel execution failed: %s\n", cudaGetErrorString(err));
            return;
        }

        ucs_status_t status_ok = UCS_OK;
        EXPECT_TRUE(mem_buffer::compare(&status_ok, status.ptr(), sizeof(status_ok),
                                        UCS_MEMORY_TYPE_CUDA));

        recvbuf.pattern_check(SEED1);
        recvbuf.pattern_fill(SEED2);
    }

    uct_ep_batch_release(m_sender->ep(0), batch);
}



_UCT_INSTANTIATE_TEST_CASE(test_gdaki, gdaki)
