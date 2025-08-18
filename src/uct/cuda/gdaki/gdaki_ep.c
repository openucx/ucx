/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "gdaki_ep.h"
#include "gdaki_iface.h"
#include <ucs/debug/log.h>
#include <ucs/type/class.h>

#include <uct/ib/mlx5/rc/rc_mlx5.h>
#include <uct/cuda/cuda_copy/cuda_copy_md.h>

#include <cuda_runtime.h>

ucs_status_t uct_gdaki_mirror(uct_gdaki_iface_t *iface, void *cpu, void **gpu_p, size_t size,
                              int release)
{
    ucs_status_t status = UCS_OK;
    doca_error_t derr;
    cudaError_t cerr;

    derr = doca_gpu_mem_alloc(iface->gpu_dev, size, 64, DOCA_GPU_MEM_TYPE_GPU,
                              gpu_p, NULL);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_gpu_mem_alloc failed: %s", doca_error_get_descr(derr));
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    cerr = cudaMemcpy(*gpu_p, cpu, size, cudaMemcpyDefault);
    if (cerr != cudaSuccess) {
        ucs_error("cudaMemcpy failed: %s\n", cudaGetErrorString(cerr));
        status = UCS_ERR_IO_ERROR;
    }

out:
    if (release) {
        ucs_free(cpu);
    }
    return status;
}

ucs_status_t uct_gdaki_ep_batch_prepare(uct_ep_h tl_ep, const uct_rma_iov_t *iov,
                                        size_t iovcnt, uint64_t signal_va,
                                        uct_rkey_t signal_rkey, uct_batch_h *batch_p)
{
    uct_gdaki_ep_t *ep = ucs_derived_of(tl_ep, uct_gdaki_ep_t);
    uct_gdaki_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_gdaki_iface_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.md, uct_ib_mlx5_md_t);
    int has_signal = (signal_va != 0);
    size_t batch_num = iovcnt + (has_signal ? 1 : 0);
    uct_gdaki_batch_t *batch, *batch_gpu;
    size_t batch_size;
    ucs_status_t status;
    doca_error_t derr;
    cudaError_t cerr;
    size_t i;

    status = uct_cuda_copy_push_ctx(iface->cuda_dev, 1, UCS_LOG_LEVEL_ERROR);
    if (status != UCS_OK) {
        return status;
    }

    batch_size = sizeof(uct_gdaki_batch_t) + batch_num * sizeof(uct_gdaki_batch_elem_t);
    batch = ucs_calloc(1, batch_size, "gdaki batch cpu");
    if (batch == NULL) {
        goto out;
    }

    derr = doca_gpu_mem_alloc(iface->gpu_dev, batch_size, 64, DOCA_GPU_MEM_TYPE_GPU,
                              (void**)&batch_gpu, NULL);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_gpu_mem_alloc failed: %s", doca_error_get_descr(derr));
        status = UCS_ERR_IO_ERROR;
        goto err2;
    }

    batch->super.tl_id = UCT_DEV_TL_GDAKI;
    batch->num = batch_num;
    batch->op = MLX5_OPCODE_RDMA_WRITE;
    batch->ep = ep->e.ep_gpu;
    for (i = 0; i < iovcnt; i++) {
        batch->list[i].e_op = batch->op;
        batch->list[i].size = iov[i].length;
        batch->list[i].src = (uintptr_t)iov[i].local_va;
        batch->list[i].dst = (uintptr_t)iov[i].remote_va;
        batch->list[i].rkey = htonl(uct_ib_md_direct_rkey(iov[i].rkey));
        batch->list[i].lkey = htonl(((uct_ib_mem_t *)iov[i].memh)->lkey);
    }

    if (has_signal) {
        batch->list[iovcnt].e_op = MLX5_OPCODE_ATOMIC_FA;
        batch->list[iovcnt].size = sizeof(uint64_t);
        batch->list[iovcnt].src = (uint64_t)&batch_gpu->atomic_buff;
        batch->list[iovcnt].dst = signal_va;
        batch->list[iovcnt].rkey = htonl(uct_ib_md_direct_rkey(signal_rkey));
        batch->mr = ibv_reg_mr(md->super.pd, &batch_gpu->atomic_buff, sizeof(uint64_t), IBV_ACCESS_LOCAL_WRITE |
                                    IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_READ |
                                    IBV_ACCESS_REMOTE_ATOMIC);
        batch->list[iovcnt].lkey = htonl(batch->mr->lkey);
    } else {
        batch->mr = NULL;
    }


    cerr = cudaMemcpy(batch_gpu, batch, batch_size, cudaMemcpyDefault);
    if (cerr != cudaSuccess) {
        ucs_error("cudaMemcpy failed: %s\n", cudaGetErrorString(cerr));
        status = UCS_ERR_IO_ERROR;
        goto err;
    }
    *batch_p = &batch_gpu->super;
    status = UCS_OK;
    goto out;

err:
    doca_gpu_mem_free(iface->gpu_dev, batch_gpu);
err2:
    ucs_free(batch);
out:
    uct_cuda_copy_pop_alloc_ctx(iface->cuda_dev);
    return status;
}

void uct_gdaki_ep_batch_release(uct_ep_h tl_ep, uct_batch_h batch)
{
    uct_gdaki_ep_t *ep = ucs_derived_of(tl_ep, uct_gdaki_ep_t);
    uct_gdaki_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_gdaki_iface_t);
    uct_gdaki_batch_t batch_cpu;
    cudaError_t cerr;

    cerr = cudaMemcpy(&batch_cpu, batch, sizeof(batch_cpu), cudaMemcpyDefault);
    if (cerr != cudaSuccess) {
        ucs_error("cudaMemcpy failed: %s\n", cudaGetErrorString(cerr));
    }

    if (batch_cpu.mr != NULL) {
        ibv_dereg_mr(batch_cpu.mr);
    }

    doca_gpu_mem_free(iface->gpu_dev, batch);
}

ucs_status_t uct_gdaki_export_qp(uct_gdaki_ep_t *ep, uct_gdaki_iface_t *iface)
{
    uct_gdaki_dev_ep_t *dev_ep;
    ucs_status_t status;
    doca_error_t derr;
    size_t sq_size = UCS_PTR_BYTE_DIFF(ep->qp.qstart, ep->qp.qend) / 64;
    size_t cq_size = UCS_BIT(ep->cq.cq_length_log);
    size_t dev_ep_size;

    derr = doca_gpu_verbs_bridge_export_qp(iface->gpu_dev, ep->qp.super.qp_num,
            ep->qp.qstart, sq_size, (uint32_t *)ep->qp.dbrec,
            ep->qp.reg->addr.ptr, 512,
            ep->cq.cq_num, ep->cq.cq_buf, cq_size,
            (uint32_t *)ep->cq.dbrec, 1, (void*)1, 1, (void*)1, 1,
            1, (void*)1, 1, (void*)1, 0, &ep->e.qp_cpu);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_gpu_export_custom_qp failed: %s", doca_error_get_descr(derr));
        return UCS_ERR_IO_ERROR;
    }

    derr = doca_gpu_verbs_get_qp_dev(ep->e.qp_cpu, &ep->e.qp_gpu);
    if (derr != DOCA_SUCCESS) {
        status = UCS_ERR_INVALID_PARAM;
        goto err_dev_ep;
    }

    dev_ep_size = sizeof(uct_gdaki_dev_ep_t) + sq_size * sizeof(uct_gdaki_op_t);
    dev_ep = ucs_calloc(1, dev_ep_size, "dev_ep");
    if (dev_ep == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_dev_ep;
    }
    dev_ep->super.tl_id = UCT_DEV_TL_GDAKI;
    dev_ep->qp = ep->e.qp_gpu;
    status = uct_gdaki_mirror(iface, dev_ep, (void**)&ep->e.ep_gpu, dev_ep_size, 1);
    if (status != UCS_OK) {
        goto err_dev_ep;
    }

    return UCS_OK;

err_dev_ep:
    // TODO check return values
    doca_gpu_verbs_unexport_qp(iface->gpu_dev, ep->e.qp_cpu);
    return status;
}

ucs_status_t uct_gdaki_ep_export_dev(uct_ep_h tl_ep, uct_dev_ep_h *dev_ep_p)
{
    uct_gdaki_ep_t *ep = ucs_derived_of(tl_ep, uct_gdaki_ep_t);

    *dev_ep_p = &ep->e.ep_gpu->super;
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_gdaki_ep_t, const uct_ep_params_t *params)
{
    uct_gdaki_iface_t *iface = ucs_derived_of(params->iface, uct_gdaki_iface_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.md, uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr = {};
    uct_ib_mlx5_qp_attr_t qp_attr = {};
    size_t cq_size;
    uint64_t *tmp_cq;
    void *cuda_cq;
    ucs_status_t status;
    cudaError_t err;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    status = uct_cuda_copy_push_ctx(iface->cuda_dev, 1, UCS_LOG_LEVEL_ERROR);
    if (status != UCS_OK) {
        return status;
    }

    //init_attr.cq_len[UCT_IB_DIR_TX] = iface->super.config.max_inl_cqe[UCT_IB_DIR_TX]; set in .create_cq
    init_attr.cq_len[UCT_IB_DIR_TX] = iface->rc_cfg.tx_qp_len * 4;
    init_attr.alloc = &iface->alloc;
    init_attr.no_cq_prep = 1;
    init_attr.flags = UCT_IB_CQ_IGNORE_OVERRUN;
    status = uct_ib_mlx5_devx_create_cq(&iface->super, UCT_IB_DIR_TX, &init_attr,
                                        &self->cq, 0, 0);
    if (status != UCS_OK) {
        goto out;
    }

    cq_size = UCS_BIT(self->cq.cq_length_log) * UCS_BIT(self->cq.cqe_size_log);
    tmp_cq = ucs_calloc(1, cq_size, "tmp cq");
    cuda_cq = self->cq.cq_buf;
    self->cq.cq_buf = tmp_cq;
    uct_ib_mlx5_fill_cq_buf(&self->cq);
    self->cq.cq_buf = cuda_cq;
    err = cudaMemcpy(cuda_cq, tmp_cq, cq_size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
        ucs_error("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return UCS_ERR_IO_ERROR;
    }
    ucs_free(tmp_cq);

    uct_ib_iface_fill_attr_rc(&iface->super, &qp_attr.super, &iface->rc_cfg,
                              iface->rc_cfg.tx_qp_len, NULL);
    qp_attr.mmio_mode = UCT_IB_MLX5_MMIO_MODE_DB;
    qp_attr.alloc = &iface->alloc;
    qp_attr.super.srq_num = 0;
    status = uct_ib_mlx5_devx_create_qp(&iface->super, &self->cq, &self->cq,
                                        &self->qp.super, &self->qp, &qp_attr);
    if (status != UCS_OK) {
        goto err_qp;
    }

    status = uct_gdaki_export_qp(self, iface);
    if (status != UCS_OK) {
        goto err_exp;
    }

    status = UCS_OK;
    goto out;

err_exp:
    uct_ib_mlx5_devx_destroy_qp(md, &self->qp.super);
err_qp:
    uct_ib_mlx5_devx_destroy_cq(md, &self->cq);
out:
    uct_cuda_copy_pop_alloc_ctx(iface->cuda_dev);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gdaki_ep_t)
{
    uct_gdaki_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                             uct_gdaki_iface_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.md, uct_ib_mlx5_md_t);
    doca_error_t derr;

    derr = doca_gpu_verbs_unexport_qp(iface->gpu_dev, self->e.qp_cpu);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_gpu_rdma_verbs_unexport_qp failed: %s", doca_error_get_descr(derr));
    }

    doca_gpu_mem_free(iface->gpu_dev, self->e.ep_gpu);

    uct_ib_mlx5_devx_destroy_qp(md, &self->qp.super);
    uct_ib_mlx5_devx_destroy_cq(md, &self->cq);
}

UCS_CLASS_DEFINE(uct_gdaki_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_gdaki_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_gdaki_ep_t, uct_ep_t);

int
uct_gdaki_base_ep_is_connected(uct_ep_h tl_ep, const uct_ep_is_connected_params_t *params)
{
    uct_gdaki_ep_t *ep = ucs_derived_of(tl_ep, uct_gdaki_ep_t);
    uct_gdaki_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                             uct_gdaki_iface_t);
    uint32_t addr_qp = 0;
    uct_rc_mlx5_base_ep_address_t *rc_addr;
    ucs_status_t status;
    struct ibv_ah_attr ah_attr;
    uint32_t qp_num;
    union ibv_gid *rgid;
    const uct_ib_address_t *ib_addr;

    status = uct_ib_mlx5_query_qp_peer_info(&iface->super,
                                            &ep->qp.super, &ah_attr,
                                            &qp_num);
    if (status != UCS_OK) {
        return 0;
    }

    if (params->field_mask & UCT_EP_IS_CONNECTED_FIELD_EP_ADDR) {
        rc_addr = (uct_rc_mlx5_base_ep_address_t*)params->ep_addr;
        addr_qp = uct_ib_unpack_uint24(rc_addr->qp_num);
    }

    if ((addr_qp != 0) && (qp_num != addr_qp)) {
        return 0;
    }

    rgid    = (ah_attr.is_global) ? &ah_attr.grh.dgid : NULL;
    ib_addr = (const uct_ib_address_t*)params->device_addr;
    return uct_ib_iface_is_same_device(ib_addr, ah_attr.dlid, rgid);
}

ucs_status_t
uct_gdaki_ep_connect_to_ep_v2(uct_ep_h ep, const uct_device_addr_t *device_addr,
                             const uct_ep_addr_t *ep_addr,
                             const uct_ep_connect_to_ep_params_t *params)
{
    uct_gdaki_ep_t *gdaki_ep = ucs_derived_of(ep, uct_gdaki_ep_t);
    uct_gdaki_iface_t *iface = ucs_derived_of(ep->iface, uct_gdaki_iface_t);
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)device_addr;
    const uct_rc_mlx5_base_ep_address_t *rc_addr = (const uct_rc_mlx5_base_ep_address_t *)ep_addr;
    uint8_t path_index = 0;
    struct ibv_ah_attr ah_attr;
    enum ibv_mtu path_mtu;
    uint32_t dest_qp_num;
    ucs_status_t status;

    if (device_addr == NULL || ep_addr == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_ib_iface_fill_ah_attr_from_addr(&iface->super, ib_addr,
                                                 path_index, &ah_attr, &path_mtu);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);
    dest_qp_num = uct_ib_unpack_uint24(rc_addr->qp_num);

    status = uct_ib_rc_mlx5_iface_common_devx_connect_qp(&iface->super, &iface->rc_cfg,
                                                         &gdaki_ep->qp.super,
                                                         dest_qp_num, &ah_attr, path_mtu,
                                                         path_index);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

ucs_status_t uct_gdaki_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_gdaki_ep_t *ep = ucs_derived_of(tl_ep, uct_gdaki_ep_t);
    uct_rc_mlx5_base_ep_address_t *rc_addr       = (uct_rc_mlx5_base_ep_address_t*)addr;

    uct_ib_pack_uint24(rc_addr->qp_num, ep->qp.super.qp_num);
    return UCS_OK;
}
 
