/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "gdaki.h"

#include <ucs/time/time.h>
#include <ucs/datastruct/string_buffer.h>
#include <uct/ib/mlx5/rc/rc_mlx5.h>
#include <uct/cuda/base/cuda_iface.h>

#include <doca_log.h>
#include <cuda.h>


#define UCT_GDAKI_DOCA_NOTUSE    1
#define UCT_GDAKI_DOCA_NOTUSEPTR (void*)1

typedef struct {
    uct_rc_iface_common_config_t      super;
    uct_rc_mlx5_iface_common_config_t mlx5;
} uct_rc_gdaki_iface_config_t;

ucs_config_field_t uct_rc_gdaki_iface_config_table[] = {
  {UCT_IB_CONFIG_PREFIX, "", NULL,
   ucs_offsetof(uct_rc_gdaki_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_common_config_table)},

  {UCT_IB_CONFIG_PREFIX, "", NULL,
   ucs_offsetof(uct_rc_gdaki_iface_config_t, mlx5),
   UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

  {NULL}
};


static UCS_CLASS_INIT_FUNC(uct_rc_gdaki_ep_t, const uct_ep_params_t *params)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(params->iface,
                                                 uct_rc_gdaki_iface_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.super.super.md,
                                          uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr = {};
    uct_ib_mlx5_cq_attr_t cq_attr      = {};
    uct_ib_mlx5_qp_attr_t qp_attr      = {};
    uct_rc_gdaki_dev_ep_t dev_ep       = {};
    ucs_status_t status;
    doca_error_t derr;
    size_t dev_ep_size;
    uct_ib_mlx5_dbrec_t dbrec;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super.super.super);

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(iface->cuda_ctx));
    if (status != UCS_OK) {
        return status;
    }

    init_attr.cq_len[UCT_IB_DIR_TX] = iface->super.super.config.tx_qp_len *
                                      UCT_IB_MLX5_MAX_BB;
    uct_ib_mlx5_cq_calc_sizes(&iface->super.super.super, UCT_IB_DIR_TX,
                              &init_attr, 0, &cq_attr);
    uct_rc_iface_fill_attr(&iface->super.super, &qp_attr.super,
                           iface->super.super.config.tx_qp_len, NULL);
    uct_ib_mlx5_wq_calc_sizes(&qp_attr);

    cq_attr.flags      |= UCT_IB_MLX5_CQ_IGNORE_OVERRUN;
    cq_attr.umem_offset = ucs_align_up_pow2(
            sizeof(uct_rc_gdaki_dev_ep_t) +
                    qp_attr.max_tx * sizeof(uct_rc_gdaki_op_t),
            ucs_get_page_size());

    qp_attr.mmio_mode     = UCT_IB_MLX5_MMIO_MODE_DB;
    qp_attr.super.srq_num = 0;
    qp_attr.umem_offset   = ucs_align_up_pow2(cq_attr.umem_offset +
                                                      cq_attr.umem_len,
                                              ucs_get_page_size());

    dev_ep_size = qp_attr.umem_offset + qp_attr.len;
    /*
     * dev_ep layout:
     * +---------------------+-------+---------+---------+
     * | counters, dbr       | ops   | cq buff | wq buff |
     * +---------------------+-------+---------+---------+
     */
    derr = doca_gpu_mem_alloc(iface->gpu_dev, dev_ep_size, ucs_get_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU, (void**)&self->ep_gpu,
                              NULL);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_gpu_mem_alloc failed: %s", doca_error_get_descr(derr));
        status = UCS_ERR_IO_ERROR;
        goto err_ctx;
    }

    /* TODO add dmabuf_fd support */
    self->umem = mlx5dv_devx_umem_reg(md->super.dev.ibv_context, self->ep_gpu,
                                      dev_ep_size, IBV_ACCESS_LOCAL_WRITE);
    if (self->umem == NULL) {
        uct_ib_check_memlock_limit_msg(md->super.dev.ibv_context,
                                       UCS_LOG_LEVEL_ERROR,
                                       "mlx5dv_devx_umem_reg(size=%zu)",
                                       dev_ep_size);
        status = UCS_ERR_NO_MEMORY;
        goto err_mem;
    }

    self->cq.devx.mem.mem       = self->umem;
    self->qp.super.devx.mem.mem = self->umem;

    dbrec.mem_id        = self->umem->umem_id;
    dbrec.offset        = ucs_offsetof(uct_rc_gdaki_dev_ep_t, cq_dbrec);
    self->cq.devx.dbrec = &dbrec;
    status = uct_ib_mlx5_devx_create_cq_common(&iface->super.super.super,
                                               UCT_IB_DIR_TX, &cq_attr,
                                               &self->cq, 0, 0);
    if (status != UCS_OK) {
        goto err_umem;
    }

    dbrec.offset              = ucs_offsetof(uct_rc_gdaki_dev_ep_t, qp_dbrec);
    self->qp.super.devx.dbrec = &dbrec;
    status = uct_ib_mlx5_devx_create_qp_common(&iface->super.super.super,
                                               &self->cq, &self->cq,
                                               &self->qp.super, &self->qp,
                                               &qp_attr);
    if (status != UCS_OK) {
        goto err_cq;
    }

    derr = doca_gpu_verbs_bridge_export_qp(
            iface->gpu_dev, self->qp.super.qp_num,
            UCS_PTR_BYTE_OFFSET(self->ep_gpu, qp_attr.umem_offset),
            qp_attr.max_tx, self->ep_gpu->qp_dbrec, self->qp.reg->addr.ptr,
            UCT_IB_MLX5_BF_REG_SIZE * 2, self->cq.cq_num,
            UCS_PTR_BYTE_OFFSET(self->ep_gpu, cq_attr.umem_offset),
            cq_attr.cq_size, self->ep_gpu->cq_dbrec, UCT_GDAKI_DOCA_NOTUSE,
            UCT_GDAKI_DOCA_NOTUSEPTR, UCT_GDAKI_DOCA_NOTUSE,
            UCT_GDAKI_DOCA_NOTUSEPTR, UCT_GDAKI_DOCA_NOTUSE,
            UCT_GDAKI_DOCA_NOTUSE, UCT_GDAKI_DOCA_NOTUSEPTR,
            UCT_GDAKI_DOCA_NOTUSE, UCT_GDAKI_DOCA_NOTUSEPTR, 0, &self->qp_cpu);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_gpu_verbs_bridge_export_qp failed: %s",
                  doca_error_get_descr(derr));
        status = UCS_ERR_INVALID_PARAM;
        goto err_qp;
    }

    derr = doca_gpu_verbs_get_qp_dev(self->qp_cpu, &self->qp_gpu);
    if (derr != DOCA_SUCCESS) {
        status = UCS_ERR_INVALID_PARAM;
        goto err_dev_ep;
    }

    dev_ep.qp = self->qp_gpu;
    dev_ep.atomic_va = iface->atomic_buff;

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemsetD8((CUdeviceptr)self->ep_gpu, 0, dev_ep_size));
    if (status != UCS_OK) {
        goto err_dev_ep;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemsetD8((CUdeviceptr)UCS_PTR_BYTE_OFFSET(self->ep_gpu,
                                                        cq_attr.umem_offset),
                       0xff, cq_attr.umem_len));
    if (status != UCS_OK) {
        goto err_dev_ep;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyHtoD((CUdeviceptr)self->ep_gpu, &dev_ep, sizeof(dev_ep)));
    if (status != UCS_OK) {
        goto err_dev_ep;
    }

    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return UCS_OK;

err_dev_ep:
    doca_gpu_verbs_unexport_qp(iface->gpu_dev, self->qp_cpu);
err_qp:
    uct_ib_mlx5_devx_destroy_qp_common(&self->qp.super);
err_cq:
    uct_ib_mlx5_devx_destroy_cq_common(&self->cq);
err_umem:
    mlx5dv_devx_umem_dereg(self->umem);
err_mem:
    doca_gpu_mem_free(iface->gpu_dev, self->ep_gpu);
err_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_gdaki_ep_t)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                                 uct_rc_gdaki_iface_t);
    doca_error_t derr;

    derr = doca_gpu_verbs_unexport_qp(iface->gpu_dev, self->qp_cpu);
    if (derr != DOCA_SUCCESS) {
        ucs_warn("doca_gpu_rdma_verbs_unexport_qp failed: %s",
                 doca_error_get_descr(derr));
    }

    uct_ib_mlx5_devx_destroy_qp_common(&self->qp.super);
    uct_ib_mlx5_devx_destroy_cq_common(&self->cq);
    mlx5dv_devx_umem_dereg(self->umem);
    doca_gpu_mem_free(iface->gpu_dev, self->ep_gpu);
}

UCS_CLASS_DEFINE(uct_rc_gdaki_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_gdaki_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_gdaki_ep_t, uct_ep_t);

static ucs_status_t
uct_rc_gdaki_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_rc_gdaki_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_gdaki_ep_t);
    uct_rc_mlx5_base_ep_address_t *rc_addr = (void*)addr;

    uct_ib_pack_uint24(rc_addr->qp_num, ep->qp.super.qp_num);
    return UCS_OK;
}

static ucs_status_t uct_rc_gdaki_iface_get_address(uct_iface_h tl_iface,
                                                   uct_iface_addr_t *addr)
{
    *(uint8_t*)addr = UCT_RC_MLX5_IFACE_ADDR_TYPE_BASIC;
    return UCS_OK;
}

static ucs_status_t
uct_rc_gdaki_ep_connect_to_ep_v2(uct_ep_h ep,
                                 const uct_device_addr_t *device_addr,
                                 const uct_ep_addr_t *ep_addr,
                                 const uct_ep_connect_to_ep_params_t *params)
{
    uct_rc_gdaki_ep_t *gdaki_ep     = ucs_derived_of(ep, uct_rc_gdaki_ep_t);
    uct_rc_gdaki_iface_t *iface     = ucs_derived_of(ep->iface,
                                                     uct_rc_gdaki_iface_t);
    const uct_ib_address_t *ib_addr = (void*)device_addr;
    const uct_rc_mlx5_base_ep_address_t *rc_addr = (void*)ep_addr;
    uint8_t path_index                           = 0;
    struct ibv_ah_attr ah_attr;
    enum ibv_mtu path_mtu;
    uint32_t dest_qp_num;
    ucs_status_t status;

    status = uct_ib_iface_fill_ah_attr_from_addr(&iface->super.super.super,
                                                 ib_addr, path_index, &ah_attr,
                                                 &path_mtu);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);
    dest_qp_num = uct_ib_unpack_uint24(rc_addr->qp_num);

    return uct_rc_mlx5_iface_common_devx_connect_qp(
            &iface->super, &gdaki_ep->qp.super, dest_qp_num, &ah_attr, path_mtu,
            path_index, iface->super.super.config.max_rd_atomic);
}

int uct_rc_gdaki_ep_is_connected(uct_ep_h tl_ep,
                                 const uct_ep_is_connected_params_t *params)
{
    uct_rc_gdaki_ep_t *ep       = ucs_derived_of(tl_ep, uct_rc_gdaki_ep_t);
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                 uct_rc_gdaki_iface_t);
    uint32_t addr_qp            = 0;
    uct_rc_mlx5_base_ep_address_t *rc_addr;
    ucs_status_t status;
    struct ibv_ah_attr ah_attr;
    uint32_t qp_num;
    union ibv_gid *rgid;
    const uct_ib_address_t *ib_addr;

    status = uct_ib_mlx5_query_qp_peer_info(&iface->super.super.super,
                                            &ep->qp.super, &ah_attr, &qp_num);
    if (status != UCS_OK) {
        return 0;
    }

    /* TODO unite code with uct_rc_mlx5_base_ep_is_connected */
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

static ucs_status_t
uct_rc_gdaki_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(tl_iface,
                                                 uct_rc_gdaki_iface_t);
    ucs_status_t status;

    status = uct_ib_iface_query(&iface->super.super.super, 0, iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* TODO:
     *  - add UCT_IFACE_FLAG_PUT_BATCH
     *  - PENDING and PUT_ZCOPY will be needed to establish rma_bw lanes
     *  - As this lane does not really support PUT_ZCOPY and PENDING, this could be
     *    causing issue when trying to send standard PUT. Eventually we must probably
     *    introduce another type of lane (rma_batch#x).
     */
    iface_attr->cap.flags = UCT_IFACE_FLAG_CONNECT_TO_EP |
                            UCT_IFACE_FLAG_INTER_NODE |
                            UCT_IFACE_FLAG_DEVICE_EP;

    iface_attr->ep_addr_len    = sizeof(uct_rc_mlx5_base_ep_address_t);
    iface_attr->iface_addr_len = sizeof(uint8_t);
    iface_attr->overhead       = UCT_RC_MLX5_IFACE_OVERHEAD;

    iface_attr->cap.put.min_zcopy = 0;
    iface_attr->cap.put.max_zcopy =
            uct_ib_iface_port_attr(&iface->super.super.super)->max_msg_sz;
    return UCS_OK;
}

static ucs_status_t uct_rc_gdaki_iface_query_v2(uct_iface_h tl_iface,
                                                uct_iface_attr_v2_t *iface_attr)
{
    if (iface_attr->field_mask & UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE) {
        iface_attr->device_mem_element_size = sizeof(
                uct_rc_gdaki_device_mem_element_t);
    }

    return UCS_OK;
}

ucs_status_t
uct_rc_gdaki_create_cq(uct_ib_iface_t *ib_iface, uct_ib_dir_t dir,
                       const uct_ib_iface_init_attr_t *init_attr,
                       int preferred_cpu, size_t inl)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(ib_iface,
                                                 uct_rc_gdaki_iface_t);

    iface->super.cq[dir].type = UCT_IB_MLX5_OBJ_TYPE_NULL;
    return UCS_OK;
}

ucs_status_t
uct_rc_gdaki_ep_get_device_ep(uct_ep_h tl_ep, uct_device_ep_h *device_ep_p)
{
    uct_rc_gdaki_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_gdaki_ep_t);

    *device_ep_p = &ep->ep_gpu->super;
    return UCS_OK;
}

ucs_status_t
uct_rc_gdaki_iface_mem_element_pack(const uct_iface_h tl_iface, uct_mem_h memh,
                                    uct_rkey_t rkey,
                                    uct_device_mem_element_t *mem_elem_p)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(tl_iface,
                                                 uct_rc_gdaki_iface_t);
    uct_rc_gdaki_device_mem_element_t mem_elem;

    mem_elem.rkey = htonl(uct_ib_md_direct_rkey(rkey));
    if (memh == NULL) {
        mem_elem.lkey = htonl(iface->atomic_mr->lkey);
    } else {
        mem_elem.lkey = htonl(((uct_ib_mem_t*)memh)->lkey);
    }

    return UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyHtoD((CUdeviceptr)mem_elem_p, &mem_elem, sizeof(mem_elem)));
}

static UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_gdaki_iface_t, uct_iface_t, uct_md_h,
                                  uct_worker_h, const uct_iface_params_t*,
                                  const uct_iface_config_t*);

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_gdaki_iface_t, uct_iface_t);

static uct_rc_iface_ops_t uct_rc_gdaki_internal_ops = {
    .super = {
        .super = {
            .iface_query_v2         = uct_rc_gdaki_iface_query_v2,
            .iface_estimate_perf    = uct_ib_iface_estimate_perf,
            .iface_vfs_refresh      = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
            .iface_mem_element_pack = uct_rc_gdaki_iface_mem_element_pack,
            .ep_query               = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
            .ep_invalidate          = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
            .ep_connect_to_ep_v2    = uct_rc_gdaki_ep_connect_to_ep_v2,
            .iface_is_reachable_v2  = (uct_iface_is_reachable_v2_func_t)ucs_empty_function_return_one_int,
            .ep_is_connected        = uct_rc_gdaki_ep_is_connected,
            .ep_get_device_ep       = uct_rc_gdaki_ep_get_device_ep,
        },
        .create_cq  = uct_rc_gdaki_create_cq,
        .destroy_cq = (uct_ib_iface_destroy_cq_func_t)ucs_empty_function_return_success,
    },
    .init_rx    = (uct_rc_iface_init_rx_func_t)ucs_empty_function_return_success,
    .cleanup_rx = (uct_rc_iface_cleanup_rx_func_t)
            ucs_empty_function_return_success,
};

static uct_iface_ops_t uct_rc_gdaki_iface_tl_ops = {
    .ep_flush          = uct_base_ep_flush,
    .ep_fence          = uct_base_ep_fence,
    .ep_create         = UCS_CLASS_NEW_FUNC_NAME(uct_rc_gdaki_ep_t),
    .ep_destroy        = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_gdaki_ep_t),
    .ep_get_address    = uct_rc_gdaki_ep_get_address,
    .ep_connect_to_ep  = uct_base_ep_connect_to_ep,
    .ep_pending_purge  = (uct_ep_pending_purge_func_t)ucs_empty_function,
    .iface_close       = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_gdaki_iface_t),
    .iface_query       = uct_rc_gdaki_iface_query,
    .iface_get_address = uct_rc_gdaki_iface_get_address,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_base_iface_is_reachable,
    .iface_flush              = (uct_iface_flush_func_t)
            ucs_empty_function_return_success,
    .iface_fence              = (uct_iface_fence_func_t)
            ucs_empty_function_return_unsupported,
    .iface_progress_enable    = (uct_iface_progress_enable_func_t)
            ucs_empty_function_return_unsupported,
    .iface_progress_disable   = (uct_iface_progress_disable_func_t)
            ucs_empty_function_return_unsupported,
    .iface_progress           = (uct_iface_progress_func_t)
            ucs_empty_function_return_unsupported,
};


static UCS_CLASS_INIT_FUNC(uct_rc_gdaki_iface_t, uct_md_h tl_md,
                           uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_gdaki_iface_config_t *config =
            ucs_derived_of(tl_config, uct_rc_gdaki_iface_config_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr = {};
    UCS_STRING_BUFFER_ONSTACK(strb, 64);
    char *gpu_name, *ib_name;
    char pci_addr[UCS_SYS_BDF_NAME_MAX];
    ucs_status_t status;
    doca_error_t derr;
    int cuda_id;

    status = uct_rc_mlx5_dp_ordering_ooo_init(md, &self->super,
                                              md->dp_ordering_cap.rc,
                                              &config->mlx5, "rc_gda");
    if (status != UCS_OK) {
        return status;
    }

    ucs_string_buffer_appendf(&strb, "%s", params->mode.device.dev_name);
    gpu_name = ucs_string_buffer_next_token(&strb, NULL, "-");
    ib_name  = ucs_string_buffer_next_token(&strb, gpu_name, "-");

    init_attr.seg_size = config->super.super.seg_size;
    init_attr.qp_type  = IBV_QPT_RC;
    init_attr.dev_name = ib_name;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_iface_common_t,
                              &uct_rc_gdaki_iface_tl_ops,
                              &uct_rc_gdaki_internal_ops, tl_md, worker, params,
                              &config->super, &config->mlx5, &init_attr);

    if (memcmp(gpu_name, UCT_DEVICE_CUDA_NAME, UCT_DEVICE_CUDA_NAME_LEN)) {
        ucs_error("wrong device name: %s\n", gpu_name);
        return status;
    }

    cuda_id = atoi(gpu_name + UCT_DEVICE_CUDA_NAME_LEN);
    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetPCIBusId(
                    pci_addr, UCS_SYS_BDF_NAME_MAX, cuda_id));
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&self->cuda_dev, cuda_id));
    if (status != UCS_OK) {
        goto err_doca;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuDevicePrimaryCtxRetain(&self->cuda_ctx, self->cuda_dev));
    if (status != UCS_OK) {
        return status;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(self->cuda_ctx));
    if (status != UCS_OK) {
        goto err_ctx_release;
    }

    derr = doca_gpu_create(pci_addr, &self->gpu_dev);
    if (derr != DOCA_SUCCESS) {
        status = UCS_ERR_IO_ERROR;
        ucs_error("doca_gpu_create failed: %s %s", doca_error_get_descr(derr),
                  pci_addr);
        goto err_ctx;
    }

    derr = doca_gpu_mem_alloc(self->gpu_dev, sizeof(uint64_t), sizeof(uint64_t),
                              DOCA_GPU_MEM_TYPE_GPU, (void**)&self->atomic_buff,
                              NULL);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_gpu_mem_alloc failed: %s", doca_error_get_descr(derr));
        status = UCS_ERR_IO_ERROR;
        goto err_doca;
    }


    self->atomic_mr = ibv_reg_mr(md->super.pd, self->atomic_buff,
                                 sizeof(uint64_t),
                                 IBV_ACCESS_LOCAL_WRITE |
                                 IBV_ACCESS_REMOTE_WRITE |
                                 IBV_ACCESS_REMOTE_READ |
                                 IBV_ACCESS_REMOTE_ATOMIC);
    if (self->atomic_mr == NULL) {
        status = UCS_ERR_IO_ERROR;
        goto err_atomic;
    }

    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return UCS_OK;

err_atomic:
    doca_gpu_mem_free(self->gpu_dev, self->atomic_buff);
err_doca:
    doca_gpu_destroy(self->gpu_dev);
err_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
err_ctx_release:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(self->cuda_dev));
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_gdaki_iface_t)
{
    ibv_dereg_mr(self->atomic_mr);
    doca_gpu_mem_free(self->gpu_dev, self->atomic_buff);
    doca_gpu_destroy(self->gpu_dev);
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(self->cuda_dev));
}

UCS_CLASS_DEFINE(uct_rc_gdaki_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_gdaki_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_gdaki_iface_t, uct_iface_t);

static ucs_status_t
uct_gdaki_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                           unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md      = ucs_derived_of(md, uct_ib_md_t);
    unsigned num_tl_devices = 0;
    uct_tl_device_resource_t *tl_devices;
    ucs_status_t status;
    CUdevice device;
    ucs_sys_device_t dev;
    ucs_sys_dev_distance_t dist;
    int num_gpus;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&num_gpus));
    if (status != UCS_OK) {
        return status;
    }

    tl_devices = ucs_malloc(sizeof(*tl_devices) * num_gpus, "gdaki_tl_devices");
    if (tl_devices == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (int i = 0; i < num_gpus; i++) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&device, i));
        if (status != UCS_OK) {
            goto err;
        }

        uct_cuda_base_get_sys_dev(device, &dev);
        status = ucs_topo_get_distance(dev, ib_md->dev.sys_dev, &dist);
        if (status != UCS_OK) {
            goto err;
        }

        /* TODO this logic should be done in UCP */
        if (dist.latency > 300.0 / UCS_NSEC_PER_SEC) {
            continue;
        }

        snprintf(tl_devices[num_tl_devices].name,
                 sizeof(tl_devices[num_tl_devices].name), "%s%d-%s:%d",
                 UCT_DEVICE_CUDA_NAME, device, uct_ib_device_name(&ib_md->dev),
                 ib_md->dev.first_port);
        tl_devices[num_tl_devices].type       = UCT_DEVICE_TYPE_NET;
        tl_devices[num_tl_devices].sys_device = dev;
        num_tl_devices++;
    }

    *num_tl_devices_p = num_tl_devices;
    *tl_devices_p     = tl_devices;
    return UCS_OK;

err:
    ucs_free(tl_devices);
    return status;
}

UCT_TL_DEFINE_ENTRY(&uct_ib_component, rc_gda, uct_gdaki_query_tl_devices,
                    uct_rc_gdaki_iface_t, "RC_GDA_",
                    uct_rc_gdaki_iface_config_table,
                    uct_rc_gdaki_iface_config_t);

static void uct_ib_doca_init(void)
{
    struct doca_log_backend *sdk_log;
    doca_error_t derr;

    derr = doca_log_backend_create_standard();
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_log_backend_create_standard failed: %d\n", derr);
        return;
    }

    derr = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_log_backend_create_with_file_sdk failed: %d\n", derr);
        return;
    }

    derr = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_ERROR);
    if (derr != DOCA_SUCCESS) {
        ucs_error("doca_log_backend_set_sdk_level failed: %d\n", derr);
        return;
    }
}

UCT_TL_INIT(&uct_ib_component, rc_gda, ctor, uct_ib_doca_init(), )
