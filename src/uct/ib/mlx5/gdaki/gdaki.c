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
#include <ucs/type/serialize.h>
#include <uct/ib/base/ib_verbs.h>
#include <uct/ib/mlx5/rc/rc_mlx5.h>
#include <uct/cuda/base/cuda_iface.h>

#include <cuda.h>


typedef struct {
    uct_rc_iface_common_config_t      super;
    uct_rc_mlx5_iface_common_config_t mlx5;
    unsigned                          num_channels;
} uct_rc_gdaki_iface_config_t;

ucs_config_field_t uct_rc_gdaki_iface_config_table[] = {
    {UCT_IB_CONFIG_PREFIX, "", NULL,
     ucs_offsetof(uct_rc_gdaki_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_rc_iface_common_config_table)},

    {UCT_IB_CONFIG_PREFIX, "", NULL,
     ucs_offsetof(uct_rc_gdaki_iface_config_t, mlx5),
     UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

    {"NUM_CHANNELS", "1",
     "Number of channels.",
     ucs_offsetof(uct_rc_gdaki_iface_config_t, num_channels),
     UCS_CONFIG_TYPE_UINT},

    {NULL}
};


ucs_status_t
uct_rc_gdaki_alloc(size_t size, size_t align, void **p_buf, CUdeviceptr *p_orig)
{
    unsigned int flag = 1;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemAlloc(p_orig, size + align - 1));
    if (status != UCS_OK) {
        return status;
    }

    *p_buf = (void*)ucs_align_up_pow2_ptr(*p_orig, align);
    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                  (CUdeviceptr)*p_buf));
    if (status != UCS_OK) {
        goto err;
    }

    return UCS_OK;

err:
    cuMemFree(*p_orig);
    return status;
}

static void uct_rc_gdaki_calc_dev_ep_layout(size_t num_channels,
                                            uct_ib_mlx5_qp_attr_t *qp_attr,
                                            size_t *cq_umem_offset_p,
                                            size_t *dev_ep_size_p)
{
    UCS_STATIC_ASSERT(sizeof(uct_rc_gdaki_dev_ep_t) == 64);
    UCS_STATIC_ASSERT(sizeof(uct_rc_gdaki_dev_qp_t) == 128);

    *cq_umem_offset_p    = sizeof(uct_rc_gdaki_dev_ep_t);
    qp_attr->umem_offset = *cq_umem_offset_p +
                           sizeof(uct_rc_gdaki_dev_qp_t) * num_channels;
    *dev_ep_size_p       = qp_attr->umem_offset + qp_attr->len * num_channels;
}

static UCS_CLASS_INIT_FUNC(uct_rc_gdaki_ep_t, const uct_ep_params_t *params)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(params->iface,
                                                 uct_rc_gdaki_iface_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.super.super.super.md,
                                          uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr = {};
    uct_ib_mlx5_cq_attr_t cq_attr      = {};
    uct_ib_mlx5_qp_attr_t qp_attr      = {};
    ucs_status_t status;
    size_t dev_ep_size;
    uct_ib_mlx5_dbrec_t dbrec;
    unsigned i;
    uct_rc_gdaki_channel_t *channel;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super.super.super);

    self->dev_ep_init = 0;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(iface->cuda_ctx));
    if (status != UCS_OK) {
        return status;
    }

    init_attr.cq_len[UCT_IB_DIR_TX] = 1;
    uct_ib_mlx5_cq_calc_sizes(&iface->super.super.super, UCT_IB_DIR_TX,
                              &init_attr, 0, &cq_attr);
    uct_rc_iface_fill_attr(&iface->super.super, &qp_attr.super,
                           iface->super.super.config.tx_qp_len, NULL);
    uct_ib_mlx5_wq_calc_sizes(&qp_attr);

    cq_attr.flags      |= UCT_IB_MLX5_CQ_IGNORE_OVERRUN;

    qp_attr.mmio_mode     = UCT_IB_MLX5_MMIO_MODE_DB;
    qp_attr.super.srq_num = 0;

    /* Disable inline scatter to TX CQE */
    qp_attr.super.max_inl_cqe[UCT_IB_DIR_TX] = 0;

    /*
     * dev_ep layout:
     * +-------------------+----------+---------+
     * | common data       | channels | wq buff |
     * +-------------------+----------+---------+
     *                    /            \
     *            +----------+- -----+---------+----+-----
     *            | cq entry | dbrec | indices | db |...
     *            +----------+- -----+---------+----+-----
     */
    uct_rc_gdaki_calc_dev_ep_layout(iface->num_channels, &qp_attr,
                                    &cq_attr.umem_offset, &dev_ep_size);

    status      = uct_rc_gdaki_alloc(dev_ep_size, ucs_get_page_size(),
                                     (void**)&self->ep_gpu, &self->ep_raw);
    if (status != UCS_OK) {
        goto err_ctx;
    }

    /* TODO add dmabuf_fd support */
    self->umem = mlx5dv_devx_umem_reg(md->super.dev.ibv_context, self->ep_gpu,
                                      dev_ep_size, IBV_ACCESS_LOCAL_WRITE);
    if (self->umem == NULL) {
        uct_ib_check_memlock_limit_msg(md->super.dev.ibv_context,
                                       UCS_LOG_LEVEL_ERROR,
                                       "mlx5dv_devx_umem_reg(ptr=%p size=%zu)",
                                       self->ep_gpu, dev_ep_size);
        status = UCS_ERR_NO_MEMORY;
        goto err_mem;
    }

    dbrec.mem_id = self->umem->umem_id;

    self->channels = ucs_calloc(iface->num_channels, sizeof(*self->channels),
                                "channels");
    if (self->channels == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_umem;
    }

    for (i = 0; i < iface->num_channels; i++) {
        channel = &self->channels[i];

        channel->cq.devx.mem.mem       = self->umem;
        channel->qp.super.devx.mem.mem = self->umem;

        dbrec.offset = ucs_offsetof(uct_rc_gdaki_dev_ep_t, qps[i].cq_dbrec);
        channel->cq.devx.dbrec = &dbrec;
        status = uct_ib_mlx5_devx_create_cq_common(&iface->super.super.super,
                                                   UCT_IB_DIR_TX, &cq_attr,
                                                   &channel->cq, 0, 0);
        if (status != UCS_OK) {
            goto err_qp;
        }

        dbrec.offset = ucs_offsetof(uct_rc_gdaki_dev_ep_t, qps[i].qp_dbrec);
        channel->qp.super.devx.dbrec = &dbrec;
        status = uct_ib_mlx5_devx_create_qp_common(&iface->super.super.super,
                                                   &channel->cq, &channel->cq,
                                                   &channel->qp.super,
                                                   &channel->qp, &qp_attr);
        if (status != UCS_OK) {
            goto err_cq;
        }

        cq_attr.umem_offset += sizeof(uct_rc_gdaki_dev_qp_t);
        qp_attr.umem_offset += qp_attr.len;
    }

    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return UCS_OK;

err_cq:
    uct_ib_mlx5_devx_destroy_cq_common(&self->channels[i].cq);
err_qp:
    while (i-- > 0) {
        uct_ib_mlx5_devx_destroy_qp_common(&self->channels[i].qp.super);
        uct_ib_mlx5_devx_destroy_cq_common(&self->channels[i].cq);
    }
    ucs_free(self->channels);
err_umem:
    mlx5dv_devx_umem_dereg(self->umem);
err_mem:
    cuMemFree(self->ep_raw);
err_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_gdaki_ep_t)
{
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                                 uct_rc_gdaki_iface_t);
    unsigned i;

    for (i = 0; i < iface->num_channels; i++) {
        if (self->dev_ep_init) {
            /* page with UAR might be or might be not registered already
             * so currently we just ignore errors. this may cause
             * use-after-free if we release page which is used by another EP
             * TODO and reference counted tracking for UAR pages */
            (void)cuMemHostUnregister(self->channels[i].qp.reg->addr.ptr);
        }
        uct_ib_mlx5_devx_destroy_qp_common(&self->channels[i].qp.super);
        uct_ib_mlx5_devx_destroy_cq_common(&self->channels[i].cq);
    }
    ucs_free(self->channels);
    mlx5dv_devx_umem_dereg(self->umem);
    cuMemFree(self->ep_raw);
}

UCS_CLASS_DEFINE(uct_rc_gdaki_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_gdaki_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_gdaki_ep_t, uct_ep_t);

static ucs_status_t
uct_rc_gdaki_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_rc_gdaki_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_gdaki_ep_t);
    uct_rc_gdaki_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_rc_gdaki_iface_t);
    void *rc_addr               = (void*)addr;
    unsigned i;

    for (i = 0; i < iface->num_channels; i++) {
        uct_ib_pack_uint24(*ucs_serialize_next(&rc_addr, uct_ib_uint24_t),
                           ep->channels[i].qp.super.qp_num);
    }
    return UCS_OK;
}

static ucs_status_t uct_rc_gdaki_iface_get_address(uct_iface_h tl_iface,
                                                   uct_iface_addr_t *addr)
{
    *(uint8_t*)addr = UCT_RC_MLX5_IFACE_ADDR_TYPE_BASIC;
    return UCS_OK;
}

static ucs_status_t
uct_rc_gdaki_ep_connect_to_ep_v2(uct_ep_h tl_ep,
                                 const uct_device_addr_t *device_addr,
                                 const uct_ep_addr_t *ep_addr,
                                 const uct_ep_connect_to_ep_params_t *params)
{
    uct_rc_gdaki_ep_t *ep           = ucs_derived_of(tl_ep, uct_rc_gdaki_ep_t);
    uct_rc_gdaki_iface_t *iface     = ucs_derived_of(tl_ep->iface,
                                                     uct_rc_gdaki_iface_t);
    const uct_ib_address_t *ib_addr = (void*)device_addr;
    uint8_t path_index                           = 0;
    struct ibv_ah_attr ah_attr;
    enum ibv_mtu path_mtu;
    uint32_t dest_qp_num;
    ucs_status_t status;
    unsigned i;

    status = uct_ib_iface_fill_ah_attr_from_addr(&iface->super.super.super,
                                                 ib_addr, path_index, &ah_attr,
                                                 &path_mtu);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);

    for (i = 0; i < iface->num_channels; i++) {
        dest_qp_num = uct_ib_unpack_uint24(
                *ucs_serialize_next(&ep_addr, uct_ib_uint24_t));
        status      = uct_rc_mlx5_iface_common_devx_connect_qp(
                &iface->super, &ep->channels[i].qp.super, dest_qp_num, &ah_attr,
                path_mtu, path_index, iface->super.super.config.max_rd_atomic);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
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
                                            &ep->channels[0].qp.super, &ah_attr,
                                            &qp_num);
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

    status = uct_ib_iface_query(&iface->super.super.super, iface_attr);
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
                            UCT_IFACE_FLAG_DEVICE_EP |
                            UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    iface_attr->ep_addr_len    = sizeof(uct_ib_uint24_t) * iface->num_channels;
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
    uct_rc_gdaki_ep_t *ep        = ucs_derived_of(tl_ep, uct_rc_gdaki_ep_t);
    uct_rc_gdaki_iface_t *iface  = ucs_derived_of(ep->super.super.iface,
                                                  uct_rc_gdaki_iface_t);
    uct_ib_mlx5_qp_attr_t qp_attr = {};
    uct_rc_gdaki_dev_ep_t *dev_ep;
    size_t cq_umem_offset, dev_ep_size;
    uct_rc_gdaki_channel_t *channel;
    ucs_status_t status;
    CUdeviceptr sq_db;
    unsigned i;

    pthread_mutex_lock(&iface->ep_init_lock);

    if (!ep->dev_ep_init) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(iface->cuda_ctx));
        if (status != UCS_OK) {
            goto out_unlock;
        }

        uct_rc_iface_fill_attr(&iface->super.super, &qp_attr.super,
                               iface->super.super.config.tx_qp_len, NULL);
        uct_ib_mlx5_wq_calc_sizes(&qp_attr);
        uct_rc_gdaki_calc_dev_ep_layout(iface->num_channels, &qp_attr,
                                        &cq_umem_offset, &dev_ep_size);

        dev_ep = ucs_calloc(1, qp_attr.umem_offset, "dev_ep");
        if (dev_ep == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out_ctx;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(
                cuMemsetD8((CUdeviceptr)ep->ep_gpu, 0, dev_ep_size));
        if (status != UCS_OK) {
            goto out_free;
        }

        dev_ep->atomic_va    = iface->atomic_buff;
        dev_ep->atomic_lkey  = htonl(iface->atomic_mr->lkey);
        dev_ep->sq_wqe_num   = qp_attr.max_tx;
        dev_ep->sq_fc_mask   = (qp_attr.max_tx >> 1) - 1;
        dev_ep->sq_wqe_daddr = UCS_PTR_BYTE_OFFSET(ep->ep_gpu,
                                                   qp_attr.umem_offset);

        for (i = 0; i < iface->num_channels; i++) {
            channel = ep->channels + i;

            (void)cuMemHostRegister(channel->qp.reg->addr.ptr,
                                    UCT_IB_MLX5_BF_REG_SIZE * 2,
                                    CU_MEMHOSTREGISTER_PORTABLE |
                                    CU_MEMHOSTREGISTER_DEVICEMAP |
                                    CU_MEMHOSTREGISTER_IOMEMORY);

            status = UCT_CUDADRV_FUNC_LOG_ERR(
                    cuMemHostGetDevicePointer(&sq_db,
                                              channel->qp.reg->addr.ptr, 0));
            if (status != UCS_OK) {
                goto out_unreg;
            }

            dev_ep->qps[i].sq_db  = (uint64_t *)sq_db;
            dev_ep->qps[i].sq_num = channel->qp.super.qp_num;
            memset(&dev_ep->qps[i].cq_buff, 0xff, 64);
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemcpyHtoD((CUdeviceptr)ep->ep_gpu,
                                                       dev_ep,
                                                       qp_attr.umem_offset));
        if (status != UCS_OK) {
            goto out_free;
        }

        ucs_free(dev_ep);
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));

        ep->dev_ep_init = 1;
    }

    *device_ep_p = &ep->ep_gpu->super;
    pthread_mutex_unlock(&iface->ep_init_lock);
    return UCS_OK;

out_unreg:
    do {
        (void)cuMemHostUnregister(ep->channels[i].qp.reg->addr.ptr);
    } while (i-- > 0);
out_free:
    ucs_free(dev_ep);
out_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
out_unlock:
    pthread_mutex_unlock(&iface->ep_init_lock);
    return status;
}

ucs_status_t
uct_rc_gdaki_iface_mem_element_pack(const uct_iface_h tl_iface, uct_mem_h memh,
                                    uct_rkey_t rkey,
                                    uct_device_mem_element_t *mem_elem_p)
{
    uct_rc_gdaki_device_mem_element_t mem_elem;

    mem_elem.rkey = htonl(uct_ib_md_direct_rkey(rkey));
    if (memh == NULL) {
        mem_elem.lkey = UCT_IB_INVALID_MKEY;
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
    int cuda_id;

    status = uct_rc_mlx5_dp_ordering_ooo_init(md, &self->super,
                                              md->dp_ordering_cap_devx.rc,
                                              md->ddp_support_dv.rc,
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
        return status;
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

    status = uct_rc_gdaki_alloc(sizeof(uint64_t), sizeof(uint64_t),
                                (void**)&self->atomic_buff, &self->atomic_raw);
    if (status != UCS_OK) {
        goto err_ctx;
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

    if (pthread_mutex_init(&self->ep_init_lock, NULL) != 0) {
        status = UCS_ERR_IO_ERROR;
        goto err_lock;
    }

    self->num_channels = config->num_channels;
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    return UCS_OK;

err_lock:
    ibv_dereg_mr(self->atomic_mr);
err_atomic:
    cuMemFree(self->atomic_raw);
err_ctx:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
err_ctx_release:
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(self->cuda_dev));
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_gdaki_iface_t)
{
    pthread_mutex_destroy(&self->ep_init_lock);
    ibv_dereg_mr(self->atomic_mr);
    cuMemFree(self->atomic_raw);
    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(self->cuda_dev));
}

UCS_CLASS_DEFINE(uct_rc_gdaki_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_gdaki_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_gdaki_iface_t, uct_iface_t);

static ucs_status_t
uct_gdaki_md_check_uar(uct_ib_mlx5_md_t *md, CUdevice cuda_dev)
{
    struct mlx5dv_devx_uar *uar;
    ucs_status_t status;
    CUcontext cuda_ctx;
    unsigned flags;

    status = uct_ib_mlx5_devx_alloc_uar(md, 0, &uar);
    if (status != UCS_OK) {
        goto out;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuDevicePrimaryCtxRetain(&cuda_ctx, cuda_dev));
    if (status != UCS_OK) {
        goto out_free_uar;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(cuda_ctx));
    if (status != UCS_OK) {
        goto out_ctx_release;
    }

    flags  = CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP |
             CU_MEMHOSTREGISTER_IOMEMORY;
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(
            cuMemHostRegister(uar->reg_addr, UCT_IB_MLX5_BF_REG_SIZE, flags));
    if (status == UCS_OK) {
        UCT_CUDADRV_FUNC_LOG_DEBUG(cuMemHostUnregister(uar->reg_addr));
    }

    UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
out_ctx_release:
    UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(cuda_dev));
out_free_uar:
    mlx5dv_devx_free_uar(uar);
out:
    return status;
}

static ucs_status_t
uct_gdaki_query_tl_devices(uct_md_h tl_md,
                           uct_tl_device_resource_t **tl_devices_p,
                           unsigned *num_tl_devices_p)
{
    static int uar_supported  = -1;
    static int peermem_loaded = -1;
    uct_ib_mlx5_md_t *md      = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    unsigned num_tl_devices   = 0;
    uct_tl_device_resource_t *tl_devices;
    ucs_status_t status;
    CUdevice device;
    ucs_sys_device_t dev;
    ucs_sys_dev_distance_t dist;
    int i, num_gpus;

    /*
    * Save the result of peermem driver check in a global flag to avoid
    * printing diag message for each MD.
    */
    if (peermem_loaded == -1) {
        peermem_loaded = !!(md->super.reg_mem_types &
                            UCS_BIT(UCS_MEMORY_TYPE_CUDA));
        if (peermem_loaded == 0) {
            ucs_diag("GDAKI not supported, please load "
                        "Nvidia peermem driver by running "
                        "\"modprobe nvidia_peermem\"");
        }
    }

    if (peermem_loaded == 0) {
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&num_gpus));
    if (status != UCS_OK) {
        return status;
    }

    tl_devices = ucs_malloc(sizeof(*tl_devices) * num_gpus, "gdaki_tl_devices");
    if (tl_devices == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (i = 0; i < num_gpus; i++) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&device, i));
        if (status != UCS_OK) {
            goto err;
        }

        /*
         * Save the result of UAR support in a global flag since to avoid the
         * overhead of checking UAR support for each GPU and MD. Assume the
         * support is the same for all GPUs and MDs in the system.
         */
        if (uar_supported == -1) {
            status = uct_gdaki_md_check_uar(md, device);
            if (status == UCS_OK) {
                uar_supported = 1;
            } else {
                ucs_diag("GDAKI not supported, please add "
                         "NVreg_RegistryDwords=\"PeerMappingOverride=1;\" "
                         "option for nvidia kernel driver");
                uar_supported = 0;
            }
        }
        if (uar_supported == 0) {
            status = UCS_ERR_NO_DEVICE;
            goto err;
        }

        uct_cuda_base_get_sys_dev(device, &dev);
        status = ucs_topo_get_distance(dev, md->super.dev.sys_dev, &dist);
        if (status != UCS_OK) {
            goto err;
        }

        /* TODO this logic should be done in UCP */
        if (dist.latency > md->super.config.gda_max_sys_latency) {
            continue;
        }

        snprintf(tl_devices[num_tl_devices].name,
                 sizeof(tl_devices[num_tl_devices].name), "%s%d-%s:%d",
                 UCT_DEVICE_CUDA_NAME, device,
                 uct_ib_device_name(&md->super.dev), md->super.dev.first_port);
        tl_devices[num_tl_devices].type       = UCT_DEVICE_TYPE_NET;
        tl_devices[num_tl_devices].sys_device = dev;
        num_tl_devices++;
    }

    *num_tl_devices_p = num_tl_devices;
    *tl_devices_p     = tl_devices;
    return UCS_OK;

err:
    ucs_free(tl_devices);
out:
    return status;
}

UCT_TL_DEFINE_ENTRY(&uct_ib_component, rc_gda, uct_gdaki_query_tl_devices,
                    uct_rc_gdaki_iface_t, "RC_GDA_",
                    uct_rc_gdaki_iface_config_table,
                    uct_rc_gdaki_iface_config_t);

UCT_TL_INIT(&uct_ib_component, rc_gda, ctor, , )
