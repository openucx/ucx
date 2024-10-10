/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/base/uct_iface.h>
#include <uct/ib/base/ib_md.h>
#include <uct/ib/mlx5/rc/rc_mlx5.h>

#include <uct/ib/mlx5/rc/rc_mlx5.inl>

#define UCT_GGA_MLX5_OPAQUE_BUF_LEN 64
#define UCT_GGA_MAX_MSG_SIZE        (2u * UCS_MBYTE)

typedef struct {
    uct_ib_md_packed_mkey_t packed_mkey;
    uct_ib_mlx5_devx_mem_t  *memh;
    uct_ib_mlx5_md_t        *md;
    uct_rkey_bundle_t       rkey_ob;
} uct_gga_mlx5_rkey_handle_t;

typedef struct {
    uct_rc_mlx5_iface_common_t  super;
} uct_gga_mlx5_iface_t;

typedef struct {
    uct_rc_iface_config_t             super;
    uct_rc_mlx5_iface_common_config_t rc_mlx5_common;
} uct_gga_mlx5_iface_config_t;

typedef struct {
    uct_ib_mlx5_dma_opaque_mr_t opaque_mr;
    void                        *buf;
    struct ibv_mr               *mr;
} uct_gga_mlx5_dma_opaque_buf_t;

typedef struct {
    uct_rc_mlx5_base_ep_t         super;
    uct_gga_mlx5_dma_opaque_buf_t dma_opaque;
} uct_gga_mlx5_ep_t;

enum {
    UCT_GGA_MLX5_EP_ADDRESS_FLAG_FLUSH_RKEY = UCS_BIT(0)
};

typedef struct {
    uint8_t         flags;
    uct_ib_uint24_t qp_num;
    uct_ib_uint24_t flush_rkey;
} UCS_S_PACKED uct_gga_mlx5_ep_address_t;

typedef struct {
    uint64_t be_sys_image_guid; /* ID of GVMI. */
} UCS_S_PACKED uct_gga_mlx5_iface_addr_t;

extern ucs_config_field_t uct_ib_md_config_table[];

ucs_config_field_t uct_gga_mlx5_iface_config_table[] = {
  {"GGA_", "", NULL,
   ucs_offsetof(uct_gga_mlx5_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"GGA_", "", NULL,
   ucs_offsetof(uct_gga_mlx5_iface_config_t, rc_mlx5_common),
   UCS_CONFIG_TYPE_TABLE(uct_rc_mlx5_common_config_table)},

  {NULL}
};

static ucs_status_t
uct_ib_mlx5_gga_mkey_pack(uct_md_h uct_md, uct_mem_h uct_memh,
                          void *address, size_t length,
                          const uct_md_mkey_pack_params_t *params,
                          void *mkey_buffer)
{
    uct_md_mkey_pack_params_t gga_params = *params;

    if (gga_params.field_mask & UCT_MD_MKEY_PACK_FIELD_FLAGS) {
        gga_params.flags      |= UCT_MD_MKEY_PACK_FLAG_EXPORT;
    } else {
        gga_params.field_mask |= UCT_MD_MKEY_PACK_FIELD_FLAGS;
        gga_params.flags       = UCT_MD_MKEY_PACK_FLAG_EXPORT;
    }

    return uct_ib_mlx5_devx_mkey_pack(uct_md, uct_memh, address, length,
                                      &gga_params, mkey_buffer);
}

static ucs_status_t
uct_gga_mlx5_rkey_unpack(uct_component_t *component, const void *rkey_buffer,
                         uct_rkey_t *rkey_p, void **handle_p)
{
    const uct_ib_md_packed_mkey_t *mkey = rkey_buffer;
    uct_gga_mlx5_rkey_handle_t *rkey_handle;

    rkey_handle = ucs_malloc(sizeof(*rkey_handle), "gga_rkey_handle");
    if (rkey_handle == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    rkey_handle->packed_mkey    = *mkey;
    /* memh and rkey_ob will be initialized on demand since MD
     * (required for memh import) is not available at this point */
    rkey_handle->memh           = NULL;
    rkey_handle->md             = NULL;
    rkey_handle->rkey_ob.rkey   = UCT_INVALID_RKEY;
    rkey_handle->rkey_ob.handle = NULL;
    rkey_handle->rkey_ob.type   = NULL;

    *rkey_p   = (uintptr_t)rkey_handle;
    *handle_p = NULL;
    return UCS_OK;
}

static void
uct_gga_mlx5_rkey_handle_dereg(uct_gga_mlx5_rkey_handle_t *rkey_handle)
{
    uct_md_mem_dereg_params_t params = {
        .field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH,
        .memh       = rkey_handle->memh
    };
    ucs_status_t status;

    if (rkey_handle->memh == NULL) {
        return;
    }

    status = uct_ib_mlx5_devx_mem_dereg(&rkey_handle->md->super.super, &params);
    if (status != UCS_OK) {
        ucs_warn("md %p: failed to deregister GGA memh %p", rkey_handle->md,
                 rkey_handle->memh);
    }

    rkey_handle->memh = NULL;
    rkey_handle->md   = NULL;
}

static ucs_status_t uct_gga_mlx5_rkey_release(uct_component_t *component,
                                              uct_rkey_t rkey, void *handle)
{
    uct_gga_mlx5_rkey_handle_t *rkey_handle = (uct_gga_mlx5_rkey_handle_t*)rkey;

    uct_gga_mlx5_rkey_handle_dereg(rkey_handle);
    ucs_free(rkey_handle);
    return UCS_OK;
}

/* Forward declaration */
static ucs_status_t
uct_ib_mlx5_gga_md_open(uct_component_t *component, const char *md_name,
                        const uct_md_config_t *uct_md_config, uct_md_h *md_p);

static uct_component_t uct_gga_component = {
    .query_md_resources = uct_ib_query_md_resources,
    .md_open            = uct_ib_mlx5_gga_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_gga_mlx5_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = uct_gga_mlx5_rkey_release,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "gga",
    .md_config          = {
        .name           = "GGA memory domain",
        .prefix         = "GGA_",
        .table          = uct_ib_md_config_table,
        .size           = sizeof(uct_ib_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_gga_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};

static UCS_F_ALWAYS_INLINE
void uct_gga_mlx5_rkey_trace(uct_ib_mlx5_md_t *md,
                             uct_gga_mlx5_rkey_handle_t *rkey_handle,
                             const char *prefix)
{
    ucs_trace("md %p: %s resolved rkey %p: rkey_ob %"PRIx64"/%p", md, prefix,
              rkey_handle, rkey_handle->rkey_ob.rkey,
              rkey_handle->rkey_ob.handle);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_gga_mlx5_rkey_resolve(uct_ib_mlx5_md_t *md,
                          uct_gga_mlx5_rkey_handle_t *rkey_handle)
{
    uct_md_h uct_md                         = &md->super.super;
    uct_md_mem_attach_params_t atach_params = { 0 };
    uct_md_mkey_pack_params_t repack_params = { 0 };
    uint64_t repack_mkey;
    ucs_status_t status;

    if (ucs_likely(rkey_handle->memh != NULL)) {
        uct_gga_mlx5_rkey_trace(md, rkey_handle, "reuse");
        return UCS_OK;
    }

    status = uct_ib_mlx5_devx_mem_attach(uct_md, &rkey_handle->packed_mkey,
                                         &atach_params,
                                         (uct_mem_h *)&rkey_handle->memh);
    if (status != UCS_OK) {
        goto err_out;
    }

    rkey_handle->md = md;

    status = uct_ib_mlx5_devx_mkey_pack(uct_md, (uct_mem_h)rkey_handle->memh,
                                        NULL, 0, &repack_params, &repack_mkey);
    if (status != UCS_OK) {
        goto err_dereg;
    }

    status = uct_ib_rkey_unpack(NULL, &repack_mkey,
                                &rkey_handle->rkey_ob.rkey,
                                &rkey_handle->rkey_ob.handle);
    uct_gga_mlx5_rkey_trace(md, rkey_handle, "new");
    return status;

err_dereg:
    uct_gga_mlx5_rkey_handle_dereg(rkey_handle);
err_out:
    return status;
}

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_gga_mlx5_iface_t, uct_iface_t);

static unsigned uct_gga_mlx5_iface_progress(uct_iface_h iface)
{
    uct_rc_mlx5_iface_common_t *rc_iface =
            ucs_derived_of(iface, uct_rc_mlx5_iface_common_t);

    return uct_rc_mlx5_iface_poll_tx(rc_iface, UCT_IB_MLX5_POLL_FLAG_HAS_EP);
}

static ucs_status_t
uct_gga_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
    ucs_status_t status;

    status = uct_ib_iface_query(&iface->super, UCT_IB_RETH_LEN, iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->ep_addr_len    = sizeof(uct_gga_mlx5_ep_address_t);
    iface_attr->iface_addr_len = sizeof(uct_gga_mlx5_iface_addr_t);
    iface_attr->cap.flags      = UCT_IFACE_FLAG_PUT_ZCOPY |
                                 UCT_IFACE_FLAG_GET_ZCOPY |
                                 UCT_IFACE_FLAG_PENDING |
                                 UCT_IFACE_FLAG_CONNECT_TO_EP |
                                 UCT_IFACE_FLAG_CB_SYNC |
                                 UCT_IFACE_FLAG_INTER_NODE;

    iface_attr->cap.event_flags = UCT_IFACE_FLAG_EVENT_SEND_COMP |
                                  UCT_IFACE_FLAG_EVENT_FD;

    iface_attr->cap.put.min_zcopy       = 1;
    iface_attr->cap.put.max_zcopy       = UCT_GGA_MAX_MSG_SIZE;
    iface_attr->cap.put.max_iov         = 1;
    iface_attr->cap.put.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;

    iface_attr->cap.get.min_zcopy       = 1;
    iface_attr->cap.get.max_zcopy       = iface->config.max_get_zcopy;
    iface_attr->cap.get.max_iov         = 1;
    iface_attr->cap.get.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;

    iface_attr->latency.c += 200e-9;

    return UCS_OK;
}

static ucs_status_t uct_gga_mlx5_ep_enable_mmo(uct_gga_mlx5_ep_t *ep)
{
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(init2init_qp_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(init2init_qp_out)] = {};
    void *qpce = UCT_IB_MLX5DV_ADDR_OF(init2init_qp_in, in, qpc_data_extension);

    UCT_IB_MLX5DV_SET(init2init_qp_in, in, opcode,
                      UCT_IB_MLX5_CMD_OP_INIT2INIT_QP);
    UCT_IB_MLX5DV_SET(init2init_qp_in, in, qpc_ext, 1);
    UCT_IB_MLX5DV_SET(init2init_qp_in, in, qpn, ep->super.tx.wq.super.qp_num);
    UCT_IB_MLX5DV_SET64(init2init_qp_in, in, opt_param_mask_95_32,
                        UCT_IB_MLX5_QPC_OPT_MASK_32_INIT2INIT_MMO);
    UCT_IB_MLX5DV_SET(qpc_ext, qpce, mmo, 1);

    return uct_ib_mlx5_devx_obj_modify(ep->super.tx.wq.super.devx.obj, in,
                                       sizeof(in), out, sizeof(out),
                                       "2INIT_QP_MMO");
}

static UCS_CLASS_INIT_FUNC(uct_gga_mlx5_ep_t, const uct_ep_params_t *params)
{
    uct_iface_t *tl_iface   = UCT_EP_PARAM_VALUE(params, iface, IFACE, NULL);
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    uct_ib_mlx5_md_t *md    = ucs_derived_of(iface->md, uct_ib_mlx5_md_t);
    int ret;
    ucs_status_t status;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_base_ep_t, params);

    ret = ucs_posix_memalign(&self->dma_opaque.buf, UCS_SYS_CACHE_LINE_SIZE,
                             UCT_GGA_MLX5_OPAQUE_BUF_LEN, "gga_dma_opaque_buf");
    if (ret != 0) {
        ucs_error("failed to allocate %u bytes DMA/MMO opaque buffer",
                  UCS_SYS_CACHE_LINE_SIZE);
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    self->dma_opaque.mr = ibv_reg_mr(md->super.pd, self->dma_opaque.buf,
                                     UCT_GGA_MLX5_OPAQUE_BUF_LEN,
                                     IBV_ACCESS_LOCAL_WRITE);

    if (self->dma_opaque.mr == NULL) {
        ucs_error("ibv_reg_mr(pd=%p, buf=%p, len=%d, 0x%x) failed to register "
                  "DMA/MMO opaque buffer: %m", md->super.pd,
                  self->dma_opaque.buf, UCT_GGA_MLX5_OPAQUE_BUF_LEN,
                  IBV_ACCESS_LOCAL_WRITE);
        status = UCS_ERR_IO_ERROR;
        goto err_free_buf;
    }

    self->dma_opaque.opaque_mr.be_vaddr =
            htobe64((uintptr_t)self->dma_opaque.mr->addr);
    self->dma_opaque.opaque_mr.be_lkey  =
            htobe32(self->dma_opaque.mr->lkey);

    status = uct_gga_mlx5_ep_enable_mmo(self);
    if (status != UCS_OK) {
        goto err_dereg_buf;
    }

    return UCS_OK;

err_dereg_buf:
    uct_ib_dereg_mr(self->dma_opaque.mr);
err_free_buf:
    ucs_free(self->dma_opaque.buf);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gga_mlx5_ep_t)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(
            self->super.super.super.super.iface, uct_rc_mlx5_iface_common_t);
    uct_ib_md_t *ib_md                = uct_ib_iface_md(&iface->super.super);
    uct_ib_mlx5_md_t *md              = ucs_derived_of(ib_md, uct_ib_mlx5_md_t);
    uint16_t outstanding;
    uint16_t wqe_count;

    outstanding = self->super.tx.wq.bb_max - self->super.super.txqp.available;
    wqe_count   = uct_ib_mlx5_txwq_num_posted_wqes(&self->super.tx.wq,
                                                   outstanding);
    ucs_assert(outstanding >= wqe_count);

    uct_ib_dereg_mr(self->dma_opaque.mr);
    ucs_free(self->dma_opaque.buf);
    uct_rc_txqp_purge_outstanding(&iface->super, &self->super.super.txqp,
                                  UCS_ERR_CANCELED, self->super.tx.wq.sw_pi, 1);
    uct_rc_iface_remove_qp(&iface->super, self->super.tx.wq.super.qp_num);
    uct_ib_mlx5_destroy_qp(md, &self->super.tx.wq.super);
    uct_ib_mlx5_qp_mmio_cleanup(&self->super.tx.wq.super, self->super.tx.wq.reg);
    ucs_list_del(&self->super.super.list);
    uct_rc_iface_add_cq_credits(&iface->super, outstanding - wqe_count);
}

UCS_CLASS_DEFINE(uct_gga_mlx5_ep_t, uct_rc_mlx5_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_gga_mlx5_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_gga_mlx5_ep_t, uct_ep_t);

static ucs_status_t
uct_gga_mlx5_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_gga_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_gga_mlx5_ep_t);
    uct_rc_mlx5_iface_common_t *iface =
            ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_common_t);
    uct_gga_mlx5_ep_address_t *gga_addr = (uct_gga_mlx5_ep_address_t*)addr;
    uct_ib_md_t *md = uct_ib_iface_md(&iface->super.super);

    uct_ib_pack_uint24(gga_addr->qp_num, ep->super.tx.wq.super.qp_num);
    if (uct_rc_iface_flush_rkey_enabled(&iface->super)) {
        gga_addr->flags = UCT_GGA_MLX5_EP_ADDRESS_FLAG_FLUSH_RKEY;
        uct_ib_pack_uint24(gga_addr->flush_rkey, md->flush_rkey >> 8);
    } else {
        gga_addr->flags = 0;
        uct_ib_pack_uint24(gga_addr->flush_rkey, 0);
    }

    return UCS_OK;
}

static ucs_status_t
uct_gga_mlx5_ep_connect_to_ep_v2(uct_ep_h tl_ep,
                                 const uct_device_addr_t *device_addr,
                                 const uct_ep_addr_t *ep_addr,
                                 const uct_ep_connect_to_ep_params_t *params)
{
    uct_gga_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_gga_mlx5_ep_t);
    uct_rc_mlx5_iface_common_t *iface =
            ucs_derived_of(tl_ep->iface, uct_rc_mlx5_iface_common_t);
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t*)device_addr;
    const uct_gga_mlx5_ep_address_t *gga_ep_addr =
            (const uct_gga_mlx5_ep_address_t*)ep_addr;
    uint32_t qp_num;
    struct ibv_ah_attr ah_attr;
    enum ibv_mtu path_mtu;
    ucs_status_t status;

    uct_ib_iface_fill_ah_attr_from_addr(&iface->super.super, ib_addr,
                                        ep->super.super.path_index, &ah_attr,
                                        &path_mtu);
    ucs_assert(path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);

    qp_num = uct_ib_unpack_uint24(gga_ep_addr->qp_num);
    status = uct_rc_mlx5_iface_common_devx_connect_qp(
            iface, &ep->super.tx.wq.super, qp_num, &ah_attr, path_mtu,
            ep->super.super.path_index, iface->super.config.max_rd_atomic);
    if (status != UCS_OK) {
        return status;
    }

    ep->super.super.atomic_mr_offset = 0;
    ep->super.super.flags           |= UCT_RC_EP_FLAG_CONNECTED;
    ep->super.super.flush_rkey       =
            (gga_ep_addr->flags & UCT_GGA_MLX5_EP_ADDRESS_FLAG_FLUSH_RKEY) ?
            (uct_ib_unpack_uint24(gga_ep_addr->flush_rkey) << 8) :
            UCT_IB_MD_INVALID_FLUSH_RKEY;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_gga_mlx5_ep_fence_put(uct_rc_mlx5_iface_common_t *iface,
                          uct_ib_mlx5_txwq_t *txwq, uct_rkey_t *rkey,
                          uint64_t *addr, uint8_t *fm_ce_se)
{
    if (ucs_unlikely(uct_rc_ep_fm(&iface->super, &txwq->fi, 1))) {
        *rkey      = uct_ib_resolve_atomic_rkey(*rkey, 0, addr);
        *fm_ce_se |= UCT_IB_MLX5_WQE_CTRL_FLAG_STRONG_ORDER;
    } else {
        *rkey = uct_ib_md_direct_rkey(*rkey);
    }
}

static ucs_status_t
uct_gga_mlx5_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                          uint64_t remote_addr, uct_rkey_t rkey,
                          uct_completion_t *comp)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uct_gga_mlx5_ep_t *gga_ep = ucs_derived_of(ep, uct_gga_mlx5_ep_t);
    uct_ib_mlx5_md_t *md      = ucs_derived_of(iface->super.super.super.md,
                                               uct_ib_mlx5_md_t);
    uct_gga_mlx5_rkey_handle_t *rkey_handle = (uct_gga_mlx5_rkey_handle_t*)rkey;
    uint8_t fm_ce_se                        = MLX5_WQE_CTRL_CQ_UPDATE;
    uct_rkey_t rkey_copy;
    ucs_status_t status;

    UCT_CHECK_LENGTH(uct_iov_total_length(iov, iovcnt), 1ul,
                     UCT_GGA_MAX_MSG_SIZE, "put_zcopy");

    /* rkey resolution doesn't depend on available resources */
    status = uct_gga_mlx5_rkey_resolve(md, rkey_handle);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    rkey_copy = rkey_handle->rkey_ob.rkey;
    uct_gga_mlx5_ep_fence_put(iface, &ep->tx.wq, &rkey_copy, &remote_addr,
                              &fm_ce_se);

    status = uct_rc_mlx5_base_ep_zcopy_post(
            ep, MLX5_OPCODE_MMO | UCT_RC_MLX5_OPCODE_FLAG_MMO_PUT, iov, iovcnt,
            0ul, 0, NULL, 0, remote_addr, rkey_copy, 0ul, 0, 0,
            &gga_ep->dma_opaque.opaque_mr, fm_ce_se,
            uct_rc_ep_send_op_completion_handler, 0, comp);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, ZCOPY,
                                 uct_iov_total_length(iov, iovcnt));
    uct_rc_ep_enable_flush_remote(&ep->super);
    return status;
}

static void
uct_gga_mlx5_ep_get_zcopy_completion_handler(uct_rc_iface_send_op_t *op,
                                             const void *resp)
{
    uct_rc_op_release_iov_get_zcopy(op);
    uct_rc_ep_send_op_completion_handler(op, resp);
}

static ucs_status_t
uct_gga_mlx5_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                          uint64_t remote_addr, uct_rkey_t rkey,
                          uct_completion_t *comp)
{
    UCT_RC_MLX5_BASE_EP_DECL(tl_ep, iface, ep);
    uct_gga_mlx5_ep_t *gga_ep = ucs_derived_of(ep, uct_gga_mlx5_ep_t);
    uct_ib_mlx5_md_t *md      = ucs_derived_of(iface->super.super.super.md,
                                               uct_ib_mlx5_md_t);
    uct_gga_mlx5_rkey_handle_t *rkey_handle = (uct_gga_mlx5_rkey_handle_t*)rkey;
    size_t total_length                     = uct_iov_total_length(iov, iovcnt);
    uint8_t fm_ce_se                        = MLX5_WQE_CTRL_CQ_UPDATE;
    uct_rkey_t rkey_copy;
    ucs_status_t status;

    UCT_CHECK_LENGTH(total_length, 1ul, iface->super.config.max_get_zcopy,
                     "get_zcopy");

    /* rkey resolution doesn't depend on available resources */
    status = uct_gga_mlx5_rkey_resolve(md, rkey_handle);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    rkey_handle = (uct_gga_mlx5_rkey_handle_t*)rkey;
    rkey_copy   = rkey_handle->rkey_ob.rkey;
    uct_rc_mlx5_ep_fence_get(iface, &ep->tx.wq, &rkey_copy, &fm_ce_se);
    status = uct_rc_mlx5_base_ep_zcopy_post(
            ep, MLX5_OPCODE_MMO | UCT_RC_MLX5_OPCODE_FLAG_MMO_GET, iov, iovcnt,
            total_length, 0, NULL, 0, remote_addr, rkey_copy, 0ul, 0, 0,
            &gga_ep->dma_opaque.opaque_mr, fm_ce_se,
            uct_gga_mlx5_ep_get_zcopy_completion_handler,
            UCT_RC_IFACE_SEND_OP_FLAG_IOV, comp);

    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY, total_length);
    }

    return status;
}

static ucs_status_t
uct_gga_mlx5_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *addr)
{
    uct_gga_mlx5_iface_addr_t *gga_addr = (uct_gga_mlx5_iface_addr_t*)addr;
    uct_ib_iface_t *iface = ucs_derived_of(tl_iface, uct_ib_iface_t);
    uct_ib_md_t *md = uct_ib_iface_md(iface);

    gga_addr->be_sys_image_guid = md->dev.dev_attr.orig_attr.sys_image_guid;
    return UCS_OK;
}

static uct_iface_ops_t uct_gga_mlx5_iface_tl_ops = {
    .ep_put_short             = ucs_empty_function_return_unsupported,
    .ep_put_bcopy             = (uct_ep_put_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_put_zcopy             = uct_gga_mlx5_ep_put_zcopy,
    .ep_get_bcopy             = ucs_empty_function_return_unsupported,
    .ep_get_zcopy             = uct_gga_mlx5_ep_get_zcopy,
    .ep_am_short              = (uct_ep_am_short_func_t)ucs_empty_function_return_unsupported,
    .ep_am_short_iov          = (uct_ep_am_short_iov_func_t)ucs_empty_function_return_unsupported,
    .ep_am_bcopy              = (uct_ep_am_bcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_am_zcopy              = (uct_ep_am_zcopy_func_t)ucs_empty_function_return_unsupported,
    .ep_atomic_cswap64        = ucs_empty_function_return_unsupported,
    .ep_atomic_cswap32        = ucs_empty_function_return_unsupported,
    .ep_atomic64_post         = ucs_empty_function_return_unsupported,
    .ep_atomic32_post         = ucs_empty_function_return_unsupported,
    .ep_atomic64_fetch        = ucs_empty_function_return_unsupported,
    .ep_atomic32_fetch        = ucs_empty_function_return_unsupported,
    .ep_pending_add           = uct_rc_ep_pending_add,
    .ep_pending_purge         = uct_rc_ep_pending_purge,
    .ep_flush                 = uct_rc_mlx5_base_ep_flush,
    .ep_fence                 = uct_rc_mlx5_base_ep_fence,
    .ep_check                 = ucs_empty_function_return_unsupported,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_gga_mlx5_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_gga_mlx5_ep_t),
    .ep_get_address           = uct_gga_mlx5_ep_get_address,
    .ep_connect_to_ep         = uct_base_ep_connect_to_ep,
    .iface_flush              = uct_rc_iface_flush,
    .iface_fence              = uct_rc_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_gga_mlx5_iface_progress,
    .iface_event_fd_get       = uct_rc_mlx5_iface_event_fd_get,
    .iface_event_arm          = uct_rc_mlx5_iface_arm,
    .iface_close              = uct_gga_mlx5_iface_t_delete,
    .iface_query              = uct_gga_mlx5_iface_query,
    .iface_get_address        = uct_gga_mlx5_iface_get_address,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_base_iface_is_reachable
};

static int
uct_gga_mlx5_iface_is_reachable_v2(const uct_iface_h tl_iface,
                                   const uct_iface_is_reachable_params_t *params)
{
    uct_ib_iface_t *iface   = ucs_derived_of(tl_iface, uct_ib_iface_t);
    uct_ib_device_t *device = uct_ib_iface_device(iface);
    const uct_gga_mlx5_iface_addr_t *iface_addr =
            (const uct_gga_mlx5_iface_addr_t*)
            UCS_PARAM_VALUE(UCT_IFACE_IS_REACHABLE_FIELD, params, iface_addr,
                            IFACE_ADDR, NULL);

    return (iface_addr != NULL) &&
           (iface_addr->be_sys_image_guid ==
            device->dev_attr.orig_attr.sys_image_guid) &&
           uct_ib_iface_is_reachable_v2(tl_iface, params);
}

static uct_rc_iface_ops_t uct_gga_mlx5_iface_ops = {
    .super = {
        .super = {
            .iface_estimate_perf   = uct_rc_iface_estimate_perf,
            .iface_vfs_refresh     = uct_rc_iface_vfs_refresh,
            .ep_query              = (uct_ep_query_func_t)ucs_empty_function,
            .ep_invalidate         = uct_rc_mlx5_base_ep_invalidate,
            .ep_connect_to_ep_v2   = uct_gga_mlx5_ep_connect_to_ep_v2,
            .iface_is_reachable_v2 = uct_gga_mlx5_iface_is_reachable_v2,
            .ep_is_connected       = ucs_empty_function_do_assert
        },
        .create_cq      = uct_rc_mlx5_iface_common_create_cq,
        .destroy_cq     = uct_rc_mlx5_iface_common_destroy_cq,
        .event_cq       = uct_rc_mlx5_iface_common_event_cq,
        .handle_failure = uct_rc_mlx5_iface_handle_failure,
    },
    .init_rx         = ucs_empty_function_return_success,
    .cleanup_rx      = ucs_empty_function,
    .fc_ctrl         = ucs_empty_function_return_unsupported,
    .fc_handler      = (uct_rc_iface_fc_handler_func_t)ucs_empty_function_do_assert,
    .cleanup_qp      = ucs_empty_function_do_assert_void,
    .ep_post_check   = uct_rc_mlx5_base_ep_post_check,
    .ep_vfs_populate = uct_rc_mlx5_base_ep_vfs_populate
};

static void uct_gga_mlx5_iface_disable_rx(uct_rc_mlx5_iface_common_t *iface)
{
    iface->super.rx.srq.quota = 0;
    iface->rx.srq.type        = UCT_IB_MLX5_OBJ_TYPE_NULL;
    iface->rx.srq.srq_num     = 0;
}

static UCS_CLASS_INIT_FUNC(uct_gga_mlx5_iface_t,
                           uct_md_h tl_md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_gga_mlx5_iface_config_t *config =
            ucs_derived_of(tl_config, uct_gga_mlx5_iface_config_t);
    uct_ib_mlx5_md_t *md                = ucs_derived_of(tl_md, uct_ib_mlx5_md_t);
    uct_ib_iface_init_attr_t init_attr  = {};
    ucs_status_t status;

    init_attr.qp_type               = IBV_QPT_RC;
    init_attr.cq_len[UCT_IB_DIR_TX] = config->super.tx_cq_len;
    init_attr.max_rd_atomic         = IBV_DEV_ATTR(&md->super.dev,
                                                   max_qp_rd_atom);
    init_attr.tx_moderation         = config->super.tx_cq_moderation;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_mlx5_iface_common_t,
                              &uct_gga_mlx5_iface_tl_ops,
                              &uct_gga_mlx5_iface_ops, tl_md, worker, params,
                              &config->super.super, &config->rc_mlx5_common,
                              &init_attr);

    status = uct_rc_mlx5_dp_ordering_ooo_init(
            &self->super, UCT_IB_MLX5_MD_FLAG_DP_ORDERING_OOO_RW_RC,
            &config->rc_mlx5_common, "gga");
    if (status != UCS_OK) {
        return status;
    }

    uct_gga_mlx5_iface_disable_rx(&self->super);

    config->super.super.fc.enable        = 0; /* FC requires AM capability */
    self->super.config.atomic_fence_flag = UCT_IB_MLX5_WQE_CTRL_FLAG_FENCE;
    self->super.super.config.fence_mode  = UCT_RC_FENCE_MODE_AUTO;

    uct_rc_iface_adjust_max_get_zcopy(&self->super.super, &config->super.super,
            UCT_GGA_MAX_MSG_SIZE,
            uct_ib_device_name(uct_ib_iface_device(&self->super.super.super)),
            "gga");

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_gga_mlx5_iface_t)
{
    uct_base_iface_progress_disable(&self->super.super.super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
}

UCS_CLASS_DEFINE(uct_gga_mlx5_iface_t, uct_rc_mlx5_iface_common_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_gga_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_gga_mlx5_iface_t, uct_iface_t);

static ucs_status_t
uct_gga_mlx5_query_tl_devices(uct_md_h md,
                              uct_tl_device_resource_t **tl_devices_p,
                              unsigned *num_tl_devices_p)
{
    uct_ib_mlx5_md_t *mlx5_md = ucs_derived_of(md, uct_ib_mlx5_md_t);

    if (strcmp(mlx5_md->super.name, UCT_IB_MD_NAME(gga)) ||
        !ucs_test_all_flags(mlx5_md->flags, UCT_IB_MLX5_MD_FLAG_DEVX |
                                            UCT_IB_MLX5_MD_FLAG_INDIRECT_XGVMI |
                                            UCT_IB_MLX5_MD_FLAG_MMO_DMA)) {
        return UCS_ERR_NO_DEVICE;
    }

    return uct_ib_device_query_ports(&mlx5_md->super.dev,
                                     UCT_IB_DEVICE_FLAG_SRQ |
                                             UCT_IB_DEVICE_FLAG_MLX5_PRM,
                                     tl_devices_p, num_tl_devices_p);
}

UCT_TL_DEFINE_ENTRY(&uct_gga_component, gga_mlx5, uct_gga_mlx5_query_tl_devices,
                    uct_gga_mlx5_iface_t, "GGA_MLX5_",
                    uct_gga_mlx5_iface_config_table,
                    uct_gga_mlx5_iface_config_t);

UCT_SINGLE_TL_INIT(&uct_gga_component, gga_mlx5, ctor,,)

/* TODO: separate memh since atomic_mr is not relevant for GGA */
static uct_md_ops_t uct_mlx5_gga_md_ops = {
    .close              = uct_ib_mlx5_devx_md_close,
    .query              = uct_ib_mlx5_devx_md_query,
    .mem_alloc          = uct_ib_mlx5_devx_device_mem_alloc,
    .mem_free           = uct_ib_mlx5_devx_device_mem_free,
    .mem_reg            = uct_ib_mlx5_devx_mem_reg,
    .mem_dereg          = uct_ib_mlx5_devx_mem_dereg,
    .mem_attach         = uct_ib_mlx5_devx_mem_attach,
    .mem_advise         = uct_ib_mem_advise,
    .mkey_pack          = uct_ib_mlx5_gga_mkey_pack,
    .detect_memory_type = ucs_empty_function_return_unsupported,
};

static ucs_status_t
uct_ib_mlx5_gga_md_open(uct_component_t *component, const char *md_name,
                        const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    const uct_ib_md_config_t *md_config = ucs_derived_of(uct_md_config,
                                                         uct_ib_md_config_t);
    struct ibv_device **ib_device_list, *ib_device;
    int num_devices, fork_init;
    ucs_status_t status;
    uct_ib_md_t *md;

    ucs_trace("opening GGA device %s", md_name);

    if (md_config->devx == UCS_NO) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* Get device list from driver */
    ib_device_list = ibv_get_device_list(&num_devices);
    if (ib_device_list == NULL) {
        ucs_debug("Failed to get GGA device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    status = uct_ib_get_device_by_name(ib_device_list, num_devices, md_name,
                                       &ib_device);
    if (status != UCS_OK) {
        goto out_free_dev_list;
    }

    status = uct_ib_fork_init(md_config, &fork_init);
    if (status != UCS_OK) {
        goto out_free_dev_list;
    }

    status = uct_ib_mlx5_devx_md_open(ib_device, md_config, &md);
    if (status != UCS_OK) {
        goto out_free_dev_list;
    }

    md->super.component = &uct_gga_component;
    md->super.ops       = &uct_mlx5_gga_md_ops;
    md->name            = UCT_IB_MD_NAME(gga);
    md->fork_init       = fork_init;
    *md_p               = &md->super;

out_free_dev_list:
    ibv_free_device_list(ib_device_list);
out:
    return status;
}
