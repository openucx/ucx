/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_mlx5.inl"

#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <ucs/async/async.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/mlx5/dv/ib_mlx5_ifc.h>

ucs_status_t uct_rc_mlx5_devx_iface_subscribe_event(uct_rc_mlx5_iface_common_t *iface,
                                                    uct_ib_mlx5_qp_t *qp,
                                                    unsigned event_num,
                                                    enum ibv_event_type event_type,
                                                    unsigned event_data)
{
#if HAVE_DECL_MLX5DV_DEVX_SUBSCRIBE_DEVX_EVENT
    uint64_t cookie;
    uint16_t event;
    int ret;

    if (iface->event_channel == NULL) {
        return UCS_OK;
    }

    event  = event_num;
    cookie = event_type | ((uint64_t)event_data << UCT_IB_MLX5_DEVX_EVENT_DATA_SHIFT);
    ret    = mlx5dv_devx_subscribe_devx_event(iface->event_channel, qp->devx.obj,
                                              sizeof(event), &event, cookie);
    if (ret) {
        ucs_error("mlx5dv_devx_subscribe_devx_event() failed: %m");
        return UCS_ERR_IO_ERROR;
    }
#endif

    return UCS_OK;
}

#if HAVE_DECL_MLX5DV_DEVX_SUBSCRIBE_DEVX_EVENT
static void
uct_rc_mlx5_devx_iface_event_handler(int fd, ucs_event_set_types_t events,
                                     void *arg)
{
    uct_rc_mlx5_iface_common_t *iface = arg;
    uct_ib_md_t *md                   = uct_ib_iface_md(&iface->super.super);
    struct mlx5dv_devx_async_event_hdr devx_event;
    uct_ib_async_event_t event;
    int ret;

    ret = mlx5dv_devx_get_event(iface->event_channel, &devx_event, sizeof(devx_event));
    if (ret < 0) {
        if (errno != EAGAIN) {
            ucs_warn("mlx5dv_devx_get_event() failed: %m");
        }
        return;
    }

    event.event_type = (enum ibv_event_type)(devx_event.cookie &
                                             UCT_IB_MLX5_DEVX_EVENT_TYPE_MASK);
    switch (event.event_type) {
    case IBV_EVENT_QP_LAST_WQE_REACHED:
        event.qp_num = devx_event.cookie >> UCT_IB_MLX5_DEVX_EVENT_DATA_SHIFT;
        break;
    default:
        ucs_warn("unexpected async event: %d", event.event_type);
        return;
    }

    uct_ib_handle_async_event(&md->dev, &event);
}
#endif

ucs_status_t uct_rc_mlx5_devx_iface_init_events(uct_rc_mlx5_iface_common_t *iface)
{
    ucs_status_t status   = UCS_OK;
#if HAVE_DECL_MLX5DV_DEVX_SUBSCRIBE_DEVX_EVENT
    uct_ib_mlx5_md_t *md  = ucs_derived_of(uct_ib_iface_md(&iface->super.super),
                                           uct_ib_mlx5_md_t);
    struct mlx5dv_devx_event_channel *event_channel;

    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_DEVX) || !md->super.dev.async_events) {
        iface->event_channel = NULL;
        return UCS_OK;
    }

    event_channel = mlx5dv_devx_create_event_channel(
            md->super.dev.ibv_context,
            MLX5_IB_UAPI_DEVX_CR_EV_CH_FLAGS_OMIT_DATA);

    if (event_channel == NULL) {
        ucs_error("mlx5dv_devx_create_event_channel() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;

    }

    status = ucs_sys_fcntl_modfl(event_channel->fd, O_NONBLOCK, 0);
    if (status != UCS_OK) {
        goto err_destroy_channel;
    }

    status = ucs_async_set_event_handler(iface->super.super.super.worker->async->mode,
                                         event_channel->fd, UCS_EVENT_SET_EVREAD,
                                         uct_rc_mlx5_devx_iface_event_handler, iface,
                                         iface->super.super.super.worker->async);
    if (status != UCS_OK) {
        goto err_destroy_channel;
    }

    iface->event_channel = event_channel;
    return UCS_OK;

err_destroy_channel:
    mlx5dv_devx_destroy_event_channel(event_channel);
    iface->event_channel = NULL;
err:
#endif
    return status;
}

void uct_rc_mlx5_devx_iface_free_events(uct_rc_mlx5_iface_common_t *iface)
{
#if HAVE_DECL_MLX5DV_DEVX_SUBSCRIBE_DEVX_EVENT
    if (iface->event_channel == NULL) {
        return;
    }

    ucs_async_remove_handler(iface->event_channel->fd, 1);
    mlx5dv_devx_destroy_event_channel(iface->event_channel);
#endif
}

static ucs_status_t
uct_rc_mlx5_devx_init_rx_common(uct_rc_mlx5_iface_common_t *iface,
                                uct_ib_mlx5_md_t *md,
                                const uct_rc_iface_common_config_t *config,
                                const struct mlx5dv_pd *dvpd, void *wq)
{
    ucs_status_t status  = UCS_ERR_NO_MEMORY;
    int len, max, stride, log_num_of_strides, wq_type;

    stride = uct_ib_mlx5_srq_stride(iface->tm.mp.num_strides);
    max    = uct_ib_mlx5_srq_max_wrs(config->super.rx.queue_len,
                                     iface->tm.mp.num_strides);
    max    = ucs_roundup_pow2(max);
    len    = max * stride;

    status = uct_ib_mlx5_md_buf_alloc(md, len, 0, &iface->rx.srq.buf,
                                      &iface->rx.srq.devx.mem, "srq buf");
    if (status != UCS_OK) {
        return status;
    }

    iface->rx.srq.devx.dbrec = uct_ib_mlx5_get_dbrec(md);
    if (!iface->rx.srq.devx.dbrec) {
        goto err_free_mem;
    }

    iface->rx.srq.db = &iface->rx.srq.devx.dbrec->db[MLX5_RCV_DBR];

    if (iface->config.srq_topo == UCT_RC_MLX5_SRQ_TOPO_CYCLIC) {
        wq_type = UCT_RC_MLX5_MP_ENABLED(iface) ?
                  UCT_IB_MLX5_SRQ_TOPO_CYCLIC_MP_RQ :
                  UCT_IB_MLX5_SRQ_TOPO_CYCLIC;
    } else {
        wq_type = UCT_RC_MLX5_MP_ENABLED(iface) ?
                  UCT_IB_MLX5_SRQ_TOPO_LIST_MP_RQ :
                  UCT_IB_MLX5_SRQ_TOPO_LIST;
    }

    UCT_IB_MLX5DV_SET  (wq, wq, wq_type,       wq_type);
    UCT_IB_MLX5DV_SET  (wq, wq, log_wq_sz,     ucs_ilog2(max));
    UCT_IB_MLX5DV_SET  (wq, wq, log_wq_stride, ucs_ilog2(stride));
    UCT_IB_MLX5DV_SET  (wq, wq, pd,            dvpd->pdn);
    UCT_IB_MLX5DV_SET  (wq, wq, dbr_umem_id,   iface->rx.srq.devx.dbrec->mem_id);
    UCT_IB_MLX5DV_SET64(wq, wq, dbr_addr,      iface->rx.srq.devx.dbrec->offset);
    UCT_IB_MLX5DV_SET  (wq, wq, wq_umem_id,    iface->rx.srq.devx.mem.mem->umem_id);

    if (UCT_RC_MLX5_MP_ENABLED(iface)) {
        /* Normalize to device's interface values (range of (-6) - 7) */
        /* cppcheck-suppress internalAstError */
        log_num_of_strides = ucs_ilog2(iface->tm.mp.num_strides) - 9;

        UCT_IB_MLX5DV_SET(wq, wq, log_wqe_num_of_strides,
                          log_num_of_strides & 0xF);
        UCT_IB_MLX5DV_SET(wq, wq, log_wqe_stride_size,
                          (ucs_ilog2(iface->super.super.config.seg_size) - 6));
    }

    iface->rx.srq.type = UCT_IB_MLX5_OBJ_TYPE_DEVX;
    uct_ib_mlx5_srq_buff_init(&iface->rx.srq, 0, max - 1,
                              iface->super.super.config.seg_size,
                              iface->tm.mp.num_strides);
    iface->super.rx.srq.quota = max - 1;

    return UCS_OK;

err_free_mem:
    uct_ib_mlx5_md_buf_free(md, iface->rx.srq.buf, &iface->rx.srq.devx.mem);
    return status;
}

#if IBV_HW_TM
ucs_status_t
uct_rc_mlx5_devx_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                            const uct_rc_iface_common_config_t *config,
                            int dc, unsigned rndv_hdr_len)
{
    uct_ib_mlx5_md_t *md  = ucs_derived_of(uct_ib_iface_md(&iface->super.super),
                                           uct_ib_mlx5_md_t);
    uct_ib_device_t *dev  = &md->super.dev;
    struct mlx5dv_pd dvpd = {};
    struct mlx5dv_cq dvcq = {};
    struct mlx5dv_obj dv  = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_xrq_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_xrq_out)] = {};
    ucs_status_t status;
    void *xrqc;

    uct_rc_mlx5_init_rx_tm_common(iface, config, rndv_hdr_len);

    dv.pd.in  = uct_ib_iface_md(&iface->super.super)->pd;
    dv.cq.in  = iface->super.super.cq[UCT_IB_DIR_RX];
    dv.pd.out = &dvpd;
    dv.cq.out = &dvcq;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_PD | MLX5DV_OBJ_CQ);

    UCT_IB_MLX5DV_SET(create_xrq_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_XRQ);
    xrqc = UCT_IB_MLX5DV_ADDR_OF(create_xrq_in, in, xrq_context);

    UCT_IB_MLX5DV_SET(xrqc, xrqc, topology, UCT_IB_MLX5_XRQC_TOPOLOGY_TAG_MATCHING);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, offload,  UCT_IB_MLX5_XRQC_OFFLOAD_RNDV);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, tag_matching_topology_context.log_matching_list_sz,
                                  ucs_ilog2(iface->tm.num_tags) + 1);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, dc,       dc);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, cqn,      dvcq.cqn);

    status = uct_rc_mlx5_devx_init_rx_common(iface, md, config, &dvpd,
                                             UCT_IB_MLX5DV_ADDR_OF(xrqc, xrqc, wq));
    if (status != UCS_OK) {
        return UCS_OK;
    }

    iface->rx.srq.devx.obj = mlx5dv_devx_obj_create(dev->ibv_context,
                                                    in, sizeof(in),
                                                    out, sizeof(out));
    if (iface->rx.srq.devx.obj == NULL) {
        ucs_error("mlx5dv_devx_obj_create(XRQ) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(create_xrq_out, out, syndrome));
        status = UCS_ERR_IO_ERROR;
        goto err_cleanup_srq;
    }

    iface->rx.srq.srq_num = UCT_IB_MLX5DV_GET(create_xrq_out, out, xrqn);
    uct_rc_mlx5_iface_tm_set_cmd_qp_len(iface);

    return UCS_OK;

err_cleanup_srq:
    uct_rc_mlx5_devx_cleanup_srq(md, &iface->rx.srq);
    return status;
}
#endif

ucs_status_t uct_rc_mlx5_devx_init_rx(uct_rc_mlx5_iface_common_t *iface,
                                      const uct_rc_iface_common_config_t *config)
{
    uct_ib_mlx5_md_t *md  = ucs_derived_of(uct_ib_iface_md(&iface->super.super),
                                           uct_ib_mlx5_md_t);
    uct_ib_device_t *dev  = &md->super.dev;
    struct mlx5dv_pd dvpd = {};
    struct mlx5dv_obj dv  = {};
    char in[UCT_IB_MLX5DV_ST_SZ_BYTES(create_rmp_in)]   = {};
    char out[UCT_IB_MLX5DV_ST_SZ_BYTES(create_rmp_out)] = {};
    ucs_status_t status;
    void *rmpc;

    dv.pd.in  = uct_ib_iface_md(&iface->super.super)->pd;
    dv.pd.out = &dvpd;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_PD);

    UCT_IB_MLX5DV_SET(create_rmp_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_RMP);
    rmpc = UCT_IB_MLX5DV_ADDR_OF(create_rmp_in, in, rmp_context);

    UCT_IB_MLX5DV_SET(rmpc, rmpc, state, UCT_IB_MLX5_RMPC_STATE_RDY);

    status = uct_rc_mlx5_devx_init_rx_common(iface, md, config, &dvpd,
                                             UCT_IB_MLX5DV_ADDR_OF(rmpc, rmpc, wq));
    if (status != UCS_OK) {
        return status;
    }

    iface->rx.srq.devx.obj = mlx5dv_devx_obj_create(dev->ibv_context,
                                                    in, sizeof(in),
                                                    out, sizeof(out));
    if (iface->rx.srq.devx.obj == NULL) {
        ucs_error("mlx5dv_devx_obj_create(RMP) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(create_rmp_out, out, syndrome));
        status = UCS_ERR_IO_ERROR;
        goto err_cleanup_srq;
    }

    iface->rx.srq.srq_num = UCT_IB_MLX5DV_GET(create_rmp_out, out, rmpn);

    return UCS_OK;

err_cleanup_srq:
    uct_rc_mlx5_devx_cleanup_srq(md, &iface->rx.srq);
    return status;
}

void uct_rc_mlx5_devx_cleanup_srq(uct_ib_mlx5_md_t *md, uct_ib_mlx5_srq_t *srq)
{
    uct_ib_mlx5_put_dbrec(srq->devx.dbrec);
    uct_ib_mlx5_md_buf_free(md, srq->buf, &srq->devx.mem);
}

ucs_status_t
uct_rc_mlx5_iface_common_devx_create_qp(uct_rc_mlx5_iface_common_t *iface,
                                        uct_ib_mlx5_qp_t *qp,
                                        uct_ib_mlx5_txwq_t *tx,
                                        uct_ib_mlx5_qp_attr_t *attr)
{
    uct_ib_iface_t *ib_iface = &iface->super.super;
    uct_ib_mlx5_md_t *md     = ucs_derived_of(uct_ib_iface_md(ib_iface),
                                              uct_ib_mlx5_md_t);
    uct_ib_mlx5_mmio_mode_t mmio_mode;
    uct_ib_mlx5_devx_uar_t *uar;
    ucs_status_t status;

    uct_ib_iface_fill_attr(ib_iface, &attr->super);

    attr->pkey_index  = ib_iface->pkey_index;
    attr->is_roce_dev = uct_ib_iface_is_roce(ib_iface);

    status = uct_ib_mlx5_get_mmio_mode(iface->super.super.super.worker,
                                       attr->mmio_mode,
                                       UCT_IB_MLX5_BF_REG_SIZE, &mmio_mode);
    if (status != UCS_OK) {
        goto err;
    }

    uar = uct_worker_tl_data_get(ib_iface->super.worker,
                                 UCT_IB_MLX5_DEVX_UAR_KEY,
                                 uct_ib_mlx5_devx_uar_t,
                                 uct_ib_mlx5_devx_uar_cmp,
                                 uct_ib_mlx5_devx_uar_init,
                                 md, mmio_mode);
    if (UCS_PTR_IS_ERR(uar)) {
        status = UCS_PTR_STATUS(uar);
        goto err;
    }

    attr->uar = uar;

    status = uct_ib_mlx5_devx_create_qp(md, qp, tx, attr);
    if (status != UCS_OK) {
        goto err_uar;
    }

    return UCS_OK;

err_uar:
    uct_worker_tl_data_put(uar, uct_ib_mlx5_devx_uar_cleanup);
err:
    return status;
}

ucs_status_t
uct_rc_mlx5_iface_common_devx_connect_qp(uct_rc_mlx5_iface_common_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         uint32_t dest_qp_num,
                                         struct ibv_ah_attr *ah_attr,
                                         enum ibv_mtu path_mtu,
                                         uint8_t path_index)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(uct_ib_iface_md(&iface->super.super),
                                          uct_ib_mlx5_md_t);
    uct_ib_mlx5_qp_connect_attr_t attr = {
        .ah_attr          = ah_attr,
        .path_index       = path_index,
        .path_mtu         = path_mtu,
        .dest_qp_num      = dest_qp_num,
        .exp_backoff      = iface->super.config.exp_backoff,
        .is_roce_dev      = uct_ib_iface_is_roce(&iface->super.super),
        .roce_ver         = uct_ib_device_roce_version(&md->super.dev,
                                                       ah_attr->port_num,
                                                       ah_attr->grh.sgid_index),
        .sl               = iface->super.super.config.sl,
        .traffic_class    = iface->super.super.config.traffic_class,
        .log_ack_req_freq = iface->config.log_ack_req_freq,
        .max_rd_atomic    = iface->super.config.max_rd_atomic,
        .min_rnr_timer    = iface->super.config.min_rnr_timer,
        .retry_cnt        = iface->super.config.retry_cnt,
        .rnr_retry        = iface->super.config.rnr_retry,
        .timeout          = iface->super.config.timeout
    };

    ucs_assert(path_mtu != UCT_IB_ADDRESS_INVALID_PATH_MTU);
    return uct_ib_mlx5_devx_connect_rc_qp(md, qp, &attr);
}
