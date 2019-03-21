/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_mlx5.h"
#include "ib_mlx5.inl"
#include "ib_mlx5_log.h"

#include <uct/ib/base/ib_verbs.h>
#include <uct/ib/base/ib_device.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <string.h>


static const char *uct_ib_mlx5_mmio_modes[] = {
    [UCT_IB_MLX5_MMIO_MODE_BF_POST]    = "bf_post",
    [UCT_IB_MLX5_MMIO_MODE_BF_POST_MT] = "bf_post_mt",
    [UCT_IB_MLX5_MMIO_MODE_DB]         = "db",
    [UCT_IB_MLX5_MMIO_MODE_AUTO]       = "auto",
    [UCT_IB_MLX5_MMIO_MODE_LAST]       = NULL
};

ucs_config_field_t uct_ib_mlx5_iface_config_table[] = {
#if HAVE_IBV_EXP_DM
    {"DM_SIZE", "2k",
     "Device Memory segment size (0 - disabled)",
     ucs_offsetof(uct_ib_mlx5_iface_config_t, dm.seg_len), UCS_CONFIG_TYPE_MEMUNITS},
    {"DM_COUNT", "1",
     "Device Memory segments count (0 - disabled)",
     ucs_offsetof(uct_ib_mlx5_iface_config_t, dm.count), UCS_CONFIG_TYPE_UINT},
#endif

    {"MMIO_MODE", "auto",
     "How to write to MMIO register when posting sends on a QP. One of the following:\n"
     " bf_post    - BlueFlame post, write the WQE fully to MMIO register.\n"
     " bf_post_mt - Thread-safe BlueFlame, same as bf_post but same MMIO register can be used\n"
     "              by multiple threads.\n"
     " db         - Doorbell mode, write only 8 bytes to MMIO register, followed by a memory\n"
     "              store fence, which makes sure the doorbell goes out on the bus.\n"
     " auto       - Select best according to worker thread mode.",
     ucs_offsetof(uct_ib_mlx5_iface_config_t, mmio_mode),
     UCS_CONFIG_TYPE_ENUM(uct_ib_mlx5_mmio_modes)},

    {NULL}
};

ucs_status_t uct_ib_mlx5_create_cq(struct ibv_context *context, int cqe,
                                   struct ibv_comp_channel *channel,
                                   int comp_vector, int ignore_overrun,
                                   size_t *inl, struct ibv_cq **cq_p)
{
#if defined(HAVE_DECL_MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE) && HAVE_DECL_MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE
    struct ibv_cq *cq;
    struct ibv_cq_init_attr_ex cq_attr = {};
    struct mlx5dv_cq_init_attr dv_attr = {};

    cq_attr.cqe = cqe;
    cq_attr.channel = channel;
    cq_attr.comp_vector = comp_vector;
    if (ignore_overrun) {
        cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_FLAGS;
        cq_attr.flags = IBV_CREATE_CQ_ATTR_IGNORE_OVERRUN;
    }
    dv_attr.comp_mask = MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE;
    dv_attr.cqe_size  = uct_ib_get_cqe_size(*inl > 32 ? 128 : 64);
    cq = ibv_cq_ex_to_cq(mlx5dv_create_cq(context, &cq_attr, &dv_attr));
    if (!cq) {
        ucs_error("mlx5dv_create_cq(cqe=%d) failed: %m", cqe);
        return UCS_ERR_IO_ERROR;
    }

    *cq_p = cq;
    *inl  = dv_attr.cqe_size / 2;
    return UCS_OK;
#else
    return uct_ib_verbs_create_cq(context, cqe, channel, comp_vector,
                                  ignore_overrun, inl, cq_p);
#endif
}

ucs_status_t uct_ib_mlx5_get_cq(struct ibv_cq *cq, uct_ib_mlx5_cq_t *mlx5_cq)
{
    uct_ib_mlx5dv_cq_t dcq = {};
    uct_ib_mlx5dv_t obj = {};
    struct mlx5_cqe64 *cqe;
    unsigned cqe_size;
    ucs_status_t status;
    int ret, i;

    obj.dv.cq.in = cq;
    obj.dv.cq.out = &dcq.dv;
    status = uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_CQ);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }

    mlx5_cq->cq_buf    = dcq.dv.buf;
    mlx5_cq->cq_ci     = 0;
    mlx5_cq->cq_sn     = 0;
    mlx5_cq->cq_length = dcq.dv.cqe_cnt;
    mlx5_cq->cq_num    = dcq.dv.cqn;
#if HAVE_STRUCT_MLX5DV_CQ_CQ_UAR
    mlx5_cq->uar       = dcq.dv.cq_uar;
#else
    /* coverity[var_deref_model] */
    mlx5_cq->uar       = uct_dv_get_info_uar0(dcq.dv.uar);
#endif
    mlx5_cq->dbrec     = dcq.dv.dbrec;
    cqe_size           = dcq.dv.cqe_size;

    /* Move buffer forward for 128b CQE, so we would get pointer to the 2nd
     * 64b when polling.
     */
    mlx5_cq->cq_buf = (char*)mlx5_cq->cq_buf + cqe_size - sizeof(struct mlx5_cqe64);

    ret = ibv_exp_cq_ignore_overrun(cq);
    if (ret != 0) {
        ucs_error("Failed to modify send CQ to ignore overrun: %s", strerror(ret));
        return UCS_ERR_UNSUPPORTED;
    }

    mlx5_cq->cqe_size_log = ucs_ilog2(cqe_size);
    ucs_assert_always((1<<mlx5_cq->cqe_size_log) == cqe_size);

    /* Set owner bit for all CQEs, so that CQE would look like it is in HW
     * ownership. In this case CQ polling functions will return immediately if
     * no any CQE ready, there is no need to check opcode for
     * MLX5_CQE_INVALID value anymore. */
    for (i = 0; i < mlx5_cq->cq_length; ++i) {
        cqe = uct_ib_mlx5_get_cqe(mlx5_cq, i);
        cqe->op_own |= MLX5_CQE_OWNER_MASK;
    }


    return UCS_OK;
}

static int
uct_ib_mlx5_iface_res_domain_cmp(uct_ib_mlx5_iface_res_domain_t *res_domain,
                                 uct_ib_md_t *md, uct_priv_worker_t *worker)
{
#if defined(HAVE_IBV_EXP_RES_DOMAIN) && HAVE_IBV_EXP_RES_DOMAIN
    return res_domain->ibv_domain->context == md->dev.ibv_context;
#elif HAVE_DECL_IBV_ALLOC_TD
    return res_domain->pd->context == md->dev.ibv_context;
#else
    return 1;
#endif
}

static ucs_status_t
uct_ib_mlx5_iface_res_domain_init(uct_ib_mlx5_iface_res_domain_t *res_domain,
                                  uct_ib_md_t *md, uct_priv_worker_t *worker)
{
#if defined(HAVE_IBV_EXP_RES_DOMAIN) && HAVE_IBV_EXP_RES_DOMAIN
    struct ibv_exp_res_domain_init_attr attr;

    attr.comp_mask    = IBV_EXP_RES_DOMAIN_THREAD_MODEL |
                        IBV_EXP_RES_DOMAIN_MSG_MODEL;
    attr.msg_model    = IBV_EXP_MSG_LOW_LATENCY;

    switch (worker->thread_mode) {
    case UCS_THREAD_MODE_SINGLE:
        attr.thread_model = IBV_EXP_THREAD_SINGLE;
        break;
    case UCS_THREAD_MODE_SERIALIZED:
        attr.thread_model = IBV_EXP_THREAD_UNSAFE;
        break;
    default:
        attr.thread_model = IBV_EXP_THREAD_SAFE;
        break;
    }

    res_domain->ibv_domain = ibv_exp_create_res_domain(md->dev.ibv_context, &attr);
    if (res_domain->ibv_domain == NULL) {
        ucs_error("ibv_exp_create_res_domain() on %s failed: %m",
                  uct_ib_device_name(&md->dev));
        return UCS_ERR_IO_ERROR;
    }
#elif HAVE_DECL_IBV_ALLOC_TD
    struct ibv_parent_domain_init_attr attr;
    struct ibv_td_init_attr td_attr;

    if (worker->thread_mode == UCS_THREAD_MODE_MULTI) {
        td_attr.comp_mask = 0;
        res_domain->td = ibv_alloc_td(md->dev.ibv_context, &td_attr);
        if (res_domain->td == NULL) {
            ucs_error("ibv_alloc_td() on %s failed: %m",
                      uct_ib_device_name(&md->dev));
            return UCS_ERR_IO_ERROR;
        }
    } else {
        res_domain->td = NULL;
        res_domain->pd = md->pd;
        return UCS_OK;
    }

    attr.td = res_domain->td;
    attr.pd = md->pd;
    attr.comp_mask = 0;
    res_domain->pd = ibv_alloc_parent_domain(md->dev.ibv_context, &attr);
    if (res_domain->pd == NULL) {
        ucs_error("ibv_alloc_parent_domain() on %s failed: %m",
                  uct_ib_device_name(&md->dev));
        ibv_dealloc_td(res_domain->td);
        return UCS_ERR_IO_ERROR;
    }
#endif
    return UCS_OK;
}

static void uct_ib_mlx5_iface_res_domain_cleanup(uct_ib_mlx5_iface_res_domain_t *res_domain)
{
#if defined(HAVE_IBV_EXP_RES_DOMAIN) && HAVE_IBV_EXP_RES_DOMAIN
    struct ibv_exp_destroy_res_domain_attr attr;
    int ret;

    attr.comp_mask = 0;
    ret = ibv_exp_destroy_res_domain(res_domain->ibv_domain->context,
                                     res_domain->ibv_domain, &attr);
    if (ret != 0) {
        ucs_warn("ibv_exp_destroy_res_domain() failed: %m");
    }
#elif HAVE_DECL_IBV_ALLOC_TD
    int ret;

    if (res_domain->td != NULL) {
        ret = ibv_dealloc_pd(res_domain->pd);
        if (ret != 0) {
            ucs_warn("ibv_dealloc_pd() failed: %m");
            return;
        }

        ret = ibv_dealloc_td(res_domain->td);
        if (ret != 0) {
            ucs_warn("ibv_dealloc_td() failed: %m");
        }
    }
#endif
}

ucs_status_t uct_ib_mlx5_iface_init_res_domain(uct_ib_iface_t *iface,
                                               uct_ib_mlx5_iface_common_t *mlx5)
{
    mlx5->res_domain = uct_worker_tl_data_get(iface->super.worker,
                                              UCT_IB_MLX5_RES_DOMAIN_KEY,
                                              uct_ib_mlx5_iface_res_domain_t,
                                              uct_ib_mlx5_iface_res_domain_cmp,
                                              uct_ib_mlx5_iface_res_domain_init,
                                              uct_ib_iface_md(iface),
                                              iface->super.worker);
    if (UCS_PTR_IS_ERR(mlx5->res_domain)) {
        return UCS_PTR_STATUS(mlx5->res_domain);
    }

    return UCS_OK;
}

void uct_ib_mlx5_iface_cleanup_res_domain(uct_ib_mlx5_iface_common_t *mlx5)
{
    uct_worker_tl_data_put(mlx5->res_domain, uct_ib_mlx5_iface_res_domain_cleanup);
}

ucs_status_t uct_ib_mlx5_iface_create_qp(uct_ib_iface_t *iface,
                                         uct_ib_mlx5_iface_common_t *mlx5,
                                         uct_ib_qp_attr_t *attr,
                                         struct ibv_qp **qp_p)
{
    uct_ib_mlx5_iface_fill_attr(iface, mlx5, attr);
    return uct_ib_iface_create_qp(iface, attr, qp_p);
}

#if !HAVE_DECL_MLX5DV_CONTEXT_FLAGS_DEVX
ucs_status_t uct_ib_mlx5_get_compact_av(uct_ib_iface_t *iface, int *compact_av)
{
    struct mlx5_wqe_av  mlx5_av;
    struct ibv_ah      *ah;
    uct_ib_address_t   *ib_addr;
    ucs_status_t        status;
    struct ibv_ah_attr  ah_attr;

    /* coverity[result_independent_of_operands] */
    ib_addr = ucs_alloca((size_t)iface->addr_size);

    status = uct_ib_iface_get_device_address(&iface->super.super,
                                             (uct_device_addr_t*)ib_addr);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_iface_fill_ah_attr_from_addr(iface, ib_addr, iface->path_bits[0], &ah_attr);
    ah_attr.is_global = iface->is_global_addr;
    status = uct_ib_iface_create_ah(iface, &ah_attr, &ah);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_mlx5_get_av(ah, &mlx5_av);

    /* copy MLX5_EXTENDED_UD_AV from the driver, if the flag is not present then
     * the device supports compact address vector. */
    *compact_av = !(mlx5_av_base(&mlx5_av)->dqp_dct & UCT_IB_MLX5_EXTENDED_UD_AV);
    return UCS_OK;
}
#endif

void uct_ib_mlx5_check_completion(uct_ib_iface_t *iface, uct_ib_mlx5_cq_t *cq,
                                  struct mlx5_cqe64 *cqe)
{
    ucs_status_t status;

    switch (cqe->op_own >> 4) {
    case MLX5_CQE_REQ_ERR:
        /* update ci before invoking error callback, since it can poll on cq */
        UCS_STATIC_ASSERT(MLX5_CQE_REQ_ERR & (UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK >> 4));
        ++cq->cq_ci;
        status = uct_ib_mlx5_completion_with_err(iface, (void*)cqe, NULL,
                                                 UCS_LOG_LEVEL_DEBUG);
        iface->ops->handle_failure(iface, cqe, status);
        return;
    case MLX5_CQE_RESP_ERR:
        /* Local side failure - treat as fatal */
        UCS_STATIC_ASSERT(MLX5_CQE_RESP_ERR & (UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK >> 4));
        ++cq->cq_ci;
        uct_ib_mlx5_completion_with_err(iface, (void*)cqe, NULL,
                                        UCS_LOG_LEVEL_FATAL);
        return;
    default:
        /* CQE might have been updated by HW. Skip it now, and it would be handled
         * in next polling. */
        return;
    }
}

static int uct_ib_mlx5_mmio_cmp(uct_ib_mlx5_mmio_reg_t *reg, uintptr_t addr,
                                unsigned bf_size)
{
    return (reg->addr.uint & ~UCT_IB_MLX5_BF_REG_SIZE) ==
           (addr & ~UCT_IB_MLX5_BF_REG_SIZE);
}

static ucs_status_t uct_ib_mlx5_mmio_init(uct_ib_mlx5_mmio_reg_t *reg,
                                          uintptr_t addr,
                                          uct_ib_mlx5_mmio_mode_t mmio_mode)
{
    reg->addr.uint = addr;
    reg->mode      = mmio_mode;
    return UCS_OK;
}

static void uct_ib_mlx5_mmio_cleanup(uct_ib_mlx5_mmio_reg_t *reg)
{
}

void uct_ib_mlx5_txwq_reset(uct_ib_mlx5_txwq_t *txwq)
{
    txwq->curr       = txwq->qstart;
    txwq->sw_pi      = 0;
    txwq->prev_sw_pi = -1;
#ifdef ENABLE_ASSERT
    txwq->hw_ci      = 0xFFFF;
#endif
    memset(txwq->qstart, 0, (char*)txwq->qend - (char*)txwq->qstart);
}

ucs_status_t uct_ib_mlx5_txwq_init(uct_priv_worker_t *worker,
                                   uct_ib_mlx5_mmio_mode_t cfg_mmio_mode,
                                   uct_ib_mlx5_txwq_t *txwq,
                                   struct ibv_qp *verbs_qp)
{
    uct_ib_mlx5_mmio_mode_t mmio_mode;
    uct_ib_mlx5dv_qp_t qp_info = {};
    uct_ib_mlx5dv_t obj = {};
    ucs_status_t status;

    obj.dv.qp.in = verbs_qp;
    obj.dv.qp.out = &qp_info.dv;

    status = uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_QP);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }

    if ((qp_info.dv.sq.stride != MLX5_SEND_WQE_BB) || !ucs_is_pow2(qp_info.dv.sq.wqe_cnt) ||
        ((qp_info.dv.bf.size != 0) && (qp_info.dv.bf.size != UCT_IB_MLX5_BF_REG_SIZE)))
    {
        ucs_error("mlx5 device parameters not suitable for transport "
                  "bf.size(%d) %d, sq.stride(%d) %d, wqe_cnt %d",
                  UCT_IB_MLX5_BF_REG_SIZE, qp_info.dv.bf.size,
                  MLX5_SEND_WQE_BB, qp_info.dv.sq.stride, qp_info.dv.sq.wqe_cnt);
        return UCS_ERR_IO_ERROR;
    }

    if (cfg_mmio_mode != UCT_IB_MLX5_MMIO_MODE_AUTO) {
        mmio_mode = cfg_mmio_mode;
    } else if (qp_info.dv.bf.size > 0) {
        if (worker->thread_mode == UCS_THREAD_MODE_SINGLE) {
            mmio_mode = UCT_IB_MLX5_MMIO_MODE_BF_POST;
        } else if (worker->thread_mode == UCS_THREAD_MODE_SERIALIZED) {
            mmio_mode = UCT_IB_MLX5_MMIO_MODE_BF_POST_MT;
        } else {
            ucs_error("unsupported thread mode for mlx5: %d", worker->thread_mode);
            return UCS_ERR_UNSUPPORTED;
        }
    } else {
        mmio_mode = UCT_IB_MLX5_MMIO_MODE_DB;
    }

    ucs_debug("tx wq %d bytes [bb=%d, nwqe=%d] mmio_mode %s",
              qp_info.dv.sq.stride * qp_info.dv.sq.wqe_cnt,
              qp_info.dv.sq.stride, qp_info.dv.sq.wqe_cnt,
              uct_ib_mlx5_mmio_modes[mmio_mode]);

    txwq->qstart     = qp_info.dv.sq.buf;
    txwq->qend       = (char*)qp_info.dv.sq.buf + (qp_info.dv.sq.stride * qp_info.dv.sq.wqe_cnt);
    txwq->reg        = uct_worker_tl_data_get(worker,
                                              UCT_IB_MLX5_WORKER_BF_KEY,
                                              uct_ib_mlx5_mmio_reg_t,
                                              uct_ib_mlx5_mmio_cmp,
                                              uct_ib_mlx5_mmio_init,
                                              (uintptr_t)qp_info.dv.bf.reg,
                                              mmio_mode);
    if (UCS_PTR_IS_ERR(txwq->reg)) {
        return UCS_PTR_STATUS(txwq->reg);
    }

    txwq->dbrec      = &qp_info.dv.dbrec[MLX5_SND_DBR];
    /* need to reserve 2x because:
     *  - on completion we only get the index of last wqe and we do not
     *    really know how many bb is there (but no more than max bb
     *  - on send we check that there is at least one bb. We know
     *  exact number of bbs once we actually are sending.
     */
    txwq->bb_max     = qp_info.dv.sq.wqe_cnt - 2 * UCT_IB_MLX5_MAX_BB;
    ucs_assert_always(txwq->bb_max > 0);
    txwq->next_fm    = 0;
    txwq->fence_beat = 0;

    uct_ib_mlx5_txwq_reset(txwq);
    return UCS_OK;
}

void uct_ib_mlx5_txwq_cleanup(uct_ib_mlx5_txwq_t* txwq)
{
    uct_worker_tl_data_put(txwq->reg, uct_ib_mlx5_mmio_cleanup);
}

ucs_status_t uct_ib_mlx5_get_rxwq(struct ibv_qp *verbs_qp, uct_ib_mlx5_rxwq_t *rxwq)
{
    uct_ib_mlx5dv_qp_t qp_info = {};
    uct_ib_mlx5dv_t obj = {};
    ucs_status_t status;

    obj.dv.qp.in = verbs_qp;
    obj.dv.qp.out = &qp_info.dv;

    status = uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_QP);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }

    if (!ucs_is_pow2(qp_info.dv.rq.wqe_cnt) ||
        qp_info.dv.rq.stride != sizeof(struct mlx5_wqe_data_seg)) {
        ucs_error("mlx5 rx wq [count=%d stride=%d] has invalid parameters",
                  qp_info.dv.rq.wqe_cnt,
                  qp_info.dv.rq.stride);
        return UCS_ERR_IO_ERROR;
    }
    rxwq->wqes            = qp_info.dv.rq.buf;
    rxwq->rq_wqe_counter  = 0;
    rxwq->cq_wqe_counter  = 0;
    rxwq->mask            = qp_info.dv.rq.wqe_cnt - 1;
    rxwq->dbrec           = &qp_info.dv.dbrec[MLX5_RCV_DBR];
    memset(rxwq->wqes, 0, qp_info.dv.rq.wqe_cnt * sizeof(struct mlx5_wqe_data_seg));

    return UCS_OK;
}

ucs_status_t uct_ib_mlx5_srq_init(uct_ib_mlx5_srq_t *srq, struct ibv_srq *verbs_srq,
                                  size_t sg_byte_count)
{
    uct_ib_mlx5dv_srq_t srq_info = {};
    uct_ib_mlx5_srq_seg_t *seg;
    uct_ib_mlx5dv_t obj = {};
    ucs_status_t status;
    unsigned i;

    obj.dv.srq.in = verbs_srq;
    obj.dv.srq.out = &srq_info.dv;

    status = uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_SRQ);
    if (status != UCS_OK) {
        return status;
    }

    if (srq_info.dv.head != 0) {
        ucs_error("SRQ head is not 0 (%d)", srq_info.dv.head);
        return UCS_ERR_NO_DEVICE;
    }

    if (srq_info.dv.stride != UCT_IB_MLX5_SRQ_STRIDE) {
        ucs_error("SRQ stride is not %lu (%d)", UCT_IB_MLX5_SRQ_STRIDE,
                  srq_info.dv.stride);
        return UCS_ERR_NO_DEVICE;
    }

    if (!ucs_is_pow2(srq_info.dv.tail + 1)) {
        ucs_error("SRQ length is not power of 2 (%d)", srq_info.dv.tail + 1);
        return UCS_ERR_NO_DEVICE;
    }

    srq->buf             = srq_info.dv.buf;
    srq->db              = srq_info.dv.dbrec;
    srq->free_idx        = srq_info.dv.tail;
    srq->ready_idx       = -1;
    srq->sw_pi           = -1;
    srq->mask            = srq_info.dv.tail;
    srq->tail            = srq_info.dv.tail;

    for (i = srq_info.dv.head; i <= srq_info.dv.tail; ++i) {
        seg = uct_ib_mlx5_srq_get_wqe(srq, i);
        seg->srq.free        = 0;
        seg->srq.desc        = NULL;
        seg->dptr.byte_count = htonl(sg_byte_count);
    }

    return UCS_OK;
}

void uct_ib_mlx5_srq_cleanup(uct_ib_mlx5_srq_t *srq, struct ibv_srq *verbs_srq)
{
    uct_ib_mlx5dv_srq_t srq_info = {};
    uct_ib_mlx5dv_t obj = {};
    ucs_status_t status;

    obj.dv.srq.in = verbs_srq;
    obj.dv.srq.out = &srq_info.dv;

    status = uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_SRQ);
    ucs_assert_always(status == UCS_OK);
    ucs_assertv_always(srq->tail == srq_info.dv.tail, "srq->tail=%d srq_info.tail=%d",
                       srq->tail, srq_info.dv.tail);
}
