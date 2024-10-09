/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ib_mlx5.h"
#include "ib_mlx5.inl"
#include "ib_mlx5_log.h"
#include <uct/ib/base/ib_verbs.h>
#include <uct/ib/base/ib_device.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <ucs/vfs/base/vfs_cb.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <string.h>


static const char *uct_ib_mlx5_mmio_modes[] = {
    [UCT_IB_MLX5_MMIO_MODE_BF_POST]    = "bf_post",
    [UCT_IB_MLX5_MMIO_MODE_BF_POST_MT] = "bf_post_mt",
    [UCT_IB_MLX5_MMIO_MODE_DB]         = "db",
    [UCT_IB_MLX5_MMIO_MODE_DB_LOCK]    = "db_lock",
    [UCT_IB_MLX5_MMIO_MODE_AUTO]       = "auto",
    [UCT_IB_MLX5_MMIO_MODE_LAST]       = NULL
};

ucs_config_field_t uct_ib_mlx5_iface_config_table[] = {
#if HAVE_IBV_DM
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
     " db_lock    - Doorbell mode, write only 8 bytes to MMIO register, guarded by spin lock.\n"
     " auto       - Select best according to worker thread mode.",
     ucs_offsetof(uct_ib_mlx5_iface_config_t, mmio_mode),
     UCS_CONFIG_TYPE_ENUM(uct_ib_mlx5_mmio_modes)},

    {"AR_ENABLE", "auto",
     "Enable Adaptive Routing (out of order) feature on SL that supports it.\n"
     "On RoCE devices, this is done by modifying the QP directly.\n"
     "SLs are selected as follows:\n"
     "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
     "+                                         + UCX_IB_AR_ENABLE=yes  + UCX_IB_AR_ENABLE=no   + UCX_IB_AR_ENABLE=try  + UCX_IB_AR_ENABLE=auto +\n"
     "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
     "+ UCX_IB_SL=auto + AR enabled on some SLs + Use 1st SL with AR    + Use 1st SL without AR + Use 1st SL with AR    + Use SL=0              +\n"
     "+                + AR enabled on all SLs  + Use SL=0              + Failure               + Use SL=0              + Use SL=0              +\n"
     "+                + AR disabled on all SLs + Failure               + Use SL=0              + Use SL=0              + Use SL=0              +\n"
     "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
     "+ UCX_IB_SL=<sl> + AR enabled on <sl>     + Use SL=<sl>           + Failure               + Use SL=<sl>           + Use SL=<sl>           +\n"
     "+                + AR disabled on <sl>    + Failure               + Use SL=<sl>           + Use SL=<sl>           + Use SL=<sl>           +\n"
     "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n",
     ucs_offsetof(uct_ib_mlx5_iface_config_t, ar_enable), UCS_CONFIG_TYPE_TERNARY_AUTO},

    {"TX_CQE_ZIP_ENABLE", "no",
     "Enable CQE zipping feature for sender side. CQE zipping reduces PCI utilization by\n"
     "merging several similar CQEs to a single CQE written by the device.",
     ucs_offsetof(uct_ib_mlx5_iface_config_t, cqe_zip_enable[UCT_IB_DIR_TX]),
     UCS_CONFIG_TYPE_BOOL},

    {"RX_CQE_ZIP_ENABLE", "no",
     "Enable CQE zipping feature for receiver side. CQE zipping reduces PCI utilization by\n"
     "merging several similar CQEs to a single CQE written by the device.",
     ucs_offsetof(uct_ib_mlx5_iface_config_t, cqe_zip_enable[UCT_IB_DIR_RX]),
     UCS_CONFIG_TYPE_BOOL},

    {NULL}
};

void uct_ib_mlx5_parse_cqe_zipping(uct_ib_mlx5_md_t *md,
                                   const uct_ib_mlx5_iface_config_t *mlx5_config,
                                   uct_ib_iface_init_attr_t *init_attr)
{
    uct_ib_dir_t dir;

    for (dir = 0; dir < UCT_IB_DIR_LAST; ++dir) {
        if (!mlx5_config->cqe_zip_enable[dir]) {
            continue;
        }

        if (md->flags & UCT_IB_MLX5_MD_FLAG_CQE64_ZIP) {
            init_attr->cqe_zip_sizes[dir] |= 64;
        }
        if (md->flags & UCT_IB_MLX5_MD_FLAG_CQE128_ZIP) {
            init_attr->cqe_zip_sizes[dir] |= 128;
        }
    }
}

ucs_status_t
uct_ib_mlx5_create_cq(uct_ib_iface_t *iface, uct_ib_dir_t dir,
                      const uct_ib_iface_init_attr_t *init_attr,
                      uct_ib_mlx5_cq_t *mlx5_cq, int preferred_cpu, size_t inl)
{
    ucs_status_t status                = UCS_OK;
#if HAVE_DECL_MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE
    uct_ib_device_t *dev               = uct_ib_iface_device(iface);
    struct ibv_cq_init_attr_ex cq_attr = {};
    struct mlx5dv_cq_init_attr dv_attr = {};
    struct ibv_cq *cq;

    uct_ib_fill_cq_attr(&cq_attr, init_attr, iface, preferred_cpu,
                        uct_ib_cq_size(iface, init_attr, dir));

    dv_attr.comp_mask = MLX5DV_CQ_INIT_ATTR_MASK_CQE_SIZE;
    dv_attr.cqe_size  = uct_ib_get_cqe_size(inl > 32 ? 128 : 64);

    cq = ibv_cq_ex_to_cq(mlx5dv_create_cq(dev->ibv_context, &cq_attr, &dv_attr));
    if (cq == NULL) {
        uct_ib_check_memlock_limit_msg(dev->ibv_context, UCS_LOG_LEVEL_ERROR,
                                       "mlx5dv_create_cq(cqe=%d)", cq_attr.cqe);
        return UCS_ERR_IO_ERROR;
    }

    iface->cq[dir]                 = cq;
    iface->config.max_inl_cqe[dir] = uct_ib_mlx5_inl_cqe(inl, dv_attr.cqe_size);
#else
    status = uct_ib_verbs_create_cq(iface, dir, init_attr, preferred_cpu, inl);
    if (status != UCS_OK) {
        return status;
    }
#endif

    status = uct_ib_mlx5_fill_cq(iface->cq[dir], mlx5_cq);
    if (status != UCS_OK) {
        ibv_destroy_cq(iface->cq[dir]);
    }

    return status;
}

void uct_ib_mlx5_destroy_cq(uct_ib_iface_t *iface, uct_ib_mlx5_cq_t *cq,
                            uct_ib_dir_t dir)
{
#if HAVE_DEVX
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.md, uct_ib_mlx5_md_t);

    if (cq->type == UCT_IB_MLX5_OBJ_TYPE_DEVX) {
        uct_ib_mlx5_devx_destroy_cq(md, cq);
        return;
    }
#endif

    uct_ib_verbs_destroy_cq(iface, dir);
}

ucs_status_t uct_ib_mlx5_fill_cq(struct ibv_cq *cq, uct_ib_mlx5_cq_t *mlx5_cq)
{
    uct_ib_mlx5dv_cq_t dcq = {};
    uct_ib_mlx5dv_t obj = {};
    ucs_status_t status;
    void *uar;

    obj.dv.cq.in = cq;
    obj.dv.cq.out = &dcq.dv;
    status = uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_CQ);
    if (status != UCS_OK) {
        return status;
    }

#if HAVE_STRUCT_MLX5DV_CQ_CQ_UAR
    uar = dcq.dv.cq_uar;
#else
    /* coverity[var_deref_model] */
    uar = uct_dv_get_info_uar0(dcq.dv.uar);
#endif

    uct_ib_mlx5_fill_cq_common(mlx5_cq, dcq.dv.cqe_cnt, dcq.dv.cqe_size,
                               dcq.dv.cqn, dcq.dv.buf, uar, dcq.dv.dbrec, 0);
    return UCS_OK;
}

void uct_ib_mlx5_fill_cq_common(uct_ib_mlx5_cq_t *cq,  unsigned cq_size,
                                unsigned cqe_size, uint32_t cqn, void *cq_buf,
                                void *uar, volatile void *dbrec, int zip)
{
    struct mlx5_cqe64 *cqe;
    int i;

    cq->cq_buf    = cq_buf;
    cq->cq_ci     = 0;
    cq->cq_sn     = 0;
    cq->cq_num    = cqn;
    cq->uar       = uar;
    cq->dbrec     = dbrec;
    cq->zip       = zip;

    /* Initializing memory is required for checking the cq_unzip.current_idx */
    memset(&cq->cq_unzip, 0, sizeof(uct_ib_mlx5_cq_unzip_t));

    /* Move buffer forward for 128b CQE, so we would get pointer to the 2nd
     * 64b when polling. */
    cq->cq_buf = UCS_PTR_BYTE_OFFSET(cq->cq_buf,
                                     cqe_size - sizeof(struct mlx5_cqe64));

    cq->cqe_size_log = ucs_ilog2(cqe_size);
    ucs_assert_always(UCS_BIT(cq->cqe_size_log) == cqe_size);

    cq->cq_length_log = ucs_ilog2(cq_size);
    ucs_assert_always(UCS_BIT(cq->cq_length_log) == cq_size);

    cq->cq_length_mask = UCS_MASK(cq->cq_length_log);

    if (cq->zip) {
        /* signature is in union with validity_iteration_count */
        cq->own_field_offset = ucs_offsetof(struct mlx5_cqe64, signature);
        cq->own_mask         = 0xff;
    } else {
        cq->own_field_offset = ucs_offsetof(struct mlx5_cqe64, op_own);
        cq->own_mask         = MLX5_CQE_OWNER_MASK;
    }

    /* Set owner bit for all CQEs, so that CQE would look like it is in HW
     * ownership. In this case CQ polling functions will return immediately if
     * no any CQE ready, there is no need to check opcode for
     * MLX5_CQE_INVALID value anymore.  */
    for (i = 0; i < cq_size; ++i) {
        cqe            = uct_ib_mlx5_get_cqe(cq, i);
        cqe->op_own   |= MLX5_CQE_OWNER_MASK;
        cqe->op_own   |= MLX5_CQE_INVALID << 4;
        cqe->signature = 0xff;
    }
}

static int
uct_ib_mlx5_res_domain_cmp(uct_ib_mlx5_res_domain_t *res_domain,
                           uct_ib_md_t *md, uct_priv_worker_t *worker)
{
    return res_domain->pd->context == md->dev.ibv_context;
}

static ucs_status_t
uct_ib_mlx5_res_domain_init(uct_ib_mlx5_res_domain_t *res_domain,
                            uct_ib_md_t *md, uct_priv_worker_t *worker)
{
    struct ibv_parent_domain_init_attr attr;
    struct ibv_td_init_attr td_attr;
    ucs_status_t status;

    td_attr.comp_mask = 0;
    res_domain->td = ibv_alloc_td(md->dev.ibv_context, &td_attr);
    if (res_domain->td == NULL) {
        ucs_debug("ibv_alloc_td() on %s failed: %m",
                  uct_ib_device_name(&md->dev));
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
        status = UCS_ERR_IO_ERROR;
        goto err_td;
    }

    return UCS_OK;

err_td:
    ibv_dealloc_td(res_domain->td);
    return status;
}

static void uct_ib_mlx5_res_domain_cleanup(uct_ib_mlx5_res_domain_t *res_domain)
{
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
}

ucs_status_t uct_ib_mlx5_iface_get_res_domain(uct_ib_iface_t *iface,
                                              uct_ib_mlx5_qp_t *qp)
{
    qp->verbs.rd = uct_worker_tl_data_get(iface->super.worker,
                                          UCT_IB_MLX5_RES_DOMAIN_KEY,
                                          uct_ib_mlx5_res_domain_t,
                                          uct_ib_mlx5_res_domain_cmp,
                                          uct_ib_mlx5_res_domain_init,
                                          uct_ib_iface_md(iface),
                                          iface->super.worker);
    if (UCS_PTR_IS_ERR(qp->verbs.rd)) {
        return UCS_PTR_STATUS(qp->verbs.rd);
    }

    qp->type = UCT_IB_MLX5_OBJ_TYPE_VERBS;

    return UCS_OK;
}

void uct_ib_mlx5_iface_put_res_domain(uct_ib_mlx5_qp_t *qp)
{
    if (qp->type == UCT_IB_MLX5_OBJ_TYPE_VERBS) {
        uct_worker_tl_data_put(qp->verbs.rd, uct_ib_mlx5_res_domain_cleanup);
    }
}

ucs_status_t uct_ib_mlx5_iface_create_qp(uct_ib_iface_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         uct_ib_mlx5_qp_attr_t *attr)
{
    ucs_status_t status;

    status = uct_ib_mlx5_iface_get_res_domain(iface, qp);
    if (status != UCS_OK) {
        goto err;
    }

    uct_ib_mlx5_iface_fill_attr(iface, qp, attr);

    status = uct_ib_iface_create_qp(iface, &attr->super, &qp->verbs.qp);
    if (status != UCS_OK) {
        goto err_put_res_domain;
    }

    qp->qp_num = qp->verbs.qp->qp_num;
    return UCS_OK;

err_put_res_domain:
    uct_ib_mlx5_iface_put_res_domain(qp);
err:
    return status;
}

#if !HAVE_DEVX
ucs_status_t uct_ib_mlx5_get_compact_av(uct_ib_iface_t *iface, int *compact_av)
{
    struct mlx5_wqe_av  mlx5_av;
    struct ibv_ah      *ah;
    uct_ib_address_t   *ib_addr;
    ucs_status_t        status;
    struct ibv_ah_attr  ah_attr;
    enum ibv_mtu        path_mtu;

    /* coverity[result_independent_of_operands] */
    ib_addr = ucs_alloca((size_t)iface->addr_size);

    status = uct_ib_iface_get_device_address(&iface->super.super,
                                             (uct_device_addr_t*)ib_addr);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_iface_fill_ah_attr_from_addr(iface, ib_addr, 0, &ah_attr, &path_mtu);
    ah_attr.is_global = iface->config.force_global_addr;
    status = uct_ib_iface_create_ah(iface, &ah_attr, "compact AV check", &ah);
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

void uct_ib_mlx5_iface_cqe_unzip_init(uct_ib_mlx5_cq_t *cq)
{
    uct_ib_mlx5_cq_unzip_t *cq_unzip = &cq->cq_unzip;
    struct mlx5_cqe64 *title_cqe     = uct_ib_mlx5_get_cqe(cq, cq->cq_ci - 1);
    struct mlx5_cqe64 *mini_cqe      = uct_ib_mlx5_get_cqe(cq, cq->cq_ci);

    if (cq->cq_unzip.title_cqe_valid == 0) {
        memcpy(&cq_unzip->title, title_cqe, sizeof(cq_unzip->title));
        cq_unzip->wqe_counter        = ntohs(title_cqe->wqe_counter);
        cq->cq_unzip.title_cqe_valid = 1;
    } else {
        cq_unzip->wqe_counter += cq_unzip->block_size;
    }
    memcpy(&cq_unzip->mini_arr, mini_cqe,
           sizeof(uct_ib_mlx5_mini_cqe8_t) * UCT_IB_MLX5_MINICQE_ARR_MAX_SIZE);
    cq_unzip->block_size = (mini_cqe->op_own >> 4) + 1;
    ucs_assertv(cq_unzip->block_size <= 7, "block_size=%u",
                cq_unzip->block_size);
    cq_unzip->miniarr_cq_idx = cq->cq_ci;
}

static void
uct_ib_mlx5_iface_cqe_unzip_fill_unique(struct mlx5_cqe64 *cqe,
                                        uct_ib_mlx5_mini_cqe8_t *mini_cqe,
                                        uct_ib_mlx5_cq_unzip_t *cq_unzip)
{
    const uint32_t net_qpn_mask = htonl(UCS_MASK(UCT_IB_QPN_ORDER));

    /* TODO: Think about the checksum field unzipping
     * The checksum field does not copied from src dst
     * because of absence of the mlx5_cqe64::check_sum field on some systems.
     */
    cqe->byte_cnt = mini_cqe->byte_cnt;
    ucs_assert(!(cqe->op_own & UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK));
    if ((cqe->op_own >> 4) == MLX5_CQE_REQ) {
        cqe->wqe_counter  = mini_cqe->wqe_counter;
        cqe->sop_drop_qpn = (cqe->sop_drop_qpn & net_qpn_mask) |
                            htonl(mini_cqe->s_wqe_opcode << UCT_IB_QPN_ORDER);
    } else {
        cqe->wqe_counter = htons(cq_unzip->wqe_counter + cq_unzip->current_idx);
    }
}

struct mlx5_cqe64 *uct_ib_mlx5_iface_cqe_unzip(uct_ib_mlx5_cq_t *cq)
{
    uct_ib_mlx5_cq_unzip_t *cq_unzip  = &cq->cq_unzip;
    uint8_t mini_cqe_idx              = cq_unzip->current_idx %
                                        UCT_IB_MLX5_MINICQE_ARR_MAX_SIZE;
    struct mlx5_cqe64 *title_cqe      = &cq_unzip->title;
    uct_ib_mlx5_mini_cqe8_t *mini_cqe = &cq_unzip->mini_arr[mini_cqe_idx];
    struct mlx5_cqe64 *cqe;
    unsigned next_cqe_idx;

    cq_unzip->current_idx++;

    uct_ib_mlx5_iface_cqe_unzip_fill_unique(title_cqe, mini_cqe, cq_unzip);

    if (cq_unzip->current_idx < cq_unzip->block_size) {
        next_cqe_idx = cq_unzip->miniarr_cq_idx + cq_unzip->current_idx;
        cqe          = uct_ib_mlx5_get_cqe(cq, next_cqe_idx);

        /* Update opcode and signature in the next CQE buffer.
         * Signature is used for simplifying the zipped CQE detection
         * during the poll.
         */
        cqe->op_own    = UCT_IB_MLX5_CQE_FORMAT_MASK;
        cqe->signature = title_cqe->signature;
    } else {
        cq_unzip->current_idx = 0;
    }

    return title_cqe;
}

struct mlx5_cqe64 *
uct_ib_mlx5_check_completion(uct_ib_iface_t *iface, uct_ib_mlx5_cq_t *cq,
                             struct mlx5_cqe64 *cqe, int flags)
{
    if (uct_ib_mlx5_check_and_init_zipped(cq, cqe)) {
        ++cq->cq_ci;
        uct_ib_mlx5_update_cqe_zipping_stats(iface, cq);
        return uct_ib_mlx5_iface_cqe_unzip(cq);
    }

    if (cqe->op_own & UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK) {
        UCS_STATIC_ASSERT(MLX5_CQE_INVALID & (UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK >> 4));
        ucs_assert((cqe->op_own >> 4) != MLX5_CQE_INVALID);
        uct_ib_mlx5_check_completion_with_err(iface, cq, cqe);
    }

    return NULL; /* No CQE */
}

void uct_ib_mlx5_check_completion_with_err(uct_ib_iface_t *iface,
                                           uct_ib_mlx5_cq_t *cq,
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
        uct_ib_mlx5_update_db_cq_ci(cq);
        return;
    case MLX5_CQE_RESP_ERR:
        /* Local side failure - treat as fatal */
        UCS_STATIC_ASSERT(MLX5_CQE_RESP_ERR & (UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK >> 4));
        ++cq->cq_ci;
        uct_ib_mlx5_completion_with_err(iface, (void*)cqe, NULL,
                                        UCS_LOG_LEVEL_FATAL);
        uct_ib_mlx5_update_db_cq_ci(cq);
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

    ucs_spinlock_init(&reg->db_lock, 0);

    return UCS_OK;
}

static void uct_ib_mlx5_mmio_cleanup(uct_ib_mlx5_mmio_reg_t *reg)
{
    ucs_spinlock_destroy(&reg->db_lock);
}

int uct_ib_mlx5_devx_uar_cmp(uct_ib_mlx5_devx_uar_t *uar,
                             uct_ib_mlx5_md_t *md,
                             uct_ib_mlx5_mmio_mode_t mmio_mode)
{
    return (uar->ctx == md->super.dev.ibv_context) &&
           (uar->super.mode == mmio_mode);
}

#if HAVE_DEVX
static ucs_status_t uct_ib_mlx5_devx_alloc_uar(uct_ib_mlx5_md_t *md,
                                               uint32_t flags,
                                               struct mlx5dv_devx_uar **uar_p)
{
    const char *uar_type_str      = (flags == UCT_IB_MLX5_UAR_ALLOC_TYPE_WC) ?
                                    "WC" : "NC_DEDICATED";
    ucs_log_level_t err_log_level = UCS_LOG_LEVEL_DIAG;
    UCS_STRING_BUFFER_ONSTACK(strb, 256);
    struct mlx5dv_devx_uar *uar;
    ucs_status_t status;
    int err;

    uar = mlx5dv_devx_alloc_uar(md->super.dev.ibv_context, flags);
    if (uar != NULL) {
        *uar_p = uar;
        return UCS_OK;
    }

    err = errno;
    ucs_string_buffer_appendf(&strb,
                              "mlx5dv_devx_alloc_uar(device=%s, flags=0x%x)"
                              "type=%s failed: %s. ",
                              uct_ib_device_name(&md->super.dev),
                              flags,
                              uar_type_str,
                              strerror(err));
    switch (err) {
    case ENOMEM:
        ucs_string_buffer_appendf(&strb,
                                  "Consider increasing PF_LOG_BAR_SIZE "
                                  "using mlxconfig tool (requires reboot)");
        err_log_level = UCS_LOG_LEVEL_ERROR;
        status        = UCS_ERR_NO_MEMORY;
        break;
    case EOPNOTSUPP:
        status = UCS_ERR_UNSUPPORTED;
        break;
    case EINVAL:
        status = UCS_ERR_INVALID_PARAM;
        break;
    default:
        status = UCS_ERR_NO_MEMORY;
        break;
    }

    ucs_log(err_log_level, "%s", ucs_string_buffer_cstr(&strb));
    return status;
}
#endif

ucs_status_t uct_ib_mlx5_devx_check_uar(uct_ib_mlx5_md_t *md)
{
#if HAVE_DEVX
    uct_ib_mlx5_devx_uar_t uar;
    ucs_status_t status;

    status = uct_ib_mlx5_devx_alloc_uar(md,
                                        UCT_IB_MLX5_UAR_ALLOC_TYPE_WC,
                                        &uar.uar);
    if (status == UCS_ERR_UNSUPPORTED) {
        status = uct_ib_mlx5_devx_alloc_uar(md,
                                            UCT_IB_MLX5_UAR_ALLOC_TYPE_NC,
                                            &uar.uar);
        if (status == UCS_ERR_UNSUPPORTED) {
            ucs_diag("%s: both WC and NC_DEDICATED UAR allocation types "
                     "are not supported", uct_ib_device_name(&md->super.dev));
            return status;
        } else if (status != UCS_OK) {
            return status;
        }
        /* NC_DEDICATED is supported - the flag is automatically set to 0 */
    } else if (status != UCS_OK) {
        /* The error is unrelated to the UAR allocation type, no fallback */
        return status;
    } else {
        /* WC is supported - set the flag to 1 */
        md->flags |= UCT_IB_MLX5_MD_FLAG_UAR_USE_WC;
    }

    uct_ib_mlx5_devx_uar_cleanup(&uar);
    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_ib_mlx5_devx_uar_init(uct_ib_mlx5_devx_uar_t *uar,
                                       uct_ib_mlx5_md_t *md,
                                       uct_ib_mlx5_mmio_mode_t mmio_mode)
{
#if HAVE_DEVX
    ucs_status_t status;
    uint32_t flags;

    /* Use UCT_IB_MLX5_MD_FLAG_UAR_USE_WC to determine the supported UAR allocation type */
    if (md->flags & UCT_IB_MLX5_MD_FLAG_UAR_USE_WC) {
        flags = UCT_IB_MLX5_UAR_ALLOC_TYPE_WC;
    } else {
        flags = UCT_IB_MLX5_UAR_ALLOC_TYPE_NC;
    }

    status = uct_ib_mlx5_devx_alloc_uar(md, flags, &uar->uar);
    if (status != UCS_OK) {
        return status;
    }

    uar->super.addr.ptr = uar->uar->reg_addr;
    uar->super.mode     = mmio_mode;
    uar->ctx            = md->super.dev.ibv_context;

    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

void uct_ib_mlx5_devx_uar_cleanup(uct_ib_mlx5_devx_uar_t *uar)
{
#if HAVE_DEVX
    mlx5dv_devx_free_uar(uar->uar);
#endif
}

void uct_ib_mlx5_txwq_reset(uct_ib_mlx5_txwq_t *txwq)
{
    txwq->curr       = txwq->qstart;
    txwq->sw_pi      = 0;
    txwq->prev_sw_pi = UINT16_MAX;
#if UCS_ENABLE_ASSERT
    txwq->hw_ci      = 0xFFFF;
    txwq->flags      = 0;
#endif
    uct_ib_fence_info_init(&txwq->fi);
    memset(txwq->qstart, 0, UCS_PTR_BYTE_DIFF(txwq->qstart, txwq->qend));

    /* Make uct_ib_mlx5_txwq_num_posted_wqes() work if no wqe has completed by
       setting number-of-segments (ds) field of the last wqe to 1 */
    uct_ib_mlx5_set_ctrl_qpn_ds(uct_ib_mlx5_txwq_get_wqe(txwq, 0xffff), 0, 1);
}

void uct_ib_mlx5_txwq_vfs_populate(uct_ib_mlx5_txwq_t *txwq, void *parent_obj)
{
    ucs_vfs_obj_add_ro_file(parent_obj, ucs_vfs_show_primitive,
                            &txwq->super.qp_num, UCS_VFS_TYPE_U32_HEX,
                            "qp_num");
    ucs_vfs_obj_add_ro_file(parent_obj, ucs_vfs_show_primitive, &txwq->sw_pi,
                            UCS_VFS_TYPE_U16, "sw_pi");
    ucs_vfs_obj_add_ro_file(parent_obj, ucs_vfs_show_primitive,
                            &txwq->prev_sw_pi, UCS_VFS_TYPE_U16, "prev_sw_pi");
    ucs_vfs_obj_add_ro_file(parent_obj, ucs_vfs_show_primitive, &txwq->qstart,
                            UCS_VFS_TYPE_POINTER, "qstart");
    ucs_vfs_obj_add_ro_file(parent_obj, ucs_vfs_show_primitive, &txwq->qend,
                            UCS_VFS_TYPE_POINTER, "qend");
    ucs_vfs_obj_add_ro_file(parent_obj, ucs_vfs_show_primitive, &txwq->bb_max,
                            UCS_VFS_TYPE_U16, "bb_max");
    ucs_vfs_obj_add_ro_file(parent_obj, ucs_vfs_show_primitive, &txwq->sig_pi,
                            UCS_VFS_TYPE_U16, "sig_pi");
#if UCS_ENABLE_ASSERT
    ucs_vfs_obj_add_ro_file(parent_obj, ucs_vfs_show_primitive, &txwq->hw_ci,
                            UCS_VFS_TYPE_U16, "hw_ci");
#endif
}

ucs_status_t
uct_ib_mlx5_get_mmio_mode(uct_priv_worker_t *worker,
                          uct_ib_mlx5_mmio_mode_t cfg_mmio_mode,
                          int need_lock, unsigned bf_size,
                          uct_ib_mlx5_mmio_mode_t *mmio_mode)
{
    ucs_assert(cfg_mmio_mode < UCT_IB_MLX5_MMIO_MODE_LAST);

    if (cfg_mmio_mode != UCT_IB_MLX5_MMIO_MODE_AUTO) {
        *mmio_mode = cfg_mmio_mode;
    } else if (need_lock) {
        *mmio_mode = UCT_IB_MLX5_MMIO_MODE_DB_LOCK;
    } else if (bf_size > 0) {
        if (worker->thread_mode == UCS_THREAD_MODE_SINGLE) {
            *mmio_mode = UCT_IB_MLX5_MMIO_MODE_BF_POST;
        } else if (worker->thread_mode == UCS_THREAD_MODE_SERIALIZED) {
            *mmio_mode = UCT_IB_MLX5_MMIO_MODE_BF_POST_MT;
        } else {
            ucs_error("unsupported thread mode for mlx5: %d", worker->thread_mode);
            return UCS_ERR_UNSUPPORTED;
        }
    } else {
        *mmio_mode = UCT_IB_MLX5_MMIO_MODE_DB;
    }

    return UCS_OK;
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

    status = uct_ib_mlx5_get_mmio_mode(worker, cfg_mmio_mode,
                                       txwq->super.verbs.rd->td == NULL,
                                       qp_info.dv.bf.size, &mmio_mode);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("tx wq %d bytes [bb=%d, nwqe=%d] mmio_mode %s",
              qp_info.dv.sq.stride * qp_info.dv.sq.wqe_cnt,
              qp_info.dv.sq.stride, qp_info.dv.sq.wqe_cnt,
              uct_ib_mlx5_mmio_modes[mmio_mode]);

    txwq->qstart     = qp_info.dv.sq.buf;
    txwq->qend       = UCS_PTR_BYTE_OFFSET(qp_info.dv.sq.buf,
                                           qp_info.dv.sq.stride * qp_info.dv.sq.wqe_cnt);
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

    /* cppcheck-suppress autoVariables */
    txwq->dbrec      = &qp_info.dv.dbrec[MLX5_SND_DBR];
    /* need to reserve 2x because:
     *  - on completion we only get the index of last wqe and we do not
     *    really know how many bb is there (but no more than max bb
     *  - on send we check that there is at least one bb. We know
     *  exact number of bbs once we actually are sending.
     */
    txwq->bb_max     = qp_info.dv.sq.wqe_cnt - 2 * UCT_IB_MLX5_MAX_BB;
    ucs_assert_always(txwq->bb_max > 0);

    uct_ib_mlx5_txwq_reset(txwq);
    return UCS_OK;
}

void *uct_ib_mlx5_txwq_get_wqe(const uct_ib_mlx5_txwq_t *txwq, uint16_t pi)
{
    uint16_t num_bb = UCS_PTR_BYTE_DIFF(txwq->qstart, txwq->qend) /
                      MLX5_SEND_WQE_BB;
    return UCS_PTR_BYTE_OFFSET(txwq->qstart, (pi % num_bb) * MLX5_SEND_WQE_BB);
}

uint16_t uct_ib_mlx5_txwq_num_posted_wqes(const uct_ib_mlx5_txwq_t *txwq,
                                          uint16_t outstanding)
{
    struct mlx5_wqe_ctrl_seg *ctrl;
    uint16_t pi, count;
    size_t wqe_size;

    /* Start iteration with the most recently completed WQE, so count from -1.
       uct_ib_mlx5_txwq_reset() sets qpn_ds in the last WQE in case no WQE has
       completed */
    pi    = txwq->prev_sw_pi - outstanding;
    count = -1;
    ucs_assert(pi == txwq->hw_ci);
    do {
        ctrl     = uct_ib_mlx5_txwq_get_wqe(txwq, pi);
        wqe_size = (ctrl->qpn_ds >> 24) * UCT_IB_MLX5_WQE_SEG_SIZE;
        pi      += (wqe_size + MLX5_SEND_WQE_BB - 1) / MLX5_SEND_WQE_BB;
        ++count;
    } while (pi != txwq->sw_pi);

    return count;
}

void uct_ib_mlx5_qp_mmio_cleanup(uct_ib_mlx5_qp_t *qp,
                                 uct_ib_mlx5_mmio_reg_t *reg)
{
    uct_ib_mlx5_devx_uar_t *uar = ucs_derived_of(reg, uct_ib_mlx5_devx_uar_t);

    switch (qp->type) {
    case UCT_IB_MLX5_OBJ_TYPE_DEVX:
        uct_worker_tl_data_put(uar, uct_ib_mlx5_devx_uar_cleanup);
        break;
    case UCT_IB_MLX5_OBJ_TYPE_VERBS:
        uct_ib_mlx5_iface_put_res_domain(qp);
        uct_worker_tl_data_put(reg, uct_ib_mlx5_mmio_cleanup);
        break;
    case UCT_IB_MLX5_OBJ_TYPE_NULL:
        ucs_fatal("qp %p: TYPE_NULL", qp);
    case UCT_IB_MLX5_OBJ_TYPE_LAST:
        if (reg != NULL) {
            uct_worker_tl_data_put(reg, uct_ib_mlx5_mmio_cleanup);
        }
    }
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
    /* cppcheck-suppress autoVariables */
    rxwq->dbrec           = &qp_info.dv.dbrec[MLX5_RCV_DBR];
    memset(rxwq->wqes, 0, qp_info.dv.rq.wqe_cnt * sizeof(struct mlx5_wqe_data_seg));

    return UCS_OK;
}

ucs_status_t
uct_ib_mlx5_verbs_srq_init(uct_ib_mlx5_srq_t *srq, struct ibv_srq *verbs_srq,
                           size_t sg_byte_count, int sge_num)
{
    uct_ib_mlx5dv_srq_t srq_info = {};
    uct_ib_mlx5dv_t obj          = {};
    ucs_status_t status;
    uint16_t stride;

    obj.dv.srq.in         = verbs_srq;
    obj.dv.srq.out        = &srq_info.dv;
#if HAVE_DEVX
    srq_info.dv.comp_mask = MLX5DV_SRQ_MASK_SRQN;
#endif

    status = uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_SRQ);
    if (status != UCS_OK) {
        return status;
    }

#if HAVE_DEVX
    srq->srq_num = srq_info.dv.srqn;
#else
    srq->srq_num = 0;
#endif

    if (srq_info.dv.head != 0) {
        ucs_error("SRQ head is not 0 (%d)", srq_info.dv.head);
        return UCS_ERR_NO_DEVICE;
    }

    stride = uct_ib_mlx5_srq_stride(sge_num);
    if (srq_info.dv.stride != stride) {
        ucs_error("SRQ stride is not %u (%d), sgenum %d",
                  stride, srq_info.dv.stride, sge_num);
        return UCS_ERR_NO_DEVICE;
    }

    if (!ucs_is_pow2(srq_info.dv.tail + 1)) {
        ucs_error("SRQ length is not power of 2 (%d)", srq_info.dv.tail + 1);
        return UCS_ERR_NO_DEVICE;
    }

    srq->buf = srq_info.dv.buf;
    srq->db  = srq_info.dv.dbrec;
    uct_ib_mlx5_srq_buff_init(srq, srq_info.dv.head, srq_info.dv.tail,
                              sg_byte_count, sge_num);

    return UCS_OK;
}

void uct_ib_mlx5_srq_buff_init(uct_ib_mlx5_srq_t *srq, uint32_t head,
                               uint32_t tail, size_t sg_byte_count, int sge_num)
{
    uct_ib_mlx5_srq_seg_t *seg;
    unsigned i, j;

    srq->free_idx  = tail;
    srq->ready_idx = UINT16_MAX;
    srq->sw_pi     = UINT16_MAX;
    srq->mask      = tail;
    srq->stride    = uct_ib_mlx5_srq_stride(sge_num);

    for (i = head; i <= tail; ++i) {
        seg = uct_ib_mlx5_srq_get_wqe(srq, i);
        seg->srq.next_wqe_index = htons((i + 1) & tail);
        seg->srq.ptr_mask       = 0;
        seg->srq.free           = 0;
        seg->srq.desc           = NULL;
        seg->srq.strides        = sge_num;
        for (j = 0; j < sge_num; ++j) {
            seg->dptr[j].byte_count = htonl(sg_byte_count);
        }
    }
}

ucs_status_t uct_ib_mlx5_modify_qp_state(uct_ib_iface_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         enum ibv_qp_state state)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.md, uct_ib_mlx5_md_t);

    ucs_debug("device %s: modify QP %p num 0x%x to state %d",
              md->super.dev.ibv_context->device->dev_name, qp, qp->qp_num,
              state);

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX) {
        return uct_ib_mlx5_devx_modify_qp_state(qp, state);
    } else {
        return uct_ib_modify_qp(qp->verbs.qp, state);
    }
}

ucs_status_t
uct_ib_mlx5_query_qp_peer_info(uct_ib_iface_t *iface, uct_ib_mlx5_qp_t *qp,
                               struct ibv_ah_attr *ah_attr, uint32_t *dest_qpn)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.md, uct_ib_mlx5_md_t);

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX) {
        return uct_ib_mlx5_devx_query_qp_peer_info(iface, qp, ah_attr,
                                                   dest_qpn);
    } else {
        return uct_ib_query_qp_peer_info(qp->verbs.qp, ah_attr, dest_qpn);
    }
}

void uct_ib_mlx5_destroy_qp(uct_ib_mlx5_md_t *md, uct_ib_mlx5_qp_t *qp)
{
    switch (qp->type) {
    case UCT_IB_MLX5_OBJ_TYPE_VERBS:
        uct_ib_destroy_qp(qp->verbs.qp);
        break;
    case UCT_IB_MLX5_OBJ_TYPE_DEVX:
        uct_ib_mlx5_devx_destroy_qp(md, qp);
        break;
    case UCT_IB_MLX5_OBJ_TYPE_NULL:
        ucs_fatal("md %p: qp %p: TYPE_NULL", md, qp);
    case UCT_IB_MLX5_OBJ_TYPE_LAST:
        break;
    }
}

size_t uct_ib_mlx5_devx_sq_length(size_t tx_qp_length)
{
    return ucs_roundup_pow2_or0(tx_qp_length * UCT_IB_MLX5_MAX_BB);
}

/* Keep the function as a separate to test SL selection */
ucs_status_t uct_ib_mlx5_select_sl(const uct_ib_iface_config_t *ib_config,
                                   ucs_ternary_auto_value_t ar_enable,
                                   uint16_t hw_sl_mask, int have_sl_mask_cap,
                                   const char *dev_name, uint8_t port_num,
                                   uint8_t *sl_p)
{
    ucs_status_t status = UCS_OK;
    const char *sl_ar_support_str;
    uint16_t sl_allow_mask, sls_with_ar, sls_without_ar;
    ucs_string_buffer_t sls_with_ar_str, sls_without_ar_str;
    char sl_str[8];
    char ar_enable_str[8];
    uint8_t sl;

    ucs_assert(have_sl_mask_cap || (hw_sl_mask == 0));

    /* which SLs are allowed by user config */
    sl_allow_mask = (ib_config->sl == UCS_ULUNITS_AUTO) ?
                    UCS_MASK(UCT_IB_SL_NUM) : UCS_BIT(ib_config->sl);

    if (have_sl_mask_cap) {
        sls_with_ar    = sl_allow_mask & hw_sl_mask;
        sls_without_ar = sl_allow_mask & ~hw_sl_mask;
    } else {
        sls_with_ar    =
        sls_without_ar = 0;
    }

    ucs_string_buffer_init(&sls_with_ar_str);
    ucs_string_buffer_init(&sls_without_ar_str);

    if (ar_enable == UCS_AUTO) {
        /* selects SL requested by a user */
        sl                    = ucs_ffs64(sl_allow_mask);
        if (have_sl_mask_cap) {
            sl_ar_support_str = (sl & sls_with_ar) ? "yes" : "no";
        } else {
            sl_ar_support_str = "unknown";
        }
    } else if (((ar_enable == UCS_YES) || (ar_enable == UCS_TRY)) &&
               (sls_with_ar != 0)) {
        /* have SLs with AR, and AR is YES/TRY */
        sl                = ucs_ffs64(sls_with_ar);
        sl_ar_support_str = "yes";
    } else if (((ar_enable == UCS_NO) || (ar_enable == UCS_TRY)) &&
               (sls_without_ar != 0)) {
        /* have SLs without AR, and AR is NO/TRY */
        sl                = ucs_ffs64(sls_without_ar);
        sl_ar_support_str = "no";
    } else if (ar_enable == UCS_TRY) {
        ucs_assert(!have_sl_mask_cap);
        sl                = ucs_ffs64(sl_allow_mask);
        sl_ar_support_str = "unknown"; /* we don't know which SLs support AR */
    } else {
        sl_ar_support_str = (ar_enable == UCS_YES) ? "with" : "without";
        goto err;
    }

    *sl_p = sl;
    ucs_debug("SL=%u (AR support - %s) was selected on %s:%u,"
              " SLs with AR support = { %s }, SLs without AR support = { %s }",
              sl, sl_ar_support_str, dev_name, port_num,
              ucs_mask_str(sls_with_ar, &sls_with_ar_str),
              ucs_mask_str(sls_without_ar, &sls_without_ar_str));

out_str_buf_clean:
    ucs_string_buffer_cleanup(&sls_with_ar_str);
    ucs_string_buffer_cleanup(&sls_without_ar_str);
    return status;

err:
    ucs_assert(ar_enable != UCS_TRY);
    ucs_config_sprintf_ulunits(sl_str, sizeof(sl_str), &ib_config->sl, NULL);
    ucs_config_sprintf_ternary_auto(ar_enable_str, sizeof(ar_enable_str),
                                    &ar_enable, NULL);
    ucs_error("AR=%s was requested for SL=%s, but %s %s AR on %s:%u,"
              " SLs with AR support = { %s }, SLs without AR support = { %s }",
              ar_enable_str, sl_str,
              have_sl_mask_cap ? "could not select SL" :
              "could not detect AR mask for SLs. Please, set SL manually",
              sl_ar_support_str, dev_name, port_num,
              ucs_mask_str(sls_with_ar, &sls_with_ar_str),
              ucs_mask_str(sls_without_ar, &sls_without_ar_str));
    status = UCS_ERR_UNSUPPORTED;
    goto out_str_buf_clean;
}

ucs_status_t
uct_ib_mlx5_iface_select_sl(uct_ib_iface_t *iface,
                            const uct_ib_mlx5_iface_config_t *ib_mlx5_config,
                            const uct_ib_iface_config_t *ib_config)
{
#if HAVE_DEVX
    uct_ib_mlx5_md_t *md = ucs_derived_of(iface->super.md, uct_ib_mlx5_md_t);
#endif
    const char *dev_name = uct_ib_device_name(uct_ib_iface_device(iface));
    uint16_t ooo_sl_mask = 0;
    ucs_status_t status;

    ucs_assert(iface->config.sl == UCT_IB_SL_NUM);

    if (uct_ib_device_is_port_roce(uct_ib_iface_device(iface),
                                   iface->config.port_num)) {
        /* Ethernet priority for RoCE devices can't be selected regardless
         * AR support requested by user, pass empty ooo_sl_mask */
        status = uct_ib_mlx5_select_sl(ib_config, UCS_NO, 0, 1, dev_name,
                                       iface->config.port_num,
                                       &iface->config.sl);
        if (status != UCS_OK) {
            goto out;
        }

        uct_ib_iface_set_reverse_sl(iface, ib_config);
        return status;
    }

#if HAVE_DEVX
    status = uct_ib_mlx5_devx_query_ooo_sl_mask(md, iface->config.port_num,
                                                &ooo_sl_mask);
    if ((status != UCS_OK) && (status != UCS_ERR_UNSUPPORTED)) {
        return status;
    }
#else
    status = UCS_ERR_UNSUPPORTED;
#endif

    status = uct_ib_mlx5_select_sl(ib_config, ib_mlx5_config->ar_enable,
                                   ooo_sl_mask, status == UCS_OK, dev_name,
                                   iface->config.port_num, &iface->config.sl);
    if (status != UCS_OK) {
        goto out;
    }

    uct_ib_iface_set_reverse_sl(iface, ib_config);

out:
    return status;
}

uint8_t uct_ib_mlx5_iface_get_counter_set_id(uct_ib_iface_t *iface)
{
    if (iface->config.counter_set_id != UCT_IB_COUNTER_SET_ID_INVALID) {
        return iface->config.counter_set_id;
    }

    return uct_ib_mlx5_devx_md_get_counter_set_id(uct_ib_mlx5_iface_md(iface),
                                                  iface->config.port_num);
}

void uct_ib_mlx5_txwq_validate_always(uct_ib_mlx5_txwq_t *wq, uint16_t num_bb,
                                      int hw_ci_updated)
{
#if UCS_ENABLE_ASSERT
    uint16_t wqe_first_bb, wqe_last_pi;
    uint16_t qp_length, max_pi;
    uint16_t hw_ci;

    /* num_bb must be non-zero and not larger than MAX_BB */
    ucs_assertv((num_bb > 0) && (num_bb <= UCT_IB_MLX5_MAX_BB), "num_bb=%u",
                num_bb);

    /* bb_max must be smaller than the full QP length */
    qp_length = UCS_PTR_BYTE_DIFF(wq->qstart, wq->qend) / MLX5_SEND_WQE_BB;
    ucs_assertv(wq->bb_max < qp_length, "bb_max=%u qp_length=%u ", wq->bb_max,
                qp_length);

    /* wq->curr and wq->sw_pi should be in sync */
    wqe_first_bb = UCS_PTR_BYTE_DIFF(wq->qstart, wq->curr) / MLX5_SEND_WQE_BB;
    ucs_assertv(wqe_first_bb == (wq->sw_pi % qp_length),
                "first_bb=%u sw_pi=%u qp_length=%u", wqe_first_bb, wq->sw_pi,
                qp_length);

    /* sw_pi must be ahead of prev_sw_pi */
    ucs_assertv(UCS_CIRCULAR_COMPARE16(wq->sw_pi, >, wq->prev_sw_pi),
                "sw_pi=%u prev_sw_pi=%u", wq->sw_pi, wq->prev_sw_pi);

    if (!hw_ci_updated) {
        return;
    }

    hw_ci = wq->hw_ci;

    /* hw_ci must be less or equal to prev_sw_pi, since we could get completion
       only for what was actually posted */
    ucs_assertv(UCS_CIRCULAR_COMPARE16(hw_ci, <=, wq->prev_sw_pi),
                "hw_ci=%u prev_sw_pi=%u", hw_ci, wq->prev_sw_pi);

    /* Check for QP overrun: our WQE's last BB index must be <= hw_ci+qp_length.
       max_pi is the largest BB index that is guaranteed to be free. */
    wqe_last_pi = wq->sw_pi + num_bb - 1;
    max_pi      = hw_ci + qp_length;
    ucs_assertv(UCS_CIRCULAR_COMPARE16(wqe_last_pi, <=, max_pi),
                "TX WQ overrun: wq=%p wqe_last_pi=%u max_pi=%u sw_pi=%u "
                "num_bb=%u hw_ci=%u qp_length=%u",
                wq, wqe_last_pi, max_pi, wq->sw_pi, num_bb, hw_ci, qp_length);
#endif
}

extern uct_tl_t UCT_TL_NAME(dc_mlx5);
extern uct_tl_t UCT_TL_NAME(rc_mlx5);
extern uct_tl_t UCT_TL_NAME(ud_mlx5);

extern uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(devx);
extern uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(dv);

void UCS_F_CTOR uct_mlx5_init(void)
{
#if defined (HAVE_MLX5_DV)
    ucs_list_add_head(&uct_ib_ops, &UCT_IB_MD_OPS_NAME(dv).list);
#endif
#if defined (HAVE_DEVX)
    ucs_list_add_head(&uct_ib_ops, &UCT_IB_MD_OPS_NAME(devx).list);
#endif

#ifdef HAVE_TL_DC
    uct_tl_register(&uct_ib_component, &UCT_TL_NAME(dc_mlx5));
#endif
#if defined (HAVE_TL_RC) && defined (HAVE_MLX5_DV)
    uct_tl_register(&uct_ib_component, &UCT_TL_NAME(rc_mlx5));
#endif
#if defined (HAVE_TL_UD) && defined (HAVE_MLX5_HW_UD)
    uct_tl_register(&uct_ib_component, &UCT_TL_NAME(ud_mlx5));
#endif
}

void UCS_F_DTOR uct_mlx5_cleanup(void)
{
#if defined (HAVE_TL_UD) && defined (HAVE_MLX5_HW_UD)
    uct_tl_unregister(&UCT_TL_NAME(ud_mlx5));
#endif
#if defined (HAVE_TL_RC) && defined (HAVE_MLX5_DV)
    uct_tl_unregister(&UCT_TL_NAME(rc_mlx5));
#endif
#ifdef HAVE_TL_DC
    uct_tl_unregister(&UCT_TL_NAME(dc_mlx5));
#endif

#if defined (HAVE_MLX5_DV)
    ucs_list_del(&UCT_IB_MD_OPS_NAME(dv).list);
#endif
#if defined (HAVE_DEVX)
    ucs_list_del(&UCT_IB_MD_OPS_NAME(devx).list);
#endif
}
