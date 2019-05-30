/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ib_mlx5.h"
#include "ib_mlx5_log.h"
#include "ib_mlx5_ifc.h"

#include <ucs/arch/bitops.h>
#include <ucs/profile/profile.h>

typedef struct uct_ib_mlx5_mem {
    uct_ib_mem_t             super;
#if HAVE_DEVX
    struct mlx5dv_devx_obj   *atomic_dvmr;
#elif HAVE_EXP_UMR
    struct ibv_mr            *atomic_mr;
#endif
} uct_ib_mlx5_mem_t;

typedef struct uct_ib_mlx5_dbrec_page {
    struct mlx5dv_devx_umem *mem;
} uct_ib_mlx5_dbrec_page_t;


uint8_t uct_ib_mlx5_md_get_atomic_mr_id(uct_ib_mlx5_md_t *md)
{
#if HAVE_EXP_UMR
    if ((md->umr_qp == NULL) || (md->umr_cq == NULL)) {
        return 0;
    }
    /* Generate atomic UMR id. We want umrs for same virtual addresses to have
     * different ids across processes.
     *
     * Usually parallel processes running on the same node as part of a single
     * job will have consecutive PIDs. For example MPI ranks, slurm spawned tasks...
     */
    return getpid() % 256;
#else
    return 0;
#endif
}

static uct_ib_md_ops_t uct_ib_mlx5_md_ops;

#if HAVE_DEVX

static ucs_status_t uct_ib_mlx5dv_reg_atomic_key(uct_ib_md_t *ibmd,
                                                 uct_ib_mem_t *ib_memh)
{
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);
    off_t offset = uct_ib_md_atomic_offset(uct_ib_mlx5_md_get_atomic_mr_id(md));
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(create_mkey_out)] = {};
    struct ibv_mr *mr = memh->super.mr;
    ucs_status_t status = UCS_OK;
    struct mlx5dv_pd dvpd = {};
    struct mlx5dv_obj dv = {};
    size_t reg_length, length, inlen;
    int list_size, i;
    void *mkc, *klm;
    uint32_t *in;
    intptr_t addr;

    if (!(md->flags & UCT_IB_MLX5_MD_FLAG_KSM)) {
        return UCS_ERR_UNSUPPORTED;
    }

    reg_length = UCT_IB_MD_MAX_MR_SIZE;
    addr       = (intptr_t)mr->addr & ~(reg_length - 1);
    length     = mr->length + (intptr_t)mr->addr - addr;
    list_size  = ucs_div_round_up(length, reg_length);
    inlen      = UCT_IB_MLX5DV_ST_SZ_BYTES(create_mkey_in) +
                 UCT_IB_MLX5DV_ST_SZ_BYTES(klm) * list_size;

    in         = ucs_calloc(1, inlen, "mkey mailbox");
    if (in == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    dv.pd.in   = md->super.pd;
    dv.pd.out  = &dvpd;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_PD);

    UCT_IB_MLX5DV_SET(create_mkey_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_MKEY);
    mkc = UCT_IB_MLX5DV_ADDR_OF(create_mkey_in, in, memory_key_mkey_entry);
    UCT_IB_MLX5DV_SET(mkc, mkc, access_mode_1_0, UCT_IB_MLX5_MKC_ACCESS_MODE_KSM);
    UCT_IB_MLX5DV_SET(mkc, mkc, a, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, rw, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, rr, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, lw, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, lr, 1);
    UCT_IB_MLX5DV_SET(mkc, mkc, pd, dvpd.pdn);
    UCT_IB_MLX5DV_SET(mkc, mkc, translations_octword_size, list_size);
    UCT_IB_MLX5DV_SET(mkc, mkc, log_entity_size, ucs_ilog2(reg_length));
    UCT_IB_MLX5DV_SET(mkc, mkc, qpn, 0xffffff);
    UCT_IB_MLX5DV_SET(mkc, mkc, mkey_7_0, offset & 0xff);
    UCT_IB_MLX5DV_SET64(mkc, mkc, start_addr, addr + offset);
    UCT_IB_MLX5DV_SET64(mkc, mkc, len, length);
    UCT_IB_MLX5DV_SET(create_mkey_in, in, translations_octword_actual_size, list_size);

    klm = UCT_IB_MLX5DV_ADDR_OF(create_mkey_in, in, klm_pas_mtt);
    for (i = 0; i < list_size; i++) {
        if (i == list_size - 1) {
            UCT_IB_MLX5DV_SET(klm, klm, byte_count, length % reg_length);
        } else {
            UCT_IB_MLX5DV_SET(klm, klm, byte_count, reg_length);
        }
        UCT_IB_MLX5DV_SET(klm, klm, mkey, mr->lkey);
        UCT_IB_MLX5DV_SET64(klm, klm, address, addr + (i * reg_length));
        klm += UCT_IB_MLX5DV_ST_SZ_BYTES(klm);
    }

    memh->atomic_dvmr = mlx5dv_devx_obj_create(md->super.dev.ibv_context, in, inlen,
                                               out, sizeof(out));
    if (memh->atomic_dvmr == NULL) {
        ucs_debug("mlx5dv_devx_obj_create(CREATE_MKEY, mode=KSM) failed: %m");
        status = UCS_ERR_UNSUPPORTED;
        md->flags &= ~UCT_IB_MLX5_MD_FLAG_KSM;
        goto out;
    }

    memh->super.atomic_rkey =
        (UCT_IB_MLX5DV_GET(create_mkey_out, out, mkey_index) << 8) |
        (offset & 0xff);

    ucs_debug("KSM registered memory %p..%p offset 0x%lx on %s rkey 0x%x",
              mr->addr, mr->addr + mr->length, offset, uct_ib_device_name(&md->super.dev),
              memh->super.atomic_rkey);
out:
    ucs_free(in);
    return status;
}

static ucs_status_t uct_ib_mlx5dv_dereg_atomic_key(uct_ib_md_t *ibmd,
                                                   uct_ib_mem_t *ib_memh)
{
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    int ret;

    ret = mlx5dv_devx_obj_destroy(memh->atomic_dvmr);
    if (ret != 0) {
        ucs_error("mlx5dv_devx_obj_destroy(MKEY, KSM) failed: %m");
        return UCS_ERR_IO_ERROR;
    }
    return UCS_OK;
}

static ucs_status_t uct_ib_mlx5_add_page(ucs_mpool_t *mp, size_t *size_p, void **page_p)
{
    uct_ib_mlx5_md_t *md = ucs_container_of(mp, uct_ib_mlx5_md_t, dbrec_pool);
    uintptr_t ps = ucs_get_page_size();
    uct_ib_mlx5_dbrec_page_t *page;
    size_t size = ucs_align_up(*size_p + sizeof(*page), ps);

    page = ucs_memalign(ps, size, "devx dbrec");
    if (page == NULL) {
        goto err;
    }

    page->mem = mlx5dv_devx_umem_reg(md->super.dev.ibv_context, page, size,
                                     IBV_ACCESS_LOCAL_WRITE);
    if (page->mem == NULL) {
        goto err_free;
    }

    *size_p = size;
    *page_p = page + 1;
    return UCS_OK;

err_free:
    ucs_free(page);
err:
    return UCS_ERR_IO_ERROR;
}

static void uct_ib_mlx5_init_dbrec(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_ib_mlx5_dbrec_page_t *page = chunk - sizeof(*page);
    uct_ib_mlx5_dbrec_t *dbrec     = obj;

    dbrec->mem_id = page->mem->umem_id;
    dbrec->offset = obj - chunk + sizeof(*page);
}

static void uct_ib_mlx5_free_page(ucs_mpool_t *mp, void *chunk)
{
    uct_ib_mlx5_dbrec_page_t *page = chunk - sizeof(*page);
    mlx5dv_devx_umem_dereg(page->mem);
    ucs_free(page);
}

static ucs_mpool_ops_t uct_ib_mlx5_dbrec_ops = {
    .chunk_alloc   = uct_ib_mlx5_add_page,
    .chunk_release = uct_ib_mlx5_free_page,
    .obj_init      = uct_ib_mlx5_init_dbrec,
    .obj_cleanup   = NULL
};

static ucs_status_t uct_ib_mlx5dv_md_open(struct ibv_device *ibv_device,
                                          const uct_ib_md_config_t *md_config,
                                          uct_ib_md_t **p_md)
{
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(query_hca_cap_out)] = {};
    uint32_t in[UCT_IB_MLX5DV_ST_SZ_DW(query_hca_cap_in)] = {};
    struct mlx5dv_context_attr dv_attr = {};
    ucs_status_t status = UCS_OK;
    struct ibv_context *ctx;
    uct_ib_device_t *dev;
    uct_ib_mlx5_md_t *md;
    void *cap;
    int ret;

#if HAVE_DECL_MLX5DV_IS_SUPPORTED
    if (!mlx5dv_is_supported(ibv_device)) {
        return UCS_ERR_UNSUPPORTED;
    }
#endif

    dv_attr.flags |= MLX5DV_CONTEXT_FLAGS_DEVX;
    ctx = mlx5dv_open_device(ibv_device, &dv_attr);
    if (ctx == NULL) {
        ucs_debug("mlx5dv_open_device(%s) failed: %m", ibv_get_device_name(ibv_device));
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    md = ucs_calloc(1, sizeof(*md), "ib_mlx5_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context;
    }

    dev              = &md->super.dev;
    dev->ibv_context = ctx;

    IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(&dev->dev_attr);
    ret = ibv_query_device_ex(dev->ibv_context, NULL, &dev->dev_attr);
    if (ret != 0) {
        ucs_error("ibv_query_device_ex() returned %d: %m", ret);
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    }

    cap = UCT_IB_MLX5DV_ADDR_OF(query_hca_cap_out, out, capability);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_MAX |
                                                   (UCT_IB_MLX5_CAP_GENERAL << 1));
    ret = mlx5dv_devx_general_cmd(ctx, in, sizeof(in), out, sizeof(out));
    if (ret != 0) {
        if ((errno == EPERM) || (errno == EPROTONOSUPPORT) ||
            (errno == EOPNOTSUPP)) {
            status = UCS_ERR_UNSUPPORTED;
            ucs_debug("mlx5dv_devx_general_cmd(QUERY_HCA_CAP) failed: %m");
        } else {
            ucs_error("mlx5dv_devx_general_cmd(QUERY_HCA_CAP) failed: %m");
            status = UCS_ERR_IO_ERROR;
        }
        goto err_free;

    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, dct)) {
        dev->flags |= UCT_IB_DEVICE_FLAG_DC;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, compact_address_vector)) {
        dev->flags |= UCT_IB_DEVICE_FLAG_AV;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, fixed_buffer_size)) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_KSM;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, atomic)) {
        int ops = UCT_IB_MLX5_ATOMIC_OPS_CMP_SWAP |
                  UCT_IB_MLX5_ATOMIC_OPS_FETCH_ADD;
        uint8_t arg_size;
        int cap_ops, mode8b;

        UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_MAX |
                                                       (UCT_IB_MLX5_CAP_ATOMIC << 1));
        ret = mlx5dv_devx_general_cmd(ctx, in, sizeof(in), out, sizeof(out));
        if (ret != 0) {
            ucs_error("mlx5dv_devx_general_cmd(QUERY_HCA_CAP, ATOMIC) failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_free;
        }

        arg_size = UCT_IB_MLX5DV_GET(atomic_caps, cap, atomic_size_qp);
        cap_ops  = UCT_IB_MLX5DV_GET(atomic_caps, cap, atomic_operations);
        mode8b   = UCT_IB_MLX5DV_GET(atomic_caps, cap, atomic_req_8B_endianness_mode);

        if ((cap_ops & ops) == ops) {
            dev->atomic_arg_sizes = sizeof(uint64_t);
            if (!mode8b) {
                dev->atomic_arg_sizes_be = sizeof(uint64_t);
            }
        }

        ops |= UCT_IB_MLX5_ATOMIC_OPS_MASKED_CMP_SWAP |
               UCT_IB_MLX5_ATOMIC_OPS_MASKED_FETCH_ADD;

        arg_size &= UCT_IB_MLX5DV_GET(query_hca_cap_out, out,
                                      capability.atomic_caps.atomic_size_dc);

        if ((cap_ops & ops) == ops) {
            dev->ext_atomic_arg_sizes = arg_size;
            if (mode8b) {
                arg_size &= ~(sizeof(uint64_t));
            }
            dev->ext_atomic_arg_sizes_be = arg_size;
        }

        dev->pci_fadd_arg_sizes  = UCT_IB_MLX5DV_GET(atomic_caps, cap, fetch_add_pci_atomic) << 2;
        dev->pci_cswap_arg_sizes = UCT_IB_MLX5DV_GET(atomic_caps, cap, compare_swap_pci_atomic) << 2;
    }

    md->super.ops = &uct_ib_mlx5_md_ops;
    status = uct_ib_md_open_common(&md->super, ibv_device, md_config);
    if (status != UCS_OK) {
        goto err_free;
    }

    status = ucs_mpool_init(&md->dbrec_pool, 0,
                            sizeof(uct_ib_mlx5_dbrec_t), 0,
                            UCS_SYS_CACHE_LINE_SIZE,
                            ucs_get_page_size() / UCS_SYS_CACHE_LINE_SIZE - 1,
                            UINT_MAX, &uct_ib_mlx5_dbrec_ops, "devx dbrec");
    if (status != UCS_OK) {
        goto err_free;
    }

    dev->flags |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
    md->flags |= UCT_IB_MLX5_MD_FLAG_DEVX;
    *p_md = &md->super;
    return status;

err_free:
    ucs_free(md);
err_free_context:
    ibv_close_device(ctx);
err:
    return status;
}

#else

/* TODO: split this function into DV and EXP implementations */
static ucs_status_t uct_ib_mlx5_check_dc(uct_ib_device_t *dev)
{
    ucs_status_t status = UCS_OK;
#if HAVE_DC_DV
    struct ibv_srq_init_attr srq_attr = {};
    struct ibv_context *ctx = dev->ibv_context;
    struct ibv_qp_init_attr_ex qp_attr = {};
    struct mlx5dv_qp_init_attr dv_attr = {};
    struct ibv_qp_attr attr = {};
    struct ibv_srq *srq;
    struct ibv_pd *pd;
    struct ibv_cq *cq;
    struct ibv_qp *qp;
    int ret;

    pd = ibv_alloc_pd(ctx);
    if (pd == NULL) {
        ucs_error("ibv_alloc_pd() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    cq = ibv_create_cq(ctx, 1, NULL, NULL, 0);
    if (cq == NULL) {
        ucs_error("ibv_create_cq() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_cq;
    }

    srq_attr.attr.max_sge   = 1;
    srq_attr.attr.max_wr    = 1;
    srq = ibv_create_srq(pd, &srq_attr);
    if (srq == NULL) {
        ucs_error("ibv_create_srq() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_srq;
    }

    qp_attr.send_cq              = cq;
    qp_attr.recv_cq              = cq;
    qp_attr.qp_type              = IBV_QPT_DRIVER;
    qp_attr.comp_mask            = IBV_QP_INIT_ATTR_PD;
    qp_attr.pd                   = pd;
    qp_attr.srq                  = srq;

    dv_attr.comp_mask            = MLX5DV_QP_INIT_ATTR_MASK_DC;
    dv_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
    dv_attr.dc_init_attr.dct_access_key = UCT_IB_KEY;

    /* create DCT qp successful means DC is supported */
    qp = mlx5dv_create_qp(ctx, &qp_attr, &dv_attr);
    if (qp == NULL) {
        goto err_qp;
    }

    attr.qp_state        = IBV_QPS_INIT;
    attr.port_num        = 1;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ  |
                           IBV_ACCESS_REMOTE_ATOMIC;
    ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE |
                                   IBV_QP_PKEY_INDEX |
                                   IBV_QP_PORT |
                                   IBV_QP_ACCESS_FLAGS);
    if (ret != 0) {
        goto err;
    }

    attr.qp_state         = IBV_QPS_RTR;
    attr.path_mtu         = IBV_MTU_256;
    attr.ah_attr.port_num = 1;

    ret = ibv_modify_qp(qp, &attr, IBV_QP_STATE |
                                   IBV_QP_MIN_RNR_TIMER |
                                   IBV_QP_AV |
                                   IBV_QP_PATH_MTU);

    if (ret == 0) {
        dev->flags |= UCT_IB_DEVICE_FLAG_DC;
    }

err:
    ibv_destroy_qp(qp);
err_qp:
    ibv_destroy_srq(srq);
err_srq:
    ibv_destroy_cq(cq);
err_cq:
    ibv_dealloc_pd(pd);
#elif HAVE_DECL_IBV_EXP_DEVICE_DC_TRANSPORT && HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_EXP_DEVICE_CAP_FLAGS
    if (dev->dev_attr.exp_device_cap_flags & IBV_EXP_DEVICE_DC_TRANSPORT) {
        dev->flags |= UCT_IB_DEVICE_FLAG_DC;
    }
#endif
    return status;
}

static ucs_status_t uct_ib_mlx5_md_umr_qp_create(uct_ib_mlx5_md_t *md)
{
#if HAVE_EXP_UMR
    struct ibv_exp_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    uint8_t port_num;
    int ret;
    uct_ib_device_t *ibdev;
    struct ibv_exp_port_attr *port_attr;
    int is_roce_v2;

    ibdev = &md->super.dev;

    if (!(ibdev->dev_attr.exp_device_cap_flags & IBV_EXP_DEVICE_UMR) ||
        !md->super.config.enable_indirect_atomic) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* TODO: fix port selection. It looks like active port should be used */
    port_num = ibdev->first_port;
    port_attr = uct_ib_device_port_attr(ibdev, port_num);

    memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    md->umr_cq = ibv_create_cq(ibdev->ibv_context, 1, NULL, NULL, 0);
    if (md->umr_cq == NULL) {
        ucs_error("failed to create UMR CQ: %m");
        goto err;
    }

    md->super.config.max_inline_klm_list =
        ucs_min(md->super.config.max_inline_klm_list,
                ibdev->dev_attr.umr_caps.max_send_wqe_inline_klms);

    qp_init_attr.qp_type             = IBV_QPT_RC;
    qp_init_attr.send_cq             = md->umr_cq;
    qp_init_attr.recv_cq             = md->umr_cq;
    qp_init_attr.cap.max_inline_data = 0;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_send_sge    = 1;
    qp_init_attr.srq                 = NULL;
    qp_init_attr.cap.max_recv_wr     = 16;
    qp_init_attr.cap.max_send_wr     = 16;
    qp_init_attr.pd                  = md->super.pd;
    qp_init_attr.comp_mask           = IBV_EXP_QP_INIT_ATTR_PD|IBV_EXP_QP_INIT_ATTR_MAX_INL_KLMS;
    qp_init_attr.max_inl_recv        = 0;
    qp_init_attr.max_inl_send_klms   = md->super.config.max_inline_klm_list;

#if HAVE_IBV_EXP_QP_CREATE_UMR
    qp_init_attr.comp_mask          |= IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
    qp_init_attr.exp_create_flags    = IBV_EXP_QP_CREATE_UMR;
#endif

    md->umr_qp = ibv_exp_create_qp(ibdev->ibv_context, &qp_init_attr);
    if (md->umr_qp == NULL) {
        ucs_error("failed to create UMR QP: %m");
        goto err_destroy_cq;
    }

    memset(&qp_attr, 0, sizeof(qp_attr));

    /* Modify QP to INIT state */
    qp_attr.qp_state                 = IBV_QPS_INIT;
    qp_attr.pkey_index               = 0;
    qp_attr.port_num                 = port_num;
    qp_attr.qp_access_flags          = UCT_IB_MEM_ACCESS_FLAGS;
    ret = ibv_modify_qp(md->umr_qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        ucs_error("Failed to modify UMR QP to INIT: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTR */
    qp_attr.qp_state                 = IBV_QPS_RTR;
    qp_attr.dest_qp_num              = md->umr_qp->qp_num;

    memset(&qp_attr.ah_attr, 0, sizeof(qp_attr.ah_attr));
    qp_attr.ah_attr.port_num         = port_num;
    qp_attr.ah_attr.dlid             = port_attr->lid;
    qp_attr.ah_attr.is_global        = 1;
    if (uct_ib_device_query_gid(ibdev, port_num, UCT_IB_MD_DEFAULT_GID_INDEX,
                                &qp_attr.ah_attr.grh.dgid, &is_roce_v2) != UCS_OK) {
        goto err_destroy_qp;
    }

    qp_attr.rq_psn                   = 0;
    qp_attr.path_mtu                 = IBV_MTU_512;
    qp_attr.min_rnr_timer            = 7;
    qp_attr.max_dest_rd_atomic       = 1;
    ret = ibv_modify_qp(md->umr_qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        ucs_error("Failed to modify UMR QP to RTR: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTS */
    qp_attr.qp_state                 = IBV_QPS_RTS;
    qp_attr.sq_psn                   = 0;
    qp_attr.timeout                  = 7;
    qp_attr.rnr_retry                = 7;
    qp_attr.retry_cnt                = 7;
    qp_attr.max_rd_atomic            = 1;
    ret = ibv_modify_qp(md->umr_qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        ucs_error("Failed to modify UMR QP to RTS: %m");
        goto err_destroy_qp;
    }

    ucs_debug("initialized UMR QP 0x%x, max_inline_klm_list %u",
              md->umr_qp->qp_num, md->super.config.max_inline_klm_list);
    return UCS_OK;

err_destroy_qp:
    ibv_destroy_qp(md->umr_qp);
err_destroy_cq:
    ibv_destroy_cq(md->umr_cq);
err:
    return UCS_ERR_IO_ERROR;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t uct_ib_mlx5_verbs_reg_atomic_key(uct_ib_md_t *ibmd,
                                                     uct_ib_mem_t *ib_memh)
{
#if HAVE_EXP_UMR
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    uct_ib_mlx5_md_t *md = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);
    off_t offset = uct_ib_md_atomic_offset(uct_ib_mlx5_md_get_atomic_mr_id(md));
    struct ibv_exp_mem_region *mem_reg = NULL;
    struct ibv_mr *mr = memh->super.mr;
    struct ibv_exp_send_wr wr, *bad_wr;
    struct ibv_exp_create_mr_in mrin;
    ucs_status_t status;
    struct ibv_mr *umr;
    struct ibv_wc wc;
    int i, list_size;
    size_t reg_length;
    int ret;

    if (md->umr_qp == NULL) {
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    /* Create and fill memory key */
    memset(&mrin, 0, sizeof(mrin));
    memset(&wr, 0, sizeof(wr));

    mrin.pd                             = md->super.pd;
    wr.exp_opcode                       = IBV_EXP_WR_UMR_FILL;
    wr.exp_send_flags                   = IBV_EXP_SEND_SIGNALED;
    wr.ext_op.umr.exp_access            = UCT_IB_MEM_ACCESS_FLAGS;

    reg_length = UCT_IB_MD_MAX_MR_SIZE;
#ifdef HAVE_EXP_UMR_KSM
    if ((md->super.dev.dev_attr.comp_mask & IBV_EXP_DEVICE_ATTR_COMP_MASK_2) &&
        (md->super.dev.dev_attr.comp_mask_2 & IBV_EXP_DEVICE_ATTR_UMR_FIXED_SIZE_CAPS) &&
        (md->super.dev.dev_attr.exp_device_cap_flags & IBV_EXP_DEVICE_UMR_FIXED_SIZE))
    {
        reg_length                      = md->super.dev.dev_attr.umr_fixed_size_caps.max_entity_size;
        list_size                       = ucs_div_round_up(mr->length, reg_length);
    } else if (mr->length < reg_length) {
        list_size                       = 1;
    } else {
        status                          = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    if (list_size > 1) {
        mrin.attr.create_flags          = IBV_EXP_MR_FIXED_BUFFER_SIZE;
        wr.ext_op.umr.umr_type          = IBV_EXP_UMR_MR_LIST_FIXED_SIZE;
    } else {
        mrin.attr.create_flags          = IBV_EXP_MR_INDIRECT_KLMS;
        wr.ext_op.umr.umr_type          = IBV_EXP_UMR_MR_LIST;
    }
#else
    if (mr->length >= reg_length) {
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    list_size                           = 1;
    mrin.attr.create_flags              = IBV_EXP_MR_INDIRECT_KLMS;
    wr.ext_op.umr.umr_type              = IBV_EXP_UMR_MR_LIST;
#endif

    mrin.attr.exp_access_flags          = UCT_IB_MEM_ACCESS_FLAGS;
    mrin.attr.max_klm_list_size         = list_size;
    mem_reg                             = ucs_calloc(list_size, sizeof(mem_reg[0]), "mem_reg");
    if (!mem_reg) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    umr = ibv_exp_create_mr(&mrin);
    if (!umr) {
        ucs_error("ibv_exp_create_mr() failed: %m");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    for (i = 0; i < list_size; i++) {
        mem_reg[i].base_addr            = (uintptr_t) mr->addr + i * reg_length;
        mem_reg[i].length               = reg_length;
        mem_reg[i].mr                   = mr;
    }

    ucs_assert(list_size >= 1);
    mem_reg[list_size - 1].length       = mr->length % reg_length;
    wr.ext_op.umr.mem_list.mem_reg_list = mem_reg;
    wr.ext_op.umr.base_addr             = (uint64_t) (uintptr_t) mr->addr + offset;
    wr.ext_op.umr.num_mrs               = list_size;
    wr.ext_op.umr.modified_mr           = umr;

    /* If the list exceeds max inline size, allocate a container object */
    if (list_size > md->super.config.max_inline_klm_list) {
        struct ibv_exp_mkey_list_container_attr in = {
            .pd                = md->super.pd,
            .mkey_list_type    = IBV_EXP_MKEY_LIST_TYPE_INDIRECT_MR,
            .max_klm_list_size = list_size
        };

        wr.ext_op.umr.memory_objects = ibv_exp_alloc_mkey_list_memory(&in);
        if (wr.ext_op.umr.memory_objects == NULL) {
            ucs_error("ibv_exp_alloc_mkey_list_memory(list_size=%d) failed: %m",
                      list_size);
            status = UCS_ERR_IO_ERROR;
            goto err_free_umr;
        }
    } else {
        wr.ext_op.umr.memory_objects = NULL;
        wr.exp_send_flags           |= IBV_EXP_SEND_INLINE;
    }

    ucs_trace_data("UMR_FILL qp 0x%x lkey 0x%x base 0x%lx [addr %lx len %zu lkey 0x%x] list_size %d",
                   md->umr_qp->qp_num, wr.ext_op.umr.modified_mr->lkey,
                   wr.ext_op.umr.base_addr, mem_reg[0].base_addr,
                   mem_reg[0].length, mem_reg[0].mr->lkey, list_size);

    /* Post UMR */
    ret = ibv_exp_post_send(md->umr_qp, &wr, &bad_wr);
    if (ret) {
        ucs_error("ibv_exp_post_send(UMR_FILL) failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free_klm_container;
    }

    /* Wait for send UMR completion */
    for (;;) {
        ret = ibv_poll_cq(md->umr_cq, 1, &wc);
        if (ret < 0) {
            ucs_error("ibv_exp_poll_cq(umr_cq) failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_free_klm_container;
        }
        if (ret == 1) {
            if (wc.status != IBV_WC_SUCCESS) {
                ucs_error("UMR_FILL completed with error: %s vendor_err %d",
                          ibv_wc_status_str(wc.status), wc.vendor_err);
                status = UCS_ERR_IO_ERROR;
                goto err_free_klm_container;
            }
            break;
        }
    }

    if (wr.ext_op.umr.memory_objects != NULL) {
        ibv_exp_dealloc_mkey_list_memory(wr.ext_op.umr.memory_objects);
    }

    ucs_debug("UMR registered memory %p..%p offset 0x%lx on %s lkey 0x%x rkey 0x%x",
              mr->addr, mr->addr + mr->length, offset, uct_ib_device_name(&md->super.dev),
              umr->lkey, umr->rkey);
    memh->atomic_mr         = umr;
    memh->super.atomic_rkey = umr->rkey;

    ucs_free(mem_reg);
    return UCS_OK;

err_free_klm_container:
    if (wr.ext_op.umr.memory_objects != NULL) {
        ibv_exp_dealloc_mkey_list_memory(wr.ext_op.umr.memory_objects);
    }
err_free_umr:
    UCS_PROFILE_CALL(ibv_dereg_mr, umr);
err:
    ucs_free(mem_reg);
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t uct_ib_mlx5_verbs_dereg_atomic_key(uct_ib_md_t *ibmd,
                                                       uct_ib_mem_t *ib_memh)
{
#if HAVE_EXP_UMR
    uct_ib_mlx5_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_mlx5_mem_t);
    int ret;

    ret = UCS_PROFILE_CALL(ibv_dereg_mr, memh->atomic_mr);
    if (ret != 0) {
        ucs_error("ibv_dereg_mr() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t uct_ib_mlx5_verbs_md_open(struct ibv_device *ibv_device,
                                              const uct_ib_md_config_t *md_config,
                                              uct_ib_md_t **p_md)
{
    ucs_status_t status = UCS_OK;
    struct ibv_context *ctx;
    uct_ib_device_t *dev;
    uct_ib_mlx5_md_t *md;
    int ret;

    ctx = ibv_open_device(ibv_device);
    if (ctx == NULL) {
        ucs_debug("ibv_open_device(%s) failed: %m", ibv_get_device_name(ibv_device));
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    md = ucs_calloc(1, sizeof(*md), "ib_mlx5_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context;
    }

    dev              = &md->super.dev;
    dev->ibv_context = ctx;

    IBV_EXP_DEVICE_ATTR_SET_COMP_MASK(&dev->dev_attr);
#if HAVE_DECL_IBV_EXP_QUERY_DEVICE
    ret = ibv_exp_query_device(dev->ibv_context, &dev->dev_attr);
#elif HAVE_DECL_IBV_QUERY_DEVICE_EX
    ret = ibv_query_device_ex(dev->ibv_context, NULL, &dev->dev_attr);
#else
#  error accel TL missconfigured
#endif
    if (ret != 0) {
        ucs_error("ibv_query_device() returned %d: %m", ret);
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    }

    if (!(uct_ib_device_spec(dev)->flags & UCT_IB_DEVICE_FLAG_MLX5_PRM)) {
        status = UCS_ERR_UNSUPPORTED;
        goto err_free;
    }

    if (IBV_EXP_HAVE_ATOMIC_HCA(&dev->dev_attr) ||
        IBV_EXP_HAVE_ATOMIC_GLOB(&dev->dev_attr) ||
        IBV_EXP_HAVE_ATOMIC_HCA_REPLY_BE(&dev->dev_attr))
    {
#ifdef HAVE_IB_EXT_ATOMICS
        if (dev->dev_attr.comp_mask & IBV_EXP_DEVICE_ATTR_EXT_ATOMIC_ARGS) {
            dev->ext_atomic_arg_sizes = dev->dev_attr.ext_atom.log_atomic_arg_sizes;
        }
#  if HAVE_MASKED_ATOMICS_ENDIANNESS
        if (dev->dev_attr.comp_mask & IBV_EXP_DEVICE_ATTR_MASKED_ATOMICS) {
            dev->ext_atomic_arg_sizes |=
                dev->dev_attr.masked_atomic.masked_log_atomic_arg_sizes;
            dev->ext_atomic_arg_sizes_be =
                dev->dev_attr.masked_atomic.masked_log_atomic_arg_sizes_network_endianness;
        }
#  endif
        dev->ext_atomic_arg_sizes &= UCS_MASK(dev->dev_attr.ext_atom.log_max_atomic_inline + 1);
#endif
        dev->atomic_arg_sizes = sizeof(uint64_t);
        if (IBV_EXP_HAVE_ATOMIC_HCA_REPLY_BE(&dev->dev_attr)) {
            dev->atomic_arg_sizes_be = sizeof(uint64_t);
        }
    }

#if HAVE_DECL_IBV_EXP_DEVICE_ATTR_PCI_ATOMIC_CAPS
    dev->pci_fadd_arg_sizes  = dev->dev_attr.pci_atomic_caps.fetch_add << 2;
    dev->pci_cswap_arg_sizes = dev->dev_attr.pci_atomic_caps.compare_swap << 2;
#endif

    status = uct_ib_mlx5_check_dc(dev);
    if (status != UCS_OK) {
        goto err_free;
    }

    md->super.ops = &uct_ib_mlx5_md_ops;
    status = uct_ib_md_open_common(&md->super, ibv_device, md_config);
    if (status != UCS_OK) {
        goto err_free;
    }

    status = uct_ib_mlx5_md_umr_qp_create(md);
    if (status != UCS_OK && status != UCS_ERR_UNSUPPORTED) {
        goto err_free;
    }

    dev->flags |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
    *p_md = &md->super;
    return UCS_OK;

err_free:
    ucs_free(md);
err_free_context:
    ibv_close_device(ctx);
err:
    return status;
}

#endif

void uct_ib_mlx5_md_cleanup(uct_ib_md_t *ibmd) {
    uct_ib_mlx5_md_t *md UCS_V_UNUSED = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);

#if HAVE_EXP_UMR
    if (md->umr_qp != NULL) {
        ibv_destroy_qp(md->umr_qp);
    }
    if (md->umr_cq != NULL) {
        ibv_destroy_cq(md->umr_cq);
    }
#endif

#if HAVE_DEVX
    ucs_mpool_cleanup(&md->dbrec_pool, 1);
#endif
}

#if HAVE_DEVX
static uct_ib_md_ops_t uct_ib_mlx5_md_ops = {
    .open             = uct_ib_mlx5dv_md_open,
    .cleanup          = uct_ib_mlx5_md_cleanup,
    .memh_struct_size = sizeof(uct_ib_mlx5_mem_t),
    .reg_atomic_key   = uct_ib_mlx5dv_reg_atomic_key,
    .dereg_atomic_key = uct_ib_mlx5dv_dereg_atomic_key,
};
#else
static uct_ib_md_ops_t uct_ib_mlx5_md_ops = {
    .open             = uct_ib_mlx5_verbs_md_open,
    .cleanup          = uct_ib_mlx5_md_cleanup,
    .memh_struct_size = sizeof(uct_ib_mlx5_mem_t),
    .reg_atomic_key   = uct_ib_mlx5_verbs_reg_atomic_key,
    .dereg_atomic_key = uct_ib_mlx5_verbs_dereg_atomic_key,
};
#endif

UCT_IB_MD_OPS(uct_ib_mlx5_md_ops, 1);

