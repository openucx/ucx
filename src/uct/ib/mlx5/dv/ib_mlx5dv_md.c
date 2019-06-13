/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/ib/mlx5/ib_mlx5.h>
#include "ib_mlx5_ifc.h"

#include <ucs/arch/bitops.h>
#include <ucs/profile/profile.h>

typedef struct uct_ib_mlx5_mem {
    uct_ib_mem_t             super;
#if HAVE_DEVX
    struct mlx5dv_devx_obj   *atomic_dvmr;
#endif
} uct_ib_mlx5_mem_t;

typedef struct uct_ib_mlx5_dbrec_page {
    struct mlx5dv_devx_umem *mem;
} uct_ib_mlx5_dbrec_page_t;


#if HAVE_DEVX

static ucs_status_t uct_ib_mlx5_devx_reg_atomic_key(uct_ib_md_t *ibmd,
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

static ucs_status_t uct_ib_mlx5_devx_dereg_atomic_key(uct_ib_md_t *ibmd,
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

static ucs_status_t uct_ib_mlx5_devx_check_odp(uct_ib_mlx5_md_t *md, void *cap)
{
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(query_hca_cap_out)] = {};
    uint32_t in[UCT_IB_MLX5DV_ST_SZ_DW(query_hca_cap_in)] = {};
    void *odp;
    int ret;

    if (!UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, pg)) {
        goto no_odp;
    }

    odp = UCT_IB_MLX5DV_ADDR_OF(query_hca_cap_out, out, capability);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR |
                                                   (UCT_IB_MLX5_CAP_ODP << 1));
    ret = mlx5dv_devx_general_cmd(md->super.dev.ibv_context, in, sizeof(in),
                                  out, sizeof(out));
    if (ret != 0) {
        ucs_error("mlx5dv_devx_general_cmd(QUERY_HCA_CAP, ODP) failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    if (!UCT_IB_MLX5DV_GET(odp_cap, odp, ud_odp_caps.send) ||
        !UCT_IB_MLX5DV_GET(odp_cap, odp, rc_odp_caps.send) ||
        !UCT_IB_MLX5DV_GET(odp_cap, odp, rc_odp_caps.write) ||
        !UCT_IB_MLX5DV_GET(odp_cap, odp, rc_odp_caps.read)) {
        goto no_odp;
    }

    if ((md->super.dev.flags & UCT_IB_DEVICE_FLAG_DC) &&
        (!UCT_IB_MLX5DV_GET(odp_cap, odp, dc_odp_caps.send) ||
         !UCT_IB_MLX5DV_GET(odp_cap, odp, dc_odp_caps.write) ||
         !UCT_IB_MLX5DV_GET(odp_cap, odp, dc_odp_caps.read))) {
        goto no_odp;
    }

    if (md->super.config.odp.max_size == UCS_MEMUNITS_AUTO) {
        if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, umr_extended_translation_offset)) {
            md->super.config.odp.max_size = 1ul << 55;
        } else {
            md->super.config.odp.max_size = 1ul << 28;
        }
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, fixed_buffer_size) &&
        UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, null_mkey) &&
        UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, umr_extended_translation_offset)) {
        md->super.dev.flags |= UCT_IB_DEVICE_FLAG_ODP_IMPLICIT;
    }

    return UCS_OK;

no_odp:
    md->super.config.odp.max_size = 0;
    return UCS_OK;
}

static uct_ib_md_ops_t uct_ib_mlx5_devx_md_ops;

static ucs_status_t uct_ib_mlx5_devx_md_open(struct ibv_device *ibv_device,
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
    md->super.config = md_config->ext;

    status = uct_ib_query_device(dev->ibv_context, &dev->dev_attr);
    if (status != UCS_OK) {
        goto err_free;
    }

    cap = UCT_IB_MLX5DV_ADDR_OF(query_hca_cap_out, out, capability);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, opcode, UCT_IB_MLX5_CMD_OP_QUERY_HCA_CAP);
    UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR |
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

    status = uct_ib_mlx5_devx_check_odp(md, cap);
    if (status != UCS_OK) {
        goto err_free;
    }

    if (UCT_IB_MLX5DV_GET(cmd_hca_cap, cap, atomic)) {
        int ops = UCT_IB_MLX5_ATOMIC_OPS_CMP_SWAP |
                  UCT_IB_MLX5_ATOMIC_OPS_FETCH_ADD;
        uint8_t arg_size;
        int cap_ops, mode8b;

        UCT_IB_MLX5DV_SET(query_hca_cap_in, in, op_mod, UCT_IB_MLX5_HCA_CAP_OPMOD_GET_CUR |
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

    md->super.ops = &uct_ib_mlx5_devx_md_ops;
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

void uct_ib_mlx5_devx_md_cleanup(uct_ib_md_t *ibmd)
{
    uct_ib_mlx5_md_t *md = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);

    ucs_mpool_cleanup(&md->dbrec_pool, 1);
}

static uct_ib_md_ops_t uct_ib_mlx5_devx_md_ops = {
    .open             = uct_ib_mlx5_devx_md_open,
    .cleanup          = uct_ib_mlx5_devx_md_cleanup,
    .memh_struct_size = sizeof(uct_ib_mlx5_mem_t),
    .reg_atomic_key   = uct_ib_mlx5_devx_reg_atomic_key,
    .dereg_atomic_key = uct_ib_mlx5_devx_dereg_atomic_key,
};

UCT_IB_MD_OPS(uct_ib_mlx5_devx_md_ops, 2);

#endif

static ucs_status_t uct_ib_mlx5dv_check_dc(uct_ib_device_t *dev)
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
#endif
    return status;
}

static uct_ib_md_ops_t uct_ib_mlx5_md_ops;

static ucs_status_t uct_ib_mlx5dv_md_open(struct ibv_device *ibv_device,
                                          const uct_ib_md_config_t *md_config,
                                          uct_ib_md_t **p_md)
{
    ucs_status_t status = UCS_OK;
    struct ibv_context *ctx;
    uct_ib_device_t *dev;
    uct_ib_mlx5_md_t *md;

#if HAVE_DECL_MLX5DV_IS_SUPPORTED
    if (!mlx5dv_is_supported(ibv_device)) {
        return UCS_ERR_UNSUPPORTED;
    }
#endif

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
    md->super.config = md_config->ext;

    status = uct_ib_query_device(dev->ibv_context, &dev->dev_attr);
    if (status != UCS_OK) {
        goto err_free;
    }

    if (!(uct_ib_device_spec(dev)->flags & UCT_IB_DEVICE_FLAG_MLX5_PRM)) {
        status = UCS_ERR_UNSUPPORTED;
        goto err_free;
    }

    if (md->super.config.odp.max_size == UCS_MEMUNITS_AUTO) {
        md->super.config.odp.max_size = 0;
    }

    if (IBV_EXP_HAVE_ATOMIC_HCA(&dev->dev_attr)) {
        dev->atomic_arg_sizes = sizeof(uint64_t);
    }

    status = uct_ib_mlx5dv_check_dc(dev);
    if (status != UCS_OK) {
        goto err_free;
    }

    md->super.ops = &uct_ib_mlx5_md_ops;
    status = uct_ib_md_open_common(&md->super, ibv_device, md_config);
    if (status != UCS_OK) {
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

static uct_ib_md_ops_t uct_ib_mlx5_md_ops = {
    .open             = uct_ib_mlx5dv_md_open,
    .cleanup          = (void*)ucs_empty_function,
    .memh_struct_size = sizeof(uct_ib_mlx5_mem_t),
    .reg_atomic_key   = (void*)ucs_empty_function_return_unsupported,
    .dereg_atomic_key = (void*)ucs_empty_function_return_unsupported,
};

UCT_IB_MD_OPS(uct_ib_mlx5_md_ops, 1);

