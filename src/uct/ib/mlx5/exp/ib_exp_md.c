/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/ib/mlx5/ib_mlx5.h>

#include <ucs/arch/bitops.h>
#include <ucs/profile/profile.h>

typedef struct uct_ib_mlx5_mem {
    uct_ib_mem_t             super;
#if HAVE_EXP_UMR
    struct ibv_mr            *atomic_mr;
#endif
} uct_ib_mlx5_mem_t;

static ucs_status_t uct_ib_mlx5_exp_md_umr_qp_create(uct_ib_mlx5_md_t *md)
{
#if HAVE_EXP_UMR
    struct ibv_exp_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    uint8_t port_num;
    int ret;
    uct_ib_device_t *ibdev;
    struct ibv_port_attr *port_attr;
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
    uct_ib_destroy_qp(md->umr_qp);
err_destroy_cq:
    ibv_destroy_cq(md->umr_cq);
err:
    return UCS_ERR_IO_ERROR;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t uct_ib_mlx5_exp_reg_atomic_key(uct_ib_md_t *ibmd,
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

static ucs_status_t uct_ib_mlx5_exp_dereg_atomic_key(uct_ib_md_t *ibmd,
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

static uct_ib_md_ops_t uct_ib_mlx5_md_ops;

static ucs_status_t uct_ib_mlx5_exp_md_open(struct ibv_device *ibv_device,
                                            const uct_ib_md_config_t *md_config,
                                            uct_ib_md_t **p_md)
{
    ucs_status_t status = UCS_OK;
    struct ibv_context *ctx;
    uct_ib_device_t *dev;
    uct_ib_mlx5_md_t *md;

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

    status = uct_ib_query_device(dev->ibv_context, &dev->dev_attr);
    if (status != UCS_OK) {
        goto err_free;
    }

    if (!(uct_ib_device_spec(dev)->flags & UCT_IB_DEVICE_FLAG_MLX5_PRM)) {
        status = UCS_ERR_UNSUPPORTED;
        goto err_free;
    }

#if HAVE_DECL_IBV_EXP_DEVICE_DC_TRANSPORT && HAVE_STRUCT_IBV_EXP_DEVICE_ATTR_EXP_DEVICE_CAP_FLAGS
    if (dev->dev_attr.exp_device_cap_flags & IBV_EXP_DEVICE_DC_TRANSPORT) {
        dev->flags |= UCT_IB_DEVICE_FLAG_DC;
    }
#endif

#if IBV_HW_TM_DC
    if (dev->dev_attr.tm_caps.capability_flags & IBV_EXP_TM_CAP_DC) {
        md->flags |= UCT_IB_MLX5_MD_FLAG_DC_TM;
    }
#endif

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

    md->super.ops = &uct_ib_mlx5_md_ops;
    status = uct_ib_md_open_common(&md->super, ibv_device, md_config);
    if (status != UCS_OK) {
        goto err_free;
    }

    status = uct_ib_mlx5_exp_md_umr_qp_create(md);
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

void uct_ib_mlx5_exp_md_cleanup(uct_ib_md_t *ibmd)
{
#if HAVE_EXP_UMR
    uct_ib_mlx5_md_t *md = ucs_derived_of(ibmd, uct_ib_mlx5_md_t);

    if (md->umr_qp != NULL) {
        uct_ib_destroy_qp(md->umr_qp);
    }
    if (md->umr_cq != NULL) {
        ibv_destroy_cq(md->umr_cq);
    }
#endif
}

static uct_ib_md_ops_t uct_ib_mlx5_md_ops = {
    .open             = uct_ib_mlx5_exp_md_open,
    .cleanup          = uct_ib_mlx5_exp_md_cleanup,
    .memh_struct_size = sizeof(uct_ib_mlx5_mem_t),
    .reg_atomic_key   = uct_ib_mlx5_exp_reg_atomic_key,
    .dereg_atomic_key = uct_ib_mlx5_exp_dereg_atomic_key,
};

UCT_IB_MD_OPS(uct_ib_mlx5_md_ops, 1);

