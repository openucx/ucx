/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "rc_mlx5.inl"

#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/mlx5/dv/ib_mlx5_ifc.h>

ucs_config_field_t uct_rc_mlx5_common_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_ib_mlx5_iface_config_table)},

  {"TX_MAX_BB", "-1",
   "Limits the number of outstanding WQE building blocks. The actual limit is\n"
   "a minimum between this value and the number of building blocks in the TX QP.\n"
   "-1 means no limit.",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, tx_max_bb), UCS_CONFIG_TYPE_UINT},

  {"TM_ENABLE", "n",
   "Enable HW tag matching",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, tm.enable), UCS_CONFIG_TYPE_BOOL},

  {"TM_LIST_SIZE", "1024",
   "Limits the number of tags posted to the HW for matching. The actual limit \n"
   "is a minimum between this value and the maximum value supported by the HW. \n"
   "-1 means no limit.",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, tm.list_size), UCS_CONFIG_TYPE_UINT},

  {"TM_SEG_SIZE", "48k",
   "Maximal size of copy-out sends when tag-matching offload is enabled.",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, tm.seg_size),
   UCS_CONFIG_TYPE_MEMUNITS},

  {"TM_MAX_BCOPY", NULL, "",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, tm.seg_size),
   UCS_CONFIG_TYPE_MEMUNITS},

  {"EXP_BACKOFF", "0",
   "Exponential Backoff Timeout Multiplier. ACK timeout will be multiplied \n"
   "by 2^EXP_BACKOFF every consecutive retry.",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, exp_backoff),
   UCS_CONFIG_TYPE_UINT},

  {NULL}
};


unsigned uct_rc_mlx5_iface_srq_post_recv(uct_rc_iface_t *iface, uct_ib_mlx5_srq_t *srq)
{
    uct_ib_mlx5_srq_seg_t *seg;
    uct_ib_iface_recv_desc_t *desc;
    uint16_t count, index, next_index;
    void *hdr;

    /* Make sure the union is right */
    UCS_STATIC_ASSERT(ucs_offsetof(uct_ib_mlx5_srq_seg_t, mlx5_srq.next_wqe_index) ==
                      ucs_offsetof(uct_ib_mlx5_srq_seg_t, srq.next_wqe_index));
    UCS_STATIC_ASSERT(ucs_offsetof(uct_ib_mlx5_srq_seg_t, dptr) ==
                      sizeof(struct mlx5_wqe_srq_next_seg));

    ucs_assert(UCS_CIRCULAR_COMPARE16(srq->ready_idx, <=, srq->free_idx));

    index = srq->ready_idx;
    for (;;) {
        next_index = index + 1;
        seg = uct_ib_mlx5_srq_get_wqe(srq, next_index & srq->mask);
        if (UCS_CIRCULAR_COMPARE16(next_index, >, srq->free_idx)) {
            if (!seg->srq.free) {
                break;
            }

            ucs_assert(next_index == (uint16_t)(srq->free_idx + 1));
            seg->srq.free  = 0;
            srq->free_idx  = next_index;
        }

        if (seg->srq.desc == NULL) {
            UCT_TL_IFACE_GET_RX_DESC(&iface->super.super, &iface->rx.mp,
                                     desc, break);

            /* Set receive data segment pointer. Length is pre-initialized. */
            hdr            = uct_ib_iface_recv_desc_hdr(&iface->super, desc);
            seg->srq.desc  = desc;
            seg->dptr.lkey = htonl(desc->lkey);
            seg->dptr.addr = htobe64((uintptr_t)hdr);
            VALGRIND_MAKE_MEM_NOACCESS(hdr, iface->super.config.seg_size);
        }

        index = next_index;
    }

    count = index - srq->sw_pi;
    ucs_assert(iface->rx.srq.available >= count);

    if (count > 0) {
        srq->ready_idx           = index;
        srq->sw_pi               = index;
        iface->rx.srq.available -= count;
        ucs_memory_cpu_store_fence();
        *srq->db = htonl(srq->sw_pi);
        ucs_assert(uct_ib_mlx5_srq_get_wqe(srq, srq->mask)->srq.next_wqe_index == 0);
    }
    return count;
}

void uct_rc_mlx5_iface_common_prepost_recvs(uct_rc_mlx5_iface_common_t *iface)
{
    iface->super.rx.srq.available = iface->super.rx.srq.quota;
    iface->super.rx.srq.quota     = 0;
    uct_rc_mlx5_iface_srq_post_recv(&iface->super, &iface->rx.srq);
}

#define UCT_RC_MLX5_DEFINE_ATOMIC_LE_HANDLER(_bits) \
    void \
    uct_rc_mlx5_common_atomic##_bits##_le_handler(uct_rc_iface_send_op_t *op, \
                                                  const void *resp) \
    { \
        uct_rc_iface_send_desc_t *desc = ucs_derived_of(op, uct_rc_iface_send_desc_t); \
        uint##_bits##_t *dest        = desc->super.buffer; \
        const uint##_bits##_t *value = resp; \
        \
        VALGRIND_MAKE_MEM_DEFINED(value, sizeof(*value)); \
        if (resp == (desc + 1)) { \
            *dest = *value; /* response in desc buffer */ \
        } else if (_bits == 32) { \
            *dest = ntohl(*value);  /* response in CQE as 32-bit value */ \
        } else if (_bits == 64) { \
            *dest = be64toh(*value); /* response in CQE as 64-bit value */ \
        } \
        \
        uct_invoke_completion(desc->super.user_comp, UCS_OK); \
        ucs_mpool_put(desc); \
    }

UCT_RC_MLX5_DEFINE_ATOMIC_LE_HANDLER(32)
UCT_RC_MLX5_DEFINE_ATOMIC_LE_HANDLER(64)

#if IBV_HW_TM
#  if ENABLE_STATS
static ucs_stats_class_t uct_rc_mlx5_tag_stats_class = {
    .name = "tag",
    .num_counters = UCT_RC_MLX5_STAT_TAG_LAST,
    .counter_names = {
        [UCT_RC_MLX5_STAT_TAG_RX_EXP]            = "rx_exp",
        [UCT_RC_MLX5_STAT_TAG_RX_EAGER_UNEXP]    = "rx_unexp_eager",
        [UCT_RC_MLX5_STAT_TAG_RX_RNDV_UNEXP]     = "rx_unexp_rndv",
        [UCT_RC_MLX5_STAT_TAG_RX_RNDV_REQ_EXP]   = "rx_exp_rndv_req",
        [UCT_RC_MLX5_STAT_TAG_RX_RNDV_REQ_UNEXP] = "rx_unexp_rndv_req",
        [UCT_RC_MLX5_STAT_TAG_RX_RNDV_FIN]       = "rx_rndv_fin",
        [UCT_RC_MLX5_STAT_TAG_LIST_ADD]          = "tx_add_op",
        [UCT_RC_MLX5_STAT_TAG_LIST_DEL]          = "tx_del_op",
        [UCT_RC_MLX5_STAT_TAG_LIST_SYNC]         = "tx_sync_op"
    }
};
#  endif


static ucs_status_t UCS_F_MAYBE_UNUSED
uct_rc_mlx5_devx_create_cmd_qp(uct_rc_mlx5_iface_common_t *iface)
{
    uct_ib_mlx5_md_t *md  = ucs_derived_of(iface->super.super.super.md,
                                           uct_ib_mlx5_md_t);
    uct_ib_device_t *dev  = &md->super.dev;
    struct ibv_ah_attr ah_attr = {};
    uct_ib_qp_attr_t attr = {};
    ucs_status_t status;

    ucs_assert(iface->tm.cmd_wq.super.super.type == UCT_IB_MLX5_OBJ_TYPE_LAST);

    attr.cap.max_send_wr  = iface->tm.cmd_qp_len;
    attr.cap.max_send_sge = 1;
    attr.ibv.pd           = md->super.pd;
    attr.ibv.send_cq      = iface->super.super.cq[UCT_IB_DIR_RX];
    attr.ibv.recv_cq      = iface->super.super.cq[UCT_IB_DIR_RX];
    attr.srq_num          = iface->rx.srq.srq_num;
    attr.port             = dev->first_port;
    status = uct_ib_mlx5_devx_create_qp(&iface->super.super,
                                        &iface->tm.cmd_wq.super.super,
                                        &iface->tm.cmd_wq.super,
                                        &attr);
    if (status != UCS_OK) {
        return status;
    }

    ah_attr.is_global     = 1;
    ah_attr.grh.dgid      = iface->super.super.gid;
    ah_attr.dlid          = uct_ib_device_port_attr(dev, attr.port)->lid;
    ah_attr.port_num      = dev->first_port;
    status = uct_rc_mlx5_iface_common_devx_connect_qp(
            iface, &iface->tm.cmd_wq.super.super,
            iface->tm.cmd_wq.super.super.qp_num, &ah_attr);
    if (status != UCS_OK) {
        goto err_destroy_qp;
    }

    return UCS_OK;

err_destroy_qp:
    uct_ib_mlx5_devx_destroy_qp(&iface->tm.cmd_wq.super.super);
    return status;
}

static struct ibv_qp * UCS_F_MAYBE_UNUSED
uct_rc_mlx5_verbs_create_cmd_qp(uct_rc_mlx5_iface_common_t *iface)
{
    uct_ib_md_t *md = uct_ib_iface_md(&iface->super.super);
    struct ibv_qp_init_attr qp_init_attr = {};
    struct ibv_qp_attr qp_attr = {};
    uct_ib_device_t *ibdev = &md->dev;
    struct ibv_port_attr *port_attr;
    ucs_status_t status;
    struct ibv_qp *qp;
    uint8_t port_num;
    int ret;

    port_num  = ibdev->first_port;
    port_attr = uct_ib_device_port_attr(ibdev, port_num);

    status = uct_ib_mlx5_iface_get_res_domain(&iface->super.super,
                                              &iface->tm.cmd_wq.super.super);
    if (status != UCS_OK) {
        goto err;
    }

    qp_init_attr.qp_type             = IBV_QPT_RC;
    qp_init_attr.send_cq             = iface->super.super.cq[UCT_IB_DIR_RX];
    qp_init_attr.recv_cq             = iface->super.super.cq[UCT_IB_DIR_RX];
    qp_init_attr.cap.max_send_sge    = 1;
    qp_init_attr.srq                 = iface->rx.srq.verbs.srq;
    qp_init_attr.cap.max_send_wr     = iface->tm.cmd_qp_len;

    qp = ibv_create_qp(md->pd, &qp_init_attr);
    if (qp == NULL) {
        ucs_error("failed to create TM control QP: %m");
        goto err_rd;
    }


    /* Modify QP to INIT state */
    qp_attr.qp_state                 = IBV_QPS_INIT;
    qp_attr.port_num                 = port_num;
    ret = ibv_modify_qp(qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        ucs_error("Failed to modify TM control QP to INIT: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTR */
    qp_attr.qp_state                 = IBV_QPS_RTR;
    qp_attr.dest_qp_num              = qp->qp_num;
    qp_attr.path_mtu                 = IBV_MTU_512;
    qp_attr.ah_attr.port_num         = port_num;
    qp_attr.ah_attr.dlid             = port_attr->lid;
    qp_attr.ah_attr.is_global        = 1;
    qp_attr.ah_attr.grh.dgid         = iface->super.super.gid;
    ret = ibv_modify_qp(qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        ucs_error("Failed to modify TM control QP to RTR: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTS */
    qp_attr.qp_state                 = IBV_QPS_RTS;
    ret = ibv_modify_qp(qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        ucs_error("Failed to modify TM control QP to RTS: %m");
        goto err_destroy_qp;
    }

    iface->tm.cmd_wq.super.super.verbs.qp = qp;
    return qp;

err_destroy_qp:
    uct_ib_destroy_qp(qp);
err_rd:
    uct_ib_mlx5_iface_put_res_domain(&iface->tm.cmd_wq.super.super);
err:
    return NULL;
}

static ucs_status_t
uct_rc_mlx5_get_cmd_qp(uct_rc_mlx5_iface_common_t *iface)
{
    struct ibv_qp *qp;
#if HAVE_STRUCT_MLX5_SRQ_CMD_QP
    iface->tm.cmd_wq.super.super.verbs.qp = NULL;
    iface->tm.cmd_wq.super.super.verbs.rd = NULL;
    iface->tm.cmd_wq.super.super.type     = UCT_IB_MLX5_OBJ_TYPE_LAST;
    qp = uct_dv_get_cmd_qp(iface->rx.srq.verbs.srq);
#else
    uct_ib_mlx5_md_t *md       = ucs_derived_of(iface->super.super.super.md,
                                                uct_ib_mlx5_md_t);

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX) {
        return uct_rc_mlx5_devx_create_cmd_qp(iface);
    } else {
        qp = uct_rc_mlx5_verbs_create_cmd_qp(iface);
    }
#endif
    iface->tm.cmd_wq.super.super.qp_num = qp->qp_num;
    return uct_ib_mlx5_txwq_init(iface->super.super.super.worker,
                                 iface->tx.mmio_mode,
                                 &iface->tm.cmd_wq.super, qp);
}
#endif

ucs_status_t uct_rc_mlx5_iface_common_tag_init(uct_rc_mlx5_iface_common_t *iface)
{
    ucs_status_t status = UCS_OK;
#if IBV_HW_TM
    int i;

    if (!UCT_RC_MLX5_TM_ENABLED(iface)) {
        return UCS_OK;
    }

    status = uct_rc_mlx5_get_cmd_qp(iface);
    if (status != UCS_OK) {
        goto err_tag_cleanup;
    }

    iface->tm.cmd_wq.ops_mask = iface->tm.cmd_qp_len - 1;
    iface->tm.cmd_wq.ops_head = iface->tm.cmd_wq.ops_tail = 0;
    iface->tm.cmd_wq.ops      = ucs_calloc(iface->tm.cmd_qp_len,
                                           sizeof(uct_rc_mlx5_srq_op_t),
                                           "srq tag ops");
    if (iface->tm.cmd_wq.ops == NULL) {
        ucs_error("Failed to allocate memory for srq tm ops array");
        status = UCS_ERR_NO_MEMORY;
        goto err_tag_cleanup;
    }

    iface->tm.list = ucs_calloc(iface->tm.num_tags + 1,
                                sizeof(uct_rc_mlx5_tag_entry_t), "tm list");
    if (iface->tm.list == NULL) {
        ucs_error("Failed to allocate memory for tag matching list");
        status = UCS_ERR_NO_MEMORY;
        goto err_cmd_wq_free;
    }

    for (i = 0; i < iface->tm.num_tags; ++i) {
        iface->tm.list[i].next = &iface->tm.list[i + 1];
    }

    iface->tm.head = &iface->tm.list[0];
    iface->tm.tail = &iface->tm.list[i];

    status = UCS_STATS_NODE_ALLOC(&iface->tm.stats, &uct_rc_mlx5_tag_stats_class,
                                  iface->stats);
    if (status != UCS_OK) {
        ucs_error("Failed to allocate tag stats: %s", ucs_status_string(status));
        goto err_cmd_wq_free;
    }

    return UCS_OK;

err_cmd_wq_free:
    ucs_free(iface->tm.cmd_wq.ops);
err_tag_cleanup:
    uct_rc_mlx5_tag_cleanup(iface);
#endif

    return status;
}

void uct_rc_mlx5_iface_common_tag_cleanup(uct_rc_mlx5_iface_common_t *iface)
{
    if (UCT_RC_MLX5_TM_ENABLED(iface)) {
        uct_ib_mlx5_destroy_qp(&iface->tm.cmd_wq.super.super);
        uct_ib_mlx5_txwq_cleanup(&iface->tm.cmd_wq.super);
        ucs_free(iface->tm.list);
        ucs_free(iface->tm.cmd_wq.ops);
        uct_rc_mlx5_tag_cleanup(iface);
    }
}

void uct_rc_mlx5_destroy_srq(uct_ib_mlx5_srq_t *srq)
{
    int ret;

    switch (srq->type) {
    case UCT_IB_MLX5_OBJ_TYPE_VERBS:
        ret = ibv_destroy_srq(srq->verbs.srq);
        if (ret) {
            ucs_warn("ibv_destroy_srq() failed: %m");
        }
        break;
    case UCT_IB_MLX5_OBJ_TYPE_DEVX:
#if HAVE_DEVX
        ret = mlx5dv_devx_obj_destroy(srq->devx.obj);
        if (ret) {
            ucs_error("mlx5dv_devx_obj_destroy(SRQ) failed: %m");
        }
        ucs_mpool_put_inline(srq->devx.dbrec);
        mlx5dv_devx_umem_dereg(srq->devx.mem);
        ucs_free(srq->buf);
#endif
        break;
    case UCT_IB_MLX5_OBJ_TYPE_LAST:
        break;
    }
}

#if IBV_HW_TM
static void uct_rc_mlx5_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_rc_mlx5_release_desc_t *release = ucs_derived_of(self,
                                                         uct_rc_mlx5_release_desc_t);
    void *ib_desc = (char*)desc - release->offset;
    ucs_mpool_put_inline(ib_desc);
}

/* tag is passed as parameter, because some (but not all!) transports may need
 * to translate TMH to LE */
ucs_status_t uct_rc_mlx5_handle_rndv(uct_rc_mlx5_iface_common_t *iface,
                                     struct ibv_tmh *tmh, uct_tag_t tag,
                                     unsigned byte_len)
{
    uct_rc_mlx5_tmh_priv_data_t *priv = (uct_rc_mlx5_tmh_priv_data_t*)tmh->reserved;
    uct_ib_md_t *ib_md                = uct_ib_iface_md(&iface->super.super);
    struct ibv_rvh *rvh;
    unsigned tm_hdrs_len;
    unsigned rndv_usr_hdr_len;
    size_t rndv_data_len;
    void *rndv_usr_hdr;
    void *rb;
    char packed_rkey[UCT_MD_COMPONENT_NAME_MAX + UCT_IB_MD_PACKED_RKEY_SIZE];

    rvh = (struct ibv_rvh*)(tmh + 1);

    /* RC uses two headers: TMH + RVH, DC uses three: TMH + RVH + RAVH.
     * So, get actual RNDV hdrs len from offsets. */
    tm_hdrs_len = sizeof(*tmh) +
                  (iface->tm.rndv_desc.offset - iface->tm.eager_desc.offset);

    rndv_usr_hdr     = (char*)tmh + tm_hdrs_len;
    rndv_usr_hdr_len = byte_len - tm_hdrs_len;
    rndv_data_len    = ntohl(rvh->len);

    /* Private TMH data may contain the first bytes of the user header, so it
       needs to be copied before that. Thus, either RVH (rc) or RAVH (dc)
       will be overwritten. That's why we saved rvh->length before. */
    ucs_assert(priv->length <= UCT_RC_MLX5_TMH_PRIV_LEN);

    memcpy((char*)rndv_usr_hdr - priv->length, &priv->data, priv->length);

    /* Create "packed" rkey to pass it in the callback */
    rb = uct_md_fill_md_name(&ib_md->super, packed_rkey);
    uct_ib_md_pack_rkey(ntohl(rvh->rkey), UCT_IB_INVALID_RKEY, rb);

    /* Do not pass flags to cb, because rkey is allocated on stack */
    return iface->tm.rndv_unexp.cb(iface->tm.rndv_unexp.arg, 0, tag,
                                   (char *)rndv_usr_hdr - priv->length,
                                   rndv_usr_hdr_len + priv->length,
                                   be64toh(rvh->va), rndv_data_len,
                                   packed_rkey);
}
#endif

#if HAVE_IBV_DM
static ucs_status_t
uct_rc_mlx5_iface_common_dm_mpool_chunk_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucs_status_t status;

    status = ucs_mpool_chunk_malloc(mp, size_p, chunk_p);
    if (status == UCS_OK) {
        memset(*chunk_p, 0, *size_p);
    }

    return status;
}

static void uct_rc_mlx5_iface_common_dm_mp_obj_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_mlx5_dm_data_t *dm         = ucs_container_of(mp, uct_mlx5_dm_data_t, mp);
    uct_rc_iface_send_desc_t* desc = (uct_rc_iface_send_desc_t*)obj;

    ucs_assert(desc->super.buffer == NULL);
    ucs_assert(dm->seg_attached < dm->seg_count);

    desc->lkey          = dm->mr->lkey;
    desc->super.buffer  = UCS_PTR_BYTE_OFFSET(dm->start_va, dm->seg_attached * dm->seg_len);
    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
    dm->seg_attached++;
}

static ucs_mpool_ops_t uct_dm_iface_mpool_ops = {
    .chunk_alloc   = uct_rc_mlx5_iface_common_dm_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_rc_mlx5_iface_common_dm_mp_obj_init,
    .obj_cleanup   = NULL
};


static int uct_rc_mlx5_iface_common_dm_device_cmp(uct_mlx5_dm_data_t *dm_data,
                                                  uct_rc_iface_t *iface,
                                                  const uct_ib_mlx5_iface_config_t *config)
{
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);

    return dm_data->device->ibv_context == dev->ibv_context;
}

static ucs_status_t
uct_rc_mlx5_iface_common_dm_tl_init(uct_mlx5_dm_data_t *data,
                                    uct_rc_iface_t *iface,
                                    const uct_ib_mlx5_iface_config_t *config)
{
    struct ibv_alloc_dm_attr dm_attr = {};
    struct mlx5dv_dm dvdm = {};
    uct_ib_mlx5dv_t obj = {};
    ucs_status_t status;

    data->seg_len      = ucs_min(ucs_align_up(config->dm.seg_len,
                                              sizeof(uct_rc_mlx5_dm_copy_data_t)),
                                 iface->super.config.seg_size);
    data->seg_count    = config->dm.count;
    data->seg_attached = 0;
    data->device       = uct_ib_iface_device(&iface->super);

    dm_attr.length     = data->seg_len * data->seg_count;
    dm_attr.comp_mask  = 0;
    data->dm           = ibv_alloc_dm(data->device->ibv_context, &dm_attr);

    if (data->dm == NULL) {
        /* TODO: prompt warning? */
        ucs_debug("ibv_alloc_dm(dev=%s length=%zu) failed: %m",
                  uct_ib_device_name(data->device), dm_attr.length);
        return UCS_ERR_NO_RESOURCE;
    }

    data->mr           = ibv_reg_dm_mr(uct_ib_iface_md(&iface->super)->pd,
                                       data->dm, 0, dm_attr.length,
                                       IBV_ACCESS_ZERO_BASED   |
                                       IBV_ACCESS_LOCAL_WRITE  |
                                       IBV_ACCESS_REMOTE_READ  |
                                       IBV_ACCESS_REMOTE_WRITE |
                                       IBV_ACCESS_REMOTE_ATOMIC);
    if (data->mr == NULL) {
        ucs_warn("ibv_reg_mr_dm() error - On Device Memory registration failed, %d %m", errno);
        status = UCS_ERR_NO_RESOURCE;
        goto failed_mr;
    }

    UCT_IB_MLX5_DV_DM(obj).in  = data->dm;
    UCT_IB_MLX5_DV_DM(obj).out = &dvdm;
    uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_DM);
    data->start_va = dvdm.buf;

    status = ucs_mpool_init(&data->mp, 0,
                            sizeof(uct_rc_iface_send_desc_t), 0, UCS_SYS_CACHE_LINE_SIZE,
                            data->seg_count, data->seg_count,
                            &uct_dm_iface_mpool_ops, "mlx5_dm_desc");
    if (status != UCS_OK) {
        goto failed_mpool;
    }

    /* DM initialization may fail due to any reason, just
     * free resources & continue without DM */
    return UCS_OK;

failed_mpool:
    ibv_dereg_mr(data->mr);
failed_mr:
    ibv_free_dm(data->dm);
    data->dm = NULL;
    return status;
}

static void uct_rc_mlx5_iface_common_dm_tl_cleanup(uct_mlx5_dm_data_t *data)
{
    ucs_assert(data->dm != NULL);
    ucs_assert(data->mr != NULL);

    ucs_mpool_cleanup(&data->mp, 1);
    ibv_dereg_mr(data->mr);
    ibv_free_dm(data->dm);
}
#endif

#if IBV_HW_TM
static void uct_rc_mlx5_init_rx_tm_common(uct_rc_mlx5_iface_common_t *iface,
                                          unsigned rndv_hdr_len)
{
    uct_ib_md_t *md       = uct_ib_iface_md(&iface->super.super);
    unsigned tmh_hdrs_len = sizeof(struct ibv_tmh) + rndv_hdr_len;

    iface->tm.eager_desc.super.cb = uct_rc_mlx5_release_desc;
    iface->tm.eager_desc.offset   = sizeof(struct ibv_tmh)
                                    - sizeof(uct_rc_mlx5_hdr_t)
                                    + iface->super.super.config.rx_headroom_offset;

    iface->tm.rndv_desc.super.cb  = uct_rc_mlx5_release_desc;
    iface->tm.rndv_desc.offset    = iface->tm.eager_desc.offset + rndv_hdr_len;

    ucs_assert(IBV_DEVICE_TM_CAPS(&md->dev, max_rndv_hdr_size) >= tmh_hdrs_len);
    iface->tm.max_rndv_data       = IBV_DEVICE_TM_CAPS(&md->dev, max_rndv_hdr_size) -
                                    tmh_hdrs_len;

    /* Init ptr array to store completions of RNDV operations. Index in
     * ptr_array is used as operation ID and is passed in "app_context"
     * of TM header. */
    ucs_ptr_array_init(&iface->tm.rndv_comps, 0, "rm_rndv_completions");
}

ucs_status_t
uct_rc_mlx5_devx_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                            const uct_rc_iface_common_config_t *config,
                            struct ibv_exp_create_srq_attr *attr)
{
#if HAVE_DEVX
    uct_ib_mlx5_md_t *md = ucs_derived_of(uct_ib_iface_md(&iface->super.super), uct_ib_mlx5_md_t);
    uct_ib_device_t *dev = &md->super.dev;
    uint32_t in[UCT_IB_MLX5DV_ST_SZ_DW(create_xrq_in)] = {};
    uint32_t out[UCT_IB_MLX5DV_ST_SZ_DW(create_xrq_out)] = {};
    ucs_status_t status = UCS_ERR_NO_MEMORY;
    uct_ib_mlx5_srq_seg_t *seg;
    struct mlx5dv_pd dvpd = {};
    struct mlx5dv_cq dvcq = {};
    struct mlx5dv_obj dv = {};
    void *xrqc, *wq;
    int len, ret, max, i;

    uct_rc_mlx5_init_rx_tm_common(iface, sizeof(struct ibv_rvh));

    max = ucs_max(config->super.rx.queue_len, IBV_DEVICE_MIN_UWQ_POST);
    max = ucs_roundup_pow2(max);
    len = max * UCT_IB_MLX5_SRQ_STRIDE;
    ret = posix_memalign(&iface->rx.srq.buf, ucs_get_page_size(), len);
    if (ret) {
        return status;
    }

    iface->rx.srq.devx.mem = mlx5dv_devx_umem_reg(dev->ibv_context,
                                                  iface->rx.srq.buf, len,
                                                  IBV_ACCESS_LOCAL_WRITE);
    if (!iface->rx.srq.devx.mem) {
        goto err_free_buf;
    }

    iface->rx.srq.devx.dbrec = ucs_mpool_get_inline(&md->dbrec_pool);
    if (!iface->rx.srq.devx.dbrec) {
        goto err_free_mem;
    }

    iface->rx.srq.db = &iface->rx.srq.devx.dbrec->db[MLX5_RCV_DBR];
    dv.pd.in         = uct_ib_iface_md(&iface->super.super)->pd;
    dv.cq.in         = iface->super.super.cq[UCT_IB_DIR_RX];
    dv.pd.out        = &dvpd;
    dv.cq.out        = &dvcq;
    mlx5dv_init_obj(&dv, MLX5DV_OBJ_PD | MLX5DV_OBJ_CQ);

    UCT_IB_MLX5DV_SET(create_xrq_in, in, opcode, UCT_IB_MLX5_CMD_OP_CREATE_XRQ);
    xrqc = UCT_IB_MLX5DV_ADDR_OF(create_xrq_in, in, xrq_context);

    UCT_IB_MLX5DV_SET(xrqc, xrqc, topology, UCT_IB_MLX5_XRQC_TOPOLOGY_TAG_MATCHING);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, offload,  UCT_IB_MLX5_XRQC_OFFLOAD_RNDV);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, tag_matching_topology_context.log_matching_list_sz,
                                  ucs_ilog2(iface->tm.num_tags));
    UCT_IB_MLX5DV_SET(xrqc, xrqc, dc,       1);
    UCT_IB_MLX5DV_SET(xrqc, xrqc, cqn,      dvcq.cqn);

    wq = UCT_IB_MLX5DV_ADDR_OF(xrqc, xrqc, wq);
    UCT_IB_MLX5DV_SET(wq, wq, log_wq_sz,     ucs_ilog2(max));
    UCT_IB_MLX5DV_SET(wq, wq, log_wq_stride, ucs_ilog2(UCT_IB_MLX5_SRQ_STRIDE));
    UCT_IB_MLX5DV_SET(wq, wq, pd,            dvpd.pdn);
    UCT_IB_MLX5DV_SET(wq, wq, dbr_umem_id,   iface->rx.srq.devx.dbrec->mem_id);
    UCT_IB_MLX5DV_SET64(wq, wq, dbr_addr,    iface->rx.srq.devx.dbrec->offset);
    UCT_IB_MLX5DV_SET(wq, wq, wq_umem_id,    iface->rx.srq.devx.mem->umem_id);

    iface->rx.srq.devx.obj = mlx5dv_devx_obj_create(dev->ibv_context, in, sizeof(in),
                                                    out, sizeof(out));
    if (iface->rx.srq.devx.obj == NULL) {
        ucs_error("mlx5dv_devx_obj_create(SRQ) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(create_xrq_out, out, syndrome));
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    }

    iface->rx.srq.type    = UCT_IB_MLX5_OBJ_TYPE_DEVX;
    iface->rx.srq.srq_num = UCT_IB_MLX5DV_GET(create_xrq_out, out, xrqn);
    iface->tm.cmd_qp_len  = (2 * iface->tm.num_tags);

    iface->rx.srq.free_idx      = max - 1;
    iface->rx.srq.ready_idx     = -1;
    iface->rx.srq.sw_pi         = -1;
    iface->rx.srq.mask          = max - 1;
    iface->rx.srq.tail          = max - 1;

    for (i = 0; i < max; ++i) {
        seg = uct_ib_mlx5_srq_get_wqe(&iface->rx.srq, i);
        seg->srq.next_wqe_index = htons((i + 1) & (max - 1));
        seg->srq.free           = 0;
        seg->srq.desc           = NULL;
        seg->dptr.byte_count    = htonl(iface->super.super.config.seg_size);
    }

    iface->super.rx.srq.quota   = max - 1;
    return UCS_OK;

err_free:
    ucs_mpool_put_inline(iface->rx.srq.devx.dbrec);
err_free_mem:
    mlx5dv_devx_umem_dereg(iface->rx.srq.devx.mem);
err_free_buf:
    ucs_free(iface->rx.srq.buf);
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_mlx5_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                                    const uct_rc_iface_common_config_t *config,
                                    struct ibv_exp_create_srq_attr *srq_init_attr,
                                    unsigned rndv_hdr_len)
{
    uct_ib_md_t *md         = uct_ib_iface_md(&iface->super.super);

    uct_rc_mlx5_init_rx_tm_common(iface, rndv_hdr_len);

#if HAVE_DECL_IBV_EXP_CREATE_SRQ
    /* Create TM-capable XRQ */
    srq_init_attr->base.attr.max_sge   = 1;
    srq_init_attr->base.attr.max_wr    = ucs_max(IBV_DEVICE_MIN_UWQ_POST,
                                                 config->super.rx.queue_len);
    srq_init_attr->base.attr.srq_limit = 0;
    srq_init_attr->base.srq_context    = iface;
    srq_init_attr->srq_type            = IBV_EXP_SRQT_TAG_MATCHING;
    srq_init_attr->pd                  = md->pd;
    srq_init_attr->cq                  = iface->super.super.cq[UCT_IB_DIR_RX];
    srq_init_attr->tm_cap.max_num_tags = iface->tm.num_tags;

    iface->tm.cmd_qp_len = (2 * iface->tm.num_tags) + 2;
    srq_init_attr->tm_cap.max_ops = iface->tm.cmd_qp_len;
    srq_init_attr->comp_mask     |= IBV_EXP_CREATE_SRQ_CQ |
                                    IBV_EXP_CREATE_SRQ_TM;

    iface->rx.srq.verbs.srq = ibv_exp_create_srq(md->dev.ibv_context, srq_init_attr);
    if (iface->rx.srq.verbs.srq == NULL) {
        ucs_error("ibv_exp_create_srq(device=%s) failed: %m",
                  uct_ib_device_name(&md->dev));
        return UCS_ERR_IO_ERROR;
    }

    iface->super.rx.srq.quota = srq_init_attr->base.attr.max_wr;
#elif HAVE_DECL_IBV_CREATE_SRQ_EX
    srq_init_attr->attr.max_sge        = 1;
    srq_init_attr->attr.max_wr         = ucs_max(IBV_DEVICE_MIN_UWQ_POST,
                                                 config->super.rx.queue_len);
    srq_init_attr->attr.srq_limit      = 0;
    srq_init_attr->srq_context         = iface;
    srq_init_attr->srq_type            = IBV_SRQT_TM;
    srq_init_attr->pd                  = md->pd;
    srq_init_attr->cq                  = iface->super.super.cq[UCT_IB_DIR_RX];
    srq_init_attr->tm_cap.max_num_tags = iface->tm.num_tags;

    iface->tm.cmd_qp_len = (2 * iface->tm.num_tags) + 2;
    srq_init_attr->tm_cap.max_ops = iface->tm.cmd_qp_len;
    srq_init_attr->comp_mask     |= IBV_SRQ_INIT_ATTR_TYPE |
                                    IBV_SRQ_INIT_ATTR_PD |
                                    IBV_SRQ_INIT_ATTR_CQ |
                                    IBV_SRQ_INIT_ATTR_TM;

    iface->rx.srq.verbs.srq = ibv_create_srq_ex(md->dev.ibv_context, srq_init_attr);
    if (iface->rx.srq.verbs.srq == NULL) {
        ucs_error("ibv_create_srq_ex(device=%s) failed: %m",
                  uct_ib_device_name(&md->dev));
        return UCS_ERR_IO_ERROR;
    }

    iface->super.rx.srq.quota = srq_init_attr->attr.max_wr;
#endif

    iface->rx.srq.type        = UCT_IB_MLX5_OBJ_TYPE_VERBS;
    ucs_debug("Tag Matching enabled: tag list size %d", iface->tm.num_tags);
    return UCS_OK;
}
#endif

void uct_rc_mlx5_tag_cleanup(uct_rc_mlx5_iface_common_t *iface)
{
#if IBV_HW_TM
    if (UCT_RC_MLX5_TM_ENABLED(iface)) {
        ucs_ptr_array_cleanup(&iface->tm.rndv_comps);
        UCS_STATS_NODE_FREE(iface->tm.stats);
    }
#endif
}

static void uct_rc_mlx5_tag_query(uct_rc_mlx5_iface_common_t *iface,
                                  uct_iface_attr_t *iface_attr,
                                  size_t max_inline, size_t max_iov)
{
#if IBV_HW_TM
    unsigned eager_hdr_size = sizeof(struct ibv_tmh);
    struct ibv_port_attr* port_attr;

    if (!UCT_RC_MLX5_TM_ENABLED(iface)) {
        return;
    }

    iface_attr->cap.flags |= UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                             UCT_IFACE_FLAG_TAG_EAGER_ZCOPY |
                             UCT_IFACE_FLAG_TAG_RNDV_ZCOPY;

    if (max_inline >= eager_hdr_size) {
        iface_attr->cap.tag.eager.max_short = max_inline - eager_hdr_size;
        iface_attr->cap.flags              |= UCT_IFACE_FLAG_TAG_EAGER_SHORT;
    }

    iface_attr->cap.tag.eager.max_bcopy = iface->super.super.config.seg_size -
                                          eager_hdr_size;
    iface_attr->cap.tag.eager.max_zcopy = iface->super.super.config.seg_size -
                                          eager_hdr_size;
    iface_attr->cap.tag.eager.max_iov   = max_iov;

    port_attr = uct_ib_iface_port_attr(&iface->super.super);
    iface_attr->cap.tag.rndv.max_zcopy  = port_attr->max_msg_sz;

    /* TMH can carry 2 additional bytes of private data */
    iface_attr->cap.tag.rndv.max_hdr    = iface->tm.max_rndv_data +
                                          UCT_RC_MLX5_TMH_PRIV_LEN;
    iface_attr->cap.tag.rndv.max_iov    = 1;

    iface_attr->cap.tag.recv.max_zcopy       = port_attr->max_msg_sz;
    iface_attr->cap.tag.recv.max_iov         = 1;
    iface_attr->cap.tag.recv.min_recv        = 0;
    iface_attr->cap.tag.recv.max_outstanding = iface->tm.num_tags;
#endif
}

ucs_status_t
uct_rc_mlx5_iface_common_dm_init(uct_rc_mlx5_iface_common_t *iface,
                                 uct_rc_iface_t *rc_iface,
                                 const uct_ib_mlx5_iface_config_t *mlx5_config)
{
#if HAVE_IBV_DM
    if ((mlx5_config->dm.seg_len * mlx5_config->dm.count) == 0) {
        goto fallback;
    }

    iface->dm.dm = uct_worker_tl_data_get(iface->super.super.super.worker,
                                          UCT_IB_MLX5_WORKER_DM_KEY,
                                          uct_mlx5_dm_data_t,
                                          uct_rc_mlx5_iface_common_dm_device_cmp,
                                          uct_rc_mlx5_iface_common_dm_tl_init,
                                          &iface->super, mlx5_config);
    if (UCS_PTR_IS_ERR(iface->dm.dm)) {
        goto fallback;
    }

    ucs_assert(iface->dm.dm->dm != NULL);
    iface->dm.seg_len = iface->dm.dm->seg_len;
    return UCS_OK;

fallback:
    iface->dm.dm = NULL;
#endif
    return UCS_OK;
}

void uct_rc_mlx5_iface_common_dm_cleanup(uct_rc_mlx5_iface_common_t *iface)
{
#if HAVE_IBV_DM
    if (iface->dm.dm) {
        uct_worker_tl_data_put(iface->dm.dm, uct_rc_mlx5_iface_common_dm_tl_cleanup);
    }
#endif
}

void uct_rc_mlx5_iface_common_query(uct_ib_iface_t *ib_iface,
                                    uct_iface_attr_t *iface_attr,
                                    size_t max_inline, size_t av_size)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(ib_iface,
                                                       uct_rc_mlx5_iface_common_t);
    uct_ib_device_t *dev = uct_ib_iface_device(ib_iface);

    /* Atomics */
    iface_attr->cap.flags        |= UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF |
                                    UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM;

    if (uct_ib_device_has_pci_atomics(dev)) {
        if (dev->pci_fadd_arg_sizes & sizeof(uint64_t)) {
            iface_attr->cap.atomic64.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_ADD);
            iface_attr->cap.atomic64.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_ADD);
        }
        if (dev->pci_cswap_arg_sizes & sizeof(uint64_t)) {
            iface_attr->cap.atomic64.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_CSWAP);
        }

        if (dev->pci_fadd_arg_sizes & sizeof(uint32_t)) {
            iface_attr->cap.atomic32.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_ADD);
            iface_attr->cap.atomic32.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_ADD);
        }
        if (dev->pci_cswap_arg_sizes & sizeof(uint32_t)) {
            iface_attr->cap.atomic32.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_CSWAP);
        }
        iface_attr->cap.flags                  |= UCT_IFACE_FLAG_ATOMIC_CPU;
    } else {
        if (dev->atomic_arg_sizes & sizeof(uint64_t)) {
            iface_attr->cap.atomic64.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_ADD);
            iface_attr->cap.atomic64.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_ADD)  |
                UCS_BIT(UCT_ATOMIC_OP_CSWAP);

            iface_attr->cap.flags              |= UCT_IFACE_FLAG_ATOMIC_DEVICE;
        }

        if (dev->ext_atomic_arg_sizes & sizeof(uint64_t)) {
            iface_attr->cap.atomic64.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_AND)  |
                UCS_BIT(UCT_ATOMIC_OP_OR)   |
                UCS_BIT(UCT_ATOMIC_OP_XOR);
            iface_attr->cap.atomic64.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_AND)  |
                UCS_BIT(UCT_ATOMIC_OP_OR)   |
                UCS_BIT(UCT_ATOMIC_OP_XOR)  |
                UCS_BIT(UCT_ATOMIC_OP_SWAP);
            iface_attr->cap.flags              |= UCT_IFACE_FLAG_ATOMIC_DEVICE;
        }

        if (dev->ext_atomic_arg_sizes & sizeof(uint32_t)) {
            iface_attr->cap.atomic32.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_ADD)  |
                UCS_BIT(UCT_ATOMIC_OP_AND)  |
                UCS_BIT(UCT_ATOMIC_OP_OR)   |
                UCS_BIT(UCT_ATOMIC_OP_XOR);
            iface_attr->cap.atomic32.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_ADD)  |
                UCS_BIT(UCT_ATOMIC_OP_AND)  |
                UCS_BIT(UCT_ATOMIC_OP_OR)   |
                UCS_BIT(UCT_ATOMIC_OP_XOR)  |
                UCS_BIT(UCT_ATOMIC_OP_SWAP) |
                UCS_BIT(UCT_ATOMIC_OP_CSWAP);
            iface_attr->cap.flags              |= UCT_IFACE_FLAG_ATOMIC_DEVICE;
        }
    }

    /* Software overhead */
    iface_attr->overhead          = 40e-9;

    /* Tag Offload */
    uct_rc_mlx5_tag_query(iface, iface_attr, max_inline,
                          UCT_RC_MLX5_TM_EAGER_ZCOPY_MAX_IOV(av_size));
}

void uct_rc_mlx5_iface_common_update_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                            uct_ib_iface_t *ib_iface)
{
#if !HAVE_DECL_MLX5DV_INIT_OBJ
    uct_ib_mlx5_update_cq_ci(ib_iface->cq[UCT_IB_DIR_TX], iface->cq[UCT_IB_DIR_TX].cq_ci);
    uct_ib_mlx5_update_cq_ci(ib_iface->cq[UCT_IB_DIR_RX], iface->cq[UCT_IB_DIR_RX].cq_ci);
#endif
}

void uct_rc_mlx5_iface_common_sync_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                          uct_ib_iface_t *ib_iface)
{
#if !HAVE_DECL_MLX5DV_INIT_OBJ
    iface->cq[UCT_IB_DIR_TX].cq_ci = uct_ib_mlx5_get_cq_ci(ib_iface->cq[UCT_IB_DIR_TX]);
    iface->cq[UCT_IB_DIR_RX].cq_ci = uct_ib_mlx5_get_cq_ci(ib_iface->cq[UCT_IB_DIR_RX]);
#endif
}

int uct_rc_mlx5_iface_commom_clean(uct_ib_mlx5_cq_t *mlx5_cq,
                                   uct_ib_mlx5_srq_t *srq, uint32_t qpn)
{
    const size_t cqe_sz       = 1ul << mlx5_cq->cqe_size_log;
    struct mlx5_cqe64 *cqe, *dest;
    uct_ib_mlx5_srq_seg_t *seg;
    unsigned pi, idx;
    uint8_t owner_bit;
    int nfreed;

    pi = mlx5_cq->cq_ci;
    for (;;) {
        cqe = uct_ib_mlx5_get_cqe(mlx5_cq, pi);
        if (uct_ib_mlx5_cqe_is_hw_owned(cqe->op_own, pi, mlx5_cq->cq_length)) {
            break;
        }

        ucs_assert((cqe->op_own >> 4) != MLX5_CQE_INVALID);

        ++pi;
        if (pi == (mlx5_cq->cq_ci + mlx5_cq->cq_length - 1)) {
            break;
        }
    }

    ucs_memory_cpu_load_fence();

    /* Remove CQEs of the destroyed QP, so the driver would not see them and try
     * to remove them itself, creating a mess with the free-list.
     */
    nfreed = 0;
    while ((int)--pi - (int)mlx5_cq->cq_ci >= 0) {
        cqe = uct_ib_mlx5_get_cqe(mlx5_cq, pi);
        if ((ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER)) == qpn) {
            idx = ntohs(cqe->wqe_counter);
            if (srq) {
                seg = uct_ib_mlx5_srq_get_wqe(srq, idx);
                seg->srq.free = 1;
                ucs_trace("cq %p: freed srq seg[%d] of qpn 0x%x",
                          mlx5_cq, idx, qpn);
            }
            ++nfreed;
        } else if (nfreed) {
            dest = uct_ib_mlx5_get_cqe(mlx5_cq, pi + nfreed);
            owner_bit = dest->op_own & MLX5_CQE_OWNER_MASK;
            memcpy((void*)(dest + 1) - cqe_sz, (void*)(cqe + 1) - cqe_sz, cqe_sz);
            dest->op_own = (dest->op_own & ~MLX5_CQE_OWNER_MASK) | owner_bit;
        }
    }

    mlx5_cq->cq_ci             += nfreed;

    return nfreed;
}

ucs_status_t
uct_rc_mlx5_iface_common_devx_connect_qp(uct_rc_mlx5_iface_common_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         uint32_t dest_qp_num,
                                         const struct ibv_ah_attr *ah_attr)
{
#if HAVE_DEVX
    uint32_t in_2rtr[UCT_IB_MLX5DV_ST_SZ_DW(init2rtr_qp_in)] = {};
    uint32_t out_2rtr[UCT_IB_MLX5DV_ST_SZ_DW(init2rtr_qp_out)] = {};
    uint32_t in_2rts[UCT_IB_MLX5DV_ST_SZ_DW(rtr2rts_qp_in)] = {};
    uint32_t out_2rts[UCT_IB_MLX5DV_ST_SZ_DW(rtr2rts_qp_out)] = {};
    const uint8_t *gid = ah_attr->grh.dgid.raw;
    uint8_t mac[6];
    void *qpc;
    int ret;

    ucs_assert_always(qp->type == UCT_IB_MLX5_OBJ_TYPE_DEVX);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, opcode, UCT_IB_MLX5_CMD_OP_INIT2RTR_QP);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, qpn, qp->qp_num);
    UCT_IB_MLX5DV_SET(init2rtr_qp_in, in_2rtr, opt_param_mask, 14);

    qpc = UCT_IB_MLX5DV_ADDR_OF(init2rtr_qp_in, in_2rtr, qpc);
    UCT_IB_MLX5DV_SET(qpc, qpc, mtu, 5);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_msg_max, 30);
    UCT_IB_MLX5DV_SET(qpc, qpc, remote_qpn, dest_qp_num);
    if (uct_ib_iface_is_roce(&iface->super.super)) {
        mac[0] = gid[8] ^ 0x02;
        memcpy(mac + 1, gid + 9, 2);
        memcpy(mac + 3, gid + 13, 3);
        memcpy(UCT_IB_MLX5DV_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32), mac, 6);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.hop_limit, 1);
    } else {
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.grh, ah_attr->is_global);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.rlid, ah_attr->dlid);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.mlid, ah_attr->src_path_bits & 0x7f);
        UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.hop_limit, ah_attr->grh.hop_limit);
    }

    memcpy(UCT_IB_MLX5DV_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip),
            &ah_attr->grh.dgid,
            UCT_IB_MLX5DV_FLD_SZ_BYTES(qpc, primary_address_path.rgid_rip));
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.vhca_port_num, ah_attr->port_num);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_ack_req_freq, 8);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_rra_max, 2);
    UCT_IB_MLX5DV_SET(qpc, qpc, atomic_mode, 5);
    UCT_IB_MLX5DV_SET(qpc, qpc, rre, 1);
    UCT_IB_MLX5DV_SET(qpc, qpc, rwe, 1);
    UCT_IB_MLX5DV_SET(qpc, qpc, rae, 1);
    UCT_IB_MLX5DV_SET(qpc, qpc, min_rnr_nak, iface->super.config.min_rnr_timer);

    ret = mlx5dv_devx_obj_modify(qp->devx.obj, in_2rtr, sizeof(in_2rtr),
                                 out_2rtr, sizeof(out_2rtr));
    if (ret) {
        ucs_error("mlx5dv_devx_obj_modify(2RTR) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(init2rtr_qp_out, out_2rtr, syndrome));
        return UCS_ERR_IO_ERROR;
    }

    UCT_IB_MLX5DV_SET(rtr2rts_qp_in, in_2rts, opcode, UCT_IB_MLX5_CMD_OP_RTR2RTS_QP);
    UCT_IB_MLX5DV_SET(rtr2rts_qp_in, in_2rts, qpn, qp->qp_num);

    qpc = UCT_IB_MLX5DV_ADDR_OF(rst2init_qp_in, in_2rts, qpc);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_ack_req_freq, 8);
    UCT_IB_MLX5DV_SET(qpc, qpc, log_sra_max, ucs_ilog2_or0(iface->super.config.max_rd_atomic));
    UCT_IB_MLX5DV_SET(qpc, qpc, retry_count, iface->super.config.retry_cnt);
    UCT_IB_MLX5DV_SET(qpc, qpc, rnr_retry, iface->super.config.rnr_retry);
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.ack_timeout, iface->super.config.timeout);
    UCT_IB_MLX5DV_SET(qpc, qpc, primary_address_path.log_rtm, iface->super.config.exp_backoff);

    ret = mlx5dv_devx_obj_modify(qp->devx.obj, in_2rts, sizeof(in_2rts),
                                 out_2rts, sizeof(out_2rts));
    if (ret) {
        ucs_error("mlx5dv_devx_obj_modify(2RTS) failed, syndrome %x: %m",
                  UCT_IB_MLX5DV_GET(rtr2rts_qp_out, out_2rts, syndrome));
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

