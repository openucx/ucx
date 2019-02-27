/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "rc_mlx5.inl"

#include <uct/api/uct.h>
#include <uct/ib/rc/base/rc_iface.h>


#if HAVE_IBV_EXP_DM
/* uct_mlx5_dm_va is used to get pointer to DM mapped into process address space */
typedef struct uct_mlx5_dm_va {
    struct ibv_exp_dm  ibv_dm;
    size_t             length;
    uint64_t           *start_va;
} uct_mlx5_dm_va_t;
#endif


ucs_config_field_t uct_rc_mlx5_common_config_table[] = {
  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, mlx5_common),
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

  {"TM_MAX_BCOPY", "48k",
   "Maximal size of copy-out sends when tag-matching offload is enabled",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, tm.max_bcopy), UCS_CONFIG_TYPE_MEMUNITS},

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

static struct ibv_qp *
uct_rc_mlx5_get_cmd_qp(uct_rc_mlx5_iface_common_t *iface)
{
#if HAVE_STRUCT_MLX5_SRQ_CMD_QP
    iface->tm.cmd_qp = NULL;
    return uct_dv_get_cmd_qp(iface->super.rx.srq.srq);
#else
    uct_ib_md_t *md = uct_ib_iface_md(&iface->super.super);
    struct ibv_qp_init_attr qp_init_attr = {};
    struct ibv_qp_attr qp_attr = {};
    uct_ib_device_t *ibdev = &md->dev;
    struct ibv_port_attr *port_attr;
    struct ibv_qp *qp;
    uint8_t port_num;
    int ret;

    port_num  = ibdev->first_port;
    port_attr = uct_ib_device_port_attr(ibdev, port_num);

    qp_init_attr.qp_type             = IBV_QPT_RC;
    qp_init_attr.send_cq             = iface->super.super.cq[UCT_IB_DIR_RX];
    qp_init_attr.recv_cq             = iface->super.super.cq[UCT_IB_DIR_RX];
    qp_init_attr.cap.max_send_sge    = 1;
    qp_init_attr.srq                 = iface->super.rx.srq.srq;
    qp_init_attr.cap.max_send_wr     = iface->tm.cmd_qp_len;

    qp = ibv_create_qp(md->pd, &qp_init_attr);
    if (qp == NULL) {
        ucs_error("failed to create TM control QP: %m");
        goto err;
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

    iface->tm.cmd_qp = qp;
    return qp;

err_destroy_qp:
    ibv_destroy_qp(qp);
err:
    return NULL;
#endif
}
#endif

ucs_status_t
uct_rc_mlx5_iface_common_tag_init(uct_rc_mlx5_iface_common_t *iface,
                                  uct_rc_mlx5_iface_common_config_t *config)
{
    ucs_status_t status = UCS_OK;
#if IBV_HW_TM
    struct ibv_qp *cmd_qp;
    int i;

    if (!UCT_RC_MLX5_TM_ENABLED(iface)) {
        return UCS_OK;
    }

    cmd_qp = uct_rc_mlx5_get_cmd_qp(iface);
    if (!cmd_qp) {
        status = UCS_ERR_NO_DEVICE;
        goto err_tag_cleanup;
    }

    status = uct_ib_mlx5_txwq_init(iface->super.super.super.worker,
                                   iface->tx.mmio_mode,
                                   &iface->tm.cmd_wq.super, cmd_qp);
    if (status != UCS_OK) {
        goto err_tag_cleanup;
    }

    iface->tm.cmd_wq.qp_num   = cmd_qp->qp_num;
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
        if (iface->tm.cmd_qp) {
            ibv_destroy_qp(iface->tm.cmd_qp);
        }
        uct_ib_mlx5_txwq_cleanup(&iface->tm.cmd_wq.super);
        ucs_free(iface->tm.list);
        ucs_free(iface->tm.cmd_wq.ops);
        uct_rc_mlx5_tag_cleanup(iface);
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

#if HAVE_IBV_EXP_DM
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
    ucs_status_t status;
    struct ibv_exp_alloc_dm_attr dm_attr;
    struct ibv_exp_reg_mr_in mr_in;

    data->seg_len      = ucs_min(ucs_align_up(config->dm.seg_len,
                                              sizeof(uct_rc_mlx5_dm_copy_data_t)),
                                 iface->super.config.seg_size);
    data->seg_count    = config->dm.count;
    data->seg_attached = 0;
    data->device       = uct_ib_iface_device(&iface->super);

    dm_attr.length     = data->seg_len * data->seg_count;
    dm_attr.comp_mask  = 0;
    data->dm           = ibv_exp_alloc_dm(data->device->ibv_context, &dm_attr);

    if (data->dm == NULL) {
        /* TODO: prompt warning? */
        ucs_debug("ibv_exp_alloc_dm(dev=%s length=%zu) failed: %m",
                  uct_ib_device_name(data->device), dm_attr.length);
        return UCS_ERR_NO_RESOURCE;
    }

    memset(&mr_in, 0, sizeof(mr_in));
    mr_in.pd           = uct_ib_iface_md(&iface->super)->pd;
    mr_in.comp_mask    = IBV_EXP_REG_MR_DM;
    mr_in.dm           = data->dm;
    mr_in.length       = dm_attr.length;
    data->mr           = ibv_exp_reg_mr(&mr_in);
    if (data->mr == NULL) {
        ucs_warn("ibv_exp_reg_mr() error - On Device Memory registration failed, %d %m", errno);
        status = UCS_ERR_NO_RESOURCE;
        goto failed_mr;
    }

    data->start_va = ((uct_mlx5_dm_va_t*)data->dm)->start_va;

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
    ibv_exp_free_dm(data->dm);
    data->dm = NULL;
    return status;
}

static void uct_rc_mlx5_iface_common_dm_tl_cleanup(uct_mlx5_dm_data_t *data)
{
    ucs_assert(data->dm != NULL);
    ucs_assert(data->mr != NULL);

    ucs_mpool_cleanup(&data->mp, 1);
    ibv_dereg_mr(data->mr);
    ibv_exp_free_dm(data->dm);
}
#endif

#if IBV_HW_TM
ucs_status_t uct_rc_mlx5_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                                    const uct_rc_mlx5_iface_common_config_t *config,
                                    struct ibv_exp_create_srq_attr *srq_init_attr,
                                    unsigned rndv_hdr_len,
                                    unsigned max_cancel_sync_ops)
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

#if HAVE_DECL_IBV_EXP_CREATE_SRQ
    /* Create TM-capable XRQ */
    srq_init_attr->base.attr.max_sge   = 1;
    srq_init_attr->base.attr.max_wr    = ucs_max(IBV_DEVICE_MIN_UWQ_POST,
                                                 config->super.super.rx.queue_len);
    srq_init_attr->base.attr.srq_limit = 0;
    srq_init_attr->base.srq_context    = iface;
    srq_init_attr->srq_type            = IBV_EXP_SRQT_TAG_MATCHING;
    srq_init_attr->pd                  = md->pd;
    srq_init_attr->cq                  = iface->super.super.cq[UCT_IB_DIR_RX];
    srq_init_attr->tm_cap.max_num_tags = iface->tm.num_tags;

    /* 2 ops for each tag (ADD + DEL) and extra ops for SYNC.
     * There can be up to "max_cancel_sync_ops" SYNC ops during cancellation.
     * Also we assume that there can be up to two pending SYNC ops during
     * unexpected messages flow. */
    iface->tm.cmd_qp_len = (2 * iface->tm.num_tags) + max_cancel_sync_ops + 2;
    srq_init_attr->tm_cap.max_ops = iface->tm.cmd_qp_len;
    srq_init_attr->comp_mask     |= IBV_EXP_CREATE_SRQ_CQ |
                                    IBV_EXP_CREATE_SRQ_TM;

    iface->super.rx.srq.srq = ibv_exp_create_srq(md->dev.ibv_context, srq_init_attr);
    if (iface->super.rx.srq.srq == NULL) {
        ucs_error("ibv_exp_create_srq(device=%s) failed: %m",
                  uct_ib_device_name(&md->dev));
        return UCS_ERR_IO_ERROR;
    }

    iface->super.rx.srq.quota = srq_init_attr->base.attr.max_wr;
#elif HAVE_DECL_IBV_CREATE_SRQ_EX
    srq_init_attr->attr.max_sge        = 1;
    srq_init_attr->attr.max_wr         = ucs_max(IBV_DEVICE_MIN_UWQ_POST,
                                                 config->super.super.rx.queue_len);
    srq_init_attr->attr.srq_limit      = 0;
    srq_init_attr->srq_context         = iface;
    srq_init_attr->srq_type            = IBV_SRQT_TM;
    srq_init_attr->pd                  = md->pd;
    srq_init_attr->cq                  = iface->super.super.cq[UCT_IB_DIR_RX];
    srq_init_attr->tm_cap.max_num_tags = iface->tm.num_tags;

    /* 2 ops for each tag (ADD + DEL) and extra ops for SYNC.
     * There can be up to "max_cancel_sync_ops" SYNC ops during cancellation.
     * Also we assume that there can be up to two pending SYNC ops during
     * unexpected messages flow. */
    iface->tm.cmd_qp_len = (2 * iface->tm.num_tags) + max_cancel_sync_ops + 2;
    srq_init_attr->tm_cap.max_ops = iface->tm.cmd_qp_len;
    srq_init_attr->comp_mask     |= IBV_SRQ_INIT_ATTR_TYPE |
                                    IBV_SRQ_INIT_ATTR_PD |
                                    IBV_SRQ_INIT_ATTR_CQ |
                                    IBV_SRQ_INIT_ATTR_TM;

    iface->super.rx.srq.srq = ibv_create_srq_ex(md->dev.ibv_context, srq_init_attr);
    if (iface->super.rx.srq.srq == NULL) {
        ucs_error("ibv_create_srq_ex(device=%s) failed: %m",
                  uct_ib_device_name(&md->dev));
        return UCS_ERR_IO_ERROR;
    }

    iface->super.rx.srq.quota = srq_init_attr->attr.max_wr;
#endif

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
    struct ibv_exp_port_attr* port_attr;

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
#if HAVE_IBV_EXP_DM
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
#if HAVE_IBV_EXP_DM
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

    if (dev->pci_fadd_arg_sizes || dev->pci_cswap_arg_sizes) {
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

ucs_status_t uct_rc_mlx5_iface_fence(uct_iface_h tl_iface, unsigned flags)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(tl_iface, uct_rc_mlx5_iface_common_t);
    uct_ib_md_t *md = uct_ib_iface_md(&iface->super.super);

    if (md->dev.pci_fadd_arg_sizes || md->dev.pci_cswap_arg_sizes) {
        iface->tx.next_fm = UCT_IB_MLX5_WQE_CTRL_FENCE_ATOMIC;
        iface->tx.fence_beat++;
    }

    UCT_TL_IFACE_STAT_FENCE(&iface->super.super.super);
    return UCS_OK;
}
