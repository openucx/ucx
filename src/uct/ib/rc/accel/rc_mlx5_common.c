/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rc_mlx5.inl"

#include <uct/api/uct.h>
#include <uct/ib/rc/base/rc_iface.h>
#include <ucs/arch/bitops.h>
#include <ucs/profile/profile.h>


ucs_config_field_t uct_rc_mlx5_common_config_table[] = {
  {UCT_IB_CONFIG_PREFIX, "", NULL,
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

  {"TM_MP_SRQ_ENABLE", "try",
   "Enable multi-packet SRQ support. Relevant for hardware tag-matching only.",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, tm.mp_enable),
   UCS_CONFIG_TYPE_TERNARY},

  {"TM_MP_NUM_STRIDES", "8",
   "Number of strides used per single receive WQE for hardware tag-matching\n"
   "unexpected messages. Can be 8 or 16 only. Relevant when MP SRQ is enabled.",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, tm.mp_num_strides),
   UCS_CONFIG_TYPE_ULUNITS},

  {"TM_MAX_BCOPY", NULL, "",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, tm.seg_size),
   UCS_CONFIG_TYPE_MEMUNITS},

  {"EXP_BACKOFF", "0",
   "Exponential Backoff Timeout Multiplier. ACK timeout will be multiplied \n"
   "by 2^EXP_BACKOFF every consecutive retry.",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, exp_backoff),
   UCS_CONFIG_TYPE_UINT},

  {"SRQ_TOPO", "cyclic,cyclic_emulated",
   "List of SRQ topology types in order of preference. Supported types are:\n"
   "\n"
   "list              SRQ is organized as a buffer containing linked list of WQEs.\n"
   "\n"
   "cyclic            SRQ is organized as a continuous array of WQEs. Requires DEVX.\n"
   "\n"
   "cyclic_emulated   SRQ is organized as a continuous array of WQEs, but HW\n"
   "                  treats it as a linked list. Doesn`t require DEVX.",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, srq_topo),
   UCS_CONFIG_TYPE_STRING_ARRAY},

  {"LOG_ACK_REQ_FREQ", "8",
   "Log of the ack frequency for requests, when using DEVX. Valid values are: 0-"
    UCS_PP_MAKE_STRING(UCT_RC_MLX5_MAX_LOG_ACK_REQ_FREQ) ".",
   ucs_offsetof(uct_rc_mlx5_iface_common_config_t, log_ack_req_freq),
   UCS_CONFIG_TYPE_UINT},

  {NULL}
};


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_iface_srq_set_seg(uct_rc_mlx5_iface_common_t *iface,
                              uct_ib_mlx5_srq_seg_t *seg)
{
    uct_ib_iface_recv_desc_t *desc;
    uint64_t desc_map;
    void *hdr;
    int i;

    desc_map = ~seg->srq.ptr_mask & UCS_MASK(iface->tm.mp.num_strides);
    ucs_for_each_bit(i, desc_map) {
        UCT_TL_IFACE_GET_RX_DESC(&iface->super.super.super, &iface->super.rx.mp,
                                 desc, return UCS_ERR_NO_MEMORY);

        /* Set receive data segment pointer. Length is pre-initialized. */
        hdr                = uct_ib_iface_recv_desc_hdr(&iface->super.super,
                                                        desc);
        seg->srq.ptr_mask |= UCS_BIT(i);
        seg->srq.desc      = desc; /* Optimization for non-MP case (1 stride) */
        seg->dptr[i].lkey  = htonl(desc->lkey);
        seg->dptr[i].addr  = htobe64((uintptr_t)hdr);
        VALGRIND_MAKE_MEM_NOACCESS(hdr, iface->super.super.config.seg_size);
    }

    return UCS_OK;
}

/* Update resources and write doorbell record */
static UCS_F_ALWAYS_INLINE void
uct_rc_mlx5_iface_update_srq_res(uct_rc_iface_t *iface, uct_ib_mlx5_srq_t *srq,
                                 uint16_t wqe_index, uint16_t count)
{
    ucs_assert(iface->rx.srq.available >= count);

    if (count == 0) {
        return;
    }

    srq->ready_idx              = wqe_index;
    srq->sw_pi                 += count;
    iface->rx.srq.available    -= count;
    ucs_memory_cpu_store_fence();
    *srq->db                    = htonl(srq->sw_pi);
}

unsigned uct_rc_mlx5_iface_srq_post_recv(uct_rc_mlx5_iface_common_t *iface)
{
    uct_ib_mlx5_srq_t *srq   = &iface->rx.srq;
    uct_rc_iface_t *rc_iface = &iface->super;
    uct_ib_mlx5_srq_seg_t *seg;
    uint16_t count, wqe_index, next_index;

    /* Make sure the union is right */
    UCS_STATIC_ASSERT(ucs_offsetof(uct_ib_mlx5_srq_seg_t, mlx5_srq.next_wqe_index) ==
                      ucs_offsetof(uct_ib_mlx5_srq_seg_t, srq.next_wqe_index));
    UCS_STATIC_ASSERT(ucs_offsetof(uct_ib_mlx5_srq_seg_t, dptr) ==
                      sizeof(struct mlx5_wqe_srq_next_seg));

    ucs_assert(UCS_CIRCULAR_COMPARE16(srq->ready_idx, <=, srq->free_idx));
    ucs_assert(rc_iface->rx.srq.available > 0);

    wqe_index = srq->ready_idx;
    for (;;) {
        next_index = wqe_index + 1;
        seg = uct_ib_mlx5_srq_get_wqe(srq, next_index);
        if (UCS_CIRCULAR_COMPARE16(next_index, >, srq->free_idx)) {
            if (!seg->srq.free) {
                break;
            }

            ucs_assert(next_index == (uint16_t)(srq->free_idx + 1));
            seg->srq.free  = 0;
            srq->free_idx  = next_index;
        }

        if (uct_rc_mlx5_iface_srq_set_seg(iface, seg) != UCS_OK) {
            break;
        }

        wqe_index = next_index;
    }

    count = wqe_index - srq->sw_pi;
    uct_rc_mlx5_iface_update_srq_res(rc_iface, srq, wqe_index, count);
    ucs_assert(uct_ib_mlx5_srq_get_wqe(srq, srq->mask)->srq.next_wqe_index == 0);
    return count;
}

unsigned uct_rc_mlx5_iface_srq_post_recv_ll(uct_rc_mlx5_iface_common_t *iface)
{
    uct_ib_mlx5_srq_t *srq     = &iface->rx.srq;
    uct_rc_iface_t *rc_iface   = &iface->super;
    uct_ib_mlx5_srq_seg_t *seg = NULL;
    uint16_t count             = 0;
    uint16_t wqe_index, next_index;

    ucs_assert(rc_iface->rx.srq.available > 0);

    wqe_index = srq->ready_idx;
    seg       = uct_ib_mlx5_srq_get_wqe(srq, wqe_index);

    for (;;) {
        next_index = ntohs(seg->srq.next_wqe_index);
        if (next_index == (srq->free_idx & srq->mask)) {
            break;
        }
        seg = uct_ib_mlx5_srq_get_wqe(srq, next_index);

        if (uct_rc_mlx5_iface_srq_set_seg(iface, seg) != UCS_OK) {
            break;
        }

        wqe_index = next_index;
        count++;
    }

    uct_rc_mlx5_iface_update_srq_res(rc_iface, srq, wqe_index, count);
    return count;
}

void uct_rc_mlx5_iface_common_prepost_recvs(uct_rc_mlx5_iface_common_t *iface)
{
    /* prepost recvs only if quota available (recvs were not preposted
     * before) */
    if (iface->super.rx.srq.quota == 0) {
        return;
    }

    iface->super.rx.srq.available = iface->super.rx.srq.quota;
    iface->super.rx.srq.quota     = 0;
    uct_rc_mlx5_iface_srq_post_recv(iface);
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
#  ifdef ENABLE_STATS
static ucs_stats_class_t uct_rc_mlx5_tag_stats_class = {
    .name          = "tag",
    .num_counters  = UCT_RC_MLX5_STAT_TAG_LAST,
    .class_id      = UCS_STATS_CLASS_ID_INVALID,
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


static UCS_F_MAYBE_UNUSED
ucs_status_t uct_rc_mlx5_devx_create_cmd_qp(uct_rc_mlx5_iface_common_t *iface)
{
    uct_ib_mlx5_md_t *md = uct_ib_mlx5_iface_md(&iface->super.super);
    uct_ib_device_t *dev = &md->super.dev;
    struct ibv_ah_attr ah_attr = {};
    uct_ib_mlx5_qp_attr_t attr = {};
    ucs_status_t status;

    ucs_assert(iface->tm.cmd_wq.super.super.type == UCT_IB_MLX5_OBJ_TYPE_LAST);

    attr.super.qp_type          = IBV_QPT_RC;
    attr.super.cap.max_send_wr  = iface->tm.cmd_qp_len;
    attr.super.cap.max_send_sge = 1;
    attr.super.ibv.pd           = md->super.pd;
    attr.super.srq_num          = iface->rx.srq.srq_num;
    attr.super.port             = dev->first_port;
    attr.mmio_mode              = iface->tx.mmio_mode;
    status = uct_ib_mlx5_devx_create_qp(&iface->super.super,
                                        &iface->cq[UCT_IB_DIR_RX],
                                        &iface->cq[UCT_IB_DIR_RX],
                                        &iface->tm.cmd_wq.super.super,
                                        &iface->tm.cmd_wq.super,
                                        &attr);
    if (status != UCS_OK) {
        return status;
    }

    ah_attr.is_global = 1;
    ah_attr.grh.dgid  = iface->super.super.gid_info.gid;
    ah_attr.dlid      = uct_ib_device_port_attr(dev, attr.super.port)->lid;
    ah_attr.port_num  = dev->first_port;
    status            = uct_rc_mlx5_iface_common_devx_connect_qp(
            iface, &iface->tm.cmd_wq.super.super,
            iface->tm.cmd_wq.super.super.qp_num, &ah_attr,
            iface->super.super.config.path_mtu, 0, 1);
    if (status != UCS_OK) {
        goto err_destroy_qp;
    }

    return UCS_OK;

err_destroy_qp:
    uct_ib_mlx5_devx_destroy_qp(md, &iface->tm.cmd_wq.super.super);
    return status;
}

static UCS_F_MAYBE_UNUSED struct ibv_qp*
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

    qp = UCS_PROFILE_CALL_ALWAYS(ibv_create_qp, md->pd, &qp_init_attr);
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
    qp_attr.ah_attr.grh.dgid         = iface->super.super.gid_info.gid;
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
    if (iface->rx.srq.type == UCT_IB_MLX5_OBJ_TYPE_DEVX) {
        return uct_rc_mlx5_devx_create_cmd_qp(iface);
    } else {
        qp = uct_rc_mlx5_verbs_create_cmd_qp(iface);
    }
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

    status = UCS_STATS_NODE_ALLOC(&iface->tm.stats,
                                  &uct_rc_mlx5_tag_stats_class,
                                  iface->stats, "");
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
    uct_ib_mlx5_md_t *md = uct_ib_mlx5_iface_md(&iface->super.super);
    uct_rc_mlx5_mp_hash_key_t key_gid;
    uint64_t key_lid;
    void *recv_buffer;

    if (!UCT_RC_MLX5_TM_ENABLED(iface)) {
        return;
    }

    uct_ib_mlx5_destroy_qp(md, &iface->tm.cmd_wq.super.super);
    uct_ib_mlx5_qp_mmio_cleanup(&iface->tm.cmd_wq.super.super,
                                iface->tm.cmd_wq.super.reg);
    ucs_free(iface->tm.list);
    ucs_free(iface->tm.cmd_wq.ops);
    uct_rc_mlx5_tag_cleanup(iface);

    kh_foreach_key(&iface->tm.tag_addrs, recv_buffer, {
        ucs_debug("destroying iface %p, with recv buffer %p offloaded to the HW",
                  iface, recv_buffer);
    });
    kh_destroy_inplace(uct_rc_mlx5_tag_addrs, &iface->tm.tag_addrs);

    if (!UCT_RC_MLX5_MP_ENABLED(iface)) {
        return;
    }

    kh_foreach_key(&iface->tm.mp.hash_lid, key_lid, {
        ucs_debug("destroying iface %p with partially received rx msg (key: %lu)",
                  iface, key_lid);
    });
    kh_destroy_inplace(uct_rc_mlx5_mp_hash_lid, &iface->tm.mp.hash_lid);

    kh_foreach_key(&iface->tm.mp.hash_gid, key_gid, {
        ucs_debug("destroying iface %p with partially received rx msg (key: %lu-%u)",
                  iface, key_gid.guid, key_gid.qp_num);
    });
    kh_destroy_inplace(uct_rc_mlx5_mp_hash_gid, &iface->tm.mp.hash_gid);

    ucs_mpool_cleanup(&iface->tm.mp.tx_mp, 1);
}

void uct_rc_mlx5_iface_fill_attr(uct_rc_mlx5_iface_common_t *iface,
                                 uct_ib_mlx5_qp_attr_t *qp_attr,
                                 unsigned max_send_wr,
                                 uct_ib_mlx5_srq_t *srq)
{
    switch (srq->type) {
    case UCT_IB_MLX5_OBJ_TYPE_VERBS:
        uct_rc_iface_fill_attr(&iface->super, &qp_attr->super, max_send_wr,
                               srq->verbs.srq);
        break;
    case UCT_IB_MLX5_OBJ_TYPE_DEVX:
        uct_rc_iface_fill_attr(&iface->super, &qp_attr->super, max_send_wr, NULL);
        qp_attr->mmio_mode = iface->tx.mmio_mode;
        break;
    case UCT_IB_MLX5_OBJ_TYPE_LAST:
        break;
    }

    qp_attr->super.srq_num = srq->srq_num;
}

ucs_status_t
uct_rc_mlx5_common_iface_init_rx(uct_rc_mlx5_iface_common_t *iface,
                                 const uct_rc_iface_common_config_t *rc_config)
{
    uct_ib_mlx5_md_t *md = uct_ib_mlx5_iface_md(&iface->super.super);
    ucs_status_t status;

    ucs_assert(iface->config.srq_topo != UCT_RC_MLX5_SRQ_TOPO_CYCLIC);

    status = uct_rc_iface_init_rx(&iface->super, rc_config,
                                  &iface->rx.srq.verbs.srq);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_mlx5_verbs_srq_init(&iface->rx.srq, iface->rx.srq.verbs.srq,
                                        iface->super.super.config.seg_size,
                                        iface->tm.mp.num_strides);
    if (status != UCS_OK) {
        goto err_free_srq;
    }

    iface->rx.srq.type = UCT_IB_MLX5_OBJ_TYPE_VERBS;
    return UCS_OK;

err_free_srq:
    uct_rc_mlx5_destroy_srq(md, &iface->rx.srq);
err:
    return status;
}

void uct_rc_mlx5_destroy_srq(uct_ib_mlx5_md_t *md, uct_ib_mlx5_srq_t *srq)
{
    switch (srq->type) {
    case UCT_IB_MLX5_OBJ_TYPE_VERBS:
        uct_ib_destroy_srq(srq->verbs.srq);
        break;
    case UCT_IB_MLX5_OBJ_TYPE_DEVX:
#if HAVE_DEVX
        uct_ib_mlx5_devx_obj_destroy(srq->devx.obj, "SRQ");
        uct_rc_mlx5_devx_cleanup_srq(md, srq);
#endif
        break;
    case UCT_IB_MLX5_OBJ_TYPE_LAST:
        break;
    }
}

void uct_rc_mlx5_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_rc_mlx5_release_desc_t *release = ucs_derived_of(self,
                                                         uct_rc_mlx5_release_desc_t);
    void *ib_desc = (char*)desc - release->offset;
    ucs_mpool_put_inline(ib_desc);
}

#if IBV_HW_TM
/* tag is passed as parameter, because some (but not all!) transports may need
 * to translate TMH to LE */
void uct_rc_mlx5_handle_unexp_rndv(uct_rc_mlx5_iface_common_t *iface,
                                   struct ibv_tmh *tmh, uct_tag_t tag,
                                   struct mlx5_cqe64 *cqe, unsigned flags,
                                   unsigned byte_len, int poll_flags)
{
    uct_rc_mlx5_tmh_priv_data_t *priv = (uct_rc_mlx5_tmh_priv_data_t*)tmh->reserved;
    struct ibv_rvh *rvh;
    unsigned tm_hdrs_len;
    unsigned rndv_usr_hdr_len;
    size_t rndv_data_len;
    void *rndv_usr_hdr;
    ucs_status_t status;
    char packed_rkey[UCT_IB_MD_PACKED_RKEY_SIZE];

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

    /* When MP XRQ is configured, RTS is always a single fragment message */
    ucs_assert(UCT_RC_MLX5_SINGLE_FRAG_MSG(flags));

    memcpy((char*)rndv_usr_hdr - priv->length, &priv->data, priv->length);

    /* Create "packed" rkey to pass it in the callback */
    uct_ib_md_pack_rkey(ntohl(rvh->rkey), UCT_IB_INVALID_MKEY, packed_rkey);

    /* Do not pass flags to cb, because rkey is allocated on stack */
    status = iface->tm.rndv_unexp.cb(iface->tm.rndv_unexp.arg, 0, tag,
                                   (char *)rndv_usr_hdr - priv->length,
                                   rndv_usr_hdr_len + priv->length,
                                   be64toh(rvh->va), rndv_data_len,
                                   packed_rkey);

    uct_rc_mlx5_iface_unexp_consumed(iface, iface->tm.rndv_desc.offset,
                                     &iface->tm.rndv_desc.super, cqe,
                                     status, ntohs(cqe->wqe_counter),
                                     poll_flags);

    UCT_RC_MLX5_TM_STAT(iface, RX_RNDV_UNEXP);
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
    .obj_cleanup   = NULL,
    .obj_str       = NULL
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
    ucs_mpool_params_t mp_params;

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

    obj.dv.dm.in  = data->dm;
    obj.dv.dm.out = &dvdm;
    uct_ib_mlx5dv_init_obj(&obj, MLX5DV_OBJ_DM);
    data->start_va = dvdm.buf;

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = sizeof(uct_rc_iface_send_desc_t);
    mp_params.elems_per_chunk = data->seg_count;
    mp_params.max_elems       = data->seg_count;
    mp_params.ops             = &uct_dm_iface_mpool_ops;
    mp_params.name            = "mlx5_dm_desc";
    status = ucs_mpool_init(&mp_params, &data->mp);
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

void uct_rc_mlx5_init_rx_tm_common(uct_rc_mlx5_iface_common_t *iface,
                                   const uct_rc_iface_common_config_t *config,
                                   unsigned rndv_hdr_len)
{
    uct_ib_md_t *md       = uct_ib_iface_md(&iface->super.super);
    unsigned tmh_hdrs_len = sizeof(struct ibv_tmh) + rndv_hdr_len;
    ucs_status_t status;

    iface->tm.eager_desc.super.cb = uct_rc_mlx5_release_desc;
    iface->tm.rndv_desc.super.cb  = uct_rc_mlx5_release_desc;

    if (UCT_RC_MLX5_MP_ENABLED(iface)) {
        iface->tm.eager_desc.offset = sizeof(struct ibv_tmh) +
                                      iface->super.super.config.rx_headroom_offset;
        iface->tm.am_desc.offset    = sizeof(uct_rc_mlx5_hdr_t) +
                                      iface->super.super.config.rx_headroom_offset;
        status = uct_iface_mpool_init(&iface->super.super.super,
                                      &iface->tm.mp.tx_mp,
                                      sizeof(uct_rc_iface_send_desc_t) +
                                      iface->tm.max_bcopy,
                                      sizeof(uct_rc_iface_send_desc_t),
                                      UCS_SYS_CACHE_LINE_SIZE,
                                      &config->super.tx.mp,
                                      iface->super.config.tx_qp_len,
                                      uct_rc_iface_send_desc_init,
                                      "tag_eager_send_desc");
        if (status != UCS_OK) {
            return;
        }

        kh_init_inplace(uct_rc_mlx5_mp_hash_lid, &iface->tm.mp.hash_lid);
        kh_init_inplace(uct_rc_mlx5_mp_hash_gid, &iface->tm.mp.hash_gid);

        iface->tm.bcopy_mp  = &iface->tm.mp.tx_mp;
        iface->tm.max_zcopy = uct_ib_iface_port_attr(&iface->super.super)->max_msg_sz;

        ucs_debug("MP WQ config: iface %p stride size %d, strides per WQE %d",
                  iface, iface->super.super.config.seg_size,
                  iface->tm.mp.num_strides);
    } else {
        iface->tm.eager_desc.offset = sizeof(struct ibv_tmh) -
                                      sizeof(uct_rc_mlx5_hdr_t) +
                                      iface->super.super.config.rx_headroom_offset;
        iface->tm.bcopy_mp          = &iface->super.tx.mp;
        iface->tm.max_zcopy         = iface->super.super.config.seg_size;
    }

    iface->tm.rndv_desc.offset    = iface->tm.eager_desc.offset + rndv_hdr_len;

    ucs_assert(IBV_DEVICE_TM_CAPS(&md->dev, max_rndv_hdr_size) >= tmh_hdrs_len);
    iface->tm.max_rndv_data       = IBV_DEVICE_TM_CAPS(&md->dev, max_rndv_hdr_size) -
                                    tmh_hdrs_len;

    /* Init ptr array to store completions of RNDV operations. Index in
     * ptr_array is used as operation ID and is passed in "app_context"
     * of TM header. */
    ucs_ptr_array_init(&iface->tm.rndv_comps, "tm_rndv_completions");

    /* Set of addresses posted to the HW. Used to avoid posting of the same
     * address more than once. */
    kh_init_inplace(uct_rc_mlx5_tag_addrs, &iface->tm.tag_addrs);
}

ucs_status_t uct_rc_mlx5_init_rx_tm(uct_rc_mlx5_iface_common_t *iface,
                                    const uct_rc_iface_common_config_t *config,
                                    struct ibv_srq_init_attr_ex *srq_attr,
                                    unsigned rndv_hdr_len)
{
    uct_ib_md_t *md = uct_ib_iface_md(&iface->super.super);
    ucs_status_t status;

    ucs_assert(iface->config.srq_topo != UCT_RC_MLX5_SRQ_TOPO_CYCLIC);

    uct_rc_mlx5_init_rx_tm_common(iface, config, rndv_hdr_len);

    ucs_assert(iface->tm.mp.num_strides == 1); /* MP XRQ is supported with DEVX only */
    srq_attr->attr.max_sge        = 1;
    srq_attr->attr.max_wr         = ucs_max(UCT_IB_MLX5_XRQ_MIN_UWQ_POST,
                                            config->super.rx.queue_len);
    srq_attr->attr.srq_limit      = 0;
    srq_attr->srq_context         = iface;
    srq_attr->srq_type            = IBV_SRQT_TM;
    srq_attr->pd                  = md->pd;
    srq_attr->cq                  = iface->super.super.cq[UCT_IB_DIR_RX];
    srq_attr->tm_cap.max_num_tags = iface->tm.num_tags;

    uct_rc_mlx5_iface_tm_set_cmd_qp_len(iface);
    srq_attr->tm_cap.max_ops      = iface->tm.cmd_qp_len;
    srq_attr->comp_mask          |= IBV_SRQ_INIT_ATTR_TYPE |
                                    IBV_SRQ_INIT_ATTR_PD |
                                    IBV_SRQ_INIT_ATTR_CQ |
                                    IBV_SRQ_INIT_ATTR_TM;

    iface->rx.srq.verbs.srq = ibv_create_srq_ex(md->dev.ibv_context, srq_attr);
    if (iface->rx.srq.verbs.srq == NULL) {
        ucs_error("ibv_create_srq_ex(device=%s) failed: %m",
                  uct_ib_device_name(&md->dev));
        return UCS_ERR_IO_ERROR;
    }

    iface->super.rx.srq.quota = srq_attr->attr.max_wr;

    status = uct_ib_mlx5_verbs_srq_init(&iface->rx.srq, iface->rx.srq.verbs.srq,
                                        iface->super.super.config.seg_size,
                                        iface->tm.mp.num_strides);
    if (status != UCS_OK) {
        goto err_free_srq;
    }

    iface->rx.srq.type        = UCT_IB_MLX5_OBJ_TYPE_VERBS;
    ucs_debug("Tag Matching enabled: tag list size %d", iface->tm.num_tags);
    return UCS_OK;

err_free_srq:
    uct_ib_destroy_srq(iface->rx.srq.verbs.srq);
    return status;
}
#endif

void uct_rc_mlx5_tag_cleanup(uct_rc_mlx5_iface_common_t *iface)
{
#if IBV_HW_TM
    if (UCT_RC_MLX5_TM_ENABLED(iface)) {
        ucs_ptr_array_cleanup(&iface->tm.rndv_comps, 1);
        UCS_STATS_NODE_FREE(iface->tm.stats);
    }
#endif
}

static void uct_rc_mlx5_tag_query(uct_rc_mlx5_iface_common_t *iface,
                                  uct_iface_attr_t *iface_attr,
                                  size_t max_inline, size_t max_tag_eager_iov)
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

    port_attr = uct_ib_iface_port_attr(&iface->super.super);
    iface_attr->cap.tag.rndv.max_zcopy       = port_attr->max_msg_sz;

    /* TMH can carry 2 additional bytes of private data */
    iface_attr->cap.tag.rndv.max_hdr         = iface->tm.max_rndv_data +
                                               UCT_RC_MLX5_TMH_PRIV_LEN;
    iface_attr->cap.tag.rndv.max_iov         = 1;
    iface_attr->cap.tag.recv.max_zcopy       = port_attr->max_msg_sz;
    iface_attr->cap.tag.recv.max_iov         = 1;
    iface_attr->cap.tag.recv.min_recv        = 0;
    iface_attr->cap.tag.recv.max_outstanding = iface->tm.num_tags;
    iface_attr->cap.tag.eager.max_iov        = max_tag_eager_iov;
    iface_attr->cap.tag.eager.max_bcopy      = iface->tm.max_bcopy - eager_hdr_size;
    iface_attr->cap.tag.eager.max_zcopy      = iface->tm.max_zcopy - eager_hdr_size;
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

#if HAVE_DECL_MLX5DV_CREATE_QP
void uct_rc_mlx5_common_fill_dv_qp_attr(uct_rc_mlx5_iface_common_t *iface,
                                        uct_ib_qp_init_attr_t *qp_attr,
                                        struct mlx5dv_qp_init_attr *dv_attr,
                                        unsigned scat2cqe_dir_mask)
{
#if HAVE_DECL_MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE
    if ((scat2cqe_dir_mask & UCS_BIT(UCT_IB_DIR_RX)) &&
        (iface->super.super.config.max_inl_cqe[UCT_IB_DIR_RX] == 0)) {
        /* make sure responder scatter2cqe is disabled */
        dv_attr->create_flags |= MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
        dv_attr->comp_mask    |= MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
    }
#endif

    if (scat2cqe_dir_mask & UCS_BIT(UCT_IB_DIR_TX)) {
        if (iface->super.super.config.max_inl_cqe[UCT_IB_DIR_TX] == 0) {
            /* Tell the driver to not signal all send WRs, so it will disable
             * requester scatter2cqe. Set this also for older driver which
             * doesn't support specific scatter2cqe flags.
             */
            qp_attr->sq_sig_all = 0;
        }
#if HAVE_DECL_MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE
        else if (!(dv_attr->create_flags &
                   MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE)) {
            /* force-enable requester scatter2cqe, regardless of SIGNAL_ALL_WR,
             * unless it was already disabled on responder side (otherwise
             * mlx5 driver check fails) */
            dv_attr->create_flags |= MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE;
            dv_attr->comp_mask    |= MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
        }
#endif
    }
}
#endif

void uct_rc_mlx5_iface_common_query(uct_ib_iface_t *ib_iface,
                                    uct_iface_attr_t *iface_attr,
                                    size_t max_inline, size_t max_tag_eager_iov)
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
    iface_attr->overhead = 40e-9;

    /* Tag Offload */
    uct_rc_mlx5_tag_query(iface, iface_attr, max_inline, max_tag_eager_iov);
}

ucs_status_t
uct_rc_mlx5_iface_common_create_cq(uct_ib_iface_t *ib_iface, uct_ib_dir_t dir,
                                   const uct_ib_iface_init_attr_t *init_attr,
                                   int preferred_cpu, size_t inl)
{
    uct_rc_mlx5_iface_common_t *iface =
            ucs_derived_of(ib_iface, uct_rc_mlx5_iface_common_t);
    uct_ib_mlx5_cq_t *uct_cq          = &iface->cq[dir];
#if HAVE_DEVX
    uct_ib_mlx5_md_t *md              = uct_ib_mlx5_iface_md(ib_iface);

    if (md->flags & UCT_IB_MLX5_MD_FLAG_DEVX_CQ) {
        return uct_ib_mlx5_devx_create_cq(ib_iface, dir, init_attr, uct_cq,
                                          preferred_cpu, inl);
    }
#endif

    uct_cq->type = UCT_IB_MLX5_OBJ_TYPE_VERBS;
    return uct_ib_mlx5_create_cq(ib_iface, dir, init_attr, uct_cq,
                                 preferred_cpu, inl);
}

void uct_rc_mlx5_iface_common_event_cq(uct_ib_iface_t *ib_iface,
                                       uct_ib_dir_t dir)
{
    uct_rc_mlx5_iface_common_t *iface = ucs_derived_of(ib_iface,
                                                       uct_rc_mlx5_iface_common_t);
    iface->cq[dir].cq_sn++;
}

void uct_rc_mlx5_iface_common_destroy_cq(uct_ib_iface_t *ib_iface,
                                         uct_ib_dir_t dir)
{
    uct_rc_mlx5_iface_common_t *iface =
            ucs_derived_of(ib_iface, uct_rc_mlx5_iface_common_t);

    uct_ib_mlx5_destroy_cq(ib_iface, &iface->cq[dir], dir);
}

int uct_rc_mlx5_iface_commom_clean(uct_ib_mlx5_cq_t *mlx5_cq,
                                   uct_ib_mlx5_srq_t *srq, uint32_t qpn)
{
    const size_t cqe_sz       = 1ul << mlx5_cq->cqe_size_log;
    struct mlx5_cqe64 *cqe, *dest, *unzipped_cqe;
    uct_ib_mlx5_srq_seg_t *seg;
    unsigned pi, idx;
    uint8_t owner_bit;
    int nfreed;

    pi = mlx5_cq->cq_ci;
    for (;;) {
        cqe = uct_ib_mlx5_get_cqe(mlx5_cq, pi);

        if (uct_ib_mlx5_cqe_is_hw_owned(mlx5_cq, cqe, pi,
                                        UCT_IB_MLX5_POLL_FLAG_CQE_ZIP)) {
            break;
        } else if (uct_ib_mlx5_check_and_init_zipped(mlx5_cq, cqe)) {
            unzipped_cqe = uct_ib_mlx5_iface_cqe_unzip(mlx5_cq);
            memcpy(cqe, unzipped_cqe, sizeof(*cqe));
        } else {
            mlx5_cq->cq_unzip.title_cqe_valid = 0;
        }

        ucs_assert((cqe->op_own >> 4) != MLX5_CQE_INVALID);

        ++pi;
        if (pi == (mlx5_cq->cq_ci + (1 << mlx5_cq->cq_length_log) - 1)) {
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
            memcpy(UCS_PTR_BYTE_OFFSET(dest + 1, -cqe_sz),
                   UCS_PTR_BYTE_OFFSET(cqe + 1, -cqe_sz), cqe_sz);
            dest->op_own = (dest->op_own & ~MLX5_CQE_OWNER_MASK) | owner_bit;
        }
    }

    mlx5_cq->cq_ci += nfreed;
    uct_ib_mlx5_update_db_cq_ci(mlx5_cq);

    return nfreed;
}
