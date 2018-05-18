/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ud_mlx5.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/base/uct_md.h>
#include <uct/base/uct_log.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <string.h>
#include <arpa/inet.h> /* For htonl */

#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/ib/mlx5/ib_mlx5.inl>
#include <uct/ib/mlx5/ib_mlx5_dv.h>

#include <uct/ib/ud/base/ud_iface.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_def.h>
#include <uct/ib/ud/base/ud_inl.h>


static ucs_config_field_t uct_ud_mlx5_iface_config_table[] = {
  {"UD_", "", NULL,
   ucs_offsetof(uct_ud_mlx5_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_ud_iface_config_table)},

  {"", "", NULL,
   ucs_offsetof(uct_ud_mlx5_iface_config_t, mlx5_common),
   UCS_CONFIG_TYPE_TABLE(uct_ud_mlx5_iface_common_config_table)},

  {NULL}
};

static UCS_F_ALWAYS_INLINE size_t
uct_ud_mlx5_ep_ctrl_av_size(uct_ud_mlx5_ep_t *ep)
{
    return sizeof(struct mlx5_wqe_ctrl_seg) + uct_ib_mlx5_wqe_av_size(&ep->av);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_mlx5_post_send(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep,
                      uint8_t se, struct mlx5_wqe_ctrl_seg *ctrl, size_t wqe_size,
                      int max_log_sge)
{
    struct mlx5_wqe_datagram_seg *dgram = (void*)(ctrl + 1);

    uct_ib_mlx5_set_ctrl_seg(ctrl, iface->tx.wq.sw_pi, MLX5_OPCODE_SEND, 0,
                             iface->super.qp->qp_num,
                             uct_ud_mlx5_tx_moderation(iface) | se, wqe_size);
    uct_ib_mlx5_set_dgram_seg(dgram, &ep->av, ep->is_global ? &ep->grh_av : NULL,
                              IBV_QPT_UD);

    uct_ib_mlx5_log_tx(&iface->super.super, IBV_QPT_UD, ctrl,
                       iface->tx.wq.qstart, iface->tx.wq.qend,
                       max_log_sge, NULL, uct_ud_dump_packet);
    iface->super.tx.available -= uct_ib_mlx5_post_send(&iface->tx.wq, ctrl,
                                                       wqe_size);
    ucs_assert((int16_t)iface->tx.wq.bb_max >= iface->super.tx.available);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_mlx5_ep_tx_skb(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep,
                      uct_ud_send_skb_t *skb, uint8_t se, int max_log_sge)
{
    size_t ctrl_av_size = uct_ud_mlx5_ep_ctrl_av_size(ep);
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_data_seg *dptr;

    ctrl = iface->tx.wq.curr;
    dptr = uct_ib_mlx5_txwq_wrap_exact(&iface->tx.wq, (void*)ctrl + ctrl_av_size);
    uct_ib_mlx5_set_data_seg(dptr, skb->neth, skb->len, skb->lkey);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, skb->neth);
    uct_ud_mlx5_post_send(iface, ep, se, ctrl, ctrl_av_size + sizeof(*dptr), max_log_sge);
}

static inline void
uct_ud_mlx5_ep_tx_inl(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep,
                      const void *buf, unsigned length, uint8_t se)
{
    size_t ctrl_av_size = uct_ud_mlx5_ep_ctrl_av_size(ep);
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;

    ctrl = iface->tx.wq.curr;
    inl = uct_ib_mlx5_txwq_wrap_exact(&iface->tx.wq, (void*)ctrl + ctrl_av_size);
    inl->byte_count = htonl(length | MLX5_INLINE_SEG);
    uct_ib_mlx5_inline_copy(inl + 1, buf, length, &iface->tx.wq);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, (uct_ud_neth_t *)buf);
    uct_ud_mlx5_post_send(iface, ep, se, ctrl,
                          ctrl_av_size + sizeof(*inl) + length, INT_MAX);
}


static void uct_ud_mlx5_ep_tx_ctl_skb(uct_ud_ep_t *ud_ep, uct_ud_send_skb_t *skb,
                                      int solicited)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(ud_ep->super.super.iface,
                                                uct_ud_mlx5_iface_t);
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(ud_ep, uct_ud_mlx5_ep_t);
    uint8_t se;

    se = solicited ? MLX5_WQE_CTRL_SOLICITED : 0;
    if (skb->len >= iface->super.config.max_inline) {
        uct_ud_mlx5_ep_tx_skb(iface, ep, skb, se, INT_MAX);
    } else {
        uct_ud_mlx5_ep_tx_inl(iface, ep, skb->neth, skb->len, se);
    }
}

static UCS_F_NOINLINE void
uct_ud_mlx5_iface_post_recv(uct_ud_mlx5_iface_t *iface)
{
    unsigned batch = iface->super.super.config.rx_max_batch;
    struct mlx5_wqe_data_seg *rx_wqes;
    uint16_t pi, next_pi, count;
    uct_ib_iface_recv_desc_t *desc;

    rx_wqes = iface->rx.wq.wqes;
    pi      = iface->rx.wq.rq_wqe_counter & iface->rx.wq.mask;

    for (count = 0; count < batch; count ++) {
        next_pi = (pi + 1) &  iface->rx.wq.mask;
        ucs_prefetch(rx_wqes + next_pi);
        UCT_TL_IFACE_GET_RX_DESC(&iface->super.super.super, &iface->super.rx.mp,
                                 desc, break);
        rx_wqes[pi].lkey = htonl(desc->lkey);
        rx_wqes[pi].addr = htobe64((uintptr_t)uct_ib_iface_recv_desc_hdr(&iface->super.super, desc));
        pi = next_pi;
    }
    if (ucs_unlikely(count == 0)) {
        ucs_debug("iface(%p) failed to post receive wqes", iface);
        return;
    }
    pi = iface->rx.wq.rq_wqe_counter + count;
    iface->rx.wq.rq_wqe_counter = pi;
    iface->super.rx.available -= count;
    ucs_memory_cpu_fence();
    *iface->rx.wq.dbrec = htonl(pi);
}

static UCS_CLASS_INIT_FUNC(uct_ud_mlx5_ep_t, uct_iface_h tl_iface)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_mlx5_iface_t);
    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_ud_ep_t, &iface->super);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_mlx5_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_ud_mlx5_ep_t, uct_ud_ep_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_mlx5_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_mlx5_ep_t, uct_ep_t);


static ucs_status_t
uct_ud_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                        const void *buffer, unsigned length)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    size_t ctrl_av_size = uct_ud_mlx5_ep_ctrl_av_size(ep);
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_ud_am_short_hdr_t *am;
    uct_ud_neth_t *neth;
    uct_ud_send_skb_t *skb;
    size_t wqe_size;

    /* data a written directly into tx wqe, so it is impossible to use
     * common ud am code
     */
    UCT_CHECK_AM_ID(id);
    UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + sizeof(hdr) + length,
                    0, iface->super.config.max_inline, "am_short");

    uct_ud_enter(&iface->super);

    skb = uct_ud_ep_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        uct_ud_leave(&iface->super);
        return UCS_ERR_NO_RESOURCE;
    }

    ctrl = iface->tx.wq.curr;
    /* Set inline segment which has AM id, AM header, and AM payload */
    inl = uct_ib_mlx5_txwq_wrap_exact(&iface->tx.wq, (void*)ctrl + ctrl_av_size);
    wqe_size = length + sizeof(*am) + sizeof(*neth);
    inl->byte_count = htonl(wqe_size | MLX5_INLINE_SEG);

    /* assume that neth and am header fit into one bb */
    ucs_assert(sizeof(*am) + sizeof(*neth) < MLX5_SEND_WQE_BB);
    neth = (void*)(inl + 1);
    uct_ud_am_set_neth(neth, &ep->super, id);

    am      = (void*)(neth + 1);
    am->hdr = hdr;
    uct_ib_mlx5_inline_copy(am + 1, buffer, length, &iface->tx.wq);

    wqe_size += ctrl_av_size + sizeof(*inl);
    UCT_CHECK_LENGTH(wqe_size, 0, UCT_IB_MLX5_MAX_SEND_WQE_SIZE, "am_short");
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, neth);
    uct_ud_mlx5_post_send(iface, ep, 0, ctrl, wqe_size, INT_MAX);

    skb->len = sizeof(*neth) + sizeof(*am);
    memcpy(skb->neth, neth, skb->len);
    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 (char *)skb->neth + skb->len, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_ud_leave(&iface->super);
    return UCS_OK;
}

static ssize_t uct_ud_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                       uct_pack_callback_t pack_cb, void *arg,
                                       unsigned flags)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    uct_ud_send_skb_t *skb;
    ucs_status_t status;
    size_t length;

    uct_ud_enter(&iface->super);

    status = uct_ud_am_common(&iface->super, &ep->super, id, &skb);
    if (status != UCS_OK) {
        uct_ud_leave(&iface->super);
        return status;
    }

    length = uct_ud_skb_bcopy(skb, pack_cb, arg);
    UCT_UD_CHECK_BCOPY_LENGTH(&iface->super, length);

    uct_ud_mlx5_ep_tx_skb(iface, ep, skb, 0, INT_MAX);
    uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    uct_ud_leave(&iface->super);
    return length;
}

static ucs_status_t
uct_ud_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                        unsigned header_length, const uct_iov_t *iov,
                        size_t iovcnt, unsigned flags, uct_completion_t *comp)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    size_t ctrl_av_size = uct_ud_mlx5_ep_ctrl_av_size(ep);
    uct_ud_send_skb_t *skb;
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_ud_neth_t *neth;
    size_t inl_size, wqe_size;

    UCT_CHECK_IOV_SIZE(iovcnt, uct_ib_iface_get_max_iov(&iface->super.super),
                       "uct_ud_mlx5_ep_am_zcopy");
    UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + header_length, 0,
                     UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(UCT_IB_MLX5_AV_FULL_SIZE),
                     "am_zcopy header");
    UCT_UD_CHECK_ZCOPY_LENGTH(&iface->super, header_length,
                              uct_iov_total_length(iov, iovcnt));

    uct_ud_enter(&iface->super);

    skb = uct_ud_ep_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        uct_ud_leave(&iface->super);
        return UCS_ERR_NO_RESOURCE;
    }

    ctrl = iface->tx.wq.curr;
    inl = uct_ib_mlx5_txwq_wrap_exact(&iface->tx.wq, (void*)ctrl + ctrl_av_size);
    inl_size = header_length + sizeof(*neth);
    inl->byte_count = htonl(inl_size | MLX5_INLINE_SEG);

    neth = (void*)(inl + 1);
    uct_ud_am_set_neth(neth, &ep->super, id);
    /* force ACK_REQ because we want to call user completion ASAP */
    neth->packet_type |= UCT_UD_PACKET_FLAG_ACK_REQ;

    uct_ib_mlx5_inline_copy(neth + 1, header, header_length, &iface->tx.wq);

    wqe_size = ucs_align_up_pow2(ctrl_av_size + inl_size + sizeof(*inl),
                                 UCT_IB_MLX5_WQE_SEG_SIZE);
    wqe_size += uct_ib_mlx5_set_data_seg_iov(&iface->tx.wq, (void *)ctrl + wqe_size,
                                             iov, iovcnt);
    ucs_assert(wqe_size <= UCT_IB_MLX5_MAX_SEND_WQE_SIZE);

    UCT_UD_EP_HOOK_CALL_TX(&ep->super, neth);
    uct_ud_mlx5_post_send(iface, ep, 0, ctrl, wqe_size,
                          UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super.super));

    skb->len = sizeof(*neth) + header_length;
    memcpy(skb->neth, neth, sizeof(*neth));
    memcpy(skb->neth + 1, header, header_length);
    uct_ud_am_set_zcopy_desc(skb, iov, iovcnt, comp);

    uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY, header_length +
                      uct_iov_total_length(iov, iovcnt));
    uct_ud_leave(&iface->super);
    return UCS_INPROGRESS;
}

static ucs_status_t
uct_ud_mlx5_ep_put_short(uct_ep_h tl_ep, const void *buffer, unsigned length,
                         uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    size_t ctrl_av_size = uct_ud_mlx5_ep_ctrl_av_size(ep);
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_ud_put_hdr_t *put_hdr;
    uct_ud_neth_t *neth;
    uct_ud_send_skb_t *skb;
    size_t wqe_size;

    uct_ud_enter(&iface->super);

    skb = uct_ud_ep_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        uct_ud_leave(&iface->super);
        return UCS_ERR_NO_RESOURCE;
    }

    ctrl = iface->tx.wq.curr;
    /* Set inline segment which has AM id, AM header, and AM payload */
    inl = uct_ib_mlx5_txwq_wrap_exact(&iface->tx.wq, (void*)ctrl + ctrl_av_size);
    wqe_size = length + sizeof(*put_hdr) + sizeof(*neth);
    inl->byte_count = htonl(wqe_size | MLX5_INLINE_SEG);

    /* assume that neth and am header fit into one bb */
    ucs_assert(sizeof(*put_hdr) + sizeof(*neth) < MLX5_SEND_WQE_BB);
    neth = (void*)(inl + 1);
    uct_ud_neth_init_data(&ep->super, neth);
    uct_ud_neth_set_type_put(&ep->super, neth);
    uct_ud_neth_ack_req(&ep->super, neth);

    put_hdr = (uct_ud_put_hdr_t *)(neth+1);
    put_hdr->rva = remote_addr;

    uct_ib_mlx5_inline_copy(put_hdr + 1, buffer, length, &iface->tx.wq);

    wqe_size += ctrl_av_size + sizeof(*inl);
    UCT_CHECK_LENGTH(wqe_size, 0, UCT_IB_MLX5_MAX_SEND_WQE_SIZE, "put_short");
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, neth);
    uct_ud_mlx5_post_send(iface, ep, 0, ctrl, wqe_size, INT_MAX);

    skb->len = sizeof(*neth) + sizeof(*put_hdr);
    memcpy(skb->neth, neth, skb->len);
    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 (char *)skb->neth + skb->len, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    uct_ud_leave(&iface->super);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_ud_mlx5_iface_poll_rx(uct_ud_mlx5_iface_t *iface, int is_async)
{
    struct mlx5_cqe64 *cqe;
    uint16_t ci;
    uct_ib_iface_recv_desc_t *desc;
    uint32_t len;
    void *packet;
    unsigned count;

    ci     = iface->rx.wq.cq_wqe_counter & iface->rx.wq.mask;
    packet = (void *)be64toh(iface->rx.wq.wqes[ci].addr);
    ucs_prefetch(packet + UCT_IB_GRH_LEN);
    desc   = (uct_ib_iface_recv_desc_t *)(packet - iface->super.super.config.rx_hdr_offset);

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super, &iface->cq[UCT_IB_DIR_RX]);
    if (cqe == NULL) {
        count = 0;
        goto out;
    }

    ucs_memory_cpu_load_fence();

    ucs_assert(0 == (cqe->op_own &
               (MLX5_INLINE_SCATTER_32|MLX5_INLINE_SCATTER_64)));
    ucs_assert(ntohs(cqe->wqe_counter) == iface->rx.wq.cq_wqe_counter);

    iface->super.rx.available++;
    iface->rx.wq.cq_wqe_counter++;
    count = 1;
    len   = ntohl(cqe->byte_cnt);
    VALGRIND_MAKE_MEM_DEFINED(packet, len);

    if (!uct_ud_iface_check_grh(&iface->super, packet + UCT_IB_GRH_LEN,
                                (ntohl(cqe->flags_rqpn) >> 28) & 3)) {
        ucs_mpool_put_inline(desc);
        goto out;
    }

    uct_ib_mlx5_log_rx(&iface->super.super, IBV_QPT_UD, cqe, packet,
                       uct_ud_dump_packet);
    uct_ud_ep_process_rx(&iface->super,
                         (uct_ud_neth_t *)(packet + UCT_IB_GRH_LEN),
                         len - UCT_IB_GRH_LEN,
                         (uct_ud_recv_skb_t *)desc, is_async);
out:
    if (iface->super.rx.available >= iface->super.super.config.rx_max_batch) {
        /* we need to try to post buffers always. Otherwise it is possible
         * to run out of rx wqes if receiver is slow and there are always
         * cqe to process
         */
        uct_ud_mlx5_iface_post_recv(iface);
    }
    return count;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_ud_mlx5_iface_poll_tx(uct_ud_mlx5_iface_t *iface)
{
    struct mlx5_cqe64 *cqe;

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super, &iface->cq[UCT_IB_DIR_TX]);
    if (cqe == NULL) {
        return 0;
    }

    ucs_memory_cpu_load_fence();

    uct_ib_mlx5_log_cqe(cqe);
    iface->super.tx.available = uct_ib_mlx5_txwq_update_bb(&iface->tx.wq,
                                                           ntohs(cqe->wqe_counter));
    return 1;
}

static unsigned uct_ud_mlx5_iface_progress(uct_iface_h tl_iface)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_mlx5_iface_t);
    ucs_status_t status;
    unsigned n, count = 0;

    uct_ud_enter(&iface->super);
    uct_ud_iface_dispatch_zcopy_comps(&iface->super);

    status = uct_ud_iface_dispatch_pending_rx(&iface->super);
    if (ucs_likely(status == UCS_OK)) {
        do {
            n = uct_ud_mlx5_iface_poll_rx(iface, 0);
            count += n;
        } while ((n > 0) && (count < iface->super.super.config.rx_max_poll));
    }

    count += uct_ud_mlx5_iface_poll_tx(iface);
    uct_ud_iface_progress_pending(&iface->super, 0);
    uct_ud_leave(&iface->super);
    return count;
}

static unsigned uct_ud_mlx5_iface_async_progress(uct_ud_iface_t *ud_iface)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(ud_iface, uct_ud_mlx5_iface_t);
    unsigned n, count;

    count = 0;
    do {
        n = uct_ud_mlx5_iface_poll_rx(iface, 1);
        count += n;
    } while (n > 0);

    count += uct_ud_mlx5_iface_poll_tx(iface);

    uct_ud_iface_progress_pending(&iface->super, 1);

    return count;
}

static ucs_status_t
uct_ud_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    ucs_status_t status;

    ucs_trace_func("");
    status = uct_ud_iface_query(iface, iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->overhead       = 80e-9; /* Software overhead */
    iface_attr->cap.am.max_iov = uct_ib_iface_get_max_iov(&iface->super);

    iface_attr->cap.am.max_hdr = UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(UCT_IB_MLX5_AV_FULL_SIZE)
                                 - sizeof(uct_ud_neth_t);

    return UCS_OK;
}

static ucs_status_t
uct_ud_mlx5_ep_create_ah(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep,
                         const uct_ib_address_t *ib_addr,
                         const uct_ud_iface_addr_t *if_addr)
{
    ucs_status_t status;
    int is_global;

    status = uct_ud_mlx5_iface_get_av(&iface->super.super, &iface->mlx5_common,
                                      ib_addr, ep->super.path_bits, &ep->av,
                                      &ep->grh_av, &is_global);
    if (status != UCS_OK) {
        return status;
    }

    ep->is_global   = is_global;
    ep->av.dqp_dct |= htonl(uct_ib_unpack_uint24(if_addr->qp_num));
    return UCS_OK;
}

static ucs_status_t
uct_ud_mlx5_ep_create_connected(uct_iface_h iface_h,
                                const uct_device_addr_t *dev_addr,
                                const uct_iface_addr_t *iface_addr,
                                uct_ep_h *new_ep_p)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(iface_h, uct_ud_mlx5_iface_t);
    uct_ud_mlx5_ep_t *ep;
    uct_ud_ep_t *new_ud_ep;
    const uct_ud_iface_addr_t *if_addr = (const uct_ud_iface_addr_t *)iface_addr;
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;
    uct_ud_send_skb_t *skb;
    ucs_status_t status, status_ah;

    uct_ud_enter(&iface->super);
    status = uct_ud_ep_create_connected_common(&iface->super, ib_addr, if_addr,
                                               &new_ud_ep, &skb);
    if (status != UCS_OK &&
        status != UCS_ERR_NO_RESOURCE &&
        status != UCS_ERR_ALREADY_EXISTS) {
        uct_ud_leave(&iface->super);
        return status;
    }

    ep = ucs_derived_of(new_ud_ep, uct_ud_mlx5_ep_t);
    *new_ep_p = &ep->super.super.super;
    if (status == UCS_ERR_ALREADY_EXISTS) {
        uct_ud_leave(&iface->super);
        return UCS_OK;
    }

    status_ah = uct_ud_mlx5_ep_create_ah(iface, ep, ib_addr, if_addr);
    if (status_ah != UCS_OK) {
        uct_ud_ep_destroy_connected(&ep->super, ib_addr, if_addr);
        *new_ep_p = NULL;
        uct_ud_leave(&iface->super);
        return status_ah;
    }

    if (status == UCS_OK) {
        uct_ud_mlx5_ep_tx_ctl_skb(&ep->super, skb, 1);
        uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
        ep->super.flags |= UCT_UD_EP_FLAG_CREQ_SENT;
    }

    uct_ud_leave(&iface->super);
    return UCS_OK;
}

static ucs_status_t
uct_ud_mlx5_ep_connect_to_ep(uct_ep_h tl_ep,
                             const uct_device_addr_t *dev_addr,
                             const uct_ep_addr_t *uct_ep_addr)
{
    ucs_status_t status;
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    const uct_ud_ep_addr_t *ep_addr = (const uct_ud_ep_addr_t *)uct_ep_addr;
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;

    ucs_trace_func("");
    status = uct_ud_ep_connect_to_ep(&ep->super, ib_addr, ep_addr);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ud_mlx5_ep_create_ah(iface, ep, ib_addr, (const uct_ud_iface_addr_t *)ep_addr);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static ucs_status_t uct_ud_mlx5_iface_arm_cq(uct_ib_iface_t *ib_iface,
                                             uct_ib_dir_t dir,
                                             int solicited)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_ud_mlx5_iface_t);
#if HAVE_DECL_MLX5DV_INIT_OBJ
    return uct_ib_mlx5dv_arm_cq(&iface->cq[dir], solicited);
#else
    uct_ib_mlx5_update_cq_ci(iface->super.super.cq[dir],
                             iface->cq[dir].cq_ci);
    return uct_ib_iface_arm_cq(ib_iface, dir, solicited);
#endif
}

static ucs_status_t uct_ud_mlx5_ep_set_failed(uct_ib_iface_t *iface,
                                              uct_ep_h ep, ucs_status_t status)
{
    return uct_set_ep_failed(&UCS_CLASS_NAME(uct_ud_mlx5_ep_t), ep,
                             &iface->super.super, status);
}

static void uct_ud_mlx5_iface_event_cq(uct_ib_iface_t *ib_iface,
                                       uct_ib_dir_t dir)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_ud_mlx5_iface_t);

    iface->cq[dir].cq_sn++;
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_ud_mlx5_iface_t)(uct_iface_t*);

static uct_ud_iface_ops_t uct_ud_mlx5_iface_ops = {
    {
    {
    .ep_put_short             = uct_ud_mlx5_ep_put_short,
    .ep_am_short              = uct_ud_mlx5_ep_am_short,
    .ep_am_bcopy              = uct_ud_mlx5_ep_am_bcopy,
    .ep_am_zcopy              = uct_ud_mlx5_ep_am_zcopy,
    .ep_pending_add           = uct_ud_ep_pending_add,
    .ep_pending_purge         = uct_ud_ep_pending_purge,
    .ep_flush                 = uct_ud_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_ud_mlx5_ep_t),
    .ep_create_connected      = uct_ud_mlx5_ep_create_connected,
    .ep_destroy               = uct_ud_ep_disconnect ,
    .ep_get_address           = uct_ud_ep_get_address,
    .ep_connect_to_ep         = uct_ud_mlx5_ep_connect_to_ep,
    .iface_flush              = uct_ud_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_ud_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_ud_mlx5_iface_progress,
    .iface_event_fd_get       = uct_ib_iface_event_fd_get,
    .iface_event_arm          = uct_ud_iface_event_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_mlx5_iface_t),
    .iface_query              = uct_ud_mlx5_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_get_address        = uct_ud_iface_get_address,
    .iface_is_reachable       = uct_ib_iface_is_reachable
    },
    .arm_cq                   = uct_ud_mlx5_iface_arm_cq,
    .event_cq                 = uct_ud_mlx5_iface_event_cq,
    .handle_failure           = uct_ud_iface_handle_failure,
    .set_ep_failed            = uct_ud_mlx5_ep_set_failed
    },
    .async_progress           = uct_ud_mlx5_iface_async_progress,
    .tx_skb                   = uct_ud_mlx5_ep_tx_ctl_skb,
    .ep_free                  = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_mlx5_ep_t)
};

static UCS_CLASS_INIT_FUNC(uct_ud_mlx5_iface_t,
                           uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_ud_mlx5_iface_config_t *config = ucs_derived_of(tl_config,
                                                        uct_ud_mlx5_iface_config_t);
    uct_ib_iface_init_attr_t init_attr = {};
    ucs_status_t status;
    int i;

    ucs_trace_func("");

    init_attr.res_domain_key = UCT_IB_IFACE_NULL_RES_DOMAIN_KEY;
    init_attr.flags          = UCT_IB_CQ_IGNORE_OVERRUN;

    UCS_CLASS_CALL_SUPER_INIT(uct_ud_iface_t, &uct_ud_mlx5_iface_ops,
                              md, worker, params, &config->super, &init_attr);

    uct_ib_iface_set_max_iov(&self->super.super, UCT_IB_MLX5_AM_ZCOPY_MAX_IOV);
    self->super.config.max_inline = UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE);

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_TX], &self->cq[UCT_IB_DIR_TX]);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_RX], &self->cq[UCT_IB_DIR_RX]);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_txwq_init(self->super.super.super.worker, &self->tx.wq,
                                   self->super.qp);
    if (status != UCS_OK) {
        return status;
    }
    self->super.tx.available = self->tx.wq.bb_max;

    status = uct_ib_mlx5_get_rxwq(self->super.qp, &self->rx.wq);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ud_mlx5_iface_common_init(&self->super.super,
                                           &self->mlx5_common, &config->mlx5_common);
    if (status != UCS_OK) {
        return status;
    }

    /* write buffer sizes */
    for (i = 0; i <= self->rx.wq.mask; i++) {
        self->rx.wq.wqes[i].byte_count = htonl(self->super.super.config.rx_payload_offset +
                                               self->super.super.config.seg_size);
    }
    while (self->super.rx.available >= self->super.super.config.rx_max_batch) {
        uct_ud_mlx5_iface_post_recv(self);
    }

    status = uct_ud_iface_complete_init(&self->super);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_ud_mlx5_iface_t)
{
    ucs_trace_func("");
    uct_ud_iface_remove_async_handlers(&self->super);
    uct_ud_enter(&self->super);
    UCT_UD_IFACE_DELETE_EPS(&self->super, uct_ud_mlx5_ep_t);
    ucs_twheel_cleanup(&self->super.async.slow_timer);
    uct_ib_mlx5_txwq_cleanup(&self->tx.wq);
    uct_ud_leave(&self->super);
}

UCS_CLASS_DEFINE(uct_ud_mlx5_iface_t, uct_ud_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_mlx5_iface_t, uct_iface_t);

static ucs_status_t
uct_ud_mlx5_query_resources(uct_md_h md,
                            uct_tl_resource_desc_t **resources_p,
                            unsigned *num_resources_p)
{
    ucs_trace_func("");
    /* TODO take transport overhead into account */
    return uct_ib_device_query_tl_resources(&ucs_derived_of(md, uct_ib_md_t)->dev,
                                            "ud_mlx5", UCT_IB_DEVICE_FLAG_MLX5_PRM,
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_ud_mlx5_tl,
                        uct_ud_mlx5_query_resources,
                        uct_ud_mlx5_iface_t,
                        "ud_mlx5",
                        "UD_MLX5_",
                        uct_ud_mlx5_iface_config_table,
                        uct_ud_mlx5_iface_config_t);
UCT_MD_REGISTER_TL(&uct_ib_mdc, &uct_ud_mlx5_tl);
