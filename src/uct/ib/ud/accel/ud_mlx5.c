/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

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
#include <uct/ib/mlx5/dv/ib_mlx5_dv.h>

#include <uct/ib/ud/base/ud_iface.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_def.h>
#include <uct/ib/ud/base/ud_inl.h>


static ucs_config_field_t uct_ud_mlx5_iface_config_table[] = {
  {"UD_", "", NULL,
   ucs_offsetof(uct_ud_mlx5_iface_config_t, super),
   UCS_CONFIG_TYPE_TABLE(uct_ud_iface_config_table)},

  {UCT_IB_CONFIG_PREFIX, "", NULL,
   ucs_offsetof(uct_ud_mlx5_iface_config_t, mlx5_common),
   UCS_CONFIG_TYPE_TABLE(uct_ib_mlx5_iface_config_table)},

  {"UD_", "", NULL,
   ucs_offsetof(uct_ud_mlx5_iface_config_t, ud_mlx5_common),
   UCS_CONFIG_TYPE_TABLE(uct_ud_mlx5_iface_common_config_table)},

  {NULL}
};

static UCS_F_ALWAYS_INLINE size_t
uct_ud_mlx5_ep_ctrl_av_size(uct_ud_mlx5_ep_t *ep)
{
    return sizeof(struct mlx5_wqe_ctrl_seg) +
           uct_ib_mlx5_wqe_av_size(&ep->peer_address.av);
}

static UCS_F_ALWAYS_INLINE size_t uct_ud_mlx5_max_am_iov()
{
    return ucs_min(UCT_IB_MLX5_AM_ZCOPY_MAX_IOV, UCT_IB_MAX_IOV);
}

static UCS_F_ALWAYS_INLINE size_t uct_ud_mlx5_max_inline()
{
    return UCT_IB_MLX5_AM_MAX_SHORT(UCT_IB_MLX5_AV_FULL_SIZE);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_mlx5_post_send(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep,
                      uint8_t ce_se, struct mlx5_wqe_ctrl_seg *ctrl,
                      size_t wqe_size, uct_ud_neth_t *neth, int max_log_sge)
{
    struct mlx5_wqe_datagram_seg *dgram = (void*)(ctrl + 1);

    ucs_assert(wqe_size <= UCT_IB_MLX5_MAX_SEND_WQE_SIZE);

    UCT_UD_EP_HOOK_CALL_TX(&ep->super, neth);

    uct_ib_mlx5_set_ctrl_seg(ctrl, iface->tx.wq.sw_pi, MLX5_OPCODE_SEND, 0,
                             iface->super.qp->qp_num,
                             uct_ud_mlx5_tx_moderation(iface, ce_se), wqe_size);
    uct_ib_mlx5_set_dgram_seg(dgram, &ep->peer_address.av,
                              ep->peer_address.is_global ?
                              &ep->peer_address.grh_av : NULL,
                              IBV_QPT_UD);

    uct_ib_mlx5_log_tx(&iface->super.super, ctrl, iface->tx.wq.qstart,
                       iface->tx.wq.qend, max_log_sge, NULL, uct_ud_dump_packet);
    iface->super.tx.available -= uct_ib_mlx5_post_send(&iface->tx.wq, ctrl,
                                                       wqe_size);
    ucs_assert((int16_t)iface->tx.wq.bb_max >= iface->super.tx.available);
}

static UCS_F_ALWAYS_INLINE struct mlx5_wqe_ctrl_seg *
uct_ud_mlx5_ep_get_next_wqe(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep,
                            size_t *wqe_size_p, void **next_seg_p)
{
    size_t ctrl_av_size = uct_ud_mlx5_ep_ctrl_av_size(ep);
    struct mlx5_wqe_ctrl_seg *ctrl;
    void *ptr;

    ucs_assert((ctrl_av_size % UCT_IB_MLX5_WQE_SEG_SIZE) == 0);

    ctrl        = iface->tx.wq.curr;
    ptr         = UCS_PTR_BYTE_OFFSET(ctrl, ctrl_av_size);

    *wqe_size_p = ctrl_av_size;
    *next_seg_p = uct_ib_mlx5_txwq_wrap_exact(&iface->tx.wq, ptr);

    return ctrl;
}

static uint16_t uct_ud_mlx5_ep_send_ctl(uct_ud_ep_t *ud_ep, uct_ud_send_skb_t *skb,
                                        const uct_ud_iov_t *iov, uint16_t iovcnt,
                                        int flags, int max_log_sge)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(ud_ep->super.super.iface,
                                                uct_ud_mlx5_iface_t);
    uct_ud_mlx5_ep_t *ep       = ucs_derived_of(ud_ep, uct_ud_mlx5_ep_t);
    struct mlx5_wqe_inl_data_seg *inl;
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_data_seg *dptr;
    uint16_t iov_index;
    size_t wqe_size;
    void *next_seg;
    uint8_t ce_se;
    uint16_t sn;

    /* set WQE flags */
    sn    = iface->tx.wq.sw_pi;
    ce_se = 0;
    if (flags & UCT_UD_IFACE_SEND_CTL_FLAG_SOLICITED) {
        ce_se |= MLX5_WQE_CTRL_SOLICITED;
    }
    if (flags & UCT_UD_IFACE_SEND_CTL_FLAG_SIGNALED) {
        ce_se |= MLX5_WQE_CTRL_CQ_UPDATE;
    }

    /* set skb header as inline (if fits the length) or as data pointer */
    ctrl = uct_ud_mlx5_ep_get_next_wqe(iface, ep, &wqe_size, &next_seg);
    if (skb->len <= uct_ud_mlx5_max_inline()) {
        inl             = next_seg;
        inl->byte_count = htonl(skb->len | MLX5_INLINE_SEG);
        wqe_size       += ucs_align_up_pow2(sizeof(*inl) + skb->len,
                                            UCT_IB_MLX5_WQE_SEG_SIZE);
        uct_ib_mlx5_inline_copy(inl + 1, skb->neth, skb->len, &iface->tx.wq);
    } else {
        ucs_assert(!(flags & UCT_UD_IFACE_SEND_CTL_FLAG_INLINE));
        dptr            = next_seg;
        wqe_size       += sizeof(*dptr);
        uct_ib_mlx5_set_data_seg(dptr, skb->neth, skb->len, skb->lkey);
    }

    /* copy IOV from descriptor to WQE */
    dptr = UCS_PTR_BYTE_OFFSET(ctrl, wqe_size);
    for (iov_index = 0; iov_index < iovcnt; ++iov_index) {
        if (iov[iov_index].length == 0) {
            continue;
        }

        dptr = uct_ib_mlx5_txwq_wrap_any(&iface->tx.wq, dptr);
        uct_ib_mlx5_set_data_seg(dptr, iov[iov_index].buffer,
                                 iov[iov_index].length, iov[iov_index].lkey);
        wqe_size += sizeof(*dptr);
        ++dptr;
    }

    uct_ud_mlx5_post_send(iface, ep, ce_se, ctrl, wqe_size, skb->neth,
                          max_log_sge);
    return sn;
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

static UCS_CLASS_INIT_FUNC(uct_ud_mlx5_ep_t, const uct_ep_params_t *params)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(params->iface,
                                                uct_ud_mlx5_iface_t);
    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_ud_ep_t, &iface->super, params);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_mlx5_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_ud_mlx5_ep_t, uct_ud_ep_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_mlx5_ep_t, uct_ep_t,
                                 const uct_ep_params_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_mlx5_ep_t, uct_ep_t);


/*
 * Generic inline+iov post-send function
 * The caller should check that header size + sg list would not exceed WQE size.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_ud_mlx5_ep_inline_iov_post(uct_ep_h tl_ep, uint8_t am_id,
                               /* inl. header */ const void *header, size_t header_size,
                               /* inl. data */   const void *data, size_t data_size,
                               /* iov data */    const uct_iov_t *iov, size_t iovcnt,
                               uint32_t packet_flags, uct_completion_t *comp,
                               unsigned stat_ops_counter, unsigned stat_bytes_counter,
                               const char *func_name)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    uct_ud_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    struct mlx5_wqe_inl_data_seg *inl;
    struct mlx5_wqe_ctrl_seg *ctrl;
    size_t inline_size, wqe_size;
    void *next_seg, *wqe_data;
    uct_ud_send_skb_t *skb;
    ucs_status_t status;
    uct_ud_neth_t *neth;

    UCT_CHECK_AM_ID(am_id);
    UCT_UD_CHECK_ZCOPY_LENGTH(&iface->super, header_size + data_size,
                              uct_iov_total_length(iov, iovcnt));
    UCT_CHECK_IOV_SIZE(iovcnt, uct_ud_mlx5_max_am_iov(), func_name);

    uct_ud_enter(&iface->super);

    skb = uct_ud_ep_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        status = UCS_ERR_NO_RESOURCE;
        goto out;
    }

    ctrl            = uct_ud_mlx5_ep_get_next_wqe(iface, ep, &wqe_size,
                                                  &next_seg);
    inl             = next_seg;
    inline_size     = sizeof(*neth) + header_size + data_size;
    inl->byte_count = htonl(inline_size | MLX5_INLINE_SEG);
    wqe_size       += sizeof(*inl) + inline_size;
    skb->len        = inline_size;

    /* set network header */
    neth              = (void*)(inl + 1);
    neth->packet_type = (am_id << UCT_UD_PACKET_AM_ID_SHIFT) |
                        ep->super.dest_ep_id |
                        packet_flags;
    uct_ud_neth_init_data(&ep->super, neth);
    if (!(packet_flags & UCT_UD_PACKET_FLAG_ACK_REQ)) {
        /* check for ACK_REQ, if not already enabled by packet_flags */
        neth->packet_type |= uct_ud_ep_req_ack(&ep->super) << UCT_UD_PACKET_ACK_REQ_SHIFT;
    }

    /* copy inline "header", assume it fits to one BB so we won't have to check
     * for QP wrap-around. This is either the "put" header or the 64-bit
     * am_short header, not the am_zcopy header.
     */
    wqe_data = UCS_PTR_BYTE_OFFSET(neth + 1, header_size);
    ucs_assert(wqe_data <= iface->tx.wq.qend);
    memcpy(neth + 1, header, header_size);

    /* copy inline "data" */
    uct_ib_mlx5_inline_copy(wqe_data, data, data_size, &iface->tx.wq);

    /* set iov to dptr */
    if (iovcnt > 0) {
        wqe_size  = ucs_align_up_pow2(wqe_size, UCT_IB_MLX5_WQE_SEG_SIZE);
        wqe_size += uct_ib_mlx5_set_data_seg_iov(&iface->tx.wq,
                                                 UCS_PTR_BYTE_OFFSET(ctrl, wqe_size),
                                                 iov, iovcnt);
    }

    uct_ud_mlx5_post_send(iface, ep, 0, ctrl, wqe_size, neth,
                          UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super.super));

    memcpy(skb->neth, neth, sizeof(*neth) + header_size);
    memcpy(UCS_PTR_BYTE_OFFSET(skb->neth + 1, header_size), data, data_size);

    if (iovcnt > 0) {
        uct_ud_skb_set_zcopy_desc(skb, iov, iovcnt, comp);
        status = UCS_INPROGRESS;
    } else {
        status = UCS_OK;
    }

    uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
    uct_ud_ep_ctl_op_del(&ep->super, UCT_UD_EP_OP_ACK|UCT_UD_EP_OP_ACK_REQ);

    UCS_STATS_UPDATE_COUNTER(ep->super.super.stats, stat_ops_counter, 1);
    UCS_STATS_UPDATE_COUNTER(ep->super.super.stats, stat_bytes_counter,
                             header_size + data_size +
                             uct_iov_total_length(iov, iovcnt));
out:
    uct_ud_leave(&iface->super);
    return status;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_ud_mlx5_ep_short_common(uct_ep_h tl_ep, uint8_t am_id,
                            /* inline header */ const void *header, size_t header_size,
                            /* inline data */   const void *data, size_t data_size,
                            uint32_t packet_flags, unsigned stat_ops_counter,
                            const char *func_name)
{
    UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + header_size + data_size, 0,
                     uct_ud_mlx5_max_inline(), func_name);
    return uct_ud_mlx5_ep_inline_iov_post(tl_ep, am_id,
                                          header, header_size,
                                          data, data_size,
                                          /* iov */ NULL, 0,
                                          packet_flags,
                                          /* completion */ NULL,
                                          stat_ops_counter,
                                          UCT_EP_STAT_BYTES_SHORT,
                                          func_name);
}

static ucs_status_t
uct_ud_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                        const void *buffer, unsigned length)
{
    return uct_ud_mlx5_ep_short_common(tl_ep, id,
                                       /* inline header */ &hdr, sizeof(hdr),
                                       /* inline data */  buffer, length,
                                       /* packet flags */ UCT_UD_PACKET_FLAG_AM,
                                       UCT_EP_STAT_AM,
                                       "uct_ud_mlx5_ep_am_short");
}

static ssize_t uct_ud_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                       uct_pack_callback_t pack_cb, void *arg,
                                       unsigned flags)
{
    uct_ud_mlx5_ep_t *ep       = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_data_seg *dptr;
    uct_ud_send_skb_t *skb;
    ucs_status_t status;
    size_t wqe_size;
    void *next_seg;
    size_t length;

    uct_ud_enter(&iface->super);

    status = uct_ud_am_skb_common(&iface->super, &ep->super, id, &skb);
    if (status != UCS_OK) {
        uct_ud_leave(&iface->super);
        return status;
    }

    length = uct_ud_skb_bcopy(skb, pack_cb, arg);
    UCT_UD_CHECK_BCOPY_LENGTH(&iface->super, length);

    ctrl = uct_ud_mlx5_ep_get_next_wqe(iface, ep, &wqe_size, &next_seg);
    dptr = next_seg;
    uct_ib_mlx5_set_data_seg(dptr, skb->neth, skb->len, skb->lkey);
    uct_ud_mlx5_post_send(iface, ep, 0, ctrl, wqe_size + sizeof(*dptr),
                          skb->neth, INT_MAX);

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
    char dummy = 0 ; /* pass dummy pointer to 0-length header to avoid compiler
                        warnings */

    UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + header_length, 0,
                     UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(UCT_IB_MLX5_AV_FULL_SIZE),
                     "am_zcopy header");
    return uct_ud_mlx5_ep_inline_iov_post(tl_ep, id,
                                          /* inl. header */  &dummy, 0,
                                          /* inl. data */    header, header_length,
                                          /* iov */          iov, iovcnt,
                                          /* packet flags */ UCT_UD_PACKET_FLAG_AM |
                                                             UCT_UD_PACKET_FLAG_ACK_REQ,
                                          /* completion */   comp,
                                          UCT_EP_STAT_AM, UCT_EP_STAT_BYTES_ZCOPY,
                                          "uct_ud_mlx5_ep_am_zcopy");
}

static ucs_status_t
uct_ud_mlx5_ep_put_short(uct_ep_h tl_ep, const void *buffer, unsigned length,
                         uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_ud_put_hdr_t puth = { .rva = remote_addr };
    return uct_ud_mlx5_ep_short_common(tl_ep, 0,
                                       /* inl. header */  &puth, sizeof(puth),
                                       /* inl. data */    buffer, length,
                                       /* packet flags */ UCT_UD_PACKET_FLAG_PUT,
                                       UCT_EP_STAT_PUT,
                                       "uct_ud_mlx5_ep_put_short");
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
    ptrdiff_t rx_hdr_offset;

    ci            = iface->rx.wq.cq_wqe_counter & iface->rx.wq.mask;
    packet        = (void *)be64toh(iface->rx.wq.wqes[ci].addr);
    ucs_prefetch(UCS_PTR_BYTE_OFFSET(packet, UCT_IB_GRH_LEN));
    rx_hdr_offset = iface->super.super.config.rx_hdr_offset;
    desc          = UCS_PTR_BYTE_OFFSET(packet, -rx_hdr_offset);

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

    if (!uct_ud_iface_check_grh(&iface->super, packet,
                                uct_ib_mlx5_cqe_is_grh_present(cqe))) {
        ucs_mpool_put_inline(desc);
        goto out;
    }

    uct_ib_mlx5_log_rx(&iface->super.super, cqe, packet, uct_ud_dump_packet);
    /* coverity[tainted_data] */
    uct_ud_ep_process_rx(&iface->super,
                         (uct_ud_neth_t *)UCS_PTR_BYTE_OFFSET(packet, UCT_IB_GRH_LEN),
                         len - UCT_IB_GRH_LEN,
                         (uct_ud_recv_skb_t *)ucs_unaligned_ptr(desc), is_async);
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
uct_ud_mlx5_iface_poll_tx(uct_ud_mlx5_iface_t *iface, int is_async)
{
    struct mlx5_cqe64 *cqe;
    uint16_t hw_ci;

    cqe = uct_ib_mlx5_poll_cq(&iface->super.super, &iface->cq[UCT_IB_DIR_TX]);
    if (cqe == NULL) {
        return 0;
    }

    ucs_memory_cpu_load_fence();

    uct_ib_mlx5_log_cqe(cqe);
    hw_ci                     = ntohs(cqe->wqe_counter);
    iface->super.tx.available = uct_ib_mlx5_txwq_update_bb(&iface->tx.wq, hw_ci);

    uct_ud_iface_send_completion(&iface->super, hw_ci, is_async);

    return 1;
}

static unsigned uct_ud_mlx5_iface_progress(uct_iface_h tl_iface)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_mlx5_iface_t);
    ucs_status_t status;
    unsigned n, count = 0;

    uct_ud_enter(&iface->super);
    uct_ud_iface_dispatch_async_comps(&iface->super);

    status = uct_ud_iface_dispatch_pending_rx(&iface->super);
    if (ucs_likely(status == UCS_OK)) {
        do {
            n = uct_ud_mlx5_iface_poll_rx(iface, 0);
            count += n;
        } while ((n > 0) && (count < iface->super.super.config.rx_max_poll));
    }

    count += uct_ud_mlx5_iface_poll_tx(iface, 0);
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
    } while ((n > 0) && (count < iface->super.rx.async_max_poll));

    count += uct_ud_mlx5_iface_poll_tx(iface, 1);

    uct_ud_iface_progress_pending(&iface->super, 1);

    return count;
}

static ucs_status_t
uct_ud_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    ucs_status_t status;

    ucs_trace_func("");

    status = uct_ud_iface_query(iface, iface_attr, uct_ud_mlx5_max_am_iov(),
                                UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(UCT_IB_MLX5_AV_FULL_SIZE)
                                - sizeof(uct_ud_neth_t));
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->overhead = 80e-9; /* Software overhead */

    return UCS_OK;
}

static ucs_status_t
uct_ud_mlx5_iface_unpack_peer_address(uct_ud_iface_t *ud_iface,
                                      const uct_ib_address_t *ib_addr,
                                      const uct_ud_iface_addr_t *if_addr,
                                      int path_index, void *address_p)
{
    uct_ud_mlx5_iface_t *iface                  =
        ucs_derived_of(ud_iface, uct_ud_mlx5_iface_t);
    uct_ud_mlx5_ep_peer_address_t *peer_address =
        (uct_ud_mlx5_ep_peer_address_t*)address_p;
    ucs_status_t status;
    int is_global;

    memset(peer_address, 0, sizeof(*peer_address));

    status = uct_ud_mlx5_iface_get_av(&ud_iface->super, &iface->ud_mlx5_common,
                                      ib_addr, path_index, &peer_address->av,
                                      &peer_address->grh_av, &is_global);
    if (status != UCS_OK) {
        return status;
    }

    peer_address->is_global   = is_global;
    peer_address->av.dqp_dct |= htonl(uct_ib_unpack_uint24(if_addr->qp_num));

    return UCS_OK;
}

static void *uct_ud_mlx5_ep_get_peer_address(uct_ud_ep_t *ud_ep)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(ud_ep, uct_ud_mlx5_ep_t);
    return &ep->peer_address;
}

static size_t uct_ud_mlx5_get_peer_address_length()
{
    return sizeof(uct_ud_mlx5_ep_peer_address_t);
}

static const char*
uct_ud_mlx5_iface_peer_address_str(const uct_ud_iface_t *iface,
                                   const void *address,
                                   char *str, size_t max_size)
{
    const uct_ud_mlx5_ep_peer_address_t *peer_address =
        (const uct_ud_mlx5_ep_peer_address_t*)address;

    uct_ib_mlx5_av_dump(str, max_size,
                        &peer_address->av, &peer_address->grh_av,
                        uct_ib_iface_is_roce((uct_ib_iface_t*)&iface->super));
    return str;
}

static ucs_status_t
uct_ud_mlx5_ep_create(const uct_ep_params_t* params, uct_ep_h *ep_p)
{
    if (ucs_test_all_flags(params->field_mask, UCT_EP_PARAM_FIELD_DEV_ADDR |
                                               UCT_EP_PARAM_FIELD_IFACE_ADDR)) {
        return uct_ud_ep_create_connected_common(params, ep_p);
    }

    return uct_ud_mlx5_ep_t_new(params, ep_p);
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

static void uct_ud_mlx5_iface_event_cq(uct_ib_iface_t *ib_iface,
                                       uct_ib_dir_t dir)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_ud_mlx5_iface_t);

    iface->cq[dir].cq_sn++;
}

static ucs_status_t uct_ud_mlx5_iface_create_qp(uct_ib_iface_t *ib_iface,
                                                uct_ib_qp_attr_t *ib_attr,
                                                struct ibv_qp **qp_p)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_ud_mlx5_iface_t);
    uct_ib_mlx5_qp_t *qp = &iface->tx.wq.super;
    uct_ib_mlx5_qp_attr_t attr = {};
    ucs_status_t status;

    attr.super     = *ib_attr;
    attr.mmio_mode = UCT_IB_MLX5_MMIO_MODE_LAST;

    status = uct_ib_mlx5_iface_create_qp(ib_iface, qp, &attr);
    if (status != UCS_OK) {
        return status;
    }

    *qp_p = qp->verbs.qp;
    return status;
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_ud_mlx5_iface_t)(uct_iface_t*);

static void uct_ud_mlx5_iface_handle_failure(uct_ib_iface_t *ib_iface, void *arg,
                                             ucs_status_t status)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(ib_iface, uct_ud_mlx5_iface_t);

    ucs_assert(status != UCS_ERR_ENDPOINT_TIMEOUT);
    /* Local side failure - treat as fatal */
    uct_ib_mlx5_completion_with_err(ib_iface, arg, &iface->tx.wq,
                                    UCS_LOG_LEVEL_FATAL);
}

static uct_ud_iface_ops_t uct_ud_mlx5_iface_ops = {
    {
    {
    .ep_put_short             = uct_ud_mlx5_ep_put_short,
    .ep_am_short              = uct_ud_mlx5_ep_am_short,
    .ep_am_short_iov          = uct_base_ep_am_short_iov,
    .ep_am_bcopy              = uct_ud_mlx5_ep_am_bcopy,
    .ep_am_zcopy              = uct_ud_mlx5_ep_am_zcopy,
    .ep_pending_add           = uct_ud_ep_pending_add,
    .ep_pending_purge         = uct_ud_ep_pending_purge,
    .ep_flush                 = uct_ud_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_check                 = uct_ud_ep_check,
    .ep_create                = uct_ud_mlx5_ep_create,
    .ep_destroy               = uct_ud_ep_disconnect ,
    .ep_get_address           = uct_ud_ep_get_address,
    .ep_connect_to_ep         = uct_ud_ep_connect_to_ep,
    .iface_flush              = uct_ud_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_ud_iface_progress_enable,
    .iface_progress_disable   = uct_ud_iface_progress_disable,
    .iface_progress           = uct_ud_mlx5_iface_progress,
    .iface_event_fd_get       = (uct_iface_event_fd_get_func_t)
                                ucs_empty_function_return_unsupported,
    .iface_event_arm          = uct_ud_iface_event_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_mlx5_iface_t),
    .iface_query              = uct_ud_mlx5_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_get_address        = uct_ud_iface_get_address,
    .iface_is_reachable       = uct_ib_iface_is_reachable
    },
    .create_cq                = uct_ib_mlx5_create_cq,
    .arm_cq                   = uct_ud_mlx5_iface_arm_cq,
    .event_cq                 = uct_ud_mlx5_iface_event_cq,
    .handle_failure           = uct_ud_mlx5_iface_handle_failure,
    },
    .async_progress           = uct_ud_mlx5_iface_async_progress,
    .send_ctl                 = uct_ud_mlx5_ep_send_ctl,
    .ep_free                  = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_mlx5_ep_t),
    .create_qp                = uct_ud_mlx5_iface_create_qp,
    .unpack_peer_address      = uct_ud_mlx5_iface_unpack_peer_address,
    .ep_get_peer_address      = uct_ud_mlx5_ep_get_peer_address,
    .get_peer_address_length  = uct_ud_mlx5_get_peer_address_length,
    .peer_address_str         = uct_ud_mlx5_iface_peer_address_str
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

    init_attr.flags                 = UCT_IB_CQ_IGNORE_OVERRUN;
    init_attr.cq_len[UCT_IB_DIR_TX] = config->super.super.tx.queue_len * UCT_IB_MLX5_MAX_BB;
    init_attr.cq_len[UCT_IB_DIR_RX] = config->super.super.rx.queue_len;

    self->tx.wq.super.type = UCT_IB_MLX5_OBJ_TYPE_LAST;

    UCS_CLASS_CALL_SUPER_INIT(uct_ud_iface_t, &uct_ud_mlx5_iface_ops,
                              md, worker, params, &config->super, &init_attr);

    self->super.config.max_inline = uct_ud_mlx5_max_inline();

    status = uct_ib_mlx5_iface_select_sl(&self->super.super,
                                         &config->mlx5_common,
                                         &config->super.super);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_TX], &self->cq[UCT_IB_DIR_TX]);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.cq[UCT_IB_DIR_RX], &self->cq[UCT_IB_DIR_RX]);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_txwq_init(self->super.super.super.worker,
                                   config->mlx5_common.mmio_mode, &self->tx.wq,
                                   self->super.qp);
    if (status != UCS_OK) {
        return status;
    }

    self->super.tx.available = self->tx.wq.bb_max;
    ucs_assert(init_attr.cq_len[UCT_IB_DIR_TX] >= self->tx.wq.bb_max);

    status = uct_ib_mlx5_get_rxwq(self->super.qp, &self->rx.wq);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(init_attr.cq_len[UCT_IB_DIR_RX] > self->rx.wq.mask);

    status = uct_ud_mlx5_iface_common_init(&self->super.super,
                                           &self->ud_mlx5_common,
                                           &config->ud_mlx5_common);
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

    return uct_ud_iface_complete_init(&self->super);
}


static UCS_CLASS_CLEANUP_FUNC(uct_ud_mlx5_iface_t)
{
    ucs_trace_func("");
    uct_ud_iface_remove_async_handlers(&self->super);
    uct_ud_enter(&self->super);
    uct_ib_mlx5_qp_mmio_cleanup(&self->tx.wq.super, self->tx.wq.reg);
    uct_ud_leave(&self->super);
}

UCS_CLASS_DEFINE(uct_ud_mlx5_iface_t, uct_ud_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_mlx5_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_mlx5_iface_t, uct_iface_t);

static ucs_status_t
uct_ud_mlx5_query_tl_devices(uct_md_h md,
                             uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);
    return uct_ib_device_query_ports(&ib_md->dev, UCT_IB_DEVICE_FLAG_MLX5_PRM,
                                     tl_devices_p, num_tl_devices_p);
}

UCT_TL_DEFINE(&uct_ib_component, ud_mlx5, uct_ud_mlx5_query_tl_devices,
              uct_ud_mlx5_iface_t, "UD_MLX5_", uct_ud_mlx5_iface_config_table,
              uct_ud_mlx5_iface_config_t);
