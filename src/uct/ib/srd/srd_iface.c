/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "srd_iface.h"
#include "srd_ep.h"
#include "srd_inl.h"
#include "srd_def.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/ib/base/ib_log.h>
#include <uct/base/uct_md.h>
#include <uct/base/uct_log.h>
#include <uct/ib/efa/ib_efa.h>
#include <ucs/arch/cpu.h>
#include <ucs/sys/math.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/queue.h>

#include <sys/poll.h>
#include <string.h>
#include <arpa/inet.h> /* For htonl */

#define UCT_SRD_IFACE_CEP_CONN_SN_MAX ((uct_srd_ep_conn_sn_t)-1)

#ifdef ENABLE_STATS
static ucs_stats_class_t uct_srd_iface_stats_class = {
    .name = "srd_iface",
    .num_counters = UCT_SRD_IFACE_STAT_LAST,
    .counter_names = {
        [UCT_SRD_IFACE_STAT_RX_DROP] = "rx_drop"
    }
};
#endif


static
ucs_status_t uct_srd_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                 const void *buffer, unsigned length)
{
    uct_srd_ep_t *ep = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_skb_t *skb;
    uct_srd_am_short_hdr_t *am_hdr;
    ucs_status_t status;

    UCT_CHECK_LENGTH(sizeof(uct_srd_neth_t) + sizeof(hdr) + length,
                     0, iface->config.max_inline, "am_short");

    uct_srd_enter(iface);

    status = uct_srd_am_skb_common(iface, ep, id, &skb);
    if (status != UCS_OK) {
        uct_srd_leave(iface);
        return status;
    }

    am_hdr      = (uct_srd_am_short_hdr_t *)(skb->neth+1);
    am_hdr->hdr = hdr;
    skb->len    = sizeof(uct_srd_neth_t) + sizeof(*am_hdr);

    iface->tx.sge[0].length = skb->len;
    iface->tx.sge[0].addr   = (uintptr_t)skb->neth;

    uct_srd_ep_tx_inlv(iface, ep, skb, buffer, length);

    uct_srd_iface_complete_tx(iface, ep, skb);
    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, sizeof(hdr) + length);
    uct_srd_leave(iface);
    return UCS_OK;
}

static ucs_status_t uct_srd_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                            const uct_iov_t *iov, size_t iovcnt)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_skb_t *skb;
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, (size_t)iface->config.max_send_sge,
                       "uct_srd_ep_am_short_iov");
    UCT_CHECK_LENGTH(sizeof(uct_srd_neth_t) + uct_iov_total_length(iov, iovcnt), 0,
                     iface->config.max_inline, "am_short");

    uct_srd_enter(iface);

    status = uct_srd_am_skb_common(iface, ep, id, &skb);
    if (status != UCS_OK) {
        uct_srd_leave(iface);
        return status;
    }

    iface->tx.sge[0].length  = skb->len = sizeof(uct_srd_neth_t);
    iface->tx.sge[0].addr    = (uintptr_t)skb->neth;
    iface->tx.wr_inl.wr_id   = (uintptr_t)skb;
    iface->tx.wr_inl.num_sge =
        uct_ib_verbs_sge_fill_iov(iface->tx.sge + 1, iov, iovcnt) + 1;

    skb->neth->psn = ep->tx.psn++;
    uct_srd_post_send(iface, ep, &iface->tx.wr_inl, IBV_SEND_INLINE,
                      iface->tx.wr_inl.num_sge);

    uct_srd_iface_complete_tx(iface, ep, skb);
    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, uct_iov_total_length(iov, iovcnt));
    uct_srd_leave(iface);

    return UCS_OK;
}

static ssize_t uct_srd_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                   uct_pack_callback_t pack_cb, void *arg,
                                   unsigned flags)
{
    uct_srd_ep_t *ep = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_skb_t *skb;
    ucs_status_t status;
    size_t length;

    uct_srd_enter(iface);

    status = uct_srd_am_skb_common(iface, ep, id, &skb);
    if (status != UCS_OK) {
        uct_srd_leave(iface);
        return status;
    }

    length = uct_srd_skb_bcopy(skb, pack_cb, arg);
    UCT_SRD_CHECK_BCOPY_LENGTH(iface, length);

    ucs_assert(iface->tx.wr_skb.num_sge == 1);
    uct_srd_ep_tx_skb(iface, ep, skb, 0, INT_MAX);
    uct_srd_iface_complete_tx(iface, ep, skb);
    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, length);
    uct_srd_leave(iface);
    return length;
}

static ucs_status_t
uct_srd_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                    unsigned header_length, const uct_iov_t *iov,
                    size_t iovcnt, unsigned flags, uct_completion_t *comp)
{
    uct_srd_ep_t *ep = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_skb_t *skb;
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, (size_t)iface->config.max_send_sge,
                       "uct_srd_ep_am_zcopy");

    UCT_CHECK_LENGTH(sizeof(uct_srd_neth_t) + sizeof(uct_srd_zcopy_desc_t) + header_length,
                     0, iface->super.config.seg_size, "am_zcopy header");

    UCT_SRD_CHECK_ZCOPY_LENGTH(iface, header_length,
                               uct_iov_total_length(iov, iovcnt));

    uct_srd_enter(iface);

    status = uct_srd_am_skb_common(iface, ep, id, &skb);
    if (status != UCS_OK) {
        uct_srd_leave(iface);
        return status;
    }
    memcpy(skb->neth + 1, header, header_length);
    skb->len = sizeof(uct_srd_neth_t) + header_length;

    iface->tx.wr_skb.num_sge = uct_ib_verbs_sge_fill_iov(iface->tx.sge + 1,
                                                         iov, iovcnt) + 1;
    uct_srd_ep_tx_skb(iface, ep, skb, 0,
                      UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super));
    iface->tx.wr_skb.num_sge = 1;

    uct_srd_skb_set_zcopy_desc(skb, iov, iovcnt, comp);
    uct_srd_iface_complete_tx(iface, ep, skb);
    UCT_TL_EP_STAT_OP(&ep->super, AM, ZCOPY, header_length +
                      uct_iov_total_length(iov, iovcnt));
    uct_srd_leave(iface);
    return UCS_INPROGRESS;
}

static void uct_srd_ep_send_completion(uct_srd_send_skb_t *skb)
{
    uct_srd_send_skb_t *q_skb;
    ucs_queue_iter_t iter;

    ucs_assert(!(skb->flags & UCT_SRD_SEND_SKB_FLAG_INVALID));

    /* If the completed skb is still in ep outstanding queue
     * remove it from the queue and call the user callback.
     * ep purge might have already removed the completed skb */
    ucs_queue_for_each_safe(q_skb, iter, &skb->ep->tx.outstanding_q, out_queue) {
        if (q_skb == skb) {
            if (ucs_unlikely((skb->flags & UCT_SRD_SEND_SKB_FLAG_COMP))) {
                uct_invoke_completion(uct_srd_comp_desc(skb)->comp, UCS_OK);
            }
            ucs_queue_del_iter(&skb->ep->tx.outstanding_q, iter);
            break;
        }
    }

    /* while queue head is flush skb, remove it and call user callback */
    ucs_queue_for_each_extract(q_skb, &skb->ep->tx.outstanding_q, out_queue,
                               q_skb->flags & UCT_SRD_SEND_SKB_FLAG_FLUSH) {
        /* outstanding flush must have completion callback. */
        ucs_assert(!(q_skb->flags & UCT_SRD_SEND_SKB_FLAG_COMP));
        uct_invoke_completion(uct_srd_comp_desc(q_skb)->comp, UCS_OK);
    }

    uct_srd_skb_release(skb, 0);
}

static void uct_srd_iface_send_completion(uct_srd_iface_t *iface,
                                          uct_srd_send_skb_t *skb)
{
    uct_srd_ep_send_completion(skb);
}

static UCS_F_NOINLINE void
uct_srd_iface_post_recv_always(uct_srd_iface_t *iface, int max)
{
    struct ibv_recv_wr *bad_wr;
    uct_ib_recv_wr_t *wrs;
    unsigned count;
    int ret;

    wrs  = ucs_alloca(sizeof *wrs  * max);

    count = uct_ib_iface_prepare_rx_wrs(&iface->super, &iface->rx.mp, wrs, max);
    if (count == 0) {
        return;
    }

    ret = ibv_post_recv(iface->qp, &wrs[0].ibwr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_recv() returned %d: %m", ret);
    }
    iface->rx.available -= count;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_iface_post_recv(uct_srd_iface_t *iface)
{
    unsigned batch = iface->super.config.rx_max_batch;

    if (iface->rx.available < batch)
        return;

    uct_srd_iface_post_recv_always(iface, batch);
}

static UCS_F_ALWAYS_INLINE unsigned
uct_srd_iface_poll_rx(uct_srd_iface_t *iface)
{
    unsigned num_wcs = iface->super.config.rx_max_poll;
    struct ibv_wc wc[num_wcs];
    ucs_status_t status;
    void *packet;
    int i;

    status = uct_ib_poll_cq(iface->super.cq[UCT_IB_DIR_RX], &num_wcs, wc);
    if (status != UCS_OK) {
        num_wcs = 0;
        goto out;
    }

    UCT_IB_IFACE_VERBS_FOREACH_RXWQE(&iface->super, i, packet, wc, num_wcs) {
        uct_ib_log_recv_completion(&iface->super, &wc[i], packet,
                                   wc[i].byte_len, uct_srd_dump_packet);
        uct_srd_ep_process_rx(iface, (uct_srd_neth_t *)packet, wc[i].byte_len,
                              (uct_srd_recv_skb_t *)wc[i].wr_id);
    }
    iface->rx.available += num_wcs;
out:
    uct_srd_iface_post_recv(iface);
    return num_wcs;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_srd_iface_poll_tx(uct_srd_iface_t *iface)
{
    unsigned num_wcs = iface->super.config.tx_max_poll;
    struct ibv_wc wc[num_wcs];
    ucs_status_t status;
    int i;

    status = uct_ib_poll_cq(iface->super.cq[UCT_IB_DIR_TX], &num_wcs, wc);
    if (status != UCS_OK) {
        num_wcs = 0;
    }

    for (i = 0; i < num_wcs; i++) {
        if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
            UCT_IB_IFACE_VERBS_COMPLETION_ERR("send", &iface->super, i, wc);
            continue;
        }

        uct_srd_iface_send_completion(iface, (uct_srd_send_skb_t *)wc[i].wr_id);
    }

    iface->tx.available += num_wcs;
    return num_wcs;
}

static void uct_srd_iface_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_srd_iface_t *iface = ucs_derived_of(tl_iface, uct_srd_iface_t);

    uct_srd_enter(iface);

    if (flags & UCT_PROGRESS_RECV) {
        iface->rx.available += iface->rx.quota;
        iface->rx.quota      = 0;
        /* let progress post the missing receives */
    }

    uct_srd_leave(iface);

    uct_base_iface_progress_enable(tl_iface, flags);
}

static unsigned uct_srd_iface_progress(uct_iface_h tl_iface)
{
    uct_srd_iface_t *iface = ucs_derived_of(tl_iface, uct_srd_iface_t);
    unsigned count;

    uct_srd_enter(iface);

    count = uct_srd_iface_poll_rx(iface);
    if (count == 0) {
        count = uct_srd_iface_poll_tx(iface);
    }

    uct_srd_iface_progress_pending(iface);
    uct_srd_leave(iface);

    return count;
}

/* Used for am zcopy only */
static ucs_status_t uct_srd_qp_max_send_sge(uct_srd_iface_t *iface,
                                            size_t *max_send_sge)
{
    uint32_t max_sge;
    ucs_status_t status;

    status = uct_ib_qp_max_send_sge(iface->qp, &max_sge);
    if (status != UCS_OK) {
        return status;
    }

    /* need to reserve 1 iov for am zcopy header */
    ucs_assert_always(max_sge > 1);

    *max_send_sge = ucs_min(max_sge - 1, UCT_IB_MAX_IOV);

    return UCS_OK;
}

static inline size_t uct_srd_ep_get_peer_address_length()
{
    return sizeof(uct_srd_ep_peer_address_t);
}

ucs_status_t
uct_srd_iface_unpack_peer_address(uct_srd_iface_t *iface,
                                  const uct_ib_address_t *ib_addr,
                                  const uct_srd_iface_addr_t *if_addr,
                                  int path_index, void *address_p)
{
    uct_ib_iface_t *ib_iface                = &iface->super;
    uct_srd_ep_peer_address_t *peer_address =
        (uct_srd_ep_peer_address_t*)address_p;
    struct ibv_ah_attr ah_attr;
    enum ibv_mtu path_mtu;
    ucs_status_t status;

    memset(peer_address, 0, sizeof(*peer_address));

    uct_ib_iface_fill_ah_attr_from_addr(ib_iface, ib_addr, path_index,
                                        &ah_attr, &path_mtu);
    status = uct_ib_iface_create_ah(ib_iface, &ah_attr, "SRD connect",
                                    &peer_address->ah);
    if (status != UCS_OK) {
        return status;
    }

    peer_address->dest_qpn = uct_ib_unpack_uint24(if_addr->qp_num);

    return UCS_OK;
}

static const char*
uct_srd_iface_peer_address_str(const uct_srd_iface_t *iface,
                               const void *address,
                               char *str, size_t max_size)
{
    const uct_srd_ep_peer_address_t *peer_address =
        (const uct_srd_ep_peer_address_t*)address;

    ucs_snprintf_zero(str, max_size, "ah=%p dest_qpn=%u",
                      peer_address->ah, peer_address->dest_qpn);
    return str;
}

static void *
uct_srd_iface_cep_get_peer_address(uct_srd_iface_t *iface,
                                   const uct_ib_address_t *ib_addr,
                                   const uct_srd_iface_addr_t *if_addr,
                                   int path_index, void *address_p)
{
    ucs_status_t status = uct_srd_iface_unpack_peer_address(iface, ib_addr,
                                                            if_addr, path_index,
                                                            address_p);

    if (status != UCS_OK) {
        ucs_fatal("iface %p: failed to get peer address", iface);
    }

    return address_p;
}

static UCS_F_ALWAYS_INLINE ucs_conn_match_queue_type_t
uct_srd_iface_cep_ep_queue_type(uct_srd_ep_t *ep)
{
    return (ep->flags & UCT_SRD_EP_FLAG_PRIVATE) ?
           UCS_CONN_MATCH_QUEUE_UNEXP :
           UCS_CONN_MATCH_QUEUE_EXP;
}

uct_srd_ep_conn_sn_t
uct_srd_iface_cep_get_conn_sn(uct_srd_iface_t *iface,
                             const uct_ib_address_t *ib_addr,
                             const uct_srd_iface_addr_t *if_addr,
                             int path_index)
{
    void *peer_address = ucs_alloca(iface->conn_match_ctx.address_length);
    return (uct_srd_ep_conn_sn_t)
           ucs_conn_match_get_next_sn(&iface->conn_match_ctx,
                                      uct_srd_iface_cep_get_peer_address(
                                          iface, ib_addr, if_addr, path_index,
                                          peer_address));
}

void uct_srd_iface_cep_insert_ep(uct_srd_iface_t *iface,
                                 const uct_ib_address_t *ib_addr,
                                 const uct_srd_iface_addr_t *if_addr,
                                 int path_index, uct_srd_ep_conn_sn_t conn_sn,
                                 uct_srd_ep_t *ep)
{
    ucs_conn_match_queue_type_t queue_type;
    void *peer_address;

    queue_type   = uct_srd_iface_cep_ep_queue_type(ep);
    peer_address = ucs_alloca(iface->conn_match_ctx.address_length);
    uct_srd_iface_cep_get_peer_address(iface, ib_addr, if_addr, path_index,
                                       peer_address);

    ucs_assert(!(ep->flags & UCT_SRD_EP_FLAG_ON_CEP));
    ucs_conn_match_insert(&iface->conn_match_ctx, peer_address,
                          conn_sn, &ep->conn_match, queue_type);
    ep->flags |= UCT_SRD_EP_FLAG_ON_CEP;
}

uct_srd_ep_t *uct_srd_iface_cep_get_ep(uct_srd_iface_t *iface,
                                       const uct_ib_address_t *ib_addr,
                                       const uct_srd_iface_addr_t *if_addr,
                                       int path_index,
                                       uct_srd_ep_conn_sn_t conn_sn,
                                       int is_private)
{
    uct_srd_ep_t *ep                        = NULL;
    ucs_conn_match_queue_type_t queue_type = is_private ?
                                             UCS_CONN_MATCH_QUEUE_UNEXP :
                                             UCS_CONN_MATCH_QUEUE_ANY;
    ucs_conn_match_elem_t *conn_match;
    void *peer_address;

    peer_address = ucs_alloca(iface->conn_match_ctx.address_length);
    uct_srd_iface_cep_get_peer_address(iface, ib_addr, if_addr,
                                       path_index, peer_address);

    conn_match = ucs_conn_match_get_elem(&iface->conn_match_ctx, peer_address,
                                         conn_sn, queue_type, is_private);
    if (conn_match == NULL) {
        return NULL;
    }

    ep = ucs_container_of(conn_match, uct_srd_ep_t, conn_match);
    ucs_assert(ep->flags & UCT_SRD_EP_FLAG_ON_CEP);

    if (is_private) {
        ep->flags &= ~UCT_SRD_EP_FLAG_ON_CEP;
    }

    return ep;
}

void uct_srd_iface_cep_remove_ep(uct_srd_iface_t *iface, uct_srd_ep_t *ep)
{
    if (!(ep->flags & UCT_SRD_EP_FLAG_ON_CEP)) {
        return;
    }

    ucs_conn_match_remove_elem(&iface->conn_match_ctx, &ep->conn_match,
                               uct_srd_iface_cep_ep_queue_type(ep));
    ep->flags &= ~UCT_SRD_EP_FLAG_ON_CEP;
}

static void uct_srd_iface_send_skb_init(uct_iface_h tl_iface, void *obj,
                                        uct_mem_h memh)
{
    uct_srd_send_skb_t *skb = obj;

    skb->lkey  = uct_ib_memh_get_lkey(memh);
    skb->flags = UCT_SRD_SEND_SKB_FLAG_INVALID;
}

static ucs_status_t
uct_srd_iface_create_qp(uct_srd_iface_t *iface,
                        const uct_srd_iface_config_t *config)
{
    uct_ib_efadv_md_t *efadv_md =
        ucs_derived_of(uct_ib_iface_md(&iface->super), uct_ib_efadv_md_t);
    const uct_ib_efadv_t *efadv = &efadv_md->efadv;
    struct ibv_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    int ret;

	memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    qp_init_attr.qp_type             = IBV_QPT_DRIVER;
    qp_init_attr.sq_sig_all          = 1;
    qp_init_attr.send_cq             = iface->super.cq[UCT_IB_DIR_TX];
    qp_init_attr.recv_cq             = iface->super.cq[UCT_IB_DIR_RX];
    qp_init_attr.cap.max_send_wr     = ucs_min(config->super.tx.queue_len,
                                               uct_ib_efadv_max_sq_wr(efadv));
    qp_init_attr.cap.max_recv_wr     = ucs_min(config->super.rx.queue_len,
                                               uct_ib_efadv_max_rq_wr(efadv));
    qp_init_attr.cap.max_send_sge    = 1 + ucs_min(config->super.tx.min_sge,
                                                   (uct_ib_efadv_max_sq_sge(efadv) - 1));
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_inline_data = ucs_min(config->super.tx.min_inline,
                                               uct_ib_efadv_inline_buf_size(efadv));

    iface->qp = efadv_create_driver_qp(efadv_md->super.pd, &qp_init_attr,
                                      EFADV_QP_DRIVER_TYPE_SRD);

    if (iface->qp == NULL) {
        ucs_error("iface=%p: failed to create %s QP on "UCT_IB_IFACE_FMT
                  " TX wr:%d sge:%d inl:%d resp:%d RX wr:%d sge:%d resp:%d: %m",
                  iface, uct_ib_qp_type_str(UCT_IB_QPT_SRD),
                  UCT_IB_IFACE_ARG(&iface->super),
                  qp_init_attr.cap.max_send_wr,
                  qp_init_attr.cap.max_send_sge,
                  qp_init_attr.cap.max_inline_data,
                  iface->super.config.max_inl_cqe[UCT_IB_DIR_TX],
                  qp_init_attr.cap.max_recv_wr,
                  qp_init_attr.cap.max_recv_sge,
                  iface->super.config.max_inl_cqe[UCT_IB_DIR_RX]);
        return UCS_ERR_IO_ERROR;
    }

    iface->config.max_inline = qp_init_attr.cap.max_inline_data;
    iface->config.tx_qp_len  = qp_init_attr.cap.max_send_wr;
    iface->tx.available      = qp_init_attr.cap.max_send_wr;
    iface->rx.available      = qp_init_attr.cap.max_recv_wr;

    ucs_debug("iface=%p: created %s QP 0x%x on "UCT_IB_IFACE_FMT
              " TX wr:%d sge:%d inl:%d resp:%d RX wr:%d sge:%d resp:%d",
              iface, uct_ib_qp_type_str(UCT_IB_QPT_SRD),
              iface->qp->qp_num, UCT_IB_IFACE_ARG(&iface->super),
              qp_init_attr.cap.max_send_wr,
              qp_init_attr.cap.max_send_sge,
              qp_init_attr.cap.max_inline_data,
              iface->super.config.max_inl_cqe[UCT_IB_DIR_TX],
              qp_init_attr.cap.max_recv_wr,
              qp_init_attr.cap.max_recv_sge,
              iface->super.config.max_inl_cqe[UCT_IB_DIR_RX]);


    memset(&qp_attr, 0, sizeof(qp_attr));
    /* Modify QP to INIT state */
    qp_attr.qp_state   = IBV_QPS_INIT;
    qp_attr.pkey_index = iface->super.pkey_index;
    qp_attr.port_num   = iface->super.config.port_num;
    qp_attr.qkey       = UCT_IB_KEY;
    ret = ibv_modify_qp(iface->qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
    if (ret) {
        ucs_error("Failed to modify SRD QP to INIT: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTR */
    qp_attr.qp_state = IBV_QPS_RTR;
    ret = ibv_modify_qp(iface->qp, &qp_attr, IBV_QP_STATE);
    if (ret) {
        ucs_error("Failed to modify SRD QP to RTR: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTS */
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    ret = ibv_modify_qp(iface->qp, &qp_attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
    if (ret) {
        ucs_error("Failed to modify SRD QP to RTS: %m");
        goto err_destroy_qp;
    }

    return UCS_OK;
err_destroy_qp:
    uct_ib_destroy_qp(iface->qp);
    return UCS_ERR_INVALID_PARAM;
}

static ucs_conn_sn_t
uct_srd_iface_conn_match_get_conn_sn(const ucs_conn_match_elem_t *elem)
{
    uct_srd_ep_t *ep = ucs_container_of(elem, uct_srd_ep_t, conn_match);
    return ep->conn_sn;
}

static const char *
uct_srd_iface_conn_match_peer_address_str(const ucs_conn_match_ctx_t *conn_match_ctx,
                                          const void *address,
                                          char *str, size_t max_size)
{
    uct_srd_iface_t *iface = ucs_container_of(conn_match_ctx,
                                              uct_srd_iface_t,
                                              conn_match_ctx);
    return uct_srd_iface_peer_address_str(iface, address, str, max_size);
}

static void
uct_srd_iface_conn_match_purge_cb(ucs_conn_match_ctx_t *conn_match_ctx,
                                  ucs_conn_match_elem_t *elem)
{
    uct_srd_ep_t *ep = ucs_container_of(elem, uct_srd_ep_t, conn_match);

    ep->flags &= ~UCT_SRD_EP_FLAG_ON_CEP;
    return UCS_CLASS_DELETE_FUNC_NAME(uct_srd_ep_t)(&ep->super.super);
}

ucs_status_t
uct_srd_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_srd_iface_t *iface = ucs_derived_of(tl_iface, uct_srd_iface_t);
    uct_srd_iface_addr_t *addr = (uct_srd_iface_addr_t *)iface_addr;

    uct_ib_pack_uint24(addr->qp_num, iface->qp->qp_num);

    return UCS_OK;
}

ucs_status_t uct_srd_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                 uct_completion_t *comp)
{
    uct_srd_iface_t *iface = ucs_derived_of(tl_iface, uct_srd_iface_t);
    uct_srd_ep_t *ep;
    ucs_status_t status;
    int i, count;

    ucs_trace_func("");

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    uct_srd_enter(iface);

    count = 0;
    ucs_ptr_array_for_each(ep, i, &iface->eps) {
        /* srd ep flush returns either ok or in progress */
        status = uct_srd_ep_flush_nolock(iface, ep, NULL);
        if ((status == UCS_INPROGRESS) || (status == UCS_ERR_NO_RESOURCE)) {
            ++count;
        }
    }

    uct_srd_leave(iface);
    if (count != 0) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super);
        return UCS_INPROGRESS;
    }

    UCT_TL_IFACE_STAT_FLUSH(&iface->super.super);
    return UCS_OK;
}

void uct_srd_iface_add_ep(uct_srd_iface_t *iface, uct_srd_ep_t *ep)
{
    ep->ep_id = ucs_ptr_array_insert(&iface->eps, ep);
}

void uct_srd_iface_remove_ep(uct_srd_iface_t *iface, uct_srd_ep_t *ep)
{
    if (ep->ep_id != UCT_SRD_EP_NULL_ID) {
        ucs_trace("iface(%p) remove ep: %p id %d", iface, ep, ep->ep_id);
        ucs_ptr_array_remove(&iface->eps, ep->ep_id);
    }
}

void uct_srd_iface_replace_ep(uct_srd_iface_t *iface,
                             uct_srd_ep_t *old_ep, uct_srd_ep_t *new_ep)
{
    void *p;
    ucs_assert_always(old_ep != new_ep);
    ucs_assert_always(old_ep->ep_id != new_ep->ep_id);
    p = ucs_ptr_array_replace(&iface->eps, old_ep->ep_id, new_ep);
    ucs_assert_always(p == (void *)old_ep);
    ucs_trace("replace_ep: old(%p) id=%d new(%p) id=%d",
              old_ep, old_ep->ep_id, new_ep, new_ep->ep_id);
    ucs_ptr_array_remove(&iface->eps, new_ep->ep_id);
}

uct_srd_send_skb_t *uct_srd_iface_ctl_skb_get(uct_srd_iface_t *iface)
{
    uct_srd_send_skb_t *skb;

    /* grow reserved skb's queue on-demand */
    skb = ucs_mpool_get(&iface->tx.mp);
    if (skb == NULL) {
        ucs_fatal("failed to allocate control skb");
    }

    VALGRIND_MAKE_MEM_DEFINED(&skb->lkey, sizeof(skb->lkey));
    skb->flags = 0;
    return skb;
}

void uct_srd_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_srd_iface_t *iface =
        ucs_container_of(self, uct_srd_iface_t, super.release_desc);

    uct_srd_enter(iface);
    uct_ib_iface_release_desc(self, desc);
    uct_srd_leave(iface);
}

static void uct_srd_iface_delete_eps(uct_srd_iface_t *iface)
{
    uct_srd_ep_t *ep;
    int i;

    ucs_ptr_array_for_each(ep, i, &iface->eps) {
        ucs_assert(!(ep->flags & UCT_SRD_EP_FLAG_ON_CEP));
        UCS_CLASS_DELETE_FUNC_NAME(uct_srd_ep_t)(&ep->super.super);
    }
}

static const void*
uct_srd_ep_get_conn_address(const ucs_conn_match_elem_t *elem)
{
    uct_srd_ep_t *ep = ucs_container_of(elem, uct_srd_ep_t, conn_match);
    return uct_srd_ep_get_peer_address(ep);
}

ucs_status_t
uct_srd_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_srd_iface_t *iface = ucs_derived_of(tl_iface, uct_srd_iface_t);
    ucs_status_t status;

    ucs_trace_func("");


    status = uct_ib_iface_query(&iface->super,
                                UCT_IB_DETH_LEN + sizeof(uct_srd_neth_t),
                                iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->cap.flags              = UCT_IFACE_FLAG_AM_BCOPY         |
                                         UCT_IFACE_FLAG_AM_ZCOPY         |
                                         UCT_IFACE_FLAG_CONNECT_TO_EP    |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                         UCT_IFACE_FLAG_PENDING          |
                                         UCT_IFACE_FLAG_EP_CHECK         |
                                         UCT_IFACE_FLAG_CB_SYNC          |
                                         UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    iface_attr->cap.am.max_short       = uct_ib_iface_hdr_size(iface->config.max_inline,
                                                               sizeof(uct_srd_neth_t));
    iface_attr->cap.am.max_bcopy       = iface->super.config.seg_size - sizeof(uct_srd_neth_t);
    iface_attr->cap.am.min_zcopy       = 0;
    iface_attr->cap.am.max_zcopy       = iface->super.config.seg_size - sizeof(uct_srd_neth_t);
    iface_attr->cap.am.align_mtu       = uct_ib_mtu_value(uct_ib_iface_port_attr(&iface->super)->active_mtu);
    iface_attr->cap.am.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.am.max_iov         = iface->config.max_send_sge;
    iface_attr->cap.am.max_hdr         = uct_ib_iface_hdr_size(iface->super.config.seg_size,
                                                               sizeof(uct_srd_neth_t) +
                                                               sizeof(uct_srd_zcopy_desc_t));

    iface_attr->iface_addr_len         = sizeof(uct_srd_iface_addr_t);
    iface_attr->ep_addr_len            = sizeof(uct_srd_ep_addr_t);
    iface_attr->max_conn_priv          = 0;

    if (iface_attr->cap.am.max_short) {
        iface_attr->cap.flags |= UCT_IFACE_FLAG_AM_SHORT;
    }

    /* TODO: set the correct values for SRD */
    iface_attr->latency.c += 30e-9;
    iface_attr->overhead   = 105e-9;

    return UCS_OK;
}

static UCS_CLASS_DECLARE_DELETE_FUNC(uct_srd_iface_t, uct_iface_t);

static uct_ib_iface_ops_t uct_srd_iface_ops = {
    .super = {
        .iface_estimate_perf = uct_base_iface_estimate_perf,
        .iface_vfs_refresh   = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
        .ep_query            = (uct_ep_query_func_t)ucs_empty_function_return_unsupported
    },
    .create_cq      = uct_ib_verbs_create_cq,
    .arm_cq         = (uct_ib_iface_arm_cq_func_t)
                      ucs_empty_function_return_unsupported,
    .event_cq       = (uct_ib_iface_event_cq_func_t)ucs_empty_function,
    .handle_failure = (uct_ib_iface_handle_failure_func_t)
                      ucs_empty_function_do_assert
};

static uct_iface_ops_t uct_srd_iface_tl_ops = {
    .ep_am_short              = uct_srd_ep_am_short,
    .ep_am_short_iov          = uct_srd_ep_am_short_iov,
    .ep_am_bcopy              = uct_srd_ep_am_bcopy,
    .ep_am_zcopy              = uct_srd_ep_am_zcopy,
    .ep_pending_add           = uct_srd_ep_pending_add,
    .ep_pending_purge         = uct_srd_ep_pending_purge,
    .ep_flush                 = uct_srd_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_check                 = uct_srd_ep_check,
    .ep_create                = uct_srd_ep_create,
    .ep_destroy               = uct_srd_ep_disconnect,
    .ep_get_address           = uct_srd_ep_get_address,
    .ep_connect_to_ep         = uct_srd_ep_connect_to_ep,
    .iface_flush              = uct_srd_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_srd_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_srd_iface_progress,
    .iface_event_fd_get       = (uct_iface_event_fd_get_func_t)
                                ucs_empty_function_return_unsupported,
    .iface_event_arm          = (uct_iface_event_arm_func_t)
                                ucs_empty_function_return_unsupported,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_srd_iface_t),
    .iface_query              = uct_srd_iface_query,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_get_address        = uct_srd_iface_get_address,
    .iface_is_reachable       = uct_ib_iface_is_reachable
};

UCS_CLASS_INIT_FUNC(uct_srd_iface_t, uct_md_h md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config)
{
    ucs_status_t status;
    size_t data_size;
    int mtu;

    uct_srd_iface_config_t *config      = ucs_derived_of(tl_config,
                                                         uct_srd_iface_config_t);
    uct_ib_iface_init_attr_t init_attr  = {};
    ucs_conn_match_ops_t conn_match_ops = {
        .get_address = uct_srd_ep_get_conn_address,
        .get_conn_sn = uct_srd_iface_conn_match_get_conn_sn,
        .address_str = uct_srd_iface_conn_match_peer_address_str,
        .purge_cb    = uct_srd_iface_conn_match_purge_cb
    };

    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_OPEN_MODE,
                    "UCT_IFACE_PARAM_FIELD_OPEN_MODE is not defined");
    if (!(params->open_mode & UCT_IFACE_OPEN_MODE_DEVICE)) {
        ucs_error("only UCT_IFACE_OPEN_MODE_DEVICE is supported");
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_trace_func("%s: iface=%p worker=%p rx_headroom=%zu",
                   params->mode.device.dev_name, self, worker,
                   (params->field_mask & UCT_IFACE_PARAM_FIELD_RX_HEADROOM) ?
                   params->rx_headroom : 0);

    status = uct_ib_device_mtu(params->mode.device.dev_name, md, &mtu);
    if (status != UCS_OK) {
        return status;
    }

    init_attr.cq_len[UCT_IB_DIR_TX] = config->super.tx.queue_len;
    init_attr.cq_len[UCT_IB_DIR_RX] = config->super.rx.queue_len;
    init_attr.rx_priv_len           = sizeof(uct_srd_recv_skb_t) -
                                      sizeof(uct_ib_iface_recv_desc_t);
    init_attr.rx_hdr_len            = sizeof(uct_srd_neth_t);
    init_attr.seg_size              = ucs_min(mtu, config->super.seg_size);
    init_attr.qp_type               = IBV_QPT_DRIVER;

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &uct_srd_iface_tl_ops,
                              &uct_srd_iface_ops, md, worker,
                              params, &config->super, &init_attr);

    /* Redefine receive desc release callback */
    self->super.release_desc.cb = uct_srd_iface_release_desc;

    status = uct_srd_iface_create_qp(self, config);
    if (status != UCS_OK) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_ptr_array_init(&self->eps, "srd_eps");

    status = uct_ib_iface_recv_mpool_init(&self->super, &config->super, params,
                                          "srd_recv_skb", &self->rx.mp);
    if (status != UCS_OK) {
        goto err_qp;
    }

    self->rx.quota =
        self->rx.available - ucs_min(self->rx.available,
                                     config->ud_common.rx_queue_len_init);
    self->rx.available -= self->rx.quota;

    ucs_mpool_grow(&self->rx.mp, self->rx.available);

    data_size = sizeof(uct_srd_ctl_hdr_t) + self->super.addr_size;
    data_size = ucs_max(data_size, self->super.config.seg_size);
    data_size = ucs_max(data_size, sizeof(uct_srd_am_short_hdr_t) +
                                   sizeof(uct_srd_neth_t));
    status = uct_iface_mpool_init(&self->super.super, &self->tx.mp,
                                  sizeof(uct_srd_send_skb_t) + data_size,
                                  sizeof(uct_srd_send_skb_t),
                                  UCT_SRD_SKB_ALIGN,
                                  &config->super.tx.mp, self->config.tx_qp_len,
                                  uct_srd_iface_send_skb_init, "srd_tx_skb");
    if (status != UCS_OK) {
        goto err_rx_mpool;
    }

    ucs_arbiter_init(&self->tx.pending_q);

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_srd_iface_stats_class,
                                  self->super.super.stats, "-%p", self);
    if (status != UCS_OK) {
        goto err_tx_mpool;
    }

    memset(&self->tx.wr_inl, 0, sizeof(self->tx.wr_inl));
    self->tx.wr_inl.opcode            = IBV_WR_SEND;
    self->tx.wr_inl.wr_id             = 0xBEEBBEEB;
    self->tx.wr_inl.wr.ud.remote_qkey = UCT_IB_KEY;
    self->tx.wr_inl.imm_data          = 0;
    self->tx.wr_inl.next              = 0;
    self->tx.wr_inl.sg_list           = self->tx.sge;

    memset(&self->tx.wr_skb, 0, sizeof(self->tx.wr_skb));
    self->tx.wr_skb.opcode            = IBV_WR_SEND;
    self->tx.wr_skb.wr_id             = 0xFAAFFAAF;
    self->tx.wr_skb.wr.ud.remote_qkey = UCT_IB_KEY;
    self->tx.wr_skb.imm_data          = 0;
    self->tx.wr_skb.next              = 0;
    self->tx.wr_skb.sg_list           = self->tx.sge;
    self->tx.wr_skb.num_sge           = 1;

    self->tx.send_sn                  = 0;
    self->tx.skb                      = NULL;
    self->super.config.sl             = uct_ib_iface_config_select_sl(&config->super);

    if (self->super.config.rx_max_batch < UCT_SRD_RX_BATCH_MIN) {
        ucs_warn("rx max batch is too low (%d < %d), performance may be impacted",
                 self->super.config.rx_max_batch, UCT_SRD_RX_BATCH_MIN);
    }

    status = uct_srd_qp_max_send_sge(self, &self->config.max_send_sge);
    if (status != UCS_OK) {
        goto err_release_stats;
    }

    while (self->rx.available >= self->super.config.rx_max_batch) {
        uct_srd_iface_post_recv(self);
    }

    ucs_conn_match_init(&self->conn_match_ctx,
                        uct_srd_ep_get_peer_address_length(),
                        UCT_SRD_IFACE_CEP_CONN_SN_MAX, &conn_match_ops);

    return UCS_OK;

err_release_stats:
    UCS_STATS_NODE_FREE(self->stats);
err_tx_mpool:
    ucs_mpool_cleanup(&self->tx.mp, 1);
err_rx_mpool:
    ucs_mpool_cleanup(&self->rx.mp, 1);
err_qp:
    uct_ib_destroy_qp(self->qp);
    ucs_ptr_array_cleanup(&self->eps, 1);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_srd_iface_t)
{
    ucs_trace_func("");

    /* TODO: proper flush and connection termination */
    uct_srd_enter(self);
    uct_base_iface_progress_disable(&self->super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    ucs_conn_match_cleanup(&self->conn_match_ctx);
    uct_srd_iface_delete_eps(self);
    ucs_debug("iface(%p): cep cleanup", self);
    ucs_mpool_cleanup(&self->tx.mp, 0);
    /* TODO: qp to error state and cleanup all wqes */
    ucs_mpool_cleanup(&self->rx.mp, 0);
    uct_ib_destroy_qp(self->qp);
    ucs_debug("iface(%p): ptr_array cleanup", self);
    ucs_ptr_array_cleanup(&self->eps, 1);
    ucs_arbiter_cleanup(&self->tx.pending_q);
    UCS_STATS_NODE_FREE(self->stats);
    uct_srd_leave(self);
}


UCS_CLASS_DEFINE(uct_srd_iface_t, uct_ib_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_srd_iface_t, uct_iface_t, uct_md_h,
                                 uct_worker_h, const uct_iface_params_t*,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_srd_iface_t, uct_iface_t);

ucs_config_field_t uct_srd_iface_config_table[] = {
    {UCT_IB_CONFIG_PREFIX, "", NULL,
     ucs_offsetof(uct_srd_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

    {"SRD_", "", NULL,
     ucs_offsetof(uct_srd_iface_config_t, ud_common),
     UCS_CONFIG_TYPE_TABLE(uct_ud_iface_common_config_table)},

    {NULL}
};

static ucs_status_t
uct_srd_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                         unsigned *num_tl_devices_p)
{
    uct_ib_md_t *ib_md      = ucs_derived_of(md, uct_ib_md_t);
    struct ibv_context *ctx = ib_md->dev.ibv_context;

    if (!uct_ib_efadv_check(ctx->device)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return uct_ib_device_query_ports(&ib_md->dev, 0, tl_devices_p,
                                     num_tl_devices_p);
}

UCT_TL_DEFINE(&uct_ib_component, srd, uct_srd_query_tl_devices,
              uct_srd_iface_t,  "SRD_",
              uct_srd_iface_config_table, uct_srd_iface_config_t);
