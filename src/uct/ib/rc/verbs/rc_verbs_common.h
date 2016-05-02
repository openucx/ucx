/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_VERBS_COMMON_H
#define UCT_RC_VERBS_COMMON_H

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>


/* definitions common to rc_verbs and dc_verbs go here */

typedef struct uct_rc_verbs_txcnt {
    uint16_t       pi;      /* producer (post_send) count */
    uint16_t       ci;      /* consumer (ibv_poll_cq) completion count */
} uct_rc_verbs_txcnt_t;


void uct_rc_verbs_txcnt_init(uct_rc_verbs_txcnt_t *txcnt);

static inline void 
uct_rc_verbs_txqp_posted(uct_rc_txqp_t *txqp, uct_rc_verbs_txcnt_t *txcnt, 
                         uct_rc_iface_t *iface, int signaled)
{
    txcnt->pi++;
    uct_rc_txqp_posted(txqp, iface, 1, signaled);
}

static inline void 
uct_rc_verbs_txqp_completed(uct_rc_txqp_t *txqp, uct_rc_verbs_txcnt_t *txcnt, uint16_t count)
{
    txcnt->ci += count;
    uct_rc_txqp_available_add(txqp, count);
}


typedef struct uct_rc_verbs_iface_common {
    struct ibv_sge         inl_sge[2];
} uct_rc_verbs_iface_common_t;

void uct_rc_verbs_iface_common_init(uct_rc_verbs_iface_common_t *iface);

ucs_status_t uct_rc_verbs_iface_prepost_recvs_common(uct_rc_iface_t *iface);

void uct_rc_verbs_iface_query_common(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr, 
                                     int max_inline, int short_desc_size);

unsigned uct_rc_verbs_iface_post_recv_always(uct_rc_iface_t *iface, unsigned max);


static inline unsigned uct_rc_verbs_iface_post_recv_common(uct_rc_iface_t *iface,
                                                           int fill)
{
    unsigned batch = iface->super.config.rx_max_batch;
    unsigned count;

    if (iface->rx.available < batch) {
        if (ucs_likely(fill == 0)) {
            return 0;
        } else {
            count = iface->rx.available;
        }
    } else {
        count = batch;
    }

    return uct_rc_verbs_iface_post_recv_always(iface, count);
}

#define UCT_RC_VERBS_IFACE_FOREACH_TXWQE(iface, i, wc, num_wcs) \
      status = uct_ib_poll_cq((iface)->super.send_cq, &num_wcs, wc); \
      if (status != UCS_OK) { \
          return; \
      } \
      UCS_STATS_UPDATE_COUNTER((iface)->stats, UCT_RC_IFACE_STAT_TX_COMPLETION, num_wcs); \
      /* it is possible to update available outside of the loop because */ \
      /* completion with error is a FATAL error */ \
      (iface)->tx.cq_available += num_wcs; \
      UCT_IB_IFACE_VERBS_FOREACH_TXWQE(&(iface)->super.super, i, wc, num_wcs) 


/* TODO: think of a better name */
static inline int uct_rc_verbs_txcq_get_comp_count(struct ibv_wc *wc)
{
    return wc->wr_id + 1;
}

static UCS_F_ALWAYS_INLINE ucs_status_t 
uct_rc_verbs_iface_poll_rx_common(uct_rc_iface_t *iface)
{
    uct_rc_hdr_t *hdr;
    int i;
    ucs_status_t status;
    unsigned num_wcs = iface->super.config.rx_max_poll;
    struct ibv_wc wc[num_wcs];
    uct_ib_iface_recv_desc_t *desc;


    status = uct_ib_poll_cq(iface->super.recv_cq, &num_wcs, wc);
    if (status != UCS_OK) {
        goto out;
    }

    UCT_IB_IFACE_VERBS_FOREACH_RXWQE(&iface->super, i, hdr, wc, num_wcs) {

        uct_ib_log_recv_completion(&iface->super, IBV_QPT_RC, &wc[i],
                                   hdr, uct_rc_ep_am_packet_dump);
        desc = (uct_ib_iface_recv_desc_t *)wc[i].wr_id;
        if (ucs_unlikely(hdr->am_id & UCT_RC_EP_FC_MASK)) {
            void *udesc;
            udesc = (char*)desc + iface->super.config.rx_headroom_offset;
            status = uct_rc_iface_handle_fc(iface, wc[i].qp_num, hdr,
                                            wc[i].byte_len - sizeof(*hdr),
                                            udesc);
            if (status == UCS_OK) {
                ucs_mpool_put_inline(desc);
            } else {
                uct_recv_desc_iface(udesc) = &iface->super.super.super;
            }
        } else {
            uct_ib_iface_invoke_am(&iface->super, hdr->am_id, hdr + 1,
                                   wc[i].byte_len - sizeof(*hdr), desc);
        }
    }
    iface->rx.available += num_wcs;
    UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_RC_IFACE_STAT_RX_COMPLETION, num_wcs);

out:
    uct_rc_verbs_iface_post_recv_common(iface, 0);
    return status;
}

static inline void 
uct_rc_verbs_iface_common_fill_inl_sge(uct_rc_verbs_iface_common_t *iface,
                                       uct_rc_am_short_hdr_t *am,
                                       uint8_t id, uint64_t hdr,
                                       const void *buffer, unsigned length)
{
    am->rc_hdr.am_id = id;
    am->am_hdr       = hdr;

    iface->inl_sge[0].addr      = (uintptr_t)am;
    iface->inl_sge[0].length    = sizeof(*am);
    iface->inl_sge[1].addr      = (uintptr_t)buffer;
    iface->inl_sge[1].length    = length;
}

#define UCT_RC_VERBS_FILL_BCOPY_WR(_wr, _sge, _length, _wr_opcode) \
    _wr.sg_list = &_sge; \
    _wr.num_sge = 1; \
    _sge.length = sizeof(uct_rc_hdr_t) + _length; \
    _wr_opcode = IBV_WR_SEND;


#define UCT_RC_VERBS_FILL_DESC_WR(_wr, _desc) \
    { \
        struct ibv_sge *sge; \
        (_wr)->next    = NULL; \
        sge            = (_wr)->sg_list; \
        sge->addr      = (uintptr_t)(desc + 1); \
        sge->lkey      = (_desc)->lkey; \
    }

#define UCT_RC_VERBS_FILL_ZCOPY_WR(_wr, _sge, _wr_opcode) \
    _wr.sg_list = _sge; \
    _wr_opcode  = IBV_WR_SEND; \
    _wr.num_sge = 2;

static inline void uct_rc_verbs_zcopy_sge_fill(struct ibv_sge *sge,
                                               unsigned header_length, const void *payload,
                                               size_t length, uct_mem_h memh)
{
    sge[0].length = sizeof(uct_rc_hdr_t) + header_length;

    sge[1].addr   = (uintptr_t)payload;
    sge[1].length = length;
    sge[1].lkey   = (memh == UCT_INVALID_MEM_HANDLE) ? 0 : ((struct ibv_mr *)memh)->lkey;
}

#endif
