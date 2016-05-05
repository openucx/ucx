/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_verbs.h"

#include <uct/ib/rc/verbs/rc_verbs_common.h>
#include <uct/api/uct.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/base/uct_pd.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <string.h>


static UCS_CLASS_INIT_FUNC(uct_dc_verbs_ep_t, uct_iface_h tl_iface)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_verbs_iface_t);
    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_dc_ep_t, &iface->super);
    self->ah = NULL;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_verbs_ep_t)
{
    ucs_trace_func("");
    if (self->ah) { 
        ibv_destroy_ah(self->ah);
        self->ah = NULL;
    }
}

UCS_CLASS_DEFINE(uct_dc_verbs_ep_t, uct_dc_ep_t);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_dc_verbs_ep_t, uct_ep_t);


static ucs_status_t uct_dc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_verbs_iface_t);

    uct_dc_iface_query(&iface->super, iface_attr);
    ucs_debug("max_inline is %d", iface->super.config.max_inline);
    uct_rc_verbs_iface_common_query(&iface->verbs_common, 
                                    &iface->super.super, iface_attr);
    /*TODO: remove flags once we have a full functionality */
    iface_attr->cap.flags           = UCT_IFACE_FLAG_AM_ZCOPY|
                                      UCT_IFACE_FLAG_AM_BCOPY|
                                      UCT_IFACE_FLAG_AM_SHORT|
                                      UCT_IFACE_FLAG_PUT_SHORT|
                                      UCT_IFACE_FLAG_PUT_BCOPY|
                                      UCT_IFACE_FLAG_PUT_ZCOPY|
                                      UCT_IFACE_FLAG_GET_BCOPY|
                                      UCT_IFACE_FLAG_GET_ZCOPY|
                                      UCT_IFACE_FLAG_AM_CB_SYNC|UCT_IFACE_FLAG_CONNECT_TO_IFACE;

    return UCS_OK;
}

static ucs_status_t
uct_dc_verbs_ep_create_connected(uct_iface_h iface_h, const uct_device_addr_t *dev_addr,
                                 const uct_iface_addr_t *iface_addr, uct_ep_h *new_ep_p)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(iface_h, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = NULL;
    const uct_ib_address_t *ib_addr = (const uct_ib_address_t *)dev_addr;
    const uct_dc_iface_addr_t *if_addr = (const uct_dc_iface_addr_t *)iface_addr;
    ucs_status_t status;

    status = UCS_CLASS_NEW(uct_dc_verbs_ep_t, &ep, iface_h);
    if (status != UCS_OK) {
        ucs_error("failed to allocate new ep");
        return status;
    }

    ep->ah = uct_ib_create_ah(&iface->super.super.super, ib_addr->lid);
    if (ep->ah == NULL) {
        *new_ep_p = NULL;
        return UCS_ERR_INVALID_ADDR;
    }

    ep->dest_qpn = uct_ib_unpack_uint24(if_addr->qp_num);
    *new_ep_p = &ep->super.super.super;
    ucs_debug("created ep %p on iface %p", ep, iface);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_iface_post_send(uct_dc_verbs_iface_t* iface, uct_dc_verbs_ep_t *ep, int dci,
                             struct ibv_exp_send_wr *wr, int send_flags)
{
    struct ibv_exp_send_wr *bad_wr;
    int ret;

    if (!(send_flags & IBV_SEND_SIGNALED)) {
        /* TODO: check tx moderation */
        send_flags |= IBV_SEND_SIGNALED;
    }

    wr->exp_send_flags    = send_flags;
    wr->wr_id             = iface->super.tx.dcis[dci].unsignaled;
    wr->dc.ah             = ep->ah;
    wr->dc.dct_number     = ep->dest_qpn;
    wr->dc.dct_access_key = UCT_IB_DC_KEY;

    uct_ib_log_exp_post_send(&iface->super.super.super, iface->super.tx.dcis[dci].qp, wr,
                             (wr->exp_opcode == IBV_WR_SEND) ? uct_rc_ep_am_packet_dump : NULL);

    UCT_IB_INSTRUMENT_RECORD_SEND_EXP_WR_LEN("uct_dc_verbs_ep_post_send", wr);

    ret = ibv_exp_post_send(iface->super.tx.dcis[dci].qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_send() returned %d (%m)", ret);
    }

    uct_rc_verbs_txqp_posted(&iface->super.tx.dcis[dci], &iface->dcis_txcnt[dci], 
                             &iface->super.super, send_flags & IBV_SEND_SIGNALED);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_iface_post_send_desc(uct_dc_verbs_iface_t *iface, 
                                  uct_dc_verbs_ep_t *ep, int dci,
                                  struct ibv_exp_send_wr *wr,
                                  uct_rc_iface_send_desc_t *desc, int send_flags)
{
    UCT_RC_VERBS_FILL_DESC_WR(wr, desc);
    uct_dc_verbs_iface_post_send(iface, ep, dci, wr, send_flags);
    uct_rc_txqp_add_send_op(&iface->super.tx.dcis[dci], &desc->super, 
                            iface->dcis_txcnt[dci].pi);
}


static inline ucs_status_t
uct_dc_verbs_ep_rdma_zcopy(uct_dc_verbs_ep_t *ep, const void *buffer, size_t length,
                           struct ibv_mr *mr, uint64_t remote_addr,
                           uct_rkey_t rkey, uct_completion_t *comp,
                           int opcode)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_dc_verbs_iface_t);
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;
    int dci;

    UCT_SKIP_ZERO_LENGTH(length);

    UCT_DC_CHECK_RES(&iface->super, &ep->super, dci);
    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.exp_opcode, opcode, sge, length, remote_addr, rkey);
    wr.next = NULL;
    uct_rc_verbs_rdma_zcopy_sge_fill(&sge, buffer, length, mr);

    uct_dc_verbs_iface_post_send(iface, ep, dci, &wr, IBV_SEND_SIGNALED);
    uct_rc_txqp_add_send_comp(&iface->super.super, &iface->super.tx.dcis[dci], comp, 
                              iface->dcis_txcnt[dci].pi);
    return UCS_INPROGRESS;
}


ucs_status_t uct_dc_verbs_ep_get_bcopy(uct_ep_h tl_ep,
                                       uct_unpack_callback_t unpack_cb,
                                       void *arg, size_t length,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;
    int dci;

    UCT_CHECK_LENGTH(length, iface->super.super.super.config.seg_size, "get_bcopy");
    UCT_DC_CHECK_RES(&iface->super, &ep->super, dci);
    UCT_RC_IFACE_GET_TX_GET_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp, 
                                       desc, unpack_cb, comp, arg, length);
    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.exp_opcode, IBV_WR_RDMA_READ, sge, length, remote_addr,
                              rkey);
    UCT_TL_EP_STAT_OP(&ep->super.super, GET, BCOPY, length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, dci, &wr, desc, IBV_SEND_SIGNALED);

    return UCS_INPROGRESS;
}

ucs_status_t uct_dc_verbs_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                       uct_mem_h memh, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    ucs_status_t status;

    status = uct_dc_verbs_ep_rdma_zcopy(ep, buffer, length, memh, remote_addr,
                                        rkey, comp, IBV_WR_RDMA_READ);
    if (status == UCS_INPROGRESS) {
        UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY, length);
    }
    return status;
}

ucs_status_t uct_dc_verbs_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    int dci;

    UCT_CHECK_LENGTH(length, iface->super.config.max_inline, "put_short");

    UCT_DC_CHECK_RES(&iface->super, &ep->super, dci);
    UCT_RC_VERBS_FILL_INL_PUT_WR(iface, remote_addr, rkey, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    uct_dc_verbs_iface_post_send(iface, ep, dci, &iface->inl_rwrite_wr, 
                                 IBV_SEND_INLINE|IBV_SEND_SIGNALED);

    return UCS_OK;
}

ssize_t uct_dc_verbs_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                  void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;
    size_t length;
    int dci;

    UCT_DC_CHECK_RES(&iface->super, &ep->super, dci);
    UCT_RC_IFACE_GET_TX_PUT_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp,
                                       desc, pack_cb, arg, length);
    UCT_RC_VERBS_FILL_RDMA_WR(wr, wr.exp_opcode, IBV_WR_RDMA_WRITE, sge, 
                              length, remote_addr, rkey);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, BCOPY, length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, dci, &wr, desc, IBV_SEND_SIGNALED);

    return length;
}


ucs_status_t uct_dc_verbs_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, size_t length,
                                       uct_mem_h memh, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    ucs_status_t status;

    status = uct_dc_verbs_ep_rdma_zcopy(ep, buffer, length, memh, remote_addr,
                                        rkey, comp, IBV_WR_RDMA_WRITE);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, ZCOPY, length);
    return status;
}

ucs_status_t uct_dc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      const void *buffer, unsigned length)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_am_short_hdr_t am;
    int dci;

    UCT_RC_CHECK_AM_SHORT(id, length, iface->super.config.max_inline);

    UCT_DC_CHECK_RES(&iface->super, &ep->super, dci);
    uct_rc_verbs_iface_fill_inl_am_sge(&iface->verbs_common, &am, id, hdr, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_dc_verbs_iface_post_send(iface, ep, dci, &iface->inl_am_wr, IBV_SEND_INLINE);

    return UCS_OK;
}

ssize_t uct_dc_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                 uct_pack_callback_t pack_cb, void *arg)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;
    size_t length;
    int dci;

    UCT_CHECK_AM_ID(id);

    UCT_DC_CHECK_RES(&iface->super, &ep->super, dci);
    UCT_RC_IFACE_GET_TX_AM_BCOPY_DESC(&iface->super.super, &iface->super.super.tx.mp, desc,
                                      id, pack_cb, arg, &length);
    UCT_RC_VERBS_FILL_AM_BCOPY_WR(wr, sge, length, wr.exp_opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, dci, &wr, desc, 0);

    return length;
}


ucs_status_t uct_dc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const void *payload,
                                      size_t length, uct_mem_h memh,
                                      uct_completion_t *comp)
{
    uct_dc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_verbs_iface_t);
    uct_dc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_dc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge[2];
    int dci, send_flags;

    UCT_RC_CHECK_AM_ZCOPY(id, header_length, length, 
                          iface->verbs_common.config.short_desc_size,
                          iface->super.super.super.config.seg_size);

    UCT_DC_CHECK_RES(&iface->super, &ep->super, dci);
    UCT_RC_IFACE_GET_TX_AM_ZCOPY_DESC(&iface->super.super, &iface->verbs_common.short_desc_mp, desc, 
                                      id, header, header_length, comp, &send_flags);
    uct_rc_verbs_am_zcopy_sge_fill(sge, header_length, payload, length, memh);
    UCT_RC_VERBS_FILL_AM_ZCOPY_WR(wr, sge, wr.exp_opcode);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY, header_length + length);
    uct_dc_verbs_iface_post_send_desc(iface, ep, dci, &wr, desc, send_flags);

    return UCS_INPROGRESS;
}

static inline ucs_status_t uct_dc_verbs_flush_dcis(uct_dc_iface_t *iface) 
{
    int i;
    int is_flush_done = 1;

    for (i = 0; i < iface->tx.ndci; i++) {
        if (uct_rc_txqp_available(&iface->tx.dcis[i]) == iface->super.config.tx_qp_len) {
            continue;
        }
        ucs_trace_data("dci %d is not flushed %d/%d", i, 
                       iface->tx.dcis[i].available, iface->super.config.tx_qp_len);
        is_flush_done = 0;
        if (uct_rc_txqp_unsignaled(&iface->tx.dcis[i]) != 0) { 
            /* TODO */
            ucs_fatal("unsignalled send is not supported!!!");
        }
    }

    return is_flush_done ? UCS_OK : UCS_INPROGRESS;
}


ucs_status_t uct_dc_verbs_iface_flush(uct_iface_h tl_iface)
{
    uct_dc_iface_t *iface = ucs_derived_of(tl_iface, uct_dc_iface_t);
    ucs_status_t status;

    status = uct_dc_verbs_flush_dcis(iface);
    if (status == UCS_OK) {
        UCT_TL_IFACE_STAT_FLUSH(&iface->super.super.super);
    } 
    else if (status == UCS_INPROGRESS) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super.super);
    }
    return status;
}

ucs_status_t uct_dc_verbs_ep_flush(uct_ep_h tl_ep)
{
    uct_dc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_dc_iface_t);
    ucs_status_t status;

    status = uct_dc_verbs_flush_dcis(iface);
    if (status == UCS_OK) {
        UCT_TL_EP_STAT_FLUSH(ucs_derived_of(tl_ep, uct_base_ep_t));
    } else if (status == UCS_INPROGRESS) {
        UCT_TL_EP_STAT_FLUSH_WAIT(ucs_derived_of(tl_ep, uct_base_ep_t));
    }
    return status;
}


static UCS_F_ALWAYS_INLINE void
uct_dc_verbs_poll_tx(uct_dc_verbs_iface_t *iface) 
{
    int i;
    ucs_status_t status;
    unsigned num_wcs = iface->super.super.super.config.rx_max_poll;
    struct ibv_wc wc[num_wcs];
    int count, dci;

    UCT_RC_VERBS_IFACE_FOREACH_TXWQE(&iface->super.super, i, wc, num_wcs) {
        count = uct_rc_verbs_txcq_get_comp_count(&wc[i]);
        ucs_assert_always(count == 1);
        dci = uct_dc_iface_dci_find(&iface->super, wc[i].qp_num);
        uct_rc_verbs_txqp_completed(&iface->super.tx.dcis[dci], &iface->dcis_txcnt[dci], count);
        ucs_trace_poll("dc tx completion on dc %d cound %d", dci, count);

        uct_rc_txqp_completion(&iface->super.tx.dcis[dci], iface->dcis_txcnt[dci].ci);
    }
}

/* TODO: make a macro that defines progress func */
static void uct_dc_verbs_iface_progress(void *arg)
{
    uct_dc_verbs_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_verbs_iface_poll_rx_common(&iface->super.super);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_dc_verbs_poll_tx(iface);
    }
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_dc_verbs_iface_t)(uct_iface_t*);

static uct_rc_iface_ops_t uct_dc_verbs_iface_ops = {
    {
        {
            .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_verbs_iface_t),
            .iface_query              = uct_dc_verbs_iface_query,
            .iface_get_device_address = uct_ib_iface_get_device_address,
            .iface_is_reachable       = uct_ib_iface_is_reachable,
            .iface_release_am_desc    = uct_ib_iface_release_am_desc, 
            .iface_get_address        = uct_dc_iface_get_address,

            .iface_flush              = uct_dc_verbs_iface_flush,

            .ep_create_connected      = uct_dc_verbs_ep_create_connected,
            .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_dc_verbs_ep_t),

            .ep_am_short              = uct_dc_verbs_ep_am_short,
            .ep_am_bcopy              = uct_dc_verbs_ep_am_bcopy,
            .ep_am_zcopy              = uct_dc_verbs_ep_am_zcopy,

            .ep_put_short             = uct_dc_verbs_ep_put_short,
            .ep_put_bcopy             = uct_dc_verbs_ep_put_bcopy,
            .ep_put_zcopy             = uct_dc_verbs_ep_put_zcopy,

            .ep_get_bcopy             = uct_dc_verbs_ep_get_bcopy,
            .ep_get_zcopy             = uct_dc_verbs_ep_get_zcopy,

            .ep_flush                 = uct_dc_verbs_ep_flush
        },
        .arm_tx_cq                = uct_ib_iface_arm_tx_cq,
        .arm_rx_cq                = uct_ib_iface_arm_rx_cq,
    },
    .fc_ctrl                  = NULL /* TODO: */
};

void uct_dc_verbs_iface_init_wrs(uct_dc_verbs_iface_t *self)
{
    
    /* Initialize inline work request */
    memset(&self->inl_am_wr, 0, sizeof(self->inl_am_wr));
    self->inl_am_wr.sg_list                 = self->verbs_common.inl_sge;
    self->inl_am_wr.num_sge                 = 2;
    self->inl_am_wr.exp_opcode              = IBV_WR_SEND;
    self->inl_am_wr.exp_send_flags          = IBV_SEND_INLINE;
    self->inl_am_wr.dc.dct_access_key       = UCT_IB_DC_KEY;

    memset(&self->inl_rwrite_wr, 0, sizeof(self->inl_rwrite_wr));
    self->inl_rwrite_wr.sg_list             = self->verbs_common.inl_sge;
    self->inl_rwrite_wr.num_sge             = 1;
    self->inl_rwrite_wr.exp_opcode          = IBV_WR_RDMA_WRITE;
    self->inl_rwrite_wr.exp_send_flags      = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    self->inl_rwrite_wr.dc.dct_access_key   = UCT_IB_DC_KEY;
}

static UCS_CLASS_INIT_FUNC(uct_dc_verbs_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    int i;
    ucs_status_t status;
    uct_dc_iface_config_t *config = ucs_derived_of(tl_config,
                                                   uct_dc_iface_config_t);
    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_dc_iface_t, &uct_dc_verbs_iface_ops, pd,
                              worker, dev_name, rx_headroom, 0, config);

    status = uct_rc_verbs_iface_common_init(&self->verbs_common, &self->super.super, &config->super);
    if (status != UCS_OK) {
        return status;
    }

    uct_dc_verbs_iface_init_wrs(self);

    status = uct_rc_verbs_iface_prepost_recvs_common(&self->super.super);
    if (status != UCS_OK) {
        goto out;
    }

    self->dcis_txcnt = ucs_malloc(self->super.tx.ndci * sizeof(uct_rc_verbs_txcnt_t), "dc");
    if (self->dcis_txcnt == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    for (i = 0; i < self->super.tx.ndci; i++) {
        uct_rc_verbs_txcnt_init(&self->dcis_txcnt[i]);
        uct_rc_txqp_available_set(&self->super.tx.dcis[i], self->super.super.config.tx_qp_len);
    }
    
    uct_worker_progress_register(worker, uct_dc_verbs_iface_progress, self);
    ucs_debug("created iface %p", self);
    return UCS_OK;
out:
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_dc_verbs_iface_t)
{
    ucs_trace_func("");
    uct_worker_progress_unregister(self->super.super.super.super.worker,
                                   uct_dc_verbs_iface_progress, self);
    ucs_free(self->dcis_txcnt);
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
}

UCS_CLASS_DEFINE(uct_dc_verbs_iface_t, uct_dc_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_dc_verbs_iface_t, uct_iface_t, uct_pd_h,
                                 uct_worker_h, const char*, size_t,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_dc_verbs_iface_t, uct_iface_t);

static
ucs_status_t uct_dc_verbs_query_resources(uct_pd_h pd,
                                          uct_tl_resource_desc_t **resources_p,
                                          unsigned *num_resources_p)
{
    return uct_dc_device_query_tl_resources(&ucs_derived_of(pd, uct_ib_pd_t)->dev,
                                            "dc", 0,
                                            resources_p, num_resources_p);
}


UCT_TL_COMPONENT_DEFINE(uct_dc_verbs_tl,
                        uct_dc_verbs_query_resources,
                        uct_dc_verbs_iface_t,
                        "dc",
                        "DC_VERBS_",
                        uct_dc_iface_config_table,
                        uct_dc_iface_config_t);
UCT_PD_REGISTER_TL(&uct_ib_pdc, &uct_dc_verbs_tl);

