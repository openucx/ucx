/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ud_iface.h"
#include "ud_ep.h"
#include "ud_inl.h"

#include <ucs/arch/cpu.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/queue.h>
#include <sys/poll.h>
#include <linux/ip.h>


#define UCT_UD_IPV4_ADDR_LEN sizeof(struct in_addr)
#define UCT_UD_IPV6_ADDR_LEN sizeof(struct in6_addr)

#if ENABLE_STATS
static ucs_stats_class_t uct_ud_iface_stats_class = {
    .name = "ud_iface",
    .num_counters = UCT_UD_IFACE_STAT_LAST,
    .counter_names = {
        [UCT_UD_IFACE_STAT_RX_DROP] = "rx_drop"
    }
};
#endif

SGLIB_DEFINE_LIST_FUNCTIONS(uct_ud_iface_peer_t, uct_ud_iface_peer_cmp, next)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_ud_iface_peer_t,
                                        UCT_UD_HASH_SIZE,
                                        uct_ud_iface_peer_hash)

static void uct_ud_iface_free_resend_skbs(uct_ud_iface_t *iface);
static void uct_ud_iface_timer(int timer_id, void *arg);

static void uct_ud_iface_free_pending_rx(uct_ud_iface_t *iface);
static void uct_ud_iface_free_async_comps(uct_ud_iface_t *iface);


void uct_ud_iface_cep_init(uct_ud_iface_t *iface)
{
    sglib_hashed_uct_ud_iface_peer_t_init(iface->peers);
}

static void
uct_ud_iface_cep_cleanup_eps(uct_ud_iface_t *iface, uct_ud_iface_peer_t *peer)
{
    uct_ud_ep_t *ep, *tmp;

    ucs_list_for_each_safe(ep, tmp, &peer->ep_list, cep_list) {
        if (ep->conn_id < peer->conn_id_last) {
            /* active connection should already be cleaned by owner */
            ucs_warn("iface (%p) peer (qpn=%d lid=%d) cleanup with %d endpoints still active",
                     iface, peer->dst_qpn, peer->dlid,
                     (int)ucs_list_length(&peer->ep_list));
            continue;
        }
        ucs_list_del(&ep->cep_list);
        ucs_trace("cep:ep_destroy(%p) conn_id %d", ep, ep->conn_id);
        uct_ep_destroy(&ep->super.super);
    }
}

void uct_ud_iface_cep_cleanup(uct_ud_iface_t *iface)
{
    uct_ud_iface_peer_t *peer;
    struct sglib_hashed_uct_ud_iface_peer_t_iterator it_peer;

    for (peer = sglib_hashed_uct_ud_iface_peer_t_it_init(&it_peer,
                                                         iface->peers);
         peer != NULL;
         peer = sglib_hashed_uct_ud_iface_peer_t_it_next(&it_peer)) {

        uct_ud_iface_cep_cleanup_eps(iface, peer);
        free(peer);
    }
}

static uct_ud_iface_peer_t *
uct_ud_iface_cep_lookup_addr(uct_ud_iface_t *iface, uint16_t dlid,
                             const union ibv_gid *dgid, uint32_t dest_qpn)
{
    uct_ud_iface_peer_t key;
    key.dlid    = dlid;
    key.dgid    = *dgid;
    key.dst_qpn = dest_qpn;
    return sglib_hashed_uct_ud_iface_peer_t_find_member(iface->peers, &key);
}

static uct_ud_iface_peer_t *
uct_ud_iface_cep_lookup_peer(uct_ud_iface_t *iface,
                             const uct_ib_address_t *src_ib_addr,
                             const uct_ud_iface_addr_t *src_if_addr)
{
    uint32_t dest_qpn = uct_ib_unpack_uint24(src_if_addr->qp_num);
    union ibv_gid dgid;
    uint8_t is_global;
    uint16_t dlid;

    uct_ib_address_unpack(src_ib_addr, &dlid, &is_global, &dgid);
    return uct_ud_iface_cep_lookup_addr(iface, dlid, &dgid, dest_qpn);
}

static uct_ud_ep_t *
uct_ud_iface_cep_lookup_ep(uct_ud_iface_peer_t *peer, uint32_t conn_id)
{
    uint32_t id;
    uct_ud_ep_t *ep;

    if (conn_id != UCT_UD_EP_CONN_ID_MAX) {
        id = conn_id;
    } else {
        id = peer->conn_id_last;
        /* TODO: O(1) lookup in this case (new connection) */
    }
    ucs_list_for_each(ep, &peer->ep_list, cep_list) {
        if (ep->conn_id == id) {
            return ep;
        }
        if (ep->conn_id < id) {
            break;
        }
    }
    return NULL;
}

static uint32_t
uct_ud_iface_cep_getid(uct_ud_iface_peer_t *peer, uint32_t conn_id)
{
    uint32_t new_id;

    if (conn_id != UCT_UD_EP_CONN_ID_MAX) {
        return conn_id;
    }
    new_id = peer->conn_id_last++;
    return new_id;
}

/* insert new ep that is connected to src_if_addr */
ucs_status_t uct_ud_iface_cep_insert(uct_ud_iface_t *iface,
                                     const uct_ib_address_t *src_ib_addr,
                                     const uct_ud_iface_addr_t *src_if_addr,
                                     uct_ud_ep_t *ep, uint32_t conn_id)
{
    uint32_t dest_qpn = uct_ib_unpack_uint24(src_if_addr->qp_num);
    uct_ud_iface_peer_t *peer;
    union ibv_gid dgid;
    uint8_t is_global;
    uct_ud_ep_t *cep;
    uint16_t dlid;

    uct_ib_address_unpack(src_ib_addr, &dlid, &is_global, &dgid);
    peer = uct_ud_iface_cep_lookup_addr(iface, dlid, &dgid, dest_qpn);
    if (peer == NULL) {
        peer = malloc(sizeof *peer);
        if (peer == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        peer->dlid    = dlid;
        peer->dgid    = dgid;
        peer->dst_qpn = dest_qpn;
        sglib_hashed_uct_ud_iface_peer_t_add(iface->peers, peer);
        ucs_list_head_init(&peer->ep_list);
        peer->conn_id_last = 0;
    }

    ep->conn_id = uct_ud_iface_cep_getid(peer, conn_id);
    if (ep->conn_id == UCT_UD_EP_CONN_ID_MAX) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (ucs_list_is_empty(&peer->ep_list)) {
            ucs_list_add_head(&peer->ep_list, &ep->cep_list);
            return UCS_OK;
    }
    ucs_list_for_each(cep, &peer->ep_list, cep_list) {
        ucs_assert_always(cep->conn_id != ep->conn_id);
        if (cep->conn_id < ep->conn_id) {
            ucs_list_insert_before(&cep->cep_list, &ep->cep_list);
            return UCS_OK;
        }
    }
    return UCS_OK;
}

void uct_ud_iface_cep_remove(uct_ud_ep_t *ep)
{
  if (ucs_list_is_empty(&ep->cep_list)) {
      return;
  }
  ucs_trace("iface(%p) cep_remove:ep(%p)", ep->super.super.iface, ep);
  ucs_list_del(&ep->cep_list);
  ucs_list_head_init(&ep->cep_list);
}

uct_ud_ep_t *uct_ud_iface_cep_lookup(uct_ud_iface_t *iface,
                                     const uct_ib_address_t *src_ib_addr,
                                     const uct_ud_iface_addr_t *src_if_addr,
                                     uint32_t conn_id)
{
    uct_ud_iface_peer_t *peer;
    uct_ud_ep_t *ep;

    peer = uct_ud_iface_cep_lookup_peer(iface, src_ib_addr, src_if_addr);
    if (peer == NULL) {
        return NULL;
    }

    ep = uct_ud_iface_cep_lookup_ep(peer, conn_id);
    if (ep && conn_id == UCT_UD_EP_CONN_ID_MAX) {
        peer->conn_id_last++;
    }
    return ep;
}

void uct_ud_iface_cep_rollback(uct_ud_iface_t *iface,
                               const uct_ib_address_t *src_ib_addr,
                               const uct_ud_iface_addr_t *src_if_addr,
                               uct_ud_ep_t *ep)
{
    uct_ud_iface_peer_t *peer;

    peer = uct_ud_iface_cep_lookup_peer(iface, src_ib_addr, src_if_addr);
    ucs_assert_always(peer != NULL);
    ucs_assert_always(peer->conn_id_last > 0);
    ucs_assert_always(ep->conn_id + 1 == peer->conn_id_last);
    ucs_assert_always(!ucs_list_is_empty(&peer->ep_list));
    ucs_assert_always(!ucs_list_is_empty(&ep->cep_list));

    peer->conn_id_last--;
    uct_ud_iface_cep_remove(ep);
}

static void uct_ud_iface_send_skb_init(uct_iface_h tl_iface, void *obj,
                                       uct_mem_h memh)
{
    uct_ud_send_skb_t *skb = obj;
    uct_ib_mem_t *ib_memh = memh;

    skb->lkey  = ib_memh->lkey;
    skb->flags = 0;
}

static ucs_status_t
uct_ud_iface_create_qp(uct_ud_iface_t *self, const uct_ud_iface_config_t *config)
{
    /* TODO: exp attrs autoconf */
    struct ibv_exp_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    int ret;

    /* Create QP */
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_context          = NULL;
    qp_init_attr.send_cq             = self->super.cq[UCT_IB_DIR_TX];
    qp_init_attr.recv_cq             = self->super.cq[UCT_IB_DIR_RX];
    qp_init_attr.srq                 = NULL; /* TODO */
    qp_init_attr.qp_type             = IBV_QPT_UD;
    qp_init_attr.sq_sig_all          = 0;

    /* TODO: cap setting */
    qp_init_attr.cap.max_send_wr     = config->super.tx.queue_len;
    qp_init_attr.cap.max_recv_wr     = config->super.rx.queue_len;
    qp_init_attr.cap.max_send_sge    = 2;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_inline_data = ucs_max(config->super.tx.min_inline,
                                               UCT_UD_MIN_INLINE);

#if HAVE_VERBS_EXP_H
    qp_init_attr.pd                  = uct_ib_iface_qp_pd(&self->super);
    qp_init_attr.comp_mask           = IBV_QP_INIT_ATTR_PD;
#if HAVE_IBV_EXP_RES_DOMAIN
    if (self->super.res_domain != NULL) {
        qp_init_attr.comp_mask      |= IBV_EXP_QP_INIT_ATTR_RES_DOMAIN;
        qp_init_attr.res_domain      = self->super.res_domain->ibv_domain;
    }
    #endif

    /* TODO: inline rcv */
#if 0
    if (mxm_ud_ep_opts(ep)->ud.ib.rx.max_inline > 0) {
        qp_init_attr.comp_mask      |= IBV_EXP_QP_INIT_ATTR_INL_RECV;
        qp_init_attr.max_inl_recv    = mxm_ud_ep_opts(ep)->ud.ib.rx.max_inline;
    }
#endif
    self->qp = ibv_exp_create_qp(uct_ib_iface_device(&self->super)->ibv_context,
                                 &qp_init_attr);
#else
    self->qp = ibv_exp_create_qp(uct_ib_iface_qp_pd(&self->super), &qp_init_attr);
#endif
    if (self->qp == NULL) {
        ucs_error("Failed to create qp: %s [inline: %u rsge: %u ssge: %u rwr: %u swr: %u]",
                  strerror(errno),
                  qp_init_attr.cap.max_inline_data, qp_init_attr.cap.max_recv_sge,
                  qp_init_attr.cap.max_send_sge, qp_init_attr.cap.max_recv_wr,
                  qp_init_attr.cap.max_send_wr);
        goto err;
    }

    self->config.max_inline = qp_init_attr.cap.max_inline_data;
    ucs_assert_always(qp_init_attr.cap.max_inline_data >= UCT_UD_MIN_INLINE);
    uct_ib_iface_set_max_iov(&self->super, qp_init_attr.cap.max_send_sge);

    memset(&qp_attr, 0, sizeof(qp_attr));
    /* Modify QP to INIT state */
    qp_attr.qp_state   = IBV_QPS_INIT;
    qp_attr.pkey_index = self->super.pkey_index;
    qp_attr.port_num   = self->super.config.port_num;
    qp_attr.qkey       = UCT_IB_KEY;
    ret = ibv_modify_qp(self->qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
    if (ret) {
        ucs_error("Failed to modify UD QP to INIT: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTR */
    qp_attr.qp_state = IBV_QPS_RTR;
    ret = ibv_modify_qp(self->qp, &qp_attr, IBV_QP_STATE);
    if (ret) {
        ucs_error("Failed to modify UD QP to RTR: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTS */
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.sq_psn = 0;
    ret = ibv_modify_qp(self->qp, &qp_attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
    if (ret) {
        ucs_error("Failed to modify UD QP to RTS: %m");
        goto err_destroy_qp;
    }

    ucs_debug("iface=%p: created qp 0x%x on %s:%d max_send_wr %u max_recv_wr %u "
              "max_inline %u",
              self, self->qp->qp_num,
              uct_ib_device_name(uct_ib_iface_device(&self->super)),
              self->super.config.port_num,
              qp_init_attr.cap.max_send_wr,
              qp_init_attr.cap.max_recv_wr,
              qp_init_attr.cap.max_inline_data);

    return UCS_OK;
err_destroy_qp:
    ibv_destroy_qp(self->qp);
err:
    return UCS_ERR_INVALID_PARAM;
}

ucs_status_t uct_ud_iface_complete_init(uct_ud_iface_t *iface)
{
    ucs_async_context_t *async = iface->super.super.worker->async;
    ucs_async_mode_t async_mode = async->mode;
    ucs_status_t status;

    iface->tx.resend_skbs_quota = iface->tx.available;

    /* TODO: make tick configurable */
    iface->async.slow_tick = ucs_time_from_msec(100);
    status = ucs_twheel_init(&iface->async.slow_timer,
                             iface->async.slow_tick / 4,
                             uct_ud_iface_get_async_time(iface));
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_async_add_timer(async_mode, iface->async.slow_tick,
                                 uct_ud_iface_timer, iface, async,
                                 &iface->async.timer_id);
    if (status != UCS_OK) {
        goto err_twheel_cleanup;
    }

    return UCS_OK;

err_twheel_cleanup:
    ucs_twheel_cleanup(&iface->async.slow_timer);
err:
    return status;
}

void uct_ud_iface_remove_async_handlers(uct_ud_iface_t *iface)
{
    uct_base_iface_progress_disable(&iface->super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);
    ucs_async_remove_handler(iface->async.timer_id, 1);
}

/* Calculate real GIDs len. Can be either 16 (RoCEv1 or RoCEv2/IPv6)
 * or 4 (RoCEv2/IPv4). This len is used for packets filtering by DGIDs.
 *
 * According to Annex17_RoCEv2 (A17.4.5.2):
 * "The first 40 bytes of user posted UD Receive Buffers are reserved for the L3
 * header of the incoming packet (as per the InfiniBand Spec Section 11.4.1.2).
 * In RoCEv2, this area is filled up with the IP header. IPv6 header uses the
 * entire 40 bytes. IPv4 headers use the 20 bytes in the second half of the
 * reserved 40 bytes area (i.e. offset 20 from the beginning of the receive
 * buffer). In this case, the content of the first 20 bytes is undefined." */
static void uct_ud_iface_calc_gid_len(uct_ud_iface_t *iface)
{
    uint16_t *local_gid_u16 = (uint16_t*)iface->super.gid.raw;

    /* Make sure that daddr in IPv4 resides in the last 4 bytes in GRH */
    UCS_STATIC_ASSERT((UCT_IB_GRH_LEN - (20 + offsetof(struct iphdr, daddr))) ==
                      UCT_UD_IPV4_ADDR_LEN);

    /* Make sure that dgid resides in the last 16 bytes in GRH */
    UCS_STATIC_ASSERT((UCT_IB_GRH_LEN - offsetof(struct ibv_grh, dgid)) ==
                      UCT_UD_IPV6_ADDR_LEN);

    /* IPv4 mapped to IPv6 looks like: 0000:0000:0000:0000:0000:ffff:????:????,
     * so check for leading zeroes and verify that 11-12 bytes are 0xff.
     * Otherwise either RoCEv1 or RoCEv2/IPv6 are used. */
    if (local_gid_u16[0] == 0x0000) {
        ucs_assert_always(local_gid_u16[5] == 0xffff);
        iface->config.gid_len = UCT_UD_IPV4_ADDR_LEN;
    } else {
        iface->config.gid_len = UCT_UD_IPV6_ADDR_LEN;
    }
}

UCS_CLASS_INIT_FUNC(uct_ud_iface_t, uct_ud_iface_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_ud_iface_config_t *config,
                    uct_ib_iface_init_attr_t *init_attr)
{
    ucs_status_t status;
    size_t data_size;
    int mtu;

    ucs_trace_func("%s: iface=%p ops=%p worker=%p rx_headroom=%zu",
                   params->mode.device.dev_name, self, ops, worker,
                   params->rx_headroom);

    if (config->super.tx.queue_len <= UCT_UD_TX_MODERATION) {
        ucs_error("%s ud iface tx queue is too short (%d <= %d)",
                  params->mode.device.dev_name,
                  config->super.tx.queue_len, UCT_UD_TX_MODERATION);
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_ib_device_mtu(params->mode.device.dev_name, md, &mtu);
    if (status != UCS_OK) {
        return status;
    }

    init_attr->rx_priv_len = sizeof(uct_ud_recv_skb_t) -
                             sizeof(uct_ib_iface_recv_desc_t);
    init_attr->rx_hdr_len  = UCT_IB_GRH_LEN + sizeof(uct_ud_neth_t);
    init_attr->tx_cq_len   = config->super.tx.queue_len;
    init_attr->rx_cq_len   = config->super.rx.queue_len;
    init_attr->seg_size    = ucs_min(mtu, config->super.super.max_bcopy);

    UCS_CLASS_CALL_SUPER_INIT(uct_ib_iface_t, &ops->super, md, worker,
                              params, &config->super, init_attr);

    if (self->super.super.worker->async == NULL) {
        ucs_error("%s ud iface must have valid async context", params->mode.device.dev_name);
        return UCS_ERR_INVALID_PARAM;
    }

    self->tx.unsignaled          = 0;
    self->tx.available           = config->super.tx.queue_len;

    self->rx.available           = config->super.rx.queue_len;
    self->rx.quota               = 0;
    self->config.tx_qp_len       = config->super.tx.queue_len;
    self->config.peer_timeout    = ucs_time_from_sec(config->peer_timeout);
    self->config.check_grh_dgid  = (config->dgid_check &&
                                    (self->super.addr_type == UCT_IB_ADDRESS_TYPE_ETH));

    if (config->slow_timer_backoff <= 0.) {
        ucs_error("The slow timer back off should be > 0 (%lf)",
                  config->slow_timer_backoff);
        return UCS_ERR_INVALID_PARAM;
    } else {
        self->config.slow_timer_backoff = config->slow_timer_backoff;
    }

    /* Redefine receive desc release callback */
    self->super.release_desc.cb  = uct_ud_iface_release_desc;

    UCT_UD_IFACE_HOOK_INIT(self);

    if (uct_ud_iface_create_qp(self, config) != UCS_OK) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_ptr_array_init(&self->eps, 0, "ud_eps");
    uct_ud_iface_cep_init(self);

    status = uct_ib_iface_recv_mpool_init(&self->super, &config->super,
                                          "ud_recv_skb", &self->rx.mp);
    if (status != UCS_OK) {
        goto err_qp;
    }

    self->rx.available = ucs_min(config->ud_common.rx_queue_len_init,
                                 config->super.rx.queue_len);
    self->rx.quota     = config->super.rx.queue_len - self->rx.available;
    ucs_mpool_grow(&self->rx.mp, self->rx.available);

    data_size = sizeof(uct_ud_ctl_hdr_t) + self->super.addr_size;
    data_size = ucs_max(data_size, self->super.config.seg_size);
    data_size = ucs_max(data_size, sizeof(uct_ud_zcopy_desc_t) + self->config.max_inline);

    status = uct_iface_mpool_init(&self->super.super, &self->tx.mp,
                                  sizeof(uct_ud_send_skb_t) + data_size,
                                  sizeof(uct_ud_send_skb_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->super.tx.mp, self->config.tx_qp_len,
                                  uct_ud_iface_send_skb_init, "ud_tx_skb");
    if (status != UCS_OK) {
        goto err_rx_mpool;
    }

    self->tx.skb = NULL;
    self->tx.skb_inl.super.len = sizeof(uct_ud_neth_t);

    ucs_queue_head_init(&self->tx.resend_skbs);
    self->tx.resend_skbs_quota = 0;

    ucs_arbiter_init(&self->tx.pending_q);

    ucs_queue_head_init(&self->tx.async_comp_q);

    ucs_queue_head_init(&self->rx.pending_q);

    self->tx.async_before_pending = 0;

    uct_ud_iface_calc_gid_len(self);

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_ud_iface_stats_class,
                                  self->super.super.stats);
    if (status != UCS_OK) {
        goto err_tx_mpool;
    }

    return UCS_OK;

err_tx_mpool:
    ucs_mpool_cleanup(&self->tx.mp, 1);
err_rx_mpool:
    ucs_mpool_cleanup(&self->rx.mp, 1);
err_qp:
    ibv_destroy_qp(self->qp);
    ucs_ptr_array_cleanup(&self->eps);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_iface_t)
{
    ucs_trace_func("");

    /* TODO: proper flush and connection termination */
    uct_ud_enter(self);
    ucs_debug("iface(%p): cep cleanup", self);
    uct_ud_iface_cep_cleanup(self);
    uct_ud_iface_free_resend_skbs(self);
    uct_ud_iface_free_async_comps(self);
    ucs_mpool_cleanup(&self->tx.mp, 0);
    /* TODO: qp to error state and cleanup all wqes */
    uct_ud_iface_free_pending_rx(self);
    ucs_mpool_cleanup(&self->rx.mp, 0);
    ibv_destroy_qp(self->qp);
    ucs_debug("iface(%p): ptr_array cleanup", self);
    ucs_ptr_array_cleanup(&self->eps);
    ucs_arbiter_cleanup(&self->tx.pending_q);
    UCS_STATS_NODE_FREE(self->stats);
    uct_ud_leave(self);
}

UCS_CLASS_DEFINE(uct_ud_iface_t, uct_ib_iface_t);

ucs_config_field_t uct_ud_iface_config_table[] = {
    {"IB_", "", NULL,
     ucs_offsetof(uct_ud_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

    {"", "", NULL,
     ucs_offsetof(uct_ud_iface_config_t, ud_common),
     UCS_CONFIG_TYPE_TABLE(uct_ud_iface_common_config_table)},

    {"TIMEOUT", "5.0m", "Transport timeout",
     ucs_offsetof(uct_ud_iface_config_t, peer_timeout), UCS_CONFIG_TYPE_TIME},
    {"SLOW_TIMER_BACKOFF", "2.0", "Timeout multiplier for resending trigger",
     ucs_offsetof(uct_ud_iface_config_t, slow_timer_backoff),
                  UCS_CONFIG_TYPE_DOUBLE},
    {"ETH_DGID_CHECK", "y",
     "Enable checking destination GID for incoming packets of Ethernet network\n"
     "Mismatched packets are silently dropped.",
     ucs_offsetof(uct_ud_iface_config_t, dgid_check), UCS_CONFIG_TYPE_BOOL},
    {NULL}
};


ucs_status_t uct_ud_iface_query(uct_ud_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    ucs_status_t status;

    status = uct_ib_iface_query(&iface->super,
                                UCT_IB_DETH_LEN + sizeof(uct_ud_neth_t),
                                iface_attr);
    if (status != UCS_OK) {
        return status;
    }

    iface_attr->cap.flags              = UCT_IFACE_FLAG_AM_SHORT         |
                                         UCT_IFACE_FLAG_AM_BCOPY         |
                                         UCT_IFACE_FLAG_AM_ZCOPY         |
                                         UCT_IFACE_FLAG_CONNECT_TO_EP    |
                                         UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                         UCT_IFACE_FLAG_PENDING          |
                                         UCT_IFACE_FLAG_CB_SYNC          |
                                         UCT_IFACE_FLAG_CB_ASYNC         |
                                         UCT_IFACE_FLAG_EVENT_SEND_COMP  |
                                         UCT_IFACE_FLAG_EVENT_RECV       |
                                         UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    iface_attr->cap.am.max_short       = iface->config.max_inline - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.max_bcopy       = iface->super.config.seg_size - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.min_zcopy       = 0;
    iface_attr->cap.am.max_zcopy       = iface->super.config.seg_size - sizeof(uct_ud_neth_t);
    iface_attr->cap.am.align_mtu       = uct_ib_mtu_value(uct_ib_iface_port_attr(&iface->super)->active_mtu);
    iface_attr->cap.am.opt_zcopy_align = UCS_SYS_PCI_MAX_PAYLOAD;
    iface_attr->cap.am.max_hdr         = iface->config.max_inline - sizeof(uct_ud_neth_t);
    /* The first iov is reserved for the header */
    iface_attr->cap.am.max_iov         = uct_ib_iface_get_max_iov(&iface->super) - 1;

    iface_attr->cap.put.max_short      = iface->config.max_inline -
                                         sizeof(uct_ud_neth_t) - sizeof(uct_ud_put_hdr_t);

    iface_attr->iface_addr_len         = sizeof(uct_ud_iface_addr_t);
    iface_attr->ep_addr_len            = sizeof(uct_ud_ep_addr_t);
    iface_attr->max_conn_priv          = 0;

    /* UD lacks of scatter to CQE support */
    iface_attr->latency.overhead      += 10e-9;

    return UCS_OK;
}

ucs_status_t
uct_ud_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    uct_ud_iface_addr_t *addr = (uct_ud_iface_addr_t *)iface_addr;

    uct_ib_pack_uint24(addr->qp_num, iface->qp->qp_num);

    return UCS_OK;
}

ucs_status_t uct_ud_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                uct_completion_t *comp)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    uct_ud_ep_t *ep;
    ucs_status_t status;
    int i, count;

    ucs_trace_func("");

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    uct_ud_enter(iface);

    if (ucs_unlikely(uct_ud_iface_has_pending_async_ev(iface))) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super);
        uct_ud_leave(iface);
        return UCS_INPROGRESS;
    }

    count = 0;
    ucs_ptr_array_for_each(ep, i, &iface->eps) {
        /* ud ep flush returns either ok or in progress */
        status = uct_ud_ep_flush_nolock(iface, ep, NULL);
        if ((status == UCS_INPROGRESS) || (status == UCS_ERR_NO_RESOURCE)) {
            ++count;
        }
    }

    uct_ud_leave(iface);
    if (count != 0) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(&iface->super.super);
        return UCS_INPROGRESS;
    }

    UCT_TL_IFACE_STAT_FLUSH(&iface->super.super);
    return UCS_OK;
}

void uct_ud_iface_add_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    uint32_t prev_gen;
    ep->ep_id = ucs_ptr_array_insert(&iface->eps, ep, &prev_gen);
}

void uct_ud_iface_remove_ep(uct_ud_iface_t *iface, uct_ud_ep_t *ep)
{
    if (ep->ep_id != UCT_UD_EP_NULL_ID) {
        ucs_trace("iface(%p) remove ep: %p id %d", iface, ep, ep->ep_id);
        ucs_ptr_array_remove(&iface->eps, ep->ep_id, 0);
    }
}

void uct_ud_iface_replace_ep(uct_ud_iface_t *iface,
                             uct_ud_ep_t *old_ep, uct_ud_ep_t *new_ep)
{
    void *p;
    ucs_assert_always(old_ep != new_ep);
    ucs_assert_always(old_ep->ep_id != new_ep->ep_id);
    p = ucs_ptr_array_replace(&iface->eps, old_ep->ep_id, new_ep);
    ucs_assert_always(p == (void *)old_ep);
    ucs_trace("replace_ep: old(%p) id=%d new(%p) id=%d", old_ep, old_ep->ep_id, new_ep, new_ep->ep_id);
    ucs_ptr_array_remove(&iface->eps, new_ep->ep_id, 0);
}


uct_ud_send_skb_t *uct_ud_iface_resend_skb_get(uct_ud_iface_t *iface)
{
    ucs_queue_elem_t *elem;
    uct_ud_send_skb_t *skb;

    /* grow reserved skb's queue on-demand */
    if (iface->tx.resend_skbs_quota > 0) {
        skb = ucs_mpool_get(&iface->tx.mp);
        if (skb == NULL) {
            ucs_fatal("failed to allocate control skb");
        }
        --iface->tx.resend_skbs_quota;
        return skb;
    } else {
        elem = ucs_queue_pull(&iface->tx.resend_skbs);
        ucs_assert(elem != NULL);
        return ucs_container_of(elem, uct_ud_send_skb_t, queue);
    }
}


static void uct_ud_iface_free_resend_skbs(uct_ud_iface_t *iface)
{
    uct_ud_send_skb_t *skb;

    iface->tx.resend_skbs_quota = 0;
    ucs_queue_for_each_extract(skb, &iface->tx.resend_skbs, queue, 1) {
        ucs_mpool_put(skb);
    }
}

static void uct_ud_ep_dispatch_err_comp(uct_ud_ep_t *ep, uct_ud_send_skb_t *skb)
{
    uct_ud_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ud_iface_t);
    ucs_status_t status;

    ucs_assert(ep->tx.err_skb_count > 0);
    --ep->tx.err_skb_count;

    if ((ep->tx.err_skb_count > 0) || (ep->flags & UCT_UD_EP_FLAG_DISCONNECTED)) {
        return;
    }

    if (ep->flags & UCT_UD_EP_FLAG_PRIVATE) {
        uct_ep_destroy(&ep->super.super);
        return;
    }

    status = iface->super.ops->set_ep_failed(&iface->super, &ep->super.super,
                                             skb->status);
    if (status != UCS_OK) {
        ucs_fatal("transport error: %s", ucs_status_string(status));
    }
}

void uct_ud_iface_dispatch_async_comps_do(uct_ud_iface_t *iface)
{
    uct_ud_comp_desc_t *cdesc;
    uct_ud_send_skb_t  *skb;
    uct_ud_ep_t *ep;

    do {
        skb = ucs_queue_pull_elem_non_empty(&iface->tx.async_comp_q,
                                            uct_ud_send_skb_t, queue);
        cdesc = uct_ud_comp_desc(skb);
        ep    = cdesc->ep;

        if (skb->flags & UCT_UD_SEND_SKB_FLAG_COMP) {
            ucs_assert(!(ep->flags & UCT_UD_EP_FLAG_DISCONNECTED));
            uct_invoke_completion(cdesc->comp, skb->status);
        }

        if (ucs_unlikely(skb->flags & UCT_UD_SEND_SKB_FLAG_ERR)) {
            uct_ud_ep_dispatch_err_comp(ep, skb);
        }

        ep->flags &= ~UCT_UD_EP_FLAG_ASYNC_COMPS;
        skb->flags = 0;
        ucs_mpool_put(skb);
    } while (!ucs_queue_is_empty(&iface->tx.async_comp_q));
}

static void uct_ud_iface_free_async_comps(uct_ud_iface_t *iface)
{
    uct_ud_send_skb_t *skb;

    while (!ucs_queue_is_empty(&iface->tx.async_comp_q)) {
        skb = ucs_queue_pull_elem_non_empty(&iface->tx.async_comp_q,
                                            uct_ud_send_skb_t, queue);
        ucs_mpool_put(skb);
    }
}

ucs_status_t uct_ud_iface_dispatch_pending_rx_do(uct_ud_iface_t *iface)
{
    int count;
    uct_ud_recv_skb_t *skb;
    uct_ud_neth_t *neth;
    unsigned max_poll = iface->super.config.rx_max_poll;

    count = 0;
    do {
        skb = ucs_queue_pull_elem_non_empty(&iface->rx.pending_q, uct_ud_recv_skb_t, u.am.queue);
        neth =  (uct_ud_neth_t *)((char *)uct_ib_iface_recv_desc_hdr(&iface->super,
                                                                     (uct_ib_iface_recv_desc_t *)skb) +
                                  UCT_IB_GRH_LEN);
        uct_ib_iface_invoke_am_desc(&iface->super,
                                    uct_ud_neth_get_am_id(neth),
                                    neth + 1,
                                    skb->u.am.len,
                                    &skb->super);
        count++;
        if (count >= max_poll) {
            return UCS_ERR_NO_RESOURCE;
        }
    } while (!ucs_queue_is_empty(&iface->rx.pending_q));

    return UCS_OK;
}

static void uct_ud_iface_free_pending_rx(uct_ud_iface_t *iface)
{
    uct_ud_recv_skb_t *skb;

    while (!ucs_queue_is_empty(&iface->rx.pending_q)) {
        skb = ucs_queue_pull_elem_non_empty(&iface->rx.pending_q, uct_ud_recv_skb_t, u.am.queue);
        ucs_mpool_put(skb);
    }
}

static inline void uct_ud_iface_async_progress(uct_ud_iface_t *iface)
{
    unsigned ev_count;
    uct_ud_iface_ops_t *ops;

    ops = ucs_derived_of(iface->super.ops, uct_ud_iface_ops_t);
    ev_count = ops->async_progress(iface);
    if (ev_count > 0) {
        uct_ud_iface_raise_pending_async_ev(iface);
    }
}

static void uct_ud_iface_timer(int timer_id, void *arg)
{
    uct_ud_iface_t *iface = arg;
    ucs_time_t now;

    uct_ud_enter(iface);
    now = uct_ud_iface_get_async_time(iface);
    ucs_trace_async("iface(%p) slow_timer_sweep: now %lu", iface, now);
    ucs_twheel_sweep(&iface->async.slow_timer, now);
    uct_ud_iface_async_progress(iface);
    uct_ud_leave(iface);
}

void uct_ud_iface_release_desc(uct_recv_desc_t *self, void *desc)
{
    uct_ud_iface_t *iface = ucs_container_of(self,
                                             uct_ud_iface_t, super.release_desc);

    uct_ud_enter(iface);
    uct_ib_iface_release_desc(self, desc);
    uct_ud_leave(iface);
}

void uct_ud_iface_handle_failure(uct_ib_iface_t *iface, void *arg,
                                 ucs_status_t status)
{
    uct_ud_tx_wnd_purge_outstanding(ucs_derived_of(iface, uct_ud_iface_t),
                                    (uct_ud_ep_t *)arg, status);
}

ucs_status_t uct_ud_iface_event_arm(uct_iface_h tl_iface, unsigned events)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);
    ucs_status_t status;

    uct_ud_enter(iface);

    status = uct_ib_iface_pre_arm(&iface->super);
    if (status != UCS_OK) {
        goto out;
    }

    /* Check if some receives were not delivered yet */
    if ((events & (UCT_EVENT_RECV | UCT_EVENT_RECV_SIG)) &&
        !ucs_queue_is_empty(&iface->rx.pending_q))
    {
        status = UCS_ERR_BUSY;
        goto out;
    }

    /* Check if some send completions were not delivered yet */
    if ((events & UCT_EVENT_SEND_COMP) &&
        !ucs_queue_is_empty(&iface->tx.async_comp_q))
    {
        status = UCS_ERR_BUSY;
        goto out;
    }

    if (events & UCT_EVENT_SEND_COMP) {
        status = iface->super.ops->arm_cq(&iface->super, UCT_IB_DIR_TX, 0);
        if (status != UCS_OK) {
            goto out;
        }
    }

    if (events & (UCT_EVENT_SEND_COMP | UCT_EVENT_RECV)) {
        /* we may get send completion through ACKs as well */
        status = iface->super.ops->arm_cq(&iface->super, UCT_IB_DIR_RX, 0);
        if (status != UCS_OK) {
            goto out;
        }
    }

    status = UCS_OK;
out:
    uct_ud_leave(iface);
    return status;
}

void uct_ud_iface_progress_enable(uct_iface_h tl_iface, unsigned flags)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);

    if (flags & UCT_PROGRESS_RECV) {
        uct_ud_enter(iface);
        iface->rx.available += iface->rx.quota;
        iface->rx.quota      = 0;
        /* let progress (possibly async) post the missing receives */
        uct_ud_leave(iface);
    }

    uct_base_iface_progress_enable(tl_iface, flags);
}
