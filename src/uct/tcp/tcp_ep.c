/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "tcp.h"

#include <ucs/async/async.h>


/* Forward declarations */
static unsigned uct_tcp_ep_progress_data_tx(void *arg);
static unsigned uct_tcp_ep_progress_data_rx(void *arg);
static unsigned uct_tcp_ep_progress_magic_number_rx(void *arg);
static unsigned uct_tcp_ep_failed_progress(void *arg);

const uct_tcp_cm_state_t uct_tcp_ep_cm_state[] = {
    [UCT_TCP_EP_CONN_STATE_CLOSED] = {
        .name        = "CLOSED",
        .tx_progress = (uct_tcp_ep_progress_t)ucs_empty_function_return_zero,
        .rx_progress = (uct_tcp_ep_progress_t)ucs_empty_function_return_zero
    },
    [UCT_TCP_EP_CONN_STATE_CONNECTING] = {
        .name        = "CONNECTING",
        .tx_progress = uct_tcp_cm_conn_progress,
        .rx_progress = uct_tcp_ep_progress_data_rx
    },
    [UCT_TCP_EP_CONN_STATE_WAITING_ACK] = {
        .name        = "WAITING_ACK",
        .tx_progress = (uct_tcp_ep_progress_t)ucs_empty_function_return_zero,
        .rx_progress = uct_tcp_ep_progress_data_rx
    },
    [UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER] = {
        .name        = "RECV_MAGIC_NUMBER",
        .tx_progress = (uct_tcp_ep_progress_t)ucs_empty_function_return_zero,
        .rx_progress = uct_tcp_ep_progress_magic_number_rx
    },
    [UCT_TCP_EP_CONN_STATE_ACCEPTING] = {
        .name        = "ACCEPTING",
        .tx_progress = (uct_tcp_ep_progress_t)ucs_empty_function_return_zero,
        .rx_progress = uct_tcp_ep_progress_data_rx
    },
    [UCT_TCP_EP_CONN_STATE_CONNECTED] = {
        .name        = "CONNECTED",
        .tx_progress = uct_tcp_ep_progress_data_tx,
        .rx_progress = uct_tcp_ep_progress_data_rx
    }
};

static inline int uct_tcp_ep_ctx_buf_empty(uct_tcp_ep_ctx_t *ctx)
{
    ucs_assert((ctx->length == 0) || (ctx->buf != NULL));

    return ctx->length == 0;
}

static inline int uct_tcp_ep_ctx_buf_need_progress(uct_tcp_ep_ctx_t *ctx)
{
    ucs_assert(ctx->offset <= ctx->length);

    return ctx->offset < ctx->length;
}

static inline ucs_status_t uct_tcp_ep_check_tx_res(uct_tcp_ep_t *ep)
{
    if (ucs_likely((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) &&
                   uct_tcp_ep_ctx_buf_empty(&ep->tx))) {
        return UCS_OK;
    } else if (ucs_unlikely(ep->conn_state == UCT_TCP_EP_CONN_STATE_CLOSED)) {
        return UCS_ERR_CONNECTION_RESET;
    } else if (ucs_unlikely(ep->conn_state ==
                            UCT_TCP_EP_CONN_STATE_ACCEPTING)) {
        ucs_assert((ep->conn_retries == 0) &&
                   !(ep->flags & (UCT_TCP_EP_FLAG_CTX_TYPE_TX |
                                  UCT_TCP_EP_FLAG_CTX_TYPE_RX)) &&
                   (ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));
        return UCS_ERR_NO_RESOURCE;
    }

    ucs_assertv((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
                (ep->conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK) ||
                ((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) &&
                 !uct_tcp_ep_ctx_buf_empty(&ep->tx)),
                "ep=%p", ep);

    uct_tcp_ep_mod_events(ep, UCS_EVENT_SET_EVWRITE, 0);
    return UCS_ERR_NO_RESOURCE;
}

static inline void uct_tcp_ep_ctx_rewind(uct_tcp_ep_ctx_t *ctx)
{
    ctx->offset = 0;
    ctx->length = 0;
}

static inline void uct_tcp_ep_ctx_init(uct_tcp_ep_ctx_t *ctx)
{
    ctx->put_sn = UINT32_MAX;
    ctx->buf    = NULL;
    uct_tcp_ep_ctx_rewind(ctx);
}

static inline void uct_tcp_ep_ctx_reset(uct_tcp_ep_ctx_t *ctx)
{
    ucs_mpool_put_inline(ctx->buf);
    ctx->buf = NULL;
    uct_tcp_ep_ctx_rewind(ctx);
}

static void uct_tcp_ep_addr_cleanup(struct sockaddr_in *sock_addr)
{
    memset(sock_addr, 0, sizeof(*sock_addr));
}

static void uct_tcp_ep_addr_init(struct sockaddr_in *sock_addr,
                                 const struct sockaddr_in *peer_addr)
{
    /* TODO: handle IPv4 and IPv6 */
    if (peer_addr == NULL) {
        uct_tcp_ep_addr_cleanup(sock_addr);
    } else {
        *sock_addr = *peer_addr;
    }
}

int uct_tcp_ep_is_self(const uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    return uct_tcp_iface_is_self_addr(iface, &ep->peer_addr);
}

static void uct_tcp_ep_cleanup(uct_tcp_ep_t *ep)
{
    uct_tcp_ep_addr_cleanup(&ep->peer_addr);

    if (ep->tx.buf != NULL) {
        uct_tcp_ep_ctx_reset(&ep->tx);
    }

    if (ep->rx.buf != NULL) {
        uct_tcp_ep_ctx_reset(&ep->rx);
    }

    uct_tcp_ep_mod_events(ep, 0, ep->events);
    ucs_close_fd(&ep->fd);
    ucs_close_fd(&ep->stale_fd);
}

static void uct_tcp_ep_ptr_map_add(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    ucs_status_t status;

    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP);
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX));

    status = ucs_ptr_map_put(&iface->ep_ptr_map, ep, 1,
                             &ep->cm_id.ptr_map_key);
    ucs_assert_always(status == UCS_OK);

    ep->flags |= UCT_TCP_EP_FLAG_ON_PTR_MAP;
}

static void uct_tcp_ep_ptr_map_del(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    ucs_status_t status;

    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP);
    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_ON_PTR_MAP);
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX));

    status = ucs_ptr_map_del(&iface->ep_ptr_map, ep->cm_id.ptr_map_key);
    ucs_assert_always(status == UCS_OK);

    ep->flags &= ~UCT_TCP_EP_FLAG_ON_PTR_MAP;
}

uct_tcp_ep_t* uct_tcp_ep_ptr_map_retrieve(uct_tcp_iface_t *iface,
                                          ucs_ptr_map_key_t ptr_map_key)
{
    uct_tcp_ep_t *ep;

    ep = ucs_ptr_map_get(&iface->ep_ptr_map, ptr_map_key);
    if (ep != NULL) {
        ucs_assert(ep->flags & UCT_TCP_EP_FLAG_ON_PTR_MAP);
        ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX));
        uct_tcp_ep_ptr_map_del(ep);
    }

    return ep;
}

static UCS_CLASS_INIT_FUNC(uct_tcp_ep_t, uct_tcp_iface_t *iface,
                           int fd, const struct sockaddr_in *dest_addr)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)

    uct_tcp_ep_addr_init(&self->peer_addr, dest_addr);

    uct_tcp_ep_ctx_init(&self->tx);
    uct_tcp_ep_ctx_init(&self->rx);

    self->events        = 0;
    self->conn_retries  = 0;
    self->fd            = fd;
    self->stale_fd      = -1;
    self->flags         = 0;
    self->conn_state    = UCT_TCP_EP_CONN_STATE_CLOSED;
    self->cm_id.conn_sn = UCT_TCP_CM_CONN_SN_MAX;

    ucs_list_head_init(&self->list);
    ucs_queue_head_init(&self->pending_q);
    ucs_queue_head_init(&self->put_comp_q);

    if (self->fd != -1) /* EP is created during accepting a connection */ {
        self->conn_retries++;
    } else if (dest_addr == NULL) {
        /* Since no socket FD and no destination address were specified for
         * new EP, it means that EP is created with CONNECT_TO_EP method */
        self->flags |= UCT_TCP_EP_FLAG_CONNECT_TO_EP;
        uct_tcp_ep_ptr_map_add(self);
    }

    uct_tcp_iface_add_ep(self);

    ucs_debug("tcp_ep %p: created on iface %p, fd %d", self, iface, self->fd);
    return UCS_OK;
}

const char *uct_tcp_ep_ctx_caps_str(uint8_t ep_ctx_caps, char *str_buffer)
{
    ucs_snprintf_zero(str_buffer, UCT_TCP_EP_CTX_CAPS_STR_MAX, "[%s:%s]",
                      (ep_ctx_caps & UCT_TCP_EP_FLAG_CTX_TYPE_TX) ?
                      "Tx" : "-",
                      (ep_ctx_caps & UCT_TCP_EP_FLAG_CTX_TYPE_RX) ?
                      "Rx" : "-");
    return str_buffer;
}

void uct_tcp_ep_change_ctx_caps(uct_tcp_ep_t *ep, uint16_t new_caps)
{
    char str_prev_ctx_caps[UCT_TCP_EP_CTX_CAPS_STR_MAX];
    char str_cur_ctx_caps[UCT_TCP_EP_CTX_CAPS_STR_MAX];

    if (ep->flags != new_caps) {
        ucs_trace("tcp_ep %p: ctx caps changed %s -> %s", ep,
                  uct_tcp_ep_ctx_caps_str(ep->flags, str_prev_ctx_caps),
                  uct_tcp_ep_ctx_caps_str(new_caps, str_cur_ctx_caps));
        ep->flags = new_caps;
    }
}

void uct_tcp_ep_add_ctx_cap(uct_tcp_ep_t *ep, uint16_t ctx_cap)
{
    ucs_assert(ctx_cap & UCT_TCP_EP_CTX_CAPS);
    uct_tcp_ep_change_ctx_caps(ep, ep->flags | ctx_cap);
}

void uct_tcp_ep_remove_ctx_cap(uct_tcp_ep_t *ep, uint16_t ctx_cap)
{
    ucs_assert(ctx_cap & UCT_TCP_EP_CTX_CAPS);
    uct_tcp_ep_change_ctx_caps(ep, ep->flags & ~ctx_cap);
}

void uct_tcp_ep_move_ctx_cap(uct_tcp_ep_t *from_ep, uct_tcp_ep_t *to_ep,
                             uint16_t ctx_cap)
{
    uct_tcp_ep_remove_ctx_cap(from_ep, ctx_cap);
    uct_tcp_ep_add_ctx_cap(to_ep, ctx_cap);
}

static int
uct_tcp_ep_failed_remove_filter(const ucs_callbackq_elem_t *elem, void *arg)
{
    uct_tcp_ep_t *ep = (uct_tcp_ep_t*)arg;

    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_FAILED);
    return (elem->cb == uct_tcp_ep_failed_progress) && (elem->arg == ep);
}

static int
uct_tcp_ep_progress_rx_remove_filter(const ucs_callbackq_elem_t *elem,
                                     void *arg)
{
    uct_tcp_ep_t *ep = (uct_tcp_ep_t*)arg;

    return (elem->cb == uct_tcp_ep_progress_data_rx) && (elem->arg == ep);
}

static UCS_CLASS_CLEANUP_FUNC(uct_tcp_ep_t)
{
    uct_tcp_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                            uct_tcp_iface_t);
    uct_tcp_ep_put_completion_t *put_comp;

    if (self->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX) {
        uct_tcp_cm_remove_ep(iface, self);
    } else {
        uct_tcp_iface_remove_ep(self);
    }

    if (self->flags & UCT_TCP_EP_FLAG_ON_PTR_MAP) {
        uct_tcp_ep_ptr_map_del(self);
    }

    uct_tcp_ep_remove_ctx_cap(self, UCT_TCP_EP_CTX_CAPS);

    ucs_queue_for_each_extract(put_comp, &self->put_comp_q, elem, 1) {
        ucs_free(put_comp);
    }

    if (self->flags & UCT_TCP_EP_FLAG_FAILED) {
        /* a failed EP callback can be still scheduled on the UCT worker,
         * remove it to prevent a callback is being invoked for the
         * destroyed EP */
        ucs_callbackq_remove_if(&iface->super.worker->super.progress_q,
                                uct_tcp_ep_failed_remove_filter, self);
    }

    ucs_callbackq_remove_if(&iface->super.worker->super.progress_q,
                            uct_tcp_ep_progress_rx_remove_filter, self);

    uct_tcp_cm_change_conn_state(self, UCT_TCP_EP_CONN_STATE_CLOSED);
    uct_tcp_ep_cleanup(self);

    ucs_debug("tcp_ep %p: destroyed on iface %p", self, iface);
}

UCS_CLASS_DEFINE(uct_tcp_ep_t, uct_base_ep_t);

UCS_CLASS_DEFINE_NAMED_NEW_FUNC(uct_tcp_ep_init, uct_tcp_ep_t, uct_tcp_ep_t,
                                uct_tcp_iface_t*, int,
                                const struct sockaddr_in*)
UCS_CLASS_DEFINE_NAMED_DELETE_FUNC(uct_tcp_ep_destroy_internal,
                                   uct_tcp_ep_t, uct_ep_t)

void uct_tcp_ep_destroy(uct_ep_h tl_ep)
{
    uct_tcp_ep_t *ep       = ucs_derived_of(tl_ep, uct_tcp_ep_t);
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);

    if (/* EPs that are connected as CONNECT_TO_EP have to be full duplex */
        !(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP) &&
        (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) &&
        ucs_test_all_flags(ep->flags, UCT_TCP_EP_CTX_CAPS)) {
        /* remove from the expected queue and then add it to the
         * unexpected queue */
        uct_tcp_cm_remove_ep(iface, ep);
        /* remove TX capability, but still will be able to receive data */
        uct_tcp_ep_remove_ctx_cap(ep, UCT_TCP_EP_FLAG_CTX_TYPE_TX);
        uct_tcp_cm_insert_ep(iface, ep);
    } else {
        uct_tcp_ep_destroy_internal(tl_ep);
    }
}

static unsigned uct_tcp_ep_failed_progress(void *arg)
{
    uct_tcp_ep_t *ep       = (uct_tcp_ep_t*)arg;
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);

    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_FAILED);
    /* Reset FAILED flag to not remove callback in the EP destructor */
    ep->flags &= ~UCT_TCP_EP_FLAG_FAILED;

    if (ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX) {
        uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CLOSED);
        uct_set_ep_failed(&UCS_CLASS_NAME(uct_tcp_ep_t),
                          &ep->super.super, &iface->super.super,
                          UCS_ERR_ENDPOINT_TIMEOUT);
    } else {
        uct_tcp_ep_destroy_internal(&ep->super.super);
    }

    return 1;
}

void uct_tcp_ep_set_failed(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface   = ucs_derived_of(ep->super.super.iface,
                                              uct_tcp_iface_t);
    uct_worker_cb_id_t cb_id = UCS_CALLBACKQ_ID_NULL;

    if (ep->flags & UCT_TCP_EP_FLAG_FAILED) {
        return;
    }

    if (ep->flags & UCT_TCP_EP_FLAG_ON_MATCH_CTX) {
        uct_tcp_cm_remove_ep(iface, ep);
        uct_tcp_iface_add_ep(ep);
    }

    uct_tcp_ep_mod_events(ep, 0, ep->events);
    ep->flags |= UCT_TCP_EP_FLAG_FAILED;
    uct_worker_progress_register_safe(&iface->super.worker->super,
                                      uct_tcp_ep_failed_progress, ep,
                                      UCS_CALLBACKQ_FLAG_ONESHOT, &cb_id);
}

static inline void uct_tcp_ep_ctx_move(uct_tcp_ep_ctx_t *to_ctx,
                                       uct_tcp_ep_ctx_t *from_ctx)
{
    if (!uct_tcp_ep_ctx_buf_need_progress(from_ctx)) {
        return;
    }

    memcpy(to_ctx, from_ctx, sizeof(*to_ctx));
    memset(from_ctx, 0, sizeof(*from_ctx));
}

static ucs_status_t uct_tcp_ep_keepalive_enable(uct_tcp_ep_t *ep)
{
#ifdef UCT_TCP_EP_KEEPALIVE
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    const int optval       = 1;
    int idle_sec;
    int intvl_sec;
    ucs_status_t status;

    if ((iface->config.keepalive.idle == 0) ||
        (iface->config.keepalive.cnt == 0) ||
        (iface->config.keepalive.intvl == 0)) {
        return UCS_OK;
    }

    idle_sec  = ucs_max(1, (int)ucs_time_to_sec(iface->config.keepalive.idle));
    intvl_sec = ucs_max(1, (int)ucs_time_to_sec(iface->config.keepalive.intvl));

    status = ucs_socket_setopt(ep->fd, IPPROTO_TCP, TCP_KEEPINTVL,
                               &intvl_sec, sizeof(intvl_sec));
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_socket_setopt(ep->fd, IPPROTO_TCP, TCP_KEEPCNT,
                               &iface->config.keepalive.cnt,
                               sizeof(iface->config.keepalive.cnt));
    if (status != UCS_OK) {
        return status;
    }

    status = ucs_socket_setopt(ep->fd, IPPROTO_TCP, TCP_KEEPIDLE,
                               &idle_sec, sizeof(idle_sec));
    if (status != UCS_OK) {
        return status;
    }

    return ucs_socket_setopt(ep->fd, SOL_SOCKET, SO_KEEPALIVE,
                             &optval, sizeof(optval));
#else /* UCT_TCP_EP_KEEPALIVE */
    return UCS_OK;
#endif /* UCT_TCP_EP_KEEPALIVE */
}

static ucs_status_t uct_tcp_ep_create_socket_and_connect(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    ucs_status_t status;

    status = ucs_socket_create(AF_INET, SOCK_STREAM, &ep->fd);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_tcp_iface_set_sockopt(iface, ep->fd,
                                       iface->config.conn_nb);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_tcp_ep_keepalive_enable(ep);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_tcp_cm_conn_start(ep);
    if (status != UCS_OK) {
        goto err;
    }

out:
    return status;

err:
    if (ep->conn_retries > 1) {
        /* if this is not the first connection establishment retry (i.e. it
         * is not called from uct_ep_create()/uct_ep_connect_to_ep()), set
         * EP as failed */
        uct_tcp_ep_set_failed(ep);
    }
    goto out;
}

void uct_tcp_ep_replace_ep(uct_tcp_ep_t *to_ep, uct_tcp_ep_t *from_ep)
{
    uct_tcp_iface_t *iface   = ucs_derived_of(to_ep->super.super.iface,
                                              uct_tcp_iface_t);
    int events               = from_ep->events;
    uct_worker_cb_id_t cb_id = UCS_CALLBACKQ_ID_NULL;

    uct_tcp_ep_mod_events(from_ep, 0, from_ep->events);
    to_ep->fd   = from_ep->fd;
    from_ep->fd = -1;
    uct_tcp_ep_mod_events(to_ep, events, 0);

    to_ep->conn_retries++;

    uct_tcp_ep_ctx_move(&to_ep->tx, &from_ep->tx);
    uct_tcp_ep_ctx_move(&to_ep->rx, &from_ep->rx);

    ucs_queue_splice(&to_ep->pending_q, &from_ep->pending_q);
    ucs_queue_splice(&to_ep->put_comp_q, &from_ep->put_comp_q);

    to_ep->flags |= from_ep->flags & (UCT_TCP_EP_FLAG_ZCOPY_TX           |
                                      UCT_TCP_EP_FLAG_PUT_RX             |
                                      UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK |
                                      UCT_TCP_EP_FLAG_PUT_RX_SENDING_ACK);

    if (uct_tcp_ep_ctx_buf_need_progress(&to_ep->rx)) {
        /* If some data was already read, we have to process it */
        uct_worker_progress_register_safe(&iface->super.worker->super,
                                          uct_tcp_ep_progress_data_rx, to_ep,
                                          UCS_CALLBACKQ_FLAG_ONESHOT, &cb_id);
    }

    /* The internal EP is not needed anymore, start failed flow for the
     * internal EP in order to destroy it from progress (to not dereference
     * already destroyed EP) */
    ucs_assert(!(from_ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX));
    uct_tcp_ep_set_failed(from_ep);
}

static ucs_status_t uct_tcp_ep_connect(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    uct_tcp_ep_t *peer_ep  = NULL;
    ucs_status_t status;

    /* Check that the connection was not issued yet. New connection has to be
     * issue form:
     * - uct_ep_connect_to_ep(), if the EP created using CONNECT_TO_EP method
     *   and a local side must connect to a peer (due to address resolution
     *   logic)
     * - uct_ep_create(), if the EP created using CONNECT_TO_IFACE method
     *   and no an internal EP was created due to connection from a peer */
    ucs_assert((ep->conn_state == UCT_TCP_EP_CONN_STATE_CLOSED) &&
               (ep->conn_retries == 0));

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTING);

    if (uct_tcp_ep_is_self(ep) ||
        (ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP)) {
        status = uct_tcp_ep_create_socket_and_connect(ep);
        if (status != UCS_OK) {
            return status;
        }
        goto out;
    }

    peer_ep = uct_tcp_cm_get_ep(iface, &ep->peer_addr, ep->cm_id.conn_sn,
                                UCT_TCP_EP_FLAG_CTX_TYPE_RX);
    if (peer_ep == NULL) {
        status = uct_tcp_ep_create_socket_and_connect(ep);
        if (status != UCS_OK) {
            return status;
        }
    } else {
        /* EP that connects to self or EP created using CONNECT_TO_EP mustn't
         * go here and always create socket and conenct to a peer */
        ucs_assert(!uct_tcp_ep_is_self(ep) &&
                   !(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP));
        ucs_assert((peer_ep != NULL) && (peer_ep->fd != -1) &&
                   !(peer_ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX));
        uct_tcp_ep_move_ctx_cap(peer_ep, ep, UCT_TCP_EP_FLAG_CTX_TYPE_RX);
        uct_tcp_ep_replace_ep(ep, peer_ep);

        uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CONNECTED);

        /* Send the connection request to the peer */
        status = uct_tcp_cm_send_event(ep, UCT_TCP_CM_CONN_REQ, 0);
        if (status == UCS_OK) {
            uct_tcp_ep_mod_events(ep, UCS_EVENT_SET_EVWRITE, 0);
        }
    }

out:
    /* Set TX capability even if something is failed for the EP (e.g. sending
     * CONN_REQ to the peer) */
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX));
    uct_tcp_ep_add_ctx_cap(ep, UCT_TCP_EP_FLAG_CTX_TYPE_TX);

    if (!uct_tcp_ep_is_self(ep) && (status == UCS_OK) &&
        !(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP)) {
        /* Move the EP to the expected queue in order to detect ghost
         * connections */
        uct_tcp_iface_remove_ep(ep);
        uct_tcp_cm_insert_ep(iface, ep);
    }

    return UCS_OK;
}

void uct_tcp_ep_set_dest_addr(const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr,
                              struct sockaddr_in *dest_addr)
{
    memset(dest_addr, 0, sizeof(*dest_addr));
    /* TODO: handle AF_INET6 */
    dest_addr->sin_family = AF_INET;
    dest_addr->sin_port   = *(const in_port_t*)iface_addr;
    dest_addr->sin_addr   = *(const struct in_addr*)
                            ucs_sockaddr_get_inet_addr((const struct sockaddr*)
                                                       dev_addr);
}

uint64_t uct_tcp_ep_get_cm_id(const uct_tcp_ep_t *ep)
{
    return (ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP) ?
           ep->cm_id.ptr_map_key : ep->cm_id.conn_sn;
}

ucs_status_t uct_tcp_ep_create(const uct_ep_params_t *params, uct_ep_h *ep_p)
{
    uct_tcp_iface_t *iface           = ucs_derived_of(params->iface,
                                                      uct_tcp_iface_t);
    uct_tcp_ep_t *ep                 = NULL;
    struct sockaddr_in *ep_dest_addr = NULL;
    struct sockaddr_in dest_addr;
    ucs_status_t status;

    if (ucs_test_all_flags(params->field_mask,
                           UCT_EP_PARAM_FIELD_DEV_ADDR |
                           UCT_EP_PARAM_FIELD_IFACE_ADDR)) {
        uct_tcp_ep_set_dest_addr(params->dev_addr, params->iface_addr,
                                 &dest_addr);
        ep_dest_addr = &dest_addr;
    }

    status = uct_tcp_ep_init(iface, -1, ep_dest_addr, &ep);
    if (status != UCS_OK) {
        return status;
    }

    if (!(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP)) {
        uct_tcp_cm_ep_set_conn_sn(ep);
        status = uct_tcp_ep_connect(ep);
        if (status != UCS_OK) {
            return status;
        }
    }

    /* cppcheck-suppress autoVariables */
    *ep_p = &ep->super.super;
    return UCS_OK;
}

ucs_status_t uct_tcp_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr)
{
    uct_tcp_ep_t *ep        = ucs_derived_of(tl_ep, uct_tcp_ep_t);
    uct_tcp_ep_addr_t *addr = (uct_tcp_ep_addr_t*)ep_addr;

    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP);

    addr->ptr_map_key = ep->cm_id.ptr_map_key;
    return uct_iface_get_address(tl_ep->iface,
                                 (uct_iface_addr_t*)&addr->iface_addr);
}

ucs_status_t uct_tcp_ep_connect_to_ep(uct_ep_h tl_ep,
                                      const uct_device_addr_t *dev_addr,
                                      const uct_ep_addr_t *ep_addr)
{
    uct_tcp_ep_t *ep                    = ucs_derived_of(tl_ep, uct_tcp_ep_t);
    uct_tcp_iface_t UCS_V_UNUSED *iface = ucs_derived_of(ep->super.super.iface,
                                                         uct_tcp_iface_t);
    uct_tcp_ep_addr_t *addr             = (uct_tcp_ep_addr_t*)ep_addr;

    ucs_assert(ep->flags & UCT_TCP_EP_FLAG_CONNECT_TO_EP);

    if (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) {
        /* CONN_REQ was already received by the EP, no need for any actions
         * anymore */
        ucs_assert(uct_tcp_ep_ptr_map_retrieve(iface,
                                               ep->cm_id.ptr_map_key) == NULL);
        return UCS_OK;
    }

    uct_tcp_ep_set_dest_addr(dev_addr, (uct_iface_addr_t*)&addr->iface_addr,
                             &ep->peer_addr);

    if (!uct_tcp_cm_ep_accept_conn(ep)) {
        ucs_assert(ep->conn_state == UCT_TCP_EP_CONN_STATE_CLOSED);
        /* EP that are created as CONNECT_TO_EP has to be full-duplex, set RX
         * capability as well as TX (that's set in uct_tcp_ep_connect()) */
        uct_tcp_ep_add_ctx_cap(ep, UCT_TCP_EP_FLAG_CTX_TYPE_RX);

        uct_tcp_ep_ptr_map_del(ep);

        /* Use remote peer connection sequence number value, since the EP has to
         * send the CONN_REQ to the peer has to find its EP in the EP PTR map */
        ep->cm_id.ptr_map_key = addr->ptr_map_key;
        return uct_tcp_ep_connect(ep);
    }

    ucs_assert(!uct_tcp_ep_is_self(ep));
    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_ACCEPTING);

    return UCS_OK;
}

void uct_tcp_ep_mod_events(uct_tcp_ep_t *ep, ucs_event_set_types_t add,
                           ucs_event_set_types_t rem)
{
    uct_tcp_iface_t *iface           = ucs_derived_of(ep->super.super.iface,
                                                      uct_tcp_iface_t);
    ucs_event_set_types_t old_events = ep->events;
    ucs_event_set_types_t new_events = (ep->events | add) & ~rem;
    ucs_status_t status;

    if (new_events != ep->events) {
        ucs_assert(ep->fd != -1);
        ep->events = new_events;
        ucs_trace("tcp_ep %p: set events to %c%c", ep,
                  (new_events & UCS_EVENT_SET_EVREAD)  ? 'r' : '-',
                  (new_events & UCS_EVENT_SET_EVWRITE) ? 'w' : '-');
        if (new_events == 0) {
            status = ucs_event_set_del(iface->event_set, ep->fd);
        } else if (old_events != 0) {
            status = ucs_event_set_mod(iface->event_set, ep->fd, ep->events,
                                       (void*)ep);
        } else {
            status = ucs_event_set_add(iface->event_set, ep->fd, ep->events,
                                       (void*)ep);
        }
        if (status != UCS_OK) {
            ucs_fatal("unable to modify event set for tcp_ep %p (fd=%d)", ep,
                      ep->fd);
        }
    }
}

static inline void uct_tcp_ep_handle_put_ack(uct_tcp_ep_t *ep,
                                             uct_tcp_ep_put_ack_hdr_t *put_ack)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    uct_tcp_ep_put_completion_t *put_comp;

    if (put_ack->sn == ep->tx.put_sn) {
        /* Since there are no other PUT operations in-flight, can remove flag
         * and decrement iface outstanding operations counter */
        ucs_assert(ep->flags & UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK);
        ep->flags &= ~UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK;
        uct_tcp_iface_outstanding_dec(iface);
    }

    ucs_queue_for_each_extract(put_comp, &ep->put_comp_q, elem,
                               (UCS_CIRCULAR_COMPARE32(put_comp->wait_put_sn,
                                                       <=, put_ack->sn))) {
        uct_invoke_completion(put_comp->comp, UCS_OK);
        ucs_mpool_put_inline(put_comp);
    }
}

void uct_tcp_ep_pending_queue_dispatch(uct_tcp_ep_t *ep)
{
    uct_pending_req_priv_queue_t *priv;

    uct_pending_queue_dispatch(priv, &ep->pending_q,
                               uct_tcp_ep_ctx_buf_empty(&ep->tx));
    if (uct_tcp_ep_ctx_buf_empty(&ep->tx)) {
        ucs_assert(ucs_queue_is_empty(&ep->pending_q));
        uct_tcp_ep_mod_events(ep, 0, UCS_EVENT_SET_EVWRITE);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_tcp_ep_tx_started(uct_tcp_ep_t *ep, const uct_tcp_am_hdr_t *hdr)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);

    ep->tx.length      += sizeof(*hdr) + hdr->length;
    iface->outstanding += ep->tx.length;
}

static UCS_F_ALWAYS_INLINE void
uct_tcp_ep_tx_completed(uct_tcp_ep_t *ep, size_t sent_length)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);

    iface->outstanding -= sent_length;
    ep->tx.offset      += sent_length;
}

static UCS_F_ALWAYS_INLINE void
uct_tcp_ep_zcopy_completed(uct_tcp_ep_t *ep, uct_completion_t *comp,
                           ucs_status_t status)
{
    ep->flags &= ~UCT_TCP_EP_FLAG_ZCOPY_TX;
    if (comp != NULL) {
        uct_invoke_completion(comp, status);
    }
}

static void uct_tcp_ep_handle_disconnected(uct_tcp_ep_t *ep, ucs_status_t status)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    uct_tcp_ep_zcopy_tx_t *ctx;

    ucs_debug("tcp_ep %p: remote disconnected", ep);

    if (ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_TX) {
        if (ep->flags & UCT_TCP_EP_FLAG_CTX_TYPE_RX) {
            uct_tcp_ep_remove_ctx_cap(ep, UCT_TCP_EP_FLAG_CTX_TYPE_RX);
            ep->flags &= ~UCT_TCP_EP_FLAG_PUT_RX_SENDING_ACK;
        }

        if (ep->flags & UCT_TCP_EP_FLAG_ZCOPY_TX) {
            /* There is ongoing AM/PUT Zcopy operation, need to notify
             * the user about the error */
            ctx = (uct_tcp_ep_zcopy_tx_t*)ep->tx.buf;
            uct_tcp_ep_zcopy_completed(ep, ctx->comp, status);
        }

        if (ep->flags & UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK) {
            /* if the EP is waiting for the acknowledgment of the started
             * PUT operation, decrease iface::outstanding counter */
            uct_tcp_iface_outstanding_dec(iface);
            ep->flags &= ~UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK;
        }

        uct_tcp_ep_tx_completed(ep, ep->tx.length - ep->tx.offset);
    }

    uct_tcp_ep_set_failed(ep);
}

static inline ucs_status_t uct_tcp_ep_handle_send_err(uct_tcp_ep_t *ep,
                                                      ucs_status_t status)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);

    ucs_assert(status != UCS_ERR_NO_PROGRESS);

    status = uct_tcp_ep_handle_io_err(ep, "send", status);
    if (status == UCS_ERR_CANCELED) {
        /* If no data were read to the allocated buffer,
         * we can safely reset it for futher re-use and to
         * avoid overwriting this buffer, because `rx::length == 0` */
        if (ep->tx.length == 0) {
            uct_tcp_ep_ctx_reset(&ep->tx);
        }
    } else {
        uct_tcp_ep_handle_disconnected(ep, status);
        if (iface->super.err_handler != NULL) {
            /* Translate the error to EP timeout error, since user expects that
             * status returned from UCT error callback is the same as it is
             * returned from the UCT EP send operation (or from UCT competion
             * callback)
             * TODO: revise this behavior */
            return UCS_ERR_ENDPOINT_TIMEOUT;
        }
    }

    return status;
}

static inline ssize_t uct_tcp_ep_send(uct_tcp_ep_t *ep)
{
    size_t sent_length;
    ucs_status_t status;

    ucs_assert(ep->tx.length > ep->tx.offset);
    sent_length = ep->tx.length - ep->tx.offset;

    status = ucs_socket_send_nb(ep->fd,
                                UCS_PTR_BYTE_OFFSET(ep->tx.buf, ep->tx.offset),
                                &sent_length);
    if (ucs_unlikely((status != UCS_OK) &&
                     (status != UCS_ERR_NO_PROGRESS))) {
        return uct_tcp_ep_handle_send_err(ep, status);
    }

    uct_tcp_ep_tx_completed(ep, sent_length);

    ucs_assert(sent_length <= SSIZE_MAX);
    return sent_length;
}

static inline ssize_t uct_tcp_ep_sendv(uct_tcp_ep_t *ep)
{
    uct_tcp_ep_zcopy_tx_t *ctx = (uct_tcp_ep_zcopy_tx_t*)ep->tx.buf;
    size_t sent_length;
    ucs_status_t status;

    ucs_assertv((ep->tx.offset < ep->tx.length) &&
                (ctx->iov_cnt > 0), "ep=%p", ep);

    status = ucs_socket_sendv_nb(ep->fd, &ctx->iov[ctx->iov_index],
                                 ctx->iov_cnt - ctx->iov_index, &sent_length);
    if (ucs_unlikely(status != UCS_OK)) {
        if (status == UCS_ERR_NO_PROGRESS) {
            ucs_assert(sent_length == 0);
            return 0;
        }

        status = uct_tcp_ep_handle_send_err(ep, status);
        uct_tcp_ep_zcopy_completed(ep, ctx->comp, status);
        return status;
    }

    uct_tcp_ep_tx_completed(ep, sent_length);

    if (ep->tx.offset != ep->tx.length) {
        ucs_iov_advance(ctx->iov, ctx->iov_cnt,
                        &ctx->iov_index, sent_length);
    } else {
        uct_tcp_ep_zcopy_completed(ep, ctx->comp, UCS_OK);
    }

    ucs_assert(sent_length <= SSIZE_MAX);
    return sent_length;
}

static int uct_tcp_ep_is_conn_closed_by_peer(ucs_status_t io_status)
{
    return (io_status == UCS_ERR_REJECTED) ||
           (io_status == UCS_ERR_CONNECTION_RESET) ||
           (io_status == UCS_ERR_NOT_CONNECTED) ||
           (io_status == UCS_ERR_TIMED_OUT);
}

/* if the caller mustn't mark the EP as failed, the function returns:
 * - UCS_ERR_CANCELED, since the error handled here is recoverable:
 *   - the peer closed the connection, since its EP is a winner of simultaneous
 *     connection establishment - just close the socket fd and wait for the
 *     connection request from the peer to acknowledge it.
 *   - the connection was dropped by the peer's host due to out-of-resources
 *     to keep the connection - need to reconnect to the peer.
 * - UCS_ERR_NO_PROGRESS, since it just informs the caller needs to try
 *   again to progress non-blocking send()/recv() operation.
 * otherwise - the result of the IO operation is returned. */
ucs_status_t uct_tcp_ep_handle_io_err(uct_tcp_ep_t *ep, const char *op_str,
                                      ucs_status_t io_status)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    ucs_status_t status;
    char str_local_addr[UCS_SOCKADDR_STRING_LEN];
    char str_remote_addr[UCS_SOCKADDR_STRING_LEN];

    ucs_assert(io_status != UCS_OK);

    if (io_status == UCS_ERR_NO_PROGRESS) {
        return UCS_ERR_NO_PROGRESS;
    }

    if (!uct_tcp_ep_is_conn_closed_by_peer(io_status)) {
        /* IO operation failed with the unrecoverable error */
        goto err;
    }

    if ((ep->conn_state == UCT_TCP_EP_CONN_STATE_ACCEPTING) ||
        (ep->conn_state == UCT_TCP_EP_CONN_STATE_RECV_MAGIC_NUMBER)) {
        ucs_debug("tcp_ep %p: detected that connection was dropped by the peer",
                  ep);
        return io_status;
    } else if ((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED) &&
               ((ep->flags & UCT_TCP_EP_CTX_CAPS) ==
                UCT_TCP_EP_FLAG_CTX_TYPE_RX) /* only RX cap */) {
        ucs_debug("tcp_ep %p: detected that [%s <-> %s]:%"PRIu64" connection was "
                  "dropped by the peer", ep,
                  ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                                   str_local_addr, UCS_SOCKADDR_STRING_LEN),
                  ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                                   str_remote_addr, UCS_SOCKADDR_STRING_LEN),
                  uct_tcp_ep_get_cm_id(ep));
        return io_status;
    } else if ((ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTING) ||
               (ep->conn_state == UCT_TCP_EP_CONN_STATE_WAITING_ACK)) {
        uct_tcp_ep_mod_events(ep, 0, ep->events);
        ucs_close_fd(&ep->fd);

        uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_CLOSED);

        status = uct_tcp_ep_create_socket_and_connect(ep);
        if (status == UCS_OK) {
            return UCS_ERR_CANCELED;
        }

        /* if connection establishment fails, the system limits
         * may not be big enough */
        ucs_error("try to increase \"net.core.somaxconn\", "
                  "\"net.core.netdev_max_backlog\", "
                  "\"net.ipv4.tcp_max_syn_backlog\" to the maximum value "
                  "on the remote node or increase %s%s%s (=%u)",
                  UCS_DEFAULT_ENV_PREFIX, UCT_TCP_CONFIG_PREFIX,
                  UCT_TCP_CONFIG_MAX_CONN_RETRIES,
                  iface->config.max_conn_retries);

        return io_status;
    } else if ((io_status == UCS_ERR_NOT_CONNECTED) &&
               (ep->conn_state == UCT_TCP_EP_CONN_STATE_CONNECTED)) {
        uct_tcp_ep_mod_events(ep, 0, ep->events);
        ucs_close_fd(&ep->fd);
        /* if this connection is needed for the local side, it will be
         * detected by the TX operations and error handling will be done */
        ucs_debug("tcp_ep %p: detected that [%s <-> %s]:%"PRIu64" connection was "
                  "closed by the peer", ep,
                  ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                                   str_local_addr, UCS_SOCKADDR_STRING_LEN),
                  ucs_sockaddr_str((const struct sockaddr*)&ep->peer_addr,
                                   str_remote_addr, UCS_SOCKADDR_STRING_LEN),
                  uct_tcp_ep_get_cm_id(ep));
        return io_status;
    }

err:
    ucs_error("tcp_ep %p (state=%s): %s(%d) failed: %s",
              ep, uct_tcp_ep_cm_state[ep->conn_state].name,
              op_str, ep->fd, ucs_status_string(io_status));
    return io_status;
}

static inline void uct_tcp_ep_handle_recv_err(uct_tcp_ep_t *ep,
                                              ucs_status_t status)
{
    status = uct_tcp_ep_handle_io_err(ep, "recv", status);
    if ((status == UCS_ERR_NO_PROGRESS) || (status == UCS_ERR_CANCELED)) {
        /* If no data were read to the allocated buffer,
         * we can safely reset it for futher re-use and to
         * avoid overwriting this buffer, because `rx::length == 0` */
        if (ep->rx.length == 0) {
            uct_tcp_ep_ctx_reset(&ep->rx);
        }
    } else {
        uct_tcp_ep_ctx_reset(&ep->rx);
        uct_tcp_ep_handle_disconnected(ep, status);
    }
}

static inline unsigned uct_tcp_ep_recv(uct_tcp_ep_t *ep, size_t recv_length)
{
    uct_tcp_iface_t UCS_V_UNUSED *iface = ucs_derived_of(ep->super.super.iface,
                                                         uct_tcp_iface_t);
    ucs_status_t status;

    if (ucs_unlikely(recv_length == 0)) {
        return 1;
    }

    status = ucs_socket_recv_nb(ep->fd, UCS_PTR_BYTE_OFFSET(ep->rx.buf,
                                                            ep->rx.length),
                                &recv_length);
    if (ucs_unlikely(status != UCS_OK)) {
        uct_tcp_ep_handle_recv_err(ep, status);
        return 0;
    }

    ucs_assertv(recv_length != 0, "ep=%p", ep);

    ep->rx.length += recv_length;
    ucs_trace_data("tcp_ep %p: recvd %zu bytes", ep, recv_length);
    ucs_assert(ep->rx.length <= (iface->config.rx_seg_size * 2));

    return 1;
}

static inline void uct_tcp_ep_check_tx_completion(uct_tcp_ep_t *ep)
{
    if (ucs_likely(!uct_tcp_ep_ctx_buf_need_progress(&ep->tx))) {
        uct_tcp_ep_ctx_reset(&ep->tx);
    } else {
        uct_tcp_ep_mod_events(ep, UCS_EVENT_SET_EVWRITE, 0);
    }
}

/* Forward declaration - the function depends on AM send
 * functions implemented below */
static void uct_tcp_ep_post_put_ack(uct_tcp_ep_t *ep);

static unsigned uct_tcp_ep_progress_data_tx(void *arg)
{
    uct_tcp_ep_t *ep = (uct_tcp_ep_t*)arg;
    unsigned ret     = 0;
    ssize_t offset;

    ucs_trace_func("ep=%p", ep);

    if (uct_tcp_ep_ctx_buf_need_progress(&ep->tx)) {
        offset = (!(ep->flags & UCT_TCP_EP_FLAG_ZCOPY_TX) ?
                  uct_tcp_ep_send(ep) : uct_tcp_ep_sendv(ep));
        if (ucs_unlikely(offset < 0)) {
            return 1;
        }

        ret = (offset > 0);

        ucs_trace_data("ep %p fd %d sent %zu/%zu bytes, moved by offset %zd",
                       ep, ep->fd, ep->tx.offset, ep->tx.length, offset);

        uct_tcp_ep_check_tx_completion(ep);
    }

    if (ep->flags & UCT_TCP_EP_FLAG_PUT_RX_SENDING_ACK) {
        uct_tcp_ep_post_put_ack(ep);
    }

    if (!ucs_queue_is_empty(&ep->pending_q)) {
        uct_tcp_ep_pending_queue_dispatch(ep);
        return ret;
    }

    if (uct_tcp_ep_ctx_buf_empty(&ep->tx)) {
        ucs_assert(ucs_queue_is_empty(&ep->pending_q));
        uct_tcp_ep_mod_events(ep, 0, UCS_EVENT_SET_EVWRITE);
    }

    return ret;
}

static inline void
uct_tcp_ep_comp_recv_am(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep,
                        uct_tcp_am_hdr_t *hdr)
{
    uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_RECV, hdr->am_id,
                       hdr + 1, hdr->length,
                       "RECV: ep %p fd %d received %zu/%zu bytes",
                       ep, ep->fd, ep->rx.offset, ep->rx.length);
    uct_iface_invoke_am(&iface->super, hdr->am_id, hdr + 1, hdr->length, 0);
}

static inline ucs_status_t
uct_tcp_ep_put_rx_advance(uct_tcp_ep_t *ep, uct_tcp_ep_put_req_hdr_t *put_req,
                          size_t recv_length)
{
    ucs_assert(!(ep->flags & UCT_TCP_EP_FLAG_PUT_RX_SENDING_ACK));
    ucs_assert(recv_length <= put_req->length);
    put_req->addr   += recv_length;
    put_req->length -= recv_length;

    if (!put_req->length) {
        uct_tcp_ep_post_put_ack(ep);

        /* EP's ctx_caps doesn't have UCT_TCP_EP_FLAG_PUT_RX flag
         * set in case of entire PUT payload was received through
         * AM protocol */
        if (ep->flags & UCT_TCP_EP_FLAG_PUT_RX) {
            ep->flags &= ~UCT_TCP_EP_FLAG_PUT_RX;
            uct_tcp_ep_ctx_reset(&ep->rx);
        }

        return UCS_OK;
    }

    return UCS_INPROGRESS;
}

static inline void uct_tcp_ep_handle_put_req(uct_tcp_ep_t *ep,
                                             uct_tcp_ep_put_req_hdr_t *put_req,
                                             size_t extra_recvd_length)
{
    size_t copied_length;
    ucs_status_t status;

    ucs_assert(put_req->addr || !put_req->length);

    copied_length  = ucs_min(put_req->length, extra_recvd_length);
    memcpy((void*)(uintptr_t)put_req->addr,
           UCS_PTR_BYTE_OFFSET(ep->rx.buf, ep->rx.offset),
           copied_length);
    ep->rx.offset += copied_length;
    ep->rx.put_sn  = put_req->sn;

    /* Remove the flag that indicates that EP is sending PUT RX ACK in order
     * to not ack the uncompleted PUT RX operation for which PUT REQ is being
     * handled here. ACK for both operations will be sent after the completion
     * of the last received PUT operation */
    ep->flags &= ~UCT_TCP_EP_FLAG_PUT_RX_SENDING_ACK;

    status = uct_tcp_ep_put_rx_advance(ep, put_req, copied_length);
    if (status == UCS_OK) {
        return;
    }

    ucs_assert(ep->rx.offset == ep->rx.length);
    uct_tcp_ep_ctx_rewind(&ep->rx);
    /* Since RX buffer and PUT request can be ovelapped, use memmove() */
    memmove(ep->rx.buf, put_req, sizeof(*put_req));
    ep->flags |= UCT_TCP_EP_FLAG_PUT_RX;
}

static unsigned uct_tcp_ep_progress_am_rx(uct_tcp_ep_t *ep)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    unsigned handled       = 0;
    uct_tcp_am_hdr_t *hdr;
    size_t recv_length;
    size_t recvd_length;
    size_t remaining;

    ucs_trace_func("ep=%p", ep);

    if (!uct_tcp_ep_ctx_buf_need_progress(&ep->rx)) {
        ucs_assert(ep->rx.buf == NULL);
        ep->rx.buf = ucs_mpool_get_inline(&iface->rx_mpool);
        if (ucs_unlikely(ep->rx.buf == NULL)) {
            ucs_warn("tcp_ep %p: unable to get a buffer from RX memory pool", ep);
            return 0;
        }

        /* post the entire AM buffer */
        recv_length = iface->config.rx_seg_size;
    } else if (ep->rx.length < sizeof(*hdr)) {
        ucs_assert((ep->rx.buf != NULL) && (ep->rx.offset == 0));

        /* do partial receive of the remaining part of the hdr
         * and post the entire AM buffer */
        recv_length = iface->config.rx_seg_size - ep->rx.length;
    } else {
        ucs_assert((ep->rx.buf != NULL) &&
                   ((ep->rx.length - ep->rx.offset) >= sizeof(*hdr)));

        /* do partial receive of the remaining user data */
        hdr          = UCS_PTR_BYTE_OFFSET(ep->rx.buf, ep->rx.offset);
        recvd_length = ep->rx.length - ep->rx.offset - sizeof(*hdr);
        recv_length  = ucs_max(0, (ssize_t)(hdr->length - recvd_length));
    }

    if (!uct_tcp_ep_recv(ep, recv_length)) {
        goto out;
    }

    /* Parse received active messages */
    while (uct_tcp_ep_ctx_buf_need_progress(&ep->rx)) {
        remaining = ep->rx.length - ep->rx.offset;
        if (remaining < sizeof(*hdr)) {
            /* Move the partially received hdr to the beginning of the buffer */
            memmove(ep->rx.buf, UCS_PTR_BYTE_OFFSET(ep->rx.buf, ep->rx.offset),
                    remaining);
            ep->rx.offset = 0;
            ep->rx.length = remaining;
            handled++;
            goto out;
        }

        hdr = UCS_PTR_BYTE_OFFSET(ep->rx.buf, ep->rx.offset);
        ucs_assertv(hdr->length <= (iface->config.rx_seg_size - sizeof(*hdr)),
                    "tcp_ep %p (conn state - %s): %u vs %zu",
                    ep, uct_tcp_ep_cm_state[ep->conn_state].name, hdr->length,
                    (iface->config.rx_seg_size - sizeof(*hdr)));

        if (remaining < (sizeof(*hdr) + hdr->length)) {
            handled++;
            goto out;
        }

        /* Full message was received */
        ep->rx.offset += sizeof(*hdr) + hdr->length;
        ucs_assert(ep->rx.offset <= ep->rx.length);

        if (ucs_likely(hdr->am_id < UCT_AM_ID_MAX)) {
            uct_tcp_ep_comp_recv_am(iface, ep, hdr);
            handled++;
        } else if (hdr->am_id == UCT_TCP_EP_PUT_REQ_AM_ID) {
            ucs_assert(hdr->length == sizeof(uct_tcp_ep_put_req_hdr_t));
            uct_tcp_ep_handle_put_req(ep, (uct_tcp_ep_put_req_hdr_t*)(hdr + 1),
                                      ep->rx.length - ep->rx.offset);
            handled++;
            if (ep->flags & UCT_TCP_EP_FLAG_PUT_RX) {
                /* It means that PUT RX is in progress and EP RX buffer
                 * is used to keep PUT header. So, we don't need to
                 * release a EP RX buffer */
                goto out;
            }
        } else if (hdr->am_id == UCT_TCP_EP_PUT_ACK_AM_ID) {
            ucs_assert(hdr->length == sizeof(uint32_t));
            uct_tcp_ep_handle_put_ack(ep, (uct_tcp_ep_put_ack_hdr_t*)(hdr + 1));
            handled++;
        } else {
            ucs_assert(hdr->am_id == UCT_TCP_EP_CM_AM_ID);
            handled += 1 + uct_tcp_cm_handle_conn_pkt(&ep, hdr + 1, hdr->length);
            /* coverity[check_after_deref] */
            if (ep == NULL) {
                goto out;
            }
        }

        ucs_assert(ep != NULL);
    }

    uct_tcp_ep_ctx_reset(&ep->rx);

out:
    return handled;
}

static inline ucs_status_t
uct_tcp_ep_am_prepare(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep,
                      uint8_t am_id, uct_tcp_am_hdr_t **hdr)
{
    ucs_status_t status;

    status = uct_tcp_ep_check_tx_res(ep);
    if (ucs_unlikely(status != UCS_OK)) {
        if (ucs_likely(status == UCS_ERR_NO_RESOURCE)) {
            goto err_no_res;
        }

        return status;
    }

    ucs_assertv(ep->tx.buf == NULL, "ep=%p", ep);

    ep->tx.buf = ucs_mpool_get_inline(&iface->tx_mpool);
    if (ucs_unlikely(ep->tx.buf == NULL)) {
        goto err_no_res;
    }

    *hdr          = ep->tx.buf;
    (*hdr)->am_id = am_id;

    return UCS_OK;

err_no_res:
    UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
    return UCS_ERR_NO_RESOURCE;
}

static unsigned uct_tcp_ep_progress_put_rx(uct_tcp_ep_t *ep)
{
    uct_tcp_ep_put_req_hdr_t *put_req;
    size_t recv_length;
    ucs_status_t status;

    put_req     = (uct_tcp_ep_put_req_hdr_t*)ep->rx.buf;
    recv_length = put_req->length;
    status      = ucs_socket_recv_nb(ep->fd, (void*)(uintptr_t)put_req->addr,
                                     &recv_length);
    if (ucs_unlikely(status != UCS_OK)) {
        uct_tcp_ep_handle_recv_err(ep, status);
        return 0;
    }

    ucs_assertv(recv_length, "ep=%p", ep);

    uct_tcp_ep_put_rx_advance(ep, put_req, recv_length);

    return 1;
}

static unsigned uct_tcp_ep_progress_data_rx(void *arg)
{
    uct_tcp_ep_t *ep = (uct_tcp_ep_t*)arg;

    if (!(ep->flags & UCT_TCP_EP_FLAG_PUT_RX)) {
        return uct_tcp_ep_progress_am_rx(ep);
    } else {
        return uct_tcp_ep_progress_put_rx(ep);
    }
}

static unsigned uct_tcp_ep_progress_magic_number_rx(void *arg)
{
    uct_tcp_ep_t *ep       = (uct_tcp_ep_t*)arg;
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    char str_local_addr[UCS_SOCKADDR_STRING_LEN];
    char str_remote_addr[UCS_SOCKADDR_STRING_LEN];
    size_t recv_length, prev_length;
    uint64_t magic_number;

    if (ep->rx.buf == NULL) {
        ep->rx.buf = ucs_mpool_get_inline(&iface->rx_mpool);
        if (ucs_unlikely(ep->rx.buf == NULL)) {
            ucs_warn("tcp_ep %p: unable to get a buffer from RX memory pool", ep);
            return 0;
        }
    }

    prev_length = ep->rx.length;
    recv_length = sizeof(magic_number) - ep->rx.length;

    if (!uct_tcp_ep_recv(ep, recv_length)) {
        /* Do not touch EP here as it could be destroyed during
         * socket error handling */
        return 0;
    }

    if (ep->rx.length < sizeof(magic_number)) {
        return ((ep->rx.length - prev_length) > 0);
    }

    magic_number = *(uint64_t*)ep->rx.buf;

    if (magic_number != UCT_TCP_MAGIC_NUMBER) {
        /* Silently close this connection and destroy its EP */
        ucs_debug("tcp_iface %p (%s): received wrong magic number (expected: "
                  "%lu, received: %"PRIu64") for ep=%p (fd=%d) from %s", iface,
                  ucs_sockaddr_str((const struct sockaddr*)&iface->config.ifaddr,
                                   str_local_addr, UCS_SOCKADDR_STRING_LEN),
                  UCT_TCP_MAGIC_NUMBER, magic_number, ep,
                  ep->fd, ucs_socket_getname_str(ep->fd, str_remote_addr,
                                                 UCS_SOCKADDR_STRING_LEN));
        goto err;
    }

    uct_tcp_ep_ctx_reset(&ep->rx);

    uct_tcp_cm_change_conn_state(ep, UCT_TCP_EP_CONN_STATE_ACCEPTING);

    return 1;

err:
    uct_tcp_ep_destroy_internal(&ep->super.super);
    return 0;
}

static inline void
uct_tcp_ep_set_outstanding_zcopy(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep,
                                 uct_tcp_ep_zcopy_tx_t *ctx, const void *header,
                                 unsigned header_length, uct_completion_t *comp)
{
    ctx->comp  = comp;
    ep->flags |= UCT_TCP_EP_FLAG_ZCOPY_TX;

    if ((header_length != 0) &&
        /* check whether a user's header was sent or not */
        (ep->tx.offset < (sizeof(uct_tcp_am_hdr_t) + header_length))) {
        ucs_assert(header_length <= iface->config.zcopy.max_hdr);
        /* if the user's header wasn't sent completely, copy it to
         * the EP TX buffer (after Zcopy context and IOVs) for
         * retransmission. iov_len is already set to the proper value */
        ctx->iov[1].iov_base = UCS_PTR_BYTE_OFFSET(ep->tx.buf,
                                                   iface->config.zcopy.hdr_offset);
        memcpy(ctx->iov[1].iov_base, header, header_length);
    }

    ctx->iov_index = 0;
    ucs_iov_advance(ctx->iov, ctx->iov_cnt, &ctx->iov_index, ep->tx.offset);
}

static inline ucs_status_t
uct_tcp_ep_am_send(uct_tcp_ep_t *ep, const uct_tcp_am_hdr_t *hdr)
{
    uct_tcp_iface_t UCS_V_UNUSED *iface = ucs_derived_of(ep->super.super.iface,
                                                         uct_tcp_iface_t);
    ssize_t offset;

    uct_tcp_ep_tx_started(ep, hdr);

    offset = uct_tcp_ep_send(ep);
    if (ucs_unlikely(offset < 0)) {
        return (ucs_status_t)offset;
    }

    uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_SEND, hdr->am_id,
                       hdr + 1, hdr->length, "SEND: ep %p fd %d sent "
                       "%zu/%zu bytes, moved by offset %zd",
                       ep, ep->fd, ep->tx.offset, ep->tx.length, offset);

    uct_tcp_ep_check_tx_completion(ep);

    return UCS_OK;
}

static const void*
uct_tcp_ep_am_sendv_get_trace_payload(uct_tcp_am_hdr_t *hdr,
                                      const void *header,
                                      const struct iovec *payload_iov,
                                      int short_sendv)
{
    if (short_sendv == 0) {
        return header;
    }

    /* If user requested trace data, we copy header and payload
     * to EP TX buffer in order to trace correct data */
    uct_am_short_fill_data(hdr + 1, *(const uint64_t*)header,
                           payload_iov->iov_base, payload_iov->iov_len);
    return (hdr + 1);
}

static inline ucs_status_t
uct_tcp_ep_am_sendv(uct_tcp_ep_t *ep, int short_sendv, uct_tcp_am_hdr_t *hdr,
                    size_t send_limit, const void *header,
                    struct iovec *iov, size_t iov_cnt)
{
    uct_tcp_iface_t UCS_V_UNUSED *iface = ucs_derived_of(ep->super.super.iface,
                                                         uct_tcp_iface_t);
    ucs_status_t status;
    size_t sent_length;

    uct_tcp_ep_tx_started(ep, hdr);

    ucs_assertv((ep->tx.length <= send_limit) &&
                (iov_cnt > 0), "ep=%p", ep);

    status = ucs_socket_sendv_nb(ep->fd, iov, iov_cnt, &sent_length);
    if (ucs_unlikely((status != UCS_OK) && (status != UCS_ERR_NO_PROGRESS))) {
        return uct_tcp_ep_handle_send_err(ep, status);
    }

    uct_tcp_ep_tx_completed(ep, sent_length);

    uct_iface_trace_am(&iface->super, UCT_AM_TRACE_TYPE_SEND, hdr->am_id,
                       /* the function will be invoked only in case of
                        * data tracing is enabled */
                       uct_tcp_ep_am_sendv_get_trace_payload(hdr, header,
                                                             &iov[2], short_sendv),
                       hdr->length, "SEND: ep %p fd %d sent %zu/%zu bytes, "
                       "moved by offset %zu, iov cnt %zu "
                       "[addr %p len %zu] [addr %p len %zu]",
                       ep, ep->fd, sent_length, ep->tx.length,
                       ep->tx.offset, iov_cnt,
                       /* print user-defined header or
                        * first iovec with a payload */
                       ((iov_cnt > 1) ? iov[1].iov_base : NULL),
                       ((iov_cnt > 1) ? iov[1].iov_len  : 0),
                       /* print first/second iovec with a payload */
                       ((iov_cnt > 2) ? iov[2].iov_base : NULL),
                       ((iov_cnt > 2) ? iov[2].iov_len  : 0));

    uct_tcp_ep_check_tx_completion(ep);

    return UCS_OK;
}

static void uct_tcp_ep_post_put_ack(uct_tcp_ep_t *ep)
{
    uct_tcp_am_hdr_t *hdr  = NULL;
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    uct_tcp_ep_put_ack_hdr_t *put_ack;
    ucs_status_t status;

    /* Make sure that we are sending nothing through this EP at the moment.
     * This check is needed to avoid mixing AM/PUT data sent from this EP
     * and this PUT ACK message */
    status = uct_tcp_ep_am_prepare(iface, ep,
                                   UCT_TCP_EP_PUT_ACK_AM_ID, &hdr);
    if (status != UCS_OK) {
        if (status == UCS_ERR_NO_RESOURCE) {
            ep->flags |= UCT_TCP_EP_FLAG_PUT_RX_SENDING_ACK;
        } else {
            ucs_error("tcp_ep %p: failed to prepare AM data", ep);
        }
        return;
    }

    /* Send PUT ACK to confirm completing PUT operations with
     * the last received sequence number == ep::rx::put_sn */
    ucs_assertv(hdr != NULL, "ep=%p", ep);
    hdr->length = sizeof(*put_ack);
    put_ack     = (uct_tcp_ep_put_ack_hdr_t*)(hdr + 1);
    put_ack->sn = ep->rx.put_sn;

    uct_tcp_ep_am_send(ep, hdr);

    /* If sending PUT ACK was OK, always remove SENDING ACK flag
     * as the function can be called from outstanding progress */
    ep->flags &= ~UCT_TCP_EP_FLAG_PUT_RX_SENDING_ACK;
}

ucs_status_t uct_tcp_ep_am_short(uct_ep_h uct_ep, uint8_t am_id, uint64_t header,
                                 const void *payload, unsigned length)
{
    uct_tcp_ep_t *ep       = ucs_derived_of(uct_ep, uct_tcp_ep_t);
    uct_tcp_iface_t *iface = ucs_derived_of(uct_ep->iface, uct_tcp_iface_t);
    uct_tcp_am_hdr_t *hdr  = NULL;
    struct iovec iov[UCT_TCP_EP_AM_SHORTV_IOV_COUNT];
    uint32_t UCS_V_UNUSED payload_length;
    size_t offset;
    ucs_status_t status;

    UCT_CHECK_LENGTH(length + sizeof(header), 0,
                     iface->config.tx_seg_size - sizeof(uct_tcp_am_hdr_t),
                     "am_short");
    UCT_CHECK_AM_ID(am_id);

    status = uct_tcp_ep_am_prepare(iface, ep, am_id, &hdr);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assertv(hdr != NULL, "ep=%p", ep);

    /* Save the length of the payload, because hdr (ep::buf)
     * can be released inside `uct_tcp_ep_am_send` call */
    hdr->length = payload_length = length + sizeof(header);

    if (length <= iface->config.sendv_thresh) {
        uct_am_short_fill_data(hdr + 1, header, payload, length);
        status = uct_tcp_ep_am_send(ep, hdr);
        if (ucs_unlikely(status != UCS_OK)) {
            return status;
        }

        UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, payload_length);
    } else {
        iov[0].iov_base = hdr;
        iov[0].iov_len  = sizeof(*hdr);

        iov[1].iov_base = &header;
        iov[1].iov_len  = sizeof(header);

        iov[2].iov_base = (void*)payload;
        iov[2].iov_len  = length;

        status = uct_tcp_ep_am_sendv(ep, 1, hdr, iface->config.tx_seg_size,
                                     &header, iov, UCT_TCP_EP_AM_SHORTV_IOV_COUNT);
        if (ucs_unlikely(status != UCS_OK)) {
            return status;
        }

        UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, payload_length);

        if (uct_tcp_ep_ctx_buf_need_progress(&ep->tx)) {
            /* Copy only user's header and payload to the TX buffer,
             * TCP AM header is placed at the beginning of the buffer */
            offset = ((ep->tx.offset >= sizeof(*hdr)) ?
                      (ep->tx.offset - sizeof(*hdr)) : 0);

            ucs_iov_copy(&iov[1], UCT_TCP_EP_AM_SHORTV_IOV_COUNT - 1,
                         offset, UCS_PTR_BYTE_OFFSET(hdr + 1, offset),
                         ep->tx.length - sizeof(*hdr) - offset,
                         UCS_IOV_COPY_TO_BUF);
        }
    }

    return status;
}

ssize_t uct_tcp_ep_am_bcopy(uct_ep_h uct_ep, uint8_t am_id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags)
{
    uct_tcp_ep_t *ep       = ucs_derived_of(uct_ep, uct_tcp_ep_t);
    uct_tcp_iface_t *iface = ucs_derived_of(uct_ep->iface, uct_tcp_iface_t);
    uct_tcp_am_hdr_t *hdr  = NULL;
    uint32_t payload_length;
    ucs_status_t status;

    UCT_CHECK_AM_ID(am_id);

    status = uct_tcp_ep_am_prepare(iface, ep, am_id, &hdr);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assertv(hdr != NULL, "ep=%p", ep);

    /* Save the length of the payload, because hdr (ep::buf)
     * can be released inside `uct_tcp_ep_am_send` call */
    hdr->length = payload_length = pack_cb(hdr + 1, arg);

    status = uct_tcp_ep_am_send(ep, hdr);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, payload_length);

    return payload_length;
}

static inline ucs_status_t
uct_tcp_ep_prepare_zcopy(uct_tcp_iface_t *iface, uct_tcp_ep_t *ep, uint8_t am_id,
                         const void *header, unsigned header_length,
                         const uct_iov_t *iov, size_t iovcnt, const char *name,
                         size_t *zcopy_payload_p, uct_tcp_ep_zcopy_tx_t **ctx_p)
{
    uct_tcp_am_hdr_t *hdr = NULL;
    size_t io_vec_cnt;
    ucs_iov_iter_t uct_iov_iter;
    uct_tcp_ep_zcopy_tx_t *ctx;
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, iface->config.zcopy.max_iov, name);
    UCT_CHECK_LENGTH(header_length, 0, iface->config.zcopy.max_hdr, name);

    status = uct_tcp_ep_am_prepare(iface, ep, am_id, &hdr);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    ucs_assertv(hdr != NULL, "ep=%p", ep);

    ctx          = ucs_derived_of(hdr, uct_tcp_ep_zcopy_tx_t);
    ctx->iov_cnt = 0;

    /* TCP transport header */
    ctx->iov[ctx->iov_cnt].iov_base = hdr;
    ctx->iov[ctx->iov_cnt].iov_len  = sizeof(*hdr);
    ctx->iov_cnt++;

    /* User-defined or TCP internal protocol header */
    if (header_length != 0) {
        ucs_assert(header != NULL);
        ctx->iov[ctx->iov_cnt].iov_base = (void*)header;
        ctx->iov[ctx->iov_cnt].iov_len  = header_length;
        ctx->iov_cnt++;
    }

    /* User-defined payload */
    ucs_iov_iter_init(&uct_iov_iter);
    io_vec_cnt       = iovcnt; 
    *zcopy_payload_p = uct_iov_to_iovec(&ctx->iov[ctx->iov_cnt], &io_vec_cnt,
                                        iov, iovcnt, SIZE_MAX, &uct_iov_iter);
    *ctx_p           = ctx;
    ctx->iov_cnt    += io_vec_cnt;

    return UCS_OK;
}

ucs_status_t uct_tcp_ep_am_zcopy(uct_ep_h uct_ep, uint8_t am_id, const void *header,
                                 unsigned header_length, const uct_iov_t *iov,
                                 size_t iovcnt, unsigned flags,
                                 uct_completion_t *comp)
{
    uct_tcp_ep_t *ep           = ucs_derived_of(uct_ep, uct_tcp_ep_t);
    uct_tcp_iface_t *iface     = ucs_derived_of(uct_ep->iface, uct_tcp_iface_t);
    uct_tcp_ep_zcopy_tx_t *ctx = NULL;
    size_t payload_length      = 0;
    ucs_status_t status;

    UCT_CHECK_LENGTH(header_length + uct_iov_total_length(iov, iovcnt), 0,
                     iface->config.rx_seg_size - sizeof(uct_tcp_am_hdr_t),
                     "am_zcopy");
    UCT_CHECK_AM_ID(am_id);

    status = uct_tcp_ep_prepare_zcopy(iface, ep, am_id, header, header_length,
                                      iov, iovcnt, "am_zcopy", &payload_length,
                                      &ctx);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    ctx->super.length = payload_length + header_length;

    status = uct_tcp_ep_am_sendv(ep, 0, &ctx->super, iface->config.rx_seg_size,
                                 header, ctx->iov, ctx->iov_cnt);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    UCT_TL_EP_STAT_OP(&ep->super, AM, ZCOPY, payload_length + header_length);

    if (uct_tcp_ep_ctx_buf_need_progress(&ep->tx)) {
        uct_tcp_ep_set_outstanding_zcopy(iface, ep, ctx, header,
                                         header_length, comp);
        return UCS_INPROGRESS;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_tcp_ep_put_comp_add(uct_tcp_ep_t *ep, uct_completion_t *comp, int wait_sn)
{
    uct_tcp_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_tcp_iface_t);
    uct_tcp_ep_put_completion_t *put_comp;

    if (comp == NULL) {
        return UCS_OK;
    }

    put_comp = ucs_mpool_get_inline(&iface->tx_mpool);
    if (ucs_unlikely(put_comp == NULL)) {
        ucs_error("tcp_ep %p: unable to allocate PUT completion from mpool",
                  ep);
        return UCS_ERR_NO_MEMORY;
    }

    put_comp->wait_put_sn = ep->tx.put_sn;
    put_comp->comp        = comp;
    ucs_queue_push(&ep->put_comp_q, &put_comp->elem);

    return UCS_OK;
}

ucs_status_t uct_tcp_ep_put_zcopy(uct_ep_h uct_ep, const uct_iov_t *iov,
                                  size_t iovcnt, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_tcp_ep_t *ep                 = ucs_derived_of(uct_ep, uct_tcp_ep_t);
    uct_tcp_iface_t *iface           = ucs_derived_of(uct_ep->iface,
                                                      uct_tcp_iface_t);
    uct_tcp_ep_zcopy_tx_t *ctx       = NULL;
    uct_tcp_ep_put_req_hdr_t put_req = {0}; /* Suppress Cppcheck false-positive */
    ucs_status_t status;

    UCT_CHECK_LENGTH(sizeof(put_req) + uct_iov_total_length(iov, iovcnt), 0,
                     UCT_TCP_EP_PUT_ZCOPY_MAX - sizeof(uct_tcp_am_hdr_t),
                     "put_zcopy");

    status = uct_tcp_ep_prepare_zcopy(iface, ep, UCT_TCP_EP_PUT_REQ_AM_ID,
                                      &put_req, sizeof(put_req),
                                      iov, iovcnt, "put_zcopy",
                                      /* Set a payload length directly to the
                                       * TX length, since PUT Zcopy doesn't
                                       * set the payload length to TCP AM hdr */
                                      &ep->tx.length, &ctx);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    ctx->super.length = sizeof(put_req);
    put_req.addr      = remote_addr;
    put_req.length    = ep->tx.length;
    put_req.sn        = ep->tx.put_sn + 1;

    status = uct_tcp_ep_am_sendv(ep, 0, &ctx->super, UCT_TCP_EP_PUT_ZCOPY_MAX,
                                 &put_req, ctx->iov, ctx->iov_cnt);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    ep->tx.put_sn++;

    if (!(ep->flags & UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK)) {
        /* Add UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK flag and increment iface
         * outstanding operations counter in order to ensure returning
         * UCS_INPROGRESS from flush functions and do progressing.
         * UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK flag has to be removed upon PUT
         * ACK message receiving if there are no other PUT operations in-flight */
        ep->flags |= UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK;
        uct_tcp_iface_outstanding_inc(iface);
    }

    UCT_TL_EP_STAT_OP(&ep->super, PUT, ZCOPY, put_req.length);

    status = uct_tcp_ep_put_comp_add(ep, comp, put_req.sn);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    if (uct_tcp_ep_ctx_buf_need_progress(&ep->tx)) {
        uct_tcp_ep_set_outstanding_zcopy(iface, ep, ctx, &put_req,
                                         sizeof(put_req), NULL);
    }

    return UCS_INPROGRESS;
}

ucs_status_t uct_tcp_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *req,
                                    unsigned flags)
{
    uct_tcp_ep_t *ep = ucs_derived_of(tl_ep, uct_tcp_ep_t);

    if (uct_tcp_ep_check_tx_res(ep) == UCS_OK) {
        return UCS_ERR_BUSY;
    }

    uct_pending_req_queue_push(&ep->pending_q, req);
    UCT_TL_EP_STAT_PEND(&ep->super);
    return UCS_OK;
}

void uct_tcp_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                              void *arg)
{
    uct_tcp_ep_t *ep = ucs_derived_of(tl_ep, uct_tcp_ep_t);
    uct_pending_req_priv_queue_t UCS_V_UNUSED *priv;

    uct_pending_queue_purge(priv, &ep->pending_q, 1, cb, arg);
}

ucs_status_t uct_tcp_ep_flush(uct_ep_h tl_ep, unsigned flags,
                              uct_completion_t *comp)
{
    uct_tcp_ep_t *ep = ucs_derived_of(tl_ep, uct_tcp_ep_t);
    ucs_status_t status;

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        /* TCP is able to cancel only pending operations, posted TX operations
         * couldn't be canceled, since some data was already sent to the peer
         * and the peer is waiting for the remaining part of the data */
        uct_ep_pending_purge(tl_ep,
                             (uct_pending_purge_callback_t)ucs_empty_function,
                             0);
        return UCS_OK;
    }

    status = uct_tcp_ep_check_tx_res(ep);
    if (status == UCS_ERR_NO_RESOURCE) {
        UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super);
        return UCS_ERR_NO_RESOURCE;
    }

    if (ep->flags & UCT_TCP_EP_FLAG_PUT_TX_WAITING_ACK) {
        status = uct_tcp_ep_put_comp_add(ep, comp, ep->tx.put_sn);
        if (status != UCS_OK) {
            return status;
        }

        return UCS_INPROGRESS;
    }

    UCT_TL_EP_STAT_FLUSH(&ep->super);
    return UCS_OK;
}

