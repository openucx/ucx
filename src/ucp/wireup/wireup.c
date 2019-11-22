/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "wireup.h"
#include "address.h"
#include "wireup_cm.h"
#include "wireup_ep.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_proxy_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_listener.h>
#include <ucp/tag/eager.h>
#include <ucs/async/async.h>
#include <ucs/datastruct/queue.h>

/*
 * Description of the protocol in UCX wiki:
 * https://github.com/openucx/ucx/wiki/Connection-establishment
 */

static size_t ucp_wireup_msg_pack(void *dest, void *arg)
{
    ucp_request_t *req = arg;
    *(ucp_wireup_msg_t*)dest = req->send.wireup;
    memcpy((ucp_wireup_msg_t*)dest + 1, req->send.buffer, req->send.length);
    return sizeof(ucp_wireup_msg_t) + req->send.length;
}

static const char* ucp_wireup_msg_str(uint8_t msg_type)
{
    switch (msg_type) {
    case UCP_WIREUP_MSG_PRE_REQUEST:
        return "PRE_REQ";
    case UCP_WIREUP_MSG_REQUEST:
        return "REQ";
    case UCP_WIREUP_MSG_REPLY:
        return "REP";
    case UCP_WIREUP_MSG_ACK:
        return "ACK";
    default:
        return "<unknown>";
    }
}

static ucp_lane_index_t ucp_wireup_get_msg_lane(ucp_ep_h ep, uint8_t msg_type)
{
    ucp_context_h   context           = ep->worker->context;
    ucp_ep_config_t *ep_config        = ucp_ep_config(ep);
    ucp_lane_index_t lane             = UCP_NULL_LANE;

    if (msg_type != UCP_WIREUP_MSG_ACK) {
        /* for request/response, try wireup_lane first */
        lane = ep_config->key.wireup_lane;
    }

    if (lane == UCP_NULL_LANE) {
        /* fallback to active messages lane */
        lane = ep_config->key.am_lane;
    }

    if (lane == UCP_NULL_LANE) {
        ucs_fatal("ep %p to %s: could not fine a lane to send CONN_%s%s",
                  ep, ucp_ep_peer_name(ep), ucp_wireup_msg_str(msg_type),
                  context->config.ext.unified_mode ?
                  ". try to set UCX_UNIFIED_MODE=n." : "");
    }

    return lane;
}

ucs_status_t ucp_wireup_msg_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep = req->send.ep;
    ssize_t packed_len;
    unsigned am_flags;

    if (req->send.wireup.type == UCP_WIREUP_MSG_REQUEST) {
        if (ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED) {
            ucs_trace("ep %p: not sending wireup message - remote already connected",
                      ep);
            goto out;
        }
    } else if (req->send.wireup.type == UCP_WIREUP_MSG_PRE_REQUEST) {
        ucs_assert (!(ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED));
    }

    /* send the active message */
    req->send.lane = ucp_wireup_get_msg_lane(ep, req->send.wireup.type);

    am_flags = 0;
    if ((req->send.wireup.type == UCP_WIREUP_MSG_REQUEST) ||
        (req->send.wireup.type == UCP_WIREUP_MSG_PRE_REQUEST)) {
        am_flags |= UCT_SEND_FLAG_SIGNALED;
    }

    VALGRIND_CHECK_MEM_IS_DEFINED(&req->send.wireup, sizeof(req->send.wireup));
    VALGRIND_CHECK_MEM_IS_DEFINED(req->send.buffer, req->send.length);

    packed_len = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], UCP_AM_ID_WIREUP,
                                 ucp_wireup_msg_pack, req, am_flags);
    if (packed_len < 0) {
        if (packed_len != UCS_ERR_NO_RESOURCE) {
            ucs_error("failed to send wireup: %s",
                      ucs_status_string((ucs_status_t)packed_len));
        }
        return (ucs_status_t)packed_len;
    }

    switch (req->send.wireup.type) {
    case UCP_WIREUP_MSG_PRE_REQUEST:
        ep->flags |= UCP_EP_FLAG_CONNECT_PRE_REQ_SENT;
        break;
    case UCP_WIREUP_MSG_REQUEST:
        ep->flags |= UCP_EP_FLAG_CONNECT_REQ_SENT;
        break;
    case UCP_WIREUP_MSG_REPLY:
        ep->flags |= UCP_EP_FLAG_CONNECT_REP_SENT;
        break;
    case UCP_WIREUP_MSG_ACK:
        ep->flags |= UCP_EP_FLAG_CONNECT_ACK_SENT;
        break;
    }

out:
    ucs_free((void*)req->send.buffer);
    ucs_free(req);
    return UCS_OK;
}

static inline int ucp_wireup_is_ep_needed(ucp_ep_h ep)
{
    return (ep != NULL) && !(ep->flags & UCP_EP_FLAG_LISTENER);
}

/*
 * @param [in] rsc_tli  Resource index for every lane.
 */
static ucs_status_t
ucp_wireup_msg_send(ucp_ep_h ep, uint8_t type, uint64_t tl_bitmap,
                    const ucp_lane_index_t *lanes2remote)
{
    ucp_request_t* req;
    ucs_status_t status;
    void *address;

    ucs_assert(ep->cfg_index != (uint8_t)-1);

    /* We cannot allocate from memory pool because it's not thread safe
     * and this function may be called from any thread
     */
    req = ucs_malloc(sizeof(*req), "wireup_msg_req");
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    req->flags                   = 0;
    req->send.ep                 = ep;
    req->send.wireup.type        = type;
    req->send.wireup.err_mode    = ucp_ep_config(ep)->key.err_mode;
    req->send.wireup.conn_sn     = ep->conn_sn;
    req->send.wireup.src_ep_ptr  = (uintptr_t)ep;
    if (ep->flags & UCP_EP_FLAG_DEST_EP) {
        req->send.wireup.dest_ep_ptr = ucp_ep_dest_ep_ptr(ep);
    } else {
        req->send.wireup.dest_ep_ptr = 0;
    }

    req->send.uct.func           = ucp_wireup_msg_progress;
    req->send.datatype           = ucp_dt_make_contig(1);
    ucp_request_send_state_init(req, ucp_dt_make_contig(1), 0);

    /* pack all addresses */
    status = ucp_address_pack(ep->worker,
                              ucp_wireup_is_ep_needed(ep) ? ep : NULL,
                              tl_bitmap, UCP_ADDRESS_PACK_FLAG_ALL,
                              lanes2remote, &req->send.length, &address);
    if (status != UCS_OK) {
        ucs_free(req);
        ucs_error("failed to pack address: %s", ucs_status_string(status));
        return status;
    }

    req->send.buffer = address;

    ucp_request_send(req, 0);
    return UCS_OK;
}

static uint64_t ucp_wireup_get_ep_tl_bitmap(ucp_ep_h ep, ucp_lane_map_t lane_map)
{
    uint64_t         tl_bitmap = 0;
    ucp_lane_index_t lane;

    ucs_for_each_bit(lane, lane_map) {
        ucs_assert(lane < UCP_MAX_LANES);
        tl_bitmap |= UCS_BIT(ucp_ep_get_rsc_index(ep, lane));
    }

    return tl_bitmap;
}

/*
 * Select remote ep address for every remote address entry (because there
 * could be multiple ep addresses per entry). This selection is used to create
 * 'lanes2remote' mapping with the remote lane index for each local lane.
 */
static void
ucp_wireup_match_p2p_lanes(ucp_ep_h ep,
                           const ucp_unpacked_address_t *remote_address,
                           const unsigned *addr_indices,
                           ucp_lane_index_t *lanes2remote)
{
    const ucp_address_entry_t *address;
    unsigned address_index;
    ucp_lane_index_t lane, remote_lane;
    unsigned *ep_addr_indexes;
    unsigned ep_addr_index;
    uint64_t UCS_V_UNUSED used_remote_lanes;

    /* Initialize the counters of ep address index for each address entry */
    ep_addr_indexes = ucs_alloca(sizeof(ep_addr_index) *
                                 remote_address->address_count);
    for (address_index = 0; address_index < remote_address->address_count;
         ++address_index) {
        ep_addr_indexes[address_index] = 0;
    }

    /* Initialize lanes2remote array */
    for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
        lanes2remote[lane] = UCP_NULL_LANE;
    }

    used_remote_lanes = 0;
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (!ucp_ep_is_lane_p2p(ep, lane)) {
            continue;
        }

        /* Select next remote ep address within the address_index as specified
         * by addr_indices argument
         */
        address_index      = addr_indices[lane];
        address            = &remote_address->address_list[address_index];
        ep_addr_index      = ep_addr_indexes[address_index]++;
        remote_lane        = address->ep_addrs[ep_addr_index].lane;
        lanes2remote[lane] = remote_lane;

        if (used_remote_lanes & UCS_BIT(remote_lane)) {
            ucs_fatal("ep %p: remote lane %d is used more than once", ep,
                      remote_lane);
        }
        used_remote_lanes |= UCS_BIT(remote_lane);

        ucs_trace("ep %p: lane[%d]->remote_lane[%d] (address[%d].ep_address[%d])",
                  ep, lane, remote_lane, address_index, ep_addr_index);
    }
}

static ucs_status_t
ucp_wireup_find_remote_p2p_addr(ucp_ep_h ep, ucp_lane_index_t remote_lane,
                               const ucp_unpacked_address_t *remote_address,
                               const uct_ep_addr_t **ep_addr_p,
                               const uct_device_addr_t **dev_addr_p)
{
    const ucp_address_entry_t *address;
    unsigned ep_addr_index;

    ucp_unpacked_address_for_each(address, remote_address) {
        for (ep_addr_index = 0; ep_addr_index < address->num_ep_addrs;
             ++ep_addr_index) {
            if (remote_lane == address->ep_addrs[ep_addr_index].lane) {
                *ep_addr_p  = address->ep_addrs[ep_addr_index].addr;
                *dev_addr_p = address->dev_addr;
                return UCS_OK;
            }
        }
    }

    return UCS_ERR_UNREACHABLE;
}

ucs_status_t
ucp_wireup_connect_local(ucp_ep_h ep,
                         const ucp_unpacked_address_t *remote_address,
                         const ucp_lane_index_t *lanes2remote)
{
    ucp_lane_index_t lane, remote_lane;
    const uct_device_addr_t *dev_addr;
    const uct_ep_addr_t *ep_addr;
    ucs_status_t status;

    ucs_trace("ep %p: connect local transports", ep);

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (!ucp_ep_is_lane_p2p(ep, lane)) {
            continue;
        }

        remote_lane = (lanes2remote == NULL) ? lane : lanes2remote[lane];

        status = ucp_wireup_find_remote_p2p_addr(ep, remote_lane, remote_address,
                                                 &ep_addr, &dev_addr);
        if (status != UCS_OK) {
            ucs_error("ep %p: no remote ep address for lane[%d]->remote_lane[%d]",
                      ep, lane, remote_lane);
           return status;
        }

        status = uct_ep_connect_to_ep(ep->uct_eps[lane], dev_addr, ep_addr);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

void ucp_wireup_remote_connected(ucp_ep_h ep)
{
    ucp_lane_index_t lane;

    if (ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED) {
        return;
    }

    ucs_trace("ep %p: remote connected", ep);
    ep->flags |= UCP_EP_FLAG_REMOTE_CONNECTED;

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (ucp_ep_is_lane_p2p(ep, lane)) {
            ucs_assert(ucp_wireup_ep_test(ep->uct_eps[lane]));
        }
        if (ucp_wireup_ep_test(ep->uct_eps[lane])) {
            ucp_wireup_ep_remote_connected(ep->uct_eps[lane]);
        }
    }

    ucs_assert(ep->flags & UCP_EP_FLAG_DEST_EP);
}


static ucs_status_t
ucp_wireup_init_lanes_by_request(ucp_worker_h worker, ucp_ep_h ep,
                                 unsigned ep_init_flags,
                                 const ucp_unpacked_address_t *remote_address,
                                 unsigned *addr_indices)
{
    ucs_status_t status = ucp_wireup_init_lanes(ep, ep_init_flags,
                                                remote_address, addr_indices);
    if (status == UCS_OK) {
        return UCS_OK;
    }

    ucp_worker_set_ep_failed(worker, ep, NULL, UCP_NULL_LANE, status);
    return status;
}


static UCS_F_NOINLINE void
ucp_wireup_process_pre_request(ucp_worker_h worker, const ucp_wireup_msg_t *msg,
                               const ucp_unpacked_address_t *remote_address)
{
    unsigned ep_init_flags = UCP_EP_INIT_CREATE_AM_LANE;
    unsigned addr_indices[UCP_MAX_LANES];
    ucs_status_t status;
    ucp_ep_h ep;

    ucs_assert(msg->type == UCP_WIREUP_MSG_PRE_REQUEST);
    ucs_assert(msg->dest_ep_ptr != 0);
    ucs_trace("got wireup pre_request from 0x%"PRIx64" src_ep 0x%lx dst_ep 0x%lx conn_sn %d",
              remote_address->uuid, msg->src_ep_ptr, msg->dest_ep_ptr, msg->conn_sn);

    /* wireup pre_request for a specific ep */
    ep = ucp_worker_get_ep_by_ptr(worker, msg->dest_ep_ptr);
    ucs_assert(ep->flags & UCP_EP_FLAG_SOCKADDR_PARTIAL_ADDR);

    ucp_ep_update_dest_ep_ptr(ep, msg->src_ep_ptr);
    ucp_ep_flush_state_reset(ep);

    if (ucp_ep_config(ep)->key.err_mode == UCP_ERR_HANDLING_MODE_PEER) {
        ep_init_flags |= UCP_EP_INIT_ERR_MODE_PEER_FAILURE;
    }

    /* initialize transport endpoints */
    status = ucp_wireup_init_lanes_by_request(worker, ep, ep_init_flags,
                                              remote_address, addr_indices);
    if (status != UCS_OK) {
        return;
    }

    status = ucp_wireup_send_request(ep);
    if (status != UCS_OK) {
        ucp_ep_cleanup_lanes(ep);
    }
}

static UCS_F_NOINLINE void
ucp_wireup_process_request(ucp_worker_h worker, const ucp_wireup_msg_t *msg,
                           const ucp_unpacked_address_t *remote_address)
{
    uint64_t remote_uuid   = remote_address->uuid;
    uint64_t tl_bitmap     = 0;
    int send_reply         = 0;
    unsigned ep_init_flags = 0;
    ucp_rsc_index_t lanes2remote[UCP_MAX_LANES];
    unsigned addr_indices[UCP_MAX_LANES];
    ucs_status_t status;
    ucp_ep_flags_t listener_flag;
    ucp_ep_h ep;

    ucs_assert(msg->type == UCP_WIREUP_MSG_REQUEST);
    ucs_trace("got wireup request from 0x%"PRIx64" src_ep 0x%lx dst_ep 0x%lx conn_sn %d",
              remote_address->uuid, msg->src_ep_ptr, msg->dest_ep_ptr, msg->conn_sn);

    if (msg->dest_ep_ptr != 0) {
        /* wireup request for a specific ep */
        ep = ucp_worker_get_ep_by_ptr(worker, msg->dest_ep_ptr);
        ucp_ep_update_dest_ep_ptr(ep, msg->src_ep_ptr);
        if (!(ep->flags & UCP_EP_FLAG_LISTENER)) {
            /* Reset flush state only if it's not a client-server wireup on
             * server side with long address exchange when listener (united with
             * flush state) should be valid until user's callback invoking */
            ucp_ep_flush_state_reset(ep);
        }
        ep_init_flags |= UCP_EP_INIT_CREATE_AM_LANE;
    } else {
        ep = ucp_ep_match_retrieve_exp(&worker->ep_match_ctx, remote_uuid,
                                       msg->conn_sn ^ (remote_uuid == worker->uuid));
        if (ep == NULL) {
            /* Create a new endpoint if does not exist */
            status = ucp_ep_new(worker, remote_address->name, "remote-request",
                                &ep);
            if (status != UCS_OK) {
                return;
            }

            /* add internal endpoint to hash */
            ep->conn_sn = msg->conn_sn;
            ucp_ep_match_insert_unexp(&worker->ep_match_ctx, remote_uuid, ep);
        } else {
            ucp_ep_flush_state_reset(ep);
        }

        ucp_ep_update_dest_ep_ptr(ep, msg->src_ep_ptr);

        /*
         * If the current endpoint already sent a connection request, we have a
         * "simultaneous connect" situation. In this case, only one of the endpoints
         * (instead of both) should respect the connect request, otherwise they
         * will end up being connected to "internal" endpoints on the remote side
         * instead of each other. We use the uniqueness of worker uuid to decide
         * which connect request should be ignored.
         */
        if ((ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED) && (remote_uuid > worker->uuid)) {
            ucs_trace("ep %p: ignoring simultaneous connect request", ep);
            ep->flags |= UCP_EP_FLAG_CONNECT_REQ_IGNORED;
            return;
        }
    }

    if (ep->flags & UCP_EP_FLAG_LISTENER) {
        /* If this is an ep on a listener (server) that received a partial
         * worker address from the client, then the following lanes initialization
         * will be done after an aux lane was already created on this ep.
         * Therefore, remove the existing aux endpoint since will need to create
         * new lanes now */
        ucp_ep_cleanup_lanes(ep);
    }

    if (msg->err_mode == UCP_ERR_HANDLING_MODE_PEER) {
        ep_init_flags |= UCP_EP_INIT_ERR_MODE_PEER_FAILURE;
    }

    /* Initialize lanes (possible destroy existing lanes) */
    status = ucp_wireup_init_lanes_by_request(worker, ep, ep_init_flags,
                                              remote_address, addr_indices);
    if (status != UCS_OK) {
        return;
    }

    ucp_wireup_match_p2p_lanes(ep, remote_address, addr_indices, lanes2remote);

    /* Send a reply if remote side does not have ep_ptr (active-active flow) or
     * there are p2p lanes (client-server flow)
     */
    send_reply = (msg->dest_ep_ptr == 0) || ucp_ep_config(ep)->p2p_lanes;

    /* Connect p2p addresses to remote endpoint */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        status = ucp_wireup_connect_local(ep, remote_address, lanes2remote);
        if (status != UCS_OK) {
            return;
        }

        tl_bitmap  = ucp_wireup_get_ep_tl_bitmap(ep,
                                                 ucp_ep_config(ep)->p2p_lanes);
        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;

        ucs_assert(send_reply);
    }

    /* mark the endpoint as connected to remote */
    if (!ucp_ep_config(ep)->p2p_lanes) {
        ucp_wireup_remote_connected(ep);
    }

    if (send_reply) {

        listener_flag = ep->flags & UCP_EP_FLAG_LISTENER;
        /* Remove this flag at this point if it's set
         * (so that address packing would be correct) */
        ep->flags &= ~UCP_EP_FLAG_LISTENER;

        ucs_trace("ep %p: sending wireup reply", ep);
        status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REPLY, tl_bitmap,
                                     lanes2remote);
        if (status != UCS_OK) {
            return;
        }

        /* Restore saved flag value */
        ep->flags |= listener_flag;
    } else {
        /* if in client-server flow, schedule invoking the user's callback
         * (if server is connected) from the main thread */
        if (ucs_test_all_flags(ep->flags,
                               (UCP_EP_FLAG_LISTENER | UCP_EP_FLAG_LOCAL_CONNECTED))) {
            ucp_listener_schedule_accept_cb(ep);
        }
    }
}

static unsigned ucp_wireup_send_msg_ack(void *arg)
{
    ucp_ep_h ep = (ucp_ep_h)arg;
    ucp_rsc_index_t rsc_tli[UCP_MAX_LANES];
    ucs_status_t status;

    /* Send ACK without any address, we've already sent it as part of the request */
    ucs_trace("ep %p: sending wireup ack", ep);

    memset(rsc_tli, UCP_NULL_RESOURCE, sizeof(rsc_tli));
    status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_ACK, 0, rsc_tli);
    return (status == UCS_OK);
}

int ucp_wireup_msg_ack_cb_pred(const ucs_callbackq_elem_t *elem, void *arg)
{
    return ((elem->arg == arg) && (elem->cb == ucp_wireup_send_msg_ack));
}

static UCS_F_NOINLINE void
ucp_wireup_process_reply(ucp_worker_h worker, const ucp_wireup_msg_t *msg,
                         const ucp_unpacked_address_t *remote_address)
{
    uct_worker_cb_id_t cb_id = UCS_CALLBACKQ_ID_NULL;
    ucs_status_t status;
    ucp_ep_h ep;
    int ack;

    ep = ucp_worker_get_ep_by_ptr(worker, msg->dest_ep_ptr);

    ucs_assert(msg->type == UCP_WIREUP_MSG_REPLY);
    ucs_assert((!(ep->flags & UCP_EP_FLAG_LISTENER)));
    ucs_trace("ep %p: got wireup reply src_ep 0x%lx dst_ep 0x%lx sn %d", ep,
              msg->src_ep_ptr, msg->dest_ep_ptr, msg->conn_sn);

    ucp_ep_match_remove_ep(&worker->ep_match_ctx, ep);
    ucp_ep_update_dest_ep_ptr(ep, msg->src_ep_ptr);
    ucp_ep_flush_state_reset(ep);

    /* Connect p2p addresses to remote endpoint */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {

        /*
         * In the wireup reply message, the lane indexes specify which
         * **receiver** ep lane should be connected to a given ep address. So we
         * don't pass 'lanes2remote' mapping, and use local lanes directly.
         */
        status = ucp_wireup_connect_local(ep, remote_address, NULL);
        if (status != UCS_OK) {
            return;
        }

        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
        ack = 1;
    } else {
        ack = 0;
    }

    ucp_wireup_remote_connected(ep);

    if (ack) {
        /* Send `UCP_WIREUP_MSG_ACK` from progress function
         * to avoid calling UCT routines from an async thread */
        uct_worker_progress_register_safe(worker->uct,
                                          ucp_wireup_send_msg_ack, ep,
                                          UCS_CALLBACKQ_FLAG_ONESHOT, &cb_id);
    }
}

static UCS_F_NOINLINE
void ucp_wireup_process_ack(ucp_worker_h worker, const ucp_wireup_msg_t *msg)
{
    ucp_ep_h ep;

    ep = ucp_worker_get_ep_by_ptr(worker, msg->dest_ep_ptr);

    ucs_assert(msg->type == UCP_WIREUP_MSG_ACK);
    ucs_trace("ep %p: got wireup ack", ep);

    ucs_assert(ep->flags & UCP_EP_FLAG_DEST_EP);
    ucs_assert(ep->flags & UCP_EP_FLAG_CONNECT_REP_SENT);
    ucs_assert(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);

    ucp_wireup_remote_connected(ep);

    /* if this ack is received as part of the client-server flow, when handling
     * a large worker address from the client, invoke the cached user callback
     * from the main thread */
    if (ep->flags & UCP_EP_FLAG_LISTENER) {
        ucp_listener_schedule_accept_cb(ep);
    }
}

static ucs_status_t ucp_wireup_msg_handler(void *arg, void *data,
                                           size_t length, unsigned flags)
{
    ucp_worker_h worker   = arg;
    ucp_wireup_msg_t *msg = data;
    ucp_unpacked_address_t remote_address;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    status = ucp_address_unpack(worker, msg + 1, UINT64_MAX, &remote_address);
    if (status != UCS_OK) {
        ucs_error("failed to unpack address: %s", ucs_status_string(status));
        goto out;
    }

    if (msg->type == UCP_WIREUP_MSG_ACK) {
        ucs_assert(remote_address.address_count == 0);
        ucp_wireup_process_ack(worker, msg);
    } else if (msg->type == UCP_WIREUP_MSG_PRE_REQUEST) {
        ucp_wireup_process_pre_request(worker, msg, &remote_address);
    } else if (msg->type == UCP_WIREUP_MSG_REQUEST) {
        ucp_wireup_process_request(worker, msg, &remote_address);
    } else if (msg->type == UCP_WIREUP_MSG_REPLY) {
        ucp_wireup_process_reply(worker, msg, &remote_address);
    } else {
        ucs_bug("invalid wireup message");
    }

    ucs_free(remote_address.address_list);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;
}

void ucp_wireup_assign_lane(ucp_ep_h ep, ucp_lane_index_t lane, uct_ep_h uct_ep,
                            int is_wireup_ep_connected, const char *info)
{
    /* If ep already exists, it's a wireup proxy, and we need to update its
     * next_ep instead of replacing it.
     */
    if (ep->uct_eps[lane] == NULL) {
        ucs_trace("ep %p: assign uct_ep[%d]=%p%s", ep, lane, uct_ep, info);
        ep->uct_eps[lane] = uct_ep;
    } else {
        ucs_assert(ucp_wireup_ep_test(ep->uct_eps[lane]));
        ucs_trace("ep %p: wireup uct_ep[%d]=%p next set to %p%s", ep, lane,
                  ep->uct_eps[lane], uct_ep, info);
        ucp_wireup_ep_set_next_ep(ep->uct_eps[lane], uct_ep);
        if (is_wireup_ep_connected) {
            ucp_wireup_ep_remote_connected(ep->uct_eps[lane]);
        }
    }
}

static uct_ep_h ucp_wireup_extract_lane(ucp_ep_h ep, ucp_lane_index_t lane)
{
    uct_ep_h uct_ep = ep->uct_eps[lane];

    if ((uct_ep != NULL) && ucp_wireup_ep_test(uct_ep)) {
        return ucp_wireup_ep_extract_next_ep(uct_ep);
    } else {
        ep->uct_eps[lane] = NULL;
        return uct_ep;
    }
}

ucs_status_t
ucp_wireup_connect_lane(ucp_ep_h ep, unsigned ep_init_flags,
                        ucp_lane_index_t lane,
                        const ucp_unpacked_address_t *remote_address,
                        unsigned addr_index)
{
    ucp_worker_h worker = ep->worker;
    int connect_aux;
    ucp_lane_index_t proxy_lane;
    ucp_rsc_index_t rsc_index;
    ucp_worker_iface_t *wiface;
    uct_ep_params_t uct_ep_params;
    uct_ep_h uct_ep;
    ucs_status_t status;

    ucs_trace("ep %p: connect lane[%d]", ep, lane);

    ucs_assert(lane != ucp_ep_get_cm_lane(ep));

    ucs_assert_always(remote_address != NULL);
    ucs_assert_always(remote_address->address_list != NULL);
    ucs_assert_always(addr_index <= remote_address->address_count);

    proxy_lane = ucp_ep_get_proxy_lane(ep, lane);
    rsc_index  = ucp_ep_get_rsc_index(ep, lane);
    wiface     = ucp_worker_iface(worker, rsc_index);

    /*
     * if the selected transport can be connected directly to the remote
     * interface, just create a connected UCT endpoint.
     */
    if ((wiface->attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) &&
        ((ep->uct_eps[lane] == NULL) || ucp_wireup_ep_test(ep->uct_eps[lane])))
    {
        if ((proxy_lane == UCP_NULL_LANE) || (proxy_lane == lane)) {
            /* create an endpoint connected to the remote interface */
            ucs_trace("ep %p: connect uct_ep[%d] to addr[%d]", ep, lane,
                      addr_index);
            uct_ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE    |
                                       UCT_EP_PARAM_FIELD_DEV_ADDR |
                                       UCT_EP_PARAM_FIELD_IFACE_ADDR;
            uct_ep_params.iface      = wiface->iface;
            uct_ep_params.dev_addr   = remote_address->address_list[addr_index].dev_addr;
            uct_ep_params.iface_addr = remote_address->address_list[addr_index].iface_addr;
            status = uct_ep_create(&uct_ep_params, &uct_ep);
            if (status != UCS_OK) {
                /* coverity[leaked_storage] */
                return status;
            }

            ucp_wireup_assign_lane(ep, lane, uct_ep, 1, "");
        }

        ucp_worker_iface_progress_ep(wiface);
        return UCS_OK;
    }

    /*
     * create a wireup endpoint which will start connection establishment
     * protocol using an auxiliary transport.
     */
    if (wiface->attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {

        /* For now, p2p transports have no reason to have proxy */
        ucs_assert_always(proxy_lane == UCP_NULL_LANE);

        if (ep->uct_eps[lane] == NULL) {
            status = ucp_wireup_ep_create(ep, &uct_ep);
            if (status != UCS_OK) {
                /* coverity[leaked_storage] */
                return status;
            }

            ucs_trace("ep %p: assign uct_ep[%d]=%p wireup", ep, lane, uct_ep);
            ep->uct_eps[lane] = uct_ep;
        } else {
            uct_ep = ep->uct_eps[lane];
            ucs_assert(ucp_wireup_ep_test(uct_ep));
        }

        if (!(ep_init_flags & (UCP_EP_INIT_CM_WIREUP_CLIENT))) {
            ucs_trace("ep %p: connect uct_ep[%d]=%p to addr[%d] wireup", ep,
                      lane, uct_ep, addr_index);
            connect_aux = !(ep_init_flags & (UCP_EP_INIT_CM_WIREUP_CLIENT |
                                             UCP_EP_INIT_CM_WIREUP_SERVER)) &&
                          (lane == ucp_ep_get_wireup_msg_lane(ep));
            status = ucp_wireup_ep_connect(ep->uct_eps[lane], ep_init_flags,
                                           rsc_index, connect_aux,
                                           remote_address);
            if (status != UCS_OK) {
                return status;
            }
        }

        ucp_worker_iface_progress_ep(wiface);

        return UCS_OK;
    }

    return UCS_ERR_UNREACHABLE;
}

ucs_status_t ucp_wireup_resolve_proxy_lanes(ucp_ep_h ep)
{
    ucp_lane_index_t lane, proxy_lane;
    uct_iface_attr_t *iface_attr;
    uct_ep_h uct_ep, signaling_ep;
    ucs_status_t status;

    /* point proxy endpoints */
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        proxy_lane = ucp_ep_get_proxy_lane(ep, lane);
        if (proxy_lane == UCP_NULL_LANE) {
            continue;
        }

        iface_attr = ucp_worker_iface_get_attr(ep->worker,
                                               ucp_ep_get_rsc_index(ep, lane));

        if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
            ucs_assert_always(iface_attr->cap.am.max_short <=
                              iface_attr->cap.am.max_bcopy);
        }

        /* Create a signaling ep to the proxy lane */
        if (proxy_lane == lane) {
            /* If proxy is to the same lane, temporarily remove the existing
             * UCT endpoint in there, so it could be assigned to the signaling
             * proxy ep. This can also be an endpoint contained inside a wireup
             * proxy, so ucp_wireup_extract_lane() handles both cases.
             */
            uct_ep = ucp_wireup_extract_lane(ep, proxy_lane);
            ucs_assert_always(uct_ep != NULL);
            status = ucp_signaling_ep_create(ep, uct_ep, 1, &signaling_ep);
            if (status != UCS_OK) {
                /* coverity[leaked_storage] */
                uct_ep_destroy(uct_ep);
                return status;
            }
        } else {
            status = ucp_signaling_ep_create(ep, ep->uct_eps[proxy_lane], 0,
                                             &signaling_ep);
            if (status != UCS_OK) {
                /* coverity[leaked_storage] */
                return status;
            }
        }

        ucs_trace("ep %p: lane[%d]=%p proxy_lane=%d", ep, lane, ep->uct_eps[lane],
                  proxy_lane);

        ucp_wireup_assign_lane(ep, lane, signaling_ep, 1, " (signaling proxy)");
    }

    return UCS_OK;
}

static void ucp_wireup_print_config(ucp_context_h context,
                                    const ucp_ep_config_key_t *key,
                                    const char *title,
                                    const unsigned *addr_indices,
                                    ucs_log_level_t log_level)
{
    char lane_info[128] = {0};
    ucp_lane_index_t lane;

    if (!ucs_log_is_enabled(log_level)) {
        return;
    }

    ucs_log(log_level, "%s: am_lane %d wireup_lane %d reachable_mds 0x%lx",
              title, key->am_lane, key->wireup_lane,
              key->reachable_md_map);

    for (lane = 0; lane < key->num_lanes; ++lane) {
        ucp_ep_config_lane_info_str(context, key, addr_indices, lane,
                                    UCP_NULL_RESOURCE, lane_info,
                                    sizeof(lane_info));
        ucs_log(log_level, "%s: %s", title, lane_info);
    }
}

int ucp_wireup_is_reachable(ucp_worker_h worker, ucp_rsc_index_t rsc_index,
                            const ucp_address_entry_t *ae)
{
    ucp_context_h context      = worker->context;
    ucp_worker_iface_t *wiface = ucp_worker_iface(worker, rsc_index);

    return (context->tl_rscs[rsc_index].tl_name_csum == ae->tl_name_csum) &&
           uct_iface_is_reachable(wiface->iface, ae->dev_addr, ae->iface_addr);
}

static void
ucp_wireup_get_reachable_mds(ucp_worker_h worker,
                             const ucp_unpacked_address_t *remote_address,
                             const ucp_ep_config_key_t *prev_key,
                             ucp_ep_config_key_t *key)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t ae_cmpts[UCP_MAX_MDS]; /* component index for each address entry */
    const ucp_address_entry_t *ae;
    ucp_rsc_index_t cmpt_index;
    ucp_rsc_index_t rsc_index;
    ucp_md_index_t dst_md_index;
    ucp_md_map_t ae_dst_md_map, dst_md_map;
    unsigned num_dst_mds;

    ae_dst_md_map = 0;
    ucs_for_each_bit(rsc_index, context->tl_bitmap) {
        ucp_unpacked_address_for_each(ae, remote_address) {
            if (ucp_wireup_is_reachable(worker, rsc_index, ae)) {
                ae_dst_md_map         |= UCS_BIT(ae->md_index);
                dst_md_index           = context->tl_rscs[rsc_index].md_index;
                ae_cmpts[ae->md_index] = context->tl_mds[dst_md_index].cmpt_index;
            }
        }
    }

    /* merge with previous configuration */
    dst_md_map  = ae_dst_md_map | prev_key->reachable_md_map;
    num_dst_mds = 0;
    ucs_for_each_bit(dst_md_index, dst_md_map) {
        cmpt_index = UCP_NULL_RESOURCE;
        /* remote md is reachable by the provided address */
        if (UCS_BIT(dst_md_index) & ae_dst_md_map) {
            cmpt_index = ae_cmpts[dst_md_index];
        }
        /* remote md is reachable by previous ep configuration */
        if (UCS_BIT(dst_md_index) & prev_key->reachable_md_map) {
            cmpt_index = ucp_ep_config_get_dst_md_cmpt(prev_key, dst_md_index);
            if (UCS_BIT(dst_md_index) & ae_dst_md_map) {
                /* we expect previous configuration will not conflict with the
                 * new one
                 */
                ucs_assert_always(cmpt_index == ae_cmpts[dst_md_index]);
            }
        }
        ucs_assert_always(cmpt_index != UCP_NULL_RESOURCE);
        key->dst_md_cmpts[num_dst_mds++] = cmpt_index;
    }
    ucs_assert(num_dst_mds == ucs_popcount(dst_md_map));

    key->reachable_md_map = dst_md_map;
}

ucs_status_t ucp_wireup_init_lanes(ucp_ep_h ep, unsigned ep_init_flags,
                                   const ucp_unpacked_address_t *remote_address,
                                   unsigned *addr_indices)
{
    ucp_worker_h worker = ep->worker;
    ucp_ep_config_key_t key;
    ucp_ep_cfg_index_t new_cfg_index;
    ucp_lane_index_t lane;
    ucs_status_t status;
    char str[32];
    ucp_wireup_ep_t *cm_wireup_ep;

    ucs_trace("ep %p: initialize lanes", ep);

    ucp_ep_config_key_reset(&key);
    ucp_ep_config_key_set_err_mode(&key, ep_init_flags);

    status = ucp_wireup_select_lanes(ep, ep_init_flags,
                                     worker->context->tl_bitmap, remote_address,
                                     addr_indices, &key);
    if (status != UCS_OK) {
        return status;
    }

    /* Get all reachable MDs from full remote address list */
    key.dst_md_cmpts = ucs_alloca(sizeof(*key.dst_md_cmpts) * UCP_MAX_MDS);
    ucp_wireup_get_reachable_mds(worker, remote_address, &ucp_ep_config(ep)->key,
                                 &key);

    /* Load new configuration */
    status = ucp_worker_get_ep_config(worker, &key, 1, &new_cfg_index);
    if (status != UCS_OK) {
        return status;
    }

    if (ep->cfg_index == new_cfg_index) {
        return UCS_OK; /* No change */
    }

    if ((ep->cfg_index != 0) && !ucp_ep_is_sockaddr_stub(ep)) {
        /*
         * TODO handle a case where we have to change lanes and reconfigure the ep:
         *
         * - if we already have uct ep connected to an address - move it to the new lane index
         * - if we don't yet have connection to an address - create it
         * - if an existing lane is not connected anymore - delete it (possibly)
         * - if the configuration has changed - replay all pending operations on all lanes -
         *   need that every pending callback would return, in case of failure, the number
         *   of lane it wants to be queued on.
         */
        ucs_debug("cannot reconfigure ep %p from [%d] to [%d]", ep, ep->cfg_index,
                  new_cfg_index);
        ucp_wireup_print_config(worker->context, &ucp_ep_config(ep)->key, "old",
			       NULL, UCS_LOG_LEVEL_ERROR);
        ucp_wireup_print_config(worker->context, &key, "new", NULL, UCS_LOG_LEVEL_ERROR);
        ucs_fatal("endpoint reconfiguration not supported yet");
    }

    cm_wireup_ep  = ucp_ep_get_cm_wireup_ep(ep);
    ep->cfg_index = new_cfg_index;
    ep->am_lane   = key.am_lane;

    snprintf(str, sizeof(str), "ep %p", ep);
    ucp_wireup_print_config(worker->context, &ucp_ep_config(ep)->key, str,
                            addr_indices, UCS_LOG_LEVEL_DEBUG);

    /* establish connections on all underlying endpoints */
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (ucp_ep_get_cm_lane(ep) == lane) {
            /* restore the cm lane after reconfiguration */
            ep->uct_eps[lane] = &cm_wireup_ep->super.super;
            continue;
        }

        status = ucp_wireup_connect_lane(ep, ep_init_flags, lane,
                                         remote_address, addr_indices[lane]);
        if (status != UCS_OK) {
            return status;
        }
    }

    status = ucp_wireup_resolve_proxy_lanes(ep);
    if (status != UCS_OK) {
        return status;
    }

    /* If we don't have a p2p transport, we're connected */
    if (!ucp_ep_config(ep)->p2p_lanes) {
        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
    }

    return UCS_OK;
}

ucs_status_t ucp_wireup_send_request(ucp_ep_h ep)
{
    ucp_rsc_index_t rsc_index;
    ucs_status_t status;
    uint64_t tl_bitmap;

    tl_bitmap = ucp_wireup_get_ep_tl_bitmap(ep, UCS_MASK(ucp_ep_num_lanes(ep)));

    /* TODO make sure such lane would exist */
    rsc_index = ucp_wireup_ep_get_aux_rsc_index(
                    ep->uct_eps[ucp_ep_get_wireup_msg_lane(ep)]);
    if (rsc_index != UCP_NULL_RESOURCE) {
        tl_bitmap |= UCS_BIT(rsc_index);
    }

    ucs_debug("ep %p: send wireup request (flags=0x%x)", ep, ep->flags);
    status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REQUEST, tl_bitmap, NULL);

    ep->flags |= UCP_EP_FLAG_CONNECT_REQ_QUEUED;

    return status;
}

static void ucp_wireup_connect_remote_purge_cb(uct_pending_req_t *self, void *arg)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_queue_head_t *queue = arg;

    ucs_trace_req("ep %p: extracted request %p from pending queue", req->send.ep,
                  req);
    ucs_queue_push(queue, (ucs_queue_elem_t*)&req->send.uct.priv);
}

ucs_status_t ucp_wireup_send_pre_request(ucp_ep_h ep)
{
    ucp_rsc_index_t rsc_tli[UCP_MAX_LANES];
    uint64_t tl_bitmap = UINT64_MAX;  /* pack full worker address */
    ucs_status_t status;

    ucs_assert(ep->flags & UCP_EP_FLAG_LISTENER);
    ucs_assert(!(ep->flags & UCP_EP_FLAG_CONNECT_PRE_REQ_QUEUED));
    memset(rsc_tli, UCP_NULL_RESOURCE, sizeof(rsc_tli));

    ucs_debug("ep %p: send wireup pre-request (flags=0x%x)", ep, ep->flags);
    status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_PRE_REQUEST, tl_bitmap, rsc_tli);

    ep->flags |= UCP_EP_FLAG_CONNECT_PRE_REQ_QUEUED;
    return status;
}

ucs_status_t ucp_wireup_connect_remote(ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucs_queue_head_t tmp_q;
    ucs_status_t status;
    ucp_request_t *req;
    uct_ep_h uct_ep;

    ucs_trace("ep %p: connect lane %d to remote peer", ep, lane);

    UCS_ASYNC_BLOCK(&ep->worker->async);

    /* checking again, with lock held, if already connected or connection is
     * in progress */
    if ((ep->flags & UCP_EP_FLAG_DEST_EP) ||
        ucp_wireup_ep_test(ep->uct_eps[lane])) {
        status = UCS_OK;
        goto out_unlock;
    }

    if (ucp_proxy_ep_test(ep->uct_eps[lane])) {
        /* signaling ep is not needed now since we will send wireup request
         * with signaling flag
         */
        uct_ep = ucp_proxy_ep_extract(ep->uct_eps[lane]);
        uct_ep_destroy(ep->uct_eps[lane]);
    } else {
        uct_ep = ep->uct_eps[lane];
    }

    ucs_assert(!(ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED));

    ucs_trace("ep %p: connect lane %d to remote peer with wireup ep", ep, lane);

    /* make ep->uct_eps[lane] a stub */
    status = ucp_wireup_ep_create(ep, &ep->uct_eps[lane]);
    if (status != UCS_OK) {
        goto err;
    }

    /* Extract all pending requests from the transport endpoint, otherwise they
     * will prevent the wireup message from being sent (because those requests
     * could not be progressed any more after switching to wireup proxy).
     */
    ucs_queue_head_init(&tmp_q);
    uct_ep_pending_purge(uct_ep, ucp_wireup_connect_remote_purge_cb, &tmp_q);

    /* the wireup ep should use the existing [am_lane] as next_ep */
    ucp_wireup_ep_set_next_ep(ep->uct_eps[lane], uct_ep);

    if (!(ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED)) {
        status = ucp_wireup_send_request(ep);
        if (status != UCS_OK) {
            goto err_destroy_wireup_ep;
        }
    }

    ucs_queue_for_each_extract(req, &tmp_q, send.uct.priv, 1) {
        ucs_trace_req("ep %p: requeue request %p after wireup request",
                      req->send.ep, req);
        status = uct_ep_pending_add(ep->uct_eps[lane], &req->send.uct,
                                    (req->send.uct.func == ucp_wireup_msg_progress) ||
                                    (req->send.uct.func == ucp_wireup_ep_progress_pending) ?
                                    UCT_CB_FLAG_ASYNC : 0);
        if (status != UCS_OK) {
            ucs_fatal("wireup proxy function must always return UCS_OK");
        }
    }

    goto out_unlock;

err_destroy_wireup_ep:
    uct_ep_destroy(ep->uct_eps[lane]);
err:
    ep->uct_eps[lane] = uct_ep; /* restore am lane */
out_unlock:
    UCS_ASYNC_UNBLOCK(&ep->worker->async);
    return status;
}

static void ucp_wireup_msg_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                                uint8_t id, const void *data, size_t length,
                                char *buffer, size_t max)
{
    ucp_context_h context       = worker->context;
    const ucp_wireup_msg_t *msg = data;
    ucp_unpacked_address_t unpacked_address;
    const ucp_address_entry_t *ae;
    ucp_tl_resource_desc_t *rsc;
    unsigned ep_addr_index;
    ucs_status_t status;
    char *p, *end;
    ucp_rsc_index_t tl;

    status = ucp_address_unpack(worker, msg + 1, ~UCP_ADDRESS_PACK_FLAG_TRACE,
                                &unpacked_address);
    if (status != UCS_OK) {
        strncpy(unpacked_address.name, "<malformed address>", UCP_WORKER_NAME_MAX);
        unpacked_address.uuid          = 0;
        unpacked_address.address_count = 0;
        unpacked_address.address_list  = NULL;
    }

    p   = buffer;
    end = buffer + max;

    snprintf(p, end - p,
             "WIREUP %s [%s uuid 0x%"PRIx64" src_ep 0x%lx dst_ep 0x%lx conn_sn %d]",
             ucp_wireup_msg_str(msg->type), unpacked_address.name,
             unpacked_address.uuid, msg->src_ep_ptr, msg->dest_ep_ptr,
             msg->conn_sn);
    p += strlen(p);

    if (unpacked_address.address_list == NULL) {
        return; /* No addresses were unpacked */
    }

    ucp_unpacked_address_for_each(ae, &unpacked_address) {
        ucs_for_each_bit(tl, context->tl_bitmap) {
            rsc = &context->tl_rscs[tl];
            if (ae->tl_name_csum == rsc->tl_name_csum) {
                snprintf(p, end - p, " "UCT_TL_RESOURCE_DESC_FMT,
                         UCT_TL_RESOURCE_DESC_ARG(&rsc->tl_rsc));
                p += strlen(p);
                break;
            }
        }
        snprintf(p, end - p, "/md[%d]", ae->md_index);
        p += strlen(p);

        for (ep_addr_index = 0; ep_addr_index < ae->num_ep_addrs;
             ++ep_addr_index) {
            snprintf(p, end - p, "/lane[%d]", ae->ep_addrs[ep_addr_index].lane);
            p += strlen(p);
        }
    }

    ucs_free(unpacked_address.address_list);
}

int ucp_worker_iface_is_tl_p2p(const uct_iface_attr_t *iface_attr)
{
    uint64_t flags = iface_attr->cap.flags;

    return (flags & UCT_IFACE_FLAG_CONNECT_TO_EP) &&
           !(flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE);
}

static ucp_err_handling_mode_t
ucp_ep_params_err_handling_mode(const ucp_ep_params_t *params)
{
    return (params->field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE) ?
           params->err_mode : UCP_ERR_HANDLING_MODE_NONE;
}

unsigned ucp_ep_init_flags(const ucp_worker_h worker,
                           const ucp_ep_params_t *params)
{
    unsigned flags = ucp_cm_ep_init_flags(worker, params);

    if (!ucp_worker_sockaddr_is_cm_proto(worker) &&
        (params->field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR)) {
        flags |= UCP_EP_INIT_CREATE_AM_LANE;
    }

    if (ucp_ep_params_err_handling_mode(params) == UCP_ERR_HANDLING_MODE_PEER) {
        flags |= UCP_EP_INIT_ERR_MODE_PEER_FAILURE;
    }

    return flags;
}

UCP_DEFINE_AM(UINT64_MAX, UCP_AM_ID_WIREUP, ucp_wireup_msg_handler,
              ucp_wireup_msg_dump, UCT_CB_FLAG_ASYNC);
