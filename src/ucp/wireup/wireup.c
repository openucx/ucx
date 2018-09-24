/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "wireup.h"
#include "address.h"
#include "wireup_ep.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_proxy_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_listener.h>
#include <ucp/tag/eager.h>
#include <ucs/arch/bitops.h>
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
    if (req->send.wireup.type == UCP_WIREUP_MSG_ACK) {
        req->send.lane = ucp_ep_get_am_lane(ep);
    } else {
        req->send.lane = ucp_ep_get_wireup_msg_lane(ep);
    }

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
            ucs_error("failed to send wireup: %s", ucs_status_string(packed_len));
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

static unsigned ucp_wireup_address_index(const unsigned *order,
                                         uint64_t tl_bitmap,
                                         ucp_rsc_index_t tl_index)
{
    return order[ucs_count_one_bits(tl_bitmap & UCS_MASK(tl_index))];
}

static inline int ucp_wireup_is_ep_needed(ucp_ep_h ep)
{
    return (ep != NULL) && !(ep->flags & UCP_EP_FLAG_LISTENER);
}

/*
 * @param [in] rsc_tli  Resource index for every lane.
 */
static ucs_status_t ucp_wireup_msg_send(ucp_ep_h ep, uint8_t type,
                                        uint64_t tl_bitmap,
                                        const ucp_rsc_index_t *rsc_tli)
{
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    ucp_request_t* req;
    ucs_status_t status;
    void *address;
    unsigned *order = ucs_alloca(ep->worker->context->num_tls * sizeof(*order));

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
    status = ucp_address_pack(ep->worker, ucp_wireup_is_ep_needed(ep) ? ep : NULL,
                              tl_bitmap, order, &req->send.length, &address);
    if (status != UCS_OK) {
        ucs_free(req);
        ucs_error("failed to pack address: %s", ucs_status_string(status));
        return status;
    }

    req->send.buffer = address;

    /* send the indices addresses that should be connected by remote side */
    for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
        rsc_index = rsc_tli[lane];
        if (rsc_index != UCP_NULL_RESOURCE) {
            req->send.wireup.tli[lane] = ucp_wireup_address_index(order,
                                                                  tl_bitmap,
                                                                  rsc_index);
        } else {
            req->send.wireup.tli[lane] = -1;
        }
    }

    ucp_request_send(req);
    return UCS_OK;
}

static ucs_status_t ucp_wireup_connect_local(ucp_ep_h ep, const uint8_t *tli,
                                             unsigned address_count,
                                             const ucp_address_entry_t *address_list)
{
    const ucp_address_entry_t *address;
    ucp_lane_index_t lane;
    ucs_status_t status;
    ucp_md_map_t UCS_V_UNUSED md_map;

    ucs_trace("ep %p: connect local transports", ep);

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (!ucp_ep_is_lane_p2p(ep, lane)) {
            continue;
        }

        address = &address_list[tli[lane]];
        status = uct_ep_connect_to_ep(ep->uct_eps[lane], address->dev_addr,
                                      address->ep_addr);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static void ucp_wireup_remote_connected(ucp_ep_h ep)
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

static UCS_F_NOINLINE void
ucp_wireup_process_pre_request(ucp_worker_h worker, const ucp_wireup_msg_t *msg,
                               const ucp_unpacked_address_t *remote_address)
{
    uint8_t addr_indices[UCP_MAX_LANES];
    ucp_ep_params_t params;
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

    params.field_mask = UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    params.err_mode   = ucp_ep_config(ep)->key.err_mode;

    /* initialize transport endpoints */
    status = ucp_wireup_init_lanes(ep, &params, UCP_EP_CREATE_AM_LANE,
                                   remote_address->address_count,
                                   remote_address->address_list, addr_indices);
    if (status != UCS_OK) {
        ucp_worker_set_ep_failed(worker, ep,
                                 ep->uct_eps[ucp_ep_get_wireup_msg_lane(ep)],
                                 ucp_ep_get_wireup_msg_lane(ep), status);
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
    uint64_t remote_uuid = remote_address->uuid;
    ucp_rsc_index_t rsc_tli[UCP_MAX_LANES];
    uint8_t addr_indices[UCP_MAX_LANES];
    ucp_lane_index_t lane, remote_lane;
    ucp_rsc_index_t rsc_index;
    ucp_ep_params_t params;
    ucs_status_t status;
    uint64_t tl_bitmap = 0;
    int send_reply = 0, reset_listener_flag = 0;
    ucp_ep_h ep;
    unsigned ep_init_flags = 0;

    ucs_assert(msg->type == UCP_WIREUP_MSG_REQUEST);
    ucs_trace("got wireup request from 0x%"PRIx64" src_ep 0x%lx dst_ep 0x%lx conn_sn %d",
              remote_address->uuid, msg->src_ep_ptr, msg->dest_ep_ptr, msg->conn_sn);

    if (msg->dest_ep_ptr != 0) {
        /* wireup request for a specific ep */
        ep = ucp_worker_get_ep_by_ptr(worker, msg->dest_ep_ptr);
        ucp_ep_update_dest_ep_ptr(ep, msg->src_ep_ptr);
        ucp_ep_flush_state_reset(ep);

        ep_init_flags |= UCP_EP_CREATE_AM_LANE;
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

    params.field_mask = UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    params.err_mode   = msg->err_mode;

    if (ep->flags & UCP_EP_FLAG_LISTENER) {
        /* If this is an ep on a listener (server) that received a partial
         * worker address from the client, then the following lanes initialization
         * will be done after an aux lane was already created on this ep.
         * Therefore, remove the existing aux endpoint since will need to create
         * new lanes now */
        ucp_ep_cleanup_lanes(ep);
        for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
            ep->uct_eps[lane] = NULL;
        }
    }

    /* Initialize lanes (possible destroy existing lanes) */
    status = ucp_wireup_init_lanes(ep, &params, ep_init_flags,
                                   remote_address->address_count,
                                   remote_address->address_list, addr_indices);
    if (status != UCS_OK) {
        return;
    }

    /* Send a reply if remote side does not have ep_ptr (active-active flow) or
     * there are p2p lanes (client-server flow)
     */
    send_reply = (msg->dest_ep_ptr == 0) || ucp_ep_config(ep)->p2p_lanes;

    /* Connect p2p addresses to remote endpoint */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        status = ucp_wireup_connect_local(ep, addr_indices,
                                          remote_address->address_count,
                                          remote_address->address_list);
        if (status != UCS_OK) {
            return;
        }

        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;

        ucs_assert(send_reply);
    }

    /* mark the endpoint as connected to remote */
    if (!ucp_ep_config(ep)->p2p_lanes) {
        ucp_wireup_remote_connected(ep);
    }

    if (send_reply) {

        if (ep->flags & UCP_EP_FLAG_LISTENER) {
            /* Remove this flag at this point (so that address packing would be correct) */
            ep->flags &= ~UCP_EP_FLAG_LISTENER;
            reset_listener_flag = 1;
        }

        /* Construct the list that tells the remote side with which address we
         * have connected to each of its lanes.
         */
        memset(rsc_tli, -1, sizeof(rsc_tli));
        for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
             rsc_index = ucp_ep_get_rsc_index(ep, lane);
             for (remote_lane = 0; remote_lane < UCP_MAX_LANES; ++remote_lane) {
                 /* If 'lane' has connected to 'remote_lane' ... */
                 if (addr_indices[lane] == msg->tli[remote_lane]) {
                     ucs_assert(ucp_worker_is_tl_p2p(worker, rsc_index));
                     rsc_tli[remote_lane] = rsc_index;
                     tl_bitmap           |= UCS_BIT(rsc_index);
                 }
             }
        }

        ucs_trace("ep %p: sending wireup reply", ep);
        status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REPLY, tl_bitmap, rsc_tli);
        if (status != UCS_OK) {
            return;
        }

        if (reset_listener_flag) {
            ep->flags |= UCP_EP_FLAG_LISTENER;
        }
    } else {
        /* if in client-server flow, schedule invoking the user's callback
         * (if server is connected) from the main thread */
        if (ucs_test_all_flags(ep->flags,
                               (UCP_EP_FLAG_LISTENER | UCP_EP_FLAG_LOCAL_CONNECTED))) {
            ucp_listener_schedule_accept_cb(ep);
        }
    }
}

static UCS_F_NOINLINE void
ucp_wireup_process_reply(ucp_worker_h worker, const ucp_wireup_msg_t *msg,
                         const ucp_unpacked_address_t *remote_address)
{
    ucp_rsc_index_t rsc_tli[UCP_MAX_LANES];
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
        status = ucp_wireup_connect_local(ep, msg->tli,
                                          remote_address->address_count,
                                          remote_address->address_list);
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
        /* Send ACK without any address, we've already sent it as part of the request */
        ucs_trace("ep %p: sending wireup ack", ep);
        memset(rsc_tli, -1, sizeof(rsc_tli));
        status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_ACK, 0, rsc_tli);
        if (status != UCS_OK) {
            return;
        }
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

    status = ucp_address_unpack(msg + 1, &remote_address);
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

static void ucp_wireup_assign_lane(ucp_ep_h ep, ucp_lane_index_t lane,
                                   uct_ep_h uct_ep, const char *info)
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
        ucp_wireup_ep_remote_connected(ep->uct_eps[lane]);
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

static ucs_status_t ucp_wireup_connect_lane(ucp_ep_h ep,
                                            const ucp_ep_params_t *params,
                                            ucp_lane_index_t lane,
                                            unsigned address_count,
                                            const ucp_address_entry_t *address_list,
                                            unsigned addr_index)
{
    ucp_worker_h worker          = ep->worker;
    ucp_rsc_index_t rsc_index    = ucp_ep_get_rsc_index(ep, lane);
    ucp_lane_index_t proxy_lane  = ucp_ep_get_proxy_lane(ep, lane);
    uct_iface_attr_t *iface_attr = &worker->ifaces[rsc_index].attr;
    uct_ep_h uct_ep;
    ucs_status_t status;

    ucs_trace("ep %p: connect lane[%d]", ep, lane);

    /*
     * if the selected transport can be connected directly to the remote
     * interface, just create a connected UCT endpoint.
     */
    if ((iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) &&
        ((ep->uct_eps[lane] == NULL) || ucp_wireup_ep_test(ep->uct_eps[lane])))
    {
        if ((proxy_lane == UCP_NULL_LANE) || (proxy_lane == lane)) {
            /* create an endpoint connected to the remote interface */
            ucs_trace("ep %p: connect uct_ep[%d] to addr[%d]", ep, lane,
                      addr_index);
            status = uct_ep_create_connected(worker->ifaces[rsc_index].iface,
                                             address_list[addr_index].dev_addr,
                                             address_list[addr_index].iface_addr,
                                             &uct_ep);
            if (status != UCS_OK) {
                /* coverity[leaked_storage] */
                return status;
            }

            ucp_wireup_assign_lane(ep, lane, uct_ep, "");
        }

        ucp_worker_iface_progress_ep(&worker->ifaces[rsc_index]);
        return UCS_OK;
    }

    /*
     * create a wireup endpoint which will start connection establishment
     * protocol using an auxiliary transport.
     */
    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {

        /* For now, p2p transports have no reason to have proxy */
        ucs_assert_always(proxy_lane == UCP_NULL_LANE);

        /* If ep already exists, it's a wireup proxy, and we need to start
         * auxiliary wireup.
         */
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
        }

        ucs_trace("ep %p: connect uct_ep[%d]=%p to addr[%d] wireup", ep, lane,
                  uct_ep, addr_index);
        status = ucp_wireup_ep_connect(ep->uct_eps[lane], params, rsc_index,
                                       lane == ucp_ep_get_wireup_msg_lane(ep),
                                       address_count, address_list);
        if (status != UCS_OK) {
            return status;
        }

        ucp_worker_iface_progress_ep(&worker->ifaces[rsc_index]);

        return UCS_OK;
    }

    return UCS_ERR_UNREACHABLE;
}

static ucs_status_t ucp_wireup_resolve_proxy_lanes(ucp_ep_h ep)
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

        iface_attr = &ep->worker->ifaces[ucp_ep_get_rsc_index(ep, lane)].attr;
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

        ucp_wireup_assign_lane(ep, lane, signaling_ep, " (signaling proxy)");
    }

    return UCS_OK;
}

static void ucp_wireup_print_config(ucp_context_h context,
                                    const ucp_ep_config_key_t *key,
                                    const char *title,
                                    uint8_t *addr_indices,
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

ucs_status_t ucp_wireup_init_lanes(ucp_ep_h ep, const ucp_ep_params_t *params,
                                   unsigned ep_init_flags, unsigned address_count,
                                   const ucp_address_entry_t *address_list,
                                   uint8_t *addr_indices)
{
    ucp_worker_h worker = ep->worker;
    ucp_ep_config_key_t key;
    uint16_t new_cfg_index;
    ucp_lane_index_t lane;
    ucs_status_t status;
    char str[32];

    ucs_trace("ep %p: initialize lanes", ep);

    status = ucp_wireup_select_lanes(ep, params, ep_init_flags, address_count,
                                     address_list, addr_indices, &key);
    if (status != UCS_OK) {
        goto err;
    }

    key.reachable_md_map |= ucp_ep_config(ep)->key.reachable_md_map;

    new_cfg_index = ucp_worker_get_ep_config(worker, &key);
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

    ep->cfg_index = new_cfg_index;
    ep->am_lane   = key.am_lane;

    snprintf(str, sizeof(str), "ep %p", ep);
    ucp_wireup_print_config(worker->context, &ucp_ep_config(ep)->key, str,
                            addr_indices, UCS_LOG_LEVEL_DEBUG);

    /* establish connections on all underlying endpoints */
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        status = ucp_wireup_connect_lane(ep, params, lane, address_count,
                                         address_list, addr_indices[lane]);
        if (status != UCS_OK) {
            goto err;
        }
    }

    status = ucp_wireup_resolve_proxy_lanes(ep);
    if (status != UCS_OK) {
        goto err;
    }

    /* If we don't have a p2p transport, we're connected */
    if (!ucp_ep_config(ep)->p2p_lanes) {
        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
    }

    return UCS_OK;

err:
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (ep->uct_eps[lane] != NULL) {
            uct_ep_destroy(ep->uct_eps[lane]);
            ep->uct_eps[lane] = NULL;
        }
    }
    return status;
}

ucs_status_t ucp_wireup_send_request(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;
    ucp_rsc_index_t rsc_tli[UCP_MAX_LANES];
    ucp_rsc_index_t rsc_index;
    uint64_t tl_bitmap = 0;
    ucp_lane_index_t lane;
    ucs_status_t status;

    for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
        if (lane < ucp_ep_num_lanes(ep)) {
            rsc_index = ucp_ep_get_rsc_index(ep, lane);
            rsc_tli[lane] = ucp_worker_is_tl_p2p(worker, rsc_index) ? rsc_index :
                                                                      UCP_NULL_RESOURCE;
            tl_bitmap |= UCS_BIT(rsc_index);
        } else {
            rsc_tli[lane] = UCP_NULL_RESOURCE;
        }
    }

    /* TODO make sure such lane would exist */
    rsc_index = ucp_wireup_ep_get_aux_rsc_index(
                    ep->uct_eps[ucp_ep_get_wireup_msg_lane(ep)]);
    if (rsc_index != UCP_NULL_RESOURCE) {
        tl_bitmap |= UCS_BIT(rsc_index);
    }

    ucs_debug("ep %p: send wireup request (flags=0x%x)", ep, ep->flags);
    status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REQUEST, tl_bitmap, rsc_tli);

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
    uint64_t tl_bitmap = -1;  /* pack full worker address */
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
        status = uct_ep_pending_add(ep->uct_eps[lane], &req->send.uct);
        ucs_assert(status == UCS_OK); /* because it's a wireup proxy */
    }

    status = UCS_OK;
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
    ucp_lane_index_t lane;
    unsigned addr_index;
    ucs_status_t status;
    char *p, *end;

    status = ucp_address_unpack(msg + 1, &unpacked_address);
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
             (msg->type == UCP_WIREUP_MSG_PRE_REQUEST ) ? "PRE_REQ" :
             (msg->type == UCP_WIREUP_MSG_REQUEST     ) ? "REQ" :
             (msg->type == UCP_WIREUP_MSG_REPLY       ) ? "REP" :
             (msg->type == UCP_WIREUP_MSG_ACK         ) ? "ACK" : "",
             unpacked_address.name, unpacked_address.uuid, msg->src_ep_ptr,
             msg->dest_ep_ptr, msg->conn_sn);
    p += strlen(p);

    for (addr_index = 0; addr_index < unpacked_address.address_count; ++addr_index) {
        ae = &unpacked_address.address_list[addr_index];
        for (rsc = context->tl_rscs; rsc < context->tl_rscs + context->num_tls; ++rsc) {
            if (ae->tl_name_csum == rsc->tl_name_csum) {
                snprintf(p, end - p, " "UCT_TL_RESOURCE_DESC_FMT,
                         UCT_TL_RESOURCE_DESC_ARG(&rsc->tl_rsc));
                p += strlen(p);
                break;
            }
        }
        snprintf(p, end - p, "/md[%d]", ae->md_index);
        p += strlen(p);

        for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
            if (msg->tli[lane] == addr_index) {
                snprintf(p, end - p, "/lane[%d]", lane);
                p += strlen(p);
            }
        }
    }

    ucs_free(unpacked_address.address_list);
}

UCP_DEFINE_AM(-1, UCP_AM_ID_WIREUP, ucp_wireup_msg_handler, 
              ucp_wireup_msg_dump, UCT_CB_FLAG_ASYNC);
