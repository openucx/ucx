/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "wireup.h"
#include "address.h"
#include "stub_ep.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/dt/dt_contig.h>
#include <ucp/tag/eager.h>
#include <ucs/arch/bitops.h>
#include <ucs/async/async.h>
#include <ucs/datastruct/queue.h>

/*
 * Description of the protocol in UCX wiki:
 * https://github.com/openucx/ucx/wiki/Connection-establishment
 */

static void ucp_wireup_msg_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                                uint8_t id, const void *data, size_t length,
                                char *buffer, size_t max)
{
    const ucp_wireup_msg_t *msg = data;
    char peer_name[UCP_WORKER_NAME_MAX + 1];
    ucp_address_entry_t *address_list, *ae;
    unsigned address_count;
    ucp_lane_index_t lane;
    uint64_t uuid;
    char *p, *end;

    ucp_address_unpack(msg + 1, &uuid, peer_name, sizeof(peer_name),
                       &address_count, &address_list);

    p   = buffer;
    end = buffer + max;
    snprintf(p, end - p, "WIREUP %s [%s uuid 0x%"PRIx64"]",
             (msg->type == UCP_WIREUP_MSG_REQUEST ) ? "REQ" :
             (msg->type == UCP_WIREUP_MSG_REPLY   ) ? "REP" :
             (msg->type == UCP_WIREUP_MSG_ACK     ) ? "ACK" : "",
             peer_name, uuid);

    p += strlen(p);
    for (ae = address_list; ae < address_list + address_count; ++ae) {
        snprintf(p, end - p, " [");
        p += strlen(p);
        for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
            if (msg->tli[lane] == (ae - address_list)) {
                snprintf(p, end - p, "%d: ", lane);
                p += strlen(p);
            }
        }
        snprintf(p, end - p, "0x%4x(%zu)]", ae->tl_name_csum, ae->tl_addr_len);
        p += strlen(p);
    }

    ucs_free(address_list);
}

static size_t ucp_wireup_msg_pack(void *dest, void *arg)
{
    ucp_request_t *req = arg;
    *(ucp_wireup_msg_t*)dest = req->send.wireup;
    memcpy((ucp_wireup_msg_t*)dest + 1, req->send.buffer, req->send.length);
    return sizeof(ucp_wireup_msg_t) + req->send.length;
}

static uct_ep_h ucp_wireup_msg_uct_ep(ucp_ep_h ep, uint8_t type)
{
    ucp_lane_index_t lane = ucp_ep_get_wireup_msg_lane(ep);
    if ((lane == UCP_NULL_LANE) || (type == UCP_WIREUP_MSG_ACK)) {
        return ucp_ep_get_am_uct_ep(ep);
    } else {
        return ep->uct_eps[lane];
    }
}

ucs_status_t ucp_wireup_msg_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep = req->send.ep;
    ssize_t packed_len;

    if (req->send.wireup.type == UCP_WIREUP_MSG_REQUEST) {
        if (ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED) {
            ucs_trace("ep %p: not sending wireup message - remote already connected",
                      ep);
            goto out;
        }
    }

    /* send the active message */
    packed_len = uct_ep_am_bcopy(ucp_wireup_msg_uct_ep(ep, req->send.wireup.type),
                                 UCP_AM_ID_WIREUP, ucp_wireup_msg_pack, req);
    if (packed_len < 0) {
        if (packed_len != UCS_ERR_NO_RESOURCE) {
            ucs_error("failed to send wireup: %s", ucs_status_string(packed_len));
        }
        return (ucs_status_t)packed_len;
    }

out:
    ucp_request_complete(req, req->cb.send, UCS_OK);
    return UCS_OK;
}

static unsigned ucp_wireup_address_index(const unsigned *order,
                                         uint64_t tl_bitmap,
                                         ucp_rsc_index_t tl_index)
{
    return order[ucs_count_one_bits(tl_bitmap & UCS_MASK(tl_index))];
}

void ucp_wireup_msg_send_completion(void *request, ucs_status_t status)
{
    ucp_request_t *req = (ucp_request_t *)request - 1;
    ucs_free((void*)req->send.buffer);
}

static ucs_status_t ucp_wireup_msg_send(ucp_ep_h ep, uint8_t type)
{
    ucp_worker_h worker = ep->worker;
    uct_ep_h uct_ep     = ucp_wireup_msg_uct_ep(ep, type);
    ucp_rsc_index_t rsc_index, aux_rsc_index;
    ucs_status_t status;
    ucp_lane_index_t lane;
    uint64_t tl_bitmap;
    unsigned order[UCP_MAX_LANES + 1];
    ucp_request_t* req;
    void *address;

    ucs_assert(ep->cfg_index != (uint8_t)-1);

    req = ucs_mpool_get(&ep->worker->req_mp);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    req->flags                   = UCP_REQUEST_FLAG_RELEASED;
    req->cb.send                 = ucp_wireup_msg_send_completion;
    req->send.uct.func           = ucp_wireup_msg_progress;
    req->send.wireup.type        = type;

    /* Make a bitmap of all addresses we are sending:
     *  REQUEST - all addresses (incl. auxiliary)
     *  REPLY   - only p2p addresses
     *  ACK     - no addresses
     */
    tl_bitmap = 0;
    if (req->send.wireup.type == UCP_WIREUP_MSG_REQUEST) {
        aux_rsc_index = ucp_stub_ep_get_aux_rsc_index(uct_ep);
        if (aux_rsc_index != UCP_NULL_RESOURCE) {
            tl_bitmap |= UCS_BIT(aux_rsc_index);
        }
    }
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        rsc_index = ucp_ep_get_rsc_index(ep, lane);
        if ((rsc_index != UCP_NULL_RESOURCE) &&
            ((req->send.wireup.type == UCP_WIREUP_MSG_REQUEST) ||
             ((req->send.wireup.type == UCP_WIREUP_MSG_REPLY) &&
              ucp_worker_is_tl_p2p(worker, rsc_index))))
        {
            tl_bitmap |= UCS_BIT(rsc_index);
        }
    }

    /* pack all addresses */
    status = ucp_address_pack(ep->worker, ep, tl_bitmap, order,
                              &req->send.length, &address);
    if (status != UCS_OK) {
        ucs_error("failed to pack address: %s", ucs_status_string(status));
        return status;
    }

    req->send.buffer = address;

    /* send the indices of runtime addresses for each operation */
    for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
        if ((lane < ucp_ep_num_lanes(ep)) &&
            (req->send.wireup.type != UCP_WIREUP_MSG_ACK))
        {
            rsc_index = ucp_ep_get_rsc_index(ep, lane);
            req->send.wireup.tli[lane] = ucp_wireup_address_index(order,
                                                                  tl_bitmap,
                                                                  rsc_index);
        } else {
            req->send.wireup.tli[lane] = -1;
        }
    }

    ucp_ep_add_pending(ep, uct_ep, req, 0);
    return UCS_OK;
}

static ucs_status_t ucp_wireup_connect_local(ucp_ep_h ep, const uint8_t *tli,
                                             unsigned address_count,
                                             const ucp_address_entry_t *address_list)
{
    ucp_worker_h worker = ep->worker;
    const ucp_address_entry_t *address;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    ucs_status_t status;
    ucp_pd_map_t UCS_V_UNUSED pd_map;

    ucs_trace("ep %p: connect local transports", ep);

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        rsc_index = ucp_ep_get_rsc_index(ep, lane);
        if (!ucp_worker_is_tl_p2p(worker, rsc_index)) {
            continue;
        }

        address = &address_list[tli[lane]];
        ucs_assert(address->tl_addr_len > 0);

        /* Check that if the lane is used for RMA/AMO, destination pd index matches */
        pd_map = ucp_lane_map_get_lane(ucp_ep_config(ep)->key.rma_lane_map, lane);
        ucs_assert((pd_map == 0) || (pd_map == UCS_BIT(address->pd_index)));

        pd_map = ucp_lane_map_get_lane(ucp_ep_config(ep)->key.amo_lane_map, lane);
        ucs_assert((pd_map == 0) || (pd_map == UCS_BIT(address->pd_index)));

        status = uct_ep_connect_to_ep(ep->uct_eps[lane], address->dev_addr,
                                      address->ep_addr);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

static void ucp_wireup_ep_remote_connected(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;

    ucs_trace("ep %p: remote connected", ep);

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        rsc_index = ucp_ep_get_rsc_index(ep, lane);
        if (ucp_worker_is_tl_p2p(worker, rsc_index)) {
            ucp_stub_ep_remote_connected(ep->uct_eps[lane]);
        }
    }
}

static void ucp_wireup_process_request(ucp_worker_h worker, const ucp_wireup_msg_t *msg,
                                       uint64_t uuid, const char *peer_name,
                                       unsigned address_count,
                                       const ucp_address_entry_t *address_list)
{
    ucp_ep_h ep = ucp_worker_ep_find(worker, uuid);
    ucs_status_t status;

    ucs_trace("ep %p: got wireup request from %s", ep, peer_name);

    if (ep == NULL) {
        /* Create a new endpoint and connect it to remote address */
        status = ucp_ep_create_connected(worker, uuid, peer_name, address_count,
                                         address_list, "remote-request", &ep);
        if (status != UCS_OK) {
            return;
        }
    } else if (ucp_ep_is_stub(ep)) {
        status = ucp_wireup_init_lanes(ep, address_count, address_list);
        if (status != UCS_OK) {
            return;
        }
    }

    /* Connect p2p addresses to remote endpoint */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        status = ucp_wireup_connect_local(ep, msg->tli, address_count, address_list);
        if (status != UCS_OK) {
            return;
        }

        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;

        ucs_trace("ep %p: sending wireup reply", ep);

        status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REPLY);
        if (status != UCS_OK) {
            return;
        }
    }
}

static void ucp_wireup_process_reply(ucp_worker_h worker, ucp_wireup_msg_t *msg,
                                     uint64_t uuid, unsigned address_count,
                                     const ucp_address_entry_t *address_list)
{
    ucp_ep_h ep = ucp_worker_ep_find(worker, uuid);
    ucs_status_t status;

    if (ep == NULL) {
        ucs_debug("ignoring connection reply - not exists");
        return;
    }

    ucs_trace("ep %p: got wireup reply", ep);


    /* Connect p2p addresses to remote endpoint */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        status = ucp_wireup_connect_local(ep, msg->tli, address_count, address_list);
        if (status != UCS_OK) {
            return;
        }

        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;

        /* If remote is connected - just send an ACK (because we already sent the address)
         * Otherwise - send a REPLY message with the ep addresses.
         */
        status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_ACK);
        if (status != UCS_OK) {
            return;
        }
    }

    if (!(ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED)) {
        ucp_wireup_ep_remote_connected(ep);
        ep->flags |= UCP_EP_FLAG_REMOTE_CONNECTED;
    }
}

static void ucp_wireup_process_ack(ucp_worker_h worker, uint64_t uuid)
{
    ucp_ep_h ep = ucp_worker_ep_find(worker, uuid);

    if (ep == NULL) {
        ucs_debug("ignoring connection ack - ep not exists");
        return;
    }

    ucs_trace("ep %p: got wireup ack", ep);

    ep->flags |= UCP_EP_FLAG_REMOTE_CONNECTED;
    ucp_wireup_ep_remote_connected(ep);
}

static ucs_status_t ucp_wireup_msg_handler(void *arg, void *data,
                                           size_t length, void *desc)
{
    ucp_worker_h worker   = arg;
    ucp_wireup_msg_t *msg = data;
    char peer_name[UCP_WORKER_NAME_MAX];
    ucp_address_entry_t *address_list;
    unsigned address_count;
    ucs_status_t status;
    uint64_t uuid;

    UCS_ASYNC_BLOCK(&worker->async);

    status = ucp_address_unpack(msg + 1, &uuid, peer_name, UCP_WORKER_NAME_MAX,
                                &address_count, &address_list);
    if (status != UCS_OK) {
        ucs_error("failed to unpack address: %s", ucs_status_string(status));
        goto out;
    }

    if (msg->type == UCP_WIREUP_MSG_ACK) {
        ucs_assert(address_count == 0);
        ucp_wireup_process_ack(worker, uuid);
    } else if (msg->type == UCP_WIREUP_MSG_REQUEST) {
        ucp_wireup_process_request(worker, msg, uuid, peer_name, address_count,
                                   address_list);
    } else if (msg->type == UCP_WIREUP_MSG_REPLY) {
        ucp_wireup_process_reply(worker, msg, uuid, address_count, address_list);
    } else {
        ucs_bug("invalid wireup message");
    }

    ucs_free(address_list);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;
}

static ucs_status_t ucp_wireup_connect_lane(ucp_ep_h ep, ucp_lane_index_t lane,
                                            unsigned address_count,
                                            const ucp_address_entry_t *address_list,
                                            unsigned addr_index)
{
    ucp_worker_h worker          = ep->worker;
    ucp_rsc_index_t rsc_index    = ucp_ep_get_rsc_index(ep, lane);
    uct_iface_attr_t *iface_attr = &worker->iface_attrs[rsc_index];
    uct_ep_h new_uct_ep;
    ucs_status_t status;

    /*
     * if the selected transport can be connected directly to the remote
     * interface, just create a connected UCT endpoint.
     */
    if ((iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) &&
        ((ep->uct_eps[lane] == NULL) || ucp_stub_ep_test(ep->uct_eps[lane])))
    {
        /* create an endpoint connected to the remote interface */
        ucs_assert(address_list[addr_index].tl_addr_len > 0);
        status = uct_ep_create_connected(worker->ifaces[rsc_index],
                                         address_list[addr_index].dev_addr,
                                         address_list[addr_index].iface_addr,
                                         &new_uct_ep);
        if (status != UCS_OK) {
            return status;
        }

        /* If ep already exists, it's a stub, and we need to update its next_ep
         * instead of replacing it.
         */
        if (ep->uct_eps[lane] == NULL) {
            ucs_trace("ep %p: assign uct_ep[%d]=%p", ep, lane, new_uct_ep);
            ep->uct_eps[lane] = new_uct_ep;
        } else {
            ucs_trace("ep %p: assign set stub_ep[%d]=%p next to %p",
                      ep, lane, ep->uct_eps[lane], new_uct_ep);
            ucp_stub_ep_set_next_ep(ep->uct_eps[lane], new_uct_ep);
            ucp_stub_ep_remote_connected(ep->uct_eps[lane]);
        }

        return UCS_OK;
    }

    /*
     * create a stub endpoint which will start connection establishment
     * protocol using an auxiliary transport.
     */
    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {

        /* If ep already exists, it's a stub, and we need to start auxiliary
         * wireup on that stub.
         */
        if (ep->uct_eps[lane] == NULL) {
            ucs_trace("ep %p: create stub_ep[%d]=%p", ep, lane, ep->uct_eps[lane]);
            status = ucp_stub_ep_create(ep, &ep->uct_eps[lane]);
            if (status != UCS_OK) {
                return status;
            }
        }

        ucs_trace("ep %p: connect stub_ep[%d]=%p", ep, lane, ep->uct_eps[lane]);
        return ucp_stub_ep_connect(ep->uct_eps[lane],
                                   ucp_ep_get_rsc_index(ep, lane),
                                   lane == ucp_ep_get_wireup_msg_lane(ep),
                                   address_count, address_list);
    }

    return UCS_ERR_UNREACHABLE;
}

ucs_status_t ucp_wireup_init_lanes(ucp_ep_h ep, unsigned address_count,
                                   const ucp_address_entry_t *address_list)
{
    ucp_worker_h worker = ep->worker;
    unsigned addr_indices[UCP_MAX_LANES];
    ucp_lane_index_t lane;
    ucs_status_t status;
    uint8_t conn_flag;

    ucs_trace("ep %p: initialize transports", ep);

    status = ucp_wireup_select_transports(ep, address_count, address_list,
                                          addr_indices);
    if (status != UCS_OK) {
        goto err;
    }

    /* establish connections on all underlying endpoint */
    conn_flag = UCP_EP_FLAG_LOCAL_CONNECTED;
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        status = ucp_wireup_connect_lane(ep, lane, address_count, address_list,
                                         addr_indices[lane]);
        if (status != UCS_OK) {
            goto err;
        }

        if (ucp_worker_is_tl_p2p(worker, ucp_ep_get_rsc_index(ep, lane))) {
            conn_flag = 0; /* If we have a p2p transport, we're not connected */
        }
    }

    ep->flags |= conn_flag;

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
    ucs_status_t status;

    if (ep->flags & UCP_EP_FLAG_CONNECT_REQ_SENT) {
        return UCS_OK;
    }

    ucs_debug("ep %p: send wireup request (flags=0x%x)", ep, ep->flags);
    status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REQUEST);
    ep->flags |= UCP_EP_FLAG_CONNECT_REQ_SENT;
    return status;
}

UCP_DEFINE_AM(-1, UCP_AM_ID_WIREUP, ucp_wireup_msg_handler, 
              ucp_wireup_msg_dump, UCT_AM_CB_FLAG_ASYNC);

