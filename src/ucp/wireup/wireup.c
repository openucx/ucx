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

    if (req->send.wireup.type == UCP_WIREUP_MSG_REQUEST) {
        if (ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED) {
            ucs_trace("ep %p: not sending wireup message - remote already connected",
                      ep);
            goto out;
        }
    }

    /* send the active message */
    if (req->send.wireup.type == UCP_WIREUP_MSG_ACK) {
        req->send.lane = ucp_ep_get_am_lane(ep);
    } else {
        req->send.lane = ucp_ep_get_wireup_msg_lane(ep);
    }
    packed_len = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], UCP_AM_ID_WIREUP,
                                 ucp_wireup_msg_pack, req, 0);
    if (packed_len < 0) {
        if (packed_len != UCS_ERR_NO_RESOURCE) {
            ucs_error("failed to send wireup: %s", ucs_status_string(packed_len));
        }
        return (ucs_status_t)packed_len;
    }

    switch (req->send.wireup.type) {
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

/*
 * @param [in] rsc_tli  Resource index for every lane.
 */
static ucs_status_t ucp_wireup_msg_send(ucp_ep_h ep, uint8_t type,
                                        uint64_t ep_uuid, uint64_t tl_bitmap,
                                        const ucp_rsc_index_t *rsc_tli)
{
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    unsigned order[UCP_MAX_LANES + 1];
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
    req->send.wireup.ep_uuid     = ep_uuid;
    req->send.uct.func           = ucp_wireup_msg_progress;
    req->send.datatype           = ucp_dt_make_contig(1);
    ucp_request_send_state_init(req, ucp_dt_make_contig(1), 0);

    /* pack all addresses */
    status = ucp_address_pack(ep->worker, ep, tl_bitmap, order,
                              &req->send.length, &address);
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

    ucs_trace("ep %p: remote connected", ep);
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (ucp_ep_is_lane_p2p(ep, lane)) {
            ucp_wireup_ep_remote_connected(ep->uct_eps[lane]);
        }
    }
}

static void ucp_wireup_process_request(ucp_worker_h worker, const ucp_wireup_msg_t *msg,
                                       uint64_t uuid, const char *peer_name,
                                       unsigned address_count,
                                       const ucp_address_entry_t *address_list)
{
    ucp_rsc_index_t rsc_tli[UCP_MAX_LANES];
    uint8_t addr_indices[UCP_MAX_LANES];
    ucp_lane_index_t lane, remote_lane;
    ucp_rsc_index_t rsc_index;
    ucp_ep_params_t params;
    ucs_status_t status;
    uint64_t tl_bitmap = 0;
    ucp_ep_h ep;

    ucs_trace("got wireup request from 0x%"PRIx64, uuid);

    if (msg->ep_uuid == worker->uuid) {
        /* Request for a new connection to the worker */
        ep = ucp_worker_ep_find(worker, uuid);
        if (ep == NULL) {
            /* Create a new endpoint if does not exist */
            status = ucp_ep_new(worker, uuid, peer_name, "remote-request", &ep);
            if (status != UCS_OK) {
                return;
            }
            ep->flags |= UCP_EP_FLAG_DEST_UUID_PEER;
        } else {
            ucs_assert(ep->flags & UCP_EP_FLAG_DEST_UUID_PEER);
        }
    } else {
        /* Reply for a client-server connection (client side) */
        ep = ucp_worker_ep_find(worker, msg->ep_uuid);
        if (ep == NULL) {
            ucs_trace("got connection request with invalid ep_uuid 0x%"PRIx64,
                      msg->ep_uuid);
            return;
        }

        /* Reinsert to hash table with destination worker uuid */
        ucs_assert(!(ep->flags & UCP_EP_FLAG_DEST_UUID_PEER));
        ucp_ep_delete_from_hash(ep);
        ep->dest_uuid = uuid;
        ep->flags    |= UCP_EP_FLAG_DEST_UUID_PEER;
        ucp_ep_add_to_hash(ep);
    }

    params.field_mask = UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    params.err_mode   = msg->err_mode;

    /* Initialize lanes (possible destroy existing lanes) */
    status = ucp_wireup_init_lanes(ep, &params, 0, address_count, address_list,
                                   addr_indices);
    if (status != UCS_OK) {
        return;
    }

    /* Connect p2p addresses to remote endpoint */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        status = ucp_wireup_connect_local(ep, addr_indices, address_count,
                                          address_list);
        if (status != UCS_OK) {
            return;
        }

        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;

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
        status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REPLY, ep->dest_uuid,
                                     tl_bitmap, rsc_tli);
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
    ucp_rsc_index_t rsc_tli[UCP_MAX_LANES];
    ucs_status_t status;
    int ack;

    if (ep == NULL) {
        ucs_debug("ignoring connection reply - not exists");
        return;
    }

    ucs_trace("ep %p: got wireup reply", ep);

    ucs_assert(ep->flags & UCP_EP_FLAG_DEST_UUID_PEER);

    /* Connect p2p addresses to remote endpoint */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        status = ucp_wireup_connect_local(ep, msg->tli, address_count, address_list);
        if (status != UCS_OK) {
            return;
        }

        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
        ack = 1;
    } else {
        ack = 0;
    }

    if (!(ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED)) {
        ucp_wireup_remote_connected(ep);
        ep->flags |= UCP_EP_FLAG_REMOTE_CONNECTED;
    }

    if (ack) {
        /* Send ACK without any address, we've already sent it as part of the request */
        ucs_trace("ep %p: sending wireup ack", ep);
        memset(rsc_tli, -1, sizeof(rsc_tli));
        status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_ACK, ep->dest_uuid, 0,
                                     rsc_tli);
        if (status != UCS_OK) {
            return;
        }
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

    ucs_assert(ep->flags & UCP_EP_FLAG_DEST_UUID_PEER);
    ucs_assert(ep->flags & UCP_EP_FLAG_CONNECT_REP_SENT);
    ucs_assert(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED);

    ep->flags |= UCP_EP_FLAG_REMOTE_CONNECTED;
    ucp_wireup_remote_connected(ep);
}

static ucs_status_t ucp_wireup_msg_handler(void *arg, void *data,
                                           size_t length, unsigned flags)
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

    if ((ep->cfg_index != 0) && !ucp_ep_is_stub(ep)) {
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

    /* Cache tag offload state in the flags for fast-path */
    if (ucp_ep_is_tag_offload_enabled(ucp_ep_config(ep))) {
        ep->flags |= UCP_EP_FLAG_TAG_OFFLOAD_ENABLED;
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

ucs_status_t ucp_wireup_send_request(ucp_ep_h ep, uint64_t ep_uuid)
{
    ucp_worker_h worker = ep->worker;
    ucp_rsc_index_t rsc_tli[UCP_MAX_LANES];
    ucp_rsc_index_t rsc_index;
    uint64_t tl_bitmap = 0;
    ucp_lane_index_t lane;
    ucs_status_t status;

    if (ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED) {
        return UCS_OK;
    }

    ucs_assert_always(!ucp_ep_is_stub(ep));

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
    status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REQUEST, ep_uuid, tl_bitmap,
                                 rsc_tli);
    ep->flags |= UCP_EP_FLAG_CONNECT_REQ_QUEUED;
    return status;
}

static void ucp_wireup_msg_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                                uint8_t id, const void *data, size_t length,
                                char *buffer, size_t max)
{
    ucp_context_h context       = worker->context;
    const ucp_wireup_msg_t *msg = data;
    char peer_name[UCP_WORKER_NAME_MAX + 1];
    ucp_address_entry_t *address_list, *ae;
    ucp_tl_resource_desc_t *rsc;
    unsigned address_count;
    ucp_lane_index_t lane;
    uint64_t uuid;
    char *p, *end;

    ucp_address_unpack(msg + 1, &uuid, peer_name, sizeof(peer_name),
                       &address_count, &address_list);

    p   = buffer;
    end = buffer + max;
    snprintf(p, end - p, "WIREUP %s [%s uuid 0x%"PRIx64" ep_uuid 0x%"PRIx64"]",
             (msg->type == UCP_WIREUP_MSG_REQUEST ) ? "REQ" :
             (msg->type == UCP_WIREUP_MSG_REPLY   ) ? "REP" :
             (msg->type == UCP_WIREUP_MSG_ACK     ) ? "ACK" : "",
             peer_name, uuid, msg->ep_uuid);

    p += strlen(p);
    for (ae = address_list; ae < address_list + address_count; ++ae) {
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
            if (msg->tli[lane] == (ae - address_list)) {
                snprintf(p, end - p, "/lane[%d]", lane);
                p += strlen(p);
            }
        }
    }

    ucs_free(address_list);
}

UCP_DEFINE_AM(-1, UCP_AM_ID_WIREUP, ucp_wireup_msg_handler, 
              ucp_wireup_msg_dump, UCT_CB_FLAG_ASYNC);
