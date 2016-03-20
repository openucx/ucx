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


static int ucp_wireup_check_runtime(uct_iface_attr_t *iface_attr,
                                    char *reason, size_t max)
{
    if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_DUP) {
        strncpy(reason, "full reliability", max);
        return 0;
    }

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY)) {
            strncpy(reason, "am_bcopy for wireup", max);
            return 0;
        }
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PENDING)) {
        strncpy(reason, "pending", max);
        return 0;
    }

    return 1;
}

static double ucp_wireup_am_score_func(ucp_worker_h worker,
                                       uct_iface_attr_t *iface_attr,
                                       char *reason, size_t max)
{

    if (!ucp_wireup_check_runtime(iface_attr, reason, max)) {
        return 0.0;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT)) {
        strncpy(reason, "am_short for tag", max);
        return 0.0;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_CB_SYNC)) {
        strncpy(reason, "sync am callback for tag", max);
        return 0.0;
    }

    if (worker->context->config.features & UCP_FEATURE_WAKEUP) {
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_WAKEUP)) {
            strncpy(reason, "wakeup", max);
            return 0.0;
        }
    }

    return 1e-3 / (iface_attr->latency + (iface_attr->overhead * 2));
}

static double ucp_wireup_rma_score_func(ucp_worker_h worker,
                                        uct_iface_attr_t *iface_attr,
                                        char *reason, size_t max)
{
    if (!ucp_wireup_check_runtime(iface_attr, reason, max)) {
        return 0.0;
    }

    /* TODO remove this requirement once we have RMA emulation */
    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_SHORT)) {
        strncpy(reason, "put_short for rma", max);
        return 0.0;
    }
    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_BCOPY)) {
        strncpy(reason, "put_bcopy for rma", max);
        return 0.0;
    }
    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_GET_BCOPY)) {
        strncpy(reason, "get_bcopy for rma", max);
        return 0.0;
    }

    /* best for 4k messages */
    return 1e-3 / (iface_attr->latency + iface_attr->overhead +
                    (4096.0 / iface_attr->bandwidth));
}

static double ucp_wireup_amo_score_func(ucp_worker_h worker,
                                        uct_iface_attr_t *iface_attr,
                                        char *reason, size_t max)
{
    uint64_t features = worker->context->config.features;

    if (!ucp_wireup_check_runtime(iface_attr, reason, max)) {
        return 0.0;
    }

    if (features & UCP_FEATURE_AMO32) {
        /* TODO remove this requirement once we have SW atomics */
        if (!ucs_test_all_flags(iface_attr->cap.flags,
                                UCT_IFACE_FLAG_ATOMIC_ADD32 |
                                UCT_IFACE_FLAG_ATOMIC_FADD32 |
                                UCT_IFACE_FLAG_ATOMIC_SWAP32 |
                                UCT_IFACE_FLAG_ATOMIC_CSWAP32))
        {
            strncpy(reason, "all 32-bit atomics", max);
            return 0.0;
        }
    }

    if (features & UCP_FEATURE_AMO64) {
        /* TODO remove this requirement once we have SW atomics */
        if (!ucs_test_all_flags(iface_attr->cap.flags,
                                UCT_IFACE_FLAG_ATOMIC_ADD64 |
                                UCT_IFACE_FLAG_ATOMIC_FADD64 |
                                UCT_IFACE_FLAG_ATOMIC_SWAP64 |
                                UCT_IFACE_FLAG_ATOMIC_CSWAP64))
        {
            strncpy(reason, "all 64-bit atomics", max);
            return 0.0;
        }
    }

    return 1e-3 / (iface_attr->latency + (iface_attr->overhead * 2));
}

/**
 * Select a local and remote transport
 */
ucs_status_t ucp_select_transport(ucp_ep_h ep,
                                  const ucp_address_entry_t *address_list,
                                  unsigned address_count, ucp_rsc_index_t pd_index,
                                  ucp_rsc_index_t *rsc_index_p,
                                  unsigned *dst_addr_index_p,
                                  ucp_wireup_score_function_t score_func,
                                  const char *title)
{
    ucp_worker_h worker = ep->worker;
    ucp_context_h context = worker->context;
    uct_tl_resource_desc_t *resource;
    const ucp_address_entry_t *ae;
    ucp_rsc_index_t rsc_index;
    double score, best_score;
    char tls_info[256];
    char tl_reason[64];
    char *p, *endp;
    uct_iface_h iface;
    int reachable;
    int found;

    found      = 0;
    best_score = 0.0;
    p          = tls_info;
    endp       = tls_info + sizeof(tls_info) - 1;
    *endp      = 0;

    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        resource   = &context->tl_rscs[rsc_index].tl_rsc;
        iface      = worker->ifaces[rsc_index];

        /* Must use only the pd the remote side explicitly requested */
        if ((pd_index != UCP_NULL_RESOURCE) &&
            (pd_index != context->tl_rscs[rsc_index].pd_index))
        {
            const char * pd_name = context->pd_rscs[pd_index].pd_name;
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : not on pd %s",
                      UCT_TL_RESOURCE_DESC_ARG(resource), pd_name);
            snprintf(p, endp - p, ", "UCT_TL_RESOURCE_DESC_FMT" - not on pd %s",
                     UCT_TL_RESOURCE_DESC_ARG(resource), pd_name);
            p += strlen(p);
            continue;
        }

        /* Get local device score */
        score = score_func(worker, &worker->iface_attrs[rsc_index], tl_reason,
                           sizeof(tl_reason));
        if (score <= 0.0) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " :  not suitable for %s, no %s",
                      UCT_TL_RESOURCE_DESC_ARG(resource), title, tl_reason);
            snprintf(p, endp - p, ", "UCT_TL_RESOURCE_DESC_FMT" - no %s",
                     UCT_TL_RESOURCE_DESC_ARG(resource), tl_reason);
            p += strlen(p);
            continue;
        }

        /* Check if remote peer is reachable using one of its devices */
        reachable = 0;
        for (ae = address_list; ae < address_list + address_count; ++ae) {
            /* Must be reachable device address, on same transport */
            reachable = !strcmp(ae->tl_name, resource->tl_name) &&
                         uct_iface_is_reachable(iface, ae->dev_addr);
            if (reachable) {
                break;
            }
        }
        if (!reachable) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : cannot reach to %s",
                      UCT_TL_RESOURCE_DESC_ARG(resource), ucp_ep_peer_name(ep));
            snprintf(p, endp - p, ", "UCT_TL_RESOURCE_DESC_FMT" - unreachable",
                     UCT_TL_RESOURCE_DESC_ARG(resource));
            p += strlen(p);
            continue;
        }

        ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : %s score %.2f",
                  UCT_TL_RESOURCE_DESC_ARG(resource), title, score);
        if (!found || (score > best_score)) {
            found             = 1;
            best_score        = score;
            *rsc_index_p      = rsc_index;
            *dst_addr_index_p = ae - address_list;
        }
    }

    if (!found) {
        ucs_error("No suitable %s transport to %s: %s", title, ucp_ep_peer_name(ep),
                  tls_info + 2);
        return UCS_ERR_UNREACHABLE;
    }

    ucs_trace("ep %p: selected for %s: " UCT_TL_RESOURCE_DESC_FMT
              " -> '%s' address[%d] score %.2f", ep, title,
              UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[*rsc_index_p].tl_rsc),
              ucp_ep_peer_name(ep), *dst_addr_index_p, best_score);
    return UCS_OK;
}

static void ucp_wireup_msg_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                                uint8_t id, const void *data, size_t length,
                                char *buffer, size_t max)
{
    const ucp_wireup_msg_t *msg = data;
    char peer_name[UCP_WORKER_NAME_MAX + 1];
    ucp_address_entry_t *address_list, *ae;
    unsigned address_count;
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
        snprintf(p, end - p, " [%s(%zu)%s]", ae->tl_name, ae->tl_addr_len,
                 ((ae - address_list) == msg->tli[UCP_EP_OP_AM] )  ? " am" :
                 ((ae - address_list) == msg->tli[UCP_EP_OP_RMA] ) ? " rma" :
                 ((ae - address_list) == msg->tli[UCP_EP_OP_AMO] ) ? " amo" :
                 "");
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
    packed_len = uct_ep_am_bcopy(ep->uct_eps[UCP_EP_OP_AM], UCP_AM_ID_WIREUP,
                                 ucp_wireup_msg_pack, req);
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

static int ucp_worker_is_tl_p2p(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    return !(worker->iface_attrs[rsc_index].cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE);
}

void ucp_wireup_msg_send_completion(void *request, ucs_status_t status)
{
    ucp_request_t *req = (ucp_request_t *)request - 1;
    ucs_free((void*)req->send.buffer);
}

static ucs_status_t ucp_wireup_msg_send(ucp_ep_h ep, uint8_t type)
{
    ucp_worker_h worker = ep->worker;
    uct_ep_h am_ep = ep->uct_eps[UCP_EP_OP_AM];
    ucp_rsc_index_t rsc_index, aux_rsc_index;
    ucs_status_t status;
    ucp_ep_op_t optype;
    uint64_t tl_bitmap;
    unsigned order[UCP_EP_OP_LAST + 1];
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
        aux_rsc_index = ucp_stub_ep_get_aux_rsc_index(am_ep);
        if (aux_rsc_index != UCP_NULL_RESOURCE) {
            tl_bitmap |= UCS_BIT(aux_rsc_index);
        }
    }
    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        rsc_index = ucp_ep_config(ep)->rscs[optype];
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
    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        if (req->send.wireup.type == UCP_WIREUP_MSG_ACK) {
            req->send.wireup.tli[optype] = -1;
        } else {
            rsc_index = ucp_ep_config(ep)->rscs[optype];
            req->send.wireup.tli[optype] = ucp_wireup_address_index(order,
                                                                    tl_bitmap,
                                                                    rsc_index);
        }
    }

    ucp_ep_add_pending(ep, ep->uct_eps[UCP_EP_OP_AM], req, 0);
    return UCS_OK;
}

static ucs_status_t ucp_wireup_connect_local(ucp_ep_h ep, const uint8_t *tli,
                                           unsigned address_count,
                                           const ucp_address_entry_t *address_list)
{
    ucp_worker_h worker = ep->worker;
    const ucp_address_entry_t *address;
    ucp_rsc_index_t rsc_index;
    ucs_status_t status;
    ucp_ep_op_t optype;

    ucs_trace("ep %p: connect local transports", ep);

    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        if (!ucp_ep_is_op_primary(ep, optype)) {
            continue;
        }

        rsc_index = ucp_ep_config(ep)->rscs[optype];
        if (!ucp_worker_is_tl_p2p(worker, rsc_index)) {
            continue;
        }

        address = &address_list[tli[optype]];
        ucs_assert(address->tl_addr_len > 0);
        status = uct_ep_connect_to_ep(ep->uct_eps[optype], address->dev_addr,
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
    ucp_ep_op_t optype;

    ucs_trace("ep %p: remote connected", ep);

    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        if (!ucp_ep_is_op_primary(ep, optype)) {
            continue;
        }

        rsc_index = ucp_ep_config(ep)->rscs[optype];
        if (!ucp_worker_is_tl_p2p(worker, rsc_index)) {
            continue;
        }

        ucp_stub_ep_remote_connected(ep->uct_eps[optype]);
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
    } else if (ep->cfg_index == 0) {
        /* Fill the transports of an existing stub endpoint */
        status = ucp_ep_init_trasports(ep, address_count, address_list);
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

        ucs_trace("ep %p: sending wireup reply", ep);

        status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REPLY);
        if (status != UCS_OK) {
            return;
        }

        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
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

ucs_status_t ucp_ep_init_trasports(ucp_ep_h ep, unsigned address_count,
                                   const ucp_address_entry_t *address_list)
{
    ucp_worker_h worker = ep->worker;
    ucp_context_h context = worker->context;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t rscs[UCP_EP_OP_LAST];
    unsigned addr_indices[UCP_EP_OP_LAST];
    ucp_rsc_index_t rsc_index;
    ucp_ep_op_t optype, dup;
    unsigned addr_index;
    ucs_status_t status;
    uct_ep_h new_uct_ep;
    int has_p2p;

    ucs_trace("ep %p: initialize transports", ep);

    ucs_assert(ep->cfg_index == 0);

    /* select best transport for every type of operation */
    has_p2p = 0;
    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        if (!(context->config.features & ucp_wireup_ep_ops[optype].features)) {
            rscs[optype]          = UCP_NULL_RESOURCE;
            addr_indices[optype] = -1;
            continue;
        }

        status = ucp_select_transport(ep, address_list, address_count,
                                      UCP_NULL_RESOURCE, &rsc_index,
                                      &addr_indices[optype],
                                      ucp_wireup_ep_ops[optype].score_func,
                                      ucp_wireup_ep_ops[optype].title);
        if (status != UCS_OK) {
            goto err;
        }

        rscs[optype] = rsc_index;
        has_p2p      = has_p2p || ucp_worker_is_tl_p2p(worker, rsc_index);
    }

    /* If one of the selected transports is p2p, we also need AM transport for
     * taking care of the wireup and sending final ACK.
     * The auxiliary wireup, if needed, will happen on the AM transport only.
     */
    if (has_p2p && (rscs[UCP_EP_OP_AM] = UCP_NULL_RESOURCE)) {
        status = ucp_select_transport(ep, address_list, address_count,
                                      UCP_NULL_RESOURCE, &rscs[UCP_EP_OP_AM],
                                      &addr_indices[UCP_EP_OP_AM],
                                      ucp_wireup_ep_ops[UCP_EP_OP_AM].score_func,
                                      ucp_wireup_ep_ops[UCP_EP_OP_AM].title);
        if (status != UCS_OK) {
            goto err;
        }
    }

    /* group eps by their configuration of transports */
    ep->cfg_index   = ucp_worker_get_ep_config(worker, rscs);

    /* save remote protection domain index for rma and amo. this is used
     * to select the remote key for these operations.
     */
    if (context->config.features & UCP_FEATURE_RMA) {
        ep->rma_dst_pdi = address_list[addr_indices[UCP_EP_OP_RMA]].pd_index;
    } else {
        ep->rma_dst_pdi = -1;
    }
    if (context->config.features & (UCP_FEATURE_AMO32|UCP_FEATURE_AMO64)) {
        ep->amo_dst_pdi = address_list[addr_indices[UCP_EP_OP_AMO]].pd_index;
    } else {
        ep->amo_dst_pdi = -1;
    }

    /* establish connections on all underlying endpoint */
    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        rsc_index  = rscs[optype];
        addr_index = addr_indices[optype];

        if (rsc_index == UCP_NULL_RESOURCE) {
            if (ep->uct_eps[optype] != NULL) {
                /* destroy stub endpoint which might have been created. it should
                 * not have any queued operations. */
                ucs_trace("ep %p: destroy stub ep %p for %s", ep,
                          ep->uct_eps[optype], ucp_wireup_ep_ops[optype].title);
                uct_ep_destroy(ep->uct_eps[optype]);
                ep->uct_eps[optype] = NULL;
            }
            continue;
        }

        /* if a transport is selected for more than once operation, create only
         * one endpoint, and use it in the duplicates.
         */
        dup = ucp_ep_config(ep)->dups[optype];
        if (dup != UCP_EP_OP_LAST) {
            ucs_assertv(dup < UCP_EP_OP_LAST, "dup=%d", dup);
            ucs_trace("ep %p: use uct_ep %p for %s as a duplicate of %s",
                      ep, ep->uct_eps[dup], ucp_wireup_ep_ops[optype].title,
                      ucp_wireup_ep_ops[dup].title);
            ep->uct_eps[optype] = ep->uct_eps[dup];
            continue;
        }

        iface_attr = &worker->iface_attrs[rsc_index];

        /* if the selected transport can be connected directly to the remote
         * interface, just create a connected uct endpoint.
         */
        if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
            /* create an endpoint connected to the remote interface */
            ucs_assert(address_list[addr_index].tl_addr_len > 0);
            status = uct_ep_create_connected(worker->ifaces[rsc_index],
                                             address_list[addr_index].dev_addr,
                                             address_list[addr_index].iface_addr,
                                             &new_uct_ep);
            if (status != UCS_OK) {
                goto err;
            }

            /* If ep already exists, it's a stub, and we need to update its next_ep
             * instead of replacing it.
             */
            if (ep->uct_eps[optype] == NULL) {
                ucs_trace("ep %p: assign uct_ep %p for %s", ep, new_uct_ep,
                          ucp_wireup_ep_ops[optype].title);
                ep->uct_eps[optype] = new_uct_ep;
            } else {
                ucs_trace("ep %p: assign set stub_ep %p next to %p for %s",
                          ep, ep->uct_eps[optype], new_uct_ep,
                          ucp_wireup_ep_ops[optype].title);
                ucp_stub_ep_set_next_ep(ep->uct_eps[optype], new_uct_ep);
                ucp_stub_ep_remote_connected(ep->uct_eps[optype]);
            }
        } else if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
            /* create a stub endpoint which will start connection establishment
             * protocol using an auxiliary transport.
             */

            /* If ep already exists, it's a stub, and we need to start auxiliary
             * wireup on this stub.
             */
            if (ep->uct_eps[optype] == NULL) {
                status = ucp_stub_ep_create(ep, optype, address_count,
                                            address_list, &ep->uct_eps[optype]);
                ucs_trace("ep %p: create set stub_ep %p for %s", ep,
                          ep->uct_eps[optype], ucp_wireup_ep_ops[optype].title);
            } else {
                status = ucp_stub_ep_connect(ep->uct_eps[optype], address_count,
                                             address_list);
                ucs_trace("ep %p: connect stub_ep %p for %s", ep,
                          ep->uct_eps[optype], ucp_wireup_ep_ops[optype].title);
            }
            if (status != UCS_OK) {
                goto err;
            }
        } else {
            status = UCS_ERR_UNREACHABLE;
            goto err;
        }
    }

    if (!has_p2p) {
        ep->flags |= UCP_EP_FLAG_LOCAL_CONNECTED;
    }

    return UCS_OK;

err:
    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        if (ep->uct_eps[optype] != NULL) {
            uct_ep_destroy(ep->uct_eps[optype]);
            ep->uct_eps[optype] = NULL;
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

ucp_wireup_ep_op_t ucp_wireup_ep_ops[] = {
    [UCP_EP_OP_AM]  = {
        .title      = "active messages",
        .features   = UCP_FEATURE_TAG,
        .score_func = ucp_wireup_am_score_func
    },
    [UCP_EP_OP_RMA] = {
        .title      = "remote memory access",
        .features   = UCP_FEATURE_RMA,
        .score_func = ucp_wireup_rma_score_func
    },
    [UCP_EP_OP_AMO] = {
        .title      = "atomics",
        .features   = UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64,
        .score_func = ucp_wireup_amo_score_func
    }
};
