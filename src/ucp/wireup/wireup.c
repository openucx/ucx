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
#include <ucs/algorithm/qsort_r.h>
#include <ucs/arch/bitops.h>
#include <ucs/async/async.h>
#include <ucs/datastruct/queue.h>

/*
 * Description of the protocol in UCX wiki:
 * https://github.com/openucx/ucx/wiki/Connection-establishment
 */

enum {
    UCP_WIREUP_LANE_USAGE_AM  = UCS_BIT(0),
    UCP_WIREUP_LANE_USAGE_RMA = UCS_BIT(1),
    UCP_WIREUP_LANE_USAGE_AMO = UCS_BIT(2)
};


typedef struct {
    ucp_rsc_index_t   rsc_index;
    unsigned          addr_index;
    ucp_rsc_index_t   dst_pd_index;
    uint32_t          usage;
    double            rma_score;
    double            amo_score;
} ucp_wireup_lane_desc_t;


static int ucp_wireup_check_runtime(const uct_iface_attr_t *iface_attr,
                                    char *reason, size_t max)
{
    if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_DUP) {
        strncpy(reason, "full reliability", max);
        return 0;
    }

    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP) {
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) ||
            (iface_attr->cap.am.max_bcopy < UCP_MIN_BCOPY))
        {
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
                                       const uct_iface_attr_t *iface_attr,
                                       char *reason, size_t max)
{

    if (!ucp_wireup_check_runtime(iface_attr, reason, max)) {
        return 0.0;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) ||
        (iface_attr->cap.am.max_bcopy < UCP_MIN_BCOPY))
    {
        strncpy(reason, "am_bcopy for tag", max);
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
                                        const uct_iface_attr_t *iface_attr,
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
    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) ||
        (iface_attr->cap.put.max_bcopy < UCP_MIN_BCOPY))
    {
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
                                        const uct_iface_attr_t *iface_attr,
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

static int ucp_wireup_check_auxiliary(const uct_iface_attr_t *iface_attr,
                                      char *reason, size_t max)
{
    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) ||
        (iface_attr->cap.am.max_bcopy < UCP_MIN_BCOPY))
    {
        strncpy(reason, "am_bcopy for wireup", max);
        return 0;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE)) {
        strncpy(reason, "connecting to iface", max);
        return 0;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PENDING)) {
        strncpy(reason, "pending", max);
        return 0;
    }

    if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_AM_CB_ASYNC)) {
        strncpy(reason, "async am callback", max);
        return 0;
    }

    return 1;
}

static double ucp_wireup_aux_score_func(ucp_worker_h worker,
                                        const uct_iface_attr_t *iface_attr,
                                        char *reason, size_t max)
{
    if (!ucp_wireup_check_auxiliary(iface_attr, reason, max)) {
        strncpy(reason, "async am callback", max);
        return 0.0;
    }

    return (1e-3 / iface_attr->latency) +
           (1e3 * ucs_max(iface_attr->cap.am.max_bcopy, iface_attr->cap.am.max_short));
}

/**
 * Select a local and remote transport
 */
static UCS_F_NOINLINE ucs_status_t
ucp_wireup_select_transport(ucp_ep_h ep, const ucp_address_entry_t *address_list,
                            unsigned address_count, ucp_wireup_score_function_t score_func,
                            uint64_t remote_pd_flags, int show_error,
                            const char *title, ucp_rsc_index_t *rsc_index_p,
                            unsigned *dst_addr_index_p, double *score_p)
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
                        uct_iface_is_reachable(iface, ae->dev_addr) &&
                        ucs_test_all_flags(ae->pd_flags, remote_pd_flags);
            if (reachable) {
                break;
            }
        }
        if (!reachable) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : cannot reach to %s with pd_flags 0x%"PRIx64,
                      UCT_TL_RESOURCE_DESC_ARG(resource), ucp_ep_peer_name(ep),
                      remote_pd_flags);
            snprintf(p, endp - p, ", "UCT_TL_RESOURCE_DESC_FMT" - cannot reach remote",
                     UCT_TL_RESOURCE_DESC_ARG(resource));
            p += strlen(p);
            if (remote_pd_flags) {
                snprintf(p, endp - p, "%s%s%s memory",
                         (remote_pd_flags & UCT_PD_FLAG_REG)   ? " registered" : "",
                         ucs_test_all_flags(remote_pd_flags, UCT_PD_FLAG_REG|
                                                             UCT_PD_FLAG_ALLOC)
                                                               ? " or"         : "",
                         (remote_pd_flags & UCT_PD_FLAG_ALLOC) ? " allocated " : "");
            } else {
                snprintf(p, endp - p, " worker");
            }
            p += strlen(p);
            continue;
        }

        ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : %s score %.2f",
                  UCT_TL_RESOURCE_DESC_ARG(resource), title, score);
        if (!found || (score > best_score)) {
            *rsc_index_p      = rsc_index;
            *dst_addr_index_p = ae - address_list;
            *score_p          = score;
            best_score        = score;
            found             = 1;
        }
    }

    if (!found) {
        if (show_error) {
            ucs_error("No suitable %s transport to %s: %s", title, ucp_ep_peer_name(ep),
                      tls_info + 2);
        }
        return UCS_ERR_UNREACHABLE;
    }

    ucs_trace("ep %p: selected for %s: " UCT_TL_RESOURCE_DESC_FMT
              " -> '%s' address[%d] score %.2f", ep, title,
              UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[*rsc_index_p].tl_rsc),
              ucp_ep_peer_name(ep), *dst_addr_index_p, best_score);
    return UCS_OK;
}

ucs_status_t ucp_wireup_select_aux_transport(ucp_ep_h ep,
                                             const ucp_address_entry_t *address_list,
                                             unsigned address_count,
                                             ucp_rsc_index_t *rsc_index_p,
                                             unsigned *addr_index_p)
{
    double score;
    return ucp_wireup_select_transport(ep, address_list, address_count,
                                       ucp_wireup_aux_score_func, 0, 1, "auxiliary",
                                       rsc_index_p, addr_index_p, &score);
}

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
        snprintf(p, end - p, "%s(%zu)]", ae->tl_name, ae->tl_addr_len);
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

static UCS_F_NOINLINE void
ucp_wireup_add_lane_desc(ucp_wireup_lane_desc_t *lane_descs,
                         ucp_lane_index_t *num_lanes_p, ucp_rsc_index_t rsc_index,
                         unsigned addr_index, ucp_rsc_index_t dst_pd_index,
                         double score, uint32_t usage)
{
    ucp_lane_index_t i;

    for (i = 0; i < *num_lanes_p; ++i) {
        if ((lane_descs[i].rsc_index == rsc_index) &&
            (lane_descs[i].addr_index == addr_index))
        {
            ucs_assert(dst_pd_index == lane_descs[i].dst_pd_index);
            lane_descs[i].usage |= usage;
            return;
        }
    }

    lane_descs[*num_lanes_p].rsc_index    = rsc_index;
    lane_descs[*num_lanes_p].addr_index   = addr_index;
    lane_descs[*num_lanes_p].dst_pd_index = dst_pd_index;
    lane_descs[*num_lanes_p].usage        = usage;
    ++(*num_lanes_p);
}

static UCS_F_NOINLINE void
ucp_wireup_address_list_remove_pd(ucp_address_entry_t *address_list,
                                  unsigned address_count,
                                  ucp_rsc_index_t pd_index)
{
    unsigned i;
    for (i = 0; i < address_count; ++i) {
        if (address_list[i].pd_index == pd_index) {
            address_list[i].pd_flags = 0;
        }
    }
}

static int ucp_wireup_compare_score(double score1, double score2)
{
    /* sort from highest score to lowest */
    return (score1 < score2) ? 1 : ((score1 > score2) ? -1 : 0);
}

static int ucp_wireup_compare_lane_rma_score(const void *elem1, const void *elem2)
{
    const ucp_wireup_lane_desc_t *desc1 = elem1;
    const ucp_wireup_lane_desc_t *desc2 = elem2;

    return ucp_wireup_compare_score(desc1->rma_score, desc2->rma_score);
}

static int ucp_wireup_compare_lane_amo_score(const void *elem1, const void *elem2,
                                             void *arg)
{
    const ucp_lane_index_t *lane1  = elem1;
    const ucp_lane_index_t *lane2  = elem2;
    const ucp_wireup_lane_desc_t *lanes = arg;

    return ucp_wireup_compare_score(lanes[*lane1].amo_score, lanes[*lane2].amo_score);
}

static UCS_F_NOINLINE ucs_status_t
ucp_wireup_add_memaccess_lanes(ucp_ep_h ep, unsigned address_count,
                               const ucp_address_entry_t *address_list,
                               ucp_wireup_lane_desc_t *lane_descs,
                               ucp_lane_index_t *num_lanes_p,
                               ucp_wireup_score_function_t score_func,
                               const char *title_fmt, uint64_t features,
                               uint32_t usage)
{
    ucp_worker_h worker   = ep->worker;
    ucp_context_h context = worker->context;
    ucp_address_entry_t *address_list_copy;
    ucp_rsc_index_t rsc_index, dst_pd_index;
    size_t address_list_size;
    double score, reg_score;
    unsigned addr_index;
    ucs_status_t status;
    char title[64];

    if (!(context->config.features & features)) {
        status = UCS_OK;
        goto out;
    }

    /* Create a copy of the address list */
    address_list_size = sizeof(*address_list_copy) * address_count;
    address_list_copy = ucs_malloc(address_list_size, "rma address list");
    if (address_list_copy == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    memcpy(address_list_copy, address_list, address_list_size);

    /* Select best transport which can reach registered memory */
    snprintf(title, sizeof(title), title_fmt, "registered");
    status = ucp_wireup_select_transport(ep, address_list_copy, address_count,
                                         score_func, UCT_PD_FLAG_REG, 1, title,
                                         &rsc_index, &addr_index, &score);
    if (status != UCS_OK) {
        goto out_free_address_list;
    }

    dst_pd_index = address_list_copy[addr_index].pd_index;
    reg_score    = score;

    /* Add to the list of lanes and remove all occurrences of the remote pd
     * from the address list, to avoid selecting the same remote pd again.*/
    ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                             dst_pd_index, score, usage);
    ucp_wireup_address_list_remove_pd(address_list_copy, address_count,
                                      dst_pd_index);

    /* Select additional transports which can access allocated memory, but only
     * if their scores are better. We need this because a remote memory block can
     * be potentially allocated using one of them, and we might get better performance
     * than the transports which support only registered remote memory.
     */
    snprintf(title, sizeof(title), title_fmt, "allocated");
    while (address_count > 0) {
        status = ucp_wireup_select_transport(ep, address_list_copy, address_count,
                                             score_func, UCT_PD_FLAG_ALLOC, 0,
                                             title, &rsc_index, &addr_index, &score);
        if ((status != UCS_OK) || (score <= reg_score)) {
            break;
        }

        /* Add lane description and remove all occurrences of the remote pd */
        dst_pd_index = address_list_copy[addr_index].pd_index;
        ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                                 dst_pd_index, score, usage);
        ucp_wireup_address_list_remove_pd(address_list_copy, address_count,
                                          dst_pd_index);
    }

    status = UCS_OK;

out_free_address_list:
    ucs_free(address_list_copy);
out:
    return status;
}

static ucs_status_t ucp_wireup_select_transports(ucp_ep_h ep, unsigned address_count,
                                                 const ucp_address_entry_t *address_list,
                                                 unsigned *addr_indices)
{
    ucp_worker_h worker   = ep->worker;
    ucp_context_h context = worker->context;
    ucp_wireup_lane_desc_t lane_descs[UCP_MAX_LANES];
    ucp_lane_index_t lane, num_lanes, num_amo_lanes;
    ucp_rsc_index_t rsc_index, dst_pd_index;
    uct_iface_attr_t *iface_attr;
    ucp_ep_config_key_t key;
    ucs_status_t status;
    unsigned addr_index;
    char reason[64];
    double score;
    int need_am;

    ucs_assert(ep->cfg_index == 0);

    num_lanes = 0;

    /* Select lanes for remote memory access */
    status = ucp_wireup_add_memaccess_lanes(ep, address_count, address_list,
                                            lane_descs, &num_lanes,
                                            ucp_wireup_rma_score_func,
                                            "remote %s memory access",
                                            UCP_FEATURE_RMA,
                                            UCP_WIREUP_LANE_USAGE_RMA);
    if (status != UCS_OK) {
        return status;
    }

    /* Select lanes for atomic operations */
    status = ucp_wireup_add_memaccess_lanes(ep, address_count, address_list,
                                            lane_descs, &num_lanes,
                                            ucp_wireup_amo_score_func,
                                            "atomic operations on %s memory",
                                            UCP_FEATURE_AMO32|UCP_FEATURE_AMO64,
                                            UCP_WIREUP_LANE_USAGE_AMO);
    if (status != UCS_OK) {
        return status;
    }

    /* Check if we need active messages, for wireup */
    need_am = 0;
    for (lane = 0; lane < num_lanes; ++lane) {
        need_am = need_am || ucp_worker_is_tl_p2p(worker,
                                                  lane_descs[lane].rsc_index);
    }

    /* Select one lane for active messages */
    if ((context->config.features & UCP_FEATURE_TAG) || need_am) {
        status = ucp_wireup_select_transport(ep, address_list, address_count,
                                             ucp_wireup_am_score_func, 0, 1,
                                             "active messages", &rsc_index,
                                             &addr_index, &score);
        if (status != UCS_OK) {
            return status;
        }

        ucp_wireup_add_lane_desc(lane_descs, &num_lanes, rsc_index, addr_index,
                                 address_list[addr_index].pd_index, score,
                                 UCP_WIREUP_LANE_USAGE_AM);

        /* TODO select transport for rendezvous, which needs high-bandwidth
         * zero-copy rma to registered memory.
         */
    }

    /* User should not create endpoints unless requested communication features */
    if (num_lanes == 0) {
        ucs_error("No transports selected to %s", ucp_ep_peer_name(ep));
        return UCS_ERR_UNREACHABLE;
    }

    /* Sort lanes according to RMA score */
    for (lane = 0; lane < num_lanes; ++lane) {
        rsc_index  = lane_descs[lane].rsc_index;
        iface_attr = &worker->iface_attrs[rsc_index];
        lane_descs[lane].rma_score = ucp_wireup_rma_score_func(worker, iface_attr,
                                                               NULL, 0);
        lane_descs[lane].amo_score = ucp_wireup_amo_score_func(worker, iface_attr,
                                                               NULL, 0);
    }
    qsort(lane_descs, num_lanes, sizeof(*lane_descs),
          ucp_wireup_compare_lane_rma_score);

    num_amo_lanes       = 0;
    memset(&key.amo_lanes, 0, sizeof(key.amo_lanes));

    /* Construct the endpoint configuration key:
     * - arrange lane description in the EP configuration
     * - create remote PD bitmap
     * - create bitmap of lanes used for RMA and AMO
     * - if AM lane exists and fits for wireup messages, select it fot his purpose.
     */
    key.num_lanes       = num_lanes;
    key.am_lane         = UCP_NULL_LANE;
    key.rma_lane_map    = 0;
    key.amo_lane_map    = 0;
    key.wireup_msg_lane = UCP_NULL_LANE;
    for (lane = 0; lane < num_lanes; ++lane) {
        rsc_index          = lane_descs[lane].rsc_index;
        dst_pd_index       = lane_descs[lane].dst_pd_index;
        key.lanes[lane]    = rsc_index;
        addr_indices[lane] = lane_descs[lane].addr_index;
        ucs_assert(lane_descs[lane].usage != 0);

        /* Active messages - add to am_lanes map, check if we can use for wireup */
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AM) {
            ucs_assert(key.am_lane == UCP_NULL_LANE);
            key.am_lane = lane;
            if (ucp_wireup_check_auxiliary(&worker->iface_attrs[rsc_index], reason,
                                           sizeof(reason)))
            {
                key.wireup_msg_lane = lane;
            } else {
                ucs_trace("will not use lane[%d] "UCT_TL_RESOURCE_DESC_FMT
                          " for wireup messages because no %s", lane,
                          UCT_TL_RESOURCE_DESC_ARG(&worker->context->tl_rscs[rsc_index].tl_rsc),
                          reason);
            }
        }

        /* RMA, AMO - add to lanes map and remote pd map */
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_RMA) {
            key.rma_lane_map |= UCS_BIT(dst_pd_index + lane * UCP_PD_INDEX_BITS);
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AMO) {
            key.amo_lanes[num_amo_lanes] = lane;
            ++num_amo_lanes;
        }
    }

    /* Sort and add AMO lanes */
    ucs_qsort_r(key.amo_lanes, num_amo_lanes, sizeof(*key.amo_lanes),
                ucp_wireup_compare_lane_amo_score, lane_descs);
    for (lane = 0; lane < num_amo_lanes; ++lane) {
        dst_pd_index      = lane_descs[key.amo_lanes[lane]].dst_pd_index;
        key.amo_lane_map |= UCS_BIT(dst_pd_index + lane * UCP_PD_INDEX_BITS);
    }

    /* Add all reachable remote pd's */
    key.reachable_pd_map = 0;
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        for (addr_index = 0; addr_index < address_count; ++addr_index) {
            if (!strcmp(address_list[addr_index].tl_name, context->tl_rscs[rsc_index].tl_rsc.tl_name) &&
                uct_iface_is_reachable(worker->ifaces[rsc_index], address_list[addr_index].dev_addr))
            {
                key.reachable_pd_map |= UCS_BIT(address_list[addr_index].pd_index);
            }
        }
    }

    /* If we did not select the AM lane for active messages, use the first p2p
     * transport, if exists. Otherwise, we don't have a lane for wireup messages,
     * and we don't need one anyway.
     */
    if (key.wireup_msg_lane == UCP_NULL_LANE) {
        for (lane = 0; lane < num_lanes; ++lane) {
            if (ucp_worker_is_tl_p2p(worker, lane_descs[lane].rsc_index)) {
                key.wireup_msg_lane = lane;
                break;
            }
        }
    }

    /* Print debug info */
    for (lane = 0; lane < num_lanes; ++lane) {
        rsc_index = lane_descs[lane].rsc_index;
        ucs_debug("ep %p: lane[%d] using rsc[%d] "UCT_TL_RESOURCE_DESC_FMT
                  " to pd[%d], for%s%s%s%s", ep, lane, rsc_index,
                  UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[rsc_index].tl_rsc),
                  lane_descs[lane].dst_pd_index,
                  (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AM)  ? " [active message]"       : "",
                  (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_RMA) ? " [remote memory access]" : "",
                  (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AMO) ? " [atomic operations]"    : "",
                  (key.wireup_msg_lane == lane         )               ? " [wireup messages]"      : "");
    }
    ucs_debug("ep %p: am_lane %d wirep_lane %d rma_lane_map 0x%"PRIx64
              " amo_lane_map 0x%"PRIx64" reachable_pds 0x%x",
              ep, key.am_lane, key.wireup_msg_lane, key.rma_lane_map,
              key.amo_lane_map, key.reachable_pd_map);

    /* Allocate/reuse configuration index */
    ep->cfg_index = ucp_worker_get_ep_config(worker, &key);

    /* Cache AM lane index on the endpoint */
    ep->am_lane   = key.am_lane;

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
    if (iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE) {
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

