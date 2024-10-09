/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2021. ALL RIGHTS RESERVED.
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

#include <ucs/async/async.h>
#include <ucs/datastruct/queue.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_listener.h>
#include <ucp/core/ucp_proxy_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/proto/proto_common.h>
#include <ucs/sys/iovec.h>
#include <ucp/tag/eager.h>

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_am.inl>

/*
 * Description of the protocol in UCX wiki:
 * https://github.com/openucx/ucx/wiki/Connection-establishment
 */


/* Validate wireup message, implemented as a macro to prevent static checker
 * warnings */
#define UCP_WIREUP_MSG_CHECK(_msg, _ep, _msg_type) \
    do { \
        ucs_assert((_msg)->type == (_msg_type)); \
        if ((_msg_type) == UCP_WIREUP_MSG_REQUEST) { \
            ucs_assert(((_msg)->dst_ep_id == UCS_PTR_MAP_KEY_INVALID) != \
                       ((_ep) != NULL)); \
        } else { \
            ucs_assert((_msg)->dst_ep_id != UCS_PTR_MAP_KEY_INVALID); \
            ucs_assert((_ep) != NULL); \
        } \
    } while (0)


size_t ucp_wireup_msg_pack(void *dest, void *arg)
{
    struct iovec *wireup_msg_iov = (struct iovec*)arg;

    return ucs_iov_copy(wireup_msg_iov, 2, 0, dest,
                        wireup_msg_iov[0].iov_len + wireup_msg_iov[1].iov_len,
                        UCS_IOV_COPY_TO_BUF);
}

const char* ucp_wireup_msg_str(uint8_t msg_type)
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
    case UCP_WIREUP_MSG_EP_CHECK:
        return "EP_CHECK";
    case UCP_WIREUP_MSG_EP_REMOVED:
        return "EP_REMOVED";
    default:
        return "<unknown>";
    }
}

static ucp_lane_index_t ucp_wireup_get_msg_lane(ucp_ep_h ep, uint8_t msg_type)
{
    ucp_context_h   context    = ep->worker->context;
    ucp_ep_config_t *ep_config = ucp_ep_config(ep);
    ucp_lane_index_t lane, fallback_lane;

    if (msg_type == UCP_WIREUP_MSG_ACK) {
        lane          = ep_config->key.am_lane;
        fallback_lane = ep_config->key.wireup_msg_lane;
    } else {
        /* for request/response, try wireup_msg_lane first */
        lane          = ep_config->key.wireup_msg_lane;
        fallback_lane = ep_config->key.am_lane;
    }

    if (lane == UCP_NULL_LANE) {
        lane = fallback_lane;
    }

    if (lane == UCP_NULL_LANE) {
        ucs_fatal("ep %p to %s: could not find a lane to send CONN_%s%s",
                  ep, ucp_ep_peer_name(ep), ucp_wireup_msg_str(msg_type),
                  context->config.ext.unified_mode ?
                  ". try to set UCX_UNIFIED_MODE=n." : "");
    }

    return lane;
}

ucs_status_t ucp_wireup_msg_progress(uct_pending_req_t *self)
{
    ucp_request_t *req  = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep         = req->send.ep;
    ucs_status_t status;
    ssize_t packed_len;
    unsigned am_flags;
    struct iovec wireup_msg_iov[2];

    UCS_ASYNC_BLOCK(&ep->worker->async);

    if (req->send.wireup.type == UCP_WIREUP_MSG_REQUEST) {
        if (ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED) {
            ucs_trace("ep %p: not sending wireup message - remote already connected",
                      ep);
            status = UCS_OK;
            goto out_free_req;
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

    wireup_msg_iov[0].iov_base = &req->send.wireup;
    wireup_msg_iov[0].iov_len  = sizeof(req->send.wireup);

    wireup_msg_iov[1].iov_base = req->send.buffer;
    wireup_msg_iov[1].iov_len  = req->send.length;

    packed_len = uct_ep_am_bcopy(ucp_ep_get_lane(ep, req->send.lane),
                                 UCP_AM_ID_WIREUP, ucp_wireup_msg_pack,
                                 wireup_msg_iov, am_flags);
    if (ucs_unlikely(packed_len < 0)) {
        status = (ucs_status_t)packed_len;
        if (ucs_likely(status == UCS_ERR_NO_RESOURCE)) {
            goto out;
        }

        ucs_diag("failed to send wireup: %s", ucs_status_string(status));
        ucp_ep_set_failed_schedule(ep, req->send.lane, status);

        status = UCS_OK;
        goto out_free_req;
    } else {
        status = UCS_OK;
    }

    switch (req->send.wireup.type) {
    case UCP_WIREUP_MSG_PRE_REQUEST:
        ucp_ep_update_flags(ep, UCP_EP_FLAG_CONNECT_PRE_REQ_SENT, 0);
        break;
    case UCP_WIREUP_MSG_REQUEST:
        ucp_ep_update_flags(ep, UCP_EP_FLAG_CONNECT_REQ_SENT, 0);
        break;
    case UCP_WIREUP_MSG_REPLY:
        ucp_ep_update_flags(ep, UCP_EP_FLAG_CONNECT_REP_SENT, 0);
        break;
    case UCP_WIREUP_MSG_ACK:
        ucp_ep_update_flags(ep, UCP_EP_FLAG_CONNECT_ACK_SENT, 0);
        break;
    }

out_free_req:
    ucs_free(req->send.buffer);
    ucp_request_mem_free(req);
out:
    UCS_ASYNC_UNBLOCK(&ep->worker->async);
    return status;
}

ucs_status_t
ucp_wireup_msg_prepare(ucp_ep_h ep, uint8_t type,
                       const ucp_tl_bitmap_t *tl_bitmap,
                       const ucp_lane_index_t *lanes2remote,
                       ucp_wireup_msg_t *msg_hdr, void **address_p,
                       size_t *address_length_p)
{
    ucp_context_h context = ep->worker->context;
    unsigned pack_flags   = ucp_worker_default_address_pack_flags(ep->worker) |
                            UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX;
    ucs_status_t status;

    msg_hdr->type      = type;
    msg_hdr->err_mode  = ucp_ep_config(ep)->key.err_mode;
    msg_hdr->conn_sn   = ep->conn_sn;
    msg_hdr->src_ep_id = ucp_ep_local_id(ep);
    if (ep->flags & UCP_EP_FLAG_REMOTE_ID) {
        msg_hdr->dst_ep_id = ucp_ep_remote_id(ep);
    } else {
        /* Destination UCP endpoint ID must be packed in case of CM */
        ucs_assert(!ucp_ep_has_cm_lane(ep));
        msg_hdr->dst_ep_id = UCS_PTR_MAP_KEY_INVALID;
    }

    /* pack all addresses */
    status = ucp_address_pack(ep->worker, ep, tl_bitmap, pack_flags,
                              context->config.ext.worker_addr_version,
                              lanes2remote, UINT_MAX, address_length_p,
                              address_p);
    if (status != UCS_OK) {
        ucs_error("failed to pack address: %s", ucs_status_string(status));
    }

    return status;
}

static ucs_status_t
ucp_wireup_msg_send(ucp_ep_h ep, uint8_t type, const ucp_tl_bitmap_t *tl_bitmap,
                    const ucp_lane_index_t *lanes2remote)
{
    ucp_request_t *req;
    ucs_status_t status;

    ucs_assert(ep->cfg_index != UCP_WORKER_CFG_INDEX_NULL);

    if (ep->flags & UCP_EP_FLAG_FAILED) {
        ucs_debug("ep %p: not sending WIREUP message (%u), because ep failed",
                  ep, type);
        return UCS_ERR_CONNECTION_RESET;
    }

    /* We cannot allocate from memory pool because it's not thread safe
     * and this function may be called from any thread
     */
    req = ucp_request_mem_alloc("wireup_msg_req");
    if (req == NULL) {
        ucs_error("failed to allocate request for sending WIREUP message");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    req->flags         = 0;
    req->send.ep       = ep;
    req->send.uct.func = ucp_wireup_msg_progress;
    req->send.datatype = ucp_dt_make_contig(1);
    ucp_request_send_state_init(req, ucp_dt_make_contig(1), 0);

    status = ucp_wireup_msg_prepare(ep, type, tl_bitmap, lanes2remote,
                                    &req->send.wireup, &req->send.buffer,
                                    &req->send.length);
    if (status != UCS_OK) {
        ucp_request_mem_free(req);
        goto err;
    }

    ucp_request_send(req);
    /* coverity[leaked_storage] */
    return UCS_OK;

err:
    ucp_ep_set_failed_schedule(ep, UCP_NULL_LANE, status);
    return status;
}

static ucp_tl_bitmap_t
ucp_wireup_get_ep_tl_bitmap(ucp_ep_h ep, ucp_lane_map_t lane_map)
{
    ucp_tl_bitmap_t tl_bitmap = UCS_STATIC_BITMAP_ZERO_INITIALIZER;
    ucp_lane_index_t lane;

    ucs_for_each_bit(lane, lane_map) {
        ucs_assert(lane < UCP_MAX_LANES);
        if (ucp_ep_get_rsc_index(ep, lane) == UCP_NULL_RESOURCE) {
            continue;
        }

        UCS_STATIC_BITMAP_SET(&tl_bitmap, ucp_ep_get_rsc_index(ep, lane));
    }

    return tl_bitmap;
}

int ucp_wireup_connect_p2p(ucp_worker_h worker, ucp_rsc_index_t rsc_index,
                           int has_cm_lane)
{
    /* The EP with CM lane has to be connected to remote EP, so prefer native
     * UCT p2p capability. */
    return has_cm_lane ?
           ucp_worker_is_tl_p2p(worker, rsc_index) :
           !ucp_worker_is_tl_2iface(worker, rsc_index);
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
    ucp_lane_index_t lane, remote_lane, num_lanes;
    unsigned *ep_addr_indexes;
    unsigned ep_addr_index;
    uint64_t used_remote_lanes;

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
    num_lanes         = ucp_ep_num_lanes(ep);
    for (lane = 0; lane < num_lanes; ++lane) {
        if (!ucp_ep_is_lane_p2p(ep, lane)) {
            continue;
        }

        /* Select next remote ep address within the address_index as specified
         * by addr_indices argument
         */
        address_index      = addr_indices[lane];
        address            = &remote_address->address_list[address_index];
        ep_addr_index      = ep_addr_indexes[address_index]++;
        ucs_assertv(ep_addr_index < address->num_ep_addrs,
                    "lane=%d/%d tl_name_csum=0x%02x address_index=%u "
                    "ep_addr_index=%u num_ep_addrs=%u",
                    lane, num_lanes, address->tl_name_csum, address_index,
                    ep_addr_index, address->num_ep_addrs);
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
                                const ucp_address_entry_t **address_entry_p,
                                const ucp_address_entry_ep_addr_t **ep_entry_p)
{
    const ucp_address_entry_t *address;
    unsigned ep_addr_index;

    ucp_unpacked_address_for_each(address, remote_address) {
        for (ep_addr_index = 0; ep_addr_index < address->num_ep_addrs;
             ++ep_addr_index) {
            if (remote_lane == address->ep_addrs[ep_addr_index].lane) {
                *address_entry_p = address;
                *ep_entry_p      = &address->ep_addrs[ep_addr_index];
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
    const ucp_address_entry_t *address_entry;
    const ucp_address_entry_ep_addr_t *ep_entry;
    ucs_status_t status;

    ucs_trace("ep %p: connect local transports", ep);
    ucs_log_indent(1);

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (!ucp_ep_is_lane_p2p(ep, lane)) {
            continue;
        }

        remote_lane = (lanes2remote == NULL) ? lane : lanes2remote[lane];

        status = ucp_wireup_find_remote_p2p_addr(ep, remote_lane, remote_address,
                                                 &address_entry, &ep_entry);
        if (status != UCS_OK) {
            ucs_error("ep %p: no remote ep address for lane[%d]->remote_lane[%d]",
                      ep, lane, remote_lane);
            goto out;
        }

        status = ucp_wireup_ep_connect_to_ep_v2(ucp_ep_get_lane(ep, lane),
                                                address_entry, ep_entry);
        if (status != UCS_OK) {
            goto out;
        }
    }

    status = UCS_OK;

out:
    ucs_log_indent(-1);
    return status;
}

static int ucp_ep_has_wireup_msg_pending(ucp_ep_h ucp_ep)
{
    ucp_wireup_ep_t *wireup_ep;
    ucp_lane_index_t lane;

    for (lane = 0; lane < ucp_ep_num_lanes(ucp_ep); ++lane) {
        wireup_ep = ucp_wireup_ep(ucp_ep_get_lane(ucp_ep, lane));
        if ((wireup_ep != NULL) && (wireup_ep->pending_count != 0)) {
            return 1;
        }
    }

    return 0;
}

static void ucp_wireup_eps_progress_sched(ucp_ep_h ucp_ep)
{
    /* Use one-shot mode to avoid the need to save prog_id */
    ucs_callbackq_add_oneshot(&ucp_ep->worker->uct->progress_q, ucp_ep,
                              ucp_wireup_eps_progress, ucp_ep);
    ucp_worker_signal_internal(ucp_ep->worker);
}

/*
 * We switch the endpoints in this function (instead in wireup code) since
 * this is guaranteed to run from the main thread.
 */
unsigned ucp_wireup_eps_progress(void *arg)
{
    ucp_ep_h ucp_ep = arg;
    ucp_wireup_ep_t *wireup_ep;
    ucs_queue_head_t tmp_pending_queue;
    ucp_lane_index_t lane;

    UCS_ASYNC_BLOCK(&ucp_ep->worker->async);

    /* If we still have pending wireup messages, send them out first */
    if (ucp_ep_has_wireup_msg_pending(ucp_ep)) {
        /* Reschedule the progress since it's one-shot */
        ucp_wireup_eps_progress_sched(ucp_ep);
        goto out_unblock;
    }

    /* If an error happened on the endpoint (but perhaps the deferred error handler,
     * ucp_worker_iface_err_handle_progress(), was not called yet, avoid changing
     * ep state, and let the error handler take care of cleanup.
     */
    if (ucp_ep->flags & UCP_EP_FLAG_FAILED) {
        ucs_trace("ep %p: not switching wireup eps to ready state because of "
                  "error", ucp_ep);
        goto out_unblock;
    }

    /* Move wireup pending queue to temporary queue and remove references to
     * the wireup progress function
     */
    ucp_wireup_eps_pending_extract(ucp_ep, &tmp_pending_queue);
    for (lane = 0; lane < ucp_ep_num_lanes(ucp_ep); ++lane) {
        wireup_ep = ucp_wireup_ep(ucp_ep_get_lane(ucp_ep, lane));
        if (wireup_ep == NULL) {
            continue;
        }

        ucs_assert(wireup_ep->flags & UCP_WIREUP_EP_FLAG_READY);
        ucs_assert(wireup_ep->super.uct_ep != NULL);

        ucs_trace("ep %p: switching wireup_ep %p to ready state", ucp_ep,
                  wireup_ep);

        /* Switch to real transport and destroy proxy endpoint (aux_ep as well) */
        ucp_proxy_ep_replace(&wireup_ep->super);
    }

    /* Replay pending requests */
    ucp_wireup_replay_pending_requests(ucp_ep, &tmp_pending_queue);
    UCS_ASYNC_UNBLOCK(&ucp_ep->worker->async);
    return 1;

out_unblock:
    UCS_ASYNC_UNBLOCK(&ucp_ep->worker->async);
    return 0;
}

void ucp_wireup_update_flags(ucp_ep_h ucp_ep, uint32_t new_flags)
{
    ucp_lane_index_t lane;
    ucp_wireup_ep_t *wireup_ep;

    for (lane = 0; lane < ucp_ep_num_lanes(ucp_ep); ++lane) {
        wireup_ep = ucp_wireup_ep(ucp_ep_get_lane(ucp_ep, lane));
        if (wireup_ep == NULL) {
            continue;
        }

        ucs_trace("ep %p: wireup_ep=%p flags=0x%x new_flags=0x%x", ucp_ep,
                  wireup_ep, wireup_ep->flags, new_flags);
        wireup_ep->flags |= new_flags;
    }
}

void ucp_wireup_remote_connected(ucp_ep_h ep)
{
    if (ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED) {
        return;
    }

    ucs_trace("ep %p: remote connected, ep_cfg[%u]", ep, ep->cfg_index);
    if (!(ep->flags & UCP_EP_FLAG_CLOSED)) {
        /* set REMOTE_CONNECTED flag if an EP is not closed, otherwise -
         * just make UCT EPs remote connected to remove WIREUP_EP for them
         * and complete flush(LOCAL) operation in UCP EP close procedure
         * (don't set REMOTE_CONNECTED flag to avoid possible wrong behavior
         * in ucp_ep_close_flushed_callback() when a peer was already
         * disconnected, but we set REMOTE_CONNECTED flag again) */
        ucp_ep_update_flags(ep, UCP_EP_FLAG_REMOTE_CONNECTED, 0);
    }

    ucp_wireup_update_flags(ep,
                            UCP_WIREUP_EP_FLAG_REMOTE_CONNECTED |
                            UCP_WIREUP_EP_FLAG_READY);
    ucp_wireup_eps_progress_sched(ep);
    ucs_assert(ep->flags & UCP_EP_FLAG_REMOTE_ID);
}

static UCS_F_ALWAYS_INLINE unsigned
ucp_ep_err_mode_init_flags(ucp_err_handling_mode_t err_mode)
{
    return (err_mode == UCP_ERR_HANDLING_MODE_PEER) ?
           UCP_EP_INIT_ERR_MODE_PEER_FAILURE : 0;
}

static UCS_F_NOINLINE void
ucp_wireup_process_pre_request(ucp_worker_h worker, ucp_ep_h ep,
                               const ucp_wireup_msg_t *msg,
                               const ucp_unpacked_address_t *remote_address)
{
    unsigned ep_init_flags = UCP_EP_INIT_CREATE_AM_LANE |
                             UCP_EP_INIT_CM_WIREUP_CLIENT |
                             ucp_ep_err_mode_init_flags(msg->err_mode);
    unsigned addr_indices[UCP_MAX_LANES];
    ucs_status_t status;

    UCP_WIREUP_MSG_CHECK(msg, ep, UCP_WIREUP_MSG_PRE_REQUEST);
    ucs_trace("got wireup pre_request from 0x%"PRIx64" src_ep_id 0x%"PRIx64
              " dst_ep_id 0x%"PRIx64" conn_sn %u address version %u/%u",
              remote_address->uuid, msg->src_ep_id, msg->dst_ep_id,
              msg->conn_sn, remote_address->addr_version,
              remote_address->dst_version);

    ucs_assert(ucp_ep_get_cm_wireup_ep(ep) != NULL);
    ucs_assert(ep->flags & UCP_EP_FLAG_CONNECT_WAIT_PRE_REQ);

    status = ucp_ep_config_err_mode_check_mismatch(ep, msg->err_mode);
    if (status != UCS_OK) {
        goto err_ep_set_failed;
    }

    /* restore the EP here to avoid access to incomplete configuration before
       this point */
    ucp_ep_update_remote_id(ep, msg->src_ep_id);

    /* initialize transport endpoints */
    status = ucp_wireup_init_lanes(ep, ep_init_flags, &ucp_tl_bitmap_max,
                                   remote_address, addr_indices);
    if (status != UCS_OK) {
        goto err_ep_set_failed;
    }

    ucp_wireup_send_request(ep);
    return;

err_ep_set_failed:
    ucp_ep_set_failed_schedule(ep, UCP_NULL_LANE, status);
}

static UCS_F_NOINLINE void
ucp_wireup_process_request(ucp_worker_h worker, ucp_ep_h ep,
                           const ucp_wireup_msg_t *msg,
                           const ucp_unpacked_address_t *remote_address)
{
    uint64_t remote_uuid      = remote_address->uuid;
    int send_reply            = 0;
    unsigned ep_init_flags    = ucp_ep_err_mode_init_flags(msg->err_mode);
    ucp_tl_bitmap_t tl_bitmap = UCS_STATIC_BITMAP_ZERO_INITIALIZER;
    ucp_lane_index_t lanes2remote[UCP_MAX_LANES];
    unsigned addr_indices[UCP_MAX_LANES];
    ucs_status_t status;
    int has_cm_lane;

    UCP_WIREUP_MSG_CHECK(msg, ep, UCP_WIREUP_MSG_REQUEST);
    ucs_trace("got wireup request from 0x%"PRIx64" src_ep_id 0x%"PRIx64
              " dst_ep_id 0x%"PRIx64" conn_sn %d address version %u/%u",
              remote_address->uuid, msg->src_ep_id, msg->dst_ep_id,
              msg->conn_sn, remote_address->addr_version,
              remote_address->dst_version);

    if (ep != NULL) {
        ucs_assert(msg->dst_ep_id != UCS_PTR_MAP_KEY_INVALID);
        ucp_ep_update_remote_id(ep, msg->src_ep_id);
        ep_init_flags |= UCP_EP_INIT_CREATE_AM_LANE;
    } else {
        ucs_assert(msg->dst_ep_id == UCS_PTR_MAP_KEY_INVALID);
        ep = ucp_ep_match_retrieve(worker, remote_uuid,
                                   msg->conn_sn ^
                                   (remote_uuid == worker->uuid),
                                   UCS_CONN_MATCH_QUEUE_EXP);
        if (ep == NULL) {
            /* Create a new endpoint if does not exist */
            status = ucp_ep_create_base(worker, ep_init_flags,
                                        remote_address->name,
                                        "remote-request", &ep);
            if (status != UCS_OK) {
                return;
            }

            /* add internal endpoint to hash */
            ep->conn_sn = msg->conn_sn;
            if (!ucp_ep_match_insert(worker, ep, remote_uuid, ep->conn_sn,
                                     UCS_CONN_MATCH_QUEUE_UNEXP)) {
                if (worker->context->config.features & UCP_FEATURE_STREAM) {
                    ucs_diag("worker %p: created the endpoint %p without"
                             " connection matching, but Stream API support was"
                             " requested on the context %p",
                             worker, ep, worker->context);
                }
            }
        } else {
            status = ucp_ep_config_err_mode_check_mismatch(ep, msg->err_mode);
            if (status != UCS_OK) {
                goto err_set_ep_failed;
            }
        }

        ucp_ep_update_remote_id(ep, msg->src_ep_id);

        /*
         * If the current endpoint already sent a connection request, we have a
         * "simultaneous connect" situation. In this case, only one of the endpoints
         * (instead of both) should respect the connect request, otherwise they
         * will end up being connected to "internal" endpoints on the remote side
         * instead of each other. We use the uniqueness of worker uuid to decide
         * which connect request should be ignored.
         */
        if ((ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED) &&
            (remote_uuid > worker->uuid)) {
            ucs_trace("ep %p: ignoring simultaneous connect request", ep);
            ucp_ep_update_flags(ep, UCP_EP_FLAG_CONNECT_REQ_IGNORED, 0);
            return;
        }
    }

    has_cm_lane = ucp_ep_has_cm_lane(ep);
    if (has_cm_lane) {
        ep_init_flags |= UCP_EP_INIT_CM_WIREUP_SERVER;
    }

    /* Initialize lanes (possible destroy existing lanes) */
    status = ucp_wireup_init_lanes(ep, ep_init_flags, &ucp_tl_bitmap_max,
                                   remote_address, addr_indices);
    if (status != UCS_OK) {
        goto err_set_ep_failed;
    }

    ucp_wireup_match_p2p_lanes(ep, remote_address, addr_indices, lanes2remote);

    /* Send a reply if remote side does not have ep_ptr (active-active flow) or
     * there are p2p lanes (client-server flow)
     */
    send_reply = /* Always send the reply in case of CM, the client's EP has to
                  * be marked as REMOTE_CONNECTED */
        has_cm_lane || (msg->dst_ep_id == UCS_PTR_MAP_KEY_INVALID) ||
        ucp_ep_config(ep)->p2p_lanes;

    /* Connect p2p addresses to remote endpoint, if at least one is true: */
    if (/* - EP has not been connected locally yet */
        !(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED) ||
        /* - EP has CM lane (it is locally connected, since CM lanes are
         *   connected) */
        has_cm_lane) {
        status = ucp_wireup_connect_local(ep, remote_address, lanes2remote);
        if (status != UCS_OK) {
            goto err_set_ep_failed;
        }

        tl_bitmap  = ucp_wireup_get_ep_tl_bitmap(ep,
                                                 ucp_ep_config(ep)->p2p_lanes);
        ucp_ep_update_flags(ep, UCP_EP_FLAG_LOCAL_CONNECTED, 0);

        ucs_assert(send_reply);
    }

    /* don't mark as connected to remote now in case of CM, since it destroys
     * CM wireup EP (if it is hidden in the CM lane) that is used for sending
     * WIREUP MSGs */
    if (!ucp_ep_config(ep)->p2p_lanes && !has_cm_lane) {
        /* mark the endpoint as connected to remote */
        ucp_wireup_remote_connected(ep);
    }

    if (send_reply) {
        ucs_trace("ep %p: sending wireup reply", ep);
        ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REPLY, &tl_bitmap, lanes2remote);
    }

    return;

err_set_ep_failed:
    ucp_ep_set_failed_schedule(ep, UCP_NULL_LANE, status);
}

static unsigned ucp_wireup_send_msg_ack(void *arg)
{
    ucp_ep_h ep = (ucp_ep_h)arg;
    ucs_status_t status;

    /* Send ACK without any address, we've already sent it as part of the request */
    ucs_trace("ep %p: sending wireup ack", ep);

    status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_ACK, &ucp_tl_bitmap_min,
                                 NULL);
    return (status == UCS_OK);
}

int ucp_wireup_msg_ack_cb_pred(const ucs_callbackq_elem_t *elem, void *arg)
{
    return ((elem->arg == arg) && (elem->cb == ucp_wireup_send_msg_ack));
}

static UCS_F_NOINLINE void
ucp_wireup_process_reply(ucp_worker_h worker, ucp_ep_h ep,
                         const ucp_wireup_msg_t *msg,
                         const ucp_unpacked_address_t *remote_address)
{
    ucs_status_t status;
    int ack;

    UCP_WIREUP_MSG_CHECK(msg, ep, UCP_WIREUP_MSG_REPLY);
    ucs_trace("ep %p: got wireup reply src_ep_id 0x%"PRIx64
              " dst_ep_id 0x%"PRIx64" sn %d", ep, msg->src_ep_id,
              msg->dst_ep_id, msg->conn_sn);

    ucp_ep_match_remove_ep(worker, ep);
    ucp_ep_update_remote_id(ep, msg->src_ep_id);

    /* Connect p2p addresses to remote endpoint */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED) ||
        ucp_ep_has_cm_lane(ep)) {

        /*
         * In the wireup reply message, the lane indexes specify which
         * **receiver** ep lane should be connected to a given ep address. So we
         * don't pass 'lanes2remote' mapping, and use local lanes directly.
         */
        status = ucp_wireup_connect_local(ep, remote_address, NULL);
        if (status != UCS_OK) {
            ucp_ep_set_failed_schedule(ep, UCP_NULL_LANE, status);
            return;
        }

        ucp_ep_update_flags(ep, UCP_EP_FLAG_LOCAL_CONNECTED, 0);
        ack = 1;
    } else {
        ack = 0;
    }

    ucp_wireup_remote_connected(ep);

    if (ack) {
        /* Send `UCP_WIREUP_MSG_ACK` from progress function
         * to avoid calling UCT routines from an async thread */
        ucs_callbackq_add_oneshot(&worker->uct->progress_q, ep,
                                  ucp_wireup_send_msg_ack, ep);
    }
}

static void ucp_ep_removed_flush_completion(ucp_request_t *req)
{
    ucs_log_level_t level = UCS_STATUS_IS_ERR(req->status) ?
                                    UCS_LOG_LEVEL_DIAG :
                                    UCS_LOG_LEVEL_DEBUG;

    ucs_log(level, "flushing ep_removed (req %p) completed with status %s", req,
            ucs_status_string(req->status));
    ucp_ep_register_disconnect_progress(req);
}

static UCS_F_NOINLINE void
ucp_wireup_send_ep_removed(ucp_worker_h worker, const ucp_wireup_msg_t *msg,
                           const ucp_unpacked_address_t *remote_address)
{
    /* 1. Request a peer failure detection support from a reply EP to be able
     *    to do discarding of lanes when destroying all UCP EPs in UCP worker
     *    destroy.
     * 2. Create UCP EP with CONNECT_TO_IFACE connection mode to not do
     *    WIREUP_MSG phase between peers which require a direct EP ID.
     * 3. Create UCP EP with AM lane only, because WIREUP_MSGs are sent using
     *    AM lane.
     * 4. Allow selecting auxiliary transports for AM lane.
     */
    unsigned ep_init_flags = UCP_EP_INIT_ERR_MODE_PEER_FAILURE |
                             UCP_EP_INIT_FLAG_INTERNAL |
                             UCP_EP_INIT_CONNECT_TO_IFACE_ONLY |
                             UCP_EP_INIT_CREATE_AM_LANE |
                             UCP_EP_INIT_CREATE_AM_LANE_ONLY |
                             UCP_EP_INIT_ALLOW_AM_AUX_TL;
    ucs_status_t status;
    ucp_ep_h reply_ep;
    unsigned addr_indices[UCP_MAX_LANES];
    ucs_status_ptr_t req;

    /* If endpoint does not exist - create a temporary endpoint to send a
     * UCP_WIREUP_MSG_EP_REMOVED reply */
    status = ucp_ep_create_base(worker, ep_init_flags, remote_address->name,
                                "wireup ep_check reply", &reply_ep);
    if (status != UCS_OK) {
        ucs_error("failed to create EP: %s", ucs_status_string(status));
        return;
    }

    /* Initialize lanes of the reply EP */
    status = ucp_wireup_init_lanes(reply_ep, ep_init_flags, &ucp_tl_bitmap_max,
                                   remote_address, addr_indices);
    if (status != UCS_OK) {
        goto out_delete_ep;
    }

    ucp_ep_update_remote_id(reply_ep, msg->src_ep_id);
    status = ucp_wireup_msg_send(reply_ep, UCP_WIREUP_MSG_EP_REMOVED,
                                 &ucp_tl_bitmap_min, NULL);
    if (status != UCS_OK) {
        goto out_cleanup_lanes;
    }

    req = ucp_ep_flush_internal(reply_ep, UCP_REQUEST_FLAG_RELEASED,
                                &ucp_request_null_param, NULL,
                                ucp_ep_removed_flush_completion, "close");
    if (UCS_PTR_IS_PTR(req)) {
        return;
    }

out_cleanup_lanes:
    ucp_ep_cleanup_lanes(reply_ep);
out_delete_ep:
    ucp_ep_delete(reply_ep);
}

static UCS_F_NOINLINE
void ucp_wireup_process_ack(ucp_worker_h worker, ucp_ep_h ep,
                            const ucp_wireup_msg_t *msg)
{
    UCP_WIREUP_MSG_CHECK(msg, ep, UCP_WIREUP_MSG_ACK);
    ucs_trace("ep %p: got wireup ack", ep);

    ucs_assert(ep->flags & UCP_EP_FLAG_REMOTE_ID);
    ucs_assert(ep->flags & UCP_EP_FLAG_CONNECT_REP_SENT);

    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        /* drop the processing of ACK since close protocol or error
         * handling is started */
        ucs_assert(ucp_ep_has_cm_lane(ep) &&
                   (ep->flags & UCP_EP_FLAG_DISCONNECTED_CM_LANE));
        return;
    }

    ucp_wireup_remote_connected(ep);
}

static ucs_status_t ucp_wireup_msg_handler(void *arg, void *data,
                                           size_t length, unsigned flags)
{
    ucp_worker_h worker   = arg;
    ucp_wireup_msg_t *msg = data;
    ucp_ep_h ep           = NULL;
    ucp_unpacked_address_t remote_address;
    ucs_status_t status;

    UCS_ASYNC_BLOCK(&worker->async);

    if (msg->dst_ep_id != UCS_PTR_MAP_KEY_INVALID) {
        UCP_WORKER_GET_EP_BY_ID(
                &ep, worker, msg->dst_ep_id,
                if (msg->type != UCP_WIREUP_MSG_EP_CHECK) { goto out; },
                "WIREUP message (%d src_ep_id 0x%" PRIx64 " sn %d)", msg->type,
                msg->src_ep_id, msg->conn_sn);

        if ((msg->type == UCP_WIREUP_MSG_EP_CHECK) && (ep != NULL)) {
            /* UCP EP is valid, no need for any other actions when handling
             * EP_CHECK message (e.g. can avoid remote address unpacking) */
            goto out;
        }
    }

    status = ucp_address_unpack(worker, msg + 1,
                                UCP_ADDRESS_PACK_FLAGS_ALL,
                                &remote_address);
    if (status != UCS_OK) {
        ucs_error("failed to unpack address: %s", ucs_status_string(status));
        goto out;
    }

    if (msg->type == UCP_WIREUP_MSG_ACK) {
        ucs_assert(remote_address.address_count == 0);
        ucp_wireup_process_ack(worker, ep, msg);
    } else if (msg->type == UCP_WIREUP_MSG_PRE_REQUEST) {
        ucp_wireup_process_pre_request(worker, ep, msg, &remote_address);
    } else if (msg->type == UCP_WIREUP_MSG_REQUEST) {
        ucp_wireup_process_request(worker, ep, msg, &remote_address);
    } else if (msg->type == UCP_WIREUP_MSG_REPLY) {
        ucp_wireup_process_reply(worker, ep, msg, &remote_address);
    } else if (msg->type == UCP_WIREUP_MSG_EP_CHECK) {
        ucs_assert((msg->dst_ep_id != UCS_PTR_MAP_KEY_INVALID) && (ep == NULL));
        ucp_wireup_send_ep_removed(worker, msg, &remote_address);
    } else if (msg->type == UCP_WIREUP_MSG_EP_REMOVED) {
        ucs_assert(msg->dst_ep_id != UCS_PTR_MAP_KEY_INVALID);
        ucp_ep_set_failed_schedule(ep, UCP_NULL_LANE, UCS_ERR_CONNECTION_RESET);
    } else {
        ucs_bug("invalid wireup message");
    }

    ucs_free(remote_address.address_list);

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return UCS_OK;
}

uct_ep_h ucp_wireup_extract_lane(ucp_ep_h ep, ucp_lane_index_t lane)
{
    uct_ep_h uct_ep = ucp_ep_get_lane(ep, lane);

    if ((uct_ep != NULL) && ucp_wireup_ep_test(uct_ep)) {
        return ucp_wireup_ep_extract_next_ep(uct_ep);
    } else {
        ucp_ep_set_lane(ep, lane, NULL);
        return uct_ep;
    }
}

static void
ucp_wireup_replay_pending_request(uct_pending_req_t *self, ucp_ep_h ucp_ep)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    ucs_assert(req->send.ep == ucp_ep);

    if ((req->flags & UCP_REQUEST_FLAG_PROTO_SEND) &&
        ((ucp_ep->cfg_index != req->send.proto_config->ep_cfg_index) ||
         ucp_ep->worker->context->config.ext.proto_request_reset)) {
        ucp_trace_req(req, "replay proto %s",
                      req->send.proto_config->proto->name);
        ucp_proto_request_restart(req);
    } else {
        ucp_trace_req(req, "replay proto %s lane %d",
                      (req->flags & UCP_REQUEST_FLAG_PROTO_SEND) ?
                      req->send.proto_config->proto->name : "N/A",
                      req->send.lane);
        ucp_request_send(req);
    }
}

void ucp_wireup_replay_pending_requests(ucp_ep_h ucp_ep,
                                        ucs_queue_head_t *tmp_pending_queue)
{
    uct_pending_req_t *uct_req;

    ucp_ep->flags |= UCP_EP_FLAG_BLOCK_FLUSH;
    /* Replay pending requests */
    ucs_queue_for_each_extract(uct_req, tmp_pending_queue, priv, 1) {
        ucp_wireup_replay_pending_request(uct_req, ucp_ep);
    }
    ucp_ep->flags &= ~UCP_EP_FLAG_BLOCK_FLUSH;
}

static void
ucp_wireup_ep_lane_set_next_ep(ucp_ep_h ep, ucp_lane_index_t lane,
                               uct_ep_h uct_ep)
{
    ucs_trace("ep %p: wireup uct_ep[%d]=%p next set to %p", ep, lane,
              ucp_ep_get_lane(ep, lane), uct_ep);
    ucp_wireup_ep_set_next_ep(ucp_ep_get_lane(ep, lane), uct_ep,
                              ucp_ep_get_rsc_index(ep, lane));
}

static int ucp_wireup_should_activate_wiface(ucp_worker_iface_t *wiface,
                                             ucp_ep_h ep, ucp_lane_index_t lane)
{
    ucp_context_h context = wiface->worker->context;

    /* Activate worker iface if: a) new protocol selection logic is disabled; or
     * b) stream support is requested, because stream API does not support new
     * protocol selection logic; or c) lane is used for checking a connection
     * state; or d) the endpoint is a mem-type ep. */
    return !context->config.ext.proto_enable ||
           (context->config.features & UCP_FEATURE_STREAM) ||
           (ucp_ep_config(ep)->key.keepalive_lane == lane) ||
           (ep->flags & UCP_EP_FLAG_INTERNAL);
}

static ucs_status_t
ucp_wireup_connect_lane_to_iface(ucp_ep_h ep, ucp_lane_index_t lane,
                                 unsigned path_index,
                                 ucp_worker_iface_t *wiface,
                                 const ucp_address_entry_t *address)
{
    uct_ep_h uct_ep = ucp_ep_get_lane(ep, lane);
    uct_ep_params_t uct_ep_params;
    ucs_status_t status;
    uct_ep_h wireup_ep;

    ucs_assert(wiface->attr.cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE);
    ucs_assertv_always((uct_ep == NULL) || ucp_wireup_ep_test(uct_ep),
                       "ep %p: lane %u (uct_ep=%p is_wireup=%d) exists", ep,
                       lane, uct_ep, ucp_wireup_ep_test(uct_ep));

    /* create an endpoint connected to the remote interface */
    ucs_trace("ep %p: connect uct_ep[%d] to addr %p", ep, lane,
              address);
    uct_ep_params.field_mask = UCT_EP_PARAM_FIELD_IFACE      |
                               UCT_EP_PARAM_FIELD_DEV_ADDR   |
                               UCT_EP_PARAM_FIELD_IFACE_ADDR |
                               UCT_EP_PARAM_FIELD_PATH_INDEX;
    uct_ep_params.iface      = wiface->iface;
    uct_ep_params.dev_addr   = address->dev_addr;
    uct_ep_params.iface_addr = address->iface_addr;
    uct_ep_params.path_index = path_index;
    status = uct_ep_create(&uct_ep_params, &uct_ep);
    if (status != UCS_OK) {
        /* coverity[leaked_storage] */
        return status;
    }

    if (ucp_ep_get_lane(ep, lane) == NULL) {
        if (/* Create wireup EP in case of CM lane is used, since a WIREUP EP is
             * used to keep user's pending requests and send WIREUP MSGs (if it
             * is WIREUP MSG lane) until CM and WIREUP_MSG phases are done. The
             * lane is added during WIREUP_MSG exchange or created as an initial
             * configuration after a connection request on a server side */
            ucp_ep_has_cm_lane(ep) ||
            /* Create wireup EP in case of presence p2p lanes and forced proto
             * restart since pending flush can be completed out of order on
             * lanes directly connected to iface without wireup EP */
            (ucp_ep_config(ep)->p2p_lanes &&
             ep->worker->context->config.ext.proto_request_reset)) {
            status = ucp_wireup_ep_create(ep, &wireup_ep);
            if (status != UCS_OK) {
                /* coverity[leaked_storage] */
                return status;
            }

            ucp_ep_set_lane(ep, lane, wireup_ep);
            ucp_wireup_ep_lane_set_next_ep(ep, lane, uct_ep);
        } else {
            /* Assign the lane without wireup EP when out-of-band address
             * exchange is used */
            ucs_trace("ep %p: assign uct_ep[%d]=%p", ep, lane, uct_ep);
            ucp_ep_set_lane(ep, lane, uct_ep);
        }
    } else {
        /* If EP already exists, it's a wireup proxy, and we need to update
         * its next_ep instead of replacing it. The wireup EP was created
         * during CM pack_cb() on a client side */
        ucs_assert(ucp_wireup_ep_test(ucp_ep_get_lane(ep, lane)));
        ucs_assert(ucp_proxy_ep_extract(ucp_ep_get_lane(ep, lane)) == NULL);
        ucs_assert(ucp_ep_has_cm_lane(ep));
        ucp_wireup_ep_lane_set_next_ep(ep, lane, uct_ep);
    }

    if (ucp_wireup_should_activate_wiface(wiface, ep, lane)) {
        ucp_worker_iface_progress_ep(wiface);
    }

    return UCS_OK;
}

static ucs_status_t
ucp_wireup_connect_lane_to_ep(ucp_ep_h ep, unsigned ep_init_flags,
                              ucp_lane_index_t lane, unsigned path_index,
                              ucp_rsc_index_t rsc_index,
                              ucp_worker_iface_t *wiface,
                              const ucp_unpacked_address_t *remote_address)
{
    int connect_aux;
    uct_ep_h uct_ep;
    ucs_status_t status;

    if (ucp_ep_get_lane(ep, lane) == NULL) {
        status = ucp_wireup_ep_create(ep, &uct_ep);
        if (status != UCS_OK) {
            /* coverity[leaked_storage] */
            return status;
        }

        ucs_trace("ep %p: assign uct_ep[%d]=%p wireup", ep, lane, uct_ep);
        ucp_ep_set_lane(ep, lane, uct_ep);
    } else {
        uct_ep = ucp_ep_get_lane(ep, lane);
        ucs_assert(ucp_wireup_ep_test(uct_ep));
    }

    ucs_trace("ep %p: connect uct_ep[%d]=%p to remote addr %p wireup", ep,
              lane, uct_ep, remote_address);
    connect_aux = !ucp_ep_init_flags_has_cm(ep_init_flags) &&
                  (lane == ucp_ep_get_wireup_msg_lane(ep));
    status      = ucp_wireup_ep_connect(ucp_ep_get_lane(ep, lane), ep_init_flags,
                                        rsc_index, path_index, connect_aux,
                                        remote_address);
    if (status != UCS_OK) {
        return status;
    }

    if (ucp_wireup_should_activate_wiface(wiface, ep, lane)) {
        ucp_worker_iface_progress_ep(wiface);
    }

    return UCS_OK;
}

static ucs_status_t
ucp_wireup_connect_lane(ucp_ep_h ep, unsigned ep_init_flags,
                        ucp_lane_index_t lane, unsigned path_index,
                        const ucp_unpacked_address_t *remote_address,
                        unsigned addr_index)
{
    ucp_worker_h worker = ep->worker;
    ucp_rsc_index_t rsc_index;
    ucp_worker_iface_t *wiface;
    ucp_address_entry_t *address;

    ucs_trace("ep %p: connect lane[%d]", ep, lane);

    ucs_assert(lane != ucp_ep_get_cm_lane(ep));

    ucs_assert_always(remote_address != NULL);
    ucs_assert_always(remote_address->address_list != NULL);
    ucs_assert_always(addr_index <= remote_address->address_count);

    rsc_index  = ucp_ep_get_rsc_index(ep, lane);
    wiface     = ucp_worker_iface(worker, rsc_index);

    /*
     * create a wireup endpoint which will start connection establishment
     * protocol using an auxiliary transport.
     */
    if (ucp_ep_is_lane_p2p(ep, lane)) {
        return ucp_wireup_connect_lane_to_ep(ep, ep_init_flags, lane,
                                             path_index, rsc_index, wiface,
                                             remote_address);
    } else if (ucp_worker_is_tl_2iface(worker, rsc_index)) {
        address = &remote_address->address_list[addr_index];
        return ucp_wireup_connect_lane_to_iface(ep, lane, path_index, wiface,
                                                address);
    } else {
        return UCS_ERR_UNREACHABLE;
    }
}

static const char *
ucp_wireup_get_lane_index_str(ucp_lane_index_t lane, char *buf, size_t max)
{
    if (lane != UCP_NULL_LANE) {
        ucs_snprintf_safe(buf, max, "%d", lane);
    } else {
        ucs_strncpy_safe(buf, "<none>", max);
    }

    return buf;
}

static void ucp_wireup_print_config(ucp_worker_h worker,
                                    const ucp_ep_config_key_t *key,
                                    const char *title,
                                    const unsigned *addr_indices,
                                    ucp_rsc_index_t cm_index,
                                    ucs_log_level_t log_level)
{
    char am_lane_str[8];
    char wireup_msg_lane_str[8];
    char cm_lane_str[8];
    char keepalive_lane_str[8];
    ucp_lane_index_t lane;

    if (!ucs_log_is_enabled(log_level)) {
        return;
    }

    ucs_log(log_level,
            "%s: am_lane %s wireup_msg_lane %s cm_lane %s keepalive_lane %s"
            " reachable_mds 0x%" PRIx64,
            title,
            ucp_wireup_get_lane_index_str(key->am_lane, am_lane_str,
                                          sizeof(am_lane_str)),
            ucp_wireup_get_lane_index_str(key->wireup_msg_lane,
                                          wireup_msg_lane_str,
                                          sizeof(wireup_msg_lane_str)),
            ucp_wireup_get_lane_index_str(key->cm_lane, cm_lane_str,
                                          sizeof(cm_lane_str)),
            ucp_wireup_get_lane_index_str(key->keepalive_lane, keepalive_lane_str,
                                          sizeof(keepalive_lane_str)),
            key->reachable_md_map);

    for (lane = 0; lane < key->num_lanes; ++lane) {
        UCS_STRING_BUFFER_ONSTACK(strb, 128);
        if (lane == key->cm_lane) {
            ucp_ep_config_cm_lane_info_str(worker, key, lane, cm_index, &strb);
        } else {
            ucp_ep_config_lane_info_str(worker, key, addr_indices, lane,
                                        UCP_NULL_RESOURCE, &strb);
        }
        ucs_log(log_level, "%s: %s", title, ucs_string_buffer_cstr(&strb));
    }

    ucs_log(log_level, "%s: err mode %d, flags 0x%x", title, key->err_mode,
            key->flags);
}

int ucp_wireup_is_reachable(ucp_ep_h ep, unsigned ep_init_flags,
                            ucp_rsc_index_t rsc_index,
                            const ucp_address_entry_t *ae,
                            char *info_str, size_t info_str_size)
{
    ucp_context_h context      = ep->worker->context;
    ucp_worker_iface_t *wiface = ucp_worker_iface(ep->worker, rsc_index);
    uct_iface_is_reachable_params_t params = {
        .field_mask  = UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR |
                       UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR,
        .device_addr = ae->dev_addr,
        .iface_addr  = ae->iface_addr,
    };

    if (info_str != NULL) {
        params.field_mask        |=
                                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING |
                                UCT_IFACE_IS_REACHABLE_FIELD_INFO_STRING_LENGTH;
        params.info_string        = info_str;
        params.info_string_length = info_str_size;
    }

    if (context->tl_rscs[rsc_index].tl_name_csum != ae->tl_name_csum) {
        return 0;
    }

    /* assume reachability is checked by CM, if EP selects lanes
     * during CM phase */
    return (ep_init_flags & UCP_EP_INIT_CM_PHASE) ||
           uct_iface_is_reachable_v2(wiface->iface, &params);
}

static void
ucp_wireup_get_reachable_mds(ucp_ep_h ep, unsigned ep_init_flags,
                             const ucp_unpacked_address_t *remote_address,
                             ucp_ep_config_key_t *key)
{
    ucp_context_h context = ep->worker->context;
    const ucp_ep_config_key_t *prev_config_key;
    ucp_rsc_index_t ae_cmpts[UCP_MAX_MDS]; /* component index for each address entry */
    const ucp_address_entry_t *ae;
    ucp_rsc_index_t cmpt_index;
    ucp_rsc_index_t rsc_index;
    ucp_md_index_t dst_md_index;
    ucp_md_map_t ae_dst_md_map, dst_md_map;
    ucp_md_map_t prev_dst_md_map;
    unsigned num_dst_mds;

    ae_dst_md_map = 0;
    UCS_STATIC_BITMAP_FOR_EACH_BIT(rsc_index, &context->tl_bitmap) {
        ucp_unpacked_address_for_each(ae, remote_address) {
            if (ucp_wireup_is_reachable(ep, ep_init_flags, rsc_index, ae, NULL, 0)) {
                ae_dst_md_map         |= UCS_BIT(ae->md_index);
                dst_md_index           = context->tl_rscs[rsc_index].md_index;
                ae_cmpts[ae->md_index] = context->tl_mds[dst_md_index].cmpt_index;
            }
        }
    }

    if (ep->cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
        prev_config_key = NULL;
        prev_dst_md_map = 0;
    } else {
        prev_config_key = &ucp_ep_config(ep)->key;
        prev_dst_md_map = prev_config_key->reachable_md_map;
    }

    /* merge with previous configuration */
    dst_md_map  = ae_dst_md_map | prev_dst_md_map;
    num_dst_mds = 0;
    ucs_for_each_bit(dst_md_index, dst_md_map) {
        cmpt_index = UCP_NULL_RESOURCE;
        /* remote md is reachable by the provided address */
        if (UCS_BIT(dst_md_index) & ae_dst_md_map) {
            cmpt_index = ae_cmpts[dst_md_index];
        }
        /* remote md is reachable by previous ep configuration */
        if (UCS_BIT(dst_md_index) & prev_dst_md_map) {
            ucs_assert(prev_config_key != NULL);
            cmpt_index = ucp_ep_config_get_dst_md_cmpt(prev_config_key, dst_md_index);
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

static void ucp_wireup_discard_uct_eps(ucp_ep_h ep, uct_ep_h *uct_eps,
                                       ucs_queue_head_t *replay_pending_queue)
{
    ucp_lane_index_t index;

    for (index = 0; index < UCP_MAX_LANES; ++index) {
        if (uct_eps[index] == NULL) {
            continue;
        }

        ucp_worker_discard_uct_ep(ep, uct_eps[index], UCP_NULL_RESOURCE,
                                  UCT_FLUSH_FLAG_LOCAL,
                                  ucp_request_purge_enqueue_cb,
                                  replay_pending_queue,
                                  (ucp_send_nbx_callback_t)ucs_empty_function,
                                  NULL);
    }
}

static int
ucp_wireup_are_all_lanes_p2p(ucp_ep_h ep, const ucp_ep_config_key_t *key)
{
    ucp_lane_index_t lane;
    ucp_rsc_index_t rsc_index;

    for (lane = 0; lane < key->num_lanes; ++lane) {
        rsc_index = ucp_ep_get_rsc_index(ep, lane);
        ucs_assert(rsc_index != UCP_NULL_RESOURCE);

        if (ucp_worker_is_tl_2iface(ep->worker, rsc_index)) {
            return 0;
        }
    }

    return 1;
}

static unsigned
ucp_ep_num_reused_lanes(ucp_ep_h ep, const ucp_lane_index_t *reuse_lane_map)
{
    unsigned num_reused = 0;
    ucp_lane_index_t lane;

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        num_reused += (reuse_lane_map[lane] != UCP_NULL_LANE);
    }

    return num_reused;
}

static int
ucp_wireup_check_is_reconfigurable(ucp_ep_h ep,
                                   const ucp_ep_config_key_t *new_key,
                                   const ucp_unpacked_address_t *remote_address,
                                   const unsigned *addr_indices)
{
    ucp_lane_index_t reuse_lane_map[UCP_MAX_LANES];
    const ucp_ep_config_key_t *old_key;

    if (ucp_ep_has_cm_lane(ep)) {
        return 1;
    }

    old_key = &ucp_ep_config(ep)->key;

    /* Verify both old/new configurations have only p2p lanes */
    if (!ucp_wireup_are_all_lanes_p2p(ep, old_key) ||
        !ucp_wireup_are_all_lanes_p2p(ep, new_key) ||
        (old_key->num_lanes != new_key->num_lanes)) {
        return 0;
    }

    ucp_ep_config_lanes_intersect(old_key, new_key, ep, remote_address,
                                  addr_indices, reuse_lane_map);

    /* For now, reconfig is supported only if no lanes are reused */
    return ucp_ep_num_reused_lanes(ep, reuse_lane_map) == 0;
}

static ucp_lane_index_t
ucp_wireup_find_non_reused_lane(ucp_ep_h ep, const ucp_ep_config_key_t *key,
                                const ucp_lane_index_t *reuse_lane_map)
{
    if (ucp_ep_has_cm_lane(ep)) {
        return key->cm_lane;
    }

    /* Just use first lane, as only non-reused lanes are allowed at the
     * moment. */
    ucs_assert(key->num_lanes > 0);
    ucs_assert(ucp_ep_num_reused_lanes(ep, reuse_lane_map) == 0);
    return 0;
}

static ucs_status_t
ucp_wireup_replace_wireup_msg_lane(ucp_ep_h ep, ucp_ep_config_key_t *key,
                                   uct_ep_h *new_uct_eps,
                                   const ucp_lane_index_t *reuse_lane_map)
{
    uct_ep_h uct_ep = NULL;
    ucp_lane_index_t old_lane, new_wireup_lane;
    ucp_wireup_ep_t *old_wireup_ep, *new_wireup_ep;
    ucp_rsc_index_t aux_rsc_index;
    int is_p2p;
    ucs_status_t status;

    /* Get old wireup lane */
    old_lane      = ucp_wireup_get_msg_lane(ep, UCP_WIREUP_MSG_REQUEST);
    old_wireup_ep = ucp_wireup_ep(ucp_ep_get_lane(ep, old_lane));
    ucs_assert_always(old_wireup_ep != NULL);

    /* Set wireup EP for new configuration's wireup lane */
    if (ucp_ep_has_cm_lane(ep)) {
        /* Use existing EP from CM lane */
        new_wireup_ep = ucp_ep_get_cm_wireup_ep(ep);
        ucs_assert(new_wireup_ep != NULL);
        aux_rsc_index = ucp_ep_get_rsc_index(ep, old_lane);
    } else {
        /* Create new EP for non-CM flow */
        status = ucp_wireup_ep_create(ep, &uct_ep);
        if (status != UCS_OK) {
            return status;
        }

        new_wireup_ep = ucp_wireup_ep(uct_ep);
        aux_rsc_index = ucp_wireup_ep_get_aux_rsc_index(
                &old_wireup_ep->super.super);
    }

    ucs_assert(aux_rsc_index != UCP_NULL_RESOURCE);
    is_p2p = ucp_ep_config_connect_p2p(ep->worker, &ucp_ep_config(ep)->key,
                                       aux_rsc_index);

    /* Move aux EP to new wireup lane */
    ucp_wireup_ep_set_aux(new_wireup_ep,
                          ucp_wireup_ep_extract_msg_ep(old_wireup_ep),
                          aux_rsc_index, is_p2p);

    /* Remove old wireup_ep as it's not needed anymore.
     * NOTICE: Next two lines are intentionally not merged with the lane
     * removal loop in ucp_wireup_check_config_intersect, because of future
     * support for non-wireup EPs reconfiguration (which will modify this
     * code). */
    uct_ep_destroy(&old_wireup_ep->super.super);
    ucp_ep_set_lane(ep, old_lane, NULL);

    /* Select CM/non-reused lane as new wireup lane */
    new_wireup_lane = ucp_wireup_find_non_reused_lane(ep, key, reuse_lane_map);

    new_uct_eps[new_wireup_lane] = &new_wireup_ep->super.super;
    key->wireup_msg_lane         = new_wireup_lane;
    return UCS_OK;
}

static ucs_status_t
ucp_wireup_check_config_intersect(ucp_ep_h ep, ucp_ep_config_key_t *new_key,
                                  const ucp_unpacked_address_t *remote_address,
                                  const unsigned *addr_indices,
                                  ucp_lane_map_t *connect_lane_bitmap,
                                  ucs_queue_head_t *replay_pending_queue)
{
    uct_ep_h new_uct_eps[UCP_MAX_LANES]            = {NULL};
    ucp_lane_index_t reuse_lane_map[UCP_MAX_LANES] = {UCP_NULL_LANE};
    ucp_ep_config_key_t *old_key;
    ucp_lane_index_t lane, reuse_lane;
    uct_ep_h uct_ep;
    ucs_status_t status;

    *connect_lane_bitmap = UCS_MASK(new_key->num_lanes);

    if ((ep->cfg_index == UCP_WORKER_CFG_INDEX_NULL) ||
        !ucp_wireup_check_is_reconfigurable(ep, new_key, remote_address,
                                            addr_indices)) {
        /* nothing to intersect with */
        return ucp_ep_realloc_lanes(ep, new_key->num_lanes);
    }

    ucs_assert(!(ep->flags & UCP_EP_FLAG_INTERNAL));

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        ucs_assert(ucp_ep_get_lane(ep, lane) != NULL);
    }

    old_key = &ucp_ep_config(ep)->key;
    ucp_ep_config_lanes_intersect(old_key, new_key, ep, remote_address,
                                  addr_indices, reuse_lane_map);

    if (ucp_ep_has_cm_lane(ep)) {
        /* CM lane has to be reused by the new EP configuration */
        ucs_assert(reuse_lane_map[ucp_ep_get_cm_lane(ep)] != UCP_NULL_LANE);
        /* wireup lane hasn't been selected by the new configuration: only this
         * function should select it */
        ucs_assertv(new_key->wireup_msg_lane == UCP_NULL_LANE,
                    "new_key->wireup_msg_lane=%u", new_key->wireup_msg_lane);
    }

    /* wireup lane has to be selected for the old configuration */
    ucs_assert(old_key->wireup_msg_lane != UCP_NULL_LANE);

    /* set the correct WIREUP MSG lane */
    reuse_lane = reuse_lane_map[old_key->wireup_msg_lane];
    if (reuse_lane != UCP_NULL_LANE) {
        /* previous wireup lane is part of the new configuration, so reuse it */
        new_key->wireup_msg_lane = reuse_lane;
    } else /* old wireup lane won't be reused */ {
        /* previous wireup lane is not part of new configuration, so add it as
         * auxiliary endpoint inside CM/non-reused lane, to be able to
         * continue wireup messages exchange */
        status = ucp_wireup_replace_wireup_msg_lane(ep, new_key, new_uct_eps,
                                                    reuse_lane_map);
        if (status != UCS_OK) {
            return status;
        }
    }

    /* Need to discard only old lanes that won't be used anymore in the new
     * configuration. Also, UCT EPs with the lane index >= old_key->num_lanes
     * could be set in case of CM, we have to not reset them */
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        reuse_lane = reuse_lane_map[lane];
        uct_ep     = ucp_ep_get_lane(ep, lane);
        if (reuse_lane == UCP_NULL_LANE) {
            if (uct_ep != NULL) {
                ucs_assert(lane != ucp_ep_get_cm_lane(ep));
                ucp_worker_discard_uct_ep(
                        ep, uct_ep, UCP_NULL_RESOURCE, UCT_FLUSH_FLAG_LOCAL,
                        (uct_pending_purge_callback_t)
                                ucs_empty_function_do_assert_void,
                        NULL, (ucp_send_nbx_callback_t)ucs_empty_function,
                        NULL);
                ucp_ep_set_lane(ep, lane, NULL);
            }
        } else if (uct_ep != NULL) {
            if (!ucp_wireup_ep_test(uct_ep) ||
                (ucp_wireup_ep(uct_ep)->super.uct_ep != NULL)) {
                /* no need to connect lane */
                *connect_lane_bitmap &= ~UCS_BIT(reuse_lane);
            }
            new_uct_eps[reuse_lane] = uct_ep;
            ucp_ep_set_lane(ep, lane, NULL);
        }

        ucs_assert(ucp_ep_get_lane(ep, lane) == NULL);
    }

    status = ucp_ep_realloc_lanes(ep, new_key->num_lanes);
    if (status != UCS_OK) {
        ucp_wireup_discard_uct_eps(ep, new_uct_eps, replay_pending_queue);
        return status;
    }

    for (lane = 0; lane < new_key->num_lanes; ++lane) {
        ucp_ep_set_lane(ep, lane, new_uct_eps[lane]);
    }

    return UCS_OK;
}

static ucs_status_t
ucp_wireup_try_select_lanes(ucp_ep_h ep, unsigned ep_init_flags,
                            const ucp_tl_bitmap_t *tl_bitmap,
                            const ucp_unpacked_address_t *remote_address,
                            unsigned *addr_indices, ucp_ep_config_key_t *key,
                            ucp_rsc_index_t *dst_md_storage)
{
    ucs_status_t status;

    ucp_ep_config_key_reset(key);
    ucp_ep_config_key_set_err_mode(key, ep_init_flags);
    ucp_ep_config_key_init_flags(key, ep_init_flags);
    key->dst_version  = remote_address->dst_version;
    key->dst_md_cmpts = dst_md_storage;

    status = ucp_wireup_select_lanes(ep, ep_init_flags, *tl_bitmap,
                                     remote_address, addr_indices, key, 1);
    if (status != UCS_OK) {
        return status;
    }

    /* Get all reachable MDs from full remote address list and join with
     * current ep configuration
     */
    ucp_wireup_get_reachable_mds(ep, ep_init_flags, remote_address, key);
    return UCS_OK;
}

ucs_status_t ucp_wireup_init_lanes(ucp_ep_h ep, unsigned ep_init_flags,
                                   const ucp_tl_bitmap_t *local_tl_bitmap,
                                   const ucp_unpacked_address_t *remote_address,
                                   unsigned *addr_indices)
{
    ucp_worker_h worker    = ep->worker;
    ucp_rsc_index_t cm_idx = UCP_NULL_RESOURCE;
    ucp_tl_bitmap_t tl_bitmap, current_tl_bitmap;
    ucp_rsc_index_t rsc_idx;
    ucp_lane_map_t connect_lane_bitmap;
    ucp_ep_config_key_t key;
    ucp_worker_cfg_index_t new_cfg_index;
    ucp_lane_index_t lane;
    ucs_status_t status;
    char str[32];
    ucs_queue_head_t replay_pending_queue;
    ucp_rsc_index_t dst_mds_mem[UCP_MAX_MDS];

    tl_bitmap = UCS_STATIC_BITMAP_AND(*local_tl_bitmap,
                                      worker->context->tl_bitmap);
    ucs_assert(!UCS_STATIC_BITMAP_IS_ZERO(tl_bitmap));

    ucs_trace("ep %p: initialize lanes", ep);
    ucs_log_indent(1);
    ucp_wireup_eps_pending_extract(ep, &replay_pending_queue);

    status = ucp_wireup_try_select_lanes(ep, ep_init_flags, &tl_bitmap,
                                         remote_address, addr_indices, &key,
                                         dst_mds_mem);
    if (status != UCS_OK) {
        goto out;
    }

    if ((ep->cfg_index != UCP_WORKER_CFG_INDEX_NULL) &&
        !ucp_ep_config_is_equal(&ucp_ep_config(ep)->key, &key) &&
        !ucp_wireup_check_is_reconfigurable(ep, &key, remote_address,
                                            addr_indices)) {
        /* Allow to choose only the lanes that were already chosen for case
         * without CM to prevent reconfiguration error.
         */
        UCS_STATIC_BITMAP_RESET_ALL(&current_tl_bitmap);
        for (lane = 0; lane < ucp_ep_config(ep)->key.num_lanes; ++lane) {
            rsc_idx = ucp_ep_config(ep)->key.lanes[lane].rsc_index;
            UCS_STATIC_BITMAP_SET(&current_tl_bitmap, rsc_idx);
            ucs_assertv(UCS_STATIC_BITMAP_GET(tl_bitmap, rsc_idx),
                        "resource that was chosen previously is unavailable: "
                        "tl_rscs[%d]=" UCT_TL_RESOURCE_DESC_FMT,
                        rsc_idx,
                        UCT_TL_RESOURCE_DESC_ARG(
                                &worker->context->tl_rscs[rsc_idx].tl_rsc));
        }
        UCS_STATIC_BITMAP_AND_INPLACE(&tl_bitmap, current_tl_bitmap);

        status = ucp_wireup_try_select_lanes(ep, ep_init_flags, &tl_bitmap,
                                             remote_address, addr_indices, &key,
                                             dst_mds_mem);
        if (status != UCS_OK) {
            goto out;
        }
    }

    status = ucp_wireup_check_config_intersect(ep, &key, remote_address,
                                               addr_indices,
                                               &connect_lane_bitmap,
                                               &replay_pending_queue);
    if (status != UCS_OK) {
        goto out;
    }

    /* Load new configuration */
    status = ucp_worker_get_ep_config(worker, &key, ep_init_flags,
                                      &new_cfg_index);
    if (status != UCS_OK) {
        goto out;
    }

    if (ep->cfg_index == new_cfg_index) {
#if UCS_ENABLE_ASSERT
        for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
            ucs_assert(ucp_ep_get_lane(ep, lane) != NULL);
        }
#endif
        status = UCS_OK; /* No change */
        goto out;
    }

    cm_idx = ep->ext->cm_idx;

    if ((ep->cfg_index != UCP_WORKER_CFG_INDEX_NULL) &&
        !ucp_wireup_check_is_reconfigurable(ep, &key, remote_address,
                                            addr_indices)) {
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
        ucp_wireup_print_config(worker, &ucp_ep_config(ep)->key, "old",
                                NULL, cm_idx, UCS_LOG_LEVEL_ERROR);
        ucp_wireup_print_config(worker, &key, "new", NULL,
                                cm_idx, UCS_LOG_LEVEL_ERROR);
        ucs_fatal("endpoint reconfiguration not supported yet");
    }

    ucp_ep_set_cfg_index(ep, new_cfg_index);
    ep->am_lane = key.am_lane;

    snprintf(str, sizeof(str), "ep %p", ep);
    ucp_wireup_print_config(worker, &ucp_ep_config(ep)->key, str,
                            addr_indices, cm_idx, UCS_LOG_LEVEL_DEBUG);

    /* establish connections on all underlying endpoints */
    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (ucp_ep_get_cm_lane(ep) == lane) {
            continue;
        }

        if (connect_lane_bitmap & UCS_BIT(lane)) {
            status = ucp_wireup_connect_lane(ep, ep_init_flags, lane,
                                             key.lanes[lane].path_index,
                                             remote_address, addr_indices[lane]);
            if (status != UCS_OK) {
                goto out;
            }
        }

        ucs_assert(ucp_ep_get_lane(ep, lane) != NULL);
    }

    /* If we don't have a p2p transport, we're connected */
    if (!ucp_ep_config(ep)->p2p_lanes) {
        ucp_ep_update_flags(ep, UCP_EP_FLAG_LOCAL_CONNECTED, 0);
    }

    ucp_worker_keepalive_add_ep(ep);
    status = UCS_OK;

out:
    ucp_wireup_replay_pending_requests(ep, &replay_pending_queue);
    ucs_log_indent(-1);
    return status;
}

ucs_status_t ucp_wireup_send_request(ucp_ep_h ep)
{
    ucp_rsc_index_t rsc_index;
    ucs_status_t status;
    ucp_tl_bitmap_t tl_bitmap;

    tl_bitmap = ucp_wireup_get_ep_tl_bitmap(ep, UCS_MASK(ucp_ep_num_lanes(ep)));

    /* TODO make sure such lane would exist */
    rsc_index = ucp_wireup_ep_get_aux_rsc_index(
            ucp_ep_get_lane(ep, ucp_ep_get_wireup_msg_lane(ep)));
    if (rsc_index != UCP_NULL_RESOURCE) {
        UCS_STATIC_BITMAP_SET(&tl_bitmap, rsc_index);
    }

    ucs_debug("ep %p: send wireup request (flags=0x%x)", ep, ep->flags);
    status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_REQUEST, &tl_bitmap, NULL);

    ucp_ep_update_flags(ep, UCP_EP_FLAG_CONNECT_REQ_QUEUED, 0);

    return status;
}

ucs_status_t ucp_wireup_send_pre_request(ucp_ep_h ep)
{
    ucs_status_t status;

    ucs_assert(ucp_ep_has_cm_lane(ep));
    ucs_assert(!(ep->flags & UCP_EP_FLAG_CONNECT_PRE_REQ_QUEUED));

    ucs_debug("ep %p: send wireup pre-request (flags=0x%x)", ep, ep->flags);
    status = ucp_wireup_msg_send(ep, UCP_WIREUP_MSG_PRE_REQUEST,
                                 &ucp_tl_bitmap_max, NULL);

    ucp_ep_update_flags(ep, UCP_EP_FLAG_CONNECT_PRE_REQ_QUEUED, 0);

    return status;
}

ucs_status_t ucp_wireup_connect_remote(ucp_ep_h ep, ucp_lane_index_t lane)
{
    uct_ep_h uct_ep = ucp_ep_get_lane(ep, lane);
    ucs_queue_head_t tmp_q;
    ucs_status_t status;
    ucp_request_t *req;
    uct_ep_h wireup_ep;

    ucs_trace("ep %p: connect lane %d to remote peer", ep, lane);

    UCS_ASYNC_BLOCK(&ep->worker->async);

    /* Checking again, with lock held, if already connected, connection is in
     * progress, or the endpoint is in failed state.
     */
    if ((ep->flags & (UCP_EP_FLAG_REMOTE_ID | UCP_EP_FLAG_FAILED)) ||
        ucp_wireup_ep_test(uct_ep)) {
        status = UCS_OK;
        goto out_unlock;
    }

    ucs_assert(!(ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED));

    ucs_trace("ep %p: connect lane %d to remote peer with wireup ep", ep, lane);

    status = ucp_wireup_ep_create(ep, &wireup_ep);
    if (status != UCS_OK) {
        goto err;
    }

    ucp_ep_set_lane(ep, lane, wireup_ep);

    /* Extract all pending requests from the transport endpoint, otherwise they
     * will prevent the wireup message from being sent (because those requests
     * could not be progressed any more after switching to wireup proxy).
     */
    ucs_queue_head_init(&tmp_q);
    uct_ep_pending_purge(uct_ep, ucp_request_purge_enqueue_cb, &tmp_q);

    /* the wireup ep should use the existing [am_lane] as next_ep */
    ucp_wireup_ep_set_next_ep(ucp_ep_get_lane(ep, lane), uct_ep,
                              ucp_ep_get_rsc_index(ep, lane));

    if (!(ep->flags & UCP_EP_FLAG_CONNECT_REQ_QUEUED)) {
        status = ucp_wireup_send_request(ep);
        if (status != UCS_OK) {
            goto err_destroy_wireup_ep;
        }
    }

    ucs_queue_for_each_extract(req, &tmp_q, send.uct.priv, 1) {
        ucs_trace_req("ep %p: requeue request %p after wireup request",
                      req->send.ep, req);
        status = uct_ep_pending_add(ucp_ep_get_lane(ep, lane), &req->send.uct,
                                    (req->send.uct.func == ucp_wireup_msg_progress) ||
                                    (req->send.uct.func == ucp_wireup_ep_progress_pending) ?
                                    UCT_CB_FLAG_ASYNC : 0);
        if (status != UCS_OK) {
            ucs_fatal("wireup proxy function must always return UCS_OK");
        }
    }

    goto out_unlock;

err_destroy_wireup_ep:
    uct_ep_destroy(ucp_ep_get_lane(ep, lane));
err:
    ucp_ep_set_lane(ep, lane, uct_ep); /* restore am lane */
out_unlock:
    UCS_ASYNC_UNBLOCK(&ep->worker->async);
    return status;
}

static uct_ep_h ucp_wireup_get_tl_ep(uct_ep_h uct_ep)
{
    ucp_wireup_ep_t *wireup_ep = ucp_wireup_ep(uct_ep);
    return (wireup_ep == NULL) ? uct_ep : wireup_ep->super.uct_ep;
}

static void
ucp_wireup_fill_is_connected_params(const ucp_address_entry_t *addr_entry,
                                    uct_ep_is_connected_params_t *params)
{
    params->field_mask = 0;
    if (addr_entry->dev_addr != NULL) {
        params->device_addr = addr_entry->dev_addr;
        params->field_mask |= UCT_EP_IS_CONNECTED_FIELD_DEVICE_ADDR;
    }

    if (addr_entry->iface_addr != NULL) {
        params->iface_addr  = addr_entry->iface_addr;
        params->field_mask |= UCT_EP_IS_CONNECTED_FIELD_IFACE_ADDR;
    }
}

/* Return index of remote p2p lane or UCP_NULL_LANE if no connection was
 * found */
static ucp_lane_index_t
ucp_wireup_get_remote_p2p_lane(ucp_ep_h ep, ucp_lane_index_t lane,
                               const ucp_address_entry_t *addr_entry)
{
    uct_ep_h uct_ep = ucp_ep_get_lane(ep, lane);
    uct_ep_is_connected_params_t params;
    const ucp_address_entry_ep_addr_t *ep_entry;

    ucp_wireup_fill_is_connected_params(addr_entry, &params);

    ucs_carray_for_each(ep_entry, addr_entry->ep_addrs,
                        addr_entry->num_ep_addrs) {
        params.field_mask |= UCT_EP_IS_CONNECTED_FIELD_EP_ADDR;
        params.ep_addr     = ep_entry->addr;

        /* Check connection to remote EP address */
        if (uct_ep_is_connected(ucp_wireup_get_tl_ep(uct_ep), &params)) {
            return ep_entry->lane;
        }
    }

    return UCP_NULL_LANE;
}

int ucp_wireup_is_lane_connected(ucp_ep_h ep, ucp_lane_index_t lane,
                                 const ucp_address_entry_t *addr_entry)
{
    uct_ep_h uct_ep = ucp_ep_get_lane(ep, lane);
    uct_ep_is_connected_params_t params;

    ucp_wireup_fill_is_connected_params(addr_entry, &params);

    if (!ucp_ep_is_lane_p2p(ep, lane)) {
        /* Check if lane is connected to remote iface */
        ucs_assertv(addr_entry->num_ep_addrs == 0, "num_ep_addrs=%u",
                    addr_entry->num_ep_addrs);
        return uct_ep_is_connected(ucp_wireup_get_tl_ep(uct_ep), &params);
    }

    return ucp_wireup_get_remote_p2p_lane(ep, lane, addr_entry) !=
           UCP_NULL_LANE;
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

    status = ucp_address_unpack(worker, msg + 1,
                                UCP_ADDRESS_PACK_FLAGS_ALL |
                                UCP_ADDRESS_PACK_FLAG_NO_TRACE,
                                &unpacked_address);
    if (status != UCS_OK) {
        strncpy(unpacked_address.name, "<malformed address>",
                UCP_WORKER_ADDRESS_NAME_MAX);
        unpacked_address.uuid          = 0;
        unpacked_address.address_count = 0;
        unpacked_address.address_list  = NULL;
    }

    p   = buffer;
    end = buffer + max;

    snprintf(p, end - p,
             "WIREUP %s [%s uuid 0x%"PRIx64" src_ep_id 0x%"PRIx64
             " dst_ep_id 0x%"PRIx64" conn_sn %d]",
             ucp_wireup_msg_str(msg->type), unpacked_address.name,
             unpacked_address.uuid, msg->src_ep_id, msg->dst_ep_id,
             msg->conn_sn);
    p += strlen(p);

    if (unpacked_address.address_list == NULL) {
        return; /* No addresses were unpacked */
    }

    ucp_unpacked_address_for_each(ae, &unpacked_address) {
        UCS_STATIC_BITMAP_FOR_EACH_BIT(tl, &context->tl_bitmap) {
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

static ucp_err_handling_mode_t
ucp_ep_params_err_handling_mode(const ucp_ep_params_t *params)
{
    return (params->field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE) ?
           params->err_mode : UCP_ERR_HANDLING_MODE_NONE;
}

static unsigned
ucp_cm_ep_init_flags(const ucp_ep_params_t *params)
{
    if (params->field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR) {
        return UCP_EP_INIT_CM_WIREUP_CLIENT | UCP_EP_INIT_CM_PHASE;
    }
    if (params->field_mask & UCP_EP_PARAM_FIELD_CONN_REQUEST) {
        return UCP_EP_INIT_CM_WIREUP_SERVER | UCP_EP_INIT_CM_PHASE;
    }

    return 0;
}

unsigned ucp_ep_init_flags(const ucp_worker_h worker,
                           const ucp_ep_params_t *params)
{
    unsigned flags = ucp_cm_ep_init_flags(params);

    if (ucp_ep_init_flags_has_cm(flags) &&
        worker->context->config.ext.cm_use_all_devices) {
        /* request AM lane for wireup MSG protocol which enables all devices */
        flags |= UCP_EP_INIT_CREATE_AM_LANE;
    }

    if (ucp_ep_params_err_handling_mode(params) == UCP_ERR_HANDLING_MODE_PEER) {
        flags |= UCP_EP_INIT_ERR_MODE_PEER_FAILURE;
    }

    return flags;
}

double ucp_wireup_iface_lat_distance_v1(const ucp_worker_iface_t *wiface)
{
    return wiface->worker->context->config.ext.proto_enable ?
        (wiface->attr.latency.c + wiface->distance.latency) :
        wiface->attr.latency.c;
}

double ucp_wireup_iface_lat_distance_v2(const ucp_worker_iface_t *wiface)
{
    ucp_context_h context = wiface->worker->context;
    ucs_linear_func_t lat = wiface->attr.latency;

    if (context->config.ext.proto_enable) {
        lat.c += wiface->distance.latency;
    }
    return ucp_tl_iface_latency(context, &lat);
}

double ucp_wireup_iface_bw_distance(const ucp_worker_iface_t *wiface)
{
    const ucs_sys_dev_distance_t *distance = &wiface->distance;
    const ucp_context_h context            = wiface->worker->context;
    uct_ppn_bandwidth_t bandwidth          = wiface->attr.bandwidth;

    if (context->config.ext.proto_enable) {
        ucp_worker_iface_add_bandwidth(&bandwidth, distance->bandwidth);
    }

    return ucp_tl_iface_bandwidth(context, &bandwidth);
}

UCP_DEFINE_AM(UINT64_MAX, UCP_AM_ID_WIREUP, ucp_wireup_msg_handler,
              ucp_wireup_msg_dump, UCT_CB_FLAG_ASYNC);
