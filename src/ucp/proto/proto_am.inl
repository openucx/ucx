/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_AM_INL_
#define UCP_PROTO_AM_INL_

#include "proto_am.h"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_ep.inl>
#include <ucp/tag/eager.h>
#include <ucp/dt/dt.h>
#include <ucs/profile/profile.h>


#define UCP_STATUS_PENDING_SWITCH (UCS_ERR_LAST - 1)

typedef void (*ucp_req_complete_func_t)(ucp_request_t *req, ucs_status_t status);


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_do_am_bcopy_single(uct_pending_req_t *self, uint8_t am_id,
                       uct_pack_callback_t pack_cb)
{
    ucp_request_t *req   = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep         = req->send.ep;
    ucp_dt_state_t state = req->send.state.dt;
    ssize_t packed_len;

    req->send.lane = ucp_ep_get_am_lane(ep);
    packed_len     = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], am_id, pack_cb,
                                     req, 0);
    if (ucs_unlikely(packed_len < 0)) {
        /* Reset the state to the previous one */
        req->send.state.dt = state;
        return (ucs_status_t)packed_len;
    }

    ucs_assertv((size_t)packed_len <= ucp_ep_get_max_bcopy(ep, req->send.lane),
                "packed_len=%zd max_bcopy=%zu",
                packed_len, ucp_ep_get_max_bcopy(ep, req->send.lane));

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_do_am_bcopy_multi(uct_pending_req_t *self, uint8_t am_id_first,
                                   uint8_t am_id_middle,
                                   uct_pack_callback_t pack_first,
                                   uct_pack_callback_t pack_middle,
                                   int enable_am_bw)
{
    ucp_request_t *req   = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep         = req->send.ep;
    ucp_dt_state_t state = req->send.state.dt;
    ssize_t packed_len;
    uct_ep_h uct_ep;
    int pending_add_res;

    req->send.lane = (!enable_am_bw || (state.offset == 0)) ? /* first part of message must be sent */
                     ucp_ep_get_am_lane(ep) :                 /* via AM lane */
                     ucp_send_request_get_am_bw_lane(req);
    uct_ep         = ep->uct_eps[req->send.lane];

    for (;;) {
        if (state.offset == 0) {
            /* First */
            packed_len = uct_ep_am_bcopy(uct_ep, am_id_first, pack_first, req, 0);
            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_bcopy_first",
                                                   packed_len, packed_len);
        } else {
            ucs_assert(state.offset < req->send.length);
            /* Middle or last */
            packed_len = uct_ep_am_bcopy(uct_ep, am_id_middle, pack_middle, req, 0);
            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_bcopy_middle",
                                                   packed_len, packed_len);
        }

        if (ucs_unlikely(packed_len < 0)) {
            /* Reset the state to the previous one */
            req->send.state.dt = state;

            if ((packed_len == UCS_ERR_NO_RESOURCE) &&
                (req->send.lane != req->send.pending_lane)) {
                /* switch to new pending lane */
                pending_add_res = ucp_request_pending_add(req, 0);
                if (!pending_add_res) {
                    /* failed to switch req to pending queue, try again */
                    continue;
                }
                return (ucs_status_t)UCP_STATUS_PENDING_SWITCH;
            } else {
                return (ucs_status_t)packed_len;
            }
        } else {
            ucs_assertv(/* The packed length has to be the same as maximum
                         * AM Bcopy for the first and middle segments */
                        ((req->send.state.dt.offset < req->send.length) &&
                         (packed_len == ucp_ep_get_max_bcopy(ep, req->send.lane))) ||
                        /* The packed length has to be the same or less than
                         * maximum AM Bcopy for the last segment */
                        (packed_len <= ucp_ep_get_max_bcopy(ep, req->send.lane)),
                        "packed_len=%zd max_bcopy=%zu",
                        packed_len, ucp_ep_get_max_bcopy(ep, req->send.lane));
            ucs_assertv(req->send.state.dt.offset <= req->send.length,
                        "offset=%zd length=%zu",
                        req->send.state.dt.offset, req->send.length);

            /* If the last segment was sent, return UCS_OK,
             * otherwise - UCS_INPROGRESS */
            if (enable_am_bw) {
                ucp_send_request_next_am_bw_lane(req);
            }
            return ((req->send.state.dt.offset < req->send.length) ?
                    UCS_INPROGRESS : UCS_OK);
        }
    }
}

static UCS_F_ALWAYS_INLINE
size_t ucp_dt_iov_copy_iov_uct(uct_iov_t *iov, size_t *iovcnt,
                               size_t max_dst_iov, ucp_dt_state_t *state,
                               const ucp_dt_iov_t *src_iov, size_t length_max,
                               ucp_md_index_t md_index, uint64_t md_flags)
{
    size_t length_it = 0;
    size_t iov_offset, max_src_iov, src_it, dst_it;
    ucp_md_index_t memh_index;

    iov_offset               = state->dt.iov.iov_offset;
    max_src_iov              = state->dt.iov.iovcnt;
    src_it                   = state->dt.iov.iovcnt_offset;
    dst_it                   = 0;
    state->dt.iov.iov_offset = 0;

    while ((dst_it < max_dst_iov) && (src_it < max_src_iov)) {
        if (src_iov[src_it].length != 0) {
            iov[dst_it].buffer   = UCS_PTR_BYTE_OFFSET(src_iov[src_it].buffer,
                                                       iov_offset);
            iov[dst_it].length   = src_iov[src_it].length - iov_offset;
            if (md_flags & UCT_MD_FLAG_NEED_MEMH) {
                ucs_assert(state->dt.iov.dt_reg != NULL);
                memh_index       = ucs_bitmap2idx(state->dt.iov.dt_reg[src_it].md_map,
                                                  md_index);
                iov[dst_it].memh = state->dt.iov.dt_reg[src_it].memh[memh_index];
            } else {
                ucs_assert(state->dt.iov.dt_reg == NULL);
                iov[dst_it].memh = UCT_MEM_HANDLE_NULL;
            }
            iov[dst_it].stride   = 0;
            iov[dst_it].count    = 1;
            length_it           += iov[dst_it].length;

            ++dst_it;
            if (length_it >= length_max) {
                iov[dst_it - 1].length  -= (length_it - length_max);
                length_it                = length_max;
                state->dt.iov.iov_offset = iov_offset + iov[dst_it - 1].length;
                break;
            }
        }
        iov_offset = 0;
        ++src_it;
    }

    state->dt.iov.iovcnt_offset = src_it;
    *iovcnt                     = dst_it;

    return length_it;
}

static UCS_F_ALWAYS_INLINE
void ucp_dt_iov_copy_uct(ucp_context_h context, uct_iov_t *iov, size_t *iovcnt,
                         size_t max_dst_iov, ucp_dt_state_t *state,
                         const ucp_dt_iov_t *src_iov, ucp_datatype_t datatype,
                         size_t length_max, ucp_md_index_t md_index,
                         ucp_mem_desc_t *mdesc)
{
    uint64_t md_flags = context->tl_mds[md_index].attr.cap.flags;
    size_t length_it  = 0;
    ucp_md_index_t memh_index;

    ucs_assert((context->tl_mds[md_index].attr.cap.flags & UCT_MD_FLAG_REG) ||
               !(md_flags & UCT_MD_FLAG_NEED_MEMH));

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if (md_flags & UCT_MD_FLAG_NEED_MEMH) {
            if (mdesc) {
                memh_index  = ucs_bitmap2idx(mdesc->memh->md_map, md_index);
                iov[0].memh = mdesc->memh->uct[memh_index];
            } else {
                memh_index  = ucs_bitmap2idx(state->dt.contig.md_map, md_index);
                iov[0].memh = state->dt.contig.memh[memh_index];
            }
        } else {
            iov[0].memh = UCT_MEM_HANDLE_NULL;
        }
        iov[0].buffer = UCS_PTR_BYTE_OFFSET(src_iov, state->offset);
        iov[0].length = length_max;
        iov[0].stride = 0;
        iov[0].count  = 1;

        *iovcnt   = 1;
        length_it = iov[0].length;
        break;
    case UCP_DATATYPE_IOV:
        length_it = ucp_dt_iov_copy_iov_uct(iov, iovcnt, max_dst_iov, state,
                                            src_iov, length_max, md_index,
                                            md_flags);
        break;
    default:
        ucs_error("Invalid data type");
    }

    state->offset += length_it;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_am_zcopy_common(ucp_request_t *req, const void *hdr,
                                 size_t hdr_size, ucp_mem_desc_t *user_hdr_desc,
                                 size_t user_hdr_size, uct_iov_t *iov,
                                 size_t max_iov, size_t max_length,
                                 uint8_t am_id, ucp_dt_state_t *state)
{
    ucp_ep_t *ep          = req->send.ep;
    ucp_md_index_t md_idx = ucp_ep_md_index(ep, req->send.lane);
    size_t iovcnt         = 0ul;
    unsigned user_hdr_iov_cnt;

    if (user_hdr_size != 0) {
        ucs_assert((req->send.length == 0) || (max_length > user_hdr_size));
        ucs_assert(max_iov > 1);

        iov[0].buffer    = user_hdr_desc + 1;
        iov[0].length    = user_hdr_size;
        iov[0].memh      = ucp_memh2uct(user_hdr_desc->memh, md_idx);
        iov[0].stride    = 0;
        iov[0].count     = 1;
        user_hdr_iov_cnt = 1;
    } else {
        user_hdr_iov_cnt = 0;
    }

    ucp_dt_iov_copy_uct(ep->worker->context, iov + user_hdr_iov_cnt, &iovcnt,
                        max_iov - user_hdr_iov_cnt, state, req->send.buffer,
                        req->send.datatype, max_length - user_hdr_size,
                        md_idx, NULL);

    return uct_ep_am_zcopy(ep->uct_eps[req->send.lane], am_id, (void*)hdr,
                           hdr_size, iov, iovcnt + user_hdr_iov_cnt, 0,
                           &req->send.state.uct_comp);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_zcopy_single_handle_status(ucp_request_t *req,
                                  const ucp_dt_state_t *dt_state,
                                  ucs_status_t status,
                                  ucp_req_complete_func_t complete_ok)
{
    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (status == UCS_OK) {
        complete_ok(req, UCS_OK);
    } else {
        /* IN_PROGRESS also goes here */
        ucp_request_send_state_advance(req, dt_state,
                                       UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
                                       status);
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_do_am_zcopy_single(uct_pending_req_t *self, uint8_t am_id,
                                    const void *hdr, size_t hdr_size,
                                    ucp_mem_desc_t *user_hdr_desc,
                                    size_t user_hdr_size,
                                    ucp_req_complete_func_t complete)
{
    ucp_request_t *req   = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep         = req->send.ep;
    size_t max_iov       = ucp_ep_config(ep)->am.max_iov;
    uct_iov_t *iov       = ucs_alloca(max_iov * sizeof(uct_iov_t));
    ucp_dt_state_t state = req->send.state.dt;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_am_lane(ep);

    status = ucp_am_zcopy_common(req, hdr, hdr_size, user_hdr_desc, user_hdr_size,
                                 iov, max_iov, req->send.length + user_hdr_size,
                                 am_id, &state);

    return ucp_am_zcopy_single_handle_status(req, &state, status, complete);
}

static UCS_F_ALWAYS_INLINE
void ucp_am_zcopy_complete_last_stage(ucp_request_t *req, ucp_dt_state_t *state,
                                      ucp_req_complete_func_t complete)
{
    ucp_request_send_state_advance(req, state,
                                   UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
                                   UCS_OK);

    /* Complete a request on a last stage if all previous AM
     * Zcopy operations completed successfully. If there are
     * operations that are in progress on other lanes, the last
     * completed operation will complete the request */
    if (req->send.state.uct_comp.count == 0) {
        complete(req, UCS_OK);
    }
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_do_am_zcopy_multi(uct_pending_req_t *self, uint8_t am_id_first,
                                   uint8_t am_id_middle,
                                   const void *hdr_first, size_t hdr_size_first,
                                   const void *hdr_middle, size_t hdr_size_middle,
                                   ucp_mem_desc_t *user_hdr_desc,
                                   size_t user_hdr_size,
                                   ucp_req_complete_func_t complete,
                                   int enable_am_bw)
{
    ucp_request_t *req    = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep          = req->send.ep;
    unsigned flag_iov_mid = 0;
    size_t iovcnt         = 0;
    ucp_dt_state_t state;
    size_t max_middle;
    size_t max_iov;
    uct_iov_t *iov;
    size_t offset;
    size_t mid_len;
    ucs_status_t status;
    uct_ep_h uct_ep;
    int pending_add_res;

    if (enable_am_bw && (req->send.state.dt.offset != 0)) {
        req->send.lane = ucp_send_request_get_am_bw_lane(req);
        ucp_send_request_add_reg_lane(req, req->send.lane);
    } else {
        req->send.lane = ucp_ep_get_am_lane(ep);
    }

    uct_ep     = ep->uct_eps[req->send.lane];
    max_middle = ucp_ep_get_max_zcopy(ep, req->send.lane) - hdr_size_middle;
    max_iov    = ucp_ep_get_max_iov(ep, req->send.lane);
    iov        = ucs_alloca(max_iov * sizeof(uct_iov_t));

    for (;;) {
        state  = req->send.state.dt;
        offset = state.offset;

        ucs_assert(max_iov > 0);
        if (UCP_DT_IS_IOV(req->send.datatype)) {
            /* This flag should guarantee middle stage usage if iovcnt exceeded */
            flag_iov_mid = ((state.dt.iov.iovcnt_offset + max_iov) <
                            state.dt.iov.iovcnt);
        } else {
            ucs_assert(UCP_DT_IS_CONTIG(req->send.datatype));
        }

        if (offset == 0) {
            /* First stage */
            ucs_assert(req->send.lane == ucp_ep_get_am_lane(ep));

            status = ucp_am_zcopy_common(req, hdr_first, hdr_size_first,
                                         user_hdr_desc, user_hdr_size, iov, max_iov,
                                         max_middle - hdr_size_first + hdr_size_middle,
                                         am_id_first, &state);

            ucs_assertv(state.offset != 0, "state must be changed on 1st stage");
            ucs_assertv(state.offset < req->send.length, "state.offset=%zu",
                        state.offset);

            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_zcopy_first",
                                                   iov[0].length, status);
        } else {
            /* Middle or last stage */
            mid_len = ucs_min(max_middle, req->send.length - offset);
            ucs_assert(offset + mid_len <= req->send.length);
            ucp_dt_iov_copy_uct(ep->worker->context, iov, &iovcnt, max_iov, &state,
                                req->send.buffer, req->send.datatype, mid_len,
                                ucp_ep_md_index(ep, req->send.lane), NULL);

            if (offset < state.offset) {
                status = uct_ep_am_zcopy(uct_ep, am_id_middle, (void*)hdr_middle,
                                         hdr_size_middle, iov, iovcnt, 0,
                                         &req->send.state.uct_comp);
            } else if (state.offset == req->send.length) {
                /* Empty IOVs on last stage */
                ucp_am_zcopy_complete_last_stage(req, &state, complete);
                return UCS_OK;
            } else {
                ucs_assert(offset == state.offset);
                /* Empty IOVs in the middle */
                ucp_request_send_state_advance(req, &state,
                                               UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
                                               UCS_OK);
                continue;
            }

            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_zcopy_middle",
                                                   iov[0].length, status);

            if (!flag_iov_mid && (offset + mid_len == req->send.length)) {
                /* Last stage */
                if (status == UCS_OK) {
                    ucp_am_zcopy_complete_last_stage(req, &state, complete);
                    return UCS_OK;
                }

                ucp_request_send_state_advance(req, &state,
                                               UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
                                               status);
                if (!UCS_STATUS_IS_ERR(status)) {
                    if (enable_am_bw) {
                        ucp_send_request_next_am_bw_lane(req);
                    }
                    return UCS_OK;
                }
            }
        }

        if (status == UCS_ERR_NO_RESOURCE) {
            if (req->send.lane != req->send.pending_lane) {
                /* switch to new pending lane */
                pending_add_res = ucp_request_pending_add(req, 0);
                if (!pending_add_res) {
                    /* failed to switch req to pending queue, try again */
                    continue;
                }
                return UCS_OK;
            }

            return UCS_ERR_NO_RESOURCE;
        }

        ucp_request_send_state_advance(req, &state,
                                       UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
                                       status);
        if (UCS_STATUS_IS_ERR(status)) {
            if (req->send.state.uct_comp.count == 0) {
               complete(req, status);
            }
            return UCS_OK;
        } else {
            if (enable_am_bw) {
                ucp_send_request_next_am_bw_lane(req);
            }
            return UCS_INPROGRESS;
        }
    }
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_get_zcopy_threshold(const ucp_request_t *req,
                              const ucp_ep_msg_config_t *msg_config,
                              size_t count, size_t max_zcopy)
{
    ucp_worker_h worker;
    ucp_lane_index_t lane;
    ucp_rsc_index_t rsc_index;
    size_t zcopy_thresh;

    if (ucs_unlikely(msg_config->max_zcopy == 0)) {
        return max_zcopy;
    }

    if (ucs_likely(UCP_DT_IS_CONTIG(req->send.datatype))) {
        return ucs_min(max_zcopy, msg_config->mem_type_zcopy_thresh[req->send.mem_type]);
    } else if (UCP_DT_IS_IOV(req->send.datatype)) {
        if (0 == count) {
            /* disable zcopy */
            zcopy_thresh = max_zcopy;
        } else if (!msg_config->zcopy_auto_thresh) {
            /* The user defined threshold or no zcopy enabled */
            zcopy_thresh = msg_config->zcopy_thresh[0];
        } else if (count <= UCP_MAX_IOV) {
            /* Using pre-calculated thresholds */
            zcopy_thresh = msg_config->zcopy_thresh[count - 1];
        } else {
            /* Calculate threshold */
            lane         = req->send.lane;
            rsc_index    = ucp_ep_config(req->send.ep)->key.lanes[lane].rsc_index;
            worker       = req->send.ep->worker;
            zcopy_thresh = ucp_ep_config_get_zcopy_auto_thresh(count,
                              &ucp_ep_md_attr(req->send.ep, lane)->reg_cost,
                              worker->context,
                              ucp_worker_iface_bandwidth(worker, rsc_index));
        }
        return ucs_min(max_zcopy, zcopy_thresh);
    } else if (UCP_DT_IS_GENERIC(req->send.datatype)) {
        return max_zcopy;
    }

    ucs_error("Unsupported datatype");

    return max_zcopy;
}

static UCS_F_ALWAYS_INLINE ssize_t
ucp_proto_get_short_max(const ucp_request_t *req,
                        const ucp_ep_msg_config_t *msg_config)
{
    return (!UCP_DT_IS_CONTIG(req->send.datatype) ||
            (req->flags & UCP_REQUEST_FLAG_SYNC) ||
            (!UCP_MEM_IS_HOST(req->send.mem_type))) ?
           -1 : msg_config->max_short;
}

static UCS_F_ALWAYS_INLINE ucp_request_t*
ucp_proto_ssend_ack_request_alloc(ucp_worker_h worker, ucs_ptr_map_key_t ep_id)
{
    ucp_request_t *req;

    req = ucp_request_get(worker);
    if (req == NULL) {
        return NULL;
    }

    req->flags              = 0;
    req->send.ep            = ucp_worker_get_ep_by_id(worker, ep_id);
    req->send.uct.func      = ucp_proto_progress_am_single;
    req->send.proto.comp_cb = ucp_request_put;
    req->send.proto.status  = UCS_OK;

    return req;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_am_bcopy_handle_status_from_pending(uct_pending_req_t *self, int multi,
                                        int tag_sync, ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    if (multi) {
        if (status == UCS_INPROGRESS) {
            return UCS_INPROGRESS;
        }

        if (ucs_unlikely(status == UCP_STATUS_PENDING_SWITCH)) {
            return UCS_OK;
        }
    } else {
        ucs_assert(status != UCS_INPROGRESS);
    }

    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        return UCS_ERR_NO_RESOURCE;
    }

    ucp_request_send_generic_dt_finish(req);
    if (tag_sync) {
        ucp_tag_eager_sync_completion(req, UCP_REQUEST_FLAG_LOCAL_COMPLETED,
                                      status);
    } else {
        ucp_request_complete_send(req, status);
    }

    return UCS_OK;
}

#endif
