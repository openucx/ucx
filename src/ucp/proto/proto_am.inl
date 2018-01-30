/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_ep.inl>
#include <ucp/dt/dt.h>
#include <ucs/debug/profile.h>

#define UCP_STATUS_PENDING_SWITCH (UCS_ERR_LAST - 1)

typedef void (*ucp_req_complete_func_t)(ucp_request_t *req, ucs_status_t status);


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_do_am_bcopy_single(uct_pending_req_t *self, uint8_t am_id,
                       uct_pack_callback_t pack_cb)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len;

    req->send.lane = ucp_ep_get_am_lane(ep);
    packed_len     = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], am_id, pack_cb,
                                     req, 0);
    if (packed_len < 0) {
        return packed_len;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_do_am_bcopy_multi(uct_pending_req_t *self, uint8_t am_id_first,
                                   uint8_t am_id_middle, uint8_t am_id_last,
                                   size_t hdr_size_middle,
                                   uct_pack_callback_t pack_first,
                                   uct_pack_callback_t pack_middle,
                                   uct_pack_callback_t pack_last,
                                   int enable_am_bw)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucs_status_t status;
    size_t max_middle;
    ssize_t packed_len;
    uct_ep_h uct_ep;
    size_t offset;
    int pending_adde_res;

    offset         = req->send.state.dt.offset;
    req->send.lane = (!enable_am_bw || !offset) ? /* first part of message must be sent */
                     ucp_ep_get_am_lane(ep) :     /* via AM lane */
                     ucp_send_request_get_next_am_bw_lane(req);
    uct_ep         = ep->uct_eps[req->send.lane];
    max_middle     = ucp_ep_get_max_bcopy(ep, req->send.lane) - hdr_size_middle;

    for (;;) {
        if (offset == 0) {
            /* First */
            packed_len = uct_ep_am_bcopy(uct_ep, am_id_first, pack_first, req, 0);
            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_bcopy_first", packed_len,
                                                   packed_len);
        } else if (offset + max_middle < req->send.length) {
            /* Middle */
            packed_len = uct_ep_am_bcopy(uct_ep, am_id_middle, pack_middle, req, 0);
            ucs_assertv((packed_len < 0) || (packed_len <= max_middle + hdr_size_middle),
                        "packed_len=%zd max_middle=%zu hdr_size_middle=%zu",
                        packed_len, max_middle, hdr_size_middle);
            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_bcopy_middle",
                                                   packed_len, packed_len);
        } else {
            /* Last */
            packed_len = uct_ep_am_bcopy(uct_ep, am_id_last, pack_last, req, 0);
            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_bcopy_last", packed_len,
                                                   packed_len);
            if (packed_len >= 0) {
                return UCS_OK;
            }
        }

        if (ucs_unlikely(packed_len < 0)) {
            if (req->send.lane != req->send.pending_lane) {
                /* switch to new pending lane */
                pending_adde_res = ucp_request_pending_add(req, &status);
                if (!pending_adde_res) {
                    /* failed to switch req to pending queue, try again */
                    continue;
                }
                ucs_assert(status == UCS_INPROGRESS);
                return UCP_STATUS_PENDING_SWITCH;
            } else {
                return packed_len;
            }
        } else {
            return UCS_INPROGRESS;
        }
    }
}

static UCS_F_ALWAYS_INLINE
void ucp_dt_iov_copy_uct(ucp_context_h context, uct_iov_t *iov, size_t *iovcnt,
                         size_t max_dst_iov, ucp_dt_state_t *state,
                         const ucp_dt_iov_t *src_iov, ucp_datatype_t datatype,
                         size_t length_max, ucp_md_index_t md_index,
                         ucp_mem_desc_t *mdesc)
{
    size_t iov_offset, max_src_iov, src_it, dst_it;
    size_t length_it = 0;
    ucp_md_index_t memh_index;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if (context->tl_mds[md_index].attr.cap.flags & UCT_MD_FLAG_REG) {
            if (mdesc) {
                memh_index  = ucp_memh_map2idx(mdesc->memh->md_map, md_index);
                iov[0].memh = mdesc->memh->uct[memh_index];
            } else {
                memh_index  = ucp_memh_map2idx(state->dt.contig.md_map, md_index);
                iov[0].memh = state->dt.contig.memh[memh_index];
            }
        } else {
            iov[0].memh = UCT_MEM_HANDLE_NULL;
        }
        iov[0].buffer = (void *)src_iov + state->offset;
        iov[0].length = length_max;
        iov[0].stride = 0;
        iov[0].count  = 1;

        *iovcnt   = 1;
        length_it = iov[0].length;
        break;
    case UCP_DATATYPE_IOV:
        iov_offset                  = state->dt.iov.iov_offset;
        max_src_iov                 = state->dt.iov.iovcnt;
        src_it                      = state->dt.iov.iovcnt_offset;
        dst_it                      = 0;
        state->dt.iov.iov_offset    = 0;
        while ((dst_it < max_dst_iov) && (src_it < max_src_iov)) {
            if (src_iov[src_it].length) {
                iov[dst_it].buffer  = src_iov[src_it].buffer + iov_offset;
                iov[dst_it].length  = src_iov[src_it].length - iov_offset;
                iov[dst_it].memh    = state->dt.iov.dt_reg[src_it].memh[0];
                iov[dst_it].stride  = 0;
                iov[dst_it].count   = 1;
                length_it          += iov[dst_it].length;

                ++dst_it;
                if (length_it >= length_max) {
                    iov[dst_it - 1].length      -= (length_it - length_max);
                    length_it                    = length_max;
                    state->dt.iov.iov_offset     = iov_offset + iov[dst_it - 1].length;
                    break;
                }
            }
            iov_offset = 0;
            ++src_it;
        }

        state->dt.iov.iovcnt_offset = src_it;
        *iovcnt                     = dst_it;
        break;
    default:
        ucs_error("Invalid data type");
    }

    state->offset += length_it;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_do_am_zcopy_single(uct_pending_req_t *self, uint8_t am_id,
                                    const void *hdr, size_t hdr_size,
                                    ucp_req_complete_func_t complete)
{
    ucp_request_t  *req    = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep           = req->send.ep;
    size_t max_iov         = ucp_ep_config(ep)->am.max_iov;
    uct_iov_t *iov         = ucs_alloca(max_iov * sizeof(uct_iov_t));
    size_t iovcnt          = 0;
    ucp_dt_state_t state   = req->send.state.dt;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_am_lane(ep);

    ucp_dt_iov_copy_uct(ep->worker->context,iov, &iovcnt, max_iov,
                        &state, req->send.buffer, req->send.datatype,
                        req->send.length, 0, NULL);

    status = uct_ep_am_zcopy(ep->uct_eps[req->send.lane], am_id, (void*)hdr,
                             hdr_size, iov, iovcnt, 0,
                             &req->send.state.uct_comp);
    if (status == UCS_OK) {
        complete(req, UCS_OK);
    } else {
        ucp_request_send_state_advance(req, &state,
                                       UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
                                       status);
    }
    return UCS_STATUS_IS_ERR(status) ? status : UCS_OK;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_do_am_zcopy_multi(uct_pending_req_t *self, uint8_t am_id_first,
                                   uint8_t am_id_middle, uint8_t am_id_last,
                                   const void *hdr_first, size_t hdr_size_first,
                                   const void *hdr_middle, size_t hdr_size_middle,
                                   ucp_req_complete_func_t complete, int enable_am_bw)
{
    ucp_request_t *req      = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep            = req->send.ep;
    unsigned flag_iov_mid   = 0;
    size_t iovcnt           = 0;
    ucp_dt_state_t state;
    size_t max_middle;
    size_t max_iov;
    uct_iov_t *iov;
    size_t offset;
    ucs_status_t status;
    uct_ep_h uct_ep;
    int pending_adde_res;

    if (UCP_DT_IS_CONTIG(req->send.datatype)) {
        if (enable_am_bw && req->send.state.dt.offset) {
            req->send.lane = ucp_send_request_get_next_am_bw_lane(req);
            ucp_send_request_add_reg_lane(req, req->send.lane);
        } else {
            req->send.lane = ucp_ep_get_am_lane(ep);
        }
    } else {
        ucs_assert(UCP_DT_IS_IOV(req->send.datatype));
        /* disable multilane for IOV datatype.
         * TODO: add IOV processing for multilane */
        req->send.lane = ucp_ep_get_am_lane(ep);
    }

    uct_ep     = ep->uct_eps[req->send.lane];
    max_middle = ucp_ep_get_max_zcopy(ep, req->send.lane) - hdr_size_middle;
    max_iov    = ucp_ep_get_max_iov(ep, req->send.lane);
    iov        = ucs_alloca(max_iov * sizeof(uct_iov_t));

    for (;;) {
        state      = req->send.state.dt;
        offset     = state.offset;

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
            ucp_dt_iov_copy_uct(ep->worker->context, iov, &iovcnt, max_iov, &state,
                                req->send.buffer,  req->send.datatype,
                                max_middle - hdr_size_first + hdr_size_middle,
                                ucp_ep_md_index(ep, req->send.lane), NULL);

            status = uct_ep_am_zcopy(uct_ep, am_id_first, (void*)hdr_first,
                                     hdr_size_first, iov, iovcnt, 0,
                                     &req->send.state.uct_comp);

            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_zcopy_first",
                                                   iov[0].length, status);
        } else if ((offset + max_middle < req->send.length) || flag_iov_mid) {
            /* Middle stage */
            ucp_dt_iov_copy_uct(ep->worker->context, iov, &iovcnt, max_iov,
                                &state, req->send.buffer, req->send.datatype,
                                max_middle, ucp_ep_md_index(ep, req->send.lane), NULL);

            status = uct_ep_am_zcopy(uct_ep, am_id_middle, (void*)hdr_middle,
                                     hdr_size_middle, iov, iovcnt, 0,
                                     &req->send.state.uct_comp);

            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_zcopy_middle",
                                                   iov[0].length, status);
        } else {
            /* Last stage */
            ucp_dt_iov_copy_uct(ep->worker->context, iov, &iovcnt, max_iov, &state,
                                req->send.buffer, req->send.datatype,
                                req->send.length - offset,
                                ucp_ep_md_index(ep, req->send.lane), NULL);

            status = uct_ep_am_zcopy(uct_ep, am_id_last, (void*)hdr_middle,
                                     hdr_size_middle, iov, iovcnt, 0,
                                     &req->send.state.uct_comp);
            UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(req, "am_zcopy_last",
                                                   iov[0].length, status);
            if (status == UCS_OK) {
                complete(req, UCS_OK);
                return UCS_OK;
            }
            ucp_request_send_state_advance(req, &state,
                                           UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
                                           status);
            if (!UCS_STATUS_IS_ERR(status)) {
                return UCS_OK;
            }
        }

        if (status == UCS_ERR_NO_RESOURCE) {
            if (req->send.lane != req->send.pending_lane) {
                /* switch to new pending lane */
                pending_adde_res = ucp_request_pending_add(req, &status);
                if (!pending_adde_res) {
                    /* failed to switch req to pending queue, try again */
                    continue;
                }
                ucs_assert(status == UCS_INPROGRESS);
                return UCS_OK;
            }
        }
        ucp_request_send_state_advance(req, &state,
                                       UCP_REQUEST_SEND_PROTO_ZCOPY_AM,
                                       status);

        return UCS_STATUS_IS_ERR(status) ? status : UCS_INPROGRESS;
    }
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_get_zcopy_threshold(const ucp_request_t *req,
                              const ucp_ep_msg_config_t *msg_config,
                              size_t count, size_t max_zcopy)
{
    ucp_worker_h     worker;
    ucp_lane_index_t lane;
    ucp_rsc_index_t  rsc_index;
    size_t           zcopy_thresh;

    if (ucs_unlikely(msg_config->max_zcopy == 0)) {
        return max_zcopy;
    }

    if (ucs_unlikely(!UCP_MEM_IS_HOST(req->send.mem_type))) {
        return 1;
    }

    if (ucs_likely(UCP_DT_IS_CONTIG(req->send.datatype))) {
        return ucs_min(max_zcopy, msg_config->zcopy_thresh[0]);
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
            lane      = req->send.lane;
            rsc_index = ucp_ep_config(req->send.ep)->key.lanes[lane].rsc_index;
            worker    = req->send.ep->worker;
            zcopy_thresh = ucp_ep_config_get_zcopy_auto_thresh(count,
                               &ucp_ep_md_attr(req->send.ep, lane)->reg_cost,
                               worker->context,
                               worker->ifaces[rsc_index].attr.bandwidth);
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
    return  (!UCP_DT_IS_CONTIG(req->send.datatype) ||
            (req->flags & UCP_REQUEST_FLAG_SYNC) ||
            (!UCP_MEM_IS_HOST(req->send.mem_type))) ?
           -1 : msg_config->max_short;
}
