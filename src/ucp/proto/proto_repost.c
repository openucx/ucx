/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_repost.h"
#include "proto_debug.h"
#include "proto_perf.h"
#include "proto_common.inl"

#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_rkey.inl>

#include <string.h>


/* Operations outstanding purge reports.
 * The lane must support all of them since we have a single protocol
 * for reposting. */
#define UCP_PROTO_REPOST_TL_CAP_FLAGS \
    (UCT_IFACE_FLAG_AM_SHORT  | UCT_IFACE_FLAG_AM_BCOPY  | \
     UCT_IFACE_FLAG_PUT_SHORT | UCT_IFACE_FLAG_PUT_BCOPY | \
     UCT_IFACE_FLAG_PUT_ZCOPY | UCT_IFACE_FLAG_GET_BCOPY | \
     UCT_IFACE_FLAG_GET_ZCOPY)


typedef struct {
    ucp_lane_index_t lane;
} ucp_proto_repost_priv_t;


typedef struct {
    const void *buffer;
    size_t     length;
} ucp_proto_repost_pack_t;


typedef struct {
    uct_completion_t comp;
    ucp_request_t    *origin;
    size_t           first_index; /* SIZE_MAX for an explicit index list */
    size_t           count;
    size_t           indices[UCP_MAX_IOV];
} ucp_proto_repost_sgl_t;


static void ucp_proto_repost_sgl_complete(uct_completion_t *comp)
{
    ucp_proto_repost_sgl_t *batch = ucs_container_of(comp,
                                                     ucp_proto_repost_sgl_t,
                                                     comp);
    uct_completion_t *origin      = &batch->origin->send.state.uct_comp;
    ucs_status_t status           = comp->status;

    ucs_mpool_put_inline(batch);
    ucp_invoke_uct_completion(origin, status);
}

uct_completion_t *ucp_proto_repost_sgl_completion(ucp_request_t *origin,
                                                  const size_t *indices,
                                                  size_t count)
{
    static ucs_mpool_ops_t ops = {
        .chunk_alloc   = ucs_mpool_chunk_malloc,
        .chunk_release = ucs_mpool_chunk_free
    };
    ucp_worker_h worker        = origin->send.ep->worker;
    ucp_proto_repost_sgl_t *batch;
    ucs_mpool_params_t params;
    ucs_status_t status;

    if (ucs_unlikely(!(worker->flags &
                       UCP_WORKER_FLAG_SGL_REPOST_MPOOL_INITIALIZED))) {
        ucs_mpool_params_reset(&params);
        params.elem_size       = sizeof(*batch);
        params.elems_per_chunk = 128;
        params.ops             = &ops;
        params.name            = "ucp_sgl_repost";
        status = ucs_mpool_init(&params, &worker->sgl_repost_mp);
        if (status != UCS_OK) {
            return NULL;
        }
        worker->flags |= UCP_WORKER_FLAG_SGL_REPOST_MPOOL_INITIALIZED;
    }

    batch = ucs_mpool_get_inline(&worker->sgl_repost_mp);
    if (batch == NULL) {
        return NULL;
    }

    batch->origin      = origin;
    batch->first_index = (indices == NULL) ? origin->send.state.dt_iter.offset :
                                             SIZE_MAX;
    batch->count       = count;
    if (indices != NULL) {
        ucs_assert(count <= ucs_static_array_size(batch->indices));
        memcpy(batch->indices, indices, count * sizeof(*indices));
    }
    batch->comp.func   = ucp_proto_repost_sgl_complete;
    batch->comp.count  = 1;
    batch->comp.status = UCS_OK;
    return &batch->comp;
}

void ucp_proto_repost_sgl_release(uct_completion_t *comp)
{
    ucs_mpool_put_inline(ucs_container_of(comp, ucp_proto_repost_sgl_t, comp));
}


ucs_status_t
ucp_proto_repost_sgl_put(ucp_request_t *origin, uct_ep_h ep,
                         void *const *buffers, const size_t *lengths,
                         uct_mem_h const *memhs, const uint64_t *remote_addrs,
                         uct_rkey_t const *rkeys, size_t count,
                         const size_t *indices)
{
    uct_completion_t *comp;
    ucs_status_t status;

    comp = ucp_proto_repost_sgl_completion(origin, indices, count);
    if (comp == NULL) {
        return UCS_ERR_NO_MEMORY;
    }
    status = uct_ep_put_sgl_zcopy(ep, buffers, lengths, memhs, remote_addrs,
                                  rkeys, NULL, NULL, count, comp);
    if (status != UCS_INPROGRESS) {
        ucp_proto_repost_sgl_release(comp);
    }
    return status;
}


static size_t ucp_proto_repost_pack(void *dest, void *arg)
{
    ucp_proto_repost_pack_t *pack = arg;

    if (pack->length > 0) {
        memcpy(dest, pack->buffer, pack->length);
    }

    return pack->length;
}

static void ucp_proto_repost_probe(const ucp_proto_init_params_t *params)
{
    ucp_proto_perf_factors_t perf_factors = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    ucp_lane_index_t lane                 = UCP_NULL_LANE;
    ucp_lane_index_t num_lanes;
    ucp_proto_repost_priv_t rpriv;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    UCS_STATIC_ASSERT(UCP_OP_ID_REPOST < UCP_PROTO_SELECT_OP_FLAGS_BASE);

    if (params->select_param->op_id_flags != UCP_OP_ID_REPOST) {
        return;
    }

    num_lanes = ucp_proto_common_find_lanes(params, 0, UCP_LANE_TYPE_LAST,
                                            UCP_PROTO_REPOST_TL_CAP_FLAGS, 0,
                                            1, 0, NULL, &lane);
    rpriv.lane = (num_lanes == 0) ? UCP_NULL_LANE : lane;

    status = ucp_proto_perf_create("repost", &perf);
    if (status != UCS_OK) {
        return;
    }

    /* Finite cost, so this instance wins over the reconfig fallback. */
    perf_factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL] = UCS_LINEAR_FUNC_ZERO;
    status = ucp_proto_perf_add_funcs(
            perf, 0, SIZE_MAX, perf_factors,
            ucp_proto_perf_node_new_data("repost", ""), NULL);
    if (status != UCS_OK) {
        ucp_proto_perf_destroy(perf);
        return;
    }

    ucp_proto_select_add_proto(params, 0, 0, perf, &rpriv, sizeof(rpriv));
}

static ucs_status_t
ucp_proto_repost_copy_data(ucp_request_t *req, void **buffer_p, size_t length)
{
    void *copy;

    if (length == 0) {
        return UCS_OK;
    }

    if (*buffer_p == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    copy = ucs_malloc(length, "ucp_proto_repost_data");
    if (copy == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    memcpy(copy, *buffer_p, length);
    *buffer_p                = copy;
    req->send.repost.buffer  = copy;
    return UCS_OK;
}

static ucs_status_t ucp_proto_repost_copy_iov(ucp_request_t *req)
{
    uct_ep_op_info_t *info = req->send.repost.info;
    size_t iovcnt          = info->rma.payload.zcopy.iovcnt;
    const uct_iov_t *iov   = info->rma.payload.zcopy.iov;
    uct_iov_t *copy;

    if (iovcnt == 0) {
        return UCS_OK;
    }

    if (iov == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    copy = ucs_malloc(iovcnt * sizeof(*copy), "ucp_proto_repost_iov");
    if (copy == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    memcpy(copy, iov, iovcnt * sizeof(*copy));
    info->rma.payload.zcopy.iov = copy;
    req->send.repost.buffer     = copy;
    return UCS_OK;
}


static void ucp_proto_repost_cleanup(ucp_request_t *req)
{
    ucs_free(req->send.repost.buffer);
    ucs_free(req->send.repost.info);
    req->send.repost.buffer = NULL;
    req->send.repost.info   = NULL;
}

static void
ucp_proto_repost_assert_fields(const uct_ep_op_info_t *info UCS_V_UNUSED)
{
    switch (info->operation) {
    case UCT_EP_OP_AM_SHORT:
        ucs_assert(ucs_test_all_flags(info->field_mask,
                                      UCT_EP_OP_INFO_FIELD_OPERATION |
                                      UCT_EP_OP_INFO_FIELD_AM));
        ucs_assert(ucs_test_all_flags(info->am.field_mask,
                                      UCT_EP_OP_INFO_AM_FIELD_AM_ID |
                                      UCT_EP_OP_INFO_AM_FIELD_HEADER_VALUE |
                                      UCT_EP_OP_INFO_AM_FIELD_PAYLOAD_DATA));
        break;
    case UCT_EP_OP_AM_BCOPY:
        ucs_assert(ucs_test_all_flags(info->field_mask,
                                      UCT_EP_OP_INFO_FIELD_OPERATION |
                                      UCT_EP_OP_INFO_FIELD_AM));
        ucs_assert(ucs_test_all_flags(info->am.field_mask,
                                      UCT_EP_OP_INFO_AM_FIELD_AM_ID |
                                      UCT_EP_OP_INFO_AM_FIELD_FLAGS |
                                      UCT_EP_OP_INFO_AM_FIELD_PAYLOAD_DATA));
        break;
    case UCT_EP_OP_PUT_SHORT:
    case UCT_EP_OP_PUT_BCOPY:
        ucs_assert(ucs_test_all_flags(info->field_mask,
                                      UCT_EP_OP_INFO_FIELD_OPERATION |
                                      UCT_EP_OP_INFO_FIELD_RMA));
        ucs_assert(ucs_test_all_flags(info->rma.field_mask,
                                      UCT_EP_OP_INFO_RMA_FIELD_REMOTE_ADDR |
                                      UCT_EP_OP_INFO_RMA_FIELD_RKEY |
                                      UCT_EP_OP_INFO_RMA_FIELD_PAYLOAD_DATA));
        break;
    case UCT_EP_OP_PUT_ZCOPY:
    case UCT_EP_OP_GET_ZCOPY:
        ucs_assert(ucs_test_all_flags(info->field_mask,
                                      UCT_EP_OP_INFO_FIELD_OPERATION |
                                      UCT_EP_OP_INFO_FIELD_RMA));
        ucs_assert(ucs_test_all_flags(info->rma.field_mask,
                                      UCT_EP_OP_INFO_RMA_FIELD_REMOTE_ADDR |
                                      UCT_EP_OP_INFO_RMA_FIELD_RKEY |
                                      UCT_EP_OP_INFO_RMA_FIELD_PAYLOAD_ZCOPY));
        break;
    case UCT_EP_OP_GET_BCOPY:
        ucs_assert(ucs_test_all_flags(info->field_mask,
                                      UCT_EP_OP_INFO_FIELD_OPERATION |
                                      UCT_EP_OP_INFO_FIELD_RMA));
        ucs_assert(ucs_test_all_flags(info->rma.field_mask,
                                      UCT_EP_OP_INFO_RMA_FIELD_REMOTE_ADDR |
                                      UCT_EP_OP_INFO_RMA_FIELD_RKEY |
                                      UCT_EP_OP_INFO_RMA_FIELD_PAYLOAD_UNPACK));
        break;
    default:
        break;
    }
}

/* Descriptor and short/bcopy bytes are valid only for the purge callback. */
static ucs_status_t
ucp_proto_repost_save(ucp_request_t *req, const uct_ep_op_info_t *op_info)
{
    uct_ep_op_info_t *info;
    ucs_status_t status;

    info = ucs_malloc(sizeof(*info), "ucp_proto_repost_info");
    if (info == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    *info                   = *op_info;
    req->send.repost.info   = info;
    req->send.repost.buffer = NULL;
    ucp_proto_repost_assert_fields(info);

    switch (info->operation) {
    case UCT_EP_OP_AM_SHORT:
    case UCT_EP_OP_AM_BCOPY:
        status = ucp_proto_repost_copy_data(req, &info->am.payload.data.buffer,
                                            info->am.payload.data.length);
        break;
    case UCT_EP_OP_PUT_SHORT:
    case UCT_EP_OP_PUT_BCOPY:
        status = ucp_proto_repost_copy_data(req, &info->rma.payload.data.buffer,
                                            info->rma.payload.data.length);
        break;
    case UCT_EP_OP_PUT_ZCOPY:
    case UCT_EP_OP_GET_ZCOPY:
        status = ucp_proto_repost_copy_iov(req);
        break;
    case UCT_EP_OP_GET_BCOPY:
        status = UCS_OK;
        break;
    default:
        status = UCS_ERR_UNSUPPORTED;
        break;
    }

    return status;
}


static void ucp_proto_repost_release(ucp_request_t *req, ucs_status_t status)
{
    ucp_proto_repost_cleanup(req);
    ucp_request_complete_send(req, status);
}

static uct_completion_t *ucp_proto_repost_origin_comp(const ucp_request_t *req)
{
    const uct_ep_op_info_t *info = req->send.repost.info;

    if ((info != NULL) && (info->field_mask & UCT_EP_OP_INFO_FIELD_COMP)) {
        return info->comp;
    }

    return NULL;
}

/* Protocol v2 stores this completion in the origin request. */
static ucp_request_t *ucp_proto_repost_origin_req(const ucp_request_t *req)
{
    uct_completion_t *origin = ucp_proto_repost_origin_comp(req);
    ucp_request_t *origin_req;

    if (origin == NULL) {
        return NULL;
    }

    ucs_assert(origin->func != ucp_ep_flush_mem_completion);
    if (origin->func == ucp_proto_repost_sgl_complete) {
        return ucs_container_of(origin, ucp_proto_repost_sgl_t, comp)->origin;
    }
    origin_req = ucs_container_of(origin, ucp_request_t, send.state.uct_comp);
    ucs_assert(origin_req->flags & UCP_REQUEST_FLAG_PROTO_SEND);
    return origin_req;
}

static int ucp_proto_repost_sgl_contains(const ucp_request_t *origin,
                                         const uct_ep_op_info_t *info,
                                         size_t index, size_t offset)
{
    const ucp_datatype_iter_t *iter = &origin->send.state.dt_iter;
    const uct_iov_t *iov            = info->rma.payload.zcopy.iov;
    size_t length                   = iter->type.sgl.lengths[index];
    size_t i;

    if (offset > length) {
        return 0;
    }
    for (i = 0; i < info->rma.payload.zcopy.iovcnt; ++i) {
        if ((iov[i].count != 1) || (iov[i].length > (length - offset)) ||
            (iov[i].buffer !=
             UCS_PTR_BYTE_OFFSET(iter->type.sgl.buffers[index], offset))) {
            return 0;
        }
        offset += iov[i].length;
    }
    return 1;
}

static ucs_status_t
ucp_proto_repost_sgl_index(ucp_request_t *req, size_t *index_p)
{
    ucp_request_t *origin  = ucp_proto_repost_origin_req(req);
    uct_ep_op_info_t *info = req->send.repost.info;
    uct_completion_t *comp = ucp_proto_repost_origin_comp(req);
    ucp_proto_repost_sgl_t *batch;
    size_t index, offset, found = SIZE_MAX;
    uintptr_t address, base;
    uint64_t remote;

    if (comp->func == ucp_proto_repost_sgl_complete) {
        batch = ucs_container_of(comp, ucp_proto_repost_sgl_t, comp);
        if (!(info->field_mask & UCT_EP_OP_INFO_FIELD_SGL_INDEX) ||
            (info->sgl_index >= batch->count)) {
            return UCS_ERR_INVALID_PARAM;
        }
        index = (batch->first_index != SIZE_MAX) ?
                batch->first_index + info->sgl_index :
                batch->indices[info->sgl_index];
        if (index >= origin->send.state.dt_iter.length) {
            return UCS_ERR_INVALID_PARAM;
        }

        offset = 0;
        if (info->rma.payload.zcopy.iovcnt != 0) {
            address = (uintptr_t)info->rma.payload.zcopy.iov[0].buffer;
            base = (uintptr_t)origin->send.state.dt_iter.type.sgl.buffers[index];
            if (address < base) {
                return UCS_ERR_INVALID_PARAM;
            }
            offset = address - base;
        }
        if (!ucp_proto_repost_sgl_contains(origin, info, index, offset)) {
            return UCS_ERR_INVALID_PARAM;
        }

        /* WQE addresses can include a transport alias, e.g. for fenced PUT.
         * The UCP element and local IOV retain the logical destination. */
        info->rma.remote_addr = origin->send.rma.sgl.remote_addrs[index] +
                                offset;
        *index_p              = index;
        return UCS_OK;
    }

    /* Legacy single-fragment submissions have no batch metadata. Accept only
     * an unambiguous match; selecting the first overlapping range is unsafe. */
    for (index = 0; index < origin->send.state.dt_iter.length; ++index) {
        remote = origin->send.rma.sgl.remote_addrs[index];
        if ((info->rma.remote_addr < remote) ||
            !ucp_proto_repost_sgl_contains(origin, info, index,
                                           info->rma.remote_addr - remote)) {
            continue;
        }
        if (found != SIZE_MAX) {
            return UCS_ERR_INVALID_PARAM;
        }
        found = index;
    }
    if (found == SIZE_MAX) {
        return UCS_ERR_INVALID_PARAM;
    }
    *index_p = found;
    return UCS_OK;
}

static ucp_rkey_h ucp_proto_repost_origin_rkey(const ucp_request_t *origin)
{
    ucp_operation_id_t op_id;

    ucs_assert(origin->send.proto_config != NULL);

    op_id = ucp_proto_select_op_id(&origin->send.proto_config->select_param);
    /* A repost request stores the copied payload where rma.rkey sits. */
    ucs_assert(op_id != UCP_OP_ID_REPOST);
    if ((op_id >= UCP_OP_ID_RNDV_FIRST) && (op_id < UCP_OP_ID_RNDV_LAST)) {
        return origin->send.rndv.rkey;
    }

    return origin->send.rma.rkey;
}

static ucs_status_t
ucp_proto_repost_tl_rkey(ucp_request_t *req, ucp_lane_index_t lane,
                         uct_rkey_t *tl_rkey_p, size_t *sgl_index_p)
{
    const uct_ep_op_info_t *info = req->send.repost.info;
    ucp_request_t *origin;
    ucp_md_index_t dst_md_index;
    ucp_rkey_h rkey;
    unsigned rkey_index;
    ucs_status_t status;

    if (ucp_proto_repost_origin_comp(req) == NULL) {
        *tl_rkey_p = info->rma.rkey;
        return UCS_OK;
    }

    origin = ucp_proto_repost_origin_req(req);
    if (origin == NULL) {
        return UCS_ERR_UNREACHABLE;
    }

    if (origin->send.state.dt_iter.dt_class == UCP_DATATYPE_SGL) {
        status = ucp_proto_repost_sgl_index(req, sgl_index_p);
        if (status != UCS_OK) {
            return status;
        }
        rkey = origin->send.rma.sgl.rkeys[*sgl_index_p];
    } else {
        rkey = ucp_proto_repost_origin_rkey(origin);
    }
    if (rkey == NULL) {
        return UCS_ERR_UNREACHABLE;
    }

    dst_md_index = ucp_ep_config(req->send.ep)->key.lanes[lane].dst_md_index;
    if ((dst_md_index == UCP_NULL_RESOURCE) ||
        !(rkey->md_map & UCS_BIT(dst_md_index))) {
        return UCS_ERR_UNREACHABLE;
    }

    rkey_index = ucs_bitmap2idx(rkey->md_map, dst_md_index);
    *tl_rkey_p = ucp_rkey_get_tl_rkey(rkey, rkey_index);
    return UCS_OK;
}

/* Copied zcopy IOV still holds the failed device memory handle. */
static ucs_status_t
ucp_proto_repost_zcopy_memh(ucp_request_t *req, ucp_lane_index_t lane,
                            size_t sgl_index)
{
    uct_ep_op_info_t *op_info = req->send.repost.info;
    ucp_request_t *origin     = ucp_proto_repost_origin_req(req);
    ucp_md_index_t md_index;
    uct_mem_h uct_memh;
    ucp_mem_h memh;
    uct_iov_t *iov;
    size_t i;

    if (origin == NULL) {
        return UCS_ERR_UNREACHABLE;
    }

    if (op_info->rma.payload.zcopy.iovcnt == 0) {
        return UCS_OK;
    }

    if (origin->send.state.dt_iter.dt_class == UCP_DATATYPE_SGL) {
        memh = (origin->send.state.dt_iter.type.sgl.memhs == NULL) ? NULL :
                origin->send.state.dt_iter.type.sgl.memhs[sgl_index];
    } else if (origin->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG) {
        memh = origin->send.state.dt_iter.type.contig.memh;
    } else {
        return UCS_ERR_UNREACHABLE;
    }
    md_index = ucp_ep_md_index(req->send.ep, lane);
    if ((memh == NULL) || !(memh->md_map & UCS_BIT(md_index))) {
        return UCS_ERR_UNREACHABLE;
    }

    uct_memh = ucp_datatype_iter_uct_memh(memh, md_index);
    iov      = (uct_iov_t*)op_info->rma.payload.zcopy.iov;
    for (i = 0; i < op_info->rma.payload.zcopy.iovcnt; ++i) {
        iov[i].memh = uct_memh;
    }

    return UCS_OK;
}


static void ucp_proto_repost_completion(uct_completion_t *self)
{
    ucp_request_t *req       = ucs_container_of(self, ucp_request_t,
                                                send.state.uct_comp);
    uct_completion_t *origin = ucp_proto_repost_origin_comp(req);

    if (req->send.repost.info != NULL) {
        req->send.repost.info->comp = NULL;
    }

    if (origin != NULL) {
        ucp_invoke_uct_completion(origin, self->status);
    }

    ucp_proto_repost_release(req, self->status);
}


static void ucp_proto_repost_abort(ucp_request_t *req, ucs_status_t status)
{
    uct_completion_t *origin = ucp_proto_repost_origin_comp(req);

    /* The repost completion is installed only after a successful save, so an
     * unsupported operation must complete the purged operation directly. */
    if (origin != NULL) {
        req->send.repost.info->comp = NULL;
        ucp_invoke_uct_completion(origin, status);
    }

    ucp_proto_repost_release(req, status);
}

static ucs_status_t ucp_proto_repost_progress(uct_pending_req_t *self)
{
    ucp_request_t *req                   = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_repost_priv_t *rpriv = req->send.proto_config->priv;
    const uct_ep_op_info_t *info         = req->send.repost.info;
    uct_rkey_t tl_rkey                   = UCT_INVALID_RKEY;
    size_t sgl_index                    = 0;
    ucp_proto_repost_pack_t pack;
    uct_completion_t *comp;
    ucs_status_t status;
    uct_ep_h uct_ep;
    ssize_t packed;

    if (rpriv->lane == UCP_NULL_LANE) {
        ucp_proto_repost_abort(req, UCS_ERR_UNREACHABLE);
        return UCS_OK;
    }

    uct_ep = ucp_ep_get_lane(req->send.ep, rpriv->lane);
    if (uct_ep == NULL) {
        ucp_proto_repost_abort(req, UCS_ERR_UNREACHABLE);
        return UCS_OK;
    }

    comp = (ucp_proto_repost_origin_comp(req) != NULL) ?
            &req->send.state.uct_comp : NULL;
    ucp_proto_repost_assert_fields(info);

    if ((info->operation == UCT_EP_OP_PUT_SHORT) ||
        (info->operation == UCT_EP_OP_PUT_BCOPY) ||
        (info->operation == UCT_EP_OP_PUT_ZCOPY) ||
        (info->operation == UCT_EP_OP_GET_BCOPY) ||
        (info->operation == UCT_EP_OP_GET_ZCOPY)) {
        status = ucp_proto_repost_tl_rkey(req, rpriv->lane, &tl_rkey,
                                           &sgl_index);
        if (status != UCS_OK) {
            ucp_proto_repost_abort(req, status);
            return UCS_OK;
        }
    }

    switch (info->operation) {
    case UCT_EP_OP_AM_SHORT:
        status = uct_ep_am_short(uct_ep, info->am.am_id, info->am.header.value,
                                 info->am.payload.data.buffer,
                                 info->am.payload.data.length);
        break;
    case UCT_EP_OP_AM_BCOPY:
        pack.buffer = info->am.payload.data.buffer;
        pack.length = info->am.payload.data.length;
        packed      = uct_ep_am_bcopy(uct_ep, info->am.am_id,
                                      ucp_proto_repost_pack, &pack,
                                      info->am.flags);
        status      = ucp_proto_bcopy_send_func_status(packed);
        break;
    case UCT_EP_OP_PUT_SHORT:
        status = uct_ep_put_short(uct_ep, info->rma.payload.data.buffer,
                                  info->rma.payload.data.length,
                                  info->rma.remote_addr, tl_rkey);
        break;
    case UCT_EP_OP_PUT_BCOPY:
        pack.buffer = info->rma.payload.data.buffer;
        pack.length = info->rma.payload.data.length;
        packed      = uct_ep_put_bcopy(uct_ep, ucp_proto_repost_pack, &pack,
                                       info->rma.remote_addr, tl_rkey);
        status      = ucp_proto_bcopy_send_func_status(packed);
        break;
    case UCT_EP_OP_PUT_ZCOPY:
        status = ucp_proto_repost_zcopy_memh(req, rpriv->lane, sgl_index);
        if (status != UCS_OK) {
            break;
        }

        status = uct_ep_put_zcopy(uct_ep, info->rma.payload.zcopy.iov,
                                  info->rma.payload.zcopy.iovcnt,
                                  info->rma.remote_addr, tl_rkey, comp);
        break;
    case UCT_EP_OP_GET_BCOPY:
        status = uct_ep_get_bcopy(uct_ep, info->rma.payload.unpack.unpack_cb,
                                  info->rma.payload.unpack.arg,
                                  info->rma.payload.unpack.length,
                                  info->rma.remote_addr, tl_rkey, comp);
        break;
    case UCT_EP_OP_GET_ZCOPY:
        status = ucp_proto_repost_zcopy_memh(req, rpriv->lane, sgl_index);
        if (status != UCS_OK) {
            break;
        }

        status = uct_ep_get_zcopy(uct_ep, info->rma.payload.zcopy.iov,
                                  info->rma.payload.zcopy.iovcnt,
                                  info->rma.remote_addr, tl_rkey, comp);
        break;
    default:
        ucp_proto_repost_abort(req, UCS_ERR_UNSUPPORTED);
        return UCS_OK;
    }

    if (status == UCS_ERR_NO_RESOURCE) {
        req->send.lane = rpriv->lane;
        return UCS_ERR_NO_RESOURCE;
    }

    /* Transport invokes the request completion only after UCS_INPROGRESS. */
    if ((status == UCS_INPROGRESS) &&
        (ucp_proto_repost_origin_comp(req) != NULL)) {
        return UCS_OK;
    }

    ucp_proto_repost_abort(req, (status == UCS_INPROGRESS) ? UCS_OK : status);
    return UCS_OK;
}

/* Keep the saved operation. Restart posts this request on the new lane. */
static ucs_status_t ucp_proto_repost_reset(ucp_request_t *req)
{
    if (ucp_proto_repost_origin_comp(req) != NULL) {
        ucp_proto_completion_init(&req->send.state.uct_comp,
                                  ucp_proto_repost_completion);
    }

    return UCS_OK;
}


ucp_proto_t ucp_repost_proto = {
    .name     = "repost",
    .desc     = "repost an undelivered operation",
    .flags    = UCP_PROTO_FLAG_INVALID,
    .dt_mask  = UCP_DT_MASK_ALL,
    .probe    = ucp_proto_repost_probe,
    .query    = ucp_proto_default_query,
    .progress = {ucp_proto_repost_progress},
    .abort    = ucp_proto_repost_abort,
    .reset    = ucp_proto_repost_reset
};


static const ucp_proto_config_t *
ucp_proto_repost_config(ucp_ep_h ep)
{
    ucp_proto_select_param_t select_param;
    ucp_proto_select_elem_t *select_elem;
    const ucp_proto_threshold_elem_t *thresh;
    ucp_memory_info_t mem_info;

    ucp_memory_info_set_host(&mem_info);
    ucp_proto_select_param_init(&select_param, UCP_OP_ID_REPOST, 0, 0,
                                UCP_DATATYPE_CONTIG, &mem_info, 1);

    select_elem = ucp_proto_select_lookup_slow(
            ep->worker, &ucp_ep_config(ep)->proto_select, 0, ep->cfg_index,
            UCP_WORKER_CFG_INDEX_NULL, &select_param);
    if (select_elem == NULL) {
        return NULL;
    }

    thresh = ucp_proto_thresholds_search_slow(select_elem->thresholds, 0);
    if ((thresh == NULL) ||
        (thresh->proto_config.proto != &ucp_repost_proto)) {
        return NULL;
    }

    return &thresh->proto_config;
}

ucs_status_t
ucp_proto_repost_submit(ucp_ep_h ep, const uct_ep_op_info_t *op_info)
{
    const ucp_proto_config_t *config;
    const ucp_proto_repost_priv_t *rpriv;
    ucp_request_t *req;
    ucs_status_t status;

    /* A memory flush completion is not in the origin request. The caller
     * completes it; a later flush uses the reconfigured lanes. */
    if ((op_info->field_mask & UCT_EP_OP_INFO_FIELD_COMP) &&
        (op_info->comp != NULL) &&
        (op_info->comp->func == ucp_ep_flush_mem_completion)) {
        return UCS_ERR_CANCELED;
    }

    /* This completion is the live repost request. Saving op_info again would
     * replace the user completion with it. */
    if ((op_info->field_mask & UCT_EP_OP_INFO_FIELD_COMP) &&
        (op_info->comp != NULL) &&
        (op_info->comp->func == ucp_proto_repost_completion)) {
        req = ucs_container_of(op_info->comp, ucp_request_t,
                                              send.state.uct_comp);
        ucp_trace_req(req, "repost restart op %s",
                      uct_ep_operation_names[op_info->operation]);
        ucp_proto_request_restart(req);
        return UCS_OK;
    }

    config = ucp_proto_repost_config(ep);
    if (config == NULL) {
        return UCS_ERR_UNREACHABLE;
    }

    rpriv = config->priv;
    if ((rpriv->lane == UCP_NULL_LANE) ||
        (ucp_ep_get_lane(ep, rpriv->lane) == NULL)) {
        return UCS_ERR_UNREACHABLE;
    }

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    req->flags                     = UCP_REQUEST_FLAG_PROTO_SEND |
                                     UCP_REQUEST_FLAG_RELEASED;
    req->send.ep                   = ep;
    req->send.state.dt_iter.offset = 0;
    req->send.state.dt_iter.length = 0;
    req->send.repost.info          = NULL;
    req->send.repost.buffer        = NULL;
    ucp_proto_request_set_proto(req, config, 0);

    status = ucp_proto_repost_save(req, op_info);
    if (status != UCS_OK) {
        /* Abort frees the descriptor, so sample it before the call. */
        int stored = (req->send.repost.info != NULL);

        /* A stored descriptor makes this abort complete the origin. Return
         * success so the purge callback does not complete it again. */
        ucp_proto_repost_abort(req, status);
        return stored ? UCS_OK : status;
    }

    if (ucp_proto_repost_origin_comp(req) != NULL) {
        ucp_proto_completion_init(&req->send.state.uct_comp,
                                  ucp_proto_repost_completion);
    }

    ucp_trace_req(req, "repost submit op %s lane %u",
                  uct_ep_operation_names[op_info->operation],
                  (unsigned)rpriv->lane);
    ucp_request_send(req);
    return UCS_OK;
}
