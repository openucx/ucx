/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
 * Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "wireup.h"
#include "wireup_cm.h"
#include "address.h"

#include <ucs/algorithm/qsort_r.h>
#include <ucs/datastruct/array.h>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/sock.h>
#include <ucp/core/ucp_ep.inl>
#include <string.h>
#include <inttypes.h>

#define UCP_WIREUP_RMA_BW_TEST_MSG_SIZE    262144
#define UCP_WIREUP_MAX_FLAGS_STRING_SIZE   50
#define UCP_WIREUP_PATH_INDEX_UNDEFINED    UINT_MAX

#define UCP_WIREUP_CHECK_AMO_FLAGS(_ae, _criteria, _context, _addr_index, _op, _size)      \
    if (!ucs_test_all_flags((_ae)->iface_attr.atomic.atomic##_size._op##_flags,            \
                            (_criteria)->remote_atomic_flags.atomic##_size._op##_flags)) { \
        char desc[256];                                                                    \
        ucs_trace("addr[%d] %s: no %s", (_addr_index),                                     \
                  ucp_find_tl_name_by_csum((_context), (_ae)->tl_name_csum),               \
                  ucp_wireup_get_missing_amo_flag_desc_##_op(                              \
                      (_ae)->iface_attr.atomic.atomic##_size._op##_flags,                  \
                      (_criteria)->remote_atomic_flags.atomic##_size._op##_flags,          \
                      (_size), desc, sizeof(desc)));                                       \
        continue;                                                                          \
    }

typedef struct ucp_wireup_atomic_flag {
    const char *name;
    const char *fetch;
} ucp_wireup_atomic_flag_t;


typedef struct {
    ucp_rsc_index_t      rsc_index;
    unsigned             addr_index;
    unsigned             path_index;
    ucp_md_index_t       dst_md_index;
    ucs_sys_device_t     dst_sys_dev;
    ucp_lane_type_mask_t lane_types;
    size_t               seg_size;
    double               score[UCP_LANE_TYPE_LAST];
} ucp_wireup_lane_desc_t;


typedef struct {
    ucp_wireup_criteria_t criteria;
    uint64_t              local_dev_bitmap;
    uint64_t              remote_dev_bitmap;
    ucp_md_map_t          md_map;
    unsigned              max_lanes;
} ucp_wireup_select_bw_info_t;


typedef struct {
    unsigned local[UCP_MAX_RESOURCES];
    unsigned remote[UCP_MAX_RESOURCES];
    unsigned local_skip[UCP_MAX_RESOURCES];
    unsigned remote_skip[UCP_MAX_RESOURCES];
} ucp_wireup_dev_usage_count;


/**
 * Global parameters for lanes selection during UCP wireup procedure
 */
typedef struct {
    ucp_ep_h                      ep;               /* UCP Endpoint */
    unsigned                      ep_init_flags;    /* Endpoint init flags */
    ucp_tl_bitmap_t               tl_bitmap;        /* TLs bitmap which can be selected */
    const ucp_unpacked_address_t  *address;         /* Remote addresses */
    int                           allow_am;         /* Shows whether emulation over AM
                                                     * is allowed or not for RMA/AMO */
    int                           show_error;       /* Global flag that controls showing
                                                     * errors from a selecting transport
                                                     * procedure */
} ucp_wireup_select_params_t;

/**
 * Context for lanes selection during UCP wireup procedure
 */
typedef struct {
    ucp_wireup_lane_desc_t    lane_descs[UCP_MAX_LANES]; /* Array of active lanes that are
                                                          * found during selection */
    ucp_lane_index_t          num_lanes;                 /* Number of active lanes */
    unsigned                  ucp_ep_init_flags;         /* Endpoint init extra flags */
    ucp_tl_bitmap_t           tl_bitmap;                 /* TL bitmap of selected resources */
} ucp_wireup_select_context_t;

UCS_ARRAY_DECLARE_TYPE(ucp_proto_select_info_array_t, unsigned,
                       ucp_wireup_select_info_t);

static const char *ucp_wireup_cmpt_flags[] = {
    [ucs_ilog2(UCT_COMPONENT_FLAG_RKEY_PTR)]     = "obtain remote memory pointer",
};

static const char *ucp_wireup_md_flags[] = {
    [ucs_ilog2(UCT_MD_FLAG_ALLOC)]               = "memory allocation",
    [ucs_ilog2(UCT_MD_FLAG_REG)]                 = "memory registration",
    [ucs_ilog2(UCT_MD_FLAG_INVALIDATE)]          = "memory invalidation",
    [ucs_ilog2(UCT_MD_FLAG_INVALIDATE_RMA)]      = "RMA memory invalidation"
};

static const char *ucp_wireup_iface_flags[] = {
    [ucs_ilog2(UCT_IFACE_FLAG_AM_SHORT)]         = "am short",
    [ucs_ilog2(UCT_IFACE_FLAG_AM_BCOPY)]         = "am bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_AM_ZCOPY)]         = "am zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_PUT_SHORT)]        = "put short",
    [ucs_ilog2(UCT_IFACE_FLAG_PUT_BCOPY)]        = "put bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_PUT_ZCOPY)]        = "put zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_GET_SHORT)]        = "get short",
    [ucs_ilog2(UCT_IFACE_FLAG_GET_BCOPY)]        = "get bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_GET_ZCOPY)]        = "get zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE)] = "peer failure handler",
    [ucs_ilog2(UCT_IFACE_FLAG_CONNECT_TO_IFACE)] = "connect to iface",
    [ucs_ilog2(UCT_IFACE_FLAG_CONNECT_TO_EP)]    = "connect to ep",
    [ucs_ilog2(UCT_IFACE_FLAG_AM_DUP)]           = "full reliability",
    [ucs_ilog2(UCT_IFACE_FLAG_CB_SYNC)]          = "sync callback",
    [ucs_ilog2(UCT_IFACE_FLAG_CB_ASYNC)]         = "async callback",
    [ucs_ilog2(UCT_IFACE_FLAG_PENDING)]          = "pending",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_SHORT)]  = "tag eager short",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_BCOPY)]  = "tag eager bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY)]  = "tag eager zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY)]   = "tag rndv zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_EP_CHECK)]         = "ep check",
    [ucs_ilog2(UCT_IFACE_FLAG_EP_KEEPALIVE)]     = "ep keepalive"
};

static const char *ucp_wireup_event_flags[] = {
    [ucs_ilog2(UCT_IFACE_FLAG_EVENT_SEND_COMP)] = "send completion event",
    [ucs_ilog2(UCT_IFACE_FLAG_EVENT_RECV)]      = "tag or active message event",
    [ucs_ilog2(UCT_IFACE_FLAG_EVENT_RECV_SIG)]  = "signaled message event"
};

static const char *ucp_wireup_peer_flags[] = {
    [ucs_ilog2(UCP_ADDR_IFACE_FLAG_CONNECT_TO_IFACE)] = "connect to iface",
    [ucs_ilog2(UCP_ADDR_IFACE_FLAG_AM_SYNC)]          = "am sync callback",
    [ucs_ilog2(UCP_ADDR_IFACE_FLAG_CB_ASYNC)]         = "async callback",
    [ucs_ilog2(UCP_ADDR_IFACE_FLAG_PUT)]              = "put",
    [ucs_ilog2(UCP_ADDR_IFACE_FLAG_GET)]              = "get",
    [ucs_ilog2(UCP_ADDR_IFACE_FLAG_TAG_EAGER)]        = "tag_eager",
    [ucs_ilog2(UCP_ADDR_IFACE_FLAG_TAG_RNDV)]         = "tag_rndv",
    [ucs_ilog2(UCP_ADDR_IFACE_FLAG_EVENT_RECV)]       = "tag_am_recv_event"
};

static ucp_wireup_atomic_flag_t ucp_wireup_atomic_desc[] = {
     [UCT_ATOMIC_OP_ADD]   = {.name = "add",   .fetch = "fetch-"},
     [UCT_ATOMIC_OP_AND]   = {.name = "and",   .fetch = "fetch-"},
     [UCT_ATOMIC_OP_OR]    = {.name = "or",    .fetch = "fetch-"},
     [UCT_ATOMIC_OP_XOR]   = {.name = "xor",   .fetch = "fetch-"},
     [UCT_ATOMIC_OP_SWAP]  = {.name = "swap",  .fetch = ""},
     [UCT_ATOMIC_OP_CSWAP] = {.name = "cswap", .fetch = ""}
};


static const char *
ucp_wireup_get_missing_flag_desc(uint64_t flags, uint64_t required_flags,
                                 const char ** flag_descs)
{
    ucs_assert((required_flags & (~flags)) != 0);
    return flag_descs[ucs_ffs64(required_flags & (~flags))];
}

static const char *
ucp_wireup_get_missing_amo_flag_desc(uint64_t flags, uint64_t required_flags,
                                     int op_size, int fetch, char *buf, size_t len)
{
    int idx;

    ucs_assert((required_flags & (~flags)) != 0);

    idx = ucs_ffs64(required_flags & (~flags));

    snprintf(buf, len, "%d-bit atomic %s%s", op_size,
             fetch ? ucp_wireup_atomic_desc[idx].fetch : "",
             ucp_wireup_atomic_desc[idx].name);

    return buf;
}

static const char *
ucp_wireup_get_missing_amo_flag_desc_op(uint64_t flags, uint64_t required_flags,
                                        int op_size, char *buf, size_t len)
{
    return ucp_wireup_get_missing_amo_flag_desc(flags, required_flags, op_size, 0, buf, len);
}

static const char *
ucp_wireup_get_missing_amo_flag_desc_fop(uint64_t flags, uint64_t required_flags,
                                         int op_size, char *buf, size_t len)
{
    return ucp_wireup_get_missing_amo_flag_desc(flags, required_flags, op_size, 1, buf, len);
}

static void ucp_wireup_init_select_flags(ucp_wireup_select_flags_t *flags,
                                         uint64_t mandatory, uint64_t optional)
{
    flags->mandatory = mandatory;
    flags->optional  = optional;
}

static int
ucp_wireup_test_select_flags(const ucp_wireup_select_flags_t *select_flags,
                             uint64_t flags, const char **flag_descs,
                             ucs_string_buffer_t *missing_flags_str)
{
    /* Check all mandatory flags are set */
    if (!ucs_test_all_flags(flags, select_flags->mandatory)) {
        const char *desc = ucp_wireup_get_missing_flag_desc(
                flags, select_flags->mandatory, flag_descs);

        /* Format message with the first missing flag */
        ucs_string_buffer_appendf(missing_flags_str, "no %s", desc);
        return 0;
    }

    /* Check if any optional flags are set */
    if (select_flags->optional && !(select_flags->optional & flags)) {
        /* Format message with a list of missing flags */
        ucs_string_buffer_appendf(missing_flags_str, "no ");
        ucs_string_buffer_append_flags(missing_flags_str,
                                       select_flags->optional, flag_descs);
        return 0;
    }

    return 1;
}

static int
ucp_wireup_check_select_flags(const uct_tl_resource_desc_t *resource,
                              uint64_t flags,
                              const ucp_wireup_select_flags_t *select_flags,
                              const char *title, const char **flag_descs,
                              char *reason, size_t max)
{
    UCS_STRING_BUFFER_ONSTACK(missing_flags_str,
                              UCP_WIREUP_MAX_FLAGS_STRING_SIZE);

    if (!ucp_wireup_test_select_flags(select_flags, flags, flag_descs,
                                      &missing_flags_str)) {
        ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : not suitable for %s, %s",
                  UCT_TL_RESOURCE_DESC_ARG(resource), title,
                  ucs_string_buffer_cstr(&missing_flags_str));

        ucs_snprintf_safe(reason, max, UCT_TL_RESOURCE_DESC_FMT " - %s",
                          UCT_TL_RESOURCE_DESC_ARG(resource),
                          ucs_string_buffer_cstr(&missing_flags_str));
        return 0;
    }

    return 1;
}

static int ucp_wireup_check_flags(const uct_tl_resource_desc_t *resource,
                                  uint64_t flags, uint64_t select_flags,
                                  const char *title, const char **flag_descs,
                                  char *reason, size_t max)
{
    ucp_wireup_select_flags_t req;

    ucp_wireup_init_select_flags(&req, select_flags, 0);
    return ucp_wireup_check_select_flags(resource, flags, &req, title,
                                         flag_descs, reason, max);
}

static int ucp_wireup_check_amo_flags(const uct_tl_resource_desc_t *resource,
                                      uint64_t flags, uint64_t required_flags,
                                      int op_size, int fetch,
                                      const char *title, char *reason,
                                      size_t max)
{
    char missing_flag_desc[256];

    if (ucs_test_all_flags(flags, required_flags)) {
        return 1;
    }

    if (required_flags) {
        ucp_wireup_get_missing_amo_flag_desc(flags, required_flags,
                                             op_size, fetch, missing_flag_desc,
                                             sizeof(missing_flag_desc));
        ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : not suitable for %s, no %s",
                  UCT_TL_RESOURCE_DESC_ARG(resource), title,
                  missing_flag_desc);
        snprintf(reason, max, UCT_TL_RESOURCE_DESC_FMT" - no %s",
                 UCT_TL_RESOURCE_DESC_ARG(resource), missing_flag_desc);
    }
    return 0;
}

static int
ucp_wireup_check_keepalive(const ucp_wireup_select_params_t *select_params,
                           ucp_rsc_index_t rsc_index, uint64_t flags,
                           const char *title, int is_keepalive,
                           const char **flag_descs, char *reason, size_t max)
{
    ucp_worker_h worker                    = select_params->ep->worker;
    ucp_context_h context                  = worker->context;
    const uct_tl_resource_desc_t *resource = &context->tl_rscs[rsc_index].tl_rsc;
    char title_keepalive[128];
    char title_ep_check[128];
    char title_am_based[128];

    if (!is_keepalive) {
        /* Keepalive is not needed */
        return 1;
    }

    ucs_snprintf_safe(title_keepalive, sizeof(title_keepalive),
                      "%s with keepalive", title);
    ucs_snprintf_safe(title_ep_check, sizeof(title_ep_check),
                      "%s with ep_check", title);
    ucs_snprintf_safe(title_am_based, sizeof(title_am_based),
                      "%s with am-based keepalive", title);

    return /* Either built-in keepalive (i.e. UCT_IFACE_FLAG_EP_KEEPALIVE) or
            * EP checking (i.e. UCT_IFACE_FLAG_EP_CHECK) with CONNECT_TO_EP or
            * CONNECT_TO_IFACE and AM_BCOPY for AM-based keepalive */
            ucp_wireup_check_flags(resource, flags,
                                   UCT_IFACE_FLAG_EP_KEEPALIVE |
                                   UCT_IFACE_FLAG_CONNECT_TO_EP,
                                   title_keepalive, ucp_wireup_iface_flags,
                                   reason, max) ||
            ucp_wireup_check_flags(resource, flags,
                                   UCT_IFACE_FLAG_EP_CHECK |
                                   UCT_IFACE_FLAG_CONNECT_TO_EP,
                                   title_ep_check, ucp_wireup_iface_flags,
                                   reason, max) ||
            ucp_wireup_check_flags(resource, flags,
                                   UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                   UCT_IFACE_FLAG_AM_BCOPY,
                                   title_am_based, ucp_wireup_iface_flags,
                                   reason, max);
}

static void
ucp_wireup_init_select_info(double score, unsigned addr_index,
                            ucp_rsc_index_t rsc_index,
                            uint8_t priority,
                            ucp_wireup_select_info_t *select_info)
{
    /* score == 0.0 could be specified only when initializing a selection info
     * to add CM lane (rsc_index == UCP_NULL_RESOURCE in this case) */
    ucs_assert((score >= 0.0) || (rsc_index == UCP_NULL_RESOURCE));

    select_info->score      = score;
    select_info->addr_index = addr_index;
    select_info->path_index = UCP_WIREUP_PATH_INDEX_UNDEFINED;
    select_info->rsc_index  = rsc_index;
    select_info->priority   = priority;
}

static size_t
ucp_wireup_bw_max_lanes(const ucp_wireup_select_params_t *select_params)
{
    return (select_params->address->dst_version < 18) ? UCP_MAX_LANES_LEGACY :
                                                        UCP_MAX_LANES;
}

static size_t
ucp_wireup_max_lanes(const ucp_wireup_select_params_t *select_params,
                     ucp_lane_type_t lane_type)
{
    return ucp_wireup_lane_type_is_fast_path(lane_type) ?
                   UCP_MAX_FAST_PATH_LANES :
                   ucp_wireup_bw_max_lanes(select_params);
}

/**
 * Select a local and remote transport
 */
static UCS_F_NOINLINE ucs_status_t ucp_wireup_select_transport(
        const ucp_wireup_select_context_t *select_ctx,
        const ucp_wireup_select_params_t *select_params,
        const ucp_wireup_criteria_t *criteria, ucp_tl_bitmap_t tl_bitmap,
        uint64_t remote_md_map, uint64_t local_dev_bitmap,
        uint64_t remote_dev_bitmap, int show_error,
        ucp_wireup_select_info_t *select_info, char *info_str,
        size_t info_str_size)
{
    UCS_STRING_BUFFER_ONSTACK(missing_flags_str,
                              UCP_WIREUP_MAX_FLAGS_STRING_SIZE);
    const ucp_unpacked_address_t *address         = select_params->address;
    ucp_ep_h ep                                   = select_params->ep;
    ucp_worker_h worker                           = ep->worker;
    ucp_context_h context                         = worker->context;
    ucp_wireup_select_info_t sinfo                = {0};
    int found                                     = 0;
    ucp_wireup_select_flags_t local_iface_flags = criteria->local_iface_flags;
    int has_cm;
    uint64_t local_md_flags;
    ucp_tl_addr_bitmap_t addr_index_map, rsc_addr_index_map;
    const ucp_wireup_lane_desc_t *lane_desc;
    unsigned addr_index;
    uct_tl_resource_desc_t *resource;
    const ucp_address_entry_t *ae;
    ucp_worker_iface_t *wiface;
    ucp_rsc_index_t rsc_index;
    ucp_rsc_index_t dev_index;
    ucp_lane_index_t lane;
    char tls_info[256];
    char *p, *endp;
    uct_iface_attr_t *iface_attr;
    uct_md_attr_v2_t *md_attr;
    const uct_component_attr_t *cmpt_attr;
    int is_reachable;
    double score;
    uint8_t priority;
    ucp_md_index_t md_index;

    p            = tls_info;
    endp         = tls_info + sizeof(tls_info) - 1;
    tls_info[0]  = '\0';
    UCS_STATIC_BITMAP_AND_INPLACE(&tl_bitmap, select_params->tl_bitmap);
    UCS_STATIC_BITMAP_AND_INPLACE(&tl_bitmap, context->tl_bitmap);
    show_error   = (select_params->show_error && show_error);

    /* Check which remote addresses satisfy the criteria */
    UCS_STATIC_BITMAP_RESET_ALL(&addr_index_map);
    ucp_unpacked_address_for_each(ae, address) {
        addr_index = ucp_unpacked_address_index(address, ae);
        if (!(remote_dev_bitmap & UCS_BIT(ae->dev_index))) {
            ucs_trace("addr[%d]: not in use, because on device[%d]",
                      addr_index, ae->dev_index);
            continue;
        } else if ((ae->md_index != UCP_NULL_RESOURCE) &&
                   !(remote_md_map & UCS_BIT(ae->md_index))) {
            ucs_trace("addr[%d]: not in use, because on md[%d]", addr_index,
                      ae->md_index);
            continue;
        }

        /* Make sure we are indeed passing all flags required by the criteria in
         * ucp packed address */
        ucs_assert(ucs_test_all_flags(UCP_ADDRESS_IFACE_EVENT_FLAGS,
                                      criteria->remote_event_flags));

        ucs_string_buffer_reset(&missing_flags_str);
        if (!ucp_wireup_test_select_flags(&criteria->remote_iface_flags,
                                          ae->iface_attr.flags,
                                          ucp_wireup_peer_flags,
                                          &missing_flags_str)) {
            ucs_trace("addr[%d] %s: %s", addr_index,
                      ucp_find_tl_name_by_csum(context, ae->tl_name_csum),
                      ucs_string_buffer_cstr(&missing_flags_str));
            continue;
        }

        if (!ucs_test_all_flags(ae->iface_attr.flags, criteria->remote_event_flags)) {
            ucs_trace("addr[%d] %s: no %s", addr_index,
                      ucp_find_tl_name_by_csum(context, ae->tl_name_csum),
                      ucp_wireup_get_missing_flag_desc(ae->iface_attr.flags,
                                                       criteria->remote_event_flags,
                                                       ucp_wireup_peer_flags));
            continue;
        }

        UCP_WIREUP_CHECK_AMO_FLAGS(ae, criteria, context, addr_index, op, 32);
        UCP_WIREUP_CHECK_AMO_FLAGS(ae, criteria, context, addr_index, op, 64);
        UCP_WIREUP_CHECK_AMO_FLAGS(ae, criteria, context, addr_index, fop, 32);
        UCP_WIREUP_CHECK_AMO_FLAGS(ae, criteria, context, addr_index, fop, 64);

        UCS_STATIC_BITMAP_SET(&addr_index_map, addr_index);
    }

    if (UCS_STATIC_BITMAP_IS_ZERO(addr_index_map)) {
        snprintf(p, endp - p, "%s  ", ucs_status_string(UCS_ERR_UNSUPPORTED));
        p += strlen(p);
        goto out;
    }

    /* For each local resource try to find the best remote address to connect to.
     * Pick the best local resource to satisfy the criteria.
     * best one has the highest score (from the dedicated score_func) and
     * has a reachable tl on the remote peer */
    UCS_STATIC_BITMAP_FOR_EACH_BIT(rsc_index, &tl_bitmap) {
        local_md_flags = criteria->local_md_flags;
        resource       = &context->tl_rscs[rsc_index].tl_rsc;
        dev_index      = context->tl_rscs[rsc_index].dev_index;
        wiface         = ucp_worker_iface(worker, rsc_index);
        iface_attr     = ucp_worker_iface_get_attr(worker, rsc_index);
        md_index       = context->tl_rscs[rsc_index].md_index;
        md_attr        = &context->tl_mds[md_index].attr;
        cmpt_attr      = ucp_cmpt_attr_by_md_index(context, md_index);

        if ((context->tl_rscs[rsc_index].flags & UCP_TL_RSC_FLAG_AUX) &&
            !(criteria->tl_rsc_flags & UCP_TL_RSC_FLAG_AUX)) {
            continue;
        }

        has_cm = ucp_ep_init_flags_has_cm(select_params->ep_init_flags);
        if (select_params->ep_init_flags & UCP_EP_INIT_CONNECT_TO_IFACE_ONLY) {
            local_iface_flags.mandatory |= UCT_IFACE_FLAG_CONNECT_TO_IFACE;
        } else if (ucp_wireup_connect_p2p(worker, rsc_index, has_cm)) {
            /* We should not need MD invalidate support for p2p lanes, since
             * both sides close the connection in case of error */
            local_md_flags &= ~(UCT_MD_FLAG_INVALIDATE     |
                                UCT_MD_FLAG_INVALIDATE_RMA |
                                UCT_MD_FLAG_INVALIDATE_AMO);
        }

        /* Check that local md and interface satisfy the criteria */
        if (!ucp_wireup_check_flags(resource, md_attr->flags, local_md_flags,
                                    criteria->title, ucp_wireup_md_flags, p,
                                    endp - p) ||
            !ucp_wireup_check_flags(resource, cmpt_attr->flags,
                                    criteria->local_cmpt_flags, criteria->title,
                                    ucp_wireup_cmpt_flags, p, endp - p) ||
            !ucp_wireup_check_flags(resource, md_attr->alloc_mem_types,
                                    criteria->alloc_mem_types, criteria->title,
                                    ucs_memory_type_names, p, endp - p) ||
            !ucp_wireup_check_flags(resource, md_attr->reg_mem_types,
                                    criteria->reg_mem_types, criteria->title,
                                    ucs_memory_type_names, p, endp - p) ||
            !ucp_wireup_check_select_flags(resource, iface_attr->cap.flags,
                                           &local_iface_flags, criteria->title,
                                           ucp_wireup_iface_flags, p,
                                           endp - p) ||
            !ucp_wireup_check_keepalive(select_params, rsc_index,
                                        iface_attr->cap.flags,
                                        criteria->title,
                                        criteria->is_keepalive,
                                        ucp_wireup_iface_flags, p,
                                        endp - p) ||
            !ucp_wireup_check_flags(resource, iface_attr->cap.event_flags,
                                    criteria->local_event_flags, criteria->title,
                                    ucp_wireup_event_flags, p, endp - p) ||
            !ucp_wireup_check_amo_flags(resource, iface_attr->cap.atomic32.op_flags,
                                        criteria->local_atomic_flags.atomic32.op_flags,
                                        32, 0, criteria->title, p, endp - p) ||
            !ucp_wireup_check_amo_flags(resource, iface_attr->cap.atomic64.op_flags,
                                        criteria->local_atomic_flags.atomic64.op_flags,
                                        64, 0, criteria->title, p, endp - p) ||
            !ucp_wireup_check_amo_flags(resource, iface_attr->cap.atomic32.fop_flags,
                                        criteria->local_atomic_flags.atomic32.fop_flags,
                                        32, 1, criteria->title, p, endp - p) ||
            !ucp_wireup_check_amo_flags(resource, iface_attr->cap.atomic64.fop_flags,
                                        criteria->local_atomic_flags.atomic64.fop_flags,
                                        64, 1, criteria->title, p, endp - p))
        {
            p += strlen(p);
            snprintf(p, endp - p, ", ");
            p += strlen(p);
            continue;
        }

        /* Check supplied tl & device bitmap */
        if (!UCS_STATIC_BITMAP_GET(tl_bitmap, rsc_index)) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : disabled by tl_bitmap",
                      UCT_TL_RESOURCE_DESC_ARG(resource));
            snprintf(p, endp - p, UCT_TL_RESOURCE_DESC_FMT" - disabled for %s, ",
                     UCT_TL_RESOURCE_DESC_ARG(resource), criteria->title);
            p += strlen(p);
            continue;
        } else if (!(local_dev_bitmap & UCS_BIT(dev_index))) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : disabled by device bitmap",
                      UCT_TL_RESOURCE_DESC_ARG(resource));
            snprintf(p, endp - p, UCT_TL_RESOURCE_DESC_FMT" - disabled for %s, ",
                     UCT_TL_RESOURCE_DESC_ARG(resource), criteria->title);
            p += strlen(p);
            continue;
        }

        if (select_ctx->num_lanes <
            ucp_wireup_max_lanes(select_params, criteria->lane_type)) {
            /* If we have not reached the lanes limit, we can select any
               combination of rsc_index/addr_index */
            rsc_addr_index_map = addr_index_map;
        } else {
            /* If we reached the lanes limit, select only existing combinations
             * of rsc_index/addr_index, to make sure lane selection result will
             * be the same when connecting to worker address and when connecting
             * to a remote ep by wireup protocol.
             */
            UCS_STATIC_BITMAP_RESET_ALL(&rsc_addr_index_map);
            for (lane = 0; lane < select_ctx->num_lanes; ++lane) {
                lane_desc = &select_ctx->lane_descs[lane];
                if (lane_desc->rsc_index == rsc_index) {
                    UCS_STATIC_BITMAP_SET(&rsc_addr_index_map,
                                          lane_desc->addr_index);
                }
            }
            UCS_STATIC_BITMAP_AND_INPLACE(&rsc_addr_index_map, addr_index_map);
        }

        is_reachable = 0;

        UCS_STATIC_BITMAP_FOR_EACH_BIT(addr_index, &rsc_addr_index_map) {
            ae = &address->address_list[addr_index];
            if (!ucp_wireup_is_reachable(ep, select_params->ep_init_flags,
                                         rsc_index, ae, info_str,
                                         info_str_size)) {
                /* Must be reachable device address, on same transport */
                continue;
            }

            score        = criteria->calc_score(wiface, md_attr, address, ae,
                                                criteria->arg);
            priority     = iface_attr->priority + ae->iface_attr.priority;
            is_reachable = 1;

            ucs_trace(UCT_TL_RESOURCE_DESC_FMT
                      "->addr[%u] : %s score %.2f priority %d",
                      UCT_TL_RESOURCE_DESC_ARG(resource),
                      addr_index, criteria->title, score, priority);

            if (!found || (ucp_score_prio_cmp(score, priority, sinfo.score,
                                              sinfo.priority) > 0)) {
                ucp_wireup_init_select_info(score, addr_index, rsc_index,
                                            priority, &sinfo);
                found = 1;
            }
        }

        /* If a local resource cannot reach any of the remote addresses,
         * generate debug message. */
        if (!is_reachable) {
            snprintf(p, endp - p, UCT_TL_RESOURCE_DESC_FMT" - %s, ",
                     UCT_TL_RESOURCE_DESC_ARG(resource),
                     ucs_status_string(UCS_ERR_UNREACHABLE));
            p += strlen(p);
        }
    }

out:
    if (p >= tls_info + 2) {
        *(p - 2) = '\0'; /* trim last "," */
    }

    if (!found) {
        if (show_error) {
            ucs_error("no %s transport to %s: %s", criteria->title,
                      address->name, tls_info);
        }

        return UCS_ERR_UNREACHABLE;
    }

    ucs_trace("ep %p: selected for %s: " UCT_TL_RESOURCE_DESC_FMT " md[%d]"
              " -> '%s' address[%d],md[%d] score %.2f",
              ep, criteria->title,
              UCT_TL_RESOURCE_DESC_ARG(
                      &context->tl_rscs[sinfo.rsc_index].tl_rsc),
              context->tl_rscs[sinfo.rsc_index].md_index, ucp_ep_peer_name(ep),
              sinfo.addr_index,
              address->address_list[sinfo.addr_index].md_index, sinfo.score);

    *select_info = sinfo;
    return UCS_OK;
}

static inline double
ucp_wireup_tl_iface_latency(const ucp_worker_iface_t *wiface,
                            const ucp_unpacked_address_t *unpacked_addr,
                            const ucp_address_iface_attr_t *remote_iface_attr)
{
    ucp_context_h context = wiface->worker->context;
    double local_lat, lat_lossy;

    if (unpacked_addr->addr_version == UCP_OBJECT_VERSION_V1) {
        local_lat = ucp_wireup_iface_lat_distance_v1(wiface);
        /* Address v1 contains just latency overhead */
        return ((local_lat + remote_iface_attr->lat_ovh) / 2) +
               (wiface->attr.latency.m * context->config.est_num_eps);
    } else {
        local_lat = ucp_wireup_iface_lat_distance_v2(wiface);
        /* FP8 is a lossy compression method, so in order to create a symmetric
         * calculation we pack/unpack the local latency as well */
        lat_lossy = ucp_wireup_fp8_pack_unpack_latency(local_lat);

        return (remote_iface_attr->lat_ovh + lat_lossy) / 2;
    }
}

static int ucp_wireup_has_slow_lanes(ucp_wireup_select_context_t *select_ctx)
{
    ucp_wireup_lane_desc_t *lane_desc;

    ucs_carray_for_each(lane_desc, select_ctx->lane_descs,
                        select_ctx->num_lanes) {
        if (!ucp_wireup_lane_types_has_fast_path(lane_desc->lane_types)) {
            return 1;
        }
    }

    return 0;
}

static int
ucp_wireup_path_index_is_equal(unsigned path_index1, unsigned path_index2)
{
    return (path_index1 == UCP_WIREUP_PATH_INDEX_UNDEFINED) ||
           (path_index2 == UCP_WIREUP_PATH_INDEX_UNDEFINED) ||
           (path_index1 == path_index2);
}

static UCS_F_NOINLINE ucs_status_t ucp_wireup_add_lane_desc(
        const ucp_wireup_select_params_t *select_params,
        const ucp_wireup_select_info_t *select_info,
        ucp_md_index_t dst_md_index, ucs_sys_device_t dst_sys_dev,
        ucp_lane_type_t lane_type, unsigned seg_size,
        ucp_wireup_select_context_t *select_ctx, int show_error)
{
    ucp_wireup_lane_desc_t *lane_desc;
    ucp_lane_type_t lane_type_iter;
    ucp_lane_index_t lane;
    ucs_log_level_t log_level;

    /* Add a new lane, but try to reuse already added lanes which are selected
     * on the same transport resources.
     */
    for (lane_desc = select_ctx->lane_descs;
         lane_desc < select_ctx->lane_descs + select_ctx->num_lanes; ++lane_desc) {
        if ((lane_desc->rsc_index == select_info->rsc_index) &&
            (lane_desc->addr_index == select_info->addr_index) &&
            ucp_wireup_path_index_is_equal(lane_desc->path_index,
                                           select_info->path_index)) {

            lane = lane_desc - select_ctx->lane_descs;
            ucs_assertv_always(dst_md_index == lane_desc->dst_md_index,
                               "lane[%d].dst_md_index=%d, dst_md_index=%d",
                               lane, lane_desc->dst_md_index, dst_md_index);

            /* The same pair of local/remote resource is already selected but
             * with a different usage - update the score and path (if needed)
             * and exit
             */
            if (!(lane_desc->lane_types & UCS_BIT(lane_type))) {
                goto out_update;
            }

            /* If adding same lane type and usage, expect same score */
            ucs_assertv_always(
                    ucp_score_cmp(lane_desc->score[lane_type],
                                  select_info->score) == 0,
                    "usage=%s lane_desc->score=%.2f select->score=%.2f",
                    ucp_lane_type_info[lane_type].short_name,
                    lane_desc->score[lane_type], select_info->score);
            goto out;
        }
    }

    /* We rely on 'ucp_wireup_search_lanes' to add fast lanes before slow
     * lanes, so that all fast lanes are inserted to the cached ucp_ep
     * structure */
    if (ucp_wireup_lane_type_is_fast_path(lane_type)) {
        /* assert we don't have slow lanes until we finished with fast lanes */
        ucs_assert_always(!ucp_wireup_has_slow_lanes(select_ctx));
    }

    if (select_ctx->num_lanes >=
        ucp_wireup_max_lanes(select_params, lane_type)) {
        log_level = show_error ? UCS_LOG_LEVEL_ERROR : UCS_LOG_LEVEL_DEBUG;
        ucs_log(log_level, "cannot add %s lane - reached limit (%d)",
                ucp_lane_type_info[lane_type].short_name,
                select_ctx->num_lanes);
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    lane_desc = &select_ctx->lane_descs[select_ctx->num_lanes];
    ++select_ctx->num_lanes;

    lane_desc->rsc_index    = select_info->rsc_index;
    lane_desc->addr_index   = select_info->addr_index;
    lane_desc->path_index   = select_info->path_index;
    lane_desc->dst_md_index = dst_md_index;
    lane_desc->dst_sys_dev  = dst_sys_dev;
    lane_desc->lane_types   = UCS_BIT(lane_type);
    lane_desc->seg_size     = seg_size;
    for (lane_type_iter = UCP_LANE_TYPE_FIRST;
         lane_type_iter < UCP_LANE_TYPE_LAST;
         ++lane_type_iter) {
        lane_desc->score[lane_type_iter] = 0.0;
    }

    if (select_info->rsc_index != UCP_NULL_RESOURCE) {
        UCS_STATIC_BITMAP_SET(&select_ctx->tl_bitmap, select_info->rsc_index);
    }

out_update:
    if (lane_desc->path_index == UCP_WIREUP_PATH_INDEX_UNDEFINED) {
        lane_desc->path_index = select_info->path_index;
    }

    lane_desc->score[lane_type] = select_info->score;
    lane_desc->lane_types      |= UCS_BIT(lane_type);
out:
    return UCS_OK;
}

static UCS_F_NOINLINE ucs_status_t
ucp_wireup_add_lane(const ucp_wireup_select_params_t *select_params,
                    const ucp_wireup_select_info_t *select_info,
                    ucp_lane_type_t lane_type, int show_error,
                    ucp_wireup_select_context_t *select_ctx)
{
    ucp_address_entry_t *addr_list = select_params->address->address_list;
    unsigned addr_index            = select_info->addr_index;

    return ucp_wireup_add_lane_desc(select_params, select_info,
                                    addr_list[addr_index].md_index,
                                    addr_list[addr_index].sys_dev, lane_type,
                                    addr_list[addr_index].iface_attr.seg_size,
                                    select_ctx,
                                    select_params->show_error && show_error);
}

static int ucp_wireup_compare_score(const void *elem1, const void *elem2,
                                    void *arg, ucp_lane_type_t lane_type)
{
    const ucp_lane_index_t *lane1       = elem1;
    const ucp_lane_index_t *lane2       = elem2;
    const ucp_wireup_lane_desc_t *lanes = arg;
    double score1, score2;

    score1 = (*lane1 == UCP_NULL_LANE) ? 0.0 : lanes[*lane1].score[lane_type];
    score2 = (*lane2 == UCP_NULL_LANE) ? 0.0 : lanes[*lane2].score[lane_type];

    /* sort from highest score to lowest */
    return (score1 < score2) ? 1 : ((score1 > score2) ? -1 : 0);
}

static int ucp_wireup_compare_lane_am_bw_score(const void *elem1, const void *elem2,
                                               void *arg)
{
    return ucp_wireup_compare_score(elem1, elem2, arg, UCP_LANE_TYPE_AM_BW);
}

static int ucp_wireup_compare_lane_rma_score(const void *elem1, const void *elem2,
                                             void *arg)
{
    return ucp_wireup_compare_score(elem1, elem2, arg, UCP_LANE_TYPE_RMA);
}

static int ucp_wireup_compare_lane_rma_bw_score(const void *elem1, const void *elem2,
                                                void *arg)
{
    return ucp_wireup_compare_score(elem1, elem2, arg, UCP_LANE_TYPE_RMA_BW);
}

static int ucp_wireup_compare_lane_amo_score(const void *elem1, const void *elem2,
                                             void *arg)
{
    return ucp_wireup_compare_score(elem1, elem2, arg, UCP_LANE_TYPE_AMO);
}

static void ucp_wireup_unset_tl_by_md(const ucp_wireup_select_params_t *sparams,
                                      const ucp_wireup_select_info_t *sinfo,
                                      ucp_tl_bitmap_t *tl_bitmap,
                                      uint64_t *remote_md_map)
{
    ucp_context_h context         = sparams->ep->worker->context;
    const ucp_address_entry_t *ae = &sparams->address->
                                        address_list[sinfo->addr_index];
    ucp_md_index_t md_index       = context->tl_rscs[sinfo->rsc_index].md_index;
    ucp_md_index_t dst_md_index   = ae->md_index;
    ucp_rsc_index_t i;

    *remote_md_map &= ~UCS_BIT(dst_md_index);

    UCS_STATIC_BITMAP_FOR_EACH_BIT(i, &context->tl_bitmap) {
        if (context->tl_rscs[i].md_index == md_index) {
            UCS_STATIC_BITMAP_RESET(tl_bitmap, i);
        }
    }
}

static UCS_F_NOINLINE ucs_status_t ucp_wireup_add_memaccess_lanes(
        const ucp_wireup_select_params_t *select_params, unsigned ep_init_flags,
        const ucp_wireup_criteria_t *criteria, ucs_memory_type_t mem_type,
        ucp_tl_bitmap_t tl_bitmap, ucp_lane_type_t lane_type,
        ucp_wireup_select_context_t *select_ctx)
{
    ucp_context_h context                = select_params->ep->worker->context;
    ucp_wireup_criteria_t mem_criteria   = *criteria;
    ucp_wireup_select_info_t select_info = {0};
    double reg_score                     = 0;
    int allow_am;
    uint64_t remote_md_map;
    ucs_status_t status;
    char title[64];

    allow_am      = select_params->allow_am &&
                    !((lane_type == UCP_LANE_TYPE_RMA) &&
                      (context->config.features & UCP_FEATURE_EXPORTED_MEMH));
    remote_md_map = UINT64_MAX;

    /* Select best transport which can reach registered memory */
    snprintf(title, sizeof(title), criteria->title, "registered");
    mem_criteria.title           = title;
    mem_criteria.local_md_flags  = UCT_MD_FLAG_REG | criteria->local_md_flags;
    mem_criteria.alloc_mem_types = 0;
    mem_criteria.reg_mem_types   = UCS_BIT(mem_type);
    mem_criteria.lane_type       = lane_type;

    status = ucp_wireup_select_transport(select_ctx, select_params,
                                         &mem_criteria, tl_bitmap,
                                         remote_md_map, UINT64_MAX, UINT64_MAX,
                                         !allow_am, &select_info, NULL, 0);
    if (status == UCS_OK) {
        /* Add to the list of lanes */
        status = ucp_wireup_add_lane(select_params, &select_info, lane_type,
                                     !allow_am, select_ctx);
        if (status == UCS_OK) {
            /* Remove all occurrences of the remote md from the address list,
             * to avoid selecting the same remote md again. */
            ucp_wireup_unset_tl_by_md(select_params, &select_info, &tl_bitmap,
                                      &remote_md_map);
            reg_score = select_info.score;
        }
    }

    /* If could not find registered memory access lane, try to use emulation */
    if (status != UCS_OK) {
        if (!allow_am) {
            return status;
        }

        select_ctx->ucp_ep_init_flags |= UCP_EP_INIT_CREATE_AM_LANE;
    }

    if (!(ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE) &&
        (mem_type != UCS_MEMORY_TYPE_HOST)) {
        /* Select transports for allocated memory only for host mem, to keep
           wire compatibility of lane selection */
        return UCS_OK;
    }

    /* Select additional transports which can access allocated memory, but
     * only if their scores are better. We need this because a remote memory
     * block can be potentially allocated using one of them, and we might get
     * better performance than the transports which support only registered
     * remote memory. */
    snprintf(title, sizeof(title), criteria->title, "allocated");
    mem_criteria.title           = title;
    mem_criteria.local_md_flags  = UCT_MD_FLAG_ALLOC | criteria->local_md_flags;
    mem_criteria.alloc_mem_types = UCS_BIT(mem_type);
    mem_criteria.reg_mem_types   = 0;
    mem_criteria.lane_type       = lane_type;

    for (;;) {
        status = ucp_wireup_select_transport(select_ctx, select_params,
                                             &mem_criteria, tl_bitmap,
                                             remote_md_map, UINT64_MAX,
                                             UINT64_MAX, 0, &select_info,
                                             NULL, 0);
        /* Break if: */
        /* - transport selection wasn't OK */
        if ((status != UCS_OK) ||
            /* - the selected transport is worse than
             *   the transport selected above */
            (ucp_score_cmp(select_info.score, reg_score) <= 0)) {
            break;
        }

        /* Add lane description and remove all occurrences of the remote md. */
        status = ucp_wireup_add_lane(select_params, &select_info, lane_type,
                                    /* do not show error */ 0, select_ctx);
        if (status != UCS_OK) {
            break;
        }

        ucp_wireup_unset_tl_by_md(select_params, &select_info, &tl_bitmap,
                                  &remote_md_map);
    }

    return UCS_OK;
}

static uint64_t ucp_ep_get_context_features(const ucp_ep_h ep)
{
    return ep->worker->context->config.features;
}

static double ucp_wireup_rma_score_func(const ucp_worker_iface_t *wiface,
                                        const uct_md_attr_v2_t *md_attr,
                                        const ucp_unpacked_address_t *unpacked_addr,
                                        const ucp_address_entry_t *remote_addr,
                                        void *arg)
{
    /* best for 4k messages */
    double local_bw;

    if (unpacked_addr->dst_version < 17) {
        local_bw = ucp_tl_iface_bandwidth(wiface->worker->context,
                                          &wiface->attr.bandwidth);
    } else {
        local_bw = ucp_wireup_iface_bw_distance(wiface);
    }

    return 1e-3 /
           (ucp_wireup_tl_iface_latency(
                wiface, unpacked_addr, &remote_addr->iface_attr) +
            wiface->attr.overhead +
            (4096.0 / ucs_min(local_bw, remote_addr->iface_attr.bandwidth)));
}

static void ucp_wireup_fill_peer_err_criteria(ucp_wireup_criteria_t *criteria,
                                              unsigned ep_init_flags)
{
    if (ep_init_flags & UCP_EP_INIT_ERR_MODE_PEER_FAILURE) {
        criteria->local_iface_flags.mandatory |=
                UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
        /* transport selection procedure will check additionally for KA or EP check
         * support */
    }
}

static double ucp_wireup_aux_score_func(const ucp_worker_iface_t *wiface,
                                        const uct_md_attr_v2_t *md_attr,
                                        const ucp_unpacked_address_t *unpacked_addr,
                                        const ucp_address_entry_t *remote_addr,
                                        void *arg)
{
    /* best end-to-end latency and larger bcopy size */
    return (1e-3 /
            (ucp_wireup_tl_iface_latency(
                wiface, unpacked_addr,  &remote_addr->iface_attr) +
             wiface->attr.overhead + remote_addr->iface_attr.overhead));
}

static void ucp_wireup_fill_aux_criteria(ucp_wireup_criteria_t *criteria,
                                         unsigned ep_init_flags,
                                         uint64_t mandatory_flags)
{
    criteria->title          = "auxiliary";
    criteria->local_md_flags = 0;
    ucp_wireup_init_select_flags(&criteria->local_iface_flags,
                                 UCT_IFACE_FLAG_AM_BCOPY |
                                 UCT_IFACE_FLAG_PENDING, 0);
    ucp_wireup_init_select_flags(&criteria->remote_iface_flags,
                                 UCP_ADDR_IFACE_FLAG_AM_SYNC, 0);

    /* CM lane doesn't require to use CONNECT_TO_IFACE for auxiliary lane */
    if (!ucp_ep_init_flags_has_cm(ep_init_flags)) {
        criteria->local_iface_flags.mandatory  |=
                UCT_IFACE_FLAG_CONNECT_TO_IFACE;
        criteria->remote_iface_flags.mandatory |=
                UCP_ADDR_IFACE_FLAG_CONNECT_TO_IFACE | mandatory_flags;
    }
    criteria->local_cmpt_flags   = 0;
    criteria->local_event_flags  = 0;
    criteria->remote_event_flags = 0;
    criteria->calc_score         = ucp_wireup_aux_score_func;
    criteria->tl_rsc_flags       = UCP_TL_RSC_FLAG_AUX; /* Can use aux transports */
    criteria->lane_type          = UCP_LANE_TYPE_LAST;

    ucp_wireup_fill_peer_err_criteria(criteria, ep_init_flags);
}

static void ucp_wireup_criteria_init(ucp_wireup_criteria_t *criteria)
{
    criteria->title              = "";
    criteria->local_md_flags     = 0;
    criteria->local_cmpt_flags   = 0;
    criteria->local_event_flags  = 0;
    criteria->remote_event_flags = 0;
    criteria->alloc_mem_types    = 0;
    criteria->reg_mem_types      = 0;
    criteria->is_keepalive       = 0;
    criteria->calc_score         = NULL;
    criteria->tl_rsc_flags       = 0;
    ucp_wireup_init_select_flags(&criteria->local_iface_flags, 0, 0);
    ucp_wireup_init_select_flags(&criteria->remote_iface_flags, 0, 0);
    memset(&criteria->remote_atomic_flags, 0,
           sizeof(criteria->remote_atomic_flags));
    memset(&criteria->local_atomic_flags, 0,
           sizeof(criteria->local_atomic_flags));
}

/**
 * Check whether emulation over AM is allowed for RMA/AMO lanes
 */
static int ucp_wireup_allow_am_emulation_layer(unsigned ep_init_flags)
{
    return !(ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE);
}

static unsigned
ucp_wireup_ep_init_flags(const ucp_wireup_select_params_t *select_params,
                         const ucp_wireup_select_context_t *select_ctx)
{
    return select_params->ep_init_flags | select_ctx->ucp_ep_init_flags;
}

static ucs_status_t
ucp_wireup_add_cm_lane(const ucp_wireup_select_params_t *select_params,
                       ucp_wireup_select_context_t *select_ctx)
{
    ucp_wireup_select_info_t select_info;

    if (!ucp_ep_init_flags_has_cm(select_params->ep_init_flags)) {
        return UCS_OK;
    }

    ucp_wireup_init_select_info(0., UINT_MAX, UCP_NULL_RESOURCE, 0,
                                &select_info);

    /* server is not a proxy because it can create all lanes connected */
    return ucp_wireup_add_lane_desc(select_params, &select_info,
                                    UCP_NULL_RESOURCE,
                                    UCS_SYS_DEVICE_ID_UNKNOWN, UCP_LANE_TYPE_CM,
                                    UINT_MAX, select_ctx, 1);
}

static ucs_status_t
ucp_wireup_add_rma_lanes(const ucp_wireup_select_params_t *select_params,
                         ucp_wireup_select_context_t *select_ctx)
{
    unsigned ep_init_flags         = ucp_wireup_ep_init_flags(select_params,
                                                              select_ctx);
    ucp_wireup_criteria_t criteria = {};
    ucp_tl_bitmap_t tl_bitmap;
    ucs_memory_type_t mem_type;
    ucs_status_t status;

    if ((!(ucp_ep_get_context_features(select_params->ep) & UCP_FEATURE_RMA) &&
         !(ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE)) ||
        (ep_init_flags & UCP_EP_INIT_CREATE_AM_LANE_ONLY)) {
        return UCS_OK;
    }

    ucp_wireup_criteria_init(&criteria);
    if (ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE) {
        criteria.title              = "copy across memory types";
        ucp_wireup_init_select_flags(&criteria.local_iface_flags,
                                     UCT_IFACE_FLAG_PUT_SHORT, 0);
        ucp_wireup_init_select_flags(&criteria.remote_iface_flags,
                                     UCP_ADDR_IFACE_FLAG_PUT, 0);
    } else {
        criteria.title              = "remote %s memory access";
        ucp_wireup_init_select_flags(&criteria.remote_iface_flags,
                                     UCP_ADDR_IFACE_FLAG_PUT |
                                     UCP_ADDR_IFACE_FLAG_GET, 0);
        ucp_wireup_init_select_flags(&criteria.local_iface_flags,
                                     UCT_IFACE_FLAG_PUT_SHORT |
                                     UCT_IFACE_FLAG_PUT_BCOPY |
                                     UCT_IFACE_FLAG_GET_BCOPY |
                                     UCT_IFACE_FLAG_PENDING, 0);
    }
    criteria.calc_score             = ucp_wireup_rma_score_func;
    ucp_wireup_fill_peer_err_criteria(&criteria, ep_init_flags);

    tl_bitmap = ucp_tl_bitmap_max;
    ucs_memory_type_for_each(mem_type) {
        status = ucp_wireup_add_memaccess_lanes(select_params, ep_init_flags,
                                                &criteria, mem_type, tl_bitmap,
                                                UCP_LANE_TYPE_RMA, select_ctx);
        if ((status != UCS_OK) && (mem_type == UCS_MEMORY_TYPE_HOST)) {
            return status;
        }
    }

    return UCS_OK;
}

double ucp_wireup_amo_score_func(const ucp_worker_iface_t *wiface,
                                 const uct_md_attr_v2_t *md_attr,
                                 const ucp_unpacked_address_t *unpacked_addr,
                                 const ucp_address_entry_t *remote_addr,
                                 void *arg)
{
    /* best one-sided latency */
    return 1e-3 /
           (ucp_wireup_tl_iface_latency(
                wiface, unpacked_addr, &remote_addr->iface_attr) +
            wiface->attr.overhead);
}

static ucs_status_t
ucp_wireup_add_amo_lanes(const ucp_wireup_select_params_t *select_params,
                         ucp_wireup_select_context_t *select_ctx)
{
    ucp_worker_h worker            = select_params->ep->worker;
    ucp_context_h context          = worker->context;
    unsigned ep_init_flags         = ucp_wireup_ep_init_flags(select_params,
                                                              select_ctx);
    ucp_wireup_criteria_t criteria = {};
    ucp_rsc_index_t rsc_index;
    ucp_tl_bitmap_t tl_bitmap;

    if (!ucs_test_flags(context->config.features, UCP_FEATURE_AMO32,
                        UCP_FEATURE_AMO64) ||
        (ep_init_flags & (UCP_EP_INIT_FLAG_MEM_TYPE |
                          UCP_EP_INIT_CREATE_AM_LANE_ONLY))) {
        return UCS_OK;
    }

    ucp_wireup_criteria_init(&criteria);
    criteria.title              = "atomic operations on %s memory";
    criteria.local_atomic_flags = criteria.remote_atomic_flags;
    criteria.calc_score         = ucp_wireup_amo_score_func;
    ucp_wireup_init_select_flags(&criteria.local_iface_flags,
                                 UCT_IFACE_FLAG_PENDING, 0);
    ucp_wireup_fill_peer_err_criteria(&criteria, ep_init_flags);
    ucp_context_uct_atomic_iface_flags(context, &criteria.remote_atomic_flags);

    /* We can use only non-p2p resources or resources which are explicitly
     * selected for atomics. Otherwise, the remote peer would not be able to
     * connect back on p2p transport.
     */
    tl_bitmap = worker->atomic_tls;
    UCS_STATIC_BITMAP_FOR_EACH_BIT(rsc_index, &context->tl_bitmap) {
        if (ucp_worker_is_tl_2iface(worker, rsc_index)) {
            UCS_STATIC_BITMAP_SET(&tl_bitmap, rsc_index);
        }
    }

    return ucp_wireup_add_memaccess_lanes(select_params, ep_init_flags,
                                          &criteria, UCS_MEMORY_TYPE_HOST,
                                          tl_bitmap, UCP_LANE_TYPE_AMO,
                                          select_ctx);
}

static double
ucp_wireup_am_score_func(const ucp_worker_iface_t *wiface,
                         const uct_md_attr_v2_t *md_attr,
                         const ucp_unpacked_address_t *unpacked_addr,
                         const ucp_address_entry_t *remote_addr, void *arg)
{
    /* best end-to-end latency */
    return 1e-3 /
           (ucp_wireup_tl_iface_latency(
                wiface, unpacked_addr, &remote_addr->iface_attr) +
            wiface->attr.overhead + remote_addr->iface_attr.overhead);
}

static double ucp_tl_iface_bandwidth_ratio(ucp_context_h context,
                                           unsigned path_index,
                                           unsigned num_paths)
{
    double ratio;

    if (UCS_CONFIG_DBL_IS_AUTO(context->config.ext.multi_path_ratio)) {
        ratio = path_index / (double)num_paths;
    } else {
        ratio = context->config.ext.multi_path_ratio * path_index;
    }

    return ucs_max(1e-5, 1.0 - ratio);
}

static double
ucp_wireup_iface_avail_bandwidth(const ucp_worker_iface_t *wiface,
                                 const ucp_unpacked_address_t *unpacked_addr,
                                 const ucp_address_entry_t *remote_addr,
                                 const ucp_wireup_dev_usage_count *dev_count)
{
    ucp_context_h context     = wiface->worker->context;
    ucp_rsc_index_t dev_index = context->tl_rscs[wiface->rsc_index].dev_index;
    double eps                = 1e-3;
    double local_bw, remote_bw;
    unsigned path_index;

    local_bw = ucp_wireup_iface_bw_distance(wiface);

    if (unpacked_addr->addr_version == UCP_OBJECT_VERSION_V2) {
        /* FP8 is a lossy compression method, so in order to create a symmetric
         * calculation we pack/unpack the local bandwidth as well */
        local_bw = UCS_FP8_PACK_UNPACK(BANDWIDTH, local_bw);
    }

    ucs_assertv(dev_count->local[dev_index] >= dev_count->local_skip[dev_index],
                "dev_count->local[%u]=%u dev_count->local_skip[%u]=%u",
                dev_index, dev_count->local[dev_index], dev_index,
                dev_count->local_skip[dev_index]);
    ucs_assertv(dev_count->remote[remote_addr->dev_index] >=
                        dev_count->remote_skip[remote_addr->dev_index],
                "dev_count->remote[%u]=%u dev_count->remote_skip[%u]=%u",
                remote_addr->dev_index,
                dev_count->remote[remote_addr->dev_index],
                remote_addr->dev_index,
                dev_count->remote_skip[remote_addr->dev_index]);

    /* Apply dev num paths ratio after fp8 pack/unpack to make sure it is not
     * neglected because of fp8 inaccuracy
     */
    path_index = dev_count->local[dev_index] - dev_count->local_skip[dev_index];
    local_bw  *= ucp_tl_iface_bandwidth_ratio(context, path_index,
                                              wiface->attr.dev_num_paths);

    path_index = dev_count->remote[remote_addr->dev_index] -
                 dev_count->remote_skip[remote_addr->dev_index];
    remote_bw  = remote_addr->iface_attr.bandwidth *
                 ucp_tl_iface_bandwidth_ratio(context, path_index,
                                              remote_addr->dev_num_paths);

    return ucs_min(local_bw, remote_bw) + (eps * (local_bw + remote_bw));
}

static ucs_linear_func_t
ucp_wireup_mem_reg_cost(ucp_context_t *context,
                        const ucp_unpacked_address_t *remote_addr,
                        const uct_md_attr_v2_t *md_attr)
{
    if (context->rcache == NULL) {
        return md_attr->reg_cost;
    }

    if (remote_addr->dst_version > UCP_RELEASE_LEGACY) {
        return ucs_linear_func_make(UCP_RCACHE_OVERHEAD_DEFAULT, 0);
    }

    /* This is needed to preserve wire-compatibility support with
     * pre UCX v1.16 versions, where rcache was used by UCT IB and KNEM MDs.
     * The cost of rcache lookup was set to 180ns there (and reported as
     * md_attr->reg_cost).
     * TODO: remove when wire compatibility is not longer supported with
     * those versions.
     */
    return ucs_linear_func_make(180e-9, 0);
}

static double
ucp_wireup_rma_bw_score_func(const ucp_worker_iface_t *wiface,
                             const uct_md_attr_v2_t *md_attr,
                             const ucp_unpacked_address_t *unpacked_addr,
                             const ucp_address_entry_t *remote_addr, void *arg)
{
    ucp_wireup_dev_usage_count *dev_count = arg;
    ucp_context_t *context                = wiface->worker->context;
    ucs_linear_func_t mem_reg_cost;

    mem_reg_cost = ucp_wireup_mem_reg_cost(context, unpacked_addr, md_attr);

    /* highest bandwidth with lowest overhead - test a message size of 256KB,
     * a size which is likely to be used for high-bw memory access protocol, for
     * how long it would take to transfer it with a certain transport. */
    return 1 / ((UCP_WIREUP_RMA_BW_TEST_MSG_SIZE /
                 ucp_wireup_iface_avail_bandwidth(wiface, unpacked_addr,
                                                  remote_addr, dev_count)) +
                ucp_wireup_tl_iface_latency(wiface, unpacked_addr,
                                            &remote_addr->iface_attr) +
                wiface->attr.overhead +
                ucs_linear_func_apply(mem_reg_cost,
                                      UCP_WIREUP_RMA_BW_TEST_MSG_SIZE));
}

static inline int
ucp_wireup_is_am_required(const ucp_wireup_select_params_t *select_params,
                          const ucp_wireup_select_context_t *select_ctx)
{
    ucp_ep_h ep            = select_params->ep;
    ucp_context_h context  = ep->worker->context;
    unsigned ep_init_flags = ucp_wireup_ep_init_flags(select_params,
                                                      select_ctx);
    ucp_lane_index_t lane;

    /* Check if we need active messages from the configurations, for wireup.
     * If not, check if am is required due to p2p transports */

    if (ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE) {
        /* Memtype ep needs only RMA lanes */
        return 0;
    }

    if (ep_init_flags & UCP_EP_INIT_CREATE_AM_LANE) {
        return 1;
    }

    if (ucp_ep_get_context_features(ep) & (UCP_FEATURE_TAG |
                                           UCP_FEATURE_STREAM |
                                           UCP_FEATURE_AM)) {
        return 1;
    }

    /* For new protocols, we need to have active message lane to handle data
     * transfers by emulation protocol, for memory types which are not supported
     * natively.
     * TODO Try to select RMA lanes for all combinations of memory types, and if
     * some are not available, require AM lane.
     */
    if (context->config.ext.proto_enable &&
        (context->num_mem_type_detect_mds > 0) &&
        (ucp_ep_get_context_features(ep) & UCP_FEATURE_RMA)) {
        return 1;
    }

    for (lane = 0; lane < select_ctx->num_lanes; ++lane) {
        if (!ucp_worker_is_tl_2iface(ep->worker,
                                     select_ctx->lane_descs[lane].rsc_index)) {
            return 1;
        }
    }

    return 0;
}

static ucs_status_t
ucp_wireup_add_am_lane(const ucp_wireup_select_params_t *select_params,
                       ucp_wireup_select_info_t *am_info,
                       ucp_wireup_select_context_t *select_ctx,
                       char *info_string, size_t info_string_length)
{
    ucp_worker_h worker            = select_params->ep->worker;
    ucp_tl_bitmap_t tl_bitmap      = select_params->tl_bitmap;
    unsigned ep_init_flags         = ucp_wireup_ep_init_flags(select_params,
                                                              select_ctx);
    ucp_wireup_criteria_t criteria = {};
    const uct_iface_attr_t *iface_attr;
    ucs_status_t status;

    if (!ucp_wireup_is_am_required(select_params, select_ctx)) {
        memset(am_info, 0, sizeof(*am_info));
        return UCS_OK;
    }

    /* Select one lane for active messages */
    for (;;) {
        ucp_wireup_criteria_init(&criteria);
        criteria.title              = "active messages";
        criteria.calc_score         = ucp_wireup_am_score_func;
        criteria.lane_type          = UCP_LANE_TYPE_AM;
        criteria.tl_rsc_flags       =
                (ep_init_flags & UCP_EP_INIT_ALLOW_AM_AUX_TL) ?
                UCP_TL_RSC_FLAG_AUX : 0;
        ucp_wireup_init_select_flags(&criteria.local_iface_flags,
                                     UCT_IFACE_FLAG_AM_BCOPY, 0);
        ucp_wireup_init_select_flags(&criteria.remote_iface_flags,
                                     UCP_ADDR_IFACE_FLAG_AM_SYNC, 0);
        ucp_wireup_fill_peer_err_criteria(&criteria,
                                          ucp_wireup_ep_init_flags(select_params,
                                                                   select_ctx));

        if (ucs_test_all_flags(ucp_ep_get_context_features(select_params->ep),
                               UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP)) {
            criteria.local_event_flags = UCP_WIREUP_UCT_EVENT_CAP_FLAGS;
        }

        status = ucp_wireup_select_transport(select_ctx, select_params,
                                             &criteria, tl_bitmap, UINT64_MAX,
                                             UINT64_MAX, UINT64_MAX, 1,
                                             am_info, info_string,
                                             info_string_length);
        if (status != UCS_OK) {
            return status;
        }

        /* If max_bcopy is too small, try again */
        iface_attr = ucp_worker_iface_get_attr(worker, am_info->rsc_index);
        if (iface_attr->cap.am.max_bcopy < UCP_MIN_BCOPY) {
            ucs_debug("ep %p: rsc_index[%d] am.max_bcopy is too small: %zu, "
                      "expected: >= %d", select_params->ep, am_info->rsc_index,
                      iface_attr->cap.am.max_bcopy, UCP_MIN_BCOPY);
            UCS_STATIC_BITMAP_RESET(&tl_bitmap, am_info->rsc_index);
            continue;
        }

        return ucp_wireup_add_lane(select_params, am_info, UCP_LANE_TYPE_AM,
                                   /* show error */ 1, select_ctx);
    }
}

static double
ucp_wireup_am_bw_score_func(const ucp_worker_iface_t *wiface,
                            const uct_md_attr_v2_t *md_attr,
                            const ucp_unpacked_address_t *unpacked_addr,
                            const ucp_address_entry_t *remote_addr, void *arg)
{
    ucp_wireup_dev_usage_count *dev_count = arg;

    /* Best single MTU bandwidth, take into account remote segment size, which
     * can be smaller than the local value (supported with worker address v2)
     */
    double size = ucs_min(wiface->attr.cap.am.max_bcopy,
                          remote_addr->iface_attr.seg_size);
    double t    = (size / ucp_wireup_iface_avail_bandwidth(
                            wiface, unpacked_addr, remote_addr, dev_count)) +
                  wiface->attr.overhead + remote_addr->iface_attr.overhead +
                  ucp_wireup_tl_iface_latency(wiface, unpacked_addr,
                                              &remote_addr->iface_attr);

    return size / t * 1e-5;
}

static double ucp_wireup_get_lane_bw(ucp_worker_h worker,
                                     const ucp_wireup_select_info_t *sinfo,
                                     const ucp_unpacked_address_t *address)
{
    ucp_worker_iface_t *wiface = ucp_worker_iface(worker, sinfo->rsc_index);
    double bw_local, bw_remote;

    if (address->dst_version < 17) {
        bw_local = ucp_tl_iface_bandwidth(worker->context,
                                          &wiface->attr.bandwidth);
    } else {
        /* Compare BW including local distance to prevent EP reconfiguration
         * since remote distance is included to remote_addr*/
        bw_local = ucp_wireup_iface_bw_distance(wiface);
    }

    bw_remote  = address->address_list[sinfo->addr_index].iface_attr.bandwidth;

    if (address->addr_version == UCP_OBJECT_VERSION_V2) {
        /* FP8 is a lossy compression method, so in order to create a symmetric
         * calculation we pack/unpack the local bandwidth as well */
        bw_local = UCS_FP8_PACK_UNPACK(BANDWIDTH, bw_local);
    }

    return ucs_min(bw_local, bw_remote);
}

static unsigned
ucp_wireup_add_fast_lanes(ucp_worker_h worker,
                          const ucp_wireup_select_params_t *select_params,
                          const ucp_proto_select_info_array_t *sinfo_array,
                          ucp_lane_type_t lane_type,
                          ucp_wireup_select_context_t *select_ctx)
{
    ucp_lane_index_t num_lanes = 0;
    double max_bw              = 0;
    ucp_context_h context      = worker->context;
    const double max_ratio     = 1. / context->config.ext.multi_lane_max_ratio;
    ucs_status_t status;
    double lane_bw;
    const ucp_wireup_select_info_t *sinfo;

    /* Iterate over all elements and calculate max BW */
    ucs_array_for_each(sinfo, sinfo_array) {
        lane_bw = ucp_wireup_get_lane_bw(worker, sinfo, select_params->address);
        max_bw  = ucs_max(lane_bw, max_bw);
    }

    /* Compare each element to max BW and filter only fast lanes */
    ucs_array_for_each(sinfo, sinfo_array) {
        lane_bw = ucp_wireup_get_lane_bw(worker, sinfo, select_params->address);

        if (lane_bw < (max_bw * max_ratio)) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT
                      " : bandwidth %.2f lower than %.2f x %.2f, dropping lane",
                      UCT_TL_RESOURCE_DESC_ARG(
                              &context->tl_rscs[sinfo->rsc_index].tl_rsc),
                      lane_bw, max_ratio, max_bw);
            continue;
        }

        status = ucp_wireup_add_lane(select_params, sinfo, lane_type,
                                     num_lanes == 0, select_ctx);
        if (status != UCS_OK) {
            break;
        }

        num_lanes++;
    }

    return num_lanes;
}

static unsigned ucp_wireup_get_current_num_lanes(
        const ucp_wireup_select_params_t *select_params, ucp_lane_type_t type)
{
    ucp_ep_h ep                = select_params->ep;
    unsigned current_num_lanes = 0;
    ucp_lane_index_t lane;

    /* First initialization (current lanes weren't chosen yet) or
     * CM is enabled (so we can reconfigure the endpoint).
     */
    if ((ep->cfg_index == UCP_WORKER_CFG_INDEX_NULL) ||
        ucp_ep_has_cm_lane(ep)) {
        return ucp_wireup_bw_max_lanes(select_params);
    }

    for (lane = 0; lane < ucp_ep_config(ep)->key.num_lanes; ++lane) {
        if (ucp_ep_config(ep)->key.lanes[lane].lane_types & UCS_BIT(type)) {
            ++current_num_lanes;
        }
    }
    return current_num_lanes;
}

static unsigned
ucp_wireup_add_bw_lanes(const ucp_wireup_select_params_t *select_params,
                        ucp_wireup_select_bw_info_t *bw_info,
                        ucp_tl_bitmap_t tl_bitmap, ucp_lane_index_t excl_lane,
                        ucp_wireup_select_context_t *select_ctx,
                        unsigned allow_extra_path)
{
    ucp_proto_select_info_array_t sinfo_array = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucp_rsc_index_t skip_dev_index            = UCP_NULL_RESOURCE;
    ucp_ep_h ep                               = select_params->ep;
    ucp_context_h context                     = ep->worker->context;
    ucp_wireup_dev_usage_count dev_count      = {};
    const uct_iface_attr_t *iface_attr;
    const ucp_address_entry_t *ae;
    ucs_status_t status;
    uint64_t local_dev_bitmap;
    uint64_t remote_dev_bitmap;
    ucp_rsc_index_t dev_index;
    ucp_rsc_index_t rsc_index;
    unsigned addr_index;
    ucp_wireup_select_info_t *sinfo;
    unsigned max_lanes, num_lanes;
    unsigned local_num_paths, remote_num_paths;

    local_dev_bitmap      = bw_info->local_dev_bitmap;
    remote_dev_bitmap     = bw_info->remote_dev_bitmap;
    bw_info->criteria.arg = &dev_count;

    /* Restrict choosing more lanes that were already chosen when CM is disabled
     * to prevent EP reconfiguration in case of dropping lanes due to low BW.
     * TODO: Remove when endpoint reconfiguration is supported.
     */
    max_lanes = ucp_wireup_get_current_num_lanes(select_params,
                                                 bw_info->criteria.lane_type);
    max_lanes = ucs_min(max_lanes, bw_info->max_lanes);

    /* lookup for requested number of lanes or limit of MD map
     * (we have to limit MD's number to avoid malloc in
     * memory registration) */
    while (ucs_array_length(&sinfo_array) < max_lanes) {
        if (excl_lane == UCP_NULL_LANE) {
            sinfo  = ucs_array_append(&sinfo_array, break);
            status = ucp_wireup_select_transport(select_ctx, select_params,
                                                 &bw_info->criteria, tl_bitmap,
                                                 UINT64_MAX, local_dev_bitmap,
                                                 remote_dev_bitmap, 0, sinfo,
                                                 NULL, 0);
            if (status != UCS_OK) {
                ucs_array_pop_back(&sinfo_array);
                break;
            }

            rsc_index         = sinfo->rsc_index;
            addr_index        = sinfo->addr_index;
            dev_index         = context->tl_rscs[rsc_index].dev_index;
            sinfo->path_index = dev_count.local[dev_index];

        } else {
            /* disqualify/count lane_desc_idx */
            addr_index      = select_ctx->lane_descs[excl_lane].addr_index;
            rsc_index       = select_ctx->lane_descs[excl_lane].rsc_index;
            dev_index       = context->tl_rscs[rsc_index].dev_index;
            excl_lane       = UCP_NULL_LANE;
            skip_dev_index  = dev_index;
        }

        /* Count how many times the LOCAL device is used */
        iface_attr = ucp_worker_iface_get_attr(ep->worker, rsc_index);
        ++dev_count.local[dev_index];

        /* Count how many times the REMOTE device is used */
        ae = &select_params->address->address_list[addr_index];
        ++dev_count.remote[ae->dev_index];

        /* Account for possible path override */
        local_num_paths  = iface_attr->dev_num_paths;
        remote_num_paths = ae->dev_num_paths;
        if (allow_extra_path &&
            ((skip_dev_index != UCP_NULL_RESOURCE) && /* clang sanitizer */
             (skip_dev_index == dev_index))) {
            /* Allow path since we skipped one */
            local_num_paths++;
            remote_num_paths++;

            /* Remove that skipped path from score computation */
            dev_count.local_skip[dev_index]      = 1;
            dev_count.remote_skip[ae->dev_index] = 1;
        }

        if (dev_count.local[dev_index] >= local_num_paths) {
            /* exclude local device if reached max concurrency level */
            local_dev_bitmap &= ~UCS_BIT(dev_index);
        }

        /* Assume symmetric num_paths */
        if (dev_count.remote[ae->dev_index] >= remote_num_paths) {
            /* exclude remote device if reached max concurrency level */
            remote_dev_bitmap &= ~UCS_BIT(ae->dev_index);
        }
    }

    bw_info->criteria.arg = NULL; /* To suppress compiler warning */
    num_lanes = ucp_wireup_add_fast_lanes(ep->worker, select_params,
                                          &sinfo_array,
                                          bw_info->criteria.lane_type,
                                          select_ctx);

    ucs_array_cleanup_dynamic(&sinfo_array);
    return num_lanes;
}

static ucs_status_t
ucp_wireup_add_am_bw_lanes(const ucp_wireup_select_params_t *select_params,
                           ucp_wireup_select_context_t *select_ctx)
{
    ucp_ep_h ep            = select_params->ep;
    ucp_context_h context  = ep->worker->context;
    unsigned ep_init_flags = ucp_wireup_ep_init_flags(select_params,
                                                      select_ctx);
    ucp_lane_index_t lane_desc_idx, am_lane;
    ucp_wireup_select_bw_info_t bw_info;
    unsigned num_am_bw_lanes;

    /* Check if we need active message BW lanes */
    if (!(ucp_ep_get_context_features(ep) &
          (UCP_FEATURE_TAG | UCP_FEATURE_AM)) ||
        (ep_init_flags & (UCP_EP_INIT_FLAG_MEM_TYPE |
                          UCP_EP_INIT_CREATE_AM_LANE_ONLY)) ||
        /* TODO: Check MAX_RNDV_LANES here for rndv/am case */
        (context->config.ext.max_eager_lanes < 2)) {
        return UCS_OK;
    }

    /* Select one lane for active messages */
    ucp_wireup_criteria_init(&bw_info.criteria);
    bw_info.criteria.title              = "high-bw active messages";
    bw_info.criteria.calc_score         = ucp_wireup_am_bw_score_func;
    bw_info.criteria.lane_type          = UCP_LANE_TYPE_AM_BW;
    ucp_wireup_init_select_flags(&bw_info.criteria.remote_iface_flags,
                                 UCP_ADDR_IFACE_FLAG_AM_SYNC, 0);
    ucp_wireup_init_select_flags(&bw_info.criteria.local_iface_flags,
                                 UCT_IFACE_FLAG_AM_BCOPY, 0);
    ucp_wireup_fill_peer_err_criteria(&bw_info.criteria, ep_init_flags);

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep),
                           UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP)) {
        bw_info.criteria.local_event_flags = UCP_WIREUP_UCT_EVENT_CAP_FLAGS;
    }

    bw_info.local_dev_bitmap  = UINT64_MAX;
    bw_info.remote_dev_bitmap = UINT64_MAX;
    bw_info.md_map            = 0;
    bw_info.max_lanes         = context->config.ext.max_eager_lanes - 1;
    /* rndv/am/zcopy proto should take max_rndv_lanes value into account */
    if (context->config.ext.proto_enable) {
        bw_info.max_lanes = ucs_max(bw_info.max_lanes,
                                    context->config.ext.max_rndv_lanes - 1);
    }

    /* am_bw_lane[0] is am_lane, so don't re-select it here */
    am_lane = UCP_NULL_LANE;
    for (lane_desc_idx = 0; lane_desc_idx < select_ctx->num_lanes; ++lane_desc_idx) {
        if (select_ctx->lane_descs[lane_desc_idx].lane_types &
                UCS_BIT(UCP_LANE_TYPE_AM)) {
            /* do not continue searching since we found AM lane (and there is
             * only one AM lane) */
            am_lane = lane_desc_idx;
            break;
        }
    }

    num_am_bw_lanes = ucp_wireup_add_bw_lanes(select_params, &bw_info,
                                              ucp_tl_bitmap_max, am_lane,
                                              select_ctx, 0);
    return ((am_lane != UCP_NULL_LANE) || (num_am_bw_lanes > 0)) ? UCS_OK :
           UCS_ERR_UNREACHABLE;
}

static void
ucp_wireup_init_rma_bw_criteria_iface_flags(ucp_rndv_mode_t rndv_mode,
                                            ucp_wireup_select_flags_t *local,
                                            ucp_wireup_select_flags_t *remote)
{
    switch (rndv_mode) {
    case UCP_RNDV_MODE_AUTO:
        local->optional  = UCT_IFACE_FLAG_GET_ZCOPY | UCT_IFACE_FLAG_PUT_ZCOPY;
        remote->optional = UCP_ADDR_IFACE_FLAG_GET  | UCP_ADDR_IFACE_FLAG_PUT;
        return;
    case UCP_RNDV_MODE_GET_ZCOPY:
        local->mandatory  = UCT_IFACE_FLAG_GET_ZCOPY;
        remote->mandatory = UCP_ADDR_IFACE_FLAG_GET;
        return;
    case UCP_RNDV_MODE_PUT_ZCOPY:
        local->mandatory  = UCT_IFACE_FLAG_PUT_ZCOPY;
        remote->mandatory = UCP_ADDR_IFACE_FLAG_PUT;
        return;
    default:
        return;
    }
}

static void ucp_wireup_criteria_iface_flags_add(
        ucp_wireup_criteria_t *criteria,
        const ucp_wireup_select_flags_t *local_iface_flags,
        const ucp_wireup_select_flags_t *remote_iface_flags)
{
    criteria->local_iface_flags.mandatory  |= local_iface_flags->mandatory;
    criteria->local_iface_flags.optional   |= local_iface_flags->optional;
    criteria->remote_iface_flags.mandatory |= remote_iface_flags->mandatory;
    criteria->remote_iface_flags.optional  |= remote_iface_flags->optional;
}

static void
ucp_wireup_iface_flags_unset(ucp_wireup_criteria_t *criteria,
                             const ucp_wireup_select_flags_t *local_iface_flags,
                             const ucp_wireup_select_flags_t *remote_iface_flags)
{
    criteria->local_iface_flags.mandatory  &= ~local_iface_flags->mandatory;
    criteria->local_iface_flags.optional   &= ~local_iface_flags->optional;
    criteria->remote_iface_flags.mandatory &= ~remote_iface_flags->mandatory;
    criteria->remote_iface_flags.optional  &= ~remote_iface_flags->optional;
}

static uint64_t
ucp_wireup_iface_get_perf_attr_flags(const ucp_worker_iface_t *wiface)
{
    uct_perf_attr_t perf_attr;
    ucs_status_t status;

    perf_attr.field_mask = UCT_PERF_ATTR_FIELD_FLAGS;
    status               = uct_iface_estimate_perf(wiface->iface, &perf_attr);
    if (status != UCS_OK) {
        return 0;
    }

    return perf_attr.flags;
}

static ucs_status_t
ucp_wireup_add_rma_bw_lanes(const ucp_wireup_select_params_t *select_params,
                            ucp_wireup_select_context_t *select_ctx)
{
    unsigned allow_extra_path          = 0;
    ucp_lane_index_t am_lane           = UCP_NULL_LANE;
    ucp_ep_h ep                        = select_params->ep;
    ucp_context_h context              = ep->worker->context;
    unsigned ep_init_flags             = ucp_wireup_ep_init_flags(select_params,
                                                                  select_ctx);
    const ucp_rndv_mode_t rndv_modes[] = {
        context->config.ext.rndv_mode,
        UCP_RNDV_MODE_GET_ZCOPY,
        UCP_RNDV_MODE_PUT_ZCOPY
    };
    ucp_wireup_select_bw_info_t bw_info;
    ucs_memory_type_t mem_type;
    size_t added_lanes;
    uint64_t md_reg_flag;
    ucp_tl_bitmap_t tl_bitmap, mem_type_tl_bitmap;
    uint8_t i;
    ucp_wireup_select_flags_t iface_rma_flags, peer_rma_flags;
    ucp_wireup_lane_desc_t *lane_desc;
    uint64_t flags;
    const ucp_worker_iface_t *wiface;

    ucp_wireup_init_select_flags(&iface_rma_flags, 0, 0);
    ucp_wireup_init_select_flags(&peer_rma_flags, 0, 0);

    if (ep_init_flags & UCP_EP_INIT_CREATE_AM_LANE_ONLY) {
        return UCS_OK;
    }

    if (ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE) {
        md_reg_flag = 0;
    } else if (ucp_ep_get_context_features(ep) &
               (UCP_FEATURE_TAG | UCP_FEATURE_AM | UCP_FEATURE_RMA)) {
        /* if needed for RNDV, need only access for remote registered memory */
        md_reg_flag = UCT_MD_FLAG_REG;
    } else {
        return UCS_OK;
    }

    ucp_wireup_criteria_init(&bw_info.criteria);
    bw_info.criteria.calc_score     = ucp_wireup_rma_bw_score_func;
    bw_info.criteria.local_md_flags = md_reg_flag;
    ucp_wireup_init_select_flags(&bw_info.criteria.local_iface_flags,
                                 UCT_IFACE_FLAG_PENDING, 0);
    ucp_wireup_fill_peer_err_criteria(&bw_info.criteria, ep_init_flags);

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep),
                           UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP)) {
        bw_info.criteria.local_event_flags = UCP_WIREUP_UCT_EVENT_CAP_FLAGS;
    }

    bw_info.local_dev_bitmap  = UINT64_MAX;
    bw_info.remote_dev_bitmap = UINT64_MAX;
    bw_info.md_map            = 0;

    /* check rkey_ptr */
    if (!(ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE) &&
         (context->config.ext.rndv_mode == UCP_RNDV_MODE_AUTO)) {

        /* We require remote memory registration and local ability to obtain
         * a pointer to the remote key. Only one is needed since we are doing
         * memory copy on the CPU.
         * Allow selecting additional lanes in case the remote memory will not be
         * registered with this memory domain, i.e with GPU memory.
         */
        bw_info.criteria.title             = "obtain remote memory pointer";
        bw_info.criteria.local_cmpt_flags |= UCT_COMPONENT_FLAG_RKEY_PTR;
        bw_info.criteria.lane_type         = UCP_LANE_TYPE_RKEY_PTR;
        bw_info.max_lanes                  = 1;

        UCP_CONTEXT_MEM_CAP_TLS(context, UCS_MEMORY_TYPE_HOST, access_mem_types,
                                tl_bitmap);
        ucp_wireup_add_bw_lanes(select_params, &bw_info, tl_bitmap,
                                UCP_NULL_LANE, select_ctx, 0);
    }

    bw_info.criteria.title            = "high-bw remote memory access";
    bw_info.criteria.lane_type        = UCP_LANE_TYPE_RMA_BW;
    bw_info.criteria.local_cmpt_flags = 0;

    if (context->config.ext.proto_enable &&
        (select_params->address->dst_version > UCP_RELEASE_LEGACY)) {
        bw_info.max_lanes = ucp_wireup_max_lanes(select_params,
                                                 bw_info.criteria.lane_type);
    } else {
        bw_info.max_lanes = context->config.ext.max_rndv_lanes;
    }

    /* If error handling is requested we require memory invalidation
     * support to provide correct data integrity in case of error */
    if (ep_init_flags & UCP_EP_INIT_ERR_MODE_PEER_FAILURE) {
        bw_info.criteria.local_md_flags |= UCT_MD_FLAG_INVALIDATE_RMA;
    }

    /* Find if an AM lane needs to be separated from RMA operations. More lanes
     * can cause fence calls to last longer at scale. Applications making heavy
     * use of fence operations usually don't request the TAG feature, hence the
     * check.
     */
    if ((select_params->address->dst_version >= 17) &&
        (context->config.features & (UCP_FEATURE_TAG | UCP_FEATURE_AM))) {
        ucs_carray_for_each(lane_desc, select_ctx->lane_descs,
                            select_ctx->num_lanes) {
            if (!(lane_desc->lane_types & UCS_BIT(UCP_LANE_TYPE_AM))) {
                continue;
            }

            wiface = ucp_worker_iface(ep->worker, lane_desc->rsc_index);
            flags  = ucp_wireup_iface_get_perf_attr_flags(wiface);
            if (!(flags & UCT_PERF_ATTR_FLAGS_TX_RX_SHARED)) {
                continue;
            }

            am_lane = lane_desc - select_ctx->lane_descs;

            /* Map AM lane to specific path_index */
            ucs_assertv((lane_desc->path_index == 0) ||
                                (lane_desc->path_index ==
                                 UCP_WIREUP_PATH_INDEX_UNDEFINED),
                        "rsc_index=%u am_lane=%u path_index=%u",
                        lane_desc->rsc_index, am_lane, lane_desc->path_index);

            lane_desc->path_index = 0;
            allow_extra_path      = 1;
            break;
        }
    }

    /* RNDV protocol can't mix different schemes, i.e. wireup has to
     * select lanes with the same iface flags depends on a requested
     * RNDV scheme.
     * First of all, try to select lanes with RNDV scheme requested
     * by user. If no lanes were selected and RNDV scheme in the
     * configuration is AUTO, try other schemes. */
    UCS_STATIC_ASSERT(UCS_MEMORY_TYPE_HOST == 0);
    for (i = 0; i < ucs_static_array_size(rndv_modes); i++) {
        /* Remove the previous iface RMA flags */
        ucp_wireup_iface_flags_unset(&bw_info.criteria, &iface_rma_flags,
                                     &peer_rma_flags);

        ucp_wireup_init_rma_bw_criteria_iface_flags(rndv_modes[i],
                                                    &iface_rma_flags,
                                                    &peer_rma_flags);

        /* Set the new iface RMA flags */
        ucp_wireup_criteria_iface_flags_add(&bw_info.criteria, &iface_rma_flags,
                                            &peer_rma_flags);

        /* Add lanes that can access the memory by short operations */
        added_lanes = 0;
        UCS_STATIC_BITMAP_RESET_ALL(&tl_bitmap);

        ucs_memory_type_for_each(mem_type) {
            UCP_CONTEXT_MEM_CAP_TLS(context, mem_type, reg_mem_types,
                                    mem_type_tl_bitmap);

            bw_info.criteria.reg_mem_types = UCS_BIT(mem_type);
            added_lanes                   += ucp_wireup_add_bw_lanes(
                                               select_params, &bw_info,
                                               UCP_TL_BITMAP_AND_NOT(
                                                 mem_type_tl_bitmap, tl_bitmap),
                                               am_lane, select_ctx,
                                               allow_extra_path);

            UCS_STATIC_BITMAP_OR_INPLACE(&tl_bitmap, mem_type_tl_bitmap);
        }

        if (added_lanes /* There are selected lanes */ ||
            /* There are no selected lanes, but a user requested
             * the exact RNDV scheme, so there is no other choice */
            (context->config.ext.rndv_mode != UCP_RNDV_MODE_AUTO)) {
            break;
        }
    }

    return UCS_OK;
}

/* Lane for transport offloaded tag interface */
static ucs_status_t
ucp_wireup_add_tag_lane(const ucp_wireup_select_params_t *select_params,
                        const ucp_wireup_select_info_t *am_info,
                        ucp_err_handling_mode_t err_mode,
                        ucp_wireup_select_context_t *select_ctx)
{
    ucp_ep_h ep                          = select_params->ep;
    ucp_wireup_select_info_t select_info = {0};
    unsigned ep_init_flags               = ucp_wireup_ep_init_flags(
                                                   select_params, select_ctx);
    ucp_wireup_criteria_t criteria       = {};
    ucs_status_t status;

    if (!(ucp_ep_get_context_features(ep) & UCP_FEATURE_TAG) ||
        (ep_init_flags & (UCP_EP_INIT_FLAG_MEM_TYPE |
                          UCP_EP_INIT_CREATE_AM_LANE_ONLY)) ||
        /* TODO: remove check below when UCP_ERR_HANDLING_MODE_PEER supports
         *       RNDV-protocol or HW TM supports fragmented protocols
         */
        (err_mode != UCP_ERR_HANDLING_MODE_NONE)) {
        return UCS_OK;
    }

    ucp_wireup_criteria_init(&criteria);
    criteria.title          = "tag_offload";
    criteria.local_md_flags = UCT_MD_FLAG_REG; /* needed for posting tags to HW */
    criteria.calc_score     = ucp_wireup_am_score_func;
    criteria.lane_type      = UCP_LANE_TYPE_TAG;
    ucp_wireup_init_select_flags(&criteria.remote_iface_flags,
                                 UCP_ADDR_IFACE_FLAG_TAG_EAGER |
                                 UCP_ADDR_IFACE_FLAG_TAG_RNDV  |
                                 UCP_ADDR_IFACE_FLAG_GET, 0);
    ucp_wireup_init_select_flags(&criteria.local_iface_flags,
                                 UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                 UCT_IFACE_FLAG_TAG_RNDV_ZCOPY  |
                                 UCT_IFACE_FLAG_GET_ZCOPY       |
                                 UCT_IFACE_FLAG_PENDING, 0);

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep),
                           UCP_FEATURE_WAKEUP)) {
        criteria.local_event_flags = UCP_WIREUP_UCT_EVENT_CAP_FLAGS;
    }

    /* Do not add tag offload lane, if selected tag lane score is lower
     * than AM score. In this case AM will be used for tag matching. */
    status = ucp_wireup_select_transport(select_ctx, select_params, &criteria,
                                         ucp_tl_bitmap_max, UINT64_MAX,
                                         UINT64_MAX, UINT64_MAX, 0,
                                         &select_info, NULL, 0);
    if ((status == UCS_OK) &&
        (ucp_score_cmp(select_info.score,
                       am_info->score) >= 0)) {
        return ucp_wireup_add_lane(select_params, &select_info,
                                   UCP_LANE_TYPE_TAG, /* show error */ 1,
                                   select_ctx);
    }

    return UCS_OK;
}

static ucp_lane_index_t
ucp_wireup_select_wireup_msg_lane(ucp_worker_h worker,
                                  unsigned ep_init_flags,
                                  const ucp_address_entry_t *address_list,
                                  const ucp_wireup_lane_desc_t *lane_descs,
                                  ucp_lane_index_t num_lanes)
{
    ucp_context_h context          = worker->context;
    ucp_lane_index_t p2p_lane      = UCP_NULL_LANE;
    ucp_wireup_criteria_t criteria = {0};
    uct_tl_resource_desc_t *resource;
    ucp_rsc_index_t rsc_index;
    uct_iface_attr_t *attrs;
    ucp_lane_index_t lane;
    unsigned addr_index;

    ucp_wireup_fill_aux_criteria(&criteria, ep_init_flags,
                                 UCP_ADDR_IFACE_FLAG_CB_ASYNC);
    for (lane = 0; lane < num_lanes; ++lane) {
        if (lane_descs[lane].rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        rsc_index  = lane_descs[lane].rsc_index;
        addr_index = lane_descs[lane].addr_index;
        resource   = &context->tl_rscs[rsc_index].tl_rsc;
        attrs      = ucp_worker_iface_get_attr(worker, rsc_index);

        /* if the current lane satisfies the wireup criteria, choose it for wireup.
         * if it doesn't take a lane with a p2p transport */
        if (ucp_wireup_check_select_flags(resource, attrs->cap.flags,
                                          &criteria.local_iface_flags,
                                          criteria.title,
                                          ucp_wireup_iface_flags, NULL, 0) &&
            ucp_wireup_check_flags(resource, attrs->cap.event_flags,
                                   criteria.local_event_flags, criteria.title,
                                   ucp_wireup_event_flags, NULL, 0) &&
            ucp_wireup_check_select_flags(
                    resource, address_list[addr_index].iface_attr.flags,
                    &criteria.remote_iface_flags, criteria.title,
                    ucp_wireup_peer_flags, NULL, 0) &&
            ucp_wireup_check_flags(resource,
                                   address_list[addr_index].iface_attr.flags,
                                   criteria.remote_event_flags, criteria.title,
                                   ucp_wireup_peer_flags, NULL, 0)) {
            return lane;
        } else if (ucp_worker_is_tl_p2p(worker, rsc_index)) {
            p2p_lane = lane;
        }
    }

    return p2p_lane;
}

static UCS_F_NOINLINE void
ucp_wireup_select_params_init(ucp_wireup_select_params_t *select_params,
                              ucp_ep_h ep, unsigned ep_init_flags,
                              const ucp_unpacked_address_t *remote_address,
                              ucp_tl_bitmap_t tl_bitmap, int show_error)
{
    select_params->ep            = ep;
    select_params->ep_init_flags = ep_init_flags;
    select_params->tl_bitmap     = tl_bitmap;
    select_params->address       = remote_address;
    select_params->allow_am      =
            ucp_wireup_allow_am_emulation_layer(ep_init_flags);
    select_params->show_error    = show_error;
}

static double
ucp_wireup_keepalive_score_func(const ucp_worker_iface_t *wiface,
                                const uct_md_attr_v2_t *md_attr,
                                const ucp_unpacked_address_t *unpacked_addr,
                                const ucp_address_entry_t *remote_addr,
                                void *arg)
{
    uct_perf_attr_t perf_attr;
    ucs_status_t status;

    perf_attr.field_mask = UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS;
    status               = ucp_worker_iface_estimate_perf(wiface, &perf_attr);
    if (status != UCS_OK) {
        ucs_warn(UCT_TL_RESOURCE_DESC_FMT
                 ": getting perf estimations failed: %s",
                 UCT_TL_RESOURCE_DESC_ARG(&wiface->worker->context->tl_rscs[
                                                  wiface->rsc_index].tl_rsc),
                 ucs_status_string(status));
        return 0;
    }

    return ucp_wireup_am_score_func(wiface, md_attr, unpacked_addr, remote_addr, arg) *
           ((double)perf_attr.max_inflight_eps / (double)SIZE_MAX);
}

static ucs_status_t
ucp_wireup_add_keepalive_lane(const ucp_wireup_select_params_t *select_params,
                              ucp_err_handling_mode_t err_mode,
                              ucp_wireup_select_context_t *select_ctx)
{
    ucp_ep_h ep                          = select_params->ep;
    ucp_worker_h worker                  = ep->worker;
    ucp_wireup_select_info_t select_info = {0};
    unsigned ep_init_flags               = ucp_wireup_ep_init_flags(
                                                   select_params, select_ctx);
    ucp_wireup_criteria_t criteria       = {};
    const ucp_tl_bitmap_t *tl_bitmap;
    ucs_status_t status;

    if ((err_mode == UCP_ERR_HANDLING_MODE_NONE) ||
        !ucp_worker_keepalive_is_enabled(worker) ||
        (ep_init_flags & UCP_EP_INIT_FLAG_INTERNAL)) {
        return UCS_OK;
    }

    if (ep_init_flags & (UCP_EP_INIT_CREATE_AM_LANE_ONLY |
                         UCP_EP_INIT_KA_FROM_EXIST_LANES)) {
        tl_bitmap = &select_ctx->tl_bitmap;
    } else {
        tl_bitmap = &select_params->tl_bitmap;
    }

    ucp_wireup_criteria_init(&criteria);
    criteria.title              = "keepalive";
    criteria.local_md_flags     = 0;
    criteria.is_keepalive       = 1;
    criteria.calc_score         = ucp_wireup_keepalive_score_func;
    /* Keepalive can also use auxiliary transports */
    criteria.tl_rsc_flags       = UCP_TL_RSC_FLAG_AUX;
    criteria.lane_type          = UCP_LANE_TYPE_KEEPALIVE;
    ucp_wireup_fill_peer_err_criteria(&criteria, ep_init_flags);

    status = ucp_wireup_select_transport(select_ctx, select_params, &criteria,
                                         *tl_bitmap, UINT64_MAX, UINT64_MAX,
                                         UINT64_MAX, 0, &select_info, NULL, 0);
    if (status == UCS_OK) {
        return ucp_wireup_add_lane(select_params, &select_info,
                                   UCP_LANE_TYPE_KEEPALIVE, /* show error */ 1,
                                   select_ctx);
    }

    return status;
}

static void
ucp_wireup_select_context_init(ucp_wireup_select_context_t *select_ctx)
{
    memset(&select_ctx->lane_descs, 0, sizeof(select_ctx->lane_descs));
    select_ctx->num_lanes         = 0;
    select_ctx->ucp_ep_init_flags = 0;
    UCS_STATIC_BITMAP_RESET_ALL(&select_ctx->tl_bitmap);
}

static UCS_F_NOINLINE ucs_status_t
ucp_wireup_search_lanes(const ucp_wireup_select_params_t *select_params,
                        ucp_err_handling_mode_t err_mode,
                        ucp_wireup_select_context_t *select_ctx,
                        char *info_string, size_t info_string_length)
{
    ucp_wireup_select_info_t am_info;
    ucs_status_t status;

    ucp_wireup_select_context_init(select_ctx);

    status = ucp_wireup_add_cm_lane(select_params, select_ctx);
    if (status != UCS_OK) {
        return status;
    }

    /* Add fast protocols first (so they'll fit in the cached-in part of
     * ucp_ep. Fast protocols are: RMA/AM/AMO/TAG */
    status = ucp_wireup_add_rma_lanes(select_params, select_ctx);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_wireup_add_amo_lanes(select_params, select_ctx);
    if (status != UCS_OK) {
        return status;
    }

    /* Add AM lane only after RMA/AMO was selected to be aware
     * about whether they need emulation over AM or not */
    status = ucp_wireup_add_am_lane(select_params, &am_info, select_ctx,
                                    info_string, info_string_length);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_wireup_add_tag_lane(select_params, &am_info, err_mode,
                                     select_ctx);
    if (status != UCS_OK) {
        return status;
    }

    /* Add slow protocols on the remaining lanes */
    status = ucp_wireup_add_rma_bw_lanes(select_params, select_ctx);
    if (status != UCS_OK) {
        return status;
    }

    /* call ucp_wireup_add_am_bw_lanes after ucp_wireup_add_am_lane to
     * allow exclude AM lane from AM_BW list */
    status = ucp_wireup_add_am_bw_lanes(select_params, select_ctx);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_wireup_add_keepalive_lane(select_params, err_mode,
                                           select_ctx);
    if (status != UCS_OK) {
        return status;
    }

    /* User should not create endpoints unless requested communication features */
    if (select_params->show_error && (select_ctx->num_lanes == 0)) {
        ucs_error("No transports selected to %s (features: 0x%"PRIx64")",
                  select_params->address->name,
                  ucp_ep_get_context_features(select_params->ep));
        return UCS_ERR_UNREACHABLE;
    }

    return UCS_OK;
}

static int
ucp_wireup_is_built_in_keepalive(
        const ucp_worker_h worker, ucp_lane_index_t lane,
        const ucp_wireup_select_params_t *select_params,
        const ucp_ep_config_key_t *key)
{
    ucp_rsc_index_t rsc_index = key->lanes[lane].rsc_index;
    return (ucp_worker_iface_get_attr(worker, rsc_index)->cap.flags &
                    UCT_IFACE_FLAG_EP_KEEPALIVE) &&
            ucp_wireup_connect_p2p(worker, rsc_index,
                                   ucp_ep_init_flags_has_cm(
                                           select_params->ep_init_flags));
}

static unsigned ucp_wireup_default_path_index(unsigned path_index)
{
    return (path_index == UCP_WIREUP_PATH_INDEX_UNDEFINED) ? 0 : path_index;
}

static ucs_status_t ucp_wireup_select_set_locality_flags(
        const ucp_wireup_select_params_t *select_params,
        unsigned *addr_indices, ucp_ep_config_key_t *key)
{
    ucp_worker_h worker = select_params->ep->worker;
    ucp_worker_iface_t *wiface;
    ucp_rsc_index_t rsc_index;
    ucp_address_entry_t *ae;
    ucp_lane_index_t lane;
    unsigned addr_index;
    uct_iface_is_reachable_params_t params;

    if (select_params->address->uuid == worker->uuid) {
        key->flags |= UCP_EP_CONFIG_KEY_FLAG_SELF;
        return UCS_OK;
    }

    /* If selected at least one shared-memory transport, it's intra-node */
    for (lane = 0; lane < key->num_lanes; ++lane) {
        rsc_index = key->lanes[lane].rsc_index;
        if ((rsc_index != UCP_NULL_RESOURCE) &&
            !(ucp_worker_iface_get_attr(worker, rsc_index)->cap.flags &
              UCT_IFACE_FLAG_INTER_NODE)) {
            key->flags |= UCP_EP_CONFIG_KEY_FLAG_INTRA_NODE;
            return UCS_OK;
        }
    }

    /* If the local context matching the corresponding remote device address,
     * it's intra-node. Only selected lanes are compared to verify reachability.
     */
    for (lane = 0; lane < key->num_lanes; ++lane) {
        rsc_index = key->lanes[lane].rsc_index;
        if (rsc_index == UCP_NULL_RESOURCE) {
            continue;
        }

        if (key->cm_lane == lane) {
            continue;
        }

        addr_index = addr_indices[lane];
        ae         = &select_params->address->address_list[addr_index];

        wiface = ucp_worker_iface(worker, rsc_index);
        if (wiface->attr.device_addr_len == 0) {
            continue;
        }

        params.field_mask  = UCT_IFACE_IS_REACHABLE_FIELD_DEVICE_ADDR |
                             UCT_IFACE_IS_REACHABLE_FIELD_IFACE_ADDR  |
                             UCT_IFACE_IS_REACHABLE_FIELD_SCOPE;
        params.device_addr = ae->dev_addr;
        params.iface_addr  = ae->iface_addr;
        params.scope       = UCT_IFACE_REACHABILITY_SCOPE_DEVICE;

        if (uct_iface_is_reachable_v2(wiface->iface, &params)) {
            key->flags |= UCP_EP_CONFIG_KEY_FLAG_INTRA_NODE;
            return UCS_OK;
        }
    }

    return UCS_OK;
}

static UCS_F_NOINLINE ucs_status_t
ucp_wireup_construct_lanes(const ucp_wireup_select_params_t *select_params,
                           ucp_wireup_select_context_t *select_ctx,
                           unsigned *addr_indices, ucp_ep_config_key_t *key)
{
    ucp_ep_h ep           = select_params->ep;
    ucp_worker_h worker   = ep->worker;
    ucp_context_h context = worker->context;
    ucp_rsc_index_t rsc_index;
    ucp_md_index_t md_index;
    ucp_lane_index_t lane;
    ucp_lane_index_t i;

    key->num_lanes = select_ctx->num_lanes;
    /* Construct the endpoint configuration key:
     * - arrange lane description in the EP configuration
     * - create remote MD bitmap
     * - if AM lane exists and fits for wireup messages, select it for this purpose.
     */
    for (lane = 0; lane < key->num_lanes; ++lane) {
        ucs_assert(select_ctx->lane_descs[lane].lane_types != 0);
        addr_indices[lane]            = select_ctx->lane_descs[lane].addr_index;
        key->lanes[lane].rsc_index    = select_ctx->lane_descs[lane].rsc_index;
        key->lanes[lane].dst_md_index = select_ctx->lane_descs[lane].dst_md_index;
        key->lanes[lane].dst_sys_dev  = select_ctx->lane_descs[lane].dst_sys_dev;
        key->lanes[lane].lane_types   = select_ctx->lane_descs[lane].lane_types;
        key->lanes[lane].seg_size     = select_ctx->lane_descs[lane].seg_size;
        key->lanes[lane].path_index   = ucp_wireup_default_path_index(
                                       select_ctx->lane_descs[lane].path_index);

        if (select_ctx->lane_descs[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_CM)) {
            ucs_assert(key->cm_lane == UCP_NULL_LANE);
            key->cm_lane = lane;
            /* CM lane can't be shared with TL lane types */
            ucs_assert(ucs_popcount(select_ctx->lane_descs[lane].lane_types) == 1);
            continue;
        }
        if (select_ctx->lane_descs[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_AM)) {
            ucs_assert(key->am_lane == UCP_NULL_LANE);
            key->am_lane = lane;
        }
        if ((select_ctx->lane_descs[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_AM_BW)) &&
            (lane < UCP_MAX_LANES - 1)) {
            key->am_bw_lanes[lane + 1] = lane;
        }
        if (select_ctx->lane_descs[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_RMA)) {
            key->rma_lanes[lane] = lane;
        }
        if (select_ctx->lane_descs[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_RMA_BW)) {
            key->rma_bw_lanes[lane] = lane;
        }
        if (select_ctx->lane_descs[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_RKEY_PTR)) {
            ucs_assert(key->rkey_ptr_lane == UCP_NULL_LANE);
            key->rkey_ptr_lane = lane;
        }
        if (select_ctx->lane_descs[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_AMO)) {
            key->amo_lanes[lane] = lane;
        }
        if (select_ctx->lane_descs[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_TAG)) {
            ucs_assert(key->tag_lane == UCP_NULL_LANE);
            key->tag_lane = lane;
        }
        if (select_ctx->lane_descs[lane].lane_types &
                    UCS_BIT(UCP_LANE_TYPE_KEEPALIVE) &&
            !ucp_wireup_is_built_in_keepalive(ep->worker, lane, select_params, key)) {
            ucs_assert(key->keepalive_lane == UCP_NULL_LANE);
            key->keepalive_lane = lane;
        }
    }

    /* Sort AM, RMA and AMO lanes according to score */
    ucs_qsort_r(key->am_bw_lanes + 1, UCP_MAX_LANES - 1,
                sizeof(ucp_lane_index_t), ucp_wireup_compare_lane_am_bw_score,
                select_ctx->lane_descs);
    ucs_qsort_r(key->rma_lanes, UCP_MAX_LANES, sizeof(ucp_lane_index_t),
                ucp_wireup_compare_lane_rma_score, select_ctx->lane_descs);
    ucs_qsort_r(key->rma_bw_lanes, UCP_MAX_LANES, sizeof(ucp_lane_index_t),
                ucp_wireup_compare_lane_rma_bw_score, select_ctx->lane_descs);
    ucs_qsort_r(key->amo_lanes, UCP_MAX_LANES, sizeof(ucp_lane_index_t),
                ucp_wireup_compare_lane_amo_score, select_ctx->lane_descs);

    /* Select lane for wireup messages, if: */
    if (/* - no CM support was requested */
        !ucp_ep_init_flags_has_cm(select_params->ep_init_flags) ||
        /* - CM support was requested, but not locally connected yet */
        !(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        key->wireup_msg_lane =
        ucp_wireup_select_wireup_msg_lane(worker,
                                          ucp_wireup_ep_init_flags(select_params,
                                                                   select_ctx),
                                          select_params->address->address_list,
                                          select_ctx->lane_descs,
                                          key->num_lanes);
    }

    for (i = 0; key->rma_bw_lanes[i] != UCP_NULL_LANE; i++) {
        lane = key->rma_bw_lanes[i];
        rsc_index = select_ctx->lane_descs[lane].rsc_index;
        md_index  = context->tl_rscs[rsc_index].md_index;

        /* Pack remote key only if needed for RMA.
         * FIXME a temporary workaround to prevent the ugni uct from using rndv. */
        if ((context->tl_mds[md_index].attr.flags & UCT_MD_FLAG_NEED_RKEY) &&
            !(strstr(context->tl_rscs[rsc_index].tl_rsc.tl_name, "ugni"))) {
            key->rma_bw_md_map |= UCS_BIT(md_index);
        }
    }

    if (key->rkey_ptr_lane != UCP_NULL_LANE) {
        rsc_index            = select_ctx->lane_descs[key->rkey_ptr_lane].rsc_index;
        md_index             = context->tl_rscs[rsc_index].md_index;
        key->rma_bw_md_map  |= UCS_BIT(md_index);
    }

    for (i = 0; key->rma_lanes[i] != UCP_NULL_LANE; i++) {
        lane             = key->rma_lanes[i];
        rsc_index        = select_ctx->lane_descs[lane].rsc_index;
        md_index         = context->tl_rscs[rsc_index].md_index;
        key->rma_md_map |= UCS_BIT(md_index);
    }

    /* use AM lane first for eager AM transport to simplify processing single/middle
     * msg packets */
    key->am_bw_lanes[0] = key->am_lane;

    return ucp_wireup_select_set_locality_flags(select_params, addr_indices,
                                                key);
}

ucs_status_t
ucp_wireup_select_lanes(ucp_ep_h ep, unsigned ep_init_flags,
                        ucp_tl_bitmap_t tl_bitmap,
                        const ucp_unpacked_address_t *remote_address,
                        unsigned *addr_indices, ucp_ep_config_key_t *key,
                        int show_error)
{
    ucp_worker_h worker                = ep->worker;
    ucp_tl_bitmap_t scalable_tl_bitmap = worker->scalable_tl_bitmap;
    /* TODO: remove initialization after all ucp_wireup_add_X_lanes functions
       will support specifying a reason */
    char wireup_info[256]              = {0};
    ucp_wireup_select_context_t select_ctx;
    ucp_wireup_select_params_t select_params;
    ucs_status_t status;

    UCS_STATIC_BITMAP_AND_INPLACE(&scalable_tl_bitmap, tl_bitmap);

    if (!UCS_STATIC_BITMAP_IS_ZERO(scalable_tl_bitmap)) {
        ucp_wireup_select_params_init(&select_params, ep, ep_init_flags,
                                      remote_address, scalable_tl_bitmap, 0);
        status = ucp_wireup_search_lanes(&select_params, key->err_mode,
                                         &select_ctx, wireup_info,
                                         sizeof(wireup_info));
        if (status == UCS_OK) {
            goto out;
        }

        /* If the transport selection based on the scalable TL bitmap wasn't
         * successful, repeat the selection procedure with full TL bitmap in
         * order to select best transports based on their scores only */
    }

    ucp_wireup_select_params_init(&select_params, ep, ep_init_flags,
                                  remote_address, tl_bitmap, show_error);
    status = ucp_wireup_search_lanes(&select_params, key->err_mode,
                                     &select_ctx, wireup_info,
                                     sizeof(wireup_info));
    if (status != UCS_OK) {
        if (wireup_info[0] != '\0') {
            ucs_diag("destination is unreachable [%s]", wireup_info);
        }

        return status;
    }

out:
    status = ucp_wireup_construct_lanes(&select_params, &select_ctx,
                                        addr_indices, key);
    if (status != UCS_OK) {
        return status;
    }

    /* Only two lanes must be created during CM phase (CM lane and TL lane) of
     * connection setup between two peers, if an AM lane only requested */
    ucs_assert(!ucs_test_all_flags(ep_init_flags,
                                   UCP_EP_INIT_CREATE_AM_LANE_ONLY |
                                   UCP_EP_INIT_CM_PHASE) ||
               (key->num_lanes == 2));

    return UCS_OK;
}

ucs_status_t
ucp_wireup_select_aux_transport(ucp_ep_h ep, unsigned ep_init_flags,
                                ucp_tl_bitmap_t tl_bitmap,
                                const ucp_unpacked_address_t *remote_address,
                                ucp_wireup_select_info_t *select_info)
{
    ucp_wireup_select_context_t select_ctx = {};
    ucp_wireup_criteria_t criteria         = {};
    ucp_wireup_select_params_t select_params;
    ucs_status_t status;

    ucp_wireup_select_params_init(&select_params, ep, ep_init_flags,
                                  remote_address, tl_bitmap, 1);

    /* Select auxiliary transport that supports async active message callback */
    ucp_wireup_fill_aux_criteria(&criteria, ep_init_flags,
                                 UCP_ADDR_IFACE_FLAG_CB_ASYNC);
    status = ucp_wireup_select_transport(&select_ctx, &select_params, &criteria,
                                         ucp_tl_bitmap_max, UINT64_MAX,
                                         UINT64_MAX, UINT64_MAX, 0,
                                         select_info, NULL, 0);
    if (status == UCS_OK) {
        return UCS_OK;
    }

    /* Fallback to an auxiliary transport without async active message callback
     * requirement */
    ucp_wireup_fill_aux_criteria(&criteria, ep_init_flags, 0);
    return ucp_wireup_select_transport(&select_ctx, &select_params, &criteria,
                                       ucp_tl_bitmap_max, UINT64_MAX,
                                       UINT64_MAX, UINT64_MAX, 1, select_info,
                                       NULL, 0);
}
