/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "wireup.h"
#include "address.h"

#include <ucs/algorithm/qsort_r.h>
#include <ucs/datastruct/queue.h>
#include <ucs/sys/string.h>
#include <ucp/core/ucp_ep.inl>
#include <string.h>
#include <inttypes.h>

#define UCP_WIREUP_RMA_BW_TEST_MSG_SIZE       262144

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


enum {
    UCP_WIREUP_LANE_USAGE_AM     = UCS_BIT(0), /* Active messages */
    UCP_WIREUP_LANE_USAGE_AM_BW  = UCS_BIT(1), /* High-BW active messages */
    UCP_WIREUP_LANE_USAGE_RMA    = UCS_BIT(2), /* Remote memory access */
    UCP_WIREUP_LANE_USAGE_RMA_BW = UCS_BIT(3), /* High-BW remote memory access */
    UCP_WIREUP_LANE_USAGE_AMO    = UCS_BIT(4), /* Atomic memory access */
    UCP_WIREUP_LANE_USAGE_TAG    = UCS_BIT(5)  /* Tag matching offload */
};


typedef struct {
    ucp_rsc_index_t   rsc_index;
    unsigned          addr_index;
    ucp_lane_index_t  proxy_lane;
    ucp_rsc_index_t   dst_md_index;
    uint32_t          usage;
    double            am_bw_score;
    double            rma_score;
    double            rma_bw_score;
    double            amo_score;
} ucp_wireup_lane_desc_t;


typedef struct {
    ucp_wireup_criteria_t criteria;
    uint64_t              local_dev_bitmap;
    uint64_t              remote_dev_bitmap;
    ucp_md_map_t          md_map;
    uint32_t              usage;
    unsigned              max_lanes;
} ucp_wireup_select_bw_info_t;


static const char *ucp_wireup_md_flags[] = {
    [ucs_ilog2(UCT_MD_FLAG_ALLOC)]               = "memory allocation",
    [ucs_ilog2(UCT_MD_FLAG_REG)]                 = "memory registration",
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
    [ucs_ilog2(UCT_IFACE_FLAG_EVENT_SEND_COMP)]  = "send completion event",
    [ucs_ilog2(UCT_IFACE_FLAG_EVENT_RECV)]       = "tag or active message event",
    [ucs_ilog2(UCT_IFACE_FLAG_EVENT_RECV_SIG)]   = "signaled message event",
    [ucs_ilog2(UCT_IFACE_FLAG_PENDING)]          = "pending",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_SHORT)]  = "tag eager short",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_BCOPY)]  = "tag eager bcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_EAGER_ZCOPY)]  = "tag eager zcopy",
    [ucs_ilog2(UCT_IFACE_FLAG_TAG_RNDV_ZCOPY)]   = "tag rndv zcopy"
};

static ucp_wireup_atomic_flag_t ucp_wireup_atomic_desc[] = {
     [UCT_ATOMIC_OP_ADD]   = {.name = "add",   .fetch = "fetch-"},
     [UCT_ATOMIC_OP_AND]   = {.name = "and",   .fetch = "fetch-"},
     [UCT_ATOMIC_OP_OR]    = {.name = "or",    .fetch = "fetch-"},
     [UCT_ATOMIC_OP_XOR]   = {.name = "xor",   .fetch = "fetch-"},
     [UCT_ATOMIC_OP_SWAP]  = {.name = "swap",  .fetch = ""},
     [UCT_ATOMIC_OP_CSWAP] = {.name = "cscap", .fetch = ""}
};


static double ucp_wireup_aux_score_func(ucp_context_h context,
                                        const uct_md_attr_t *md_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_address_iface_attr_t *remote_iface_attr);

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

static int ucp_wireup_check_flags(const uct_tl_resource_desc_t *resource,
                                  uint64_t flags, uint64_t required_flags,
                                  const char *title, const char ** flag_descs,
                                  char *reason, size_t max)
{
    const char *missing_flag_desc;

    if (ucs_test_all_flags(flags, required_flags)) {
        return 1;
    }

    if (required_flags) {
        missing_flag_desc = ucp_wireup_get_missing_flag_desc(flags, required_flags,
                                                             flag_descs);
        ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : not suitable for %s, no %s",
                  UCT_TL_RESOURCE_DESC_ARG(resource), title,
                  missing_flag_desc);
        snprintf(reason, max, UCT_TL_RESOURCE_DESC_FMT" - no %s",
                 UCT_TL_RESOURCE_DESC_ARG(resource), missing_flag_desc);
    }
    return 0;
}

static int ucp_wireup_check_amo_flags(const uct_tl_resource_desc_t *resource,
                                      uint64_t flags, uint64_t required_flags,
                                      int op_size, int fetch,
                                      const char *title, char *reason, size_t max)
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

static int ucp_wireup_is_reachable(ucp_worker_h worker, ucp_rsc_index_t rsc_index,
                                   const ucp_address_entry_t *ae)
{
    ucp_context_h context = worker->context;
    return (context->tl_rscs[rsc_index].tl_name_csum == ae->tl_name_csum) &&
           uct_iface_is_reachable(worker->ifaces[rsc_index].iface, ae->dev_addr,
                                  ae->iface_addr);
}

/**
 * Select a local and remote transport
 */
static UCS_F_NOINLINE ucs_status_t
ucp_wireup_select_transport(ucp_ep_h ep, const ucp_address_entry_t *address_list,
                            unsigned address_count, const ucp_wireup_criteria_t *criteria,
                            uint64_t tl_bitmap, uint64_t remote_md_map,
                            uint64_t local_dev_bitmap, uint64_t remote_dev_bitmap,
                            int show_error, ucp_rsc_index_t *rsc_index_p,
                            unsigned *dst_addr_index_p, double *score_p)
{
    ucp_worker_h worker = ep->worker;
    ucp_context_h context = worker->context;
    uct_tl_resource_desc_t *resource;
    const ucp_address_entry_t *ae;
    ucp_rsc_index_t rsc_index;
    double score, best_score;
    char tls_info[256];
    char *p, *endp;
    uct_iface_attr_t *iface_attr;
    uct_md_attr_t *md_attr;
    uint64_t addr_index_map;
    unsigned addr_index;
    int reachable;
    int found;
    uint8_t priority, best_score_priority;
    float epsilon; /* a small value to overcome float imprecision */

    found       = 0;
    best_score  = 0.0;
    best_score_priority = 0;
    p           = tls_info;
    endp        = tls_info + sizeof(tls_info) - 1;
    tls_info[0] = '\0';

    /* Check which remote addresses satisfy the criteria */
    addr_index_map = 0;
    for (ae = address_list; ae < address_list + address_count; ++ae) {
        addr_index = ae - address_list;
        if (!(remote_dev_bitmap & UCS_BIT(ae->dev_index))) {
            ucs_trace("addr[%d]: not in use, because on device[%d]",
                      addr_index, ae->dev_index);
            continue;
        } else if (!(remote_md_map & UCS_BIT(ae->md_index))) {
            ucs_trace("addr[%d]: not in use, because on md[%d]", addr_index,
                      ae->md_index);
            continue;
        } else if (!ucs_test_all_flags(ae->md_flags, criteria->remote_md_flags)) {
            ucs_trace("addr[%d] %s: no %s", addr_index,
                      ucp_find_tl_name_by_csum(context, ae->tl_name_csum),
                      ucp_wireup_get_missing_flag_desc(ae->md_flags,
                                                       criteria->remote_md_flags,
                                                       ucp_wireup_md_flags));
            continue;
        }

        /* Make sure we are indeed passing all flags required by the criteria in
         * ucp packed address */
        ucs_assert(ucs_test_all_flags(UCP_ADDRESS_IFACE_FLAGS,
                                      criteria->remote_iface_flags));

        if (!ucs_test_all_flags(ae->iface_attr.cap_flags, criteria->remote_iface_flags)) {
            ucs_trace("addr[%d] %s: no %s", addr_index,
                      ucp_find_tl_name_by_csum(context, ae->tl_name_csum),
                      ucp_wireup_get_missing_flag_desc(ae->iface_attr.cap_flags,
                                                       criteria->remote_iface_flags,
                                                       ucp_wireup_iface_flags));
            continue;
        }

        UCP_WIREUP_CHECK_AMO_FLAGS(ae, criteria, context, addr_index, op, 32);
        UCP_WIREUP_CHECK_AMO_FLAGS(ae, criteria, context, addr_index, op, 64);
        UCP_WIREUP_CHECK_AMO_FLAGS(ae, criteria, context, addr_index, fop, 32);
        UCP_WIREUP_CHECK_AMO_FLAGS(ae, criteria, context, addr_index, fop, 64);

        addr_index_map |= UCS_BIT(addr_index);
    }

    if (!addr_index_map) {
         snprintf(p, endp - p, "%s  ", ucs_status_string(UCS_ERR_UNSUPPORTED));
         p += strlen(p);
    }

    /* For each local resource try to find the best remote address to connect to.
     * Pick the best local resource to satisfy the criteria.
     * best one has the highest score (from the dedicated score_func) and
     * has a reachable tl on the remote peer */
    for (rsc_index = 0; addr_index_map && (rsc_index < context->num_tls); ++rsc_index) {
        resource     = &context->tl_rscs[rsc_index].tl_rsc;
        iface_attr   = &worker->ifaces[rsc_index].attr;
        md_attr      = &context->tl_mds[context->tl_rscs[rsc_index].md_index].attr;

        if ((context->tl_rscs[rsc_index].flags & UCP_TL_RSC_FLAG_AUX) &&
            !(criteria->tl_rsc_flags & UCP_TL_RSC_FLAG_AUX)) {
            continue;
        }

        /* Check that local md and interface satisfy the criteria */
        if (!ucp_wireup_check_flags(resource, md_attr->cap.flags,
                                    criteria->local_md_flags, criteria->title,
                                    ucp_wireup_md_flags, p, endp - p) ||
            !ucp_wireup_check_flags(resource, iface_attr->cap.flags,
                                    criteria->local_iface_flags, criteria->title,
                                    ucp_wireup_iface_flags, p, endp - p) ||
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
        if (!(tl_bitmap & UCS_BIT(rsc_index))) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : disabled by tl_bitmap",
                      UCT_TL_RESOURCE_DESC_ARG(resource));
            snprintf(p, endp - p, UCT_TL_RESOURCE_DESC_FMT" - disabled for %s, ",
                     UCT_TL_RESOURCE_DESC_ARG(resource), criteria->title);
            p += strlen(p);
            continue;
        } else if (!(local_dev_bitmap & UCS_BIT(context->tl_rscs[rsc_index].dev_index))) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : disabled by device bitmap",
                      UCT_TL_RESOURCE_DESC_ARG(resource));
            snprintf(p, endp - p, UCT_TL_RESOURCE_DESC_FMT" - disabled for %s, ",
                     UCT_TL_RESOURCE_DESC_ARG(resource), criteria->title);
            p += strlen(p);
            continue;
        }

        reachable = 0;

        for (ae = address_list; ae < address_list + address_count; ++ae) {
            if (!(addr_index_map & UCS_BIT(ae - address_list)) ||
                !ucp_wireup_is_reachable(worker, rsc_index, ae))
            {
                /* Must be reachable device address, on same transport */
                continue;
            }

            reachable = 1;

            score = criteria->calc_score(context, md_attr, iface_attr,
                                         &ae->iface_attr);
            ucs_assert(score >= 0.0);

            priority = iface_attr->priority + ae->iface_attr.priority;

            ucs_trace(UCT_TL_RESOURCE_DESC_FMT "->addr[%zd] : %s score %.2f priority %d",
                      UCT_TL_RESOURCE_DESC_ARG(resource), ae - address_list,
                      criteria->title, score, priority);

            /* First comparing score, if score equals to current best score,
             * comparing priority with the priority of best score */
            epsilon = (score + best_score) * (1e-6);
            if (!found || (score > (best_score + epsilon)) ||
                ((fabs(score - best_score) < epsilon) && (priority > best_score_priority))) {
                *rsc_index_p      = rsc_index;
                *dst_addr_index_p = ae - address_list;
                *score_p          = score;
                best_score        = score;
                best_score_priority = priority;
                found             = 1;
            }
        }

        /* If a local resource cannot reach any of the remote addresses, generate
         * debug message.
         */
        if (!reachable) {
            snprintf(p, endp - p, UCT_TL_RESOURCE_DESC_FMT" - %s, ",
                     UCT_TL_RESOURCE_DESC_ARG(resource),
                     ucs_status_string(UCS_ERR_UNREACHABLE));
            p += strlen(p);
        }
    }

    if (p >= tls_info + 2) {
        *(p - 2) = '\0'; /* trim last "," */
    }

    if (!found) {
        if (show_error) {
            ucs_error("no %s transport to %s: %s", criteria->title,
                      ucp_ep_peer_name(ep), tls_info);
        }
        return UCS_ERR_UNREACHABLE;
    }

    ucs_trace("ep %p: selected for %s: " UCT_TL_RESOURCE_DESC_FMT " md[%d]"
              " -> '%s' address[%d],md[%d] score %.2f", ep, criteria->title,
              UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[*rsc_index_p].tl_rsc),
              context->tl_rscs[*rsc_index_p].md_index,
              ucp_ep_peer_name(ep), *dst_addr_index_p,
              address_list[*dst_addr_index_p].md_index, best_score);
    return UCS_OK;
}

static inline double ucp_wireup_tl_iface_latency(ucp_context_h context,
                                                 const uct_iface_attr_t *iface_attr,
                                                 const ucp_address_iface_attr_t *remote_iface_attr)
{
    return ucs_max(iface_attr->latency.overhead, remote_iface_attr->lat_ovh) +
           (iface_attr->latency.growth * context->config.est_num_eps);
}

static UCS_F_NOINLINE void
ucp_wireup_add_lane_desc(ucp_wireup_lane_desc_t *lane_descs,
                         ucp_lane_index_t *num_lanes_p, ucp_rsc_index_t rsc_index,
                         unsigned addr_index, ucp_rsc_index_t dst_md_index,
                         double score, uint32_t usage, int is_proxy)
{
    ucp_wireup_lane_desc_t *lane_desc;
    ucp_lane_index_t lane, proxy_lane;
    int proxy_changed;

    /* Add a new lane, but try to reuse already added lanes which are selected
     * on the same transport resources.
     */
    proxy_changed = 0;
    for (lane_desc = lane_descs; lane_desc < lane_descs + (*num_lanes_p); ++lane_desc) {
        if ((lane_desc->rsc_index == rsc_index) &&
            (lane_desc->addr_index == addr_index))
        {
            lane = lane_desc - lane_descs;
            ucs_assertv_always(dst_md_index == lane_desc->dst_md_index,
                               "lane[%d].dst_md_index=%d, dst_md_index=%d",
                               lane, lane_desc->dst_md_index, dst_md_index);
            ucs_assertv_always(!(lane_desc->usage & usage), "lane[%d]=0x%x |= 0x%x",
                               lane, lane_desc->usage, usage);
            if (is_proxy && (lane_desc->proxy_lane == UCP_NULL_LANE)) {
                /* New lane is a proxy, and found existing non-proxy lane with
                 * same resource. So that lane should be used by the proxy.
                 */
                proxy_lane = lane;
                goto out_add_lane;
            } else if (!is_proxy && (lane_desc->proxy_lane == lane)) {
                /* New lane is not a proxy, but found existing proxy lane which
                 * could use the new lane. It also means we should be able to
                 * add our new lane.
                 */
                lane_desc->proxy_lane = *num_lanes_p;
                proxy_changed = 1;
            } else if (!is_proxy && (lane_desc->proxy_lane == UCP_NULL_LANE)) {
                /* Found non-proxy lane with same resource - don't add */
                ucs_assert_always(!proxy_changed);
                lane_desc->usage |= usage;
                goto out_update_score;
            }
        }
    }

    /* If a proxy cannot find other lane with same resource, proxy to self */
    proxy_lane = is_proxy ? (*num_lanes_p) : UCP_NULL_LANE;

out_add_lane:
    lane_desc = &lane_descs[*num_lanes_p];
    ++(*num_lanes_p);

    lane_desc->rsc_index    = rsc_index;
    lane_desc->addr_index   = addr_index;
    lane_desc->proxy_lane   = proxy_lane;
    lane_desc->dst_md_index = dst_md_index;
    lane_desc->usage        = usage;
    lane_desc->am_bw_score  = 0.0;
    lane_desc->rma_score    = 0.0;
    lane_desc->rma_bw_score = 0.0;
    lane_desc->amo_score    = 0.0;

out_update_score:
    if (usage & UCP_WIREUP_LANE_USAGE_AM_BW) {
        lane_desc->am_bw_score = score;
    }
    if (usage & UCP_WIREUP_LANE_USAGE_RMA) {
        lane_desc->rma_score = score;
    }
    if (usage & UCP_WIREUP_LANE_USAGE_RMA_BW) {
        lane_desc->rma_bw_score = score;
    }
    if (usage & UCP_WIREUP_LANE_USAGE_AMO) {
        lane_desc->amo_score = score;
    }
}

#define UCP_WIREUP_COMPARE_SCORE(_elem1, _elem2, _arg, _token) \
    ({ \
        const ucp_lane_index_t *lane1 = (_elem1); \
        const ucp_lane_index_t *lane2 = (_elem2); \
        const ucp_wireup_lane_desc_t *lanes = (_arg); \
        double score1, score2; \
        \
        score1 = (*lane1 == UCP_NULL_LANE) ? 0.0 : lanes[*lane1]._token##_score; \
        score2 = (*lane2 == UCP_NULL_LANE) ? 0.0 : lanes[*lane2]._token##_score; \
        /* sort from highest score to lowest */ \
        (score1 < score2) ? 1 : ((score1 > score2) ? -1 : 0); \
    })

static int ucp_wireup_compare_lane_am_bw_score(const void *elem1, const void *elem2,
                                               void *arg)
{
    return UCP_WIREUP_COMPARE_SCORE(elem1, elem2, arg, am_bw);
}

static int ucp_wireup_compare_lane_rma_score(const void *elem1, const void *elem2,
                                             void *arg)
{
    return UCP_WIREUP_COMPARE_SCORE(elem1, elem2, arg, rma);
}

static int ucp_wireup_compare_lane_rma_bw_score(const void *elem1, const void *elem2,
                                             void *arg)
{
    return UCP_WIREUP_COMPARE_SCORE(elem1, elem2, arg, rma_bw);
}

static int ucp_wireup_compare_lane_amo_score(const void *elem1, const void *elem2,
                                             void *arg)
{
    return UCP_WIREUP_COMPARE_SCORE(elem1, elem2, arg, amo);
}

static uint64_t ucp_wireup_unset_tl_by_md(ucp_ep_h ep, uint64_t tl_bitmap,
                                          ucp_rsc_index_t rsc_index)
{
    ucp_context_h context    = ep->worker->context;
    ucp_rsc_index_t md_index = context->tl_rscs[rsc_index].md_index;
    ucp_rsc_index_t i;

    for (i = 0; i < context->num_tls; i++) {
        if (context->tl_rscs[i].md_index == md_index) {
            tl_bitmap &= ~UCS_BIT(i);
        }
    }

    return tl_bitmap;
}

static UCS_F_NOINLINE ucs_status_t
ucp_wireup_add_memaccess_lanes(ucp_ep_h ep, unsigned address_count,
                               const ucp_address_entry_t *address_list,
                               ucp_wireup_lane_desc_t *lane_descs,
                               ucp_lane_index_t *num_lanes_p,
                               const ucp_wireup_criteria_t *criteria,
                               uint64_t tl_bitmap, uint32_t usage,
                               int select_best, int show_error)
{
    ucp_wireup_criteria_t mem_criteria = *criteria;
    ucp_address_entry_t *address_list_copy;
    ucp_rsc_index_t rsc_index, dst_md_index;
    size_t address_list_size;
    double score, reg_score;
    uint64_t remote_md_map;
    unsigned addr_index;
    ucs_status_t status;
    char title[64];

    remote_md_map = -1;

    /* Create a copy of the address list */
    address_list_size = sizeof(*address_list_copy) * address_count;
    address_list_copy = ucs_malloc(address_list_size, "rma address list");
    if (address_list_copy == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    memcpy(address_list_copy, address_list, address_list_size);

    /* Select best transport which can reach registered memory */
    snprintf(title, sizeof(title), criteria->title, "registered");
    mem_criteria.title           = title;
    mem_criteria.remote_md_flags = UCT_MD_FLAG_REG | criteria->remote_md_flags;
    status = ucp_wireup_select_transport(ep, address_list_copy, address_count,
                                         &mem_criteria, tl_bitmap, remote_md_map,
                                         -1, -1, show_error,
                                         &rsc_index, &addr_index, &score);
    if (status != UCS_OK) {
        goto out_free_address_list;
    }

    dst_md_index = address_list_copy[addr_index].md_index;
    reg_score    = score;

    /* Add to the list of lanes and remove all occurrences of the remote md
     * from the address list, to avoid selecting the same remote md again.*/
    ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                             dst_md_index, score, usage, 0);
    remote_md_map &= ~UCS_BIT(dst_md_index);
    tl_bitmap = ucp_wireup_unset_tl_by_md(ep, tl_bitmap, rsc_index);

    /* Select additional transports which can access allocated memory, but only
     * if their scores are better. We need this because a remote memory block can
     * be potentially allocated using one of them, and we might get better performance
     * than the transports which support only registered remote memory.
     */
    if (select_best) {
        snprintf(title, sizeof(title), criteria->title, "allocated");
        mem_criteria.title           = title;
        mem_criteria.remote_md_flags = UCT_MD_FLAG_ALLOC | criteria->remote_md_flags;
    } else if (ep->worker->context->tl_rscs[rsc_index].tl_rsc.dev_type == UCT_DEVICE_TYPE_SHM) {
        /* special case for SHM: do not try to lookup additional lanes when
         * SHM transport detected (another transport will be significantly
         * slower) */
        goto out_free_address_list;
    }

    while (address_count > 0) {
        status = ucp_wireup_select_transport(ep, address_list_copy, address_count,
                                             &mem_criteria, tl_bitmap, remote_md_map,
                                             -1, -1, 0, &rsc_index,
                                             &addr_index, &score);
        if ((status != UCS_OK) ||
            (select_best && (score <= reg_score))) {
            break;
        }

        /* Add lane description and remove all occurrences of the remote md */
        dst_md_index = address_list_copy[addr_index].md_index;
        ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                                 dst_md_index, score, usage, 0);
        remote_md_map &= ~UCS_BIT(dst_md_index);
        tl_bitmap = ucp_wireup_unset_tl_by_md(ep, tl_bitmap, rsc_index);
    }

    status = UCS_OK;

out_free_address_list:
    ucs_free(address_list_copy);
out:
    return select_best ? status : UCS_OK;
}

static uint64_t ucp_ep_get_context_features(ucp_ep_h ep)
{
    return ep->worker->context->config.features;
}

static double ucp_wireup_rma_score_func(ucp_context_h context,
                                        const uct_md_attr_t *md_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* best for 4k messages */
    return 1e-3 / (ucp_wireup_tl_iface_latency(context, iface_attr, remote_iface_attr) +
                   iface_attr->overhead +
                   (4096.0 / ucs_min(iface_attr->bandwidth, remote_iface_attr->bandwidth)));
}

static int ucp_wireup_ep_params_is_err_mode_peer(const ucp_ep_params_t *params)
{
    return (params->field_mask & UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE) &&
           (params->err_mode == UCP_ERR_HANDLING_MODE_PEER);
}

static void ucp_wireup_fill_ep_params_criteria(ucp_wireup_criteria_t *criteria,
                                               const ucp_ep_params_t *params)
{
    if (ucp_wireup_ep_params_is_err_mode_peer(params)) {
        criteria->local_iface_flags |= UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
    }
}

static void ucp_wireup_fill_aux_criteria(ucp_wireup_criteria_t *criteria,
                                         const ucp_ep_params_t *params)
{
    criteria->title              = "auxiliary";
    criteria->local_md_flags     = 0;
    criteria->remote_md_flags    = 0;
    criteria->local_iface_flags  = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                   UCT_IFACE_FLAG_AM_BCOPY |
                                   UCT_IFACE_FLAG_PENDING;
    criteria->remote_iface_flags = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                   UCT_IFACE_FLAG_AM_BCOPY |
                                   UCT_IFACE_FLAG_CB_ASYNC;
    criteria->calc_score         = ucp_wireup_aux_score_func;
    criteria->tl_rsc_flags       = UCP_TL_RSC_FLAG_AUX; /* Can use aux transports */

    ucp_wireup_fill_ep_params_criteria(criteria, params);
}

static void ucp_wireup_clean_amo_criteria(ucp_wireup_criteria_t *criteria)
{
    memset(&criteria->remote_atomic_flags, 0,
           sizeof(criteria->remote_atomic_flags));
    memset(&criteria->local_atomic_flags, 0,
           sizeof(criteria->local_atomic_flags));
}

static ucs_status_t ucp_wireup_add_rma_lanes(ucp_ep_h ep, const ucp_ep_params_t *params,
                                             unsigned ep_init_flags, unsigned address_count,
                                             const ucp_address_entry_t *address_list,
                                             ucp_wireup_lane_desc_t *lane_descs,
                                             ucp_lane_index_t *num_lanes_p,
                                             int *need_am)
{
    ucp_wireup_criteria_t criteria = {0};
    ucs_status_t status;
    int allow_am;

    if (!(ucp_ep_get_context_features(ep) & UCP_FEATURE_RMA) &&
        !(ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE)) {
        return UCS_OK;
    }

    if (ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE) {
        criteria.title              = "copy across memory types";
        criteria.remote_iface_flags = UCT_IFACE_FLAG_PUT_SHORT;
        criteria.local_iface_flags  = criteria.remote_iface_flags;
    } else {
        criteria.title              = "remote %s memory access";
        criteria.remote_iface_flags = UCT_IFACE_FLAG_PUT_SHORT |
                                      UCT_IFACE_FLAG_PUT_BCOPY |
                                      UCT_IFACE_FLAG_GET_BCOPY;
        criteria.local_iface_flags  = criteria.remote_iface_flags |
                                      UCT_IFACE_FLAG_PENDING;
    }
    criteria.calc_score         = ucp_wireup_rma_score_func;
    criteria.tl_rsc_flags       = 0;
    ucp_wireup_fill_ep_params_criteria(&criteria, params);

    allow_am = !(ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE) &&
               !ucp_wireup_ep_params_is_err_mode_peer(params);
    status = ucp_wireup_add_memaccess_lanes(ep, address_count, address_list,
                                            lane_descs, num_lanes_p, &criteria,
                                            -1, UCP_WIREUP_LANE_USAGE_RMA, 1,
                                            !allow_am);
    if (status == UCS_OK) {
        return status; /* using transport RMA operations */
    } else if (allow_am) {
        *need_am = 1;  /* using emulation over active messages */
        return UCS_OK;
    } else {
        return status;
    }
}

double ucp_wireup_amo_score_func(ucp_context_h context,
                                 const uct_md_attr_t *md_attr,
                                 const uct_iface_attr_t *iface_attr,
                                 const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* best one-sided latency */
    return 1e-3 / (ucp_wireup_tl_iface_latency(context, iface_attr, remote_iface_attr) +
                   iface_attr->overhead);
}

static ucs_status_t ucp_wireup_add_amo_lanes(ucp_ep_h ep, const ucp_ep_params_t *params,
                                             unsigned ep_init_flags,
                                             unsigned address_count,
                                             const ucp_address_entry_t *address_list,
                                             ucp_wireup_lane_desc_t *lane_descs,
                                             ucp_lane_index_t *num_lanes_p)
{
    ucp_worker_h worker            = ep->worker;
    ucp_context_h context          = worker->context;
    ucp_wireup_criteria_t criteria = {0};
    ucp_rsc_index_t rsc_index;
    uint64_t tl_bitmap;

    if (!ucs_test_flags(context->config.features, UCP_FEATURE_AMO32, UCP_FEATURE_AMO64) ||
        (ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE)) {
        return UCS_OK;
    }

    ucp_context_uct_atomic_iface_flags(context, &criteria.remote_atomic_flags);

    criteria.title              = "atomic operations on %s memory";
    criteria.local_iface_flags  = criteria.remote_iface_flags |
                                  UCT_IFACE_FLAG_PENDING;
    criteria.local_atomic_flags = criteria.remote_atomic_flags;
    criteria.calc_score         = ucp_wireup_amo_score_func;
    ucp_wireup_fill_ep_params_criteria(&criteria, params);

    /* We can use only non-p2p resources or resources which are explicitly
     * selected for atomics. Otherwise, the remote peer would not be able to
     * connect back on p2p transport.
     */
    tl_bitmap = worker->atomic_tls;
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        if (!ucp_worker_is_tl_p2p(worker, rsc_index)) {
            tl_bitmap |= UCS_BIT(rsc_index);
        }
    }

    return ucp_wireup_add_memaccess_lanes(ep, address_count, address_list,
                                          lane_descs, num_lanes_p, &criteria,
                                          tl_bitmap, UCP_WIREUP_LANE_USAGE_AMO,
                                          1, 1);
}

static double ucp_wireup_am_score_func(ucp_context_h context,
                                       const uct_md_attr_t *md_attr,
                                       const uct_iface_attr_t *iface_attr,
                                       const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* best end-to-end latency */
    return 1e-3 / (ucp_wireup_tl_iface_latency(context, iface_attr, remote_iface_attr) +
                   iface_attr->overhead + remote_iface_attr->overhead);
}

static double ucp_wireup_rma_bw_score_func(ucp_context_h context,
                                           const uct_md_attr_t *md_attr,
                                           const uct_iface_attr_t *iface_attr,
                                           const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* highest bandwidth with lowest overhead - test a message size of 256KB,
     * a size which is likely to be used for high-bw memory access protocol, for
     * how long it would take to transfer it with a certain transport. */
    return 1 / ((UCP_WIREUP_RMA_BW_TEST_MSG_SIZE /
                ucs_min(iface_attr->bandwidth, remote_iface_attr->bandwidth)) +
                ucp_wireup_tl_iface_latency(context, iface_attr, remote_iface_attr) +
                iface_attr->overhead + md_attr->reg_cost.overhead +
                (UCP_WIREUP_RMA_BW_TEST_MSG_SIZE * md_attr->reg_cost.growth));
}

static int ucp_wireup_is_lane_proxy(ucp_ep_h ep, ucp_rsc_index_t rsc_index,
                                    uint64_t remote_cap_flags)
{
    return !ucp_worker_is_tl_p2p(ep->worker, rsc_index) &&
           ((remote_cap_flags & UCP_WORKER_UCT_RECV_EVENT_CAP_FLAGS) ==
            UCT_IFACE_FLAG_EVENT_RECV_SIG);
}

static inline int ucp_wireup_is_am_required(ucp_ep_h ep,
                                            const ucp_ep_params_t *params,
                                            unsigned ep_init_flags,
                                            ucp_wireup_lane_desc_t *lane_descs,
                                            int num_lanes_p)
{
    ucp_lane_index_t lane;

    /* Check if we need active messages from the configurations, for wireup.
     * If not, check if am is required due to p2p transports */

    if ((ep_init_flags & UCP_EP_CREATE_AM_LANE) ||
        (params->field_mask & UCP_EP_PARAM_FIELD_SOCK_ADDR)) {
        return 1;
    }

    if (!(ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE) &&
        (ucp_ep_get_context_features(ep) & (UCP_FEATURE_TAG | UCP_FEATURE_STREAM))) {
        return 1;
    }

    for (lane = 0; lane < num_lanes_p; ++lane) {
        if (ucp_worker_is_tl_p2p(ep->worker, lane_descs[lane].rsc_index)) {
            return 1;
        }
    }

    return 0;
}

static ucs_status_t ucp_wireup_add_am_lane(ucp_ep_h ep, const ucp_ep_params_t *params,
                                           unsigned ep_init_flags, unsigned address_count,
                                           const ucp_address_entry_t *address_list,
                                           ucp_wireup_lane_desc_t *lane_descs,
                                           ucp_lane_index_t *num_lanes_p,
                                           double *am_score,
                                           ucp_err_handling_mode_t err_mode)
{
    ucp_wireup_criteria_t criteria = {0};
    ucp_rsc_index_t rsc_index;
    ucs_status_t status;
    unsigned addr_index;
    int is_proxy;

    if (!ucp_wireup_is_am_required(ep, params, ep_init_flags, lane_descs,
                                   *num_lanes_p)) {
        return UCS_OK;
    }

    /* Select one lane for active messages */
    criteria.title              = "active messages";
    criteria.remote_iface_flags = UCT_IFACE_FLAG_AM_BCOPY |
                                  UCT_IFACE_FLAG_CB_SYNC;
    criteria.local_iface_flags  = UCT_IFACE_FLAG_AM_BCOPY;
    criteria.calc_score         = ucp_wireup_am_score_func;
    ucp_wireup_fill_ep_params_criteria(&criteria, params);

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep), UCP_FEATURE_TAG |
                                                            UCP_FEATURE_WAKEUP)) {
        criteria.local_iface_flags |= UCP_WORKER_UCT_UNSIG_EVENT_CAP_FLAGS;
    }

    status = ucp_wireup_select_transport(ep, address_list, address_count, &criteria,
                                         -1, -1, -1, -1, 1, &rsc_index, &addr_index,
                                         am_score);
    if (status != UCS_OK) {
        return status;
    }

    /* If the remote side is not p2p and has only signaled-am wakeup, it may
     * deactivate its interface and wait for signaled active message to wake up.
     * Use a proxy lane which would send the first active message as signaled to
     * make sure the remote interface will indeed wake up.
     */
    is_proxy = ucp_wireup_is_lane_proxy(ep, rsc_index,
                                        address_list[addr_index].iface_attr.cap_flags);

    ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                             address_list[addr_index].md_index, *am_score,
                             UCP_WIREUP_LANE_USAGE_AM, is_proxy);

    return UCS_OK;
}

static double ucp_wireup_am_bw_score_func(ucp_context_h context,
                                          const uct_md_attr_t *md_attr,
                                          const uct_iface_attr_t *iface_attr,
                                          const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* best single MTU bandwidth */
    double size = iface_attr->cap.am.max_bcopy;
    double time = (size / ucs_min(iface_attr->bandwidth,
                                  remote_iface_attr->bandwidth)) +
                  iface_attr->overhead + remote_iface_attr->overhead;
    return size / time * 1e-5;
}

static int ucp_wireup_is_ep_single_lane(ucp_ep_h ep, ucp_rsc_index_t rsc_index)
{
    return (ep->worker->context->tl_rscs[rsc_index].tl_rsc.dev_type == UCT_DEVICE_TYPE_SHM) ||
           (ep->worker->context->tl_rscs[rsc_index].tl_rsc.dev_type == UCT_DEVICE_TYPE_SELF);
}

static ucs_status_t ucp_wireup_add_bw_lanes(ucp_ep_h ep,
                                            unsigned address_count,
                                            const ucp_address_entry_t *address_list,
                                            const ucp_wireup_select_bw_info_t *bw_info,
                                            int allow_proxy, uint64_t tl_bitmap,
                                            ucp_wireup_lane_desc_t *lane_descs,
                                            ucp_lane_index_t *num_lanes_p)
{
    ucp_context_h context = ep->worker->context;
    ucs_status_t status;
    int num_lanes;
    uint64_t local_dev_bitmap;
    uint64_t remote_dev_bitmap;
    ucp_md_map_t md_map;
    ucp_rsc_index_t rsc_index;
    unsigned addr_index;
    double score;
    int is_proxy;

    status             = UCS_ERR_UNREACHABLE;
    num_lanes          = 0;
    md_map             = bw_info->md_map;
    local_dev_bitmap   = bw_info->local_dev_bitmap;
    remote_dev_bitmap  = bw_info->remote_dev_bitmap;

    /* lookup for requested number of lanes or limit of MD map
     * (we have to limit MD's number to avoid malloc in
     * memory registration) */
    while ((num_lanes < bw_info->max_lanes) &&
           (ucs_count_one_bits(md_map) < UCP_MAX_OP_MDS)) {
        status = ucp_wireup_select_transport(ep, address_list, address_count,
                                             &bw_info->criteria, tl_bitmap, -1,
                                             local_dev_bitmap, remote_dev_bitmap,
                                             0, &rsc_index, &addr_index, &score);
        if (status != UCS_OK) {
            break;
        }

        is_proxy = allow_proxy &&
                   ucp_wireup_is_lane_proxy(ep, rsc_index,
                                            address_list[addr_index].iface_attr.cap_flags);

        ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                                 address_list[addr_index].md_index, score,
                                 bw_info->usage, is_proxy);
        md_map |= UCS_BIT(context->tl_rscs[rsc_index].md_index);
        num_lanes++;

        local_dev_bitmap  &= ~UCS_BIT(context->tl_rscs[rsc_index].dev_index);
        remote_dev_bitmap &= ~UCS_BIT(address_list[addr_index].dev_index);

        if (ucp_wireup_is_ep_single_lane(ep, rsc_index)) {
            /* special case for SHM: do not try to lookup additional lanes when
             * SHM transport detected (another transport will be significantly
             * slower) */
            break;
        }
    }

    return UCS_OK;
}

static ucs_status_t ucp_wireup_add_am_bw_lanes(ucp_ep_h ep, const ucp_ep_params_t *params,
                                               unsigned ep_init_flags, unsigned address_count,
                                               const ucp_address_entry_t *address_list,
                                               ucp_wireup_lane_desc_t *lane_descs,
                                               ucp_lane_index_t *num_lanes_p)
{
    ucp_context_h context = ep->worker->context;
    ucp_wireup_select_bw_info_t bw_info;
    ucp_lane_index_t lane_desc_idx;
    ucp_rsc_index_t rsc_index;
    unsigned addr_index;

    /* Check if we need active messages, for wireup */
    if (!(ucp_ep_get_context_features(ep) & UCP_FEATURE_TAG) ||
        (ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE)          ||
        (ep->worker->context->config.ext.max_eager_lanes < 2)) {
        return UCS_OK;
    }

    /* Select one lane for active messages */
    bw_info.criteria.title              = "high-bw active messages";
    bw_info.criteria.local_md_flags     = 0;
    bw_info.criteria.remote_md_flags    = 0;
    bw_info.criteria.remote_iface_flags = UCT_IFACE_FLAG_AM_BCOPY |
                                          UCT_IFACE_FLAG_CB_SYNC;
    bw_info.criteria.local_iface_flags  = UCT_IFACE_FLAG_AM_BCOPY;
    bw_info.criteria.calc_score         = ucp_wireup_am_bw_score_func;
    bw_info.criteria.tl_rsc_flags       = 0;
    ucp_wireup_clean_amo_criteria(&bw_info.criteria);
    ucp_wireup_fill_ep_params_criteria(&bw_info.criteria, params);

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep), UCP_FEATURE_TAG |
                                                            UCP_FEATURE_WAKEUP)) {
        bw_info.criteria.local_iface_flags |= UCP_WORKER_UCT_UNSIG_EVENT_CAP_FLAGS;
    }

    bw_info.local_dev_bitmap  = -1;
    bw_info.remote_dev_bitmap = -1;
    bw_info.md_map            = 0;
    bw_info.max_lanes         = ep->worker->context->config.ext.max_eager_lanes - 1;
    bw_info.usage             = UCP_WIREUP_LANE_USAGE_AM_BW;

    /* am_bw_lane[0] is am_lane, so don't re-select it here */
    for (lane_desc_idx = 0; lane_desc_idx < *num_lanes_p; ++lane_desc_idx) {
        if (lane_descs[lane_desc_idx].usage & UCP_WIREUP_LANE_USAGE_AM) {
            addr_index                 = lane_descs[lane_desc_idx].addr_index;
            rsc_index                  = lane_descs[lane_desc_idx].rsc_index;
            bw_info.md_map            |= UCS_BIT(context->tl_rscs[rsc_index].md_index);
            bw_info.local_dev_bitmap  &= ~UCS_BIT(context->tl_rscs[rsc_index].dev_index);
            bw_info.remote_dev_bitmap &= ~UCS_BIT(address_list[addr_index].dev_index);
            if (ucp_wireup_is_ep_single_lane(ep, rsc_index)) {
                /* if AM lane is SELF or SHMEM - then do not use more lanes */
                return UCS_OK;
            } else {
                break; /* do not continue searching due to we found
                          AM lane (and there is only one lane) */
            }
        }
    }

    return ucp_wireup_add_bw_lanes(ep, address_count, address_list, &bw_info, 1,
                                   -1, lane_descs, num_lanes_p);
}

static ucs_status_t ucp_wireup_add_rma_bw_lanes(ucp_ep_h ep,
                                                const ucp_ep_params_t *params,
                                                unsigned ep_init_flags, unsigned address_count,
                                                const ucp_address_entry_t *address_list,
                                                ucp_wireup_lane_desc_t *lane_descs,
                                                ucp_lane_index_t *num_lanes_p)
{
    ucp_wireup_select_bw_info_t bw_info;
    uct_memory_type_t mem_type;

    if ((ucp_ep_get_context_features(ep) & UCP_FEATURE_RMA) ||
        (ep_init_flags & UCP_EP_INIT_FLAG_MEM_TYPE)) {
        /* if needed for RMA, need also access for remote allocated memory */
        bw_info.criteria.remote_md_flags = bw_info.criteria.local_md_flags = 0;
    } else {
        if (!(ucp_ep_get_context_features(ep) & UCP_FEATURE_TAG)) {
            return UCS_OK;
        }
        /* if needed for RNDV, need only access for remote registered memory */
        bw_info.criteria.remote_md_flags = bw_info.criteria.local_md_flags = UCT_MD_FLAG_REG;
    }

    bw_info.criteria.title              = "high-bw remote memory access";
    bw_info.criteria.remote_iface_flags = UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY;
    bw_info.criteria.local_iface_flags  = bw_info.criteria.remote_iface_flags |
                                          UCT_IFACE_FLAG_PENDING;
    bw_info.criteria.calc_score         = ucp_wireup_rma_bw_score_func;
    bw_info.criteria.tl_rsc_flags       = 0;
    ucp_wireup_clean_amo_criteria(&bw_info.criteria);
    ucp_wireup_fill_ep_params_criteria(&bw_info.criteria, params);

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep),
                           UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP)) {
        bw_info.criteria.local_iface_flags |= UCP_WORKER_UCT_UNSIG_EVENT_CAP_FLAGS;
    }

    bw_info.local_dev_bitmap  = -1;
    bw_info.remote_dev_bitmap = -1;
    bw_info.md_map            = 0;
    bw_info.max_lanes         = ep->worker->context->config.ext.max_rndv_lanes;
    bw_info.usage             = UCP_WIREUP_LANE_USAGE_RMA_BW;

    for (mem_type = 0; mem_type < UCT_MD_MEM_TYPE_LAST; mem_type++) {
        if (!ep->worker->context->mem_type_tls[mem_type]) {
            continue;
        }

        ucp_wireup_add_bw_lanes(ep, address_count, address_list, &bw_info, 0,
                                ep->worker->context->mem_type_tls[mem_type],
                                lane_descs, num_lanes_p);
    }

    return UCS_OK;
}

/* Lane for transport offloaded tag interface */
static ucs_status_t ucp_wireup_add_tag_lane(ucp_ep_h ep, unsigned address_count,
                                            const ucp_address_entry_t *address_list,
                                            ucp_wireup_lane_desc_t *lane_descs,
                                            ucp_lane_index_t *num_lanes_p,
                                            double am_score,
                                            ucp_err_handling_mode_t err_mode)
{
    ucp_wireup_criteria_t criteria = {0};
    ucp_rsc_index_t rsc_index;
    ucs_status_t status;
    unsigned addr_index;
    double score;
    int is_proxy;

    if (!(ucp_ep_get_context_features(ep) & UCP_FEATURE_TAG) ||
        /* TODO: remove check below when UCP_ERR_HANDLING_MODE_PEER supports
         *       RNDV-protocol or HW TM supports fragmented protocols
         */
        (err_mode != UCP_ERR_HANDLING_MODE_NONE)) {
        return UCS_OK;
    }

    criteria.title              = "tag_offload";
    criteria.local_md_flags     = UCT_MD_FLAG_REG; /* needed for posting tags to HW */
    criteria.remote_md_flags    = UCT_MD_FLAG_REG; /* needed for posting tags to HW */
    criteria.remote_iface_flags = /* the same as local_iface_flags */
    criteria.local_iface_flags  = UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
                                  UCT_IFACE_FLAG_TAG_RNDV_ZCOPY  |
                                  UCT_IFACE_FLAG_GET_ZCOPY       |
                                  UCT_IFACE_FLAG_PENDING;
    criteria.calc_score         = ucp_wireup_am_score_func;

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep), UCP_FEATURE_WAKEUP)) {
        criteria.local_iface_flags |= UCP_WORKER_UCT_UNSIG_EVENT_CAP_FLAGS;
    }

    /* Do not add tag offload lane, if selected tag lane score is lower
     * than AM score. In this case AM will be used for tag macthing. */
    status = ucp_wireup_select_transport(ep, address_list, address_count, &criteria,
                                         -1, -1, -1, -1, 0, &rsc_index, &addr_index,
                                         &score);
    if ((status != UCS_OK) || (am_score > score)) {
        goto out;
    }

    /* If the remote side is not p2p and has only signaled wakeup, it may
     * deactivate its interface and wait for signaled tag message to wake up.
     * Use a proxy lane which would send the first tag message as signaled to
     * make sure the remote interface will indeed wake up.
     */
    is_proxy = ucp_wireup_is_lane_proxy(ep, rsc_index,
                                        address_list[addr_index].iface_attr.cap_flags);

    ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                             address_list[addr_index].md_index, score,
                             UCP_WIREUP_LANE_USAGE_TAG, is_proxy);

out:
    return UCS_OK;
}

static ucp_lane_index_t
ucp_wireup_select_wireup_msg_lane(ucp_worker_h worker,
                                  const ucp_ep_params_t *ep_params,
                                  const ucp_address_entry_t *address_list,
                                  const ucp_wireup_lane_desc_t *lane_descs,
                                  ucp_lane_index_t num_lanes)
{
    ucp_context_h context          = worker->context;
    ucp_lane_index_t p2p_lane      = UCP_NULL_LANE;
    ucp_wireup_criteria_t criteria = {0};
    uct_tl_resource_desc_t *resource;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    unsigned addr_index;

    ucp_wireup_fill_aux_criteria(&criteria, ep_params);
    for (lane = 0; lane < num_lanes; ++lane) {
        rsc_index  = lane_descs[lane].rsc_index;
        addr_index = lane_descs[lane].addr_index;
        resource   = &context->tl_rscs[rsc_index].tl_rsc;

        /* if the current lane satisfies the wireup criteria, choose it for wireup.
         * if it doesn't take a lane with a p2p transport */
        if (ucp_wireup_check_flags(resource,
                                   worker->ifaces[rsc_index].attr.cap.flags,
                                   criteria.local_iface_flags, criteria.title,
                                   ucp_wireup_iface_flags, NULL, 0) &&
            ucp_wireup_check_flags(resource,
                                   address_list[addr_index].iface_attr.cap_flags,
                                   criteria.remote_iface_flags, criteria.title,
                                   ucp_wireup_iface_flags, NULL, 0))
         {
             return lane;
         } else if (ucp_worker_is_tl_p2p(worker, rsc_index)) {
             p2p_lane = lane;
         }
    }

    return p2p_lane;
}

static uint64_t
ucp_wireup_get_reachable_mds(ucp_worker_h worker, unsigned address_count,
                             const ucp_address_entry_t *address_list)
{
    ucp_context_h context  = worker->context;
    uint64_t reachable_mds = 0;
    const ucp_address_entry_t *ae;
    ucp_rsc_index_t rsc_index;

    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        for (ae = address_list; ae < address_list + address_count; ++ae) {
            if (ucp_wireup_is_reachable(worker, rsc_index, ae)) {
                reachable_mds |= UCS_BIT(ae->md_index);
            }
        }
    }

    return reachable_mds;
}

ucs_status_t ucp_wireup_select_lanes(ucp_ep_h ep, const ucp_ep_params_t *params,
                                     unsigned ep_init_flags, unsigned address_count,
                                     const ucp_address_entry_t *address_list,
                                     uint8_t *addr_indices,
                                     ucp_ep_config_key_t *key)
{
    ucp_worker_h worker   = ep->worker;
    ucp_context_h context = worker->context;
    double am_score       = 0.0;
    ucp_wireup_lane_desc_t lane_descs[UCP_MAX_LANES];
    ucp_rsc_index_t rsc_index;
    ucp_md_index_t md_index;
    ucp_lane_index_t lane;
    ucp_lane_index_t i;
    ucs_status_t status;
    int need_am = 0;

    memset(lane_descs, 0, sizeof(lane_descs));
    ucp_ep_config_key_reset(key);
    ucp_ep_config_key_set_params(key, params);

    status = ucp_wireup_add_rma_lanes(ep, params, ep_init_flags, address_count,
                                      address_list, lane_descs, &key->num_lanes,
                                      &need_am);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_wireup_add_amo_lanes(ep, params, ep_init_flags, address_count,
                                      address_list, lane_descs, &key->num_lanes);
    if (status != UCS_OK) {
        return status;
    }

    if (need_am) {
        ep_init_flags |= UCP_EP_CREATE_AM_LANE;
    }

    status = ucp_wireup_add_am_lane(ep, params, ep_init_flags, address_count,
                                    address_list, lane_descs, &key->num_lanes,
                                    &am_score, key->err_mode);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_wireup_add_rma_bw_lanes(ep, params, ep_init_flags, address_count,
                                         address_list, lane_descs, &key->num_lanes);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_wireup_add_tag_lane(ep, address_count, address_list,
                                     lane_descs, &key->num_lanes, am_score,
                                     key->err_mode);
    if (status != UCS_OK) {
        return status;
    }

    /* call ucp_wireup_add_am_bw_lanes after ucp_wireup_add_am_lane to
     * allow exclude AM lane from AM_BW list */
    status = ucp_wireup_add_am_bw_lanes(ep, params, ep_init_flags, address_count,
                                        address_list, lane_descs, &key->num_lanes);
    if (status != UCS_OK) {
        return status;
    }

    /* User should not create endpoints unless requested communication features */
    if (key->num_lanes == 0) {
        ucs_error("No transports selected to %s (features: 0x%lx)",
                  ucp_ep_peer_name(ep), ucp_ep_get_context_features(ep));
        return UCS_ERR_UNREACHABLE;
    }

    /* Construct the endpoint configuration key:
     * - arrange lane description in the EP configuration
     * - create remote MD bitmap
     * - if AM lane exists and fits for wireup messages, select it for this purpose.
     */
    for (lane = 0; lane < key->num_lanes; ++lane) {
        ucs_assert(lane_descs[lane].usage != 0);
        key->lanes[lane].rsc_index    = lane_descs[lane].rsc_index;
        key->lanes[lane].proxy_lane   = lane_descs[lane].proxy_lane;
        key->lanes[lane].dst_md_index = lane_descs[lane].dst_md_index;
        addr_indices[lane]            = lane_descs[lane].addr_index;

        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AM) {
            ucs_assert(key->am_lane == UCP_NULL_LANE);
            key->am_lane = lane;
        }
        if ((lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AM_BW) &&
            (lane < UCP_MAX_LANES - 1)) {
            key->am_bw_lanes[lane + 1] = lane;
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_RMA) {
            key->rma_lanes[lane] = lane;
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_RMA_BW) {
            key->rma_bw_lanes[lane] = lane;
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AMO) {
            key->amo_lanes[lane] = lane;
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_TAG) {
            ucs_assert(key->tag_lane == UCP_NULL_LANE);
            key->tag_lane = lane;
        }
    }

    /* Sort AM, RMA and AMO lanes according to score */
    ucs_qsort_r(key->am_bw_lanes + 1, UCP_MAX_LANES - 1, sizeof(ucp_lane_index_t),
                ucp_wireup_compare_lane_am_bw_score, lane_descs);
    ucs_qsort_r(key->rma_lanes, UCP_MAX_LANES, sizeof(ucp_lane_index_t),
                ucp_wireup_compare_lane_rma_score, lane_descs);
    ucs_qsort_r(key->rma_bw_lanes, UCP_MAX_LANES, sizeof(ucp_lane_index_t),
                ucp_wireup_compare_lane_rma_bw_score, lane_descs);
    ucs_qsort_r(key->amo_lanes, UCP_MAX_LANES, sizeof(ucp_lane_index_t),
                ucp_wireup_compare_lane_amo_score, lane_descs);

    /* Get all reachable MDs from full remote address list */
    key->reachable_md_map = ucp_wireup_get_reachable_mds(worker, address_count,
                                                         address_list);

    /* Select lane for wireup messages */
    key->wireup_lane  = ucp_wireup_select_wireup_msg_lane(worker, params,
                                                          address_list,
                                                          lane_descs,
                                                          key->num_lanes);

    /* add to map first UCP_MAX_OP_MDS fastest MD's */
    for (i = 0;
         (key->rma_bw_lanes[i] != UCP_NULL_LANE) &&
         (ucs_count_one_bits(key->rma_bw_md_map) < UCP_MAX_OP_MDS); i++) {
        lane = key->rma_bw_lanes[i];
        rsc_index = lane_descs[lane].rsc_index;
        md_index  = worker->context->tl_rscs[rsc_index].md_index;

        /* Pack remote key only if needed for RMA.
         * FIXME a temporary workaround to prevent the ugni uct from using rndv. */
        if ((context->tl_mds[md_index].attr.cap.flags & UCT_MD_FLAG_NEED_RKEY) &&
            !(strstr(context->tl_rscs[rsc_index].tl_rsc.tl_name, "ugni"))) {
            key->rma_bw_md_map |= UCS_BIT(md_index);
        }
    }

    /* use AM lane first for eager AM transport to simplify processing single/middle
     * msg packets */
    key->am_bw_lanes[0] = key->am_lane;

    return UCS_OK;
}

static double ucp_wireup_aux_score_func(ucp_context_h context,
                                        const uct_md_attr_t *md_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* best end-to-end latency and larger bcopy size */
    return (1e-3 / (ucp_wireup_tl_iface_latency(context, iface_attr, remote_iface_attr) +
            iface_attr->overhead + remote_iface_attr->overhead));
}

ucs_status_t ucp_wireup_select_aux_transport(ucp_ep_h ep,
                                             const ucp_ep_params_t *params,
                                             const ucp_address_entry_t *address_list,
                                             unsigned address_count,
                                             ucp_rsc_index_t *rsc_index_p,
                                             unsigned *addr_index_p)
{
    ucp_wireup_criteria_t criteria = {0};
    double score;

    ucp_wireup_fill_aux_criteria(&criteria, params);
    return ucp_wireup_select_transport(ep, address_list, address_count,
                                       &criteria, -1, -1, -1, -1, 1, rsc_index_p,
                                       addr_index_p, &score);
}

ucs_status_t ucp_wireup_select_sockaddr_transport(ucp_ep_h ep,
                                                  const ucp_ep_params_t *params,
                                                  ucp_rsc_index_t *rsc_index_p)
{
    ucp_worker_h worker   = ep->worker;
    ucp_context_h context = worker->context;
    char saddr_str[UCS_SOCKADDR_STRING_LEN];
    ucp_tl_resource_desc_t *resource;
    ucp_rsc_index_t tl_id;
    ucp_md_index_t md_index;
    uct_md_h md;

    for (tl_id = 0; tl_id < context->num_tls; ++tl_id) {
        resource = &context->tl_rscs[tl_id];
        if (!(resource->flags & UCP_TL_RSC_FLAG_SOCKADDR)) {
            continue;
        }

        md_index = resource->md_index;
        md       = context->tl_mds[md_index].md;
        ucs_assert(context->tl_mds[md_index].attr.cap.flags & UCT_MD_FLAG_SOCKADDR);

        if (uct_md_is_sockaddr_accessible(md, &params->sockaddr, UCT_SOCKADDR_ACC_REMOTE)) {
            /* TODO use score to prefer best tl rather than using first one */
            *rsc_index_p = tl_id;
            return UCS_OK;
        }

        ucs_debug("md %s cannot reach %s", context->tl_mds[md_index].rsc.md_name,
                  ucs_sockaddr_str(params->sockaddr.addr, saddr_str, sizeof(saddr_str)));
    }

    return UCS_ERR_UNREACHABLE;
}
