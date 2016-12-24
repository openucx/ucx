/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "wireup.h"
#include "address.h"

#include <ucs/algorithm/qsort_r.h>
#include <ucp/core/ucp_ep.inl>
#include <string.h>
#include <inttypes.h>


enum {
    UCP_WIREUP_LANE_USAGE_AM   = UCS_BIT(0),
    UCP_WIREUP_LANE_USAGE_RMA  = UCS_BIT(1),
    UCP_WIREUP_LANE_USAGE_AMO  = UCS_BIT(2),
    UCP_WIREUP_LANE_USAGE_RNDV = UCS_BIT(3)
};


typedef struct {
    ucp_rsc_index_t   rsc_index;
    unsigned          addr_index;
    ucp_rsc_index_t   dst_md_index;
    uint32_t          usage;
    double            rma_score;
    double            amo_score;
} ucp_wireup_lane_desc_t;


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
    [ucs_ilog2(UCT_IFACE_FLAG_ATOMIC_ADD32)]     = "32-bit atomic add",
    [ucs_ilog2(UCT_IFACE_FLAG_ATOMIC_ADD64)]     = "64-bit atomic add",
    [ucs_ilog2(UCT_IFACE_FLAG_ATOMIC_FADD32)]    = "32-bit atomic fetch-add",
    [ucs_ilog2(UCT_IFACE_FLAG_ATOMIC_FADD64)]    = "64-bit atomic fetch-add",
    [ucs_ilog2(UCT_IFACE_FLAG_ATOMIC_SWAP32)]    = "32-bit atomic swap",
    [ucs_ilog2(UCT_IFACE_FLAG_ATOMIC_SWAP64)]    = "64-bit atomic swap",
    [ucs_ilog2(UCT_IFACE_FLAG_ATOMIC_CSWAP32)]   = "32-bit atomic compare-swap",
    [ucs_ilog2(UCT_IFACE_FLAG_ATOMIC_CSWAP64)]   = "64-bit atomic compare-swap",
    [ucs_ilog2(UCT_IFACE_FLAG_CONNECT_TO_IFACE)] = "connect to iface",
    [ucs_ilog2(UCT_IFACE_FLAG_CONNECT_TO_EP)]    = "connect to ep",
    [ucs_ilog2(UCT_IFACE_FLAG_AM_DUP)]           = "full reliability",
    [ucs_ilog2(UCT_IFACE_FLAG_AM_CB_SYNC)]       = "sync am callback",
    [ucs_ilog2(UCT_IFACE_FLAG_AM_CB_ASYNC)]      = "async am callback",
    [ucs_ilog2(UCT_IFACE_FLAG_WAKEUP)]           = "wakeup",
    [ucs_ilog2(UCT_IFACE_FLAG_PENDING)]          = "pending"
};

static double ucp_wireup_aux_score_func(const uct_md_attr_t *md_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_address_iface_attr_t *remote_iface_attr);

static ucp_wireup_criteria_t ucp_wireup_aux_criteria = {
    .title              = "auxiliary",
    .local_md_flags     = 0,
    .remote_md_flags    = 0,
    .local_iface_flags  = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                          UCT_IFACE_FLAG_AM_BCOPY |
                          UCT_IFACE_FLAG_PENDING,
    .remote_iface_flags = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                          UCT_IFACE_FLAG_AM_BCOPY |
                          UCT_IFACE_FLAG_AM_CB_ASYNC,
    .calc_score         = ucp_wireup_aux_score_func
};

static const char *
ucp_wireup_get_missing_flag_desc(uint64_t flags, uint64_t required_flags,
                                 const char ** flag_descs)
{
    ucs_assert((required_flags & (~flags)) != 0);
    return flag_descs[ucs_ffs64(required_flags & (~flags))];
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

static int ucp_wireup_is_reachable(ucp_worker_h worker, ucp_rsc_index_t rsc_index,
                                   const ucp_address_entry_t *ae)
{
    ucp_context_h context = worker->context;
    return (context->tl_rscs[rsc_index].tl_name_csum == ae->tl_name_csum) &&
           uct_iface_is_reachable(worker->ifaces[rsc_index], ae->dev_addr, ae->iface_addr);
}

/**
 * Select a local and remote transport
 */
static UCS_F_NOINLINE ucs_status_t
ucp_wireup_select_transport(ucp_ep_h ep, const ucp_address_entry_t *address_list,
                            unsigned address_count, const ucp_wireup_criteria_t *criteria,
                            uint64_t tl_bitmap, uint64_t remote_md_map, int show_error,
                            ucp_rsc_index_t *rsc_index_p, unsigned *dst_addr_index_p,
                            double *score_p)
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
        if (!(remote_md_map & UCS_BIT(ae->md_index))) {
            ucs_trace("addr[%d]: not in use, because on md[%d]", addr_index,
                      ae->md_index);
            continue;
        }
        if (!ucs_test_all_flags(ae->md_flags, criteria->remote_md_flags)) {
            ucs_trace("addr[%d]: no %s", addr_index,
                      ucp_wireup_get_missing_flag_desc(ae->md_flags,
                                                       criteria->remote_md_flags,
                                                       ucp_wireup_md_flags));
            continue;
        }

        /* Make sure we are indeed passing all flags required by the criteria in
         * ucp packed address */
        ucs_assert(ucs_test_all_flags(UCP_ADDRESS_IFACE_FLAGS, criteria->remote_iface_flags));

        if (!ucs_test_all_flags(ae->iface_attr.cap_flags, criteria->remote_iface_flags)) {
            ucs_trace("addr[%d]: no %s", addr_index,
                      ucp_wireup_get_missing_flag_desc(ae->iface_attr.cap_flags,
                                                       criteria->remote_iface_flags,
                                                       ucp_wireup_iface_flags));
            continue;
        }

        addr_index_map |= UCS_BIT(addr_index);
    }

    if (!addr_index_map) {
         snprintf(p, endp - p, "not supported by peer  ");
         p += strlen(p);
    }

    /* For each local resource try to find the best remote address to connect to.
     * Pick the best local resource to satisfy the criteria.
     * best one has the highest score (from the dedicated score_func) and
     * has a reachable tl on the remote peer */
    for (rsc_index = 0; addr_index_map && (rsc_index < context->num_tls); ++rsc_index) {
        resource     = &context->tl_rscs[rsc_index].tl_rsc;
        iface_attr   = &worker->iface_attrs[rsc_index];
        md_attr      = &context->md_attrs[context->tl_rscs[rsc_index].md_index];

        /* Check that local md and interface satisfy the criteria */
        if (!ucp_wireup_check_flags(resource, md_attr->cap.flags,
                                    criteria->local_md_flags, criteria->title,
                                    ucp_wireup_md_flags, p, endp - p) ||
            !ucp_wireup_check_flags(resource, iface_attr->cap.flags,
                                    criteria->local_iface_flags, criteria->title,
                                    ucp_wireup_iface_flags, p, endp - p))
        {
            p += strlen(p);
            snprintf(p, endp - p, ", ");
            p += strlen(p);
            continue;
        }

        /* Check supplied tl bitmap */
        if (!(tl_bitmap & UCS_BIT(rsc_index))) {
            ucs_trace(UCT_TL_RESOURCE_DESC_FMT " : disabled by tl_bitmap",
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

            score = criteria->calc_score(md_attr, iface_attr, &ae->iface_attr);
            ucs_assert(score >= 0.0);

            priority = iface_attr->priority + ae->iface_attr.priority;

            ucs_trace(UCT_TL_RESOURCE_DESC_FMT "->addr[%zd] : %s score %.2f",
                      UCT_TL_RESOURCE_DESC_ARG(resource), ae - address_list,
                      criteria->title, score);

            /* First comparing score, if score equals to current best score,
             * comparing priority with the priority of best score */
            if (!found || (score > best_score) ||
                ((score == best_score) && (priority > best_score_priority))) {
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
            snprintf(p, endp - p, UCT_TL_RESOURCE_DESC_FMT" - cannot reach peer, ",
                     UCT_TL_RESOURCE_DESC_ARG(resource));
            p += strlen(p);
        }
    }

    if (p >= tls_info + 2) {
        *(p - 2) = '\0'; /* trim last "," */
    }

    if (!found) {
        if (show_error) {
            ucs_error("No %s transport to %s: %s", criteria->title,
                      ucp_ep_peer_name(ep), tls_info);
        }
        return UCS_ERR_UNREACHABLE;
    }

    ucs_trace("ep %p: selected for %s: " UCT_TL_RESOURCE_DESC_FMT
              " -> '%s' address[%d],md[%d] score %.2f", ep, criteria->title,
              UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[*rsc_index_p].tl_rsc),
              ucp_ep_peer_name(ep), *dst_addr_index_p,
              address_list[*dst_addr_index_p].md_index, best_score);
    return UCS_OK;
}

static UCS_F_NOINLINE void
ucp_wireup_add_lane_desc(ucp_wireup_lane_desc_t *lane_descs,
                         ucp_lane_index_t *num_lanes_p, ucp_rsc_index_t rsc_index,
                         unsigned addr_index, ucp_rsc_index_t dst_md_index,
                         double score, uint32_t usage)
{
    ucp_wireup_lane_desc_t *lane_desc;

    for (lane_desc = lane_descs; lane_desc < lane_descs + (*num_lanes_p); ++lane_desc) {
        if ((lane_desc->rsc_index == rsc_index) &&
            (lane_desc->addr_index == addr_index))
        {
            ucs_assertv_always(dst_md_index == lane_desc->dst_md_index,
                               "lane[%d].dst_md_index=%d, dst_md_index=%d",
                               (int)(lane_desc - lane_descs), lane_desc->dst_md_index,
                               dst_md_index);
            ucs_assertv_always(!(lane_desc->usage & usage), "lane[%d]=0x%x |= 0x%x",
                               (int)(lane_desc - lane_descs), lane_desc->usage,
                               usage);
            lane_desc->usage |= usage;
            goto out_update_score;
        }
    }

    lane_desc = &lane_descs[*num_lanes_p];
    ++(*num_lanes_p);

    lane_desc->rsc_index    = rsc_index;
    lane_desc->addr_index   = addr_index;
    lane_desc->dst_md_index = dst_md_index;
    lane_desc->usage        = usage;
    lane_desc->rma_score    = 0.0;
    lane_desc->amo_score    = 0.0;

out_update_score:
    if (usage & UCP_WIREUP_LANE_USAGE_RMA) {
        lane_desc->rma_score = score;
    }
    if (usage & UCP_WIREUP_LANE_USAGE_AMO) {
        lane_desc->amo_score = score;
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
                               const ucp_wireup_criteria_t *criteria,
                               uint64_t tl_bitmap, uint32_t usage)
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
    mem_criteria.remote_md_flags = UCT_MD_FLAG_REG;
    status = ucp_wireup_select_transport(ep, address_list_copy, address_count,
                                         &mem_criteria, tl_bitmap, remote_md_map,
                                         1, &rsc_index, &addr_index, &score);
    if (status != UCS_OK) {
        goto out_free_address_list;
    }

    dst_md_index = address_list_copy[addr_index].md_index;
    reg_score    = score;

    /* Add to the list of lanes and remove all occurrences of the remote md
     * from the address list, to avoid selecting the same remote md again.*/
    ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                             dst_md_index, score, usage);
    remote_md_map &= ~UCS_BIT(dst_md_index);

    /* Select additional transports which can access allocated memory, but only
     * if their scores are better. We need this because a remote memory block can
     * be potentially allocated using one of them, and we might get better performance
     * than the transports which support only registered remote memory.
     */
    snprintf(title, sizeof(title), criteria->title, "allocated");
    mem_criteria.title           = title;
    mem_criteria.remote_md_flags = UCT_MD_FLAG_ALLOC;

    while (address_count > 0) {
        status = ucp_wireup_select_transport(ep, address_list_copy, address_count,
                                             &mem_criteria, tl_bitmap, remote_md_map,
                                             0, &rsc_index, &addr_index, &score);
        if ((status != UCS_OK) || (score <= reg_score)) {
            break;
        }

        /* Add lane description and remove all occurrences of the remote md */
        dst_md_index = address_list_copy[addr_index].md_index;
        ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                                 dst_md_index, score, usage);
        remote_md_map &= ~UCS_BIT(dst_md_index);
    }

    status = UCS_OK;

out_free_address_list:
    ucs_free(address_list_copy);
out:
    return status;
}

static uint64_t ucp_ep_get_context_features(ucp_ep_h ep)
{
    return ep->worker->context->config.features;
}

static double ucp_wireup_rma_score_func(const uct_md_attr_t *md_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* best for 4k messages */
    return 1e-3 / (iface_attr->latency + iface_attr->overhead +
                    (4096.0 / ucs_min(iface_attr->bandwidth, remote_iface_attr->bandwidth)));
}

static ucs_status_t ucp_wireup_add_rma_lanes(ucp_ep_h ep, unsigned address_count,
                                             const ucp_address_entry_t *address_list,
                                             ucp_wireup_lane_desc_t *lane_descs,
                                             ucp_lane_index_t *num_lanes_p)
{
    ucp_wireup_criteria_t criteria;

    if (!(ucp_ep_get_context_features(ep) & UCP_FEATURE_RMA)) {
        return UCS_OK;
    }

    criteria.title              = "remote %s memory access";
    criteria.local_md_flags     = 0;
    criteria.remote_md_flags    = 0;
    criteria.remote_iface_flags = UCT_IFACE_FLAG_PUT_SHORT |
                                  UCT_IFACE_FLAG_PUT_BCOPY |
                                  UCT_IFACE_FLAG_GET_BCOPY;
    criteria.local_iface_flags  = criteria.remote_iface_flags |
                                  UCT_IFACE_FLAG_PENDING;
    criteria.calc_score         = ucp_wireup_rma_score_func;

    return ucp_wireup_add_memaccess_lanes(ep, address_count, address_list,
                                          lane_descs, num_lanes_p, &criteria,
                                          -1, UCP_WIREUP_LANE_USAGE_RMA);
}

double ucp_wireup_amo_score_func(const uct_md_attr_t *md_attr,
                                 const uct_iface_attr_t *iface_attr,
                                 const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* best one-sided latency */
    return 1e-3 / (iface_attr->latency + iface_attr->overhead);
}

static ucs_status_t ucp_wireup_add_amo_lanes(ucp_ep_h ep, unsigned address_count,
                                             const ucp_address_entry_t *address_list,
                                             ucp_wireup_lane_desc_t *lane_descs,
                                             ucp_lane_index_t *num_lanes_p)
{
    ucp_worker_h worker   = ep->worker;
    ucp_context_h context = worker->context;
    ucp_wireup_criteria_t criteria;
    ucp_rsc_index_t rsc_index;
    uint64_t tl_bitmap;

    criteria.remote_iface_flags = ucp_context_uct_atomic_iface_flags(context);
    if (criteria.remote_iface_flags == 0) {
        return UCS_OK;
    }

    criteria.title              = "atomic operations on %s memory";
    criteria.local_md_flags     = 0;
    criteria.remote_md_flags    = 0;
    criteria.local_iface_flags  = criteria.remote_iface_flags |
                                  UCT_IFACE_FLAG_PENDING;
    criteria.calc_score         = ucp_wireup_amo_score_func;

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
                                          tl_bitmap, UCP_WIREUP_LANE_USAGE_AMO);
}

static double ucp_wireup_am_score_func(const uct_md_attr_t *md_attr,
                                       const uct_iface_attr_t *iface_attr,
                                       const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* best end-to-end latency */
    return 1e-3 / (iface_attr->latency + iface_attr->overhead + remote_iface_attr->overhead);
}

static double ucp_wireup_rndv_score_func(const uct_md_attr_t *md_attr,
                                         const uct_iface_attr_t *iface_attr,
                                         const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* highest bandwidth with lowest overhead - test a message size of 256KB,
     * a size which is likely to be used with the Rendezvous protocol, for
     * how long it would take to transfer it with a certain transport. */
    return 1 / ((262144 / iface_attr->bandwidth) + iface_attr->overhead);
}

static ucs_status_t ucp_wireup_add_am_lane(ucp_ep_h ep, unsigned address_count,
                                           const ucp_address_entry_t *address_list,
                                           ucp_wireup_lane_desc_t *lane_descs,
                                           ucp_lane_index_t *num_lanes_p)
{
    ucp_wireup_criteria_t criteria;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    ucs_status_t status;
    unsigned addr_index;
    double score;
    int need_am;

    /* Check if we need active messages, for wireup */
    if (!(ucp_ep_get_context_features(ep) & UCP_FEATURE_TAG)) {
        need_am = 0;
        for (lane = 0; lane < *num_lanes_p; ++lane) {
            need_am = need_am || ucp_worker_is_tl_p2p(ep->worker,
                                                      lane_descs[lane].rsc_index);
        }
        if (!need_am) {
            return UCS_OK;
        }
    }

    /* Select one lane for active messages */
    criteria.title              = "active messages";
    criteria.local_md_flags     = 0;
    criteria.remote_md_flags    = 0;
    criteria.remote_iface_flags = UCT_IFACE_FLAG_AM_BCOPY |
                                  UCT_IFACE_FLAG_AM_CB_SYNC;
    criteria.local_iface_flags  = UCT_IFACE_FLAG_AM_BCOPY;
    criteria.calc_score         = ucp_wireup_am_score_func;

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep), UCP_FEATURE_TAG |
                                                            UCP_FEATURE_WAKEUP)) {
        criteria.remote_iface_flags |= UCT_IFACE_FLAG_WAKEUP;
    }

    status = ucp_wireup_select_transport(ep, address_list, address_count, &criteria,
                                         -1, -1, 1, &rsc_index, &addr_index, &score);
    if (status != UCS_OK) {
        return status;
    }

    ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                             address_list[addr_index].md_index, score,
                             UCP_WIREUP_LANE_USAGE_AM);
    return UCS_OK;
}

static ucs_status_t ucp_wireup_add_rndv_lane(ucp_ep_h ep, unsigned address_count,
                                             const ucp_address_entry_t *address_list,
                                             ucp_wireup_lane_desc_t *lane_descs,
                                             ucp_lane_index_t *num_lanes_p)
{
    ucp_wireup_criteria_t criteria;
    ucp_rsc_index_t rsc_index;
    ucs_status_t status;
    unsigned addr_index;
    double score;

    if (!(ucp_ep_get_context_features(ep) & UCP_FEATURE_TAG)) {
        return UCS_OK;
    }

    /* Select one lane for the Rendezvous protocol (for the actual data. not for rts/rtr) */
    criteria.title              = "rendezvous";
    criteria.local_md_flags     = UCT_MD_FLAG_REG;
    criteria.remote_md_flags    = UCT_MD_FLAG_REG;  /* TODO not all ucts need reg on remote side */
    criteria.remote_iface_flags = UCT_IFACE_FLAG_GET_ZCOPY;
    criteria.local_iface_flags  = UCT_IFACE_FLAG_GET_ZCOPY |
                                  UCT_IFACE_FLAG_PENDING;
    criteria.calc_score         = ucp_wireup_rndv_score_func;

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep), UCP_FEATURE_WAKEUP)) {
        criteria.remote_iface_flags |= UCT_IFACE_FLAG_WAKEUP;
    }

    status = ucp_wireup_select_transport(ep, address_list, address_count, &criteria,
                                         -1, -1, 0, &rsc_index, &addr_index, &score);
    if ((status == UCS_OK) &&
        /* a temporary workaround to prevent the ugni uct from using rndv */
        (strstr(ep->worker->context->tl_rscs[rsc_index].tl_rsc.tl_name, "ugni") == NULL)) {
         ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                                 address_list[addr_index].md_index, score,
                                 UCP_WIREUP_LANE_USAGE_RNDV);
    }

    return UCS_OK;
}

static ucp_lane_index_t
ucp_wireup_select_wireup_msg_lane(ucp_worker_h worker,
                                  const ucp_address_entry_t *address_list,
                                  const ucp_wireup_lane_desc_t *lane_descs,
                                  ucp_lane_index_t num_lanes)
{
    ucp_context_h context     = worker->context;
    ucp_lane_index_t p2p_lane = UCP_NULL_RESOURCE;
    uct_tl_resource_desc_t *resource;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;
    unsigned addr_index;

    for (lane = 0; lane < num_lanes; ++lane) {
        rsc_index  = lane_descs[lane].rsc_index;
        addr_index = lane_descs[lane].addr_index;
        resource   = &context->tl_rscs[rsc_index].tl_rsc;

        /* if the current lane satisfies the wireup criteria, choose it for wireup.
         * if it doesn't take a lane with a p2p transport */
        if (ucp_wireup_check_flags(resource,
                                   worker->iface_attrs[rsc_index].cap.flags,
                                   ucp_wireup_aux_criteria.local_iface_flags,
                                   ucp_wireup_aux_criteria.title,
                                   ucp_wireup_iface_flags, NULL, 0) &&
            ucp_wireup_check_flags(resource,
                                   address_list[addr_index].iface_attr.cap_flags,
                                   ucp_wireup_aux_criteria.remote_iface_flags,
                                   ucp_wireup_aux_criteria.title,
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

ucs_status_t ucp_wireup_select_lanes(ucp_ep_h ep, unsigned address_count,
                                     const ucp_address_entry_t *address_list,
                                     uint8_t *addr_indices,
                                     ucp_ep_config_key_t *key)
{
    ucp_worker_h worker            = ep->worker;
    ucp_lane_index_t num_amo_lanes = 0;
    ucp_wireup_lane_desc_t lane_descs[UCP_MAX_LANES];
    ucp_rsc_index_t rsc_index, dst_md_index;
    ucp_lane_index_t lane;
    ucs_status_t status;

    memset(lane_descs, 0, sizeof(lane_descs));
    memset(key, 0, sizeof(*key));

    status = ucp_wireup_add_rma_lanes(ep, address_count, address_list,
                                      lane_descs, &key->num_lanes);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_wireup_add_amo_lanes(ep, address_count, address_list,
                                      lane_descs, &key->num_lanes);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_wireup_add_am_lane(ep, address_count, address_list,
                                    lane_descs, &key->num_lanes);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_wireup_add_rndv_lane(ep, address_count, address_list,
                                      lane_descs, &key->num_lanes);
    if (status != UCS_OK) {
        return status;
    }

    /* User should not create endpoints unless requested communication features */
    if (key->num_lanes == 0) {
        ucs_error("No transports selected to %s (features: 0x%lx)",
                  ucp_ep_peer_name(ep), ucp_ep_get_context_features(ep));
        return UCS_ERR_UNREACHABLE;
    }

    /* Sort lanes according to RMA score */
    qsort(lane_descs, key->num_lanes, sizeof(*lane_descs),
          ucp_wireup_compare_lane_rma_score);

    /* Construct the endpoint configuration key:
     * - arrange lane description in the EP configuration
     * - create remote MD bitmap
     * - create bitmap of lanes used for RMA and AMO
     * - if AM lane exists and fits for wireup messages, select it for this purpose.
     */
    key->am_lane   = UCP_NULL_LANE;
    key->rndv_lane = UCP_NULL_LANE;
    for (lane = 0; lane < key->num_lanes; ++lane) {
        rsc_index          = lane_descs[lane].rsc_index;
        dst_md_index       = lane_descs[lane].dst_md_index;
        key->lanes[lane]   = rsc_index;
        addr_indices[lane] = lane_descs[lane].addr_index;
        ucs_assert(lane_descs[lane].usage != 0);

        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AM) {
            ucs_assert(key->am_lane == UCP_NULL_LANE);
            key->am_lane = lane;
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_RMA) {
            key->rma_lane_map |= UCS_BIT(dst_md_index + lane * UCP_MD_INDEX_BITS);
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AMO) {
            key->amo_lanes[num_amo_lanes] = lane;
            ++num_amo_lanes;
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_RNDV) {
            ucs_assert(key->rndv_lane == UCP_NULL_LANE);
            key->rndv_lane = lane;
        }
    }

    /* Sort and add AMO lanes */
    ucs_qsort_r(key->amo_lanes, num_amo_lanes, sizeof(*key->amo_lanes),
                ucp_wireup_compare_lane_amo_score, lane_descs);
    for (lane = 0; lane < UCP_MAX_LANES; ++lane) {
        if (lane < num_amo_lanes) {
            dst_md_index      = lane_descs[key->amo_lanes[lane]].dst_md_index;
            key->amo_lane_map |= UCS_BIT(dst_md_index + lane * UCP_MD_INDEX_BITS);
        } else {
            key->amo_lanes[lane] = UCP_NULL_LANE;
        }
    }

    key->reachable_md_map = ucp_wireup_get_reachable_mds(worker, address_count,
                                                         address_list);
    key->wireup_msg_lane  = ucp_wireup_select_wireup_msg_lane(worker, address_list,
                                                              lane_descs, key->num_lanes);
    return UCS_OK;
}

static double ucp_wireup_aux_score_func(const uct_md_attr_t *md_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_address_iface_attr_t *remote_iface_attr)
{
    /* best end-to-end latency and larger bcopy size */
    return (1e-3 / (iface_attr->latency + iface_attr->overhead + remote_iface_attr->overhead)) +
           (1e3 * ucs_max(iface_attr->cap.am.max_bcopy, iface_attr->cap.am.max_short));
}

ucs_status_t ucp_wireup_select_aux_transport(ucp_ep_h ep,
                                             const ucp_address_entry_t *address_list,
                                             unsigned address_count,
                                             ucp_rsc_index_t *rsc_index_p,
                                             unsigned *addr_index_p)
{
    double score;
    return ucp_wireup_select_transport(ep, address_list, address_count,
                                       &ucp_wireup_aux_criteria, -1, -1, 1,
                                       rsc_index_p, addr_index_p, &score);
}
