/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "wireup.h"
#include "address.h"

#include <ucp/core/ucp_ep.inl>
#include <string.h>
#include <inttypes.h>


enum {
    UCP_WIREUP_LANE_USAGE_AM  = UCS_BIT(0),
    UCP_WIREUP_LANE_USAGE_RMA = UCS_BIT(1),
    UCP_WIREUP_LANE_USAGE_AMO = UCS_BIT(2)
};

typedef struct {
    ucp_rsc_index_t   rsc_index;
    unsigned          addr_index;
    ucp_rsc_index_t   dst_pd_index;
    double            score;
    uint32_t          usage;
} ucp_wireup_lane_desc_t;


static const char *ucp_wireup_pd_flags[] = {
    [ucs_ilog2(UCT_PD_FLAG_ALLOC)]               = "memory allocation",
    [ucs_ilog2(UCT_PD_FLAG_REG)]                 = "memory registration",
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

static double ucp_wireup_aux_score_func(const uct_pd_attr_t *pd_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_wireup_iface_attr_t *remote_iface_attr);

static ucp_wireup_criteria_t ucp_wireup_aux_criteria = {
    .title              = "auxiliary",
    .local_pd_flags     = 0,
    .remote_pd_flags    = 0,
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
           uct_iface_is_reachable(worker->ifaces[rsc_index], ae->dev_addr);
}

/**
 * Select a local and remote transport
 */
static UCS_F_NOINLINE ucs_status_t
ucp_wireup_select_transport(ucp_ep_h ep, const ucp_address_entry_t *address_list,
                            unsigned address_count, const ucp_wireup_criteria_t *criteria,
                            uint64_t remote_pd_map, int show_error,
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
    uct_iface_h iface;
    uct_iface_attr_t *iface_attr;
    uct_pd_attr_t *pd_attr;
    uint64_t addr_index_map;
    unsigned addr_index;
    int reachable;
    int found;

    found       = 0;
    best_score  = 0.0;
    p           = tls_info;
    endp        = tls_info + sizeof(tls_info) - 1;
    tls_info[0] = '\0';

    /* Check which remote addresses satisfy the criteria */
    addr_index_map = 0;
    for (ae = address_list; ae < address_list + address_count; ++ae) {
        addr_index = ae - address_list;
        if (!(remote_pd_map & UCS_BIT(ae->pd_index))) {
            ucs_trace("addr[%d]: not in use, because on pd[%d]", addr_index,
                      ae->pd_index);
            continue;
        }
        if (!ucs_test_all_flags(ae->pd_flags, criteria->remote_pd_flags)) {
            ucs_trace("addr[%d]: no %s", addr_index,
                      ucp_wireup_get_missing_flag_desc(ae->pd_flags,
                                                       criteria->remote_pd_flags,
                                                       ucp_wireup_pd_flags));
            continue;
        }
        if (!ucs_test_all_flags(ae->iface_attr.cap_flags, criteria->remote_iface_flags)) {
            ucs_trace("addr[%d]: no %s", addr_index,
                      ucp_wireup_get_missing_flag_desc(ae->iface_attr.cap_flags,
                                                       criteria->remote_iface_flags,
                                                       ucp_wireup_iface_flags));
            continue;
        }

        addr_index_map |= UCS_BIT(addr_index);
    }

    /* For each local resource try to find the best remote address to connect to */
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        resource     = &context->tl_rscs[rsc_index].tl_rsc;
        iface        = worker->ifaces[rsc_index];
        iface_attr   = &worker->iface_attrs[rsc_index];
        pd_attr      = &context->pd_attrs[context->tl_rscs[rsc_index].pd_index];

        /* Check that local pd and interface satisfy the criteria */
        if (!ucp_wireup_check_flags(resource, pd_attr->cap.flags,
                                    criteria->local_pd_flags, criteria->title,
                                    ucp_wireup_pd_flags, p, endp - p) ||
            !ucp_wireup_check_flags(resource, iface_attr->cap.flags,
                                    criteria->local_iface_flags, criteria->title,
                                    ucp_wireup_iface_flags, p, endp - p))
        {
            p += strlen(p);
            snprintf(p, endp - p, ", ");
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

            score = criteria->calc_score(pd_attr, iface_attr, &ae->iface_attr);
            ucs_assert(score >= 0.0);

            ucs_trace(UCT_TL_RESOURCE_DESC_FMT "->addr[%zd] : %s score %.2f",
                      UCT_TL_RESOURCE_DESC_ARG(resource), ae - address_list,
                      criteria->title, score);
            if (!found || (score > best_score)) {
                *rsc_index_p      = rsc_index;
                *dst_addr_index_p = ae - address_list;
                *score_p          = score;
                best_score        = score;
                found             = 1;
            }
        }

        /* If a local resource cannot reach any of the remote addresses, generate
         * debug message.
         */
        if (!reachable) {
            snprintf(p, endp - p, UCT_TL_RESOURCE_DESC_FMT" - cannot reach remote worker, ",
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
              " -> '%s' address[%d],pd[%d] score %.2f", ep, criteria->title,
              UCT_TL_RESOURCE_DESC_ARG(&context->tl_rscs[*rsc_index_p].tl_rsc),
              ucp_ep_peer_name(ep), *dst_addr_index_p,
              address_list[*dst_addr_index_p].pd_index, best_score);
    return UCS_OK;
}

static UCS_F_NOINLINE void
ucp_wireup_add_lane_desc(ucp_wireup_lane_desc_t *lane_descs,
                         ucp_lane_index_t *num_lanes_p, ucp_rsc_index_t rsc_index,
                         unsigned addr_index, ucp_rsc_index_t dst_pd_index,
                         double score, uint32_t usage)
{
    ucp_wireup_lane_desc_t *lane_desc;

    for (lane_desc = lane_descs; lane_desc < lane_descs + (*num_lanes_p); ++lane_desc) {
        if ((lane_desc->rsc_index == rsc_index) &&
            (lane_desc->addr_index == addr_index))
        {
            ucs_assertv_always(dst_pd_index == lane_desc->dst_pd_index,
                               "lane[%d].dst_pd_index=%d, dst_pd_index=%d",
                               (int)(lane_desc - lane_descs), lane_desc->dst_pd_index,
                               dst_pd_index);
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
    lane_desc->dst_pd_index = dst_pd_index;
    lane_desc->usage        = usage;
    lane_desc->score        = 0.0;

out_update_score:
    if (usage & UCP_WIREUP_LANE_USAGE_RMA) {
        lane_desc->score = score;
    }
}

static int ucp_wireup_compare_lane_desc_score(const void *elem1, const void *elem2)
{
    const ucp_wireup_lane_desc_t *desc1 = elem1;
    const ucp_wireup_lane_desc_t *desc2 = elem2;

    /* sort from highest score to lowest */
    return (desc1->score < desc2->score) ? 1 :
                    ((desc1->score > desc2->score) ? -1 : 0);
}

static UCS_F_NOINLINE ucs_status_t
ucp_wireup_add_memaccess_lanes(ucp_ep_h ep, unsigned address_count,
                               const ucp_address_entry_t *address_list,
                               ucp_wireup_lane_desc_t *lane_descs,
                               ucp_lane_index_t *num_lanes_p,
                               const ucp_wireup_criteria_t *criteria,
                               uint32_t usage)
{
    ucp_wireup_criteria_t mem_criteria = *criteria;
    ucp_address_entry_t *address_list_copy;
    ucp_rsc_index_t rsc_index, dst_pd_index;
    size_t address_list_size;
    double score, reg_score;
    uint64_t remote_pd_map;
    unsigned addr_index;
    ucs_status_t status;
    char title[64];

    remote_pd_map = -1;

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
    mem_criteria.remote_pd_flags = UCT_PD_FLAG_REG;
    status = ucp_wireup_select_transport(ep, address_list_copy, address_count,
                                         &mem_criteria, remote_pd_map, 1,
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
    remote_pd_map &= ~UCS_BIT(dst_pd_index);

    /* Select additional transports which can access allocated memory, but only
     * if their scores are better. We need this because a remote memory block can
     * be potentially allocated using one of them, and we might get better performance
     * than the transports which support only registered remote memory.
     */
    snprintf(title, sizeof(title), criteria->title, "allocated");
    mem_criteria.title           = title;
    mem_criteria.remote_pd_flags = UCT_PD_FLAG_ALLOC;

    while (address_count > 0) {
        status = ucp_wireup_select_transport(ep, address_list_copy, address_count,
                                             &mem_criteria, remote_pd_map, 0,
                                             &rsc_index, &addr_index, &score);
        if ((status != UCS_OK) || (score <= reg_score)) {
            break;
        }

        /* Add lane description and remove all occurrences of the remote pd */
        dst_pd_index = address_list_copy[addr_index].pd_index;
        ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                                 dst_pd_index, score, usage);
        remote_pd_map &= ~UCS_BIT(dst_pd_index);
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

static double ucp_wireup_rma_score_func(const uct_pd_attr_t *pd_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_wireup_iface_attr_t *remote_iface_attr)
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
    criteria.local_pd_flags     = 0;
    criteria.remote_pd_flags    = 0;
    criteria.remote_iface_flags = UCT_IFACE_FLAG_PUT_SHORT |
                                  UCT_IFACE_FLAG_PUT_BCOPY |
                                  UCT_IFACE_FLAG_GET_BCOPY;
    criteria.local_iface_flags  = criteria.remote_iface_flags |
                                  UCT_IFACE_FLAG_PENDING;
    criteria.calc_score         = ucp_wireup_rma_score_func;

    return ucp_wireup_add_memaccess_lanes(ep, address_count, address_list,
                                          lane_descs, num_lanes_p, &criteria,
                                          UCP_WIREUP_LANE_USAGE_RMA);
}

static double ucp_wireup_amo_score_func(const uct_pd_attr_t *pd_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_wireup_iface_attr_t *remote_iface_attr)
{
    /* best one-sided latency */
    return 1e-3 / (iface_attr->latency + iface_attr->overhead);
}

static ucs_status_t ucp_wireup_add_amo_lanes(ucp_ep_h ep, unsigned address_count,
                                             const ucp_address_entry_t *address_list,
                                             ucp_wireup_lane_desc_t *lane_descs,
                                             ucp_lane_index_t *num_lanes_p)
{
    ucp_wireup_criteria_t criteria;

    criteria.remote_iface_flags = 0;

    if (ucp_ep_get_context_features(ep) & UCP_FEATURE_AMO32) {
        criteria.remote_iface_flags |= UCT_IFACE_FLAG_ATOMIC_ADD32 |
                                       UCT_IFACE_FLAG_ATOMIC_FADD32 |
                                       UCT_IFACE_FLAG_ATOMIC_SWAP32 |
                                       UCT_IFACE_FLAG_ATOMIC_CSWAP32;
    }
    if (ucp_ep_get_context_features(ep) & UCP_FEATURE_AMO64) {
        criteria.remote_iface_flags |= UCT_IFACE_FLAG_ATOMIC_ADD64 |
                                       UCT_IFACE_FLAG_ATOMIC_FADD64 |
                                       UCT_IFACE_FLAG_ATOMIC_SWAP64 |
                                       UCT_IFACE_FLAG_ATOMIC_CSWAP64;
    }
    if (criteria.remote_iface_flags == 0) {
        return UCS_OK;
    }

    criteria.title              = "atomic operations on %s memory";
    criteria.local_pd_flags     = 0;
    criteria.remote_pd_flags    = 0;
    criteria.local_iface_flags  = criteria.remote_iface_flags |
                                  UCT_IFACE_FLAG_PENDING;
    criteria.calc_score         = ucp_wireup_amo_score_func;

    return ucp_wireup_add_memaccess_lanes(ep, address_count, address_list,
                                          lane_descs, num_lanes_p, &criteria,
                                          UCP_WIREUP_LANE_USAGE_AMO);
}

static double ucp_wireup_am_score_func(const uct_pd_attr_t *pd_attr,
                                       const uct_iface_attr_t *iface_attr,
                                       const ucp_wireup_iface_attr_t *remote_iface_attr)
{
    /* best end-to-end latency */
    return 1e-3 / (iface_attr->latency + iface_attr->overhead + remote_iface_attr->overhead);
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
    criteria.local_pd_flags     = 0;
    criteria.remote_pd_flags    = 0;
    criteria.remote_iface_flags = UCT_IFACE_FLAG_AM_BCOPY |
                                  UCT_IFACE_FLAG_AM_CB_SYNC;
    criteria.local_iface_flags  = UCT_IFACE_FLAG_AM_BCOPY;
    criteria.calc_score         = ucp_wireup_am_score_func;

    if (ucs_test_all_flags(ucp_ep_get_context_features(ep), UCP_FEATURE_TAG |
                                                            UCP_FEATURE_WAKEUP)) {
        criteria.remote_iface_flags |= UCT_IFACE_FLAG_WAKEUP;
    }

    status = ucp_wireup_select_transport(ep, address_list, address_count, &criteria,
                                         -1, 1, &rsc_index, &addr_index, &score);
    if (status != UCS_OK) {
        return status;
    }

    ucp_wireup_add_lane_desc(lane_descs, num_lanes_p, rsc_index, addr_index,
                             address_list[addr_index].pd_index, score,
                             UCP_WIREUP_LANE_USAGE_AM);
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
ucp_wireup_get_reachable_pds(ucp_worker_h worker, unsigned address_count,
                             const ucp_address_entry_t *address_list)
{
    ucp_context_h context  = worker->context;
    uint64_t reachable_pds = 0;
    const ucp_address_entry_t *ae;
    ucp_rsc_index_t rsc_index;

    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        for (ae = address_list; ae < address_list + address_count; ++ae) {
            if (ucp_wireup_is_reachable(worker, rsc_index, ae)) {
                reachable_pds |= UCS_BIT(ae->pd_index);
            }
        }
    }

    return reachable_pds;
}

ucs_status_t ucp_wireup_select_lanes(ucp_ep_h ep, unsigned address_count,
                                     const ucp_address_entry_t *address_list,
                                     uint8_t *addr_indices,
                                     ucp_ep_config_key_t *key)
{
    ucp_worker_h worker            = ep->worker;
    ucp_wireup_lane_desc_t lane_descs[UCP_MAX_LANES];
    ucp_rsc_index_t rsc_index, dst_pd_index;
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

    /* User should not create endpoints unless requested communication features */
    if (key->num_lanes == 0) {
        ucs_error("No transports selected to %s", ucp_ep_peer_name(ep));
        return UCS_ERR_UNREACHABLE;
    }

    /* Sort lanes according to RMA score */
    qsort(lane_descs, key->num_lanes, sizeof(*lane_descs),
          ucp_wireup_compare_lane_desc_score);

    /* Construct the endpoint configuration key:
     * - arrange lane description in the EP configuration
     * - create remote PD bitmap
     * - create bitmap of lanes used for RMA and AMO
     * - if AM lane exists and fits for wireup messages, select it fot his purpose.
     */
    key->am_lane   = UCP_NULL_LANE;
    for (lane = 0; lane < key->num_lanes; ++lane) {
        rsc_index          = lane_descs[lane].rsc_index;
        dst_pd_index       = lane_descs[lane].dst_pd_index;
        key->lanes[lane]   = rsc_index;
        addr_indices[lane] = lane_descs[lane].addr_index;
        ucs_assert(lane_descs[lane].usage != 0);

        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AM) {
            ucs_assert(key->am_lane == UCP_NULL_LANE);
            key->am_lane = lane;
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_RMA) {
            key->rma_lane_map |= UCS_BIT(dst_pd_index + lane * UCP_PD_INDEX_BITS);
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AMO) {
            /* TODO different priority map for atomics */
            key->amo_lane_map |= UCS_BIT(dst_pd_index + lane * UCP_PD_INDEX_BITS);
        }
    }

    key->reachable_pd_map = ucp_wireup_get_reachable_pds(worker, address_count,
                                                         address_list);
    key->wireup_msg_lane  = ucp_wireup_select_wireup_msg_lane(worker, address_list,
                                                              lane_descs, key->num_lanes);
    return UCS_OK;
}

static double ucp_wireup_aux_score_func(const uct_pd_attr_t *pd_attr,
                                        const uct_iface_attr_t *iface_attr,
                                        const ucp_wireup_iface_attr_t *remote_iface_attr)
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
                                       &ucp_wireup_aux_criteria, -1, 1,
                                       rsc_index_p, addr_index_p, &score);
}
