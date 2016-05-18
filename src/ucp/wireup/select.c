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
    uint16_t tl_name_csum;
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
        resource     = &context->tl_rscs[rsc_index].tl_rsc;
        tl_name_csum = context->tl_rscs[rsc_index].tl_name_csum;
        iface        = worker->ifaces[rsc_index];

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
            reachable = (tl_name_csum == ae->tl_name_csum) &&
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

ucs_status_t ucp_wireup_select_transports(ucp_ep_h ep, unsigned address_count,
                                          const ucp_address_entry_t *address_list,
                                          unsigned *addr_indices)
{
    ucp_worker_h worker   = ep->worker;
    ucp_context_h context = worker->context;
    ucp_wireup_lane_desc_t lane_descs[UCP_MAX_LANES];
    ucp_lane_index_t lane, num_lanes;
    ucp_rsc_index_t rsc_index, dst_pd_index;
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

    /* Sort lanes according to RMA score
     * TODO do it for AMOs as well
     */
    for (lane = 0; lane < num_lanes; ++lane) {
        rsc_index = lane_descs[lane].rsc_index;
        lane_descs[lane].score =
                        ucp_wireup_rma_score_func(worker,
                                                  &worker->iface_attrs[rsc_index],
                                                  NULL, 0);
    }
    qsort(lane_descs, num_lanes, sizeof(*lane_descs),
          ucp_wireup_compare_lane_desc_score);

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
            key.rma_lane_map  |= UCS_BIT(dst_pd_index + lane * UCP_PD_INDEX_BITS);
        }
        if (lane_descs[lane].usage & UCP_WIREUP_LANE_USAGE_AMO) {
            /* TODO different priority map for atomics */
            key.amo_lane_map  |= UCS_BIT(dst_pd_index + lane * UCP_PD_INDEX_BITS);
        }
    }

    /* Add all reachable remote pd's */
    key.reachable_pd_map = 0;
    for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
        for (addr_index = 0; addr_index < address_count; ++addr_index) {
            if ((address_list[addr_index].tl_name_csum == context->tl_rscs[rsc_index].tl_name_csum) &&
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
                  (key.am_lane        == lane          )        ? " [active message]"       : "",
                  ucp_lane_map_get_lane(key.rma_lane_map, lane) ? " [remote memory access]" : "",
                  ucp_lane_map_get_lane(key.amo_lane_map, lane) ? " [atomic operations]"    : "",
                  (key.wireup_msg_lane == lane         )        ? " [wireup messages]"      : "");
    }
    ucs_debug("ep %p: rma_lane_map 0x%"PRIx64" amo_lane_map 0x%"PRIx64" reachable_pds 0x%x",
              ep, key.rma_lane_map, key.amo_lane_map, key.reachable_pd_map);

    /* Allocate/reuse configuration index */
    ep->cfg_index = ucp_worker_get_ep_config(worker, &key);

    /* Cache AM lane index on the endpoint */
    ep->am_lane   = key.am_lane;

    return UCS_OK;
}

