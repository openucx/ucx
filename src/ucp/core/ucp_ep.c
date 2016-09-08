/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_ep.h"
#include "ucp_worker.h"
#include "ucp_ep.inl"
#include "ucp_request.inl"

#include <ucp/wireup/stub_ep.h>
#include <ucp/wireup/wireup.h>
#include <ucp/tag/eager.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <string.h>


ucs_status_t ucp_ep_new(ucp_worker_h worker, uint64_t dest_uuid,
                        const char *peer_name, const char *message,
                        ucp_ep_h *ep_p)
{
    ucs_status_t status;
    ucp_ep_config_key_t key;
    ucp_ep_h ep;
    khiter_t hash_it;
    int hash_extra_status = 0;

    ep = ucs_calloc(1, sizeof(*ep), "ucp ep");
    if (ep == NULL) {
        ucs_error("Failed to allocate ep");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* EP configuration without any lanes */
    memset(&key, 0, sizeof(key));
    key.rma_lane_map     = 0;
    key.amo_lane_map     = 0;
    key.reachable_md_map = 0;
    key.am_lane          = UCP_NULL_RESOURCE;
    key.rndv_lane        = UCP_NULL_RESOURCE;
    key.wireup_msg_lane  = UCP_NULL_LANE;
    key.num_lanes        = 0;
    memset(key.amo_lanes, UCP_NULL_LANE, sizeof(key.amo_lanes));

    ep->worker           = worker;
    ep->dest_uuid        = dest_uuid;
    ep->cfg_index        = ucp_worker_get_ep_config(worker, &key);
    ep->am_lane          = UCP_NULL_LANE;
    ep->flags            = 0;
#if ENABLE_DEBUG_DATA
    ucs_snprintf_zero(ep->peer_name, UCP_WORKER_NAME_MAX, "%s", peer_name);
#endif

    hash_it = kh_put(ucp_worker_ep_hash, &worker->ep_hash, dest_uuid,
                     &hash_extra_status);
    if (ucs_unlikely(hash_it == kh_end(&worker->ep_hash))) {
        ucs_error("Hash failed with ep %p to %s 0x%"PRIx64"->0x%"PRIx64" %s "
                  "with status %d", ep, peer_name, worker->uuid, ep->dest_uuid,
                  message, hash_extra_status);
        status = UCS_ERR_NO_RESOURCE;
        goto err_free_ep;
    }
    kh_value(&worker->ep_hash, hash_it) = ep;

    *ep_p = ep;
    ucs_debug("created ep %p to %s 0x%"PRIx64"->0x%"PRIx64" %s", ep, peer_name,
              worker->uuid, ep->dest_uuid, message);
    return UCS_OK;

err_free_ep:
    ucs_free(ep);
err:
    return status;
}

static void ucp_ep_delete_from_hash(ucp_ep_h ep)
{
    khiter_t hash_it;

    hash_it = kh_get(ucp_worker_ep_hash, &ep->worker->ep_hash, ep->dest_uuid);
    if (hash_it != kh_end(&ep->worker->ep_hash)) {
        kh_del(ucp_worker_ep_hash, &ep->worker->ep_hash, hash_it);
    }
}

static void ucp_ep_delete(ucp_ep_h ep)
{
    ucp_ep_delete_from_hash(ep);
    ucs_free(ep);
}

ucs_status_t ucp_ep_create_stub(ucp_worker_h worker, uint64_t dest_uuid,
                                const char *message, ucp_ep_h *ep_p)
{
    ucs_status_t status;
    ucp_ep_config_key_t key;
    ucp_ep_h ep = NULL;

    status = ucp_ep_new(worker, dest_uuid, "??", message, &ep);
    if (status != UCS_OK) {
        goto err;
    }

    /* all operations will use the first lane, which is a stub endpoint */
    memset(&key, 0, sizeof(key));
    key.rma_lane_map     = 1;
    key.amo_lane_map     = 1;
    key.reachable_md_map = 0; /* TODO */
    key.am_lane          = 0;
    key.rndv_lane        = 0;
    key.wireup_msg_lane  = 0;
    key.lanes[0]         = UCP_NULL_RESOURCE;
    key.num_lanes        = 1;
    memset(key.amo_lanes, UCP_NULL_LANE, sizeof(key.amo_lanes));

    ep->cfg_index        = ucp_worker_get_ep_config(worker, &key);
    ep->am_lane          = 0;

    status = ucp_stub_ep_create(ep, &ep->uct_eps[0]);
    if (status != UCS_OK) {
        goto err_destroy_uct_eps;
    }

    *ep_p = ep;
    return UCS_OK;

err_destroy_uct_eps:
    uct_ep_destroy(ep->uct_eps[0]);
    ucp_ep_delete(ep);
err:
    return status;
}

int ucp_ep_is_stub(ucp_ep_h ep)
{
    return ucp_ep_get_rsc_index(ep, 0) == UCP_NULL_RESOURCE;
}

ucs_status_t ucp_ep_create(ucp_worker_h worker, const ucp_address_t *address,
                           ucp_ep_h *ep_p)
{
    char peer_name[UCP_WORKER_NAME_MAX];
    uint8_t addr_indices[UCP_MAX_LANES];
    ucp_address_entry_t *address_list;
    unsigned address_count;
    ucs_status_t status;
    uint64_t dest_uuid;
    ucp_ep_h ep;

    UCS_ASYNC_BLOCK(&worker->async);

    status = ucp_address_unpack(address, &dest_uuid, peer_name, sizeof(peer_name),
                                &address_count, &address_list);
    if (status != UCS_OK) {
        ucs_error("failed to unpack remote address: %s", ucs_status_string(status));
        goto out;
    }

    ep = ucp_worker_ep_find(worker, dest_uuid);
    if (ep != NULL) {
        /* TODO handle a case where the existing endpoint is incomplete */
        *ep_p = ep;
        status = UCS_OK;
        goto out_free_address;
    }

    /* allocate endpoint */
    status = ucp_ep_new(worker, dest_uuid, peer_name, "from api call", &ep);
    if (status != UCS_OK) {
        goto out_free_address;
    }

    /* initialize transport endpoints */
    status = ucp_wireup_init_lanes(ep, address_count, address_list, addr_indices);
    if (status != UCS_OK) {
        goto err_destroy_ep;
    }

    /* send initial wireup message */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        status = ucp_wireup_send_request(ep);
        if (status != UCS_OK) {
            goto err_destroy_ep;
        }
    }

    *ep_p = ep;
    goto out_free_address;

err_destroy_ep:
    ucp_ep_destroy(ep);
out_free_address:
    ucs_free(address_list);
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

static void ucp_ep_destory_uct_eps(ucp_ep_h ep)
{
    ucp_lane_index_t lane;
    uct_ep_h uct_ep;

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        uct_ep = ep->uct_eps[lane];
        if (uct_ep == NULL) {
            continue;
        }
        uct_ep_pending_purge(uct_ep, ucp_request_release_pending_send, NULL);
        ucs_debug("destroy ep %p lane %d uct_ep %p", ep, lane, uct_ep);
        uct_ep_destroy(uct_ep);
    }
}

ucs_status_ptr_t ucp_disconnect_nb(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;

    ucs_debug("disconnect ep %p", ep);

    UCS_ASYNC_BLOCK(&worker->async);
    ucp_ep_delete_from_hash(ep);
    ucp_ep_destory_uct_eps(ep);
    UCS_ASYNC_UNBLOCK(&worker->async);

    ucs_free(ep);

    return NULL; /* TODO implement non-blocking flow */
}

void ucp_ep_destroy(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;
    ucs_status_ptr_t *ureq;
    ucp_request_t *req;

    ureq = ucp_disconnect_nb(ep);
    if (ureq == NULL) {
        return;
    } else if (UCS_PTR_IS_ERR(ureq)) {
        ucs_warn("disconnect failed: %s", ucs_status_string(UCS_PTR_STATUS(ureq)));
        return;
    } else {
        req = (ucp_request_t*)ureq - 1;
        while (!(req->flags & UCP_REQUEST_FLAG_COMPLETED)) {
            ucp_worker_progress(worker);
        }
    }
}

int ucp_ep_config_is_equal(const ucp_ep_config_key_t *key1,
                           const ucp_ep_config_key_t *key2)
{
    ucp_lane_index_t lane;


    if ((key1->num_lanes        != key2->num_lanes) ||
        (key1->rma_lane_map     != key2->rma_lane_map) ||
        (key1->amo_lane_map     != key2->amo_lane_map) ||
        memcmp(key1->amo_lanes, key2->amo_lanes, sizeof(key1->amo_lanes)) ||
        (key1->reachable_md_map != key2->reachable_md_map) ||
        (key1->am_lane          != key2->am_lane) ||
        (key1->rndv_lane        != key2->rndv_lane) ||
        (key1->wireup_msg_lane  != key2->wireup_msg_lane))
    {
        return 0;
    }

    for (lane = 0; lane < key1->num_lanes; ++lane) {
        if (key1->lanes[lane] != key2->lanes[lane]) {
            return 0;
        }
    }

    return 1;
}

void ucp_ep_config_init(ucp_worker_h worker, ucp_ep_config_t *config)
{
    ucp_context_h context = worker->context;
    ucp_ep_rma_config_t *rma_config;
    uct_iface_attr_t *iface_attr;
    ucp_rsc_index_t rsc_index;
    uct_md_attr_t *md_attr;
    ucp_lane_index_t lane;
    double zcopy_thresh, rndv_thresh, numerator, denumerator;

    /* Default thresholds */
    config->zcopy_thresh      = SIZE_MAX;
    config->sync_zcopy_thresh = -1;
    config->bcopy_thresh      = context->config.ext.bcopy_thresh;
    config->rndv_thresh       = SIZE_MAX;
    config->sync_rndv_thresh  = SIZE_MAX;
    config->max_rndv_get_zcopy = SIZE_MAX;

    /* Configuration for active messages */
    if (config->key.am_lane != UCP_NULL_LANE) {
        lane        = config->key.am_lane;
        rsc_index   = config->key.lanes[lane];
        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr  = &worker->iface_attrs[rsc_index];
            md_attr     = &context->md_attrs[context->tl_rscs[rsc_index].md_index];

            if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT) {
                config->max_eager_short  = iface_attr->cap.am.max_short -
                                           sizeof(ucp_eager_hdr_t);
                config->max_am_short     = iface_attr->cap.am.max_short -
                                           sizeof(uint64_t);
            }

            if (iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) {
                config->max_am_bcopy     = iface_attr->cap.am.max_bcopy;
            }

            if ((iface_attr->cap.flags & UCT_IFACE_FLAG_AM_ZCOPY) &&
                (md_attr->cap.flags & UCT_MD_FLAG_REG))
            {
                config->max_am_zcopy  = iface_attr->cap.am.max_zcopy;

                if (context->config.ext.zcopy_thresh == UCS_CONFIG_MEMUNITS_AUTO) {
                    /* auto */
                    zcopy_thresh = md_attr->reg_cost.overhead / (
                                            (1.0 / context->config.ext.bcopy_bw) -
                                            (1.0 / iface_attr->bandwidth) -
                                            md_attr->reg_cost.growth);
                    if (zcopy_thresh < 0) {
                        config->zcopy_thresh      = SIZE_MAX;
                        config->sync_zcopy_thresh = -1;
                    } else {
                        config->zcopy_thresh      = zcopy_thresh;
                        config->sync_zcopy_thresh = zcopy_thresh;
                    }
                } else {
                    config->zcopy_thresh      = context->config.ext.zcopy_thresh;
                    config->sync_zcopy_thresh = context->config.ext.zcopy_thresh;
                }
            }
        } else {
            config->max_am_bcopy = UCP_MIN_BCOPY; /* Stub endpoint */
        }
    }

    /* Configuration for remote memory access */
    for (lane = 0; lane < config->key.num_lanes; ++lane) {
        if (ucp_lane_map_get_lane(config->key.rma_lane_map, lane) == 0) {
            continue;
        }

        rma_config = &config->rma[lane];
        rsc_index  = config->key.lanes[lane];
        iface_attr = &worker->iface_attrs[rsc_index];
        if (rsc_index != UCP_NULL_RESOURCE) {
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_SHORT) {
                rma_config->max_put_short = iface_attr->cap.put.max_short;
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_BCOPY) {
                rma_config->max_put_bcopy = iface_attr->cap.put.max_bcopy;
            }
            if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_BCOPY) {
                rma_config->max_get_bcopy = iface_attr->cap.get.max_bcopy;
            }
        } else {
            rma_config->max_put_bcopy = UCP_MIN_BCOPY; /* Stub endpoint */
        }
    }

    /* Configuration for Rendezvous data */
    if (config->key.rndv_lane != UCP_NULL_LANE) {
        lane        = config->key.rndv_lane;
        rsc_index   = config->key.lanes[lane];
        if (rsc_index != UCP_NULL_RESOURCE) {
            iface_attr = &worker->iface_attrs[rsc_index];
            md_attr    = &context->md_attrs[context->tl_rscs[rsc_index].md_index];

            if (iface_attr->cap.flags & UCT_IFACE_FLAG_GET_ZCOPY) {

                if (context->config.ext.rndv_thresh == UCS_CONFIG_MEMUNITS_AUTO) {
                    /* auto */
                    /* Make UCX calculate the rndv threshold on its own.
                     * We do it by finding the message size at which rndv and eager_zcopy get
                     * the same latency. Starting this message size (rndv_thresh), rndv's latency
                     * would be lower and the reached bandwidth would be higher.
                     * The latency function for eager_zcopy is:
                     * [ reg_cost.overhead + size * md_attr->reg_cost.growth +
                     * max(size/bw , size/bcopy_bw) + overhead ]
                     * The latency function for Rendezvous is:
                     * [ reg_cost.overhead + size * md_attr->reg_cost.growth + latency + overhead +
                     *   reg_cost.overhead + size * md_attr->reg_cost.growth + overhead + latency +
                     *   size/bw + latency + overhead + latency ]
                     * Isolating the 'size' yields the rndv_thresh.
                     * The used latency functions for eager_zcopy and rndv are also specified in
                     * the UCX wiki */
                    numerator = ((2 * iface_attr->overhead) + (4 * iface_attr->latency) +
                                 md_attr->reg_cost.overhead);
                    denumerator = (ucs_max((1.0 / iface_attr->bandwidth),(1.0 / context->config.ext.bcopy_bw)) -
                                   md_attr->reg_cost.growth - (1.0 / iface_attr->bandwidth));

                    if (denumerator > 0) {
                        rndv_thresh = numerator / denumerator;
                    } else {
                        rndv_thresh = context->config.ext.rndv_thresh_fallback;
                        ucs_debug("The denumerator is %f. Setting the rndv threshold to %f",
                                  denumerator, rndv_thresh);
                    }

                    /* for the 'auto' mode in the rndv_threshold, we enforce the usage of rndv
                     * to a value that can be set by the user.
                     * to disable rndv, need to set a high value for the rndv_threshold
                     * (without the 'auto' mode)*/
                    config->rndv_thresh        = rndv_thresh;
                    config->sync_rndv_thresh   = rndv_thresh;
                    config->max_rndv_get_zcopy = iface_attr->cap.get.max_zcopy;

                } else {
                    config->rndv_thresh        = context->config.ext.rndv_thresh;
                    config->sync_rndv_thresh   = context->config.ext.rndv_thresh;
                    config->max_rndv_get_zcopy = iface_attr->cap.get.max_zcopy;
                }
                ucs_debug("the rendezvous threshold is: %zu", config->rndv_thresh);
            }
        } else {
            ucs_debug("There was no transport selected for rndv");
        }
    } else {
        ucs_debug("There was no lane for rndv");
    }
}
