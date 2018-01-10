/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WIREUP_EP_H_
#define UCP_WIREUP_EP_H_

#include "address.h"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_proxy_ep.h>
#include <ucs/datastruct/queue_types.h>


/**
 * Stub endpoint flags
 */
enum {
    UCP_WIREUP_EP_FLAG_READY           = UCS_BIT(0), /**< next_ep is fully connected */
    UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED = UCS_BIT(1), /**< Debug: next_ep connected to remote */
};


/**
 * Wireup proxy endpoint, to hold off send requests until wireup process completes.
 * It is placed instead UCT endpoint before it's fully connected, and for AM
 * endpoint it also contains an auxiliary endpoint which can send wireup messages.
 */
struct ucp_wireup_ep {
    ucp_proxy_ep_t            super;         /**< Derive from ucp_proxy_ep_t */
    ucs_queue_head_t          pending_q;     /**< Queue of pending operations */
    uct_ep_h                  aux_ep;        /**< Used to wireup the "real" endpoint */
    uct_ep_h                  sockaddr_ep;   /**< Used for client-server wireup */
    ucp_rsc_index_t           aux_rsc_index; /**< Index of auxiliary transport */
    volatile uint32_t         pending_count; /**< Number of pending wireup operations */
    volatile uint32_t         flags;         /**< Connection state flags */
    uct_worker_cb_id_t        progress_id;   /**< ID of progress function */
};

typedef struct ucp_wireup_client_data {
    ucp_err_handling_mode_t err_mode;
    uint64_t                ep_uuid;
    /* packed worker address follows */
} UCS_S_PACKED ucp_wireup_sockaddr_priv_t;

/**
 * Create a proxy endpoint for wireup.
 */
ucs_status_t ucp_wireup_ep_create(ucp_ep_h ep, uct_ep_h *ep_p);


/**
 * @return Auxiliary resource index used by the wireup endpoint.
 *   If the endpoint is not a wireup endpoint, return UCP_NULL_RESOURCE.
 */
ucp_rsc_index_t ucp_wireup_ep_get_aux_rsc_index(uct_ep_h uct_ep);


/**
 * Create endpoint for the real transport, which we would eventually connect.
 * After this function is called, it would be possible to send wireup messages
 * on this endpoint, if connect_aux is 1.
 *
 * @param [in]  uct_ep       Stub endpoint to connect.
 * @param [in]  rsc_index    Resource of the real transport.
 * @param [in]  connect_aux  Whether to connect the auxiliary transport, for
 *                          sending
 */
ucs_status_t ucp_wireup_ep_connect(uct_ep_h uct_ep, const ucp_ep_params_t *params,
                                   ucp_rsc_index_t rsc_index, int connect_aux,
                                   unsigned address_count,
                                   const ucp_address_entry_t *address_list);

ucs_status_t ucp_wireup_ep_connect_to_sockaddr(uct_ep_h uct_ep,
                                               const ucp_ep_params_t *params);

void ucp_wireup_ep_set_next_ep(uct_ep_h uct_ep, uct_ep_h next_ep);

uct_ep_h ucp_wireup_ep_extract_next_ep(uct_ep_h uct_ep);

void ucp_wireup_ep_remote_connected(uct_ep_h uct_ep);

int ucp_wireup_ep_test(uct_ep_h uct_ep);

int ucp_wireup_ep_is_owner(uct_ep_h uct_ep, uct_ep_h owned_ep);

void ucp_wireup_ep_disown(uct_ep_h uct_ep, uct_ep_h owned_ep);

#endif
