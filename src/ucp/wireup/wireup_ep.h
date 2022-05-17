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
    /* next_ep should replace wireup_ep */
    UCP_WIREUP_EP_FLAG_READY            = UCS_BIT(0),

    /* Debug: next_ep connected to remote address */
    UCP_WIREUP_EP_FLAG_LOCAL_CONNECTED  = UCS_BIT(1),

    /* Remote peer has connected to next_ep */
    UCP_WIREUP_EP_FLAG_REMOTE_CONNECTED = UCS_BIT(2),

    /* Send client id */
    UCP_WIREUP_EP_FLAG_SEND_CLIENT_ID   = UCS_BIT(3),

    /* Indicates that aux_ep is CONNECT_TO_EP */
    UCP_WIREUP_EP_FLAG_AUX_P2P          = UCS_BIT(4)
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
    struct sockaddr_storage   cm_remote_sockaddr;  /**< sockaddr of the remote peer -
                                                        used only on the client side
                                                        in a client-server flow */
    struct sockaddr_storage   cm_local_sockaddr;   /**< local sockaddr
                                                        used only on the client side
                                                        in a client-server flow */
    ucp_rsc_index_t           aux_rsc_index; /**< Index of auxiliary transport */
    volatile uint32_t         pending_count; /**< Number of pending wireup operations */
    volatile uint32_t         flags;         /**< Connection state flags */
    uct_worker_cb_id_t        progress_id;   /**< ID of progress function */
    unsigned                  ep_init_flags; /**< UCP wireup EP init flags */
    /**< TLs which are available on client side resolved device */
    ucp_tl_bitmap_t           cm_resolve_tl_bitmap;
    /**< Destination resource indicies used for checking intersection between
         between two configurations in case of CM */
    ucp_rsc_index_t           dst_rsc_indices[UCP_MAX_LANES];
};


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
 * @param [in]  uct_ep            Stub endpoint to connect.
 * @param [in]  ucp_ep_init_flags Initial flags of UCP EP.
 * @param [in]  rsc_index         Resource of the real transport.
 * @param [in]  path_index        Path index the transport endpoint should use.
 * @param [in]  connect_aux       Whether to connect the auxiliary transport,
 *                                for sending.
 * @param [in]  remote_address    Remote address connect to.
 */
ucs_status_t ucp_wireup_ep_connect(uct_ep_h uct_ep, unsigned ucp_ep_init_flags,
                                   ucp_rsc_index_t rsc_index,
                                   unsigned path_index, int connect_aux,
                                   const ucp_unpacked_address_t *remote_address);

void ucp_wireup_ep_pending_queue_purge(uct_ep_h uct_ep,
                                       uct_pending_purge_callback_t cb,
                                       void *arg);

void ucp_wireup_ep_set_aux(ucp_wireup_ep_t *wireup_ep, uct_ep_h uct_ep,
                           ucp_rsc_index_t rsc_index, int is_p2p);

void ucp_wireup_ep_discard_aux_ep(ucp_wireup_ep_t *wireup_ep,
                                  unsigned ep_flush_flags,
                                  uct_pending_purge_callback_t purge_cb,
                                  void *purge_arg);

int ucp_wireup_ep_has_next_ep(ucp_wireup_ep_t *wireup_ep);

void ucp_wireup_ep_set_next_ep(uct_ep_h uct_ep, uct_ep_h next_ep,
                               ucp_rsc_index_t rsc_index);

uct_ep_h ucp_wireup_ep_extract_next_ep(uct_ep_h uct_ep);

void ucp_wireup_ep_destroy_next_ep(ucp_wireup_ep_t *wireup_ep);

void ucp_wireup_ep_remote_connected(uct_ep_h uct_ep, int ready);

int ucp_wireup_ep_test(uct_ep_h uct_ep);

int ucp_wireup_aux_ep_is_owner(ucp_wireup_ep_t *wireup_ep, uct_ep_h owned_ep);

int ucp_wireup_ep_is_owner(uct_ep_h uct_ep, uct_ep_h owned_ep);

void ucp_wireup_ep_disown(uct_ep_h uct_ep, uct_ep_h owned_ep);

uct_ep_h ucp_wireup_ep_get_msg_ep(ucp_wireup_ep_t *wireup_ep);

ucs_status_t ucp_wireup_ep_progress_pending(uct_pending_req_t *self);

ucp_wireup_ep_t *ucp_wireup_ep(uct_ep_h uct_ep);

#endif
