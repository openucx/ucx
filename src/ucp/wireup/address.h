/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_ADDRESS_H_
#define UCP_ADDRESS_H_

#include "wireup.h"

#include <uct/api/uct.h>
#include <ucp/core/ucp_context.h>


/* Which iface flags would be packed in the address */
enum {
    UCP_ADDRESS_IFACE_FLAGS =
         UCT_IFACE_FLAG_CONNECT_TO_IFACE |
         UCT_IFACE_FLAG_CB_SYNC |
         UCT_IFACE_FLAG_CB_ASYNC |
         UCT_IFACE_FLAG_AM_BCOPY |
         UCT_IFACE_FLAG_PUT_SHORT |
         UCT_IFACE_FLAG_PUT_BCOPY |
         UCT_IFACE_FLAG_PUT_ZCOPY |
         UCT_IFACE_FLAG_GET_BCOPY |
         UCT_IFACE_FLAG_GET_ZCOPY |
         UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
         UCT_IFACE_FLAG_TAG_RNDV_ZCOPY  |
         UCP_UCT_IFACE_ATOMIC32_FLAGS |
         UCP_UCT_IFACE_ATOMIC64_FLAGS |
         UCT_IFACE_FLAG_EVENT_RECV |
         UCT_IFACE_FLAG_EVENT_RECV_SIG |
         UCT_IFACE_FLAG_PENDING
};


/**
 * Remote interface attributes.
 */
struct ucp_address_iface_attr {
    uint64_t                   cap_flags;     /* Interface capability flags */
    double                     overhead;      /* Interface performance - overhead */
    double                     bandwidth;     /* Interface performance - bandwidth */
    int                        priority;      /* Priority of device */
    double                     lat_ovh;       /* latency overhead */
};


/**
 * Address entry.
 */
struct ucp_address_entry {
    const uct_device_addr_t    *dev_addr;      /* Points to device address */
    const uct_iface_addr_t     *iface_addr;    /* Interface address, NULL if not available */
    const uct_ep_addr_t        *ep_addr;       /* Endpoint address, NULL if not available */
    ucp_address_iface_attr_t   iface_attr;     /* Interface attributes information */
    uint64_t                   md_flags;       /* MD reg/alloc flags */
    uint16_t                   tl_name_csum;   /* Checksum of transport name */
    ucp_rsc_index_t            md_index;       /* Memory domain index */
    ucp_rsc_index_t            dev_index;      /* Device index */
};


/**
 * Pack multiple addresses into a buffer, of resources specified in rsc_bitmap.
 * For every resource in rcs_bitmap:
 *    - if iface is CONNECT_TO_IFACE, pack interface address
 *    - if iface is CONNECT_TO_EP, and ep != NULL, and it has a uct_ep on this
 *      resource, pack endpoint address.
 *
 * @param [in]  worker      Worker object whose interface addresses to pack.
 * @param [in]  ep          Endpoint object whose uct_ep addresses to pack.
 *                            Can be set to NULL, to take addresses only from worker.
 * @param [in]  tl_bitmap   Specifies the resources whose transport address
 *                           (ep or iface) should be packed.
 * @param [out] order       If != NULL, filled with the order of addresses as they
 *                           were packed. For example: first entry in the array is
 *                           the address index of the first transport specified
 *                           by tl_bitmap. The array should be large enough to
 *                           hold all transports specified by tl_bitmap.
 * @param [out] size_p      Filled with buffer size.
 * @param [out] buffer_p    Filled with pointer to packed buffer. It should be
 *                           released by ucs_free().
 */
ucs_status_t ucp_address_pack(ucp_worker_h worker, ucp_ep_h ep, uint64_t tl_bitmap,
                              unsigned *order, size_t *size_p, void **buffer_p);


/**
 * Unpack a list of addresses.
 *
 * @param [in]  buffer           Buffer with data to unpack.
 * @param [out] remote_uuid_p    Filled with remote worker uuid.
 * @param [out] remote_name      Filled with remote worker name.
 * @param [in]  max              Maximal length on @a remote_name.
 * @param [out] address_count_p  Filled with amount of addresses in the list.
 * @param [out] address_list_p   Filled with pointer to unpacked address list.
 *                                It should be released by ucs_free().
 *
 * @note Entries in the address list could point into the data buffer, so it
 *       should not be released as long as the list is used.
 *
 * @note The address list should be released by ucs_free().
 */
ucs_status_t ucp_address_unpack(const void *buffer, uint64_t *remote_uuid_p,
                                char *remote_name, size_t max,
                                unsigned *address_count_p,
                                ucp_address_entry_t **address_list_p);


#endif
