/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_ADDRESS_H_
#define UCP_ADDRESS_H_

#include <uct/api/uct.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/sys/math.h>


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
         UCT_IFACE_FLAG_GET_SHORT |
         UCT_IFACE_FLAG_GET_BCOPY |
         UCT_IFACE_FLAG_GET_ZCOPY |
         UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
         UCT_IFACE_FLAG_TAG_RNDV_ZCOPY  |
         UCT_IFACE_FLAG_PENDING
};


/* Which iface event flags would be packed in the address */
enum {
    UCP_ADDRESS_IFACE_EVENT_FLAGS = UCT_IFACE_FLAG_EVENT_RECV
};


enum {
    /* Add worker UUID */
    UCP_ADDRESS_PACK_FLAG_WORKER_UUID = UCS_BIT(0),

    /* Pack worker name */
    UCP_ADDRESS_PACK_FLAG_WORKER_NAME = UCS_BIT(1),

    /* Pack device addresses */
    UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR = UCS_BIT(2),

    /* Pack interface addresses */
    UCP_ADDRESS_PACK_FLAG_IFACE_ADDR  = UCS_BIT(3),

    /* Pack endpoint addresses */
    UCP_ADDRESS_PACK_FLAG_EP_ADDR     = UCS_BIT(4),

    /* Pack TL resource index */
    UCP_ADDRESS_PACK_FLAG_TL_RSC_IDX  = UCS_BIT(5),

    /* Pack system device id */
    UCP_ADDRESS_PACK_FLAG_SYS_DEVICE  = UCS_BIT(6),

    UCP_ADDRESS_PACK_FLAG_LAST,

    /* A bitmap of all flags: UCP_ADDRESS_PACK_FLAG_LAST is the last bit plus 1,
     * so UCP_ADDRESS_PACK_FLAG_LAST<<1 is the next bit plus 2. If we subtract 3
     * we get the next bit minus 1.
     */
    UCP_ADDRESS_PACK_FLAGS_ALL        = (UCP_ADDRESS_PACK_FLAG_LAST << 1) - 3,

    /* Default packing flags for client-server protocol */
    UCP_ADDRESS_PACK_FLAGS_CM_DEFAULT = UCP_ADDRESS_PACK_FLAG_IFACE_ADDR |
                                        UCP_ADDRESS_PACK_FLAG_EP_ADDR,

    /* Suppress debug tracing */
    UCP_ADDRESS_PACK_FLAG_NO_TRACE    = UCS_BIT(16)
};


/**
 * Remote interface attributes.
 */
struct ucp_address_iface_attr {
    uint64_t                    cap_flags;    /* Interface capability flags */
    uint64_t                    event_flags;  /* Interface event capability flags */
    double                      overhead;     /* Interface performance - overhead */
    uct_ppn_bandwidth_t         bandwidth;    /* Interface performance - bandwidth */
    int                         priority;     /* Priority of device */
    double                      lat_ovh;      /* Latency overhead */
    ucp_rsc_index_t             dst_rsc_index;/* Destination resource index */
    ucp_tl_iface_atomic_flags_t atomic;       /* Atomic operations */
};

typedef struct ucp_address_entry_ep_addr {
    ucp_lane_index_t            lane;         /* Lane index (local or remote) */
    const uct_ep_addr_t         *addr;        /* Pointer to ep address */
} ucp_address_entry_ep_addr_t;

/**
 * Address entry.
 */
struct ucp_address_entry {
    const uct_device_addr_t     *dev_addr;      /* Points to device address */
    const uct_iface_addr_t      *iface_addr;    /* Interface address, NULL if not available */
    unsigned                    num_ep_addrs;   /* How many endpoint address are in ep_addrs */
    ucp_address_entry_ep_addr_t ep_addrs[UCP_MAX_LANES]; /* Endpoint addresses */
    ucp_address_iface_attr_t    iface_attr;     /* Interface attributes information */
    uint64_t                    md_flags;       /* MD reg/alloc flags */
    unsigned                    dev_num_paths;  /* Number of paths on the device */
    uint16_t                    tl_name_csum;   /* Checksum of transport name */
    ucp_md_index_t              md_index;       /* Memory domain index */
    ucs_sys_device_t            sys_dev;        /* System device id */
    ucp_rsc_index_t             dev_index;      /* Device index */
};


/**
 * Unpacked remote address
 */
struct ucp_unpacked_address {
    uint64_t                    uuid;           /* Remote worker UUID */
    /* Remote worker address name */
    char                        name[UCP_WORKER_ADDRESS_NAME_MAX];
    unsigned                    address_count;  /* Length of address list */
    ucp_address_entry_t         *address_list;  /* Pointer to address list */
};


/* Iterate over entries in an unpacked address */
#define ucp_unpacked_address_for_each(_elem, _unpacked_address) \
    for (_elem = (_unpacked_address)->address_list; \
         _elem < (_unpacked_address)->address_list + (_unpacked_address)->address_count; \
         ++_elem)


/* Return the index of a specific entry in an unpacked address */
#define ucp_unpacked_address_index(_unpacked_address, _ae) \
    ((int)((_ae) - (_unpacked_address)->address_list))


/**
 * Pack multiple addresses into a buffer, of resources specified in rsc_bitmap.
 * For every resource in rcs_bitmap:
 *    - if iface is CONNECT_TO_IFACE, pack interface address
 *    - if iface is CONNECT_TO_EP, and ep != NULL, and it has a uct_ep on this
 *      resource, pack endpoint address.
 *
 * @param [in]  worker        Worker object whose interface addresses to pack.
 * @param [in]  ep            Endpoint object whose uct_ep addresses to pack.
 *                            Can be set to NULL, to take addresses only from worker.
 * @param [in]  tl_bitmap     Specifies the resources whose transport address
 *                            (ep or iface) should be packed.
 * @param [in]  pack_flags    UCP_ADDRESS_PACK_FLAG_xx flags to specify address
 *                            format.
 * @param [in]  lanes2remote  If NULL, the lane index in each packed ep address
 *                            will be the local lane index. Otherwise, specifies
 *                            which lane index should be packed in the ep address
 *                            for each local lane.
 * @param [out] size_p        Filled with buffer size.
 * @param [out] buffer_p      Filled with pointer to packed buffer. It should be
 *                            released by ucs_free().
 */
ucs_status_t ucp_address_pack(ucp_worker_h worker, ucp_ep_h ep,
                              const ucp_tl_bitmap_t *tl_bitmap,
                              unsigned pack_flags,
                              const ucp_lane_index_t *lanes2remote,
                              size_t *size_p, void **buffer_p);


/**
 * Unpack a list of addresses.
 *
 * @param [in]  worker           Worker object.
 * @param [in]  buffer           Buffer with data to unpack.
 * @param [in]  unpack_flags     UCP_ADDRESS_PACK_FLAG_xx flags to specify
 *                               address format, must be the same as the address
 *                               which was packed by @ref ucp_address_pack.
 * @param [out] unpacked_address Filled with remote address data.
 *
 * @note Entries in the address list could point into the data buffer, so it
 *       should not be released as long as the remote address is used.
 *
 * @note The address list inside @ref ucp_remote_address_t should be released
 *       by ucs_free().
 */
ucs_status_t ucp_address_unpack(ucp_worker_h worker, const void *buffer,
                                unsigned unpack_flags,
                                ucp_unpacked_address_t *unpacked_address);


#endif
