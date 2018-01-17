/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TYPES_H_
#define UCP_TYPES_H_

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/sys/preprocessor.h>
#include <stdint.h>


#define UCP_WORKER_NAME_MAX          32   /* Worker name for debugging */
#define UCP_MIN_BCOPY                64   /* Minimal size for bcopy */

/* Resources */
#define UCP_MAX_RESOURCES            UINT8_MAX
#define UCP_NULL_RESOURCE            ((ucp_rsc_index_t)-1)
typedef uint8_t                      ucp_rsc_index_t;

/* MDs */
#define UCP_UINT_TYPE(_bits)         typedef UCS_PP_TOKENPASTE(UCS_PP_TOKENPASTE(uint, _bits), _t)
#define UCP_MD_INDEX_BITS            64  /* How many bits are in MD index */
typedef ucp_rsc_index_t              ucp_md_index_t;
#define UCP_MAX_MDS                  ucs_min(UCP_MD_INDEX_BITS, UCP_MAX_RESOURCES)
#define UCP_MAX_OP_MDS               3  /* maximal number of MDs per single op */
UCP_UINT_TYPE(UCP_MD_INDEX_BITS)     ucp_md_map_t;

/* Lanes */
#define UCP_MAX_LANES                8
#define UCP_NULL_LANE                ((ucp_lane_index_t)-1)
typedef uint8_t                      ucp_lane_index_t;
UCP_UINT_TYPE(UCP_MAX_LANES)         ucp_lane_map_t;


/* Forward declarations */
typedef struct ucp_request              ucp_request_t;
typedef struct ucp_recv_desc            ucp_recv_desc_t;
typedef struct ucp_address_iface_attr   ucp_address_iface_attr_t;
typedef struct ucp_address_entry        ucp_address_entry_t;
typedef struct ucp_wireup_ep            ucp_wireup_ep_t;
typedef struct ucp_proto                ucp_proto_t;
typedef struct ucp_worker_iface         ucp_worker_iface_t;


/**
 * Active message codes
 */
enum {
    UCP_AM_ID_WIREUP            =  1, /* Connection establishment */

    UCP_AM_ID_EAGER_ONLY        =  2, /* Single packet eager TAG */
    UCP_AM_ID_EAGER_FIRST       =  3, /* First eager fragment */
    UCP_AM_ID_EAGER_MIDDLE      =  4, /* Middle eager fragment */

    UCP_AM_ID_EAGER_SYNC_ONLY   =  6, /* Single packet eager-sync */
    UCP_AM_ID_EAGER_SYNC_FIRST  =  7, /* First eager-sync fragment */
    UCP_AM_ID_EAGER_SYNC_ACK    =  8, /* Eager-sync acknowledge */

    UCP_AM_ID_RNDV_RTS          =  9, /* Ready-to-Send to init rendezvous */
    UCP_AM_ID_RNDV_ATS          =  10, /* Ack-to-Send after finishing a get operation */
    UCP_AM_ID_RNDV_RTR          =  11, /* Ready-to-Receive rendezvous for a receiver
                                          with a generic datatype */
    UCP_AM_ID_RNDV_DATA         =  12, /* Rndv data fragments when using software
                                          rndv (bcopy) */
    UCP_AM_ID_OFFLOAD_SYNC_ACK  =  14, /* Eager sync ack for tag offload proto */

    UCP_AM_ID_STREAM_DATA       =  15, /* Eager STREAM packet */

    UCP_AM_ID_RNDV_ATP          =  16, /* Ack-to-put complete after finishing a put_zcopy */

    UCP_AM_ID_LAST
};


/**
 * Atomic operations mode.
 */
typedef enum {
    UCP_ATOMIC_MODE_CPU,     /* Use CPU-based atomics */
    UCP_ATOMIC_MODE_DEVICE,  /* Use device-based atomics */
    UCP_ATOMIC_MODE_GUESS,   /* If all transports support CPU AMOs only (no DEVICE),
                              * the CPU is selected, otherwise DEVICE is selected */
    UCP_ATOMIC_MODE_LAST
} ucp_atomic_mode_t;


/**
 * Communication scheme in RNDV protocol.
 */
typedef enum {
    UCP_RNDV_MODE_GET_ZCOPY, /* Use get_zcopy scheme in RNDV protocol */
    UCP_RNDV_MODE_PUT_ZCOPY, /* Use put_zcopy scheme in RNDV protocol */
    UCP_RNDV_MODE_AUTO,      /* Runtime automatically chooses optimal scheme to use */
    UCP_RNDV_MODE_LAST
} ucp_rndv_mode_t;

/**
 * Active message tracer.
 */
typedef void (*ucp_am_tracer_t)(ucp_worker_h worker, uct_am_trace_type_t type,
                                uint8_t id, const void *data, size_t length,
                                char *buffer, size_t max);


/**
 * Internal callback for UCP requests
 */
typedef void (*ucp_request_callback_t)(ucp_request_t *req);


#endif
