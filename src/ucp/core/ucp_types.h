/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TYPES_H_
#define UCP_TYPES_H_

#include <ucp/api/ucp.h>
#include <ucs/type/float8.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/bitmap.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/math.h>
#include <stdint.h>


/* Typedef by number of bits */
#define UCP_UINT_TYPE(_bits)         typedef UCS_PP_TOKENPASTE3(uint, _bits, _t)


#define UCP_WORKER_ADDRESS_NAME_MAX  32 /* Worker address name for debugging */
#define UCP_MIN_BCOPY                64 /* Minimal size for bcopy */
#define UCP_FEATURE_AMO              (UCP_FEATURE_AMO32|UCP_FEATURE_AMO64)


/* Resources */
#define UCP_MAX_RESOURCES            128
#define UCP_NULL_RESOURCE            ((ucp_rsc_index_t)-1)
typedef uint8_t                      ucp_rsc_index_t;


/* MDs */
#define UCP_MD_INDEX_BITS            64  /* How many bits are in MD index */
typedef ucp_rsc_index_t              ucp_md_index_t;
#define UCP_MAX_MDS                  ucs_min(UCP_MD_INDEX_BITS, UCP_MAX_RESOURCES)
#define UCP_MAX_OP_MDS               4  /* maximal number of MDs per single op */
UCP_UINT_TYPE(UCP_MD_INDEX_BITS)     ucp_md_map_t;


/* Lanes */
#define UCP_MAX_LANES                6
#define UCP_NULL_LANE                ((ucp_lane_index_t)-1)
typedef uint8_t                      ucp_lane_index_t;
typedef uint8_t                      ucp_lane_map_t;


/* System devices */
#define UCP_MAX_SYS_DEVICES          64
UCP_UINT_TYPE(UCP_MAX_SYS_DEVICES)   ucp_sys_dev_map_t;


/* Worker configuration index for endpoint and rkey */
typedef uint8_t                      ucp_worker_cfg_index_t;
#define UCP_WORKER_MAX_EP_CONFIG     64
#define UCP_WORKER_MAX_RKEY_CONFIG   128
#define UCP_WORKER_CFG_INDEX_NULL    UINT8_MAX


/* Forward declarations */
typedef struct ucp_request            ucp_request_t;
typedef struct ucp_recv_desc          ucp_recv_desc_t;
typedef struct ucp_address_iface_attr ucp_address_iface_attr_t;
typedef struct ucp_address_entry      ucp_address_entry_t;
typedef struct ucp_unpacked_address   ucp_unpacked_address_t;
typedef struct ucp_wireup_ep          ucp_wireup_ep_t;
typedef struct ucp_request_send_proto ucp_request_send_proto_t;
typedef struct ucp_worker_iface       ucp_worker_iface_t;
typedef struct ucp_worker_cm          ucp_worker_cm_t;
typedef struct ucp_rma_proto          ucp_rma_proto_t;
typedef struct ucp_amo_proto          ucp_amo_proto_t;
typedef struct ucp_ep_config          ucp_ep_config_t;
typedef struct ucp_ep_config_key      ucp_ep_config_key_t;
typedef struct ucp_rkey_config_key    ucp_rkey_config_key_t;
typedef struct ucp_proto              ucp_proto_t;
typedef struct ucp_mem_desc           ucp_mem_desc_t;


/**
 * UCP TL bitmap
 *
 * Bitmap type for representing which TL resources are in use.
 */
typedef ucs_bitmap_t(UCP_MAX_RESOURCES) ucp_tl_bitmap_t;


/**
 * Max possible value of TL bitmap (all bits are 1)
 */
extern const ucp_tl_bitmap_t ucp_tl_bitmap_max;


/**
 * Min possible value of TL bitmap (all bits are 0)
 */
extern const ucp_tl_bitmap_t ucp_tl_bitmap_min;


#define UCT_TL_BITMAP_FMT          "0x%lx 0x%lx"
#define UCT_TL_BITMAP_ARG(_bitmap) (_bitmap)->bits[0], (_bitmap)->bits[1]


/**
 * Perform bitwise AND on a TL bitmap and a negation of a bitmap and return the result
 *
 * @param _bitmap1 First operand
 * @param _bitmap2 Second operand
 *
 * @return A new bitmap, which is the logical AND NOT of the operands
 */
#define UCP_TL_BITMAP_AND_NOT(_bitmap1, _bitmap2) \
    UCS_BITMAP_AND(_bitmap1, UCS_BITMAP_NOT(_bitmap2, UCP_MAX_RESOURCES), \
                   UCP_MAX_RESOURCES)


/* Pack bandwidth as bytes/second, range: 512 MB/s to 4 TB/s */
UCS_FP8_DECLARE_TYPE(BANDWIDTH, 512 * UCS_MBYTE, 4 * UCS_TBYTE)


/* Pack latency as nanoseconds, range: 16 nsec to 131 usec */
UCS_FP8_DECLARE_TYPE(LATENCY, UCS_BIT(4), UCS_BIT(17))


/* Pack overhead as nanoseconds, range: 1 nsec to 4 usec */
UCS_FP8_DECLARE_TYPE(OVERHEAD, UCS_BIT(0), UCS_BIT(12))


/**
 * Operation for which protocol is selected
 */
typedef enum {
    UCP_OP_ID_TAG_SEND,
    UCP_OP_ID_TAG_SEND_SYNC,
    UCP_OP_ID_AM_SEND,
    UCP_OP_ID_AM_SEND_REPLY,
    UCP_OP_ID_PUT,
    UCP_OP_ID_GET,
    UCP_OP_ID_AMO_POST,
    UCP_OP_ID_AMO_FETCH,
    UCP_OP_ID_AMO_CSWAP,
    UCP_OP_ID_API_LAST,

    /* Internal rendezvous operations */
    UCP_OP_ID_RNDV_FIRST = UCP_OP_ID_API_LAST,
    UCP_OP_ID_RNDV_SEND  = UCP_OP_ID_RNDV_FIRST,
    UCP_OP_ID_RNDV_RECV,
    UCP_OP_ID_RNDV_RECV_DROP,
    UCP_OP_ID_RNDV_LAST,

    UCP_OP_ID_LAST = UCP_OP_ID_RNDV_LAST
} ucp_operation_id_t;


/**
 * Active message codes
 */
typedef enum {
    UCP_AM_ID_FIRST             =  1, /* First valid AM ID */
    UCP_AM_ID_WIREUP            =  UCP_AM_ID_FIRST, /* Connection establishment */

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

    UCP_AM_ID_PUT               =  17, /* Remote memory write */
    UCP_AM_ID_GET_REQ           =  18, /* Remote memory read request */
    UCP_AM_ID_GET_REP           =  19, /* Remote memory read reply */
    UCP_AM_ID_ATOMIC_REQ        =  20, /* Remote memory atomic request */
    UCP_AM_ID_ATOMIC_REP        =  21, /* Remote memory atomic reply */
    UCP_AM_ID_CMPL              =  22, /* Remote memory operation completion */
    UCP_AM_ID_AM_SINGLE         =  23, /* Single fragment user defined AM */
    UCP_AM_ID_AM_FIRST          =  24, /* First fragment of user defined AM */
    UCP_AM_ID_AM_MIDDLE         =  25, /* Middle or last fragment of user
                                          defined AM */
    UCP_AM_ID_AM_SINGLE_REPLY   =  26, /* Single fragment user defined AM
                                          carrying remote ep for reply */
    UCP_AM_ID_LAST
} ucp_am_id_t;


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
    UCP_RNDV_MODE_AUTO, /* Runtime automatically chooses optimal scheme to use */
    UCP_RNDV_MODE_GET_ZCOPY, /* Use get_zcopy scheme in RNDV protocol */
    UCP_RNDV_MODE_PUT_ZCOPY, /* Use put_zcopy scheme in RNDV protocol */
    UCP_RNDV_MODE_GET_PIPELINE, /* Use pipelined get_zcopy scheme in RNDV protocol */
    UCP_RNDV_MODE_PUT_PIPELINE, /* Use pipelined put_zcopy scheme in RNDV protocol */
    UCP_RNDV_MODE_AM, /* Use active-messages based RNDV protocol */
    UCP_RNDV_MODE_RKEY_PTR, /* Use rkey_ptr in RNDV protocol */
    UCP_RNDV_MODE_LAST
} ucp_rndv_mode_t;


/* Versions enumeration used for various UCP objects (e.g. ucp worker address,
 * sockaddr data structure, etc).
 */
typedef enum {
    UCP_OBJECT_VERSION_V1,
    UCP_OBJECT_VERSION_V2,
    UCP_OBJECT_VERSION_LAST
} ucp_object_version_t;


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
