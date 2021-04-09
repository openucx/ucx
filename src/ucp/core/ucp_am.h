/**
 * Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_AM_H_
#define UCP_AM_H_


#include <ucs/datastruct/array.h>
#include <ucp/rndv/rndv.h>


#define ucp_am_hdr_from_rts(_rts) \
    ({ \
        UCS_STATIC_ASSERT(sizeof((_rts)->hdr) == sizeof(ucp_am_hdr_t)); \
        ((ucp_am_hdr_t*)&(_rts)->hdr); \
    })


enum {
    UCP_AM_CB_PRIV_FIRST_FLAG = UCS_BIT(15),

    /* Indicates that cb was set with ucp_worker_set_am_recv_handler */
    UCP_AM_CB_PRIV_FLAG_NBX   = UCP_AM_CB_PRIV_FIRST_FLAG
};


/**
 * Data that is stored about each callback registered with a worker
 */
typedef struct ucp_am_entry {
    union {
        ucp_am_callback_t      cb_old;   /* user defined callback, used by legacy API */
        ucp_am_recv_callback_t cb;       /* user defined callback */
    };
    void                       *context;   /* user defined callback argument */
    unsigned                   flags;      /* flags affecting callback behavior
                                              (set by the user) */
} ucp_am_entry_t;


typedef union {
    struct {
        uint16_t             am_id;         /* index into callback array */
        uint16_t             flags;         /* operation flags */
        uint32_t             header_length; /* user header length */
    };

    uint64_t                 u64;     /* this is used to ensure the size of
                                         the header is 64 bytes and aligned */
} UCS_S_PACKED ucp_am_hdr_t;


typedef struct {
    ucp_am_hdr_t             super;
    uint64_t                 ep_id; /* ep which can be used for reply */
} UCS_S_PACKED ucp_am_reply_hdr_t;


typedef struct {
    ucp_am_reply_hdr_t       super;
    uint64_t                 msg_id;     /* method to match parts of the same AM */
    size_t                   total_size; /* length of buffer needed for all data */
} UCS_S_PACKED ucp_am_first_hdr_t;


typedef struct {
    uint64_t                 msg_id;     /* method to match parts of the same AM */
    size_t                   offset;     /* offset in the entire AM buffer */
    uint64_t                 ep_id;      /* ep which can be used for reply */
} UCS_S_PACKED ucp_am_mid_hdr_t;


typedef struct {
    ucs_list_link_t          list;        /* entry into list of unfinished AM's */
    size_t                   remaining;   /* how many bytes left to receive */
} ucp_am_first_desc_t;


ucs_status_t ucp_am_init(ucp_worker_h worker);

void ucp_am_cleanup(ucp_worker_h worker);

void ucp_am_ep_init(ucp_ep_h ep);

void ucp_am_ep_cleanup(ucp_ep_h ep);

size_t ucp_am_max_header_size(ucp_worker_h worker);

ucs_status_t ucp_am_rndv_process_rts(void *arg, void *data, size_t length,
                                     unsigned tl_flags);

UCS_ARRAY_DECLARE_TYPE(ucp_am_cbs, unsigned, ucp_am_entry_t)


#endif
