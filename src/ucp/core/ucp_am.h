/**
 * Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_AM_H_
#define UCP_AM_H_

#include "ucp_ep.h"


#define UCP_AM_CB_BLOCK_SIZE 16


/**
 * Data that is stored about each callback registered with a worker
 */
typedef struct ucp_am_entry {
    ucp_am_callback_t        cb;       /* user defined callback*/
    void                     *context; /* user defined callback argument */
    unsigned                 flags;    /* flags affecting callback behavior */
} ucp_am_entry_t;


typedef struct ucp_am_context {
    ucp_am_entry_t           *cbs;          /* array of callbacks and their data */
    size_t                   cbs_array_len; /* len of callbacks array */
} ucp_am_context_t;


typedef union {
    struct {
        uint16_t             am_id;   /* index into callback array */
        uint16_t             flags;   /* operation flags */
        uint32_t             padding;
    };

    uint64_t                 u64;     /* this is used to ensure the size of
                                         the header is 64 bytes and aligned */
} UCS_S_PACKED ucp_am_hdr_t;


typedef struct {
    ucp_am_hdr_t             super;
    uintptr_t                ep_ptr; /* ep which can be used for reply */
} UCS_S_PACKED ucp_am_reply_hdr_t;


typedef struct {
    ucp_am_reply_hdr_t       super;
    uint64_t                 msg_id;     /* method to match parts of the same AM */
    size_t                   total_size; /* length of buffer needed for all data */
} UCS_S_PACKED ucp_am_first_hdr_t;


typedef struct {
    uint64_t                 msg_id;     /* method to match parts of the same AM */
    size_t                   offset;     /* offset in the entire AM buffer */
    uintptr_t                ep_ptr;     /* ep which can be used for reply */
} UCS_S_PACKED ucp_am_mid_hdr_t;


typedef struct {
    ucs_list_link_t          list;        /* entry into list of unfinished AM's */
    size_t                   remaining;   /* how many bytes left to receive */
} ucp_am_first_desc_t;


ucs_status_t ucp_am_init(ucp_worker_h worker);

void ucp_am_cleanup(ucp_worker_h worker);

void ucp_am_ep_init(ucp_ep_h ep);

void ucp_am_ep_cleanup(ucp_ep_h ep);

#endif
