/**
 * Copyright (C) Los Alamos National Security, LLC. 2019 ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_ep.h"

#define UCP_AM_CB_BLOCK_SIZE 16


typedef union {
    struct {
        uint32_t     length;      /* length of an AM. Ideally it would be size_t
                                   * but we want to keep this struct at 64 bits
                                   * to fit in uct_ep_am_short header. MAX_SHORT
                                   * or b/zcopy MTU
                                   * should be much smaller than this anyway */
        uint16_t     am_id;       /* Index into callback array */
        uint16_t     flags;       /* currently unused in this header 
                                     because replies require long header
                                     defined by @ref ucp_am_send_flags */
    } am_hdr;

    uint64_t u64;                 /* This is used to ensure the size of
                                     the header is 64 bytes and aligned */
} ucp_am_hdr_t;

typedef struct {
    ucp_am_hdr_t super;
    uintptr_t    ep_ptr;
} UCS_S_PACKED ucp_am_reply_hdr_t;

typedef struct {
    size_t            total_size; /* length of buffer needed for all data */
    uint64_t          msg_id;     /* method to match parts of the same AM */
    uintptr_t         ep;         /* end point ptr, used for maintaing list 
                                     of arrivals */
    size_t            offset;     /* how far this message goes into large
                                     the entire AM buffer */
    uint16_t          am_id;      /* index into callback array */
} UCS_S_PACKED ucp_am_long_hdr_t;

typedef struct {
    ucs_list_link_t   list;       /* entry into list of unfinished AM's */
    ucp_recv_desc_t  *all_data;   /* buffer for all parts of the AM */
    uint64_t          msg_id;     /* way to match up all parts of AM */
    size_t            left;
} ucp_am_unfinished_t;

void ucp_am_ep_init(ucp_ep_h ep);

void ucp_am_ep_cleanup(ucp_ep_h ep);
