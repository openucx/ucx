/**
 * Copyright (C) Los Alamos National Security, LLC. 2018 ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_ep.h"

#define AM_BLOCK 16

typedef struct{
    uint32_t     total_size; /* length of an AM. Ideally it would be size_t
                              * but we want to keep this struct at 64 bits
                              * to fit in uct_ep_am_short header. MAX_SHORT
                              * should be much smaller than this anyway */
    uint32_t     am_id;      /* Index into callback array */
} UCS_S_PACKED ucp_am_hdr_t;

typedef struct{
    size_t            total_size; /* length of buffer needed for all data */
    uint16_t          am_id;      /* index into callback array */
    uint64_t          msg_id;     /* method to match parts of the same AM */
    uintptr_t         ep;         /* end point ptr, used for maintaing list 
                                     of arrivals */
    size_t            offset;     /* how far this message goes into large
                                     the entire AM buffer */
} UCS_S_PACKED ucp_am_long_hdr_t;

typedef struct{
    ucs_list_link_t   unfinished; /* entry into list of unfinished AM's */
    ucp_recv_desc_t  *all_data;   /* buffer for all parts of the AM */
    uint64_t          msg_id;     /* way to match up all parts of AM */
    size_t            left;
} ucp_am_unfinished_t;

void ucp_am_ep_init(ucp_ep_h ep);

void ucp_am_ep_cleanup(ucp_ep_h ep);
