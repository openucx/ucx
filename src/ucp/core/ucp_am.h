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

enum {
  UCP_PACKED_RKEY_MAX_SIZE = 256 ,   /* Max supported size for a packed rkey */
  UCP_AM_RDMA_IOVEC_0_MAX_SIZE = 32, /* Amount of iovec[0] carried in AM request */
  UCP_AM_RDMA_THRESHOLD = 65536      /* If iovec[1] is shorter than this, use the non-RDMA path */
};

/* Set UCP_AM_RDMA_VERIFY to 1 if you want the receiver of an AM RDMA to check that the RDMA was performed OK */
#define UCP_AM_RDMA_VERIFY 1

typedef struct {
  size_t            total_size; /* length of buffer needed for all data */
  uint64_t          msg_id;     /* method to match parts of the same AM */
  uintptr_t         ep_ptr;         /* end point ptr, used for maintaing list
                                   of arrivals */
  char              rkey_buffer[UCP_PACKED_RKEY_MAX_SIZE] ; /* Packed remote key */
  uintptr_t         address;     /* Address for RDMA */
  size_t            iovec_0_length ; /* Amount of data transferred by iovec[0], for checking */
  char              iovec_0[UCP_AM_RDMA_IOVEC_0_MAX_SIZE] ; /* iovec[0] carried in request */
  uint16_t          am_id;      /* index into callback array */
#if defined(UCP_AM_RDMA_VERIFY)
  char              iovec_1_first_byte; /* First byte of iovec[1], fro checking */
  char              iovec_1_last_byte; /* Last byte of iovec[1], for checking */
#endif
} UCS_S_PACKED ucp_am_rdma_header_t ;

typedef struct {
  uint64_t          msg_id;     /* method to match parts of the same AM */
  uintptr_t         ep_ptr;     /* end point ptr, used for maintaing list
                                   of arrivals */
} UCS_S_PACKED ucp_am_rdma_completion_header_t ;

typedef struct {
    ucs_list_link_t   list;       /* entry into list of unfinished AM's */
    ucp_recv_desc_t  *all_data;   /* buffer for all parts of the AM */
    uint64_t          msg_id;     /* way to match up all parts of AM */
    size_t            left;
} ucp_am_unfinished_t;

typedef struct {
  ucs_list_link_t   list;                                  /* entry into list of unfinished AM's */
  ucs_status_t      status;                                /* status of the rdma */
  ucp_request_t    *req;                                   /* active message request */
  uint64_t          msg_id;                                /* way to match up all parts of AM */
  ucp_mem_h         memh;                                  /* memory handle for mapping to the adapter */
  ucp_rkey_h        rkey;                                  /* remote memory key */
  ucp_send_callback_t cb ;                                 /* callback to drive when the AM is complete */
  ucp_am_rdma_header_t rdma_header ;                       /* What the client initially sends to the AM server */
} ucp_am_rdma_client_unfinished_t ;

typedef struct {
  ucs_list_link_t   list;                        /* entry into list of unfinished AM's */
  ucp_recv_desc_t  *all_data;                    /* buffer for all parts of the AM */
  uint64_t          msg_id;                      /* way to match up all parts of AM */
  size_t            total_size;                  /* size of data for AM */
  ucp_mem_h         memh;                        /* memory handle for mapping to the adapter */
  ucp_rkey_h        rkey;                        /* key for remote memory */
  ucp_request_t     *request;                    /* request for completion AM */
#if defined(UCP_AM_RDMA_VERIFY)
  size_t            iovec_0_length;              /* Amount of data transferred by iovec[0], for checking */
#endif
  ucp_am_rdma_completion_header_t rdma_completion_header ; /* What the client sents when RDMA is complete */
  uint16_t          am_id ;                      /* active message function index */
#if defined(UCP_AM_RDMA_VERIFY)
  char              iovec_1_first_byte;          /* First byte of iovec[1], fro checking */
  char              iovec_1_last_byte;           /* Last byte of iovec[1], for checking */
#endif
} ucp_am_rdma_server_unfinished_t ;

void ucp_am_ep_init(ucp_ep_h ep);

void ucp_am_ep_cleanup(ucp_ep_h ep);
