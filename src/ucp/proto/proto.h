/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_H_
#define UCP_PROTO_H_


/**
 * Defines functions for a protocol, on all possible data types.
 */
typedef struct ucp_proto {
    uct_pending_callback_t     contig_short;           /* Progress short data */
    uct_pending_callback_t     contig_bcopy_single;    /* Progress bcopy single fragment */
    uct_pending_callback_t     contig_bcopy_multi;     /* Progress bcopy multi-fragment */
    uct_pending_callback_t     contig_zcopy_single;    /* Progress zcopy single fragment */
    uct_pending_callback_t     contig_zcopy_multi;     /* Progress zcopy multi-fragment */
    uct_completion_callback_t  contig_zcopy_completion;/* Callback for UCT zcopy completion */
    uct_pending_callback_t     generic_single;         /* Progress bcopy single fragment, generic dt */
    uct_pending_callback_t     generic_multi;          /* Progress bcopy multi-fragment, generic dt */
    size_t                     only_hdr_size;          /* Header size for single / short */
    size_t                     first_hdr_size;         /* Header size for first of multi */
    size_t                     mid_hdr_size;           /* Header size for rest of multi */
} ucp_proto_t;


#endif
