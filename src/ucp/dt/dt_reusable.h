/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_REUSABLE_H_
#define UCP_DT_REUSABLE_H_

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>

#define UCP_DT_IS_REUSABLE(_datatype) \
    ((((_datatype) & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_IOV_R) || \
     (((_datatype) & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_STRIDE_R))

#define UCP_DT_GET_REUSABLE(_datatype) (&ucp_dt_ptr(_datatype)->reusable)

typedef struct ucp_dt_reusable {
    size_t length;            /* Optimization: cache the length of the datatype */
    union {
        uct_mem_h* iov_memh;  /* Array of (UCP-)IOV memory handles */
        uct_mem_h stride_memh;/* Handle for the full extent of the stride */
    };

    uct_mem_h nc_memh;        /* Non-contiguous registration - single handle */
    uct_md_t *nc_md;          /* Memory domain used for creating contig_memh */
    uct_completion_t nc_comp; /* Non-contiguous registration completion */
    ucs_status_t nc_status;   /* Non-contiguous registration status */
    uct_iov_t *nc_iov;        /* Optimization: cache the registration layout */
    size_t nc_iovcnt;         /*   - and the registration layout length */
} ucp_dt_reusable_t;

void ucp_dt_reusable_destroy(ucp_dt_reusable_t *dt);

void ucp_dt_reusable_completion(uct_completion_t *self, ucs_status_t status);

#endif
