/**
 * Copyright (C) Mellanox Technologies Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_H_
#define UCP_DT_H_

#include "dt_contig.h"
#include "dt_iov.h"
#include "dt_generic.h"

#include <ucp/core/ucp_types.h>
#include <uct/api/uct.h>
#include <ucp/api/ucp.h>


/**
 * Memory registration state of a buffer/operation
 */
typedef struct ucp_dt_reg {
    ucp_md_map_t                  md_map;    /* Map of used memory domains */
    uct_mem_h                     memh[UCP_MAX_OP_MDS];
} ucp_dt_reg_t;


/**
 * State of progressing sent/receive operation on a datatype.
 */
typedef struct ucp_dt_state {
    size_t                        offset;  /* Total offset in overall payload. */
    union {
        ucp_dt_reg_t              contig;
        struct {
            size_t                iov_offset;     /* Offset in the IOV item */
            size_t                iovcnt_offset;  /* The IOV item to start copy */
            size_t                iovcnt;         /* Number of IOV buffers */
            ucp_dt_reg_t          *dt_reg;        /* Pointer to IOV memh[iovcnt] */
        } iov;
        struct {
            void                  *state;
        } generic;
    } dt;
} ucp_dt_state_t;


size_t ucp_dt_pack(ucp_datatype_t datatype, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length);

ucs_status_t ucp_mem_type_unpack(ucp_worker_h worker, void *buffer,
                                 const void *recv_data, size_t recv_length,
                                 uct_memory_type_t mem_type);

#endif /* UCP_DT_H_ */

