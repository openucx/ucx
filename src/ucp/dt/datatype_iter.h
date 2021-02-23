/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_DATATYPE_ITER_H_
#define UCP_DATATYPE_ITER_H_

#include "dt.h"
#include "dt_generic.h"

#include <ucp/api/ucp.h>
#include <ucs/memory/memtype_cache.h>


/*
 * Iterator on a datatype, used to produce data from send buffer or consume data
 * into a receive buffer.
 */
typedef struct {
    ucp_dt_class_t    dt_class; /* Datatype class (contig/iov/...) */
    ucs_memory_info_t mem_info; /* Memory type and locality, needed to
                                   pack/unpack */
    size_t            length; /* Total packed flat length */
    size_t            offset; /* Current flat offset */
    union {
        struct {
            void                  *buffer;    /* Contiguous buffer pointer */
            ucp_dt_reg_t          reg;        /* Memory registration state */
        } contig;
        struct {
            ucp_dt_generic_t      *dt_gen;    /* Generic datatype handle */
            void                  *state;     /* User-defined state */
        } generic;
        struct {
            const ucp_dt_iov_t    *iov;       /* IOV list */
            size_t                iov_index;  /* Index of current IOV item */
            size_t                iov_offset; /* Offset in the current IOV item */
            /* TODO support memory registration with IOV */
            /* TODO duplicate the iov array, and save the "start offset" instead
             * of "iov_length" in each element, this way we don't need to keep
             * "iov_offset" field in the iterator, because the "flat length"
             * will be enough to calculate it:
             *   iov_offset = iter.length - iter.iov[iter.iov_index].start_offset
             */
        } iov;
    } type;
} ucp_datatype_iter_t;


#endif
