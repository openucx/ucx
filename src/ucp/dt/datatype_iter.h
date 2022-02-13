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
#include <ucp/core/ucp_mm.h>
#include <ucs/memory/memtype_cache.h>
#include <ucs/datastruct/string_buffer.h>


/*
 * dt_mask argument which contains all possible datatypes
 */
#define UCP_DT_MASK_ALL UCS_MASK(UCP_DATATYPE_CLASS_MASK + 1)

/*
 * dt_mask argument which contains contiguous datatype and iov datatype
 */
#define UCP_DT_MASK_CONTIG_IOV \
    (UCS_BIT(UCP_DATATYPE_CONTIG) | UCS_BIT(UCP_DATATYPE_IOV))


/*
 * Iterator on a datatype, used to produce data from send buffer or consume data
 * into a receive buffer.
 */
typedef struct {
    ucp_dt_class_t    dt_class; /* Datatype class (contig/iov/...) */
    ucp_memory_info_t mem_info; /* Memory type and locality, needed to
                                   pack/unpack */
    size_t            length; /* Total packed flat length */
    size_t            offset; /* Current flat offset */
    union {
        struct {
            void                  *buffer;    /* Contiguous buffer pointer */
            ucp_mem_h             memh;       /* Memory registration handle */
        } contig;
        struct {
            ucp_dt_generic_t      *dt_gen;    /* Generic datatype handle */
            void                  *state;     /* User-defined state */
        } generic;
        struct {
            const ucp_dt_iov_t    *iov;       /* IOV list */
#if UCS_ENABLE_ASSERT
            size_t                iov_count;  /* Number of IOV items */
#endif
            size_t                iov_index;  /* Index of current IOV item */
            size_t                iov_offset; /* Offset in the current IOV item */
            ucp_mem_h             *memh;
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


ucs_status_t ucp_datatype_iter_iov_mem_reg(ucp_context_h context,
                                           ucp_datatype_iter_t *dt_iter,
                                           ucp_md_map_t md_map,
                                           unsigned uct_flags);


void ucp_datatype_iter_iov_mem_dereg(ucp_context_h context,
                                     ucp_datatype_iter_t *dt_iter);


size_t ucp_datatype_iter_iov_next_iov(const ucp_datatype_iter_t *dt_iter,
                                      size_t max_length,
                                      ucp_rsc_index_t memh_index,
                                      ucp_datatype_iter_t *next_iter,
                                      uct_iov_t *iov, size_t max_iov);


void ucp_datatype_iter_str(const ucp_datatype_iter_t *dt_iter,
                           ucs_string_buffer_t *strb);

#endif
