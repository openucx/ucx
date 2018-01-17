/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_CONTIG_H_
#define UCP_DT_CONTIG_H_

#include <ucp/api/ucp.h>
#include <ucs/debug/assert.h>


/**
 * Context for memcpy pack callback.
 */
typedef struct {
    const void                    *src;
    size_t                        length;
} ucp_memcpy_pack_context_t;


size_t ucp_memcpy_pack(void *dest, void *arg);


static inline size_t ucp_contig_dt_elem_size(ucp_datatype_t datatype)
{
    return datatype >> UCP_DATATYPE_SHIFT;
}

#define UCP_DT_IS_CONTIG(_datatype) \
          (((_datatype) & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_CONTIG)

static inline size_t ucp_contig_dt_length(ucp_datatype_t datatype, size_t count)
{
    ucs_assert(UCP_DT_IS_CONTIG(datatype));
    return count * ucp_contig_dt_elem_size(datatype);
}

#endif
