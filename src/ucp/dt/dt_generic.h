/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_GENERIC_H_
#define UCP_DT_GENERIC_H_

#include <ucp/api/ucp.h>


/**
 * Generic datatype structure.
 */
typedef struct ucp_dt_generic {
    void                     *context;
    ucp_generic_dt_ops_t     ops;
} ucp_dt_generic_t;


#define UCP_DT_IS_GENERIC(_datatype) \
    (((_datatype) & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_GENERIC)


static UCS_F_ALWAYS_INLINE
ucp_dt_generic_t* ucp_dt_to_generic(ucp_datatype_t datatype)
{
    return (ucp_dt_generic_t*)(void*)(datatype & ~UCP_DATATYPE_CLASS_MASK);
}


static UCS_F_ALWAYS_INLINE
ucp_datatype_t ucp_dt_from_generic(ucp_dt_generic_t* dt_gen)
{
    return ((uintptr_t)dt_gen) | UCP_DATATYPE_GENERIC;
}

#endif
