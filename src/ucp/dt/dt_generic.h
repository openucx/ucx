/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DATATYPE_H_
#define UCP_DATATYPE_H_

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

void ucp_dt_generic_create(ucp_dt_generic_t *dt,
                           const ucp_generic_dt_ops_t *ops,
                           void *context);

#endif
