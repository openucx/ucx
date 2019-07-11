/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/base/uct_cm.h>
#include <ucs/sys/sock.h>
#include "rdmacm_def.h"


/**
 * An rdmacm connection manager
 */
typedef struct uct_rdmacm_cm {
    uct_cm_t                  super;
    struct rdma_event_channel *ev_ch;
    uct_worker_h              worker;
} uct_rdmacm_cm_t;

UCS_CLASS_DECLARE(uct_rdmacm_cm_t, uct_component_h, uct_worker_h);
UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_cm_t, uct_cm_t, uct_component_h, uct_worker_h);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_cm_t, uct_cm_t);

static inline size_t uct_rdmacm_cm_get_max_conn_priv()
{
    return UCT_RDMACM_TCP_PRIV_DATA_LEN - sizeof(uct_rdmacm_priv_data_hdr_t);
}
