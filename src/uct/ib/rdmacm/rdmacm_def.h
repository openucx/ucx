/*
 *  * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 *   * See file LICENSE for terms.
 *    */

#ifndef UCT_RDMACM_H
#define UCT_RDMACM_H

#include <uct/api/uct.h>
#include <uct/api/uct_def.h>
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <ucs/type/class.h>
#include <ucs/time/time.h>
#include <ucs/async/async.h>
#include <rdma/rdma_cma.h>
#include <sys/poll.h>

#define UCT_RDMACM_TL_NAME              "rdmacm"
#define UCT_RDMACM_UDP_PRIV_DATA_LEN    136   /** See rdma_accept(3) */
#define UCT_RDMACM_TCP_PRIV_DATA_LEN    56    /** See rdma_connect(3) */


typedef struct uct_rdmacm_iface   uct_rdmacm_iface_t;
typedef struct uct_rdmacm_ep      uct_rdmacm_ep_t;

typedef struct uct_rdmacm_priv_data_hdr {
    uint8_t length;     /* length of the private data */
    int8_t  status;
} uct_rdmacm_priv_data_hdr_t;

typedef struct uct_rdmacm_ctx {
    struct rdma_cm_id  *cm_id;
    uct_rdmacm_ep_t    *ep;
    ucs_list_link_t    list;    /* for list of used cm_ids */
} uct_rdmacm_ctx_t;

ucs_status_t uct_rdmacm_resolve_addr(struct rdma_cm_id *cm_id,
                                     struct sockaddr *addr, int timeout_ms,
                                     ucs_log_level_t log_level);

ucs_status_t uct_rdmacm_ep_resolve_addr(uct_rdmacm_ep_t *ep);

ucs_status_t uct_rdmacm_ep_set_cm_id(uct_rdmacm_iface_t *iface, uct_rdmacm_ep_t *ep);

#endif /* UCT_RDMACM_H */
