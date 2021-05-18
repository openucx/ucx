/**
* Copyright (C) Mellanox Technologies Ltd. 2019-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RDMACM_CM_H
#define UCT_RDMACM_CM_H

#include <uct/base/uct_cm.h>
#include <ucs/datastruct/khash.h>
#include <ucs/sys/string.h>

#include <rdma/rdma_cma.h>


KHASH_MAP_INIT_INT64(uct_rdmacm_cm_cqs, struct ibv_cq*);


#define UCT_RDMACM_TCP_PRIV_DATA_LEN    56    /** See rdma_connect(3) */
#define UCT_RDMACM_EP_FLAGS_STRING_LEN  128   /** A string to hold the
                                                  representation of the ep flags */
#define UCT_RDMACM_EP_STRING_LEN        192   /** A string to hold the ep info */


typedef struct uct_rdmacm_priv_data_hdr {
    uint8_t length;     /* length of the private data */
    uint8_t status;
} uct_rdmacm_priv_data_hdr_t;


/**
 * An rdmacm connection manager
 */
typedef struct uct_rdmacm_cm {
    uct_cm_t                   super;
    struct rdma_event_channel  *ev_ch;
    khash_t(uct_rdmacm_cm_cqs) cqs;

    struct {
        struct sockaddr        *src_addr;
        double                 timeout;
    } config;
} uct_rdmacm_cm_t;


typedef struct uct_rdmacm_cm_config {
    uct_cm_config_t super;
    char            *src_addr;
    double          timeout;
} uct_rdmacm_cm_config_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_rdmacm_cm_t, uct_cm_t, uct_component_h,
                           uct_worker_h, const uct_cm_config_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rdmacm_cm_t, uct_cm_t);

static UCS_F_ALWAYS_INLINE ucs_async_context_t *
uct_rdmacm_cm_get_async(uct_rdmacm_cm_t *cm)
{
    uct_priv_worker_t *wpriv = ucs_derived_of(cm->super.iface.worker,
                                              uct_priv_worker_t);

    return wpriv->async;
}

static inline void
uct_rdmacm_cm_id_to_dev_name(struct rdma_cm_id *cm_id, char *dev_name)
{
    ucs_snprintf_zero(dev_name, UCT_DEVICE_NAME_MAX, "%s:%d",
                      ibv_get_device_name(cm_id->verbs->device),
                      cm_id->port_num);
}

ucs_status_t uct_rdmacm_cm_destroy_id(struct rdma_cm_id *id);

ucs_status_t uct_rdmacm_cm_ack_event(struct rdma_cm_event *event);

ucs_status_t uct_rdmacm_cm_reject(uct_rdmacm_cm_t *cm, struct rdma_cm_id *id);

ucs_status_t uct_rdmacm_cm_get_cq(uct_rdmacm_cm_t *cm, struct ibv_context *verbs,
                                  struct ibv_cq **cq);

void uct_rdmacm_cm_cqs_cleanup(uct_rdmacm_cm_t *cm);

size_t uct_rdmacm_cm_get_max_conn_priv();

#endif
