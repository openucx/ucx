/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019-2021. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RDMACM_CM_H
#define UCT_RDMACM_CM_H

#include <uct/base/uct_cm.h>
#include <ucs/datastruct/khash.h>
#include <ucs/sys/string.h>
#include <ucs/type/spinlock.h>
#include <ucs/datastruct/bitmap.h>
#if HAVE_DEVX
#include <uct/ib/mlx5/ib_mlx5.h>
#endif

#include <rdma/rdma_cma.h>


KHASH_MAP_INIT_INT64(uct_rdmacm_cm_device_contexts, struct uct_rdmacm_cm_device_context*);


#define UCT_RDMACM_TCP_PRIV_DATA_LEN            56    /** See rdma_connect(3) */
#define UCT_RDMACM_EP_FLAGS_STRING_LEN          128   /** A string to hold the
                                                          representation of the ep flags */
#define UCT_RDMACM_EP_STRING_LEN                192   /** A string to hold the ep info */


typedef struct uct_rdmacm_priv_data_hdr {
    uint8_t length;     /* length of the private data */
} UCS_S_PACKED uct_rdmacm_priv_data_hdr_t;

typedef struct uct_rdmacm_priv_data_hdr_old {
    uct_rdmacm_priv_data_hdr_t hdr;
    uint8_t status;
} UCS_S_PACKED uct_rdmacm_priv_data_hdr_old_t;

/**
 * An rdmacm connection manager
 */
typedef struct uct_rdmacm_cm {
    uct_cm_t                               super;
    struct rdma_event_channel              *ev_ch;
    khash_t(uct_rdmacm_cm_device_contexts) ctxs;

    struct {
        struct sockaddr                    *src_addr;
        double                             timeout;
        ucs_ternary_auto_value_t           reserved_qpn;
    } config;
} uct_rdmacm_cm_t;


typedef struct uct_rdmacm_cm_config {
    uct_cm_config_t          super;
    char                     *src_addr;
    double                   timeout;
    ucs_ternary_auto_value_t reserved_qpn;
} uct_rdmacm_cm_config_t;


/** A reserved qpn block */
typedef struct uct_rdmacm_cm_reserved_qpn_blk {
    uint32_t               first_qpn;             /** Number of the first qpn in the block */
    uint32_t               next_avail_qpn_offset; /** Offset of next available qpn */
    uint32_t               refcount;              /** The counter of qpns which were created and hasn't been destroyed */
    ucs_list_link_t        entry;                 /** List link of blocks */
#ifdef HAVE_DEVX
    struct mlx5dv_devx_obj *obj;                  /** The devx obj used to create the block */
#endif
} uct_rdmacm_cm_reserved_qpn_blk_t;


typedef struct uct_rdmacm_cm_device_context {
    int             use_reserved_qpn;
    ucs_spinlock_t  lock;                         /** Avoid competed condition on the qpn resource for multi-threads */
    ucs_list_link_t blk_list;
    uint32_t        log_reserved_qpn_granularity;
    uint32_t        num_dummy_qps;
    struct ibv_cq   *cq;
    uint8_t         eth_ports;
} uct_rdmacm_cm_device_context_t;


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

size_t uct_rdmacm_cm_get_max_conn_priv();

ucs_status_t uct_rdmacm_cm_get_device_context(uct_rdmacm_cm_t *cm,
                                              struct ibv_context *verbs,
                                              uct_rdmacm_cm_device_context_t **ctx_p);

ucs_status_t
uct_rdmacm_cm_reserved_qpn_blk_alloc(uct_rdmacm_cm_device_context_t *ctx,
                                     struct ibv_context *verbs,
                                     ucs_log_level_t err_level,
                                     uct_rdmacm_cm_reserved_qpn_blk_t **blk_p);

void uct_rdmacm_cm_reserved_qpn_blk_release(
        uct_rdmacm_cm_reserved_qpn_blk_t *blk);

#define RDMACM_HDR_FIRST_BIT 6
#define RDMACM_HDR_REJECT_BIT RDMACM_HDR_FIRST_BIT
#define RDMACM_HDR_SHORT_BIT (RDMACM_HDR_FIRST_BIT + 1)

static UCS_F_ALWAYS_INLINE uint8_t
uct_rdmacm_hdr_get_length(const uct_rdmacm_priv_data_hdr_t *hdr)
{
    return hdr->length & UCS_MASK(RDMACM_HDR_FIRST_BIT);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rdmacm_hdr_get_status(const uct_rdmacm_priv_data_hdr_t *hdr)
{
    ucs_status_t status = UCS_OK;

    if (hdr->length & UCS_BIT(RDMACM_HDR_SHORT_BIT)) {
        if (hdr->length & UCS_BIT(RDMACM_HDR_REJECT_BIT)) {
            status = UCS_ERR_REJECTED;
        }
    } else {
        status = *(uint8_t *)(hdr + 1);
    }
    return status;
}

static UCS_F_ALWAYS_INLINE void
uct_rdmacm_hdr_set_length(uct_rdmacm_priv_data_hdr_t *hdr, uint8_t length)
{
    uint8_t *status;

    hdr->length = length;
    if (length >= (UCT_RDMACM_TCP_PRIV_DATA_LEN - sizeof(*hdr))) {
        hdr->length |= UCS_BIT(RDMACM_HDR_SHORT_BIT);
    } else {
        status = (uint8_t *)(hdr + 1);
        *status = UCS_OK;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_rdmacm_hdr_set_reject(uct_rdmacm_priv_data_hdr_old_t *old_hdr)
{
    old_hdr->hdr.length = 0;
    old_hdr->status = UCS_ERR_REJECTED;
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_rdmacm_hdr_get_hdr_size(const uct_rdmacm_priv_data_hdr_t *hdr)
{
    uint8_t size = sizeof(*hdr);
    if ((hdr->length & UCS_BIT(RDMACM_HDR_SHORT_BIT)) == 0) {
        size++;
    }
    return size;
}

static UCS_F_ALWAYS_INLINE void *
uct_rdmacm_hdr_get_data(const uct_rdmacm_priv_data_hdr_t *hdr)
{
    return UCS_PTR_BYTE_OFFSET(hdr, uct_rdmacm_hdr_get_hdr_size(hdr));
}
#endif
