/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* Copyright (C) 2021 Broadcom. ALL RIGHTS RESERVED. The term “Broadcom”
* refers to Broadcom Inc. and/or its subsidiaries.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_VERBS_H
#define UCT_IB_VERBS_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <infiniband/verbs.h>

#include <errno.h>
#include <string.h>

#include <ucs/type/status.h>
#include <ucs/debug/log.h>

/* Read device properties */
#if HAVE_DECL_IBV_QUERY_DEVICE_EX

#  define IBV_DEV_ATTR(_dev, _attr)        ((_dev)->dev_attr.orig_attr._attr)

typedef struct ibv_device_attr_ex uct_ib_device_attr;

static inline ucs_status_t uct_ib_query_device(struct ibv_context *ctx,
                                               uct_ib_device_attr* attr) {
    int ret;

    attr->comp_mask = 0;
    ret = ibv_query_device_ex(ctx, NULL, attr);
    if (ret != 0) {
        ucs_error("ibv_query_device_ex(%s) returned %d: %m",
                  ibv_get_device_name(ctx->device), ret);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

#else

#  define IBV_DEV_ATTR(_dev, _attr)        ((_dev)->dev_attr._attr)

typedef struct ibv_device_attr uct_ib_device_attr;

static inline ucs_status_t uct_ib_query_device(struct ibv_context *ctx,
                                               uct_ib_device_attr* attr) {
    int ret;

    ret = ibv_query_device(ctx, attr);
    if (ret != 0) {
        ucs_error("ibv_query_device(%s) returned %d: %m",
                  ibv_get_device_name(ctx->device), ret);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

#endif


#if !HAVE_DECL_IBV_ACCESS_RELAXED_ORDERING
#  define IBV_ACCESS_RELAXED_ORDERING               0
#endif

#if !HAVE_DECL_IBV_ACCESS_ON_DEMAND
#  define IBV_ACCESS_ON_DEMAND                      0
#endif

/*
 * DC support
 */
#define IBV_DEVICE_HAS_DC(dev)                      (dev->flags & UCT_IB_DEVICE_FLAG_DC)


/*
 * Atomics support
 */
#define IBV_DEVICE_ATOMIC_HCA(dev)              (IBV_DEV_ATTR(dev, atomic_cap) == IBV_ATOMIC_HCA)
#define IBV_DEVICE_ATOMIC_GLOB(dev)             (IBV_DEV_ATTR(dev, atomic_cap) == IBV_ATOMIC_GLOB)


/*
 * On-demand paging support
 */
#if HAVE_STRUCT_IBV_DEVICE_ATTR_EX_ODP_CAPS
#  define IBV_DEVICE_HAS_ODP(_dev)                  ((_dev)->dev_attr.odp_caps.general_caps & IBV_ODP_SUPPORT)
#else
#  define IBV_DEVICE_HAS_ODP(_dev)                  0
#endif


/* Ethernet link layer */
#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
#  define IBV_PORT_IS_LINK_LAYER_ETHERNET(_attr)    ((_attr)->link_layer == IBV_LINK_LAYER_ETHERNET)
#else
#  define IBV_PORT_IS_LINK_LAYER_ETHERNET(_attr)    0
#endif

#if HAVE_DECL_IBV_QPF_GRH_REQUIRED
#  define uct_ib_grh_required(_attr)                ((_attr)->flags & IBV_QPF_GRH_REQUIRED)
#else
#  define uct_ib_grh_required(_attr)                0
#endif

/* Dummy structure declaration, when not present in verbs.h */
#if !HAVE_IBV_DM
    struct ibv_dm;
#endif

typedef uint8_t uct_ib_uint24_t[3];

static inline void uct_ib_pack_uint24(uct_ib_uint24_t buf, uint32_t val)
{
    buf[0] = (val >> 0)  & 0xFF;
    buf[1] = (val >> 8)  & 0xFF;
    buf[2] = (val >> 16) & 0xFF;
}

static inline uint32_t uct_ib_unpack_uint24(const uct_ib_uint24_t buf)
{
    return buf[0] | ((uint32_t)buf[1] << 8) | ((uint32_t)buf[2] << 16);
}

static inline void uct_ib_destroy_qp(struct ibv_qp *qp)
{
    int ret;

    ret = ibv_destroy_qp(qp);
    if (ret) {
        ucs_warn("ibv_destroy_qp() failed: %m");
    }
}

static inline void uct_ib_destroy_srq(struct ibv_srq *srq)
{
    int ret;

    ret = ibv_destroy_srq(srq);
    if (ret) {
        ucs_warn("ibv_destroy_srq() failed: %m");
    }
}

static inline ucs_status_t uct_ib_qp_max_send_sge(struct ibv_qp *qp,
                                                  uint32_t *max_send_sge)
{
    struct ibv_qp_attr qp_attr;
    struct ibv_qp_init_attr qp_init_attr;
    int ret;

    ret = ibv_query_qp(qp, &qp_attr, IBV_QP_CAP, &qp_init_attr);
    if (ret) {
        ucs_error("Failed to query UD QP(ret=%d): %m", ret);
        return UCS_ERR_IO_ERROR;
    }

    *max_send_sge = qp_attr.cap.max_send_sge;

    return UCS_OK;
}

static inline ucs_status_t
uct_ib_query_qp_peer_info(struct ibv_qp *qp, struct ibv_ah_attr *ah_attr,
                          uint32_t *dest_qpn)
{
    struct ibv_qp_attr qp_attr           = {};
    struct ibv_qp_init_attr qp_init_attr = {};
    int ret;

    ret = ibv_query_qp(qp, &qp_attr, IBV_QP_AV | IBV_QP_DEST_QPN,
                       &qp_init_attr);
    if (ret) {
        ucs_error("failed to query qp 0x%u (ret=%d): %m", qp->qp_num, ret);
        return UCS_ERR_IO_ERROR;
    }

    *dest_qpn = qp_attr.dest_qp_num;

    memcpy(ah_attr, &qp_attr.ah_attr, sizeof(*ah_attr));

    return UCS_OK;
}

#endif /* UCT_IB_VERBS_H */
