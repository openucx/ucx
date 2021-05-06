/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_EFA_H_
#define UCT_IB_EFA_H_


#include <ucs/type/status.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_md.h>
#include <infiniband/efadv.h>


typedef struct uct_ib_efadv {
    struct efadv_device_attr efadv_attr;
} uct_ib_efadv_t;


typedef struct uct_ib_efadv_md {
    uct_ib_md_t     super;
    uct_ib_efadv_t  efadv; /* EFA-specific cached device attributes */
} uct_ib_efadv_md_t;


static inline int uct_ib_efadv_has_rdma_read(const uct_ib_efadv_t *efadv)
{
#ifdef HAVE_DECL_EFA_DV_RDMA_READ
    return (efadv->efadv_attr.device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_READ);
#else
    return 0;
#endif
}

static inline uint32_t uct_ib_efadv_max_sq_wr(const uct_ib_efadv_t *efadv)
{
    return efadv->efadv_attr.max_sq_wr;
}

static inline uint32_t uct_ib_efadv_max_rq_wr(const uct_ib_efadv_t *efadv)
{
    return efadv->efadv_attr.max_rq_wr;
}

static inline uint16_t uct_ib_efadv_max_sq_sge(const uct_ib_efadv_t *efadv)
{
    return efadv->efadv_attr.max_sq_sge;
}

static inline uint16_t uct_ib_efadv_max_rq_sge(const uct_ib_efadv_t *efadv)
{
    return efadv->efadv_attr.max_rq_sge;
}

static inline uint16_t uct_ib_efadv_inline_buf_size(const uct_ib_efadv_t *efadv)
{
    return efadv->efadv_attr.inline_buf_size;
}

static inline uint32_t uct_ib_efadv_max_rdma_size(const uct_ib_efadv_t *efadv)
{
#ifdef HAVE_DECL_EFA_DV_RDMA_READ
    return efadv->efadv_attr.max_rdma_size;
#else
    return 0;
#endif
}

ucs_status_t uct_ib_efadv_query(struct ibv_context *ctx,
                                struct efadv_device_attr *efadv_attr);
#endif
