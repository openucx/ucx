/**
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <infiniband/efadv.h>
#include <infiniband/verbs.h>

#include <errno.h>
#include <stdlib.h>

#include "fake.h"
#include "verbs.h"
#include "config.h"

int efadv_query_device(struct ibv_context *context,
                       struct efadv_device_attr *attr, uint32_t inlen)
{
    if ((context == NULL) || (inlen != sizeof(efa_dev_attr))) {
        return EINVAL;
    }

    if ((context->device->node_type != IBV_NODE_UNSPECIFIED) ||
        (context->device->transport_type != IBV_TRANSPORT_UNSPECIFIED)) {
        return ENOTSUP;
    }

    memcpy(attr, &efa_dev_attr, sizeof(efa_dev_attr));
    return 0;
}

struct ibv_qp *efadv_create_driver_qp_impl(struct ibv_pd           *pd,
                                           struct ibv_qp_init_attr *attr,
                                           uint32_t driver_qp_type)
{
    struct fake_qp *fqp;
    struct ibv_qp *qp;


    fqp = calloc(1, sizeof(*fqp));
    if (fqp == NULL) {
        return NULL;
    }

    fqp->fpd = (struct fake_pd*)pd;

    fqp->qp_ex.wr_start        = dev_qp_wr_start;
    fqp->qp_ex.wr_rdma_read    = dev_qp_wr_rdma_read;
    fqp->qp_ex.wr_rdma_write   = dev_qp_wr_rdma_write;
    fqp->qp_ex.wr_set_sge_list = dev_qp_wr_set_sge_list;
    fqp->qp_ex.wr_set_ud_addr  = dev_qp_wr_set_ud_addr;
    fqp->qp_ex.wr_complete     = dev_qp_wr_complete;

    list_init(&fqp->recv_reqs);

    qp             = &fqp->qp_ex.qp_base;
    qp->qp_context = pd->context;
    qp->context    = pd->context;
    qp->qp_type    = driver_qp_type;
    qp->send_cq    = attr->send_cq;
    qp->recv_cq    = attr->recv_cq;
    qp->pd         = pd;
    qp->state      = IBV_QPS_RESET;

    lock();
    qp->qp_num = ++fake_qpn;
    array_append(&fake_qps, &fqp, sizeof(fqp));
    unlock();
    return qp;
}

struct ibv_qp *efadv_create_driver_qp(struct ibv_pd           *pd,
                                      struct ibv_qp_init_attr *attr,
                                      uint32_t                driver_qp_type)
{
    if ((attr->qp_type != IBV_QPT_DRIVER) ||
        (driver_qp_type != EFADV_QP_DRIVER_TYPE_SRD)) {
        return NULL;
    }

    return efadv_create_driver_qp_impl(pd, attr, driver_qp_type);
}

struct ibv_qp *efadv_create_qp_ex(struct ibv_context         *context,
                                  struct ibv_qp_init_attr_ex *attr_ex,
                                  struct efadv_qp_init_attr  *efa_attr,
                                  uint32_t                   inlen)
{
    struct ibv_qp_init_attr attr = {
        .qp_type = efa_attr->driver_qp_type,
        .send_cq = attr_ex->send_cq,
        .recv_cq = attr_ex->recv_cq
    };
    (void)context;

    if (inlen != sizeof(*efa_attr)) {
        return NULL;
    }

    return efadv_create_driver_qp_impl(attr_ex->pd, &attr,
                                       efa_attr->driver_qp_type);
}
