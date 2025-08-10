/**
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef __VERBS_H
#define __VERBS_H

#include <infiniband/verbs.h>

/* QP related callback */
void dev_qp_wr_start(struct ibv_qp_ex *qp_ex);
void dev_qp_wr_rdma_read(struct ibv_qp_ex *qp_ex, uint32_t rkey,
                         uint64_t remote_addr);
void dev_qp_wr_rdma_write(struct ibv_qp_ex *qp_ex, uint32_t rkey,
                          uint64_t remote_addr);
void dev_qp_wr_set_sge_list(struct ibv_qp_ex *qp, size_t num_sge,
                            const struct ibv_sge *sg_list);
void(dev_qp_wr_set_ud_addr)(struct ibv_qp_ex *qp, struct ibv_ah *ah,
                            uint32_t remote_qpn, uint32_t remote_qkey);
int dev_qp_wr_complete(struct ibv_qp_ex *qp);
#endif /* __VERBS_H */
