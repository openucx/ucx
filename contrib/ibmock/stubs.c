/**
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "verbs.h"

#include <stdio.h>
#include <stdlib.h>


int ibv_cmd_destroy_srq(struct ibv_srq *srq)
{
    (void)srq;
    return ENOTSUP;
}

int ibv_cmd_destroy_wq(struct ibv_wq *wq)
{
    (void)wq;
    return ENOTSUP;
}

int ibv_cmd_destroy_cq(struct ibv_cq *cq)
{
    (void)cq;
    return ENOTSUP;
}

struct ibv_open_xrcd_resp;
struct ibv_open_xrcd;
struct verbs_xrcd;

int ibv_cmd_open_xrcd(struct ibv_context *context, struct verbs_xrcd *xrcd,
                      int vxrcd_size, struct ibv_xrcd_init_attr *attr,
                      struct ibv_open_xrcd *cmd, size_t cmd_size,
                      struct ibv_open_xrcd_resp *resp, size_t resp_size)
{
    (void)context;
    (void)xrcd;
    (void)vxrcd_size;
    (void)attr;
    (void)cmd;
    (void)cmd_size;
    (void)resp;
    (void)resp_size;
    return ENOTSUP;
}

int ibv_cmd_close_xrcd(struct verbs_xrcd *xrcd)
{
    (void)xrcd;
    return ENOTSUP;
}

int ibv_cmd_attach_mcast(struct ibv_qp *qp, const union ibv_gid *gid,
                         uint16_t lid)
{
    (void)qp;
    (void)gid;
    (void)lid;
    return ENOTSUP;
}

int ibv_cmd_detach_mcast(struct ibv_qp *qp, const union ibv_gid *gid,
                         uint16_t lid)
{
    (void)qp;
    (void)gid;
    (void)lid;
    return ENOTSUP;
}

struct ibv_create_qp_resp_ex;
struct ibv_create_qp_ex;
struct verbs_qp *qp;

int ibv_cmd_create_qp_ex2(struct ibv_context *context, struct verbs_qp *qp,
                          int vqp_sz, struct ibv_qp_init_attr_ex *qp_attr,
                          struct ibv_create_qp_ex *cmd, size_t cmd_core_size,
                          size_t cmd_size, struct ibv_create_qp_resp_ex *resp,
                          size_t resp_core_size, size_t resp_size)
{
    (void)context;
    (void)qp;
    (void)vqp_sz;
    (void)qp_attr;
    (void)cmd;
    (void)cmd_core_size;
    (void)cmd_size;
    (void)resp;
    (void)resp_core_size;
    (void)resp_size;

    return ENOTSUP;
}

struct verbs_mr;

int ibv_cmd_query_mr(struct ibv_pd *pd, struct verbs_mr *vmr,
                     uint32_t mr_handle)
{
    (void)pd;
    (void)vmr;
    (void)mr_handle;

    return ENOTSUP;
}

int ibv_cmd_create_flow(struct ibv_qp *qp, struct ibv_flow *flow_id,
                        struct ibv_flow_attr *flow_attr, void *ucmd,
                        size_t ucmd_size)
{
    (void)qp;
    (void)flow_id;
    (void)flow_attr;
    (void)ucmd;
    (void)ucmd_size;

    return ENOTSUP;
}

struct verbs_cq;
struct ibv_create_cq_ex;

int ibv_cmd_create_cq_ex(struct ibv_context *context,
                         const struct ibv_cq_init_attr_ex *cq_attr,
                         struct verbs_cq *cq, struct ibv_create_cq_ex *cmd,
                         size_t cmd_size,
                         struct ib_uverbs_ex_create_cq_resp *resp,
                         size_t resp_size, uint32_t cmd_flags)
{
    (void)context;
    (void)cq_attr;
    (void)cq;
    (void)cmd;
    (void)cmd_size;
    (void)resp;
    (void)resp_size;
    (void)cmd_flags;

    return ENOTSUP;
}

int ibv_cmd_dealloc_mw(struct ibv_mw *mw)
{
    (void)mw;

    return ENOTSUP;
}

int ibv_cmd_dealloc_pd(struct ibv_pd *pd)
{
    (void)pd;

    return ENOTSUP;
}

struct ibv_command_buffer;

struct ibv_mr *ibv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset,
                                 size_t length, uint64_t iova, int fd,
                                 int access)
{
    (void)pd;
    (void)offset;
    (void)length;
    (void)iova;
    (void)fd;
    (void)access;
    return NULL; /* not supported */
}

int ibv_cmd_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length,
                          uint64_t iova, int fd, int access,
                          struct verbs_mr *vmr,
                          struct ibv_command_buffer *driver)
{
    (void)pd;
    (void)offset;
    (void)length;
    (void)iova;
    (void)fd;
    (void)access;
    (void)vmr;
    (void)driver;

    return ENOTSUP;
}

struct ibv_query_port;

int ibv_cmd_query_port(struct ibv_context *context, uint8_t port_num,
                       struct ibv_port_attr *port_attr,
                       struct ibv_query_port *cmd, size_t cmd_size)
{
    (void)context;
    (void)port_num;
    (void)port_attr;
    (void)cmd;
    (void)cmd_size;

    return ENOTSUP;
}

struct ibv_modify_qp;

int ibv_cmd_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                      int attr_mask, struct ibv_modify_qp *cmd, size_t cmd_size)
{
    (void)qp;
    (void)attr;
    (void)attr_mask;
    (void)cmd;
    (void)cmd_size;

    return ENOTSUP;
}

struct ibv_srq *ibv_create_srq(struct ibv_pd *pd,
                               struct ibv_srq_init_attr *srq_init_attr)
{
    (void)pd;
    (void)srq_init_attr;
    return NULL;
}
