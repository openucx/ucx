/**
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef __FAKE_H
#define __FAKE_H

#include <infiniband/verbs.h>

#include <stdint.h>

#include "lib.h"

struct fake_device {
    struct ibv_device dev;
    int               id;
};

struct fake_ah {
    struct ibv_ah      ah;
    struct ibv_ah_attr attr;
};

struct fake_pd {
    struct ibv_pd pd;
    uint32_t      lkey;
    array_t       mrs; /* memory registrations */
    array_t       qps; /* queue pairs created */
};

struct fake_mr {
    struct ibv_mr  mr;
    struct fake_pd *fpd;
};

struct fake_qp {
    struct ibv_qp_ex   qp_ex;
    struct fake_pd     *fpd;

    /* Write in preparation */
    struct ibv_send_wr sr;
    struct ibv_ah      *ah;
    uint32_t           remote_qpn;

    /* Posted receives */
    struct list        recv_reqs;
};

struct fake_cq {
    struct ibv_cq cq;
    struct list   wcs;
};

struct fake_cqe {
    struct list    list;
    struct ibv_wc  wc;
    struct fake_cq *fcq; /* owner CQ */
    void (*free)(void*);
};

/* Header for serialized payloads */
struct fake_hdr {
    union ibv_gid gid;
    unsigned      opcode;
    unsigned      src_qp;
    unsigned      qpn;
    struct {
        union ibv_gid src_gid;
        uint32_t      rkey;
        uint64_t      addr;
        int           count;
        size_t        len;
    } rdma;
};

struct fake_recv_wr {
    struct list        list;
    struct ibv_recv_wr wr;
    struct fake_cqe    fcqe;
    struct ibv_sge     sge[];
};

extern int fake_qpn;
extern array_t fake_qps;

#endif /* __FAKE_H */
