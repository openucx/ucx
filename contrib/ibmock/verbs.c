/**
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "verbs.h"
#include "lib.h"
#include "fake.h"

#include <sys/uio.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>

#define NUM_DEVS   2
#define SYS_PATH   "/tmp/ibmock"
#define DUMMY_PKEY 65535 /* Dummy pkey with full membership for now */

static pthread_mutex_t global_lock = PTHREAD_MUTEX_INITIALIZER;

void lock(void)
{
    pthread_mutex_lock(&global_lock);
}

void unlock(void)
{
    pthread_mutex_unlock(&global_lock);
}

bool verbs_allow_disassociate_destroy = 0;

enum be_mode {
    BE_NOTSET,
    BE_LOOPBACK
};

static enum be_mode be_mode_get(void)
{
    return BE_LOOPBACK;
}

struct ibv_device **ibv_get_device_list(int *num_devices)
{
    struct fake_device *fake;
    struct ibv_device **devs;
    int i;

    *num_devices = NUM_DEVS;
    devs         = calloc(*num_devices + 1, sizeof(*devs));
    if (devs == NULL) {
        return NULL;
    }

    for (i = 0; i < *num_devices; i++) {
        fake = calloc(1, sizeof(*fake));
        if (fake == NULL) {
            goto failure;
        }

        fake->id = i;

        devs[i]                 = &fake->dev;
        devs[i]->node_type      = IBV_NODE_UNSPECIFIED;
        devs[i]->transport_type = IBV_TRANSPORT_UNSPECIFIED;

        sprintf(devs[i]->name, "rdmap%d", i);
        sprintf(devs[i]->dev_name, "uverbs%d", i);
        sprintf(devs[i]->dev_path, "%s/sys/class/infiniband/rdmap%d",
                SYS_PATH, i);
        sprintf(devs[i]->ibdev_path, "%s/sys/class/infiniband/uverbs%d",
                SYS_PATH, i);
    }

    devs[i] = NULL;
    return devs;

failure:
    while (i-- > 0) {
        free(devs[i]);
    }

    free(devs);
    return NULL;
}

void ibv_free_device_list(struct ibv_device **list)
{
    int i;

    for (i = 0; i < NUM_DEVS; i++) {
        free(list[i]);
    }

    free(list);
}

const char *ibv_get_sysfs_path(void)
{
    return SYS_PATH;
}

const char *ibv_get_device_name(struct ibv_device *device)
{
    return device->name;
}

int ibv_fork_init(void)
{
    return 0;
}

static int vctx_query_port(struct ibv_context *context, uint8_t port_num,
                           struct ibv_port_attr *port_attr,
                           size_t port_attr_len)
{
    (void)context;

    if ((port_num != 1) || (port_attr_len != sizeof(*port_attr))) {
        return EINVAL;
    }

    memcpy(port_attr, &efa_ib_port_attr, sizeof(*port_attr));
    return 0;
}

#undef ibv_query_port
int ibv_query_port(struct ibv_context *context, uint8_t port_num,
                   struct _compat_ibv_port_attr *port_attr)
{
    return vctx_query_port(context, port_num, (struct ibv_port_attr*)port_attr,
                           sizeof(struct ibv_port_attr));
}

struct fake_recv_wr *fake_recv_wr_create(struct ibv_recv_wr *wr)
{
    (void)wr;
    return NULL;
}

static int dev_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                         struct ibv_recv_wr **bad_wr)
{
    struct fake_qp *fqp = (struct fake_qp*)qp;
    struct fake_recv_wr *recv_wr;

    lock();
    for (; wr; wr = wr->next) {
        recv_wr = calloc(1, sizeof(*recv_wr) +
                         ( wr->num_sge * sizeof(*recv_wr->sge)));
        if (recv_wr == NULL) {
            if (bad_wr) {
                *bad_wr = wr;
            }
            unlock();
            return -1;
        }

        memcpy(&recv_wr->wr, wr, sizeof(*wr));
        memcpy(recv_wr->sge, wr->sg_list, sizeof(*recv_wr->sge) * wr->num_sge);
        list_add_tail(&fqp->recv_reqs, &recv_wr->list);
    }

    unlock();
    return 0;
}

#define MAX_SGE  64
#define GRH_SIZE 40

static void fake_recv_wr_free(void *ptr)
{
    struct fake_recv_wr *recv = container_of(ptr, struct fake_recv_wr, fcqe);
    free(recv);
}

static enum ibv_wc_opcode dev_wc_opcode(enum ibv_wr_opcode opcode)
{
    if (opcode == IBV_WR_SEND) {
        return IBV_WC_SEND;
    } else if (opcode == IBV_WR_RDMA_READ) {
        return IBV_WC_RDMA_READ;
    }

    return IBV_WC_RDMA_READ;
}

static int
rx_send(struct fake_qp *fqp, struct fake_hdr *hdr, struct iovec *iov, int count)
{
    struct fake_recv_wr *recv;
    struct fake_cq *fcq;
    struct ibv_sge *sge;
    struct ibv_recv_wr *wr;
    int i, j, src_len, dst_len, len;
    size_t src_off, dst_off, total = 0;

    if (list_is_empty(&fqp->recv_reqs)) {
        return 1;
    }

    src_off = sizeof(*hdr);
    dst_off = 0;

    recv = list_first(&fqp->recv_reqs);
    list_del(&recv->list);

    wr  = &recv->wr;
    sge = recv->sge;

    for (i = 0, j = 0; i < count; i++, src_off = 0) {
        while (src_off < iov[i].iov_len) {
            src_len = iov[i].iov_len - src_off;
            dst_len = sge[j].length - dst_off;
            if (!dst_len) {
                j++;
                dst_off = 0;
                if (j >= wr->num_sge) {
                    fprintf(stderr, "ibmock: error: posted RX too short\n");
                    return -1;
                }
            }

            len = min(src_len, dst_len);
            memcpy((void*)sge[j].addr + dst_off, iov[i].iov_base + src_off,
                   len);
            src_off += len;
            dst_off += len;
            total   += len;
        }
    }

    recv->fcqe.wc.status   = IBV_WC_SUCCESS;
    recv->fcqe.wc.wr_id    = wr->wr_id;
    recv->fcqe.wc.src_qp   = hdr->src_qp;
    recv->fcqe.wc.qp_num   = fqp->qp_ex.qp_base.qp_num;
    recv->fcqe.wc.byte_len = total;
    recv->fcqe.free        = fake_recv_wr_free;

    fcq = (struct fake_cq*)fqp->qp_ex.qp_base.recv_cq;
    list_add_tail(&fcq->wcs, &recv->fcqe.list);
    return 1;
}

static int
rx_rdma(struct fake_qp *fqp, struct fake_hdr *hdr, struct iovec *iov, int count)
{
    int found = 0;
    struct fake_mr **mr;
    struct {
        void     *addr;
        uint32_t len;
    } __attribute__((packed)) dest[hdr->rdma.count];
    int i;
    size_t src_off, dst_off, total, len;

    array_foreach(mr, &fqp->fpd->mrs) {
        if ((*mr)->mr.lkey == hdr->rdma.rkey) {
            /* TODO check addr ranges, deduplicate mr key lookup */
            found = 1;
        }
    }

    if (!found) {
        fprintf(stderr, "ibmock: rx rdma rkey not found PD of QP#%d\n",
               fqp->qp_ex.qp_base.qp_num);
        return 0;
    }

    i       = 0;
    src_off = sizeof(*hdr);
    dst_off = 0;
    for (dst_off = 0; dst_off < sizeof(dest);) {
        while (i < count && src_off >= iov[i].iov_len) {
            src_off = 0;
            i++;
        }

        if (i >= count) {
            fprintf(stderr, "ibmock: IO error\n");
            return 0;
        }

        len = min(sizeof(dest) - dst_off, iov[i].iov_len - src_off);
        memcpy((void*)dest + dst_off, iov[i].iov_base + src_off, len);
        dst_off += len;
        src_off += len;
    }

    total = 0;
    for (i = 0; i < hdr->rdma.count; i++) {
        memcpy(dest[i].addr, (void*)hdr->rdma.addr + total, dest[i].len);
        total += dest[i].len;
    }

    assert(total == hdr->rdma.len);
    return 1;
}

static int dev_rx_cb(struct iovec *iov, int count)
{
    struct fake_hdr *hdr = iov[0].iov_base;
    uint8_t *gid         = hdr->gid.raw;
    struct fake_qp *fqp  = NULL;
    struct fake_qp **entry;
    int ret;

    if (gid[13] != be_mode_get()) {
        return 0; /* not ours */
    }

    /* Lookup QP */
    array_foreach(entry, &fake_qps) {
        if ((*entry)->qp_ex.qp_base.qp_num == hdr->qpn) {
            fqp = *entry;
            break;
        }
    }

    if (fqp == NULL) {
        return 1;
    }

    if (hdr->opcode == IBV_WR_SEND) {
        ret = rx_send(fqp, hdr, iov, count);
    } else if (hdr->opcode == IBV_WR_RDMA_READ) {
        ret = rx_rdma(fqp, hdr, iov, count);
    }

    return ret;
}

static void dev_send_comp(void *arg, int ret)
{
    struct fake_cqe *fcqe = arg;

    if (ret < 0) {
        fcqe->wc.status = IBV_WC_GENERAL_ERR;
    }

    list_add_tail(&fcqe->fcq->wcs, &fcqe->list);
}

static int dev_wr_send_serialize(struct ibv_qp *qp, struct ibv_send_wr *wr,
                                 struct ibv_ah *ah, uint32_t remote_qpn)
{
    union {
        struct fake_hdr hdr;
        char            _padding[sizeof(struct fake_hdr) + GRH_SIZE];
    } u;
    struct fake_hdr *hdr = &u.hdr;
    size_t total         = 0;
    struct iovec iov[2 * MAX_SGE + 1];
    struct fake_ah *fah;
    struct fake_cqe *fcqe;
    struct ibv_wc *wc;
    int i, ret, count;

    if ((wr->opcode != IBV_WR_SEND) && (wr->opcode != IBV_WR_RDMA_READ)) {
        return -1;
    }

    fah = (struct fake_ah*)ah;

    memcpy(&hdr->gid, &fah->attr.grh.dgid, sizeof(hdr->gid));
    hdr->opcode = wr->opcode;
    hdr->src_qp = qp->qp_num;
    hdr->qpn    = remote_qpn;

    iov[0].iov_base = hdr;
    iov[0].iov_len  = sizeof(*hdr);

    if (qp->qp_type == IBV_QPT_UD) {
        iov[0].iov_len += GRH_SIZE;
    }

    for (i = 0; i < wr->num_sge; i++) {
        if (wr->opcode == IBV_WR_SEND) {
            iov[1 + i].iov_base = (void*)wr->sg_list[i].addr;
            iov[1 + i].iov_len  = wr->sg_list[i].length;

            total += iov[1 + i].iov_len;
        } else if (wr->opcode == IBV_WR_RDMA_READ) {
            iov[1 + (2 * i)].iov_base = &wr->sg_list[i].addr;
            iov[1 + (2 * i)].iov_len  = sizeof(wr->sg_list[i].addr);
            iov[2 + (2 * i)].iov_base = &wr->sg_list[i].length;
            iov[2 + (2 * i)].iov_len  = sizeof(wr->sg_list[i].length);

            total += wr->sg_list[i].length;
        }
    }

    count = wr->num_sge + 1;
    if (wr->opcode == IBV_WR_RDMA_READ) {
        hdr->rdma.rkey  = wr->wr.rdma.rkey;
        hdr->rdma.addr  = wr->wr.rdma.remote_addr;
        hdr->rdma.len   = total;
        hdr->rdma.count = wr->num_sge;

        count += wr->num_sge;
    }

    /* Pre-generate completion queue entry */
    fcqe = malloc(sizeof(*fcqe));
    if (fcqe == NULL) {
        return -1;
    }

    fcqe->fcq  = (struct fake_cq*)qp->send_cq;
    fcqe->free = free;

    wc           = &fcqe->wc;
    wc->status   = IBV_WC_SUCCESS;
    wc->wr_id    = wr->wr_id;
    wc->opcode   = dev_wc_opcode(wr->opcode);
    wc->qp_num   = remote_qpn;
    wc->src_qp   = qp->qp_num;
    wc->byte_len = total;

    /* TODO: Use actual backend for multi process/nodes */
    ret = dev_rx_cb(iov, count);
    if (ret == 0) {
        free(fcqe);
        return -1;
    }

    dev_send_comp(fcqe, ret);
    return 0;
}

static int dev_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
                         struct ibv_send_wr **bad_wr)
{
    int found[MAX_SGE];
    struct fake_qp *fqp = (struct fake_qp*)qp;
    struct fake_mr **mr;
    int i, ret, total;

    lock();
    for (; wr; wr = wr->next) {
        if (wr->opcode != IBV_WR_SEND) {
            *bad_wr = wr;
            ret     = -EINVAL;
            goto fail;
        }

        if (wr->num_sge > MAX_SGE) {
            ret = -EINVAL;
            goto fail;
        }

        if (!(wr->send_flags & IBV_SEND_INLINE)) {
            total = 0;
            memset(found, 0, sizeof(found));
            array_foreach(mr, &fqp->fpd->mrs) {
                for (i = 0; i < wr->num_sge; i++) {
                    if ((*mr)->mr.lkey == wr->sg_list[i].lkey) {
                        total   += !found[i];
                        found[i] = 1;
                    }
                }

                if (total == wr->num_sge) {
                    break;
                }
            }

            if (total != wr->num_sge) {
                ret = -EINVAL;
                goto fail;
            }
        }


        if (dev_wr_send_serialize(qp, wr, wr->wr.ud.ah, wr->wr.ud.remote_qpn)) {
            ret = -ENOMEM;
            goto fail;
        }
    }

    unlock();
    return 0;

fail:
    unlock();
    return ret;
}

static int dev_poll_cq(struct ibv_cq *cq, int num_entries, struct ibv_wc *wc)
{
    struct fake_cq *fcq = (struct fake_cq*)cq;
    struct fake_cqe *fcqe;
    int i;

    lock();
    for (i = 0; i < num_entries; i++) {
        if (list_is_empty(&fcq->wcs)) {
            break;
        }

        fcqe = (struct fake_cqe*)list_first(&fcq->wcs);
        list_del(&fcqe->list);
        wc[i] = fcqe->wc;
        fcqe->free(fcqe);
    }

    unlock();
    return i;
}

struct ibv_context *ibv_open_device(struct ibv_device *device)
{
    struct verbs_context *vctx;
    int fds[2];

    vctx = calloc(1, sizeof(*vctx));
    if (vctx == NULL) {
        return NULL;
    }

    vctx->context.device = malloc(sizeof(struct fake_device));
    if (vctx->context.device == NULL) {
        free(vctx);
        return NULL;
    }
    vctx->context.ops.post_send = dev_post_send;
    vctx->context.ops.post_recv = dev_post_recv;
    vctx->context.ops.poll_cq   = dev_poll_cq;

    if (pipe(fds)) {
        free(vctx);
        return NULL;
    }

    vctx->context.cmd_fd           = fds[0];
    vctx->context.async_fd         = fds[1];
    vctx->context.num_comp_vectors = 8,

    memcpy(vctx->context.device, device, sizeof(struct fake_device));
    vctx->context.abi_compat = __VERBS_ABI_IS_EXTENDED;
    vctx->query_port         = vctx_query_port;

    return &vctx->context;
}

int ibv_close_device(struct ibv_context *context)
{
    close(context->async_fd);
    close(context->cmd_fd);
    free(context->device);
    free(verbs_get_ctx(context));
    return 0;
}

struct ibv_pd *ibv_alloc_pd(struct ibv_context *context)
{
    struct fake_pd *fpd = calloc(1, sizeof(*fpd));
    struct ibv_pd *pd;

    if (fpd == NULL) {
        return NULL;
    }

    array_init(&fpd->mrs, sizeof(struct fake_mr*));
    array_init(&fpd->qps, sizeof(struct fake_qp*));

    pd          = &fpd->pd;
    pd->context = context;
    return pd;
}

int ibv_dealloc_pd(struct ibv_pd *pd)
{
    struct fake_pd *fpd = (struct fake_pd*)pd;

    array_cleanup(&fpd->mrs);
    array_cleanup(&fpd->qps);
    free(fpd);
    return 0;
}

struct ibv_cq *ibv_create_cq(struct ibv_context *context, int cqe,
                             void *cq_context, struct ibv_comp_channel *channel,
                             int comp_vector)
{
    struct fake_cq *fcq;
    struct ibv_cq *cq;

    (void)comp_vector;

    fcq = calloc(1, sizeof(*fcq));
    if (fcq == NULL) {
        return NULL;
    }

    cq             = &fcq->cq;
    cq->context    = context;
    cq->channel    = channel;
    cq->cqe        = cqe;
    cq->cq_context = cq_context;

    list_init(&fcq->wcs);
    return cq;
}

int ibv_destroy_cq(struct ibv_cq *cq)
{
    struct fake_cq *fcq = (struct fake_cq*)cq;
    struct fake_cqe *fcqe;

    while (!list_is_empty(&fcq->wcs)) {
        fcqe = (struct fake_cqe*)list_first(&fcq->wcs);
        list_del(&fcqe->list);
        fcqe->free(fcqe);
    }

    free(fcq);
    return 0;
}

int fake_qpn = 0;
array_t fake_qps;

struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *attr)
{
    struct fake_qp *fqp;
    struct ibv_qp *qp;

    if ((attr->qp_type != IBV_QPT_DRIVER) && (attr->qp_type != IBV_QPT_UD)) {
        return NULL; /* RC is not supported */
    }

    if ((attr->cap.max_inline_data > efa_dev_attr.inline_buf_size) ||
        (attr->cap.max_send_sge > efa_dev_attr.max_sq_sge)) {
        return NULL;
    }

    fqp = calloc(1, sizeof(*fqp));
    if (fqp == NULL) {
        return NULL;
    }

    lock();
    fqp->fpd = (struct fake_pd*)pd;
    list_init(&fqp->recv_reqs);

    qp          = &fqp->qp_ex.qp_base;
    qp->context = pd->context;
    qp->qp_type = attr->qp_type;
    qp->send_cq = attr->send_cq;
    qp->recv_cq = attr->recv_cq;
    qp->pd      = pd;
    qp->srq     = attr->srq;
    qp->state   = IBV_QPS_RESET;
    qp->qp_num  = ++fake_qpn;

    array_append(&fake_qps, &fqp, sizeof(fqp));
    unlock();
    return qp;
}


int ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask)
{
    (void)attr_mask;

    lock();

    switch (attr->qp_state) {
    case IBV_QPS_INIT:
        if (qp->state != IBV_QPS_RESET) {
            goto fail;
        }
        break;
    case IBV_QPS_RTR:
        if (qp->state != IBV_QPS_INIT) {
            goto fail;
        }

        break;
    case IBV_QPS_RTS:
        if (qp->state != IBV_QPS_RTR) {
            goto fail;
        }

        break;
    default:
        unlock();
        return ENOTSUP;
    }

    qp->state = attr->qp_state;
    unlock();
    return 0;

fail:
    unlock();
    return EINVAL;
}

int ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask,
                 struct ibv_qp_init_attr *init_attr)
{
    struct ibv_qp_init_attr qp_init_attr = efa_ib_qp_init_attr;
    (void)attr_mask;

    lock();
    qp_init_attr.qp_context = qp->context;
    qp_init_attr.send_cq    = qp->send_cq;
    qp_init_attr.recv_cq    = qp->recv_cq;
    qp_init_attr.qp_type    = qp->qp_type;

    memcpy(attr, &efa_ib_qp_attr, sizeof(efa_ib_qp_attr));
    memcpy(init_attr, &qp_init_attr, sizeof(qp_init_attr));
    unlock();
    return 0;
}

int ibv_destroy_qp(struct ibv_qp *qp)
{
    struct fake_qp **entry, *fqp = (struct fake_qp*)qp;
    struct fake_recv_wr *fake_wr;

    lock();
    array_foreach(entry, &fake_qps) {
        if (fqp == *entry) {
            array_remove(&fake_qps, entry);

            while (!list_is_empty(&fqp->recv_reqs)) {
                fake_wr = list_first(&fqp->recv_reqs);
                list_del(&fake_wr->list);
                free(fake_wr);
            }

            free(fqp);
            unlock();
            return 0;
        }
    }

    fprintf(stderr, "ibmock: failed to destroy QP#%d\n", qp->qp_num);
    unlock();
    return -1;
}

/* verbs.h header re/post defines ibv_reg_mr() */
#undef ibv_reg_mr

struct ibv_mr *ibv_reg_mr_iova2(struct ibv_pd *pd, void *addr, size_t length,
                                uint64_t iova, unsigned int access)
{
    struct fake_pd *fpd = (struct fake_pd*)pd;
    struct fake_mr *fmr;
    struct ibv_mr *mr;
    (void)iova;

    fmr = calloc(1, sizeof(*fmr));
    if (fmr == NULL) {
        return NULL;
    }

    lock();
    mr          = &fmr->mr;
    mr->context = pd->context;
    mr->pd      = pd;
    mr->addr    = addr;
    mr->length  = length;
    mr->handle  = access;
    mr->rkey    = mr->lkey = ++fpd->lkey;

    fmr->fpd = fpd;
    array_append(&fpd->mrs, &fmr, sizeof(fmr));
    unlock();

    return mr;
}

struct ibv_mr *
ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access)
{
    return ibv_reg_mr_iova2(pd, addr, length, (uint64_t)addr, access);
}

int ibv_dereg_mr(struct ibv_mr *mr)
{
    struct fake_mr **entry, *fmr = (struct fake_mr*)mr;

    lock();
    array_foreach(entry, &fmr->fpd->mrs) {
        if (fmr == *entry) {
            array_remove(&fmr->fpd->mrs, entry);
            free(fmr);
            unlock();
            return 0;
        }
    }

    free(fmr);
    fprintf(stderr, "ibmock: failed to deregister MR lkey=%x\n", mr->lkey);
    unlock();
    return -1;
}

int ibv_query_pkey(struct ibv_context *context, uint8_t port_num, int index,
                   uint16_t *pkey)
{
    (void)context;
    (void)port_num;
    (void)index;

    if ((port_num != 1) || (index != 0)) {
        return EINVAL;
    }

    *pkey = DUMMY_PKEY;
    return 0;
}

int ibv_query_gid(struct ibv_context *context, uint8_t port_num, int index,
                  union ibv_gid *gid)
{
    struct fake_device *fdevice = (struct fake_device*)context->device;
    /* Construct an arbitrary GID based on default GID prefix */
    uint8_t addr[16] = {0xfe,        0x80,          0x00,     0x00, 0x00, 0x00,
                        0x00,        0x00,          0x04,     0x07, 0x64, 0xff,
                        fdevice->id, be_mode_get(), port_num, index};

    (void)context;
    memcpy(gid->raw, addr, sizeof(gid->raw));
    return 0;
}

int ibv_query_device(struct ibv_context *context,
                     struct ibv_device_attr *device_attr)
{
    (void)context;
    memcpy(device_attr, &efa_ibv_dev_attr, sizeof(*device_attr));
    return 0;
}

__be64 ibv_get_device_guid(struct ibv_device *device)
{
    (void)device;
    errno = ENOTSUP;
    return 0;
}

int ibv_get_device_index(struct ibv_device *device)
{
    struct fake_device *fake = (struct fake_device*)device;

    return fake->id;
}

struct ibv_ah *ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr)
{
    struct fake_ah *fah = calloc(1, sizeof(*fah));
    static int ah_handle;

    if (fah == NULL) {
        return NULL;
    }

    lock();
    fah->ah.context = pd->context;
    fah->ah.pd      = pd;
    fah->ah.handle  = ++ah_handle;
    unlock();

    memcpy(&fah->attr, attr, sizeof(*attr));
    return &fah->ah;
}

int ibv_destroy_ah(struct ibv_ah *ah)
{
    free(ah);
    return 0;
}

struct ibv_qp_ex *ibv_qp_to_qp_ex(struct ibv_qp *qp)
{
    return (struct ibv_qp_ex*)qp;
}

void dev_qp_wr_start(struct ibv_qp_ex *qp_ex)
{
    struct fake_qp *fqp = (struct fake_qp*)qp_ex;

    memset(&fqp->sr, 0, sizeof(fqp->sr));
}

void dev_qp_wr_rdma_read(struct ibv_qp_ex *qp_ex, uint32_t rkey,
                         uint64_t remote_addr)
{
    struct fake_qp *fqp = (struct fake_qp*)qp_ex;

    fqp->sr.opcode              = IBV_WR_RDMA_READ;
    fqp->sr.wr.rdma.rkey        = rkey;
    fqp->sr.wr.rdma.remote_addr = remote_addr;
}

void dev_qp_wr_set_sge_list(struct ibv_qp_ex *qp_ex, size_t num_sge,
                            const struct ibv_sge *sg_list)
{
    struct fake_qp *fqp = (struct fake_qp*)qp_ex;

    fqp->sr.sg_list = (struct ibv_sge*)sg_list;
    fqp->sr.num_sge = num_sge;
}

void dev_qp_wr_set_ud_addr(struct ibv_qp_ex *qp_ex, struct ibv_ah *ah,
                           uint32_t remote_qpn, uint32_t remote_qkey)
{
    struct fake_qp *fqp = (struct fake_qp*)qp_ex;

    (void)remote_qkey;

    fqp->ah         = ah;
    fqp->remote_qpn = remote_qpn;
}

int dev_qp_wr_complete(struct ibv_qp_ex *qp_ex)
{
    struct fake_qp *fqp = (struct fake_qp*)qp_ex;
    int ret;

    lock();
    fqp->sr.wr_id = qp_ex->wr_id;

    ret = dev_wr_send_serialize(&qp_ex->qp_base, &fqp->sr, fqp->ah,
                                fqp->remote_qpn);
    unlock();
    return ret;
}

__attribute__((constructor)) void verbs_ctor(void)
{
    array_init(&fake_qps, sizeof(struct fake_qp*));
}

__attribute__((destructor)) void verbs_dtor(void)
{
    array_cleanup(&fake_qps);
}

/* rdma-core copy/paste */
void ibv_copy_ah_attr_from_kern(struct ibv_ah_attr *dst,
                                struct ib_uverbs_ah_attr *src)
{
    memcpy(dst->grh.dgid.raw, src->grh.dgid, sizeof dst->grh.dgid);
    dst->grh.flow_label    = src->grh.flow_label;
    dst->grh.sgid_index    = src->grh.sgid_index;
    dst->grh.hop_limit     = src->grh.hop_limit;
    dst->grh.traffic_class = src->grh.traffic_class;

    dst->dlid          = src->dlid;
    dst->sl            = src->sl;
    dst->src_path_bits = src->src_path_bits;
    dst->static_rate   = src->static_rate;
    dst->is_global     = src->is_global;
    dst->port_num      = src->port_num;
}

/* rdma-core copy/paste */
const char *ibv_node_type_str(enum ibv_node_type node_type)
{
    static const char *const node_type_str[] = {
        [IBV_NODE_CA]          = "InfiniBand channel adapter",
        [IBV_NODE_SWITCH]      = "InfiniBand switch",
        [IBV_NODE_ROUTER]      = "InfiniBand router",
        [IBV_NODE_RNIC]        = "iWARP NIC",
        [IBV_NODE_USNIC]       = "usNIC",
        [IBV_NODE_USNIC_UDP]   = "usNIC UDP",
        [IBV_NODE_UNSPECIFIED] = "unspecified",
    };

    if ((node_type < IBV_NODE_CA) || (node_type > IBV_NODE_UNSPECIFIED)) {
        return "unknown";
    }

    return node_type_str[node_type];
}

/* rdma-core copy/paste */
const char *ibv_port_state_str(enum ibv_port_state port_state)
{
    static const char *const port_state_str[] = {
        [IBV_PORT_NOP]          = "no state change (NOP)",
        [IBV_PORT_DOWN]         = "down",
        [IBV_PORT_INIT]         = "init",
        [IBV_PORT_ARMED]        = "armed",
        [IBV_PORT_ACTIVE]       = "active",
        [IBV_PORT_ACTIVE_DEFER] = "active defer"
    };

    if ((port_state < IBV_PORT_NOP) || (port_state > IBV_PORT_ACTIVE_DEFER)) {
        return "unknown";
    }

    return port_state_str[port_state];
}

/* rdma-core copy/paste */
const char *ibv_event_type_str(enum ibv_event_type event)
{
    static const char *const event_type_str[] = {
        [IBV_EVENT_CQ_ERR]        = "CQ error",
        [IBV_EVENT_QP_FATAL]      = "local work queue catastrophic error",
        [IBV_EVENT_QP_REQ_ERR]    = "invalid request local work queue error",
        [IBV_EVENT_QP_ACCESS_ERR] = "local access violation work queue error",
        [IBV_EVENT_COMM_EST]      = "communication established",
        [IBV_EVENT_SQ_DRAINED]    = "send queue drained",
        [IBV_EVENT_PATH_MIG]      = "path migrated",
        [IBV_EVENT_PATH_MIG_ERR]  = "path migration request error",
        [IBV_EVENT_DEVICE_FATAL]  = "local catastrophic error",
        [IBV_EVENT_PORT_ACTIVE]   = "port active",
        [IBV_EVENT_PORT_ERR]      = "port error",
        [IBV_EVENT_LID_CHANGE]    = "LID change",
        [IBV_EVENT_PKEY_CHANGE]   = "P_Key change",
        [IBV_EVENT_SM_CHANGE]     = "SM change",
        [IBV_EVENT_SRQ_ERR]       = "SRQ catastrophic error",
        [IBV_EVENT_SRQ_LIMIT_REACHED]   = "SRQ limit reached",
        [IBV_EVENT_QP_LAST_WQE_REACHED] = "last WQE reached",
        [IBV_EVENT_CLIENT_REREGISTER]   = "client reregistration",
        [IBV_EVENT_GID_CHANGE]          = "GID table change",
        [IBV_EVENT_WQ_FATAL]            = "WQ fatal"
    };

    if ((event < IBV_EVENT_CQ_ERR) || (event > IBV_EVENT_WQ_FATAL)) {
        return "unknown";
    }

    return event_type_str[event];
}

/* rdma-core copy/paste */
const char *ibv_wc_status_str(enum ibv_wc_status status)
{
    static const char *const wc_status_str[] = {
        [IBV_WC_SUCCESS]            = "success",
        [IBV_WC_LOC_LEN_ERR]        = "local length error",
        [IBV_WC_LOC_QP_OP_ERR]      = "local QP operation error",
        [IBV_WC_LOC_EEC_OP_ERR]     = "local EE context operation error",
        [IBV_WC_LOC_PROT_ERR]       = "local protection error",
        [IBV_WC_WR_FLUSH_ERR]       = "Work Request Flushed Error",
        [IBV_WC_MW_BIND_ERR]        = "memory management operation error",
        [IBV_WC_BAD_RESP_ERR]       = "bad response error",
        [IBV_WC_LOC_ACCESS_ERR]     = "local access error",
        [IBV_WC_REM_INV_REQ_ERR]    = "remote invalid request error",
        [IBV_WC_REM_ACCESS_ERR]     = "remote access error",
        [IBV_WC_REM_OP_ERR]         = "remote operation error",
        [IBV_WC_RETRY_EXC_ERR]      = "transport retry counter exceeded",
        [IBV_WC_RNR_RETRY_EXC_ERR]  = "RNR retry counter exceeded",
        [IBV_WC_LOC_RDD_VIOL_ERR]   = "local RDD violation error",
        [IBV_WC_REM_INV_RD_REQ_ERR] = "remote invalid RD request",
        [IBV_WC_REM_ABORT_ERR]      = "aborted error",
        [IBV_WC_INV_EECN_ERR]       = "invalid EE context number",
        [IBV_WC_INV_EEC_STATE_ERR]  = "invalid EE context state",
        [IBV_WC_FATAL_ERR]          = "fatal error",
        [IBV_WC_RESP_TIMEOUT_ERR]   = "response timeout error",
        [IBV_WC_GENERAL_ERR]        = "general error",
        [IBV_WC_TM_ERR]             = "TM error",
        [IBV_WC_TM_RNDV_INCOMPLETE] = "TM software rendezvous",
    };

    if ((status < IBV_WC_SUCCESS) || (status > IBV_WC_TM_RNDV_INCOMPLETE)) {
        return "unknown";
    }

    return wc_status_str[status];
}
