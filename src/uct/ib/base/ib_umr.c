/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ib_umr.h"

#define UCT_IB_UMR_ACCESS_FLAGS  (IBV_ACCESS_LOCAL_WRITE   | \
                                  IBV_ACCESS_REMOTE_WRITE  | \
                                  IBV_ACCESS_REMOTE_READ   | \
                                  IBV_ACCESS_REMOTE_ATOMIC)

// TODO: utilize double "pipe" of UMR indirection by splitting requests?

static void uct_ib_umr_destroy(uct_ib_umr_t *umr)
{
    switch (umr->wr.ext_op.umr.umr_type) {
    case IBV_EXP_UMR_MR_LIST:
        ucs_free(umr->mem_iov);
        break;
    }

    ucs_mpool_put_inline(umr);
}

static void uct_ib_umr_destroy_cb(uct_completion_t *self, ucs_status_t status)
{
    uct_ib_umr_t *umr = ucs_container_of(self, uct_ib_umr_t, comp);
    uct_ib_umr_destroy(umr);
}

ucs_status_t uct_ib_umr_init(uct_ib_md_t *md, unsigned klm_cnt, uct_ib_umr_t *umr)
{
    struct ibv_exp_create_mr_in mrin = {0};

    umr->is_inline  = 0; /* Temporary */

    /* Create memory key */
    mrin.pd                       = md->pd;
#ifdef HAVE_EXP_UMR_NEW_API
    mrin.attr.create_flags        = IBV_EXP_MR_INDIRECT_KLMS;
    mrin.attr.exp_access_flags    = UCT_IB_UMR_ACCESS_FLAGS | IBV_EXP_ACCESS_ALLOCATE_MR;
    mrin.attr.max_klm_list_size   = klm_cnt;
#else
    mrin.attr.create_flags        = IBV_MR_NONCONTIG_MEM;
    mrin.attr.access_flags        = UCT_IB_MEM_ACCESS_FLAGS;
    mrin.attr.max_reg_descriptors = klm_cnt;
#endif

    umr->mr = ibv_exp_create_mr(&mrin);
    if (!umr->mr) {
        umr->klms = 0;
        ucs_error("Failed to create modified_mr: %m");
        return UCS_ERR_NO_MEMORY;
    }

    umr->comp.func  = uct_ib_umr_destroy_cb;
    umr->klms       = klm_cnt;
    return UCS_OK;
}

void uct_ib_umr_finalize(uct_ib_umr_t *umr)
{
    ibv_dereg_mr(umr->mr);
    umr->mr = NULL;
}

static inline
ucs_status_t uct_ib_umr_fill_wr(uct_ib_md_t *md, const uct_iov_t *iov,
                                size_t iovcnt, uct_ib_umr_t *umr)
{
    unsigned mem_idx;
    const uct_iov_t *entry = iov;
    unsigned entry_idx = 0;
    struct ibv_mr *ib_mr;

    if (!umr->is_inline) {
        return UCS_ERR_UNSUPPORTED; // TODO: support...
    }

    umr->wr.exp_opcode             = IBV_EXP_WR_UMR_FILL;
    umr->wr.exp_send_flags         = IBV_EXP_SEND_INLINE |
                                     IBV_EXP_SEND_SIGNALED;
    umr->wr.ext_op.umr.exp_access  = UCT_IB_UMR_ACCESS_FLAGS;
    umr->wr.ext_op.umr.modified_mr = umr->mr;
    umr->wr.ext_op.umr.base_addr   = (uint64_t) entry->buffer;


    umr->wr.ext_op.umr.umr_type = IBV_EXP_UMR_MR_LIST;
    umr->wr.ext_op.umr.mem_list.mem_reg_list = umr->mem_iov;
    while (entry_idx < iovcnt) {
        ib_mr = md->umr.get_mr(entry->memh);
        if (ib_mr->pd != umr->mr->pd) {
            return UCS_ERR_INVALID_PARAM;
        }

        umr->mem_iov[entry_idx].base_addr = (uint64_t) entry->buffer;
        umr->mem_iov[entry_idx].mr        = ib_mr;
        umr->mem_iov[entry_idx].length    = entry->length;

        entry = &iov[++entry_idx];
    }
    mem_idx = iovcnt;

    umr->wr.ext_op.umr.num_mrs = mem_idx;
    return UCS_OK;
}

ucs_status_t uct_ib_umr_create(uct_ib_md_t *md, const uct_iov_t *iov,
                               size_t iovcnt, uct_ep_t *tl_ep,
                               ep_post_dereg_f dereg_f, uct_ib_umr_t **umr_p)
{
#if (HAVE_EXP_UMR && HAVE_EXP_UMR_NEW_API)
    ucs_status_t status;

    uct_ib_umr_t *umr = ucs_mpool_get_inline(&md->umr.mp);

    status = uct_ib_umr_fill_wr(md, iov, iovcnt, umr);
    if (status != UCS_OK) {
        uct_ib_umr_destroy(umr);
        return status;
    }

    umr->dereg_f = dereg_f;
    umr->tl_ep   = tl_ep;
    *umr_p       = umr;
    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t uct_ib_umr_post_md(uct_ib_md_t *md,
                                       struct ibv_exp_send_wr *wr)
{
    struct ibv_wc wc;
    struct ibv_exp_send_wr *bad_wr;

    /* Post UMR */
    int ret = ibv_exp_post_send(md->umr.qp, wr, &bad_wr);
    if (ret) {
        ucs_error("ibv_exp_post_send(QP UMR) failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    /* Wait for send UMR completion */
    for (;;) {
        ret = ibv_poll_cq(md->umr.cq, 1, &wc);
        if (ret < 0) {
            ucs_error("ibv_exp_poll_cq(umr_cq) failed: %m");
            return UCS_ERR_IO_ERROR;
        }
        if (ret == 1) {
            if (wc.status != IBV_WC_SUCCESS) {
                ucs_error("UMR_FILL completed with error: %s vendor_err %d",
                        ibv_wc_status_str(wc.status), wc.vendor_err);
                return UCS_ERR_IO_ERROR;
            }
            break;
        }
    }

    return UCS_OK;
}

void uct_ib_umr_dereg_offset(uct_ep_t *tl_ep, struct ibv_exp_send_wr *wr,
                             uct_completion_t *comp)
{
    uct_ib_md_t *md = (uct_ib_md_t*)tl_ep;
    (void) uct_ib_umr_post_md(md, wr);
    uct_invoke_completion(comp, UCS_OK);
}

ucs_status_t uct_ib_umr_reg_offset(uct_ib_md_t *md, struct ibv_mr *mr,
                                   off_t offset, struct ibv_mr **offset_mr,
                                   uct_ib_umr_t **umr_p)
{
#if (HAVE_EXP_UMR || HAVE_EXP_UMR_NEW_API)
    uct_ib_umr_t *umr;
    ucs_status_t status;
    uct_ib_mem_t ib_mem = {
        .mr = mr
    };
    uct_iov_t iov = {
        .buffer = mr->addr + offset,
        .length = mr->length - offset,
        .memh = &ib_mem,
        .stride = 0,
        .count = 0,
    };

    if (ucs_unlikely(md->umr.qp == NULL)) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_ib_umr_create(md, &iov, 1, (uct_ep_t*)md,
                               uct_ib_umr_dereg_offset, &umr);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_umr_post_md(md, &umr->wr);
    if (status != UCS_OK) {
        uct_ib_umr_destroy(umr);
        return status;
    }

    *offset_mr = umr->mr;
    *umr_p = umr;
    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_ib_umr_dereg_nc(uct_ib_umr_t *umr)
{
#if (HAVE_EXP_UMR || HAVE_EXP_UMR_NEW_API)
    struct ibv_exp_send_wr *wr = &umr->wr;

    memset(wr, 0, sizeof(*wr));
    wr->exp_opcode             = IBV_EXP_WR_UMR_INVALIDATE;
    wr->exp_send_flags         = IBV_EXP_SEND_INLINE |
                                 IBV_EXP_SEND_SIGNALED;
    wr->ext_op.umr.modified_mr = umr->mr;

    umr->dereg_f(umr->tl_ep, wr, &umr->comp);

    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}
