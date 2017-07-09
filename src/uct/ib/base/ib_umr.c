/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ib_umr.h"

#define MAX_UMR_REPEAT_COUNT  ((uint32_t)-1)
#define MAX_UMR_REPEAT_CYCLE  ((uint32_t)-1)
#define MAX_UMR_REPEAT_STRIDE ((uint16_t)-1)
#define MAX_UMR_REPEAT_LENGTH ((uint16_t)-1)

#define UCT_IB_UMR_ACCESS_FLAGS  (IBV_ACCESS_LOCAL_WRITE   | \
                                  IBV_ACCESS_REMOTE_WRITE  | \
                                  IBV_ACCESS_REMOTE_READ   | \
                                  IBV_ACCESS_REMOTE_ATOMIC)

// TODO: utilize double "pipe" of UMR indirection by splitting requests?

static void uct_ib_umr_destroy(uct_ib_umr_t *umr)
{
    switch (umr->wr.ext_op.umr.umr_type) {
    case IBV_EXP_UMR_REPEAT:
        ucs_free(umr->mem_strided);
        ucs_free(umr->repeat_count);
        ucs_free(umr->repeat_length);
        ucs_free(umr->repeat_stride);
        break;
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

    if (!klm_cnt) {
        klm_cnt = IBV_DEVICE_UMR_CAPS(&md->dev.dev_attr, max_send_wqe_inline_klms);
        umr->is_inline  = 1;
    } else {
        umr->is_inline  = 0;
    }

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
    unsigned dim_idx;
    unsigned mem_idx;
    unsigned ilv_idx;
    const uct_iov_t *tmp, *entry = iov;
    unsigned entry_idx = 0;
    struct ibv_mr *ib_mr;
    size_t cycle_length;

    if (!umr->is_inline) {
        return UCS_ERR_UNSUPPORTED; // TODO: support...
    }

    umr->wr.exp_opcode             = IBV_EXP_WR_UMR_FILL;
    umr->wr.exp_send_flags         = IBV_EXP_SEND_INLINE |
                                     IBV_EXP_SEND_SIGNALED;
    umr->wr.ext_op.umr.exp_access  = UCT_IB_UMR_ACCESS_FLAGS;
    umr->wr.ext_op.umr.modified_mr = umr->mr;
    umr->wr.ext_op.umr.base_addr   = (uint64_t) entry->buffer;

    if (entry->stride) {
        umr->wr.ext_op.umr.umr_type = IBV_EXP_UMR_REPEAT;
        umr->wr.ext_op.umr.mem_list.rb.mem_repeat_block_list = umr->mem_strided;
        umr->wr.ext_op.umr.mem_list.rb.repeat_count          = umr->repeat_count;
        umr->wr.ext_op.umr.mem_list.rb.stride_dim            = umr->stride_dim;
        umr->repeat_count[0]                                 = entry->count;
        if (entry->count > MAX_UMR_REPEAT_COUNT) {
            return UCS_ERR_UNSUPPORTED;
        }

        mem_idx = 0;
        cycle_length = 0;
        while (entry_idx < iovcnt) {
            if (umr->repeat_count[0] != entry->count) {
                return UCS_ERR_UNSUPPORTED;
            }

            ib_mr = md->umr.get_mr(entry->memh);
            if (ib_mr->pd != umr->mr->pd) {
                return UCS_ERR_INVALID_PARAM;
            }

            for (tmp = entry, ilv_idx = entry->ilv_ratio; ilv_idx > 0; ilv_idx--) {
                entry = tmp; /* repeat the same group of entries*/
                dim_idx = umr->stride_dim * mem_idx;
                umr->mem_strided[mem_idx].base_addr  = (uint64_t) entry->buffer;
                umr->mem_strided[mem_idx].mr         = ib_mr;
                umr->mem_strided[mem_idx].stride     = &umr->repeat_stride[dim_idx];
                umr->mem_strided[mem_idx].byte_count = &umr->repeat_length[dim_idx];

                do {
                    if ((entry->length > MAX_UMR_REPEAT_STRIDE) ||
                        (entry->stride > MAX_UMR_REPEAT_LENGTH)) {
                        return UCS_ERR_UNSUPPORTED;
                    }

                    umr->repeat_length[dim_idx] = entry->length;
                    umr->repeat_stride[dim_idx] = entry->stride;
                    cycle_length += entry->length;
                    dim_idx++;

                    entry = &iov[++entry_idx];
                } while (entry->buffer == NULL);
                mem_idx++;
            }
        }
        if (cycle_length > MAX_UMR_REPEAT_CYCLE) {
            return UCS_ERR_UNSUPPORTED;
        }
    } else {
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
    }

    umr->wr.ext_op.umr.num_mrs = mem_idx;
    return UCS_OK;
}

static inline
ucs_status_t uct_ib_umr_update_wr(const uct_iov_t *iov, size_t iovcnt,
                                  uct_ib_umr_t *umr)
{
    unsigned mem_idx = 0, iov_idx;
    uint64_t addr;

    if (iov[0].stride) {
        for (iov_idx = 0; iov_idx < iovcnt; iov_idx++) {
            addr = (uint64_t) iov[iov_idx].buffer;
            if (addr) {
                umr->mem_strided[mem_idx++].base_addr = addr;
            }
        }
    } else {
        for (iov_idx = 0; iov_idx < iovcnt; iov_idx++) {
            addr = (uint64_t) iov[iov_idx].buffer;
            if (addr) {
                umr->mem_iov[mem_idx++].base_addr = addr;
            }
        }
    }

    return UCS_OK;
}

static inline
ucs_status_t uct_ib_md_calc_required_klms(const struct ibv_exp_device_attr *dev_attr,
                                          const uct_iov_t *iov, size_t iovcnt,
                                          unsigned *klms_needed,
                                          unsigned *stride_dim,
                                          unsigned *depth)
{
    unsigned iov_idx, klm_cnt = 0;
    unsigned iov_depth, umr_depth = 0;
    unsigned max_depth = IBV_DEVICE_UMR_CAPS(dev_attr, max_umr_recursion_depth);
    unsigned max_dim = IBV_DEVICE_UMR_CAPS(dev_attr, max_umr_stride_dimension);
    unsigned dim_cnt = (unsigned)-1, dim_check = (unsigned)-1;

    for (iov_idx = 0; iov_idx < iovcnt; iov_idx++) {
        uct_mem_h iov_memh = iov[iov_idx].memh;
        if (ucs_unlikely(!iov_memh)) {
            return UCS_ERR_INVALID_PARAM;
        }

        /* Check if recursion depth limit is exceeded */
        iov_depth = ((uct_ib_mem_t*)(iov_memh))->umr_depth;
        if (iov_depth > umr_depth) {
            if (iov_depth >= max_depth) {
                return UCS_ERR_UNSUPPORTED;
            }
            umr_depth = iov_depth;
        }

        /* Count stride dimension and KLMs required */
        if (iov[iov_idx].buffer == NULL) {
            if (++dim_cnt >= max_dim) {
                return UCS_ERR_UNSUPPORTED;
            }
        } else {
            if (dim_cnt != dim_check) {
                if (dim_check == (unsigned)-1) {
                    dim_check = dim_cnt;
                } else {
                    return UCS_ERR_INVALID_PARAM;
                }
            }
            if (iov[iov_idx].stride) {
                dim_cnt = 1;
                klm_cnt += iov[iov_idx].ilv_ratio;
            } else {
                dim_cnt = 0;
                klm_cnt++;
            }
        }
    }

    /* Check last iov array element */
    if (dim_cnt != dim_check) {
        if (dim_check == (unsigned)-1) {
            dim_check = dim_cnt;
        } else {
            return UCS_ERR_INVALID_PARAM;
        }
    }

    if (!max_dim && dim_check) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (klm_cnt > IBV_DEVICE_UMR_CAPS(dev_attr, max_klm_list_size)) {
        return UCS_ERR_UNSUPPORTED;
    }

    *klms_needed = klm_cnt;
    *stride_dim = dim_check;
    *depth = umr_depth + 1;
    return UCS_OK;
}

static inline
ucs_status_t uct_ib_umr_alloc(uct_ib_md_t *md, unsigned klms,
                              unsigned stride_dim, uct_ib_umr_t **umr_p)
{
    uct_ib_umr_t *umr = ucs_mpool_get_inline(&md->umr.mp);

    if (klms > umr->klms) {
        ucs_status_t status;
        uct_ib_umr_finalize(umr);
        status = uct_ib_umr_init(md, klms, umr);
        if (status != UCS_OK) {
            ucs_mpool_put_inline(umr);
            return status;
        }
    }

    if (stride_dim) {
        umr->stride_dim = stride_dim;
        umr->mem_strided = ucs_malloc(klms * sizeof(*umr->mem_strided), "umr_repeat");
        if (umr->mem_strided == NULL) {
            goto alloc_none;
        }

        umr->repeat_count = ucs_malloc(stride_dim * sizeof(size_t), "umr_count");
        if (umr->repeat_count == NULL) {
            goto alloc_strided;
        }

        umr->repeat_length = ucs_malloc(stride_dim * klms * sizeof(size_t), "umr_length");
        if (umr->repeat_length == NULL) {
            goto alloc_count;
        }

        umr->repeat_stride = ucs_malloc(stride_dim * klms * sizeof(size_t), "umr_stride");
        if (umr->repeat_stride == NULL) {
            goto alloc_length;
        }
    } else {
        umr->mem_iov = ucs_malloc(klms * sizeof(struct ibv_exp_mem_region), "umr_iov");
        if (umr->mem_iov == NULL) {
            goto alloc_none;
        }
    }

    umr->comp.count = 1;
    *umr_p = umr;
    return UCS_OK;

alloc_length:
    ucs_free(umr->repeat_length);
alloc_count:
    ucs_free(umr->repeat_count);
alloc_strided:
    ucs_free(umr->mem_strided);
alloc_none:
    ucs_mpool_put_inline(umr);
    return UCS_ERR_NO_MEMORY;
}

ucs_status_t uct_ib_umr_create(uct_ib_md_t *md, const uct_iov_t *iov,
                               size_t iovcnt, uct_ep_t *tl_ep,
                               ep_post_dereg_f dereg_f, uct_ib_umr_t **umr_p)
{
#if (HAVE_EXP_UMR && HAVE_EXP_UMR_NEW_API)
    ucs_status_t status;
    unsigned klms_needed;
    unsigned stride_dim;
    unsigned umr_depth;
    uct_ib_umr_t *umr;

    if (!IBV_EXP_HAVE_UMR(&md->dev.dev_attr)) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_ib_md_calc_required_klms(&md->dev.dev_attr, iov, iovcnt,
                                          &klms_needed, &stride_dim, &umr_depth);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_umr_alloc(md, klms_needed, stride_dim, &umr);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_umr_fill_wr(md, iov, iovcnt, umr);
    if (status != UCS_OK) {
        uct_ib_umr_destroy(umr);
        return status;
    }

    umr->depth   = umr_depth;
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

ucs_status_t uct_ib_umr_reg_nc(uct_md_h uct_md, const uct_iov_t *iov,
                               size_t iovcnt, uct_ep_h tl_ep,
                               ep_post_dereg_f dereg_f, uct_ib_mem_t *memh,
                               struct ibv_exp_send_wr **wr_p)
{
#if (HAVE_EXP_UMR || HAVE_EXP_UMR_NEW_API)
    uct_ib_umr_t *umr;
    ucs_status_t status;

    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    if (ucs_unlikely(md->umr.qp == NULL)) {
        return UCS_ERR_UNSUPPORTED;
    }

    UCS_STATS_UPDATE_COUNTER(md->stats, UCT_IB_MD_STAT_MEM_REG_NC, +1);
    if (ucs_unlikely(memh->umr == NULL)) {
        status = uct_ib_umr_create(md, iov, iovcnt, tl_ep, dereg_f, &umr);
        if (status != UCS_OK) {
            return status;
        }

        memh->mr        = umr->mr;
        memh->umr       = umr;
        memh->lkey      = umr->mr->lkey;
        memh->flags     = UCT_IB_MEM_FLAG_NC_MR;
        memh->umr_depth = umr->depth;
        *wr_p              = &umr->wr;
        return UCS_OK;
    }

    *wr_p = &memh->umr->wr;
    return uct_ib_umr_update_wr(iov, iovcnt, memh->umr);
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



