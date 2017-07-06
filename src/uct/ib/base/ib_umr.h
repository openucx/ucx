/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * Copyright (C) The University of Tennessee and The University
 *               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_UMR_H_
#define UCT_IB_UMR_H_

#include "ib_md.h"

typedef void(*ep_post_dereg_f)(uct_ep_h tl_ep, struct ibv_exp_send_wr *wr,
                               uct_completion_t *comp);

typedef struct uct_ib_umr {
    unsigned klms;
    unsigned depth;
    int is_inline;
    struct ibv_mr *mr;
    struct ibv_exp_send_wr wr;

    uct_completion_t comp;   /* completion routine */
    ep_post_dereg_f dereg_f; /* endpoint WR posting function pointer */
    uct_ep_t *tl_ep;         /* registering endpoint - for cleanup */
    struct ibv_exp_mem_region *mem_iov;
} uct_ib_umr_t;

ucs_status_t uct_ib_umr_init(uct_ib_md_t *md, unsigned klm_cnt, uct_ib_umr_t *umr);

void uct_ib_umr_finalize(uct_ib_umr_t *umr);


ucs_status_t uct_ib_umr_reg_offset(uct_ib_md_t *md, struct ibv_mr *mr,
                                   off_t offset, struct ibv_mr **offset_mr,
                                   uct_ib_umr_t **umr_p);

ucs_status_t uct_ib_umr_dereg_nc(uct_ib_umr_t *umr);

#endif
