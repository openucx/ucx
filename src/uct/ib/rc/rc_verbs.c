/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_iface.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_context.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/tl/context.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <string.h>
#include <arpa/inet.h> /* For htonl */

#define UCT_IB_MAX_WC 32


typedef struct {
    uct_rc_ep_t        super;

    struct {
        unsigned       available;
    } tx;
} uct_rc_verbs_ep_t;

typedef struct {
    uct_rc_iface_t     super;
    struct ibv_send_wr inl_rwrite_wr;
    struct ibv_sge     inl_sg;
} uct_rc_verbs_iface_t;


static ucs_status_t uct_rc_verbs_query_resources(uct_context_h context,
                                                 uct_resource_desc_t **resources_p,
                                                 unsigned *num_resources_p)
{
    /* TODO take transport overhead into account */
    return uct_ib_query_resources(context, 0, resources_p, num_resources_p);
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_ep_t, uct_iface_h tl_iface)
{
    UCS_CLASS_CALL_SUPER_INIT(tl_iface);
    self->tx.available = UCT_RC_TX_QP_LEN; /* TODO */
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rc_verbs_ep_t, uct_rc_ep_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, uct_iface_h);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);


static ucs_status_t uct_rc_verbs_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                              unsigned length,
                                              uint64_t remote_addr,
                                              uct_rkey_t rkey)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    struct ibv_send_wr *bad_wr;
    int UCS_V_UNUSED ret;

    if (ep->tx.available == 0) {
        return UCS_ERR_WOULD_BLOCK;
    }

    iface->inl_rwrite_wr.wr.rdma.remote_addr = remote_addr;
    iface->inl_rwrite_wr.wr.rdma.rkey        = ntohl(rkey);
    iface->inl_sg.addr                       = (uintptr_t)buffer;
    iface->inl_sg.length                     = length;

    ret = ibv_post_send(ep->super.qp, &iface->inl_rwrite_wr, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);

    --ep->tx.available;
    ++iface->super.tx.outstanding;
    return UCS_OK;
}

static void uct_rc_verbs_iface_progress(void *arg)
{
    uct_rc_verbs_iface_t *iface = arg;
    uct_rc_verbs_ep_t *ep;
    struct ibv_wc wc[UCT_IB_MAX_WC];
    int i, ret;

    ret = ibv_poll_cq(iface->super.super.send_cq, UCT_IB_MAX_WC, wc);
    if (ret > 0) {
        for (i = 0; i < ret; ++i) {
            if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
                ucs_fatal("Send completion with error: %s", ibv_wc_status_str(wc[i].status));
            }

            ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, wc[i].qp_num), uct_rc_verbs_ep_t);
            ucs_assert(ep != NULL);
            ++ep->tx.available;
        }
        iface->super.tx.outstanding -= ret;
    } else if (ucs_unlikely(ret < 0)) {
        ucs_fatal("Failed to poll send CQ");
    }
}

static ucs_status_t uct_rc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    uct_rc_iface_query(iface, iface_attr);
    iface_attr->max_short = 50;  /* TODO max_inline */
    return UCS_OK;
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_iface_t)(uct_iface_t*);

uct_iface_ops_t uct_rc_verbs_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_iface_t),
    .iface_get_address   = uct_rc_iface_get_address,
    .iface_flush         = uct_rc_iface_flush,
    .ep_get_address      = uct_rc_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_rc_ep_connect_to_ep,
    .iface_query         = uct_rc_verbs_iface_query,
    .ep_put_short        = uct_rc_verbs_ep_put_short,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_ep_t),
};

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_iface_t, uct_context_h context,
                           const char *dev_name, uct_iface_config_t *tl_config)
{
    UCS_CLASS_CALL_SUPER_INIT(&uct_rc_verbs_iface_ops, context, dev_name);

    /* Initialize inline work request */
    self->inl_rwrite_wr.wr_id               = 0;
    self->inl_rwrite_wr.next                = NULL;
    self->inl_rwrite_wr.sg_list             = &self->inl_sg;
    self->inl_rwrite_wr.num_sge             = 1;
    self->inl_rwrite_wr.opcode              = IBV_WR_RDMA_WRITE;
    self->inl_rwrite_wr.send_flags          = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    self->inl_rwrite_wr.imm_data            = 0;
    self->inl_rwrite_wr.wr.rdma.remote_addr = 0;
    self->inl_rwrite_wr.wr.rdma.rkey        = 0;
    self->inl_sg.addr                       = 0;
    self->inl_sg.length                     = 0;
    self->inl_sg.lkey                       = 0;

    ucs_notifier_chain_add(&context->progress_chain, uct_rc_verbs_iface_progress,
                           self);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_iface_t)
{
    uct_context_h context = self->super.super.super.pd->context;
    ucs_notifier_chain_remove(&context->progress_chain, uct_rc_verbs_iface_progress, self);
}

UCS_CLASS_DEFINE(uct_rc_verbs_iface_t, uct_rc_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_iface_t, uct_iface_t, uct_context_h,
                                 const char*, uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_iface_t, uct_iface_t);

static uct_tl_ops_t uct_rc_verbs_tl_ops = {
    .query_resources     = uct_rc_verbs_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_rc_verbs_iface_t),
    .rkey_unpack         = uct_ib_rkey_unpack,
};

static void uct_rc_verbs_register(uct_context_t *context)
{
    uct_register_tl(context, "rc_verbs", uct_rc_iface_config_table,
                    sizeof(uct_rc_iface_config_t), &uct_rc_verbs_tl_ops);
}

UCS_COMPONENT_DEFINE(uct_context_t, rc_verbs, uct_rc_verbs_register, ucs_empty_function, 0)
