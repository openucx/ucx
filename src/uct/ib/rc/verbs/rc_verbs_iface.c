/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_verbs.h"
#include "rc_verbs_common.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_log.h>
#include <uct/base/uct_pd.h>
#include <ucs/arch/bitops.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <string.h>

static uct_rc_iface_ops_t uct_rc_verbs_iface_ops;

static inline unsigned uct_rc_verbs_iface_post_recv(uct_rc_verbs_iface_t *iface,
                                                    int fill)
{
    unsigned batch = iface->super.super.config.rx_max_batch;
    unsigned count;

    if (iface->super.rx.available < batch) {
        if (!fill) {
            return 0;
        } else {
            count = iface->super.rx.available;
        }
    } else {
        count = batch;
    }

    return uct_rc_verbs_iface_post_recv_always(&iface->super, count);
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_iface_poll_tx(uct_rc_verbs_iface_t *iface)
{
    uct_rc_verbs_ep_t *ep;
    uint16_t count;
    int i;
    unsigned num_wcs = iface->super.super.config.tx_max_poll;
    struct ibv_wc wc[num_wcs];
    ucs_status_t status;

    UCT_RC_VERBS_IFACE_FOREACH_TXWQE(&iface->super, i, wc, num_wcs) {
        count = uct_rc_verbs_txcq_get_comp_count(&wc[i]);
        ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, wc[i].qp_num),
                            uct_rc_verbs_ep_t);
        ucs_assert(ep != NULL);
        uct_rc_verbs_txqp_completed(&ep->super.txqp, &ep->txcnt, count);
        uct_rc_ep_process_tx_completion(&iface->super, &ep->super,
                                        ep->txcnt.ci);
    }
}

void uct_rc_verbs_iface_progress(void *arg)
{
    uct_rc_verbs_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_verbs_iface_poll_rx_common(&iface->super);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_rc_verbs_iface_poll_tx(iface);
    }
}

static ucs_status_t uct_rc_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    uct_rc_iface_query(&iface->super, iface_attr);
    uct_rc_verbs_iface_common_query(&iface->verbs_common, &iface->super, iface_attr);
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_verbs_iface_config_t *config =
                    ucs_derived_of(tl_config, uct_rc_verbs_iface_config_t);
    ucs_status_t status;
    struct ibv_qp_cap cap;
    struct ibv_qp *qp;

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_verbs_iface_ops, pd,
                              worker, dev_name, rx_headroom, 0, &config->super);

    /* Initialize inline work request */
    /* TODO: move to common macro */
    memset(&self->inl_am_wr, 0, sizeof(self->inl_am_wr));
    self->inl_am_wr.sg_list                 = self->verbs_common.inl_sge;
    self->inl_am_wr.num_sge                 = 2;
    self->inl_am_wr.opcode                  = IBV_WR_SEND;
    self->inl_am_wr.send_flags              = IBV_SEND_INLINE;

    memset(&self->inl_rwrite_wr, 0, sizeof(self->inl_rwrite_wr));
    self->inl_rwrite_wr.sg_list             = self->verbs_common.inl_sge;
    self->inl_rwrite_wr.num_sge             = 1;
    self->inl_rwrite_wr.opcode              = IBV_WR_RDMA_WRITE;
    self->inl_rwrite_wr.send_flags          = IBV_SEND_SIGNALED | IBV_SEND_INLINE;

    status = uct_rc_verbs_iface_common_init(&self->verbs_common, 
                                            &self->super, config);
    if (status != UCS_OK) {
        return status;
    }

    /* Create a dummy QP in order to find out max_inline */
    status = uct_rc_iface_qp_create(&self->super, IBV_QPT_RC, &qp, &cap);
    if (status != UCS_OK) {
        goto err;
    }
    ibv_destroy_qp(qp);
    self->verbs_common.config.max_inline = cap.max_inline_data;

    status = uct_rc_verbs_iface_prepost_recvs_common(&self->super);
    if (status != UCS_OK) {
        goto err;
    }

    return UCS_OK;

err:
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_iface_t)
{
    uct_rc_verbs_iface_common_cleanup(&self->verbs_common);
}

UCS_CLASS_DEFINE(uct_rc_verbs_iface_t, uct_rc_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_iface_t, uct_iface_t, uct_pd_h,
                                 uct_worker_h, const char*, size_t,
                                 const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_iface_t, uct_iface_t);


static uct_rc_iface_ops_t uct_rc_verbs_iface_ops = {
    {
    {
    .iface_query              = uct_rc_verbs_iface_query,
    .iface_flush              = uct_rc_iface_flush,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_iface_t),
    .iface_release_am_desc    = uct_ib_iface_release_am_desc,
    .iface_wakeup_open        = uct_ib_iface_wakeup_open,
    .iface_wakeup_get_fd      = uct_ib_iface_wakeup_get_fd,
    .iface_wakeup_arm         = uct_ib_iface_wakeup_arm,
    .iface_wakeup_wait        = uct_ib_iface_wakeup_wait,
    .iface_wakeup_signal      = uct_ib_iface_wakeup_signal,
    .iface_wakeup_close       = uct_ib_iface_wakeup_close,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_get_address           = uct_rc_ep_get_address,
    .ep_connect_to_ep         = uct_rc_ep_connect_to_ep,
    .iface_get_device_address = uct_ib_iface_get_device_address,
    .iface_is_reachable       = uct_ib_iface_is_reachable,
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_verbs_ep_t),
    .ep_am_short              = uct_rc_verbs_ep_am_short,
    .ep_am_bcopy              = uct_rc_verbs_ep_am_bcopy,
    .ep_am_zcopy              = uct_rc_verbs_ep_am_zcopy,
    .ep_put_short             = uct_rc_verbs_ep_put_short,
    .ep_put_bcopy             = uct_rc_verbs_ep_put_bcopy,
    .ep_put_zcopy             = uct_rc_verbs_ep_put_zcopy,
    .ep_get_bcopy             = uct_rc_verbs_ep_get_bcopy,
    .ep_get_zcopy             = uct_rc_verbs_ep_get_zcopy,
    .ep_atomic_add64          = uct_rc_verbs_ep_atomic_add64,
    .ep_atomic_fadd64         = uct_rc_verbs_ep_atomic_fadd64,
    .ep_atomic_swap64         = uct_rc_verbs_ep_atomic_swap64,
    .ep_atomic_cswap64        = uct_rc_verbs_ep_atomic_cswap64,
    .ep_atomic_add32          = uct_rc_verbs_ep_atomic_add32,
    .ep_atomic_fadd32         = uct_rc_verbs_ep_atomic_fadd32,
    .ep_atomic_swap32         = uct_rc_verbs_ep_atomic_swap32,
    .ep_atomic_cswap32        = uct_rc_verbs_ep_atomic_cswap32,
    .ep_pending_add           = uct_rc_ep_pending_add,
    .ep_pending_purge         = uct_rc_ep_pending_purge,
    .ep_flush                 = uct_rc_verbs_ep_flush
    },
    .arm_tx_cq                = uct_ib_iface_arm_tx_cq,
    .arm_rx_cq                = uct_ib_iface_arm_rx_cq,
    },
    .fc_ctrl                  = uct_rc_verbs_ep_fc_ctrl
};

static ucs_status_t uct_rc_verbs_query_resources(uct_pd_h pd,
                                                 uct_tl_resource_desc_t **resources_p,
                                                 unsigned *num_resources_p)
{
    return uct_ib_device_query_tl_resources(&ucs_derived_of(pd, uct_ib_pd_t)->dev,
                                            "rc", 0,
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_rc_verbs_tl,
                        uct_rc_verbs_query_resources,
                        uct_rc_verbs_iface_t,
                        "rc",
                        "RC_VERBS_",
                        uct_rc_verbs_iface_config_table,
                        uct_rc_verbs_iface_config_t);
UCT_PD_REGISTER_TL(&uct_ib_pdc, &uct_rc_verbs_tl);
