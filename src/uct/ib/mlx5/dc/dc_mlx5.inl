/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "dc_mlx5.h"
#include "dc_mlx5_ep.h"


#include <uct/ib/mlx5/rc/rc_mlx5.inl>
#include "uct/ib/rc/base/rc_iface.h"
#include "uct/ib/rc/base/rc_ep.h"


static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_update_tx_res(uct_dc_mlx5_iface_t *iface, uct_ib_mlx5_txwq_t *txwq,
                          uct_rc_txqp_t *txqp, uint16_t hw_ci)
{
    uct_rc_txqp_available_set(txqp, uct_ib_mlx5_txwq_update_bb(txwq, hw_ci));
    ucs_assert(uct_rc_txqp_available(txqp) <= txwq->bb_max);

    uct_rc_iface_update_reads(&iface->super.super);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_get_arbiter_params(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                               ucs_arbiter_t **waitq_p,
                               ucs_arbiter_group_t **group_p,
                               uint8_t *pool_index_p)
{
    *pool_index_p = uct_dc_mlx5_ep_pool_index(ep);

    if (ep->dci != UCT_DC_MLX5_EP_NO_DCI) {
        *waitq_p = uct_dc_mlx5_iface_tx_waitq(iface);
        *group_p = uct_dc_mlx5_ep_arb_group(iface, ep);
    } else {
        *waitq_p = uct_dc_mlx5_iface_dci_waitq(iface, *pool_index_p);
        *group_p = &ep->arb_group;
    }
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_ep_push_pending_req(ucs_arbiter_group_t *group,
                                uct_pending_req_t *r, int push_to_head)
{
    if (push_to_head) {
        uct_pending_req_arb_group_push_head(group, r);
    } else {
        uct_pending_req_arb_group_push(group, r);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_ep_pending_common_shared(uct_dc_mlx5_iface_t *iface,
                                     uct_dc_mlx5_ep_t *ep, uct_pending_req_t *r,
                                     unsigned flags, int push_to_head,
                                     int schedule)
{
    ucs_arbiter_group_t *group = uct_dc_mlx5_ep_shared_arb_group(iface, ep);

    UCS_STATIC_ASSERT(sizeof(uct_dc_mlx5_pending_req_priv) <=
                      UCT_PENDING_REQ_PRIV_LEN);
    uct_dc_mlx5_pending_req_priv(r)->ep = ep;
    uct_dc_mlx5_ep_push_pending_req(group, r, push_to_head);

    if (schedule) {
        ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface), group);
    }
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_ep_pending_common(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep,
                              uct_pending_req_t *r, unsigned flags,
                              int push_to_head, int schedule)
{
    ucs_arbiter_group_t *group;

    UCT_TL_EP_STAT_PEND(&ep->super);

    if (uct_dc_mlx5_iface_is_policy_shared(iface)) {
        uct_dc_mlx5_ep_pending_common_shared(iface, ep, r, flags, push_to_head,
                                             schedule);
        return;
    }

    group = &ep->arb_group;

    uct_dc_mlx5_ep_push_pending_req(group, r, push_to_head);

    if (!schedule) {
        return;
    }

    if (ep->dci == UCT_DC_MLX5_EP_NO_DCI) {
        /* no dci:
         * Do not grab dci here. Instead put the group on dci allocation
         * arbiter. This way we can assure fairness between all eps waiting for
         * dci allocation. Relevant for dcs, dcs_quota and dcs_hybrid policies.
         */
        uct_dc_mlx5_iface_schedule_dci_alloc(iface, ep);
    } else if (uct_dc_mlx5_iface_dci_has_tx_resources(iface, ep->dci)) {
        ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface), group);
    }
}
