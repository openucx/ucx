/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2016-2020. ALL RIGHTS RESERVED.

* See file LICENSE for terms.
*/

#ifndef UCT_DC_EP_H
#define UCT_DC_EP_H

#include <uct/api/uct.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/sys/compiler_def.h>

#include "dc_mlx5.h"

#define UCT_DC_MLX5_EP_NO_DCI    ((uint8_t)-1)
#define UCT_DC_MLX5_HW_DCI_INDEX 0


#define UCT_DC_MLX5_TXQP_DECL(_txqp, _txwq) \
    uct_rc_txqp_t UCS_V_UNUSED *_txqp; \
    uct_ib_mlx5_txwq_t UCS_V_UNUSED *_txwq;


#define UCT_DC_MLX5_IFACE_TXQP_GET(_iface, _ep, _txqp, _txwq) \
    UCT_DC_MLX5_IFACE_TXQP_DCI_GET(_iface, (_ep)->dci, _txqp, _txwq)


enum uct_dc_mlx5_ep_flags {
    /* DCI pool EP assigned to according to it's lag port */
    UCT_DC_MLX5_EP_FLAG_POOL_INDEX_MASK     = UCS_MASK(5),

    /* EP is in the tx_wait state. See description of the dcs+quota dci
       selection policy above */
    UCT_DC_MLX5_EP_FLAG_TX_WAIT             = UCS_BIT(5),

    /* EP has GRH address. Used by dc_mlx5 endpoint */
    UCT_DC_MLX5_EP_FLAG_GRH                 = UCS_BIT(6),

    /* Flush cancel was executed on EP */
    UCT_DC_MLX5_EP_FLAG_FLUSH_CANCEL        = UCS_BIT(7),

    /* Error handler already called or flush(CANCEL) disabled it */
    UCT_DC_MLX5_EP_FLAG_ERR_HANDLER_INVOKED = UCS_BIT(8),

    /* EP supports flush remote operation */
    UCT_DC_MLX5_EP_FLAG_FLUSH_RKEY          = UCS_BIT(9),

    /* Flush remote operation should be invoked */
    UCT_DC_MLX5_EP_FLAG_FLUSH_REMOTE        = UCS_BIT(10),

#if UCS_ENABLE_ASSERT
    /* EP was invalidated without DCI */
    UCT_DC_MLX5_EP_FLAG_INVALIDATED         = UCS_BIT(11)
#else
    UCT_DC_MLX5_EP_FLAG_INVALIDATED         = 0
#endif
};

/* Address-vector for link-local scope */
typedef struct uct_dc_mlx5_base_av {
    uint32_t              dqp_dct;
    uint16_t              rlid;
} UCS_S_PACKED uct_dc_mlx5_base_av_t;

struct uct_dc_mlx5_ep {
    uct_base_ep_t         super;
    ucs_arbiter_group_t   arb_group;
    uint8_t               dci;
    uint8_t               atomic_mr_id;
    uint16_t              flags;
    uint16_t              flush_rkey_hi;
    uct_rc_fc_t           fc;
    uct_dc_mlx5_base_av_t av;
    uint8_t               dci_channel_index;
};

typedef struct {
    uct_dc_mlx5_ep_t                    super;
    struct mlx5_grh_av                  grh_av;
} uct_dc_mlx5_grh_ep_t;

typedef struct {
    uct_pending_req_priv_arb_t arb;
    uct_dc_mlx5_ep_t           *ep;
} uct_dc_mlx5_pending_req_priv_t;


UCS_CLASS_DECLARE(uct_dc_mlx5_ep_t, uct_dc_mlx5_iface_t*,
                  const uct_dc_mlx5_iface_addr_t*, uct_ib_mlx5_base_av_t*,
                  uint8_t, const uct_dc_mlx5_dci_config_t*);

UCS_CLASS_DECLARE(uct_dc_mlx5_grh_ep_t, uct_dc_mlx5_iface_t*,
                  const uct_dc_mlx5_iface_addr_t*, uct_ib_mlx5_base_av_t*,
                  uint8_t, struct mlx5_grh_av*,
                  const uct_dc_mlx5_dci_config_t*);

UCS_CLASS_DECLARE_DELETE_FUNC(uct_dc_mlx5_ep_t, uct_ep_t);

ucs_status_t uct_dc_mlx5_ep_put_short(uct_ep_h tl_ep, const void *payload,
                                      unsigned length, uint64_t remote_addr,
                                      uct_rkey_t rkey);

ssize_t uct_dc_mlx5_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                 void *arg, uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_dc_mlx5_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_get_bcopy(uct_ep_h tl_ep,
                                      uct_unpack_callback_t unpack_cb,
                                      void *arg, size_t length,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                     const void *buffer, unsigned length);

ucs_status_t uct_dc_mlx5_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                         const uct_iov_t *iov, size_t iovcnt);

ssize_t uct_dc_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                uct_pack_callback_t pack_cb, void *arg,
                                unsigned flags);

ucs_status_t uct_dc_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                     unsigned header_length, const uct_iov_t *iov,
                                     size_t iovcnt, unsigned flags,
                                     uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_atomic64_post(uct_ep_h ep, unsigned opcode, uint64_t value,
                                          uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_dc_mlx5_ep_atomic32_post(uct_ep_h ep, unsigned opcode, uint32_t value,
                                          uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_dc_mlx5_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                           uint64_t value, uint64_t *result,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                           uint32_t value, uint32_t *result,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp);

#if IBV_HW_TM
ucs_status_t uct_dc_mlx5_ep_tag_eager_short(uct_ep_h tl_ep, uct_tag_t tag,
                                            const void *data, size_t length);

ssize_t uct_dc_mlx5_ep_tag_eager_bcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                       uint64_t imm,
                                       uct_pack_callback_t pack_cb,
                                       void *arg, unsigned flags);

ucs_status_t uct_dc_mlx5_ep_tag_eager_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                            uint64_t imm, const uct_iov_t *iov,
                                            size_t iovcnt, unsigned flags,
                                            uct_completion_t *comp);

ucs_status_ptr_t uct_dc_mlx5_ep_tag_rndv_zcopy(uct_ep_h tl_ep, uct_tag_t tag,
                                               const void *header,
                                               unsigned header_length,
                                               const uct_iov_t *iov,
                                               size_t iovcnt, unsigned flags,
                                               uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_tag_rndv_request(uct_ep_h tl_ep, uct_tag_t tag,
                                             const void* header,
                                             unsigned header_length,
                                             unsigned flags);

ucs_status_t uct_dc_mlx5_iface_tag_recv_zcopy(uct_iface_h tl_iface,
                                              uct_tag_t tag,
                                              uct_tag_t tag_mask,
                                              const uct_iov_t *iov,
                                              size_t iovcnt,
                                              uct_tag_context_t *ctx);

ucs_status_t uct_dc_mlx5_iface_tag_recv_cancel(uct_iface_h tl_iface,
                                               uct_tag_context_t *ctx,
                                               int force);
#endif

ucs_status_t uct_dc_mlx5_ep_fence(uct_ep_h tl_ep, unsigned flags);

ucs_status_t uct_dc_mlx5_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp);

ucs_status_t uct_dc_mlx5_ep_qp_to_err(uct_dc_mlx5_ep_t *ep);

ucs_status_t uct_dc_mlx5_ep_invalidate(uct_ep_h tl_ep, unsigned flags);

ucs_status_t uct_dc_mlx5_ep_fc_pure_grant_send(uct_dc_mlx5_ep_t *ep,
                                               uct_rc_iface_send_op_t *send_op);

unsigned uct_dc_mlx5_ep_dci_release_progress(void *arg);

void
uct_dc_mlx5_ep_fc_pure_grant_send_completion(uct_rc_iface_send_op_t *send_op,
                                             const void *resp);

ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_pending_wait(ucs_arbiter_t *arbiter,
                                      ucs_arbiter_group_t *group,
                                      ucs_arbiter_elem_t *elem,
                                      void *arg);

ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_dcs_pending_tx(ucs_arbiter_t *arbiter,
                                        ucs_arbiter_group_t *group,
                                        ucs_arbiter_elem_t *elem,
                                        void *arg);

ucs_arbiter_cb_result_t
uct_dc_mlx5_iface_dci_do_rand_pending_tx(ucs_arbiter_t *arbiter,
                                         ucs_arbiter_group_t *group,
                                         ucs_arbiter_elem_t *elem,
                                         void *arg);

ucs_status_t uct_dc_mlx5_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *r,
                                        unsigned flags);

void uct_dc_mlx5_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb, void *arg);

void uct_dc_mlx5_ep_do_pending_fc(uct_dc_mlx5_ep_t *fc_ep,
                                  uct_dc_fc_request_t *fc_req);

static UCS_F_ALWAYS_INLINE uint8_t
uct_dc_mlx5_ep_pool_index(uct_dc_mlx5_ep_t *ep)
{
    return ep->flags & UCT_DC_MLX5_EP_FLAG_POOL_INDEX_MASK;
}

static UCS_F_ALWAYS_INLINE uct_dc_mlx5_pending_req_priv_t *
uct_dc_mlx5_pending_req_priv(uct_pending_req_t *req)
{
    return (uct_dc_mlx5_pending_req_priv_t *)&(req)->priv;
}

static UCS_F_ALWAYS_INLINE int uct_dc_mlx5_iface_is_dci_rand(uct_dc_mlx5_iface_t *iface)
{
    return iface->tx.policy == UCT_DC_TX_POLICY_RAND;
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_is_hw_dcs(const uct_dc_mlx5_iface_t *iface)
{
    return iface->tx.policy == UCT_DC_TX_POLICY_HW_DCS;
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_is_policy_shared(const uct_dc_mlx5_iface_t *iface)
{
    return iface->tx.policy >= UCT_DC_TX_POLICY_SHARED_FIRST;
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_is_dcs_quota_or_hybrid(const uct_dc_mlx5_iface_t *iface)
{
    return UCS_BIT(iface->tx.policy) & (UCS_BIT(UCT_DC_TX_POLICY_DCS_QUOTA) |
                                        UCS_BIT(UCT_DC_TX_POLICY_DCS_HYBRID));
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_is_hybrid(const uct_dc_mlx5_iface_t *iface)
{
    return iface->tx.policy == UCT_DC_TX_POLICY_DCS_HYBRID;
}

static UCS_F_ALWAYS_INLINE ucs_arbiter_group_t*
uct_dc_mlx5_ep_rand_arb_group(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    ucs_assert(uct_dc_mlx5_iface_is_policy_shared(iface) &&
               (ep->dci != UCT_DC_MLX5_EP_NO_DCI));
    /* If DCI random policy is used, DCI is always assigned to EP */
    return &uct_dc_mlx5_iface_dci(iface, ep->dci)->arb_group;
}

static UCS_F_ALWAYS_INLINE ucs_arbiter_group_t*
uct_dc_mlx5_ep_arb_group(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    return (uct_dc_mlx5_iface_is_policy_shared(iface)) ?
            uct_dc_mlx5_ep_rand_arb_group(iface, ep) : &ep->arb_group;
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_dci_sched_tx(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    if (uct_dc_mlx5_iface_is_policy_shared(iface)) {
        ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface),
                                   uct_dc_mlx5_ep_rand_arb_group(iface, ep));
    } else if (uct_dc_mlx5_iface_dci_has_tx_resources(iface, ep->dci)) {
        ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface),
                                   &ep->arb_group);
    }
}

static UCS_F_ALWAYS_INLINE uct_dc_mlx5_ep_t *
uct_dc_mlx5_ep_from_dci(uct_dc_mlx5_iface_t *iface, uint8_t dci_index)
{
    /* Can be used with dcs* policies only, with rand policy every dci may
     * be used by many eps */
    ucs_assert(!uct_dc_mlx5_iface_is_policy_shared(iface));
    return uct_dc_mlx5_iface_dci(iface, dci_index)->ep;
}

void uct_dc_mlx5_ep_handle_failure(uct_dc_mlx5_ep_t *ep,
                                   struct mlx5_cqe64 *cqe,
                                   ucs_status_t status);

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_init_dci_config(uct_dc_mlx5_dci_config_t *dci_config,
                            uint8_t path_index, uint8_t max_rd_atomic)
{
    dci_config->path_index    = path_index % UCT_DC_MLX5_IFACE_MAX_DCI_POOLS;
    dci_config->max_rd_atomic = max_rd_atomic;
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_is_hw_dci(const uct_dc_mlx5_iface_t *iface, uint8_t dci)
{
    return dci == iface->tx.hybrid_hw_dci;
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_is_dci_shared(uct_dc_mlx5_iface_t *iface, uint8_t dci)
{
    return uct_dc_mlx5_iface_dci(iface, dci)->flags & UCT_DC_DCI_FLAG_SHARED;
}

ucs_status_t static UCS_F_ALWAYS_INLINE
uct_dc_mlx5_dci_pool_init_dci(uct_dc_mlx5_iface_t *iface, uint8_t pool_index,
                              uint8_t dci_index)
{
    uct_dc_mlx5_dci_pool_t *pool = &iface->tx.dci_pool[pool_index];
    uct_dc_dci_t *dci            = uct_dc_mlx5_iface_dci(iface, dci_index);
    uint8_t num_channels         = 1;
    ucs_status_t status;

    ucs_assertv(ucs_array_length(&pool->stack) < iface->tx.ndci,
                "stack length exceeded ndci");

    if (uct_dc_mlx5_iface_is_hw_dcs(iface) ||
        uct_dc_mlx5_is_hw_dci(iface, dci_index)) {
        num_channels = iface->tx.num_dci_channels;
    }

    status = uct_dc_mlx5_iface_create_dci(iface, dci_index, 1, num_channels);
    if (status != UCS_OK) {
        ucs_error("iface %p: failed to create dci %u at pool %u", iface,
                  dci_index, pool_index);
        return status;
    }

    dci->path_index = pool->config.path_index;
    dci->pool_index = pool_index;

    if (uct_dc_mlx5_iface_is_policy_shared(iface) ||
        uct_dc_mlx5_is_hw_dci(iface, dci_index)) {
        dci->flags |= UCT_DC_DCI_FLAG_SHARED;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_mlx5_ep_basic_init(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    size_t dcis_array_size;
    uct_dc_dci_t *dci;

    ucs_arbiter_group_init(&ep->arb_group);

    if ((uct_dc_mlx5_iface_is_hw_dcs(iface) ||
         uct_dc_mlx5_iface_is_hybrid(iface)) &&
        ucs_array_is_empty(&iface->tx.dcis)) {
        uct_dc_mlx5_iface_resize_and_fill_dcis(iface, 1);
        uct_dc_mlx5_dci_pool_init_dci(iface, uct_dc_mlx5_ep_pool_index(ep),
                                      UCT_DC_MLX5_HW_DCI_INDEX);
    }

    if (uct_dc_mlx5_iface_is_dci_rand(iface)) {
        /* coverity[dont_call] */
        ep->dci               = rand_r(&iface->tx.rand_seed) % iface->tx.ndci;
        ep->dci_channel_index = 0;
        dcis_array_size       = ucs_max(ep->dci + 1,
                                        ucs_array_length(&iface->tx.dcis));
        uct_dc_mlx5_iface_resize_and_fill_dcis(iface, dcis_array_size);

        if (!uct_dc_mlx5_is_dci_valid(uct_dc_mlx5_iface_dci(iface, ep->dci))) {
            uct_dc_mlx5_dci_pool_init_dci(iface, uct_dc_mlx5_ep_pool_index(ep),
                                          ep->dci);
        }
    } else if (uct_dc_mlx5_iface_is_hw_dcs(iface)) {
        ep->dci               = UCT_DC_MLX5_HW_DCI_INDEX;
        dci                   = uct_dc_mlx5_iface_dci(iface, ep->dci);
        ep->dci_channel_index = dci->next_channel_index++;
    } else {
        /* Hybrid or software dcs */
        ep->dci               = UCT_DC_MLX5_EP_NO_DCI;
        ep->dci_channel_index = 0;
    }

    return uct_rc_fc_init(&ep->fc,
                          &iface->super.super UCS_STATS_ARG(ep->super.stats));
}

static int
uct_dc_mlx5_iface_dci_can_alloc(uct_dc_mlx5_iface_t *iface, uint8_t pool_index)
{
    uct_dc_mlx5_dci_pool_t *pool = &iface->tx.dci_pool[pool_index];

    return ucs_likely(pool->stack_top < iface->tx.ndci);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_dcis_array_copy(void *dst, void *src, size_t length)
{
    uct_dc_dci_t *src_dcis = (uct_dc_dci_t*)src;
    uct_dc_dci_t *dst_dcis = (uct_dc_dci_t*)dst;
    size_t i;

    memcpy(dst_dcis, src_dcis, sizeof(uct_dc_dci_t) * length);

    /* txqp is a queue and need to splice tail */
    for (i = 0; i < length; ++i) {
        if (uct_dc_mlx5_is_dci_valid(&src_dcis[i])) {
            ucs_queue_head_init(&dst_dcis[i].txqp.outstanding);
            ucs_queue_splice(&dst_dcis[i].txqp.outstanding,
                             &src_dcis[i].txqp.outstanding);
        }
    }
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_dci_can_alloc_or_create(uct_dc_mlx5_iface_t *iface,
                                          uint8_t pool_index)
{
    uct_dc_mlx5_dci_pool_t *pool = &iface->tx.dci_pool[pool_index];
    uint8_t dci_index;
    ucs_status_t status;

    ucs_assert(!uct_dc_mlx5_iface_is_policy_shared(iface));

    if (ucs_likely(pool->stack_top < ucs_array_length(&pool->stack))) {
        return 1;
    }

    dci_index = ucs_array_length(&iface->tx.dcis);
    if (ucs_array_length(&pool->stack) >= iface->tx.ndci) {
        return 0;
    }

    /* Append a new dci */
    status = uct_dc_mlx5_iface_resize_and_fill_dcis(iface, dci_index + 1);
    if (status != UCS_OK) {
        return 0;
    }

    status = uct_dc_mlx5_dci_pool_init_dci(iface, pool_index, dci_index);
    if (status != UCS_OK) {
        return 0;
    }

    *ucs_array_append(&pool->stack, return UCS_ERR_NO_MEMORY) = dci_index;
    return 1;
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_check_tx(uct_dc_mlx5_iface_t *iface)
{
#if UCS_ENABLE_ASSERT
    uint8_t pool_index;

    for (pool_index = 0; pool_index < iface->tx.num_dci_pools; ++pool_index) {
        if (uct_dc_mlx5_iface_dci_can_alloc(iface, pool_index)) {
            ucs_assertv(ucs_arbiter_is_empty(
                                uct_dc_mlx5_iface_dci_waitq(iface, pool_index)),
                        "dc_iface %p pool %d: can allocate dci, but pending is "
                        "not empty",
                        iface, pool_index);
        }
    }
#endif
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_progress_pending(uct_dc_mlx5_iface_t *iface,
                                   uint8_t pool_index)
{
    ucs_arbiter_t *dci_waitq = uct_dc_mlx5_iface_dci_waitq(iface, pool_index);

    do {
        /**
         * Pending op on the tx_waitq can complete with the UCS_OK
         * status without actually sending anything on the dci.
         * In this case pending ops on the waitq may never be
         * scheduled.
         *
         * So we keep progressing pending while dci_waitq is not
         * empty and it is possible to allocate a dci.
         * NOTE: in case of rand dci allocation policy, dci_waitq is always
         * empty.
         */
        if (!uct_dc_mlx5_iface_is_policy_shared(iface) &&
            uct_dc_mlx5_iface_dci_can_alloc(iface, pool_index)) {
            ucs_arbiter_dispatch(dci_waitq, 1,
                                 uct_dc_mlx5_iface_dci_do_pending_wait, NULL);
        }
        ucs_arbiter_dispatch(uct_dc_mlx5_iface_tx_waitq(iface), 1,
                             iface->tx.pend_cb, NULL);

    } while (ucs_unlikely(!ucs_arbiter_is_empty(dci_waitq) &&
                          uct_dc_mlx5_iface_dci_can_alloc(iface, pool_index)));
}

static inline int uct_dc_mlx5_iface_dci_ep_can_send(uct_dc_mlx5_ep_t *ep)
{
    uct_dc_mlx5_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_dc_mlx5_iface_t);
    return (!(ep->flags & UCT_DC_MLX5_EP_FLAG_TX_WAIT)) &&
           uct_rc_fc_has_resources(&iface->super.super, &ep->fc) &&
           uct_dc_mlx5_iface_dci_has_tx_resources(iface, ep->dci);
}

static UCS_F_ALWAYS_INLINE
void uct_dc_mlx5_iface_schedule_dci_alloc(uct_dc_mlx5_iface_t *iface,
                                          uct_dc_mlx5_ep_t *ep)
{
    ucs_arbiter_t *waitq;

    /* If FC window is empty the group will be scheduled when grant is received */
    if (uct_rc_fc_has_resources(&iface->super.super, &ep->fc)) {
        waitq = uct_dc_mlx5_iface_dci_waitq(iface,
                                            uct_dc_mlx5_ep_pool_index(ep));
        ucs_arbiter_group_schedule(waitq, &ep->arb_group);
    }
}

static UCS_F_ALWAYS_INLINE uint8_t
uct_dc_mlx5_iface_dci_pool_index(uct_dc_mlx5_iface_t *iface, uint8_t dci_index)
{
    return uct_dc_mlx5_iface_dci(iface, dci_index)->pool_index;
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_dci_release(uct_dc_mlx5_iface_t *iface, uint8_t dci_index)
{
    uint8_t pool_index           = uct_dc_mlx5_iface_dci_pool_index(iface,
                                                                    dci_index);
    uct_dc_mlx5_dci_pool_t *pool = &iface->tx.dci_pool[pool_index];

    ucs_trace_data("iface %p: release dci %d from ep %p", iface, dci_index,
                   uct_dc_mlx5_ep_from_dci(iface, dci_index));

    uct_dc_mlx5_iface_dci(iface, dci_index)->ep = NULL;
    pool->stack_top--;
    ucs_assertv(pool->stack_top >= 0, "dci pool underflow, stack_top=%d",
                (int)pool->stack_top);
    ucs_assert(pool->release_stack_top < pool->stack_top);
    ucs_array_elem(&pool->stack, pool->stack_top) = dci_index;
}

/* Release endpoint's DCI below, if the endpoint does not have outstanding
 * operations */
static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_dci_put(uct_dc_mlx5_iface_t *iface, uint8_t dci_index)
{
    uct_dc_mlx5_ep_t *ep;
    ucs_arbiter_t *waitq;
    uint8_t pool_index;

    ucs_assert(dci_index != UCT_DC_MLX5_EP_NO_DCI);

    if (uct_dc_mlx5_iface_is_policy_shared(iface)) {
        return;
    }

    ep = uct_dc_mlx5_ep_from_dci(iface, dci_index);

    if (ucs_unlikely(ep == NULL)) {
        if (!uct_dc_mlx5_iface_dci_has_outstanding(iface, dci_index) &&
            !uct_dc_mlx5_is_hw_dci(iface, dci_index)) {
            uct_dc_mlx5_iface_dci_release(iface, dci_index);
        }
        return;
    }

    pool_index = uct_dc_mlx5_ep_pool_index(ep);
    ucs_assert(iface->tx.dci_pool[pool_index].stack_top > 0);

    if (uct_dc_mlx5_iface_dci_has_outstanding(iface, dci_index)) {
        if (uct_dc_mlx5_iface_is_dcs_quota_or_hybrid(iface)) {
            /* in tx_wait state:
             * -  if there are no eps are waiting for dci allocation
             *    ep goes back to normal state
             */
            if (ep->flags & UCT_DC_MLX5_EP_FLAG_TX_WAIT) {
                waitq = uct_dc_mlx5_iface_dci_waitq(iface, pool_index);
                if (!ucs_arbiter_is_empty(waitq)) {
                    return;
                }
                ep->flags &= ~UCT_DC_MLX5_EP_FLAG_TX_WAIT;
            }
        }
        ucs_arbiter_group_schedule(uct_dc_mlx5_iface_tx_waitq(iface), &ep->arb_group);
        return;
    }

    uct_dc_mlx5_iface_dci_release(iface, dci_index);

    ucs_assert(ep->dci != UCT_DC_MLX5_EP_NO_DCI);
    ep->dci    = UCT_DC_MLX5_EP_NO_DCI;
    ep->flags &= ~UCT_DC_MLX5_EP_FLAG_TX_WAIT;

    uct_dc_mlx5_iface_dci(iface, dci_index)->ep = NULL;

    /* it is possible that dci is released while ep still has scheduled pending ops.
     * move the group to the 'wait for dci alloc' state
     */
    ucs_arbiter_group_desched(uct_dc_mlx5_iface_tx_waitq(iface), &ep->arb_group);
    uct_dc_mlx5_iface_schedule_dci_alloc(iface, ep);
}

static inline void
uct_dc_mlx5_iface_dci_alloc(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    /* take a first available dci from stack.
     * There is no need to check txqp because
     * dci must have resources to transmit.
     */
    uint8_t pool_index           = uct_dc_mlx5_ep_pool_index(ep);
    uct_dc_mlx5_dci_pool_t *pool = &iface->tx.dci_pool[pool_index];

    ucs_assert(!uct_dc_mlx5_iface_is_policy_shared(iface));
    ucs_assert(pool->release_stack_top < pool->stack_top);
    ucs_assertv(pool->stack_top < ucs_array_length(&pool->stack),
                "stack_top=%u, array_length(stack)=%u", pool->stack_top,
                ucs_array_length(&pool->stack));
    ep->dci = ucs_array_elem(&pool->stack, pool->stack_top);
    ucs_assert(!uct_dc_mlx5_is_hw_dci(iface, ep->dci));
    ucs_assert(uct_dc_mlx5_ep_from_dci(iface, ep->dci) == NULL);
    uct_dc_mlx5_iface_dci(iface, ep->dci)->ep = ep;
    pool->stack_top++;
    if (ep->flags & UCT_DC_MLX5_EP_FLAG_INVALIDATED) {
        (void)uct_dc_mlx5_ep_qp_to_err(ep);
    }

    ucs_assertv(pool->stack_top > 0, "dci pool overflow, stack_top=%d",
                (int)pool->stack_top);
    ucs_trace_data("iface %p: allocate dci %d for ep %p", iface, ep->dci, ep);
}

static UCS_F_ALWAYS_INLINE void
uct_dc_mlx5_iface_dci_schedule_release(uct_dc_mlx5_iface_t *iface, uint8_t dci)
{
    uct_worker_h worker = &iface->super.super.super.super.worker->super;
    uint8_t pool_index = uct_dc_mlx5_iface_dci_pool_index(iface, dci);
    uint8_t stack_top;

    ucs_assert(!uct_dc_mlx5_iface_is_policy_shared(iface));

    /* adding current DCI into release stack and mark pool for
     * processing, see details in @ref uct_dc_mlx5_dci_pool_t description */
    stack_top = ++iface->tx.dci_pool[pool_index].release_stack_top;
    ucs_assert(stack_top < iface->tx.dci_pool[pool_index].stack_top);

    iface->tx.dci_pool_release_bitmap |= UCS_BIT(pool_index);
    ucs_array_elem(&iface->tx.dci_pool[pool_index].stack, stack_top) = dci;
    ucs_callbackq_add_oneshot(&worker->progress_q, iface,
                              uct_dc_mlx5_ep_dci_release_progress, iface);
}

static UCS_F_ALWAYS_INLINE int
uct_dc_mlx5_iface_dci_detach(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    uint8_t dci_index = ep->dci;

    ucs_assert(!uct_dc_mlx5_iface_is_policy_shared(iface));
    ucs_assert(dci_index != UCT_DC_MLX5_EP_NO_DCI);
    ucs_assert(iface->tx.dci_pool[uct_dc_mlx5_ep_pool_index(ep)].stack_top > 0);

    if (uct_dc_mlx5_iface_dci_has_outstanding(iface, dci_index) ||
        uct_dc_mlx5_is_hw_dci(iface, dci_index)) {
        return 0;
    }

    ep->dci    = UCT_DC_MLX5_EP_NO_DCI;
    ep->flags &= ~UCT_DC_MLX5_EP_FLAG_TX_WAIT;

    uct_dc_mlx5_iface_dci_schedule_release(iface, dci_index);

    return 1;
}

int uct_dc_mlx5_ep_is_connected(const uct_ep_h tl_ep,
                                const uct_ep_is_connected_params_t *params);

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_mlx5_set_ep_to_hw_dcs(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    if (!uct_dc_mlx5_iface_is_hybrid(iface) ||
        !uct_dc_mlx5_iface_dci_has_tx_resources(iface,
                                                UCT_DC_MLX5_HW_DCI_INDEX)) {
        UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
        return UCS_ERR_NO_RESOURCE;
    }

    ep->dci = UCT_DC_MLX5_HW_DCI_INDEX;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_dc_mlx5_iface_dci_get(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep)
{
    uint8_t pool_index = uct_dc_mlx5_ep_pool_index(ep);
    uct_dc_dci_t *dci;
    ucs_arbiter_t *waitq;
    uct_rc_txqp_t *txqp;
    int16_t available;

    ucs_assert(!iface->super.super.config.tx_moderation);

    if (ep->dci == UCT_DC_MLX5_EP_NO_DCI) {
        goto try_alloc;
    }

    dci  = uct_dc_mlx5_iface_dci(iface, ep->dci);

    if (uct_dc_mlx5_is_dci_shared(iface, ep->dci)) {
        if (uct_dc_mlx5_iface_dci_has_tx_resources(iface, ep->dci)) {
            return UCS_OK;
        } else {
            UCS_STATS_UPDATE_COUNTER(dci->txqp.stats, UCT_RC_TXQP_STAT_QP_FULL,
                                     1);
            goto out_no_res;
        }
    }

    /* dci is already assigned - keep using it */
    if (uct_dc_mlx5_iface_is_dcs_quota_or_hybrid(iface) &&
        (ep->flags & UCT_DC_MLX5_EP_FLAG_TX_WAIT)) {
        goto out_no_res;
    }

    /* if dci has sent more than quota, and there are eps waiting for dci
         * allocation ep goes into tx_wait state.
         */
    txqp      = &dci->txqp;
    available = uct_rc_txqp_available(txqp);
    waitq     = uct_dc_mlx5_iface_dci_waitq(iface, pool_index);
    if (uct_dc_mlx5_iface_is_dcs_quota_or_hybrid(iface) &&
        (available <= iface->tx.available_quota) &&
        !ucs_arbiter_is_empty(waitq)) {
        ep->flags |= UCT_DC_MLX5_EP_FLAG_TX_WAIT;
        goto out_no_res;
    }

    if (available <= 0) {
        UCS_STATS_UPDATE_COUNTER(txqp->stats, UCT_RC_TXQP_STAT_QP_FULL, 1);
        goto out_no_res;
    }

    return UCS_OK;

try_alloc:
    if (uct_dc_mlx5_iface_dci_can_alloc_or_create(iface, pool_index)) {
        waitq = uct_dc_mlx5_iface_dci_waitq(iface, pool_index);
        ucs_assert(ucs_arbiter_is_empty(waitq));

        uct_dc_mlx5_iface_dci_alloc(iface, ep);
        return UCS_OK;
    } 

    return uct_dc_mlx5_set_ep_to_hw_dcs(iface, ep);

out_no_res:
    /* we will have to wait until someone releases dci */
    UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
    return UCS_ERR_NO_RESOURCE;
}

ucs_status_t uct_dc_mlx5_ep_check_fc(uct_dc_mlx5_iface_t *iface, uct_dc_mlx5_ep_t *ep);

static inline struct mlx5_grh_av *uct_dc_mlx5_ep_get_grh(uct_dc_mlx5_ep_t *ep)
{
   return (ep->flags & UCT_DC_MLX5_EP_FLAG_GRH) ?
          &(ucs_derived_of(ep, uct_dc_mlx5_grh_ep_t)->grh_av) : NULL;
}


#define UCT_DC_MLX5_CHECK_DCI_RES(_iface, _ep) \
    { \
        ucs_status_t _status = uct_dc_mlx5_iface_dci_get(_iface, _ep); \
        if (ucs_unlikely(_status != UCS_OK)) { \
            return _status; \
        } \
    }


#define UCT_DC_CHECK_RES_PTR(_iface, _ep) \
    { \
        { \
            ucs_status_t status = uct_dc_mlx5_iface_dci_get(_iface, _ep); \
            if (ucs_unlikely(status != UCS_OK)) { \
                return UCS_STATUS_PTR(status); \
            } \
        } \
        UCT_RC_CHECK_NUM_RDMA_READ_RET(&(_iface)->super.super, \
                                       UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE)) \
    }


/**
 * All operations are not allowed if no RDMA_READ credits. Otherwise operations
 * ordering can be broken. If some AM sends added to the pending queue after
 * RDMA_READ operation, it may be stuck there until RDMA_READ credits arrive,
 * therefore need to block even AM sends, until all resources are available.
 */
#define UCT_DC_MLX5_CHECK_RES(_iface, _ep) \
    { \
        UCT_DC_MLX5_CHECK_DCI_RES(_iface, _ep) \
        UCT_RC_CHECK_NUM_RDMA_READ_RET(&(_iface)->super.super, \
                                       UCS_ERR_NO_RESOURCE) \
    }


/* First, check whether we have FC window. If hard threshold is reached, credit
 * request will be sent by "fc_ctrl" as a separate message. TX resources
 * are checked after FC, because fc credits request may consume latest
 * available TX resources. */
#define UCT_DC_CHECK_RES_AND_FC(_iface, _ep) \
    { \
        UCT_DC_MLX5_CHECK_RES(_iface, _ep) \
        if (ucs_unlikely((_ep)->fc.fc_wnd <= \
                         (_iface)->super.super.config.fc_hard_thresh)) { \
            ucs_status_t _status = uct_dc_mlx5_ep_check_fc(_iface, _ep); \
            if (ucs_unlikely(_status != UCS_OK)) { \
                if ((_ep)->fc.fc_wnd <= 0) { \
                    UCS_STATS_UPDATE_COUNTER((_ep)->fc.stats, \
                                             UCT_RC_FC_STAT_NO_CRED, 1); \
                    UCS_STATS_UPDATE_COUNTER((_ep)->super.stats, \
                                             UCT_EP_STAT_NO_RES, 1); \
                } \
                return _status; \
            } \
        } \
        if (!uct_dc_mlx5_iface_is_policy_shared(_iface)) { \
            uct_rc_iface_check_pending(&(_iface)->super.super, \
                                       &(_ep)->arb_group); \
        } \
    }


#endif
