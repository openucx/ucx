/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_VERBS_H
#define UCT_RC_VERBS_H

#include "rc_iface.h"
#include "rc_ep.h"

#include <ucs/type/class.h>


/**
 * RC mlx5 interface configuration
 */
typedef struct uct_rc_verbs_iface_config {
    uct_rc_iface_config_t  super;
    size_t                 max_am_hdr;
    /* TODO flags for exp APIs */
} uct_rc_verbs_iface_config_t;


/**
 * RC verbs communication context.
 */
typedef struct uct_rc_verbs_ep {
    uct_rc_ep_t        super;

    struct {
        uint16_t       post_count;
        uint16_t       completion_count;
        unsigned       available;
    } tx;
} uct_rc_verbs_ep_t;


/**
 * RC verbs remote endpoint.
 */
typedef struct uct_rc_verbs_iface {
    uct_rc_iface_t     super;

    ucs_mpool_h        short_desc_mp;
    struct ibv_send_wr inl_am_wr;
    struct ibv_send_wr inl_rwrite_wr;
    struct ibv_sge     inl_sge[2];

    struct {
        size_t               short_desc_size;
        uct_completion_callback_t  atomic32_completoin;
        uct_completion_callback_t  atomic64_completoin;
        size_t               max_inline;
    } config;
} uct_rc_verbs_iface_t;


/*
 * We can post if we're not starting further that max_pi.
 * See also uct_rc_mlx5_calc_max_pi().
 */
#define UCT_RC_VERBS_CHECK_RES(_iface, _ep) \
    if (!uct_rc_iface_have_tx_cqe_avail(&(_iface)->super)) { \
        UCS_STATS_UPDATE_COUNTER((_iface)->super.stats, UCT_RC_IFACE_STAT_NO_CQE, 1); \
        return UCS_ERR_NO_RESOURCE; \
    } \
    if ((_ep)->tx.available == 0) { \
        UCS_STATS_UPDATE_COUNTER((_ep)->super.stats, UCT_RC_EP_STAT_QP_FULL, 1); \
        return UCS_ERR_NO_RESOURCE; \
    }


UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);

ucs_status_t uct_rc_verbs_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey);

ucs_status_t uct_rc_verbs_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                       void *arg, size_t length, uint64_t remote_addr,
                                       uct_rkey_t rkey);

ucs_status_t uct_rc_verbs_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, size_t length,
                                       uct_mem_h memh, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_get_bcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                       uct_mem_h memh, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      const void *buffer, unsigned length);

ucs_status_t uct_rc_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                      uct_pack_callback_t pack_cb, void *arg,
                                      size_t length);

ucs_status_t uct_rc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const void *payload,
                                      size_t length, uct_mem_h memh,
                                      uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_rc_verbs_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint64_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_rc_verbs_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint32_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_flush(uct_ep_h tl_ep);

#endif
