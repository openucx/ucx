/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_MLX5_H
#define UCT_RC_MLX5_H

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <ucs/datastruct/queue.h>
#include <ucs/type/class.h>


#define UCT_RC_MLX5_MAX_BB   4 /* Max number of BB per WQE */


enum {
    UCT_RC_MLX5_IFACE_STAT_RX_INL_32,
    UCT_RC_MLX5_IFACE_STAT_RX_INL_64,
    UCT_RC_MLX5_IFACE_STAT_LAST
};


typedef struct uct_rc_mlx5_recv_desc uct_rc_mlx5_recv_desc_t;
struct uct_rc_mlx5_recv_desc {
    uct_ib_iface_recv_desc_t   super;
    ucs_queue_elem_t           queue;
} UCS_S_PACKED;


/**
 * RC mlx5 interface configuration
 */
typedef struct uct_rc_mlx5_iface_config {
    uct_rc_iface_config_t  super;
    /* TODO wc_mode, UAR mode SnB W/A... */
} uct_rc_mlx5_iface_config_t;


/**
 * RC remote endpoint
 */
typedef struct uct_rc_mlx5_ep {
    uct_rc_ep_t      super;
    unsigned         qp_num;

    struct {
        uct_ib_mlx5_txwq_t  wq;
    } tx;
} uct_rc_mlx5_ep_t;


/**
 * RC communication interface
 */
typedef struct {
    uct_rc_iface_t         super;

    struct {
        uct_ib_mlx5_cq_t   cq;
        ucs_mpool_h        atomic_desc_mp;
    } tx;

    struct {
        uct_ib_mlx5_cq_t   cq;
        ucs_queue_head_t   desc_q;
        void               *buf;
        uint32_t           *db;
        uint16_t           head;
        uint16_t           tail;
        uint16_t           sw_pi;
    } rx;

    UCS_STATS_NODE_DECLARE(stats);

} uct_rc_mlx5_iface_t;


/*
 * We can post if we're not starting further that max_pi.
 * See also uct_rc_mlx5_calc_max_pi().
 */
#define UCT_RC_MLX5_CHECK_RES(_iface, _ep) \
    if (!uct_rc_iface_have_tx_cqe_avail(&(_iface)->super)) { \
        UCS_STATS_UPDATE_COUNTER((_iface)->super.stats, UCT_RC_IFACE_STAT_NO_CQE, 1); \
        return UCS_ERR_NO_RESOURCE; \
    } \
    if ((_ep)->super.available == 0) { \
        UCS_STATS_UPDATE_COUNTER((_ep)->super.stats, UCT_RC_EP_STAT_QP_FULL, 1); \
        return UCS_ERR_NO_RESOURCE; \
    }

UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_mlx5_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_mlx5_ep_t, uct_ep_t);

ucs_status_t uct_rc_mlx5_ep_put_short(uct_ep_h tl_ep, const void *buffer, unsigned length,
                                      uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_rc_mlx5_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                      void *arg, size_t length, uint64_t remote_addr,
                                      uct_rkey_t rkey);

ucs_status_t uct_rc_mlx5_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, size_t length,
                                      uct_mem_h memh, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_get_bcopy(uct_ep_h tl_ep,
                                      uct_unpack_callback_t unpack_cb,
                                      void *arg, size_t length,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                      uct_mem_h memh, uint64_t remote_addr,
                                      uct_rkey_t rkey, uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                     const void *payload, unsigned length);

ucs_status_t uct_rc_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                     uct_pack_callback_t pack_cb, void *arg,
                                     size_t length);

ucs_status_t uct_rc_mlx5_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                     unsigned header_length, const void *payload,
                                     size_t length, uct_mem_h memh,
                                     uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_rc_mlx5_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint64_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint64_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_rc_mlx5_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint32_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uint32_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_mlx5_ep_flush(uct_ep_h tl_ep);

#endif
