/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_VERBS_H
#define UCT_RC_VERBS_H

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <ucs/type/class.h>

#define UCT_RC_VERBS_IFACE_FOREACH_TXWQE(_iface, _i, _wc, _num_wcs) \
      status = uct_ib_poll_cq((_iface)->super.cq[UCT_IB_DIR_TX], &_num_wcs, _wc); \
      if (status != UCS_OK) { \
          return 0; \
      } \
      UCS_STATS_UPDATE_COUNTER((_iface)->stats, \
                               UCT_RC_IFACE_STAT_TX_COMPLETION, _num_wcs); \
      for (_i = 0; _i < _num_wcs; ++_i)


typedef struct uct_rc_verbs_txcnt {
    uint16_t       pi;      /* producer (post_send) count */
    uint16_t       ci;      /* consumer (ibv_poll_cq) completion count */
} uct_rc_verbs_txcnt_t;

/**
 * RC verbs communication context.
 */
typedef struct uct_rc_verbs_ep {
    uct_rc_ep_t            super;
    uct_rc_verbs_txcnt_t   txcnt;
} uct_rc_verbs_ep_t;


/**
 * RC verbs interface configuration.
 */
typedef struct uct_rc_verbs_iface_config {
    uct_rc_iface_config_t              super;
    size_t                             max_am_hdr;
    unsigned                           tx_max_wr;
    uct_rc_fc_config_t                 fc;
} uct_rc_verbs_iface_config_t;


/**
 * RC verbs interface.
 */
typedef struct uct_rc_verbs_iface {
    uct_rc_iface_t              super;
    struct ibv_send_wr          inl_am_wr;
    struct ibv_send_wr          inl_rwrite_wr;
    struct ibv_sge              inl_sge[2];
    uct_rc_am_short_hdr_t       am_inl_hdr;
    ucs_mpool_t                 short_desc_mp;
    uct_rc_iface_send_desc_t   *fc_desc; /* used when max_inline is zero */
    struct {
        size_t                  short_desc_size;
        size_t                  max_inline;
        unsigned                tx_max_wr;
    } config;
} uct_rc_verbs_iface_t;


UCS_CLASS_DECLARE(uct_rc_verbs_ep_t);
UCS_CLASS_DECLARE_INIT_FUNC(uct_rc_verbs_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);

ucs_status_t uct_rc_verbs_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey);

ssize_t uct_rc_verbs_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                  void *arg, uint64_t remote_addr,
                                  uct_rkey_t rkey);

ucs_status_t uct_rc_verbs_ep_put_zcopy(uct_ep_h tl_ep,
                                       const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_get_bcopy(uct_ep_h tl_ep,
                                       uct_unpack_callback_t unpack_cb,
                                       void *arg, size_t length,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_get_zcopy(uct_ep_h tl_ep,
                                       const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      const void *buffer, unsigned length);

ssize_t uct_rc_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                 uct_pack_callback_t pack_cb, void *arg,
                                 unsigned flags);

ucs_status_t uct_rc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const uct_iov_t *iov,
                                      size_t iovcnt, unsigned flags,
                                      uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint64_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint32_t *result, uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic64_post(uct_ep_h tl_ep, unsigned opcode, uint64_t value,
                                           uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_rc_verbs_ep_atomic32_post(uct_ep_h tl_ep, unsigned opcode, uint32_t value,
                                           uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_rc_verbs_ep_atomic64_fetch(uct_ep_h tl_ep, uct_atomic_op_t opcode,
                                            uint64_t value, uint64_t *result,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_atomic32_fetch(uct_ep_h tl_ep, uct_atomic_op_t opcode,
                                            uint32_t value, uint32_t *result,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                   uct_completion_t *comp);

ucs_status_t uct_rc_verbs_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                     uct_rc_fc_request_t *req);

#endif
