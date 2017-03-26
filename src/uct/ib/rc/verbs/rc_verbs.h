/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_RC_VERBS_H
#define UCT_RC_VERBS_H

#include <uct/ib/rc/base/rc_iface.h>
#include <uct/ib/rc/base/rc_ep.h>
#include <ucs/type/class.h>

#include "rc_verbs_common.h"


/**
 * RC verbs communication context.
 */
typedef struct uct_rc_verbs_ep {
    uct_rc_ep_t            super;
    uct_rc_verbs_txcnt_t   txcnt;
#if HAVE_IBV_EX_HW_TM
    struct ibv_qp          *tm_qp;
#endif
} uct_rc_verbs_ep_t;


/**
 * RC verbs interface configuration.
 */
typedef struct uct_rc_verbs_iface_config {
    uct_rc_iface_config_t              super;
    uct_rc_verbs_iface_common_config_t verbs_common;
    uct_rc_fc_config_t                 fc;
#if HAVE_IBV_EX_HW_TM
    struct {
        int                            enable;
        unsigned                       list_size;
        unsigned                       rndv_queue_len;
    } tm;
#endif
} uct_rc_verbs_iface_config_t;


/**
 * RC verbs interface.
 */
typedef struct uct_rc_verbs_iface {
    uct_rc_iface_t              super;
    struct ibv_send_wr          inl_am_wr;
    struct ibv_send_wr          inl_rwrite_wr;
    uct_rc_verbs_iface_common_t verbs_common;
#if HAVE_IBV_EX_HW_TM
    struct {
        uct_rc_srq_t            xrq;       /* TM XRQ */
        unsigned                tag_available;
        uint8_t                 enabled;
        struct {
            void                     *arg; /* User defined arg */
            uct_tag_unexp_eager_cb_t cb;   /* Callback for unexpected eager messages */
        } eager_unexp;

        struct {
            void                     *arg; /* User defined arg */
            uct_tag_unexp_rndv_cb_t  cb;   /* Callback for unexpected rndv messages */
        } rndv_unexp;
    } tm;
#endif
    struct {
        unsigned                tx_max_wr;
    } config;

    void (*progress)(void*); /* Progress function (either regular or TM aware) */
} uct_rc_verbs_iface_t;


#define UCT_RC_VERBS_CHECK_AM_SHORT(_iface, _id, _length, _max_inline) \
     UCT_CHECK_AM_ID(_id); \
     UCT_CHECK_LENGTH(sizeof(uct_rc_am_short_hdr_t) + _length + \
                      (_iface)->verbs_common.config.notag_hdr_size, \
                      _max_inline, "am_short");

#define UCT_RC_VERBS_CHECK_AM_ZCOPY(_iface, _id, _header_len, _len, _desc_size, _seg_size) \
     UCT_RC_CHECK_AM_ZCOPY_DATA(_id, _header_len, _len, _seg_size) \
     UCT_CHECK_LENGTH(sizeof(uct_rc_hdr_t) + _header_len + \
                      (_iface)->verbs_common.config.notag_hdr_size, \
                      _desc_size, "am_zcopy header");

#define UCT_RC_VERBS_GET_TX_TM_DESC(_iface, _mp, _desc, _hdr, _len) \
     { \
         UCT_RC_IFACE_GET_TX_DESC(&(_iface)->super, _mp, _desc) \
         hdr = _desc + 1; \
         len = uct_rc_verbs_notag_header_fill(_iface, _hdr); \
     }

#define UCT_RC_VERBS_GET_TX_AM_BCOPY_DESC(_iface, _mp, _desc, _id, _pack_cb, \
                                          _arg, _length, _data_length) \
     { \
         void *hdr; \
         size_t len; \
         UCT_RC_VERBS_GET_TX_TM_DESC(_iface, _mp, _desc, hdr, len) \
         (_desc)->super.handler = (uct_rc_send_handler_t)ucs_mpool_put; \
         uct_rc_bcopy_desc_fill(hdr + len, _id, _pack_cb, _arg, &(_data_length)); \
         _length = _data_length + len + sizeof(uct_rc_hdr_t); \
     }

#define UCT_RC_VERBS_GET_TX_AM_ZCOPY_DESC(_iface, _mp, _desc, _id, _header, \
                                          _header_length, _comp, _send_flags, _sge) \
     { \
         void *hdr; \
         size_t len; \
         UCT_RC_VERBS_GET_TX_TM_DESC(_iface, _mp, _desc, hdr, len) \
         uct_rc_zcopy_desc_set_comp(_desc, _comp, _send_flags); \
         uct_rc_zcopy_desc_set_header(hdr + len, _id, _header, _header_length); \
         _sge.length = sizeof(uct_rc_hdr_t) + header_length + len; \
     }


#if HAVE_IBV_EX_HW_TM

/* For RNDV TM enabling 2 QPs should be created, one is for sending WRs and
 * another one for HW (device will use it for RDMA reads and sending RNDV
 * Complete messages).*/
typedef struct uct_rc_verbs_ep_tm_address {
    uct_rc_ep_address_t         super;
    uct_ib_uint24_t             tm_qp_num;
} UCS_S_PACKED uct_rc_verbs_ep_tm_address_t;


#  define UCT_RC_VERBS_TAG_MIN_POSTED  33

#  define UCT_RC_VERBS_TM_ENABLED(_iface) \
       (IBV_DEVICE_TM_CAPS(uct_ib_iface_device(&(_iface)->super.super), max_num_tags) && \
        (_iface)->tm.enabled)

#  define UCT_RC_VERBS_TM_CONFIG(_config, _field)  (_config)->tm._field

ucs_status_t uct_rc_verbs_ep_tag_qp_create(uct_rc_verbs_iface_t *iface,
                                           uct_rc_verbs_ep_t *ep);

ucs_status_t uct_rc_verbs_ep_tag_qp_destroy(uct_rc_verbs_ep_t *ep);

ucs_status_t uct_rc_verbs_ep_tag_get_address(uct_ep_h tl_ep,
                                             uct_ep_addr_t *addr);

ucs_status_t uct_rc_verbs_ep_tag_connect_to_ep(uct_ep_h tl_ep,
                                               const uct_device_addr_t *dev_addr,
                                               const uct_ep_addr_t *ep_addr);
#else

#  define UCT_RC_VERBS_TM_ENABLED(_iface)   0
#  define uct_rc_verbs_ep_tag_qp_create     ucs_empty_function_return_unsupported
#  define uct_rc_verbs_ep_tag_qp_destroy    ucs_empty_function_return_unsupported
#  define uct_rc_verbs_ep_tag_get_address   ucs_empty_function_return_unsupported
#  define uct_rc_verbs_ep_tag_connect_to_ep ucs_empty_function_return_unsupported

#endif /* HAVE_IBV_EX_HW_TM */


static UCS_F_ALWAYS_INLINE size_t
uct_rc_verbs_notag_header_fill(uct_rc_verbs_iface_t *iface, void *hdr)
{
#if HAVE_IBV_EX_HW_TM
    struct ibv_tm_info tm_info;
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super.super);

    if (UCT_RC_VERBS_TM_ENABLED(iface)) {
        tm_info.op = IBV_TM_OP_NO_TAG;
        return ibv_pack_tm_info(dev->ibv_context, hdr, &tm_info);
    }
#endif

    return 0;
}


UCS_CLASS_DECLARE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);

void uct_rc_verbs_ep_am_packet_dump(uct_base_iface_t *iface,
                                    uct_am_trace_type_t type,
                                    void *data, size_t length,
                                    size_t valid_length,
                                    char *buffer, size_t max);

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
                                 uct_pack_callback_t pack_cb, void *arg);

ucs_status_t uct_rc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const uct_iov_t *iov,
                                      size_t iovcnt, uct_completion_t *comp);

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

ucs_status_t uct_rc_verbs_ep_flush(uct_ep_h tl_ep, unsigned flags,
                                   uct_completion_t *comp);

void uct_rc_verbs_iface_progress(void *arg);

ucs_status_t uct_rc_verbs_ep_fc_ctrl(uct_ep_t *tl_ep, unsigned op,
                                     uct_rc_fc_request_t *req);

#endif
