/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "rc_verbs_common.h"

#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <uct/ib/rc/base/rc_iface.h>


ucs_config_field_t uct_rc_verbs_iface_common_config_table[] = {
  {"MAX_AM_HDR", "128",
   "Buffer size to reserve for active message headers. If set to 0, the transport will\n"
   "not support zero-copy active messages.",
   ucs_offsetof(uct_rc_verbs_iface_common_config_t, max_am_hdr), UCS_CONFIG_TYPE_MEMUNITS},

  {"TX_MAX_WR", "-1",
   "Limits the number of outstanding posted work requests. The actual limit is\n"
   "a minimum between this value and the TX queue length. -1 means no limit.",
   ucs_offsetof(uct_rc_verbs_iface_common_config_t, tx_max_wr), UCS_CONFIG_TYPE_UINT},

  {NULL}
};

void uct_rc_verbs_iface_common_query(uct_rc_verbs_iface_common_t *verbs_iface,
                                     uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    /* PUT */
    iface_attr->cap.put.max_short = verbs_iface->config.max_inline;
    iface_attr->cap.put.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.put.min_zcopy = 0;
    iface_attr->cap.put.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;
    iface_attr->cap.put.max_iov   = uct_ib_iface_get_max_iov(&iface->super);

    /* GET */
    iface_attr->cap.get.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.get.min_zcopy = iface->super.config.max_inl_resp + 1;
    iface_attr->cap.get.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;
    iface_attr->cap.get.max_iov   = uct_ib_iface_get_max_iov(&iface->super);

    /* AM */
    iface_attr->cap.am.max_short  = verbs_iface->config.max_inline - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_bcopy  = iface->super.config.seg_size - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.min_zcopy  = 0;
    iface_attr->cap.am.max_zcopy  = iface->super.config.seg_size - sizeof(uct_rc_hdr_t);
    /* The first IOV is reserved for the header */
    iface_attr->cap.am.max_iov    = uct_ib_iface_get_max_iov(&iface->super) - 1;

    /* TODO: may need to change for dc/rc */
    iface_attr->cap.am.max_hdr    = verbs_iface->config.short_desc_size - sizeof(uct_rc_hdr_t);

    iface_attr->cap.flags        |= UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    /* Software overhead */
    iface_attr->overhead          = 75e-9;
}

unsigned uct_rc_verbs_iface_post_recv_always(uct_rc_iface_t *iface,
                                             uct_rc_srq_t *srq, unsigned max)
{
    struct ibv_recv_wr *bad_wr;
    uct_ib_recv_wr_t *wrs;
    unsigned count;
    int ret;

    wrs  = ucs_alloca(sizeof *wrs  * max);

    count = uct_ib_iface_prepare_rx_wrs(&iface->super, &iface->rx.mp,
                                        wrs, max);
    if (ucs_unlikely(count == 0)) {
        return 0;
    }

    ret = ibv_post_srq_recv(srq->srq, &wrs[0].ibwr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_srq_recv() returned %d: %m", ret);
    }
    srq->available -= count;

    return count;
}

ucs_status_t uct_rc_verbs_iface_prepost_recvs_common(uct_rc_iface_t *iface,
                                                     uct_rc_srq_t *srq)
{
    while (srq->available > 0) {
        if (uct_rc_verbs_iface_post_recv_common(iface, srq, 1) == 0) {
            ucs_error("failed to post receives");
            return UCS_ERR_NO_MEMORY;
        }
    }
    return UCS_OK;
}

void uct_rc_verbs_iface_common_cleanup(uct_rc_verbs_iface_common_t *self)
{
    ucs_mpool_cleanup(&self->short_desc_mp, 1);
}


ucs_status_t uct_rc_verbs_iface_common_init(uct_rc_verbs_iface_common_t *iface,
                                            uct_rc_iface_t *rc_iface,
                                            uct_rc_verbs_iface_common_config_t *config,
                                            uct_rc_iface_config_t *rc_config)
{
    ucs_status_t status;
    size_t am_hdr_size;

    memset(iface->inl_sge, 0, sizeof(iface->inl_sge));

    /* Configuration */
    am_hdr_size = ucs_max(config->max_am_hdr, sizeof(uct_rc_hdr_t));
    iface->config.short_desc_size = ucs_max(UCT_RC_MAX_ATOMIC_SIZE, am_hdr_size);

    /* Create AM headers and Atomic mempool */
    status = uct_iface_mpool_init(&rc_iface->super.super,
                                  &iface->short_desc_mp,
                                  sizeof(uct_rc_iface_send_desc_t) +
                                      iface->config.short_desc_size,
                                  sizeof(uct_rc_iface_send_desc_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &rc_config->super.tx.mp,
                                  rc_iface->config.tx_qp_len,
                                  uct_rc_iface_send_desc_init,
                                  "rc_verbs_short_desc");
    if (status != UCS_OK) {
        return status;
    }
    return UCS_OK;
}

void uct_rc_verbs_txcnt_init(uct_rc_verbs_txcnt_t *txcnt)
{
    txcnt->pi = txcnt->ci = 0;
}

static
void uct_rc_ep_get_zcopy_handler(uct_rc_iface_send_op_t *op, const void *resp)
{
    uct_rc_iface_send_desc_t *desc = ucs_derived_of(op, uct_rc_iface_send_desc_t);
    char *desc_ptr                 = (char *)(desc + 1);
    size_t desc_offset             = 0;
    uct_rc_iface_send_iov_save_desc_t *iov_desc;
    size_t iov_it, iovcnt, length;

    ucs_assert(NULL != desc->super.buffer);
    iov_desc = desc->super.buffer;
    iovcnt   = iov_desc->iovcnt;

    /* copy the payload arrived to the mpool descriptor to the user buffers */
    for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
        length = iov_desc->length[iov_it];
        memcpy(iov_desc->buffer[iov_it], desc_ptr + desc_offset, length);
        desc_offset += length;
    }

    ucs_mpool_put(desc->super.buffer);
    ucs_mpool_put(desc);
}

ucs_status_t
uct_rc_iface_verbs_iov_rdma_read_callback(uct_iface_h tl_iface, uint32_t length,
                                          void **desc_iov_p, size_t *desc_offset_p,
                                          uint64_t *addr_p, uint32_t *lkey_p)
{
    uct_rc_iface_send_desc_t *desc  = *desc_iov_p;
    char *desc_ptr                  = NULL;
    uct_rc_iface_send_iov_save_desc_t *iov_desc;
    uct_rc_iface_t *iface ;

    if (NULL == desc) {
        iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
        UCT_RC_IFACE_GET_TX_DESC(iface, &iface->tx.mp, desc);
        UCT_RC_IFACE_GET_TX_DESC(iface, &iface->tx.mp, iov_desc);
        VALGRIND_MAKE_MEM_DEFINED(desc + 1, iface->super.config.seg_size);

        /* schedule a callback to copy payload to the user buffer at completion */
        desc->super.handler = uct_rc_ep_get_zcopy_handler;
        desc->super.buffer  = iov_desc;
        iov_desc->iovcnt    = 0;
        *desc_iov_p         = desc;
    }

    iov_desc = desc->super.buffer;

    ucs_assert(NULL != iov_desc);
    ucs_assert(iov_desc->iovcnt < UCT_IB_MAX_IOV);

    /* save offset to extract payload from descriptor later */
    iov_desc->length[iov_desc->iovcnt] = length;
    iov_desc->buffer[iov_desc->iovcnt] = (void *) *addr_p;
    ++iov_desc->iovcnt;
    desc_ptr        = (char *)(desc + 1);
    *addr_p         = (uintptr_t)(desc_ptr + *desc_offset_p);
    *lkey_p         = desc->lkey;
    *desc_offset_p += length;

    return UCS_OK;
}

ucs_status_t uct_rc_iface_verbs_iov_callback(uct_iface_h tl_iface, uint32_t length,
                                             void **desc_iov_p, size_t *desc_offset_p,
                                             uint64_t *addr_p, uint32_t *lkey_p)
{

    uct_rc_iface_send_desc_t *desc  = *desc_iov_p;
    char *desc_ptr                  = NULL;
    uct_rc_iface_t *iface ;

    if (NULL == desc) {
        iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
        UCT_RC_IFACE_GET_TX_DESC(iface, &iface->tx.mp, desc);
        desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
        *desc_iov_p = desc;
    }

    desc_ptr = (char *)(desc + 1);

    /* copy payload from user buffer to the descriptor */
    memcpy(desc_ptr + *desc_offset_p, (void *) *addr_p, length);
    *addr_p         = (uintptr_t)(desc_ptr + *desc_offset_p);
    *lkey_p         = desc->lkey;
    *desc_offset_p += length;

    return UCS_OK;
}

