/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "rc_verbs_common.h"

#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <uct/ib/rc/base/rc_iface.h>


ucs_config_field_t uct_rc_verbs_iface_config_table[] = {
  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_verbs_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {"MAX_AM_HDR", "128",
   "Buffer size to reserve for active message headers. If set to 0, the transport will\n"
   "not support zero-copy active messages.",
   ucs_offsetof(uct_rc_verbs_iface_config_t, max_am_hdr), UCS_CONFIG_TYPE_MEMUNITS},

  {NULL}
};

static int
uct_rc_verbs_is_ext_atomic_supported(struct ibv_exp_device_attr *dev_attr,
                                     size_t atomic_size)
{
#ifdef HAVE_IB_EXT_ATOMICS
    struct ibv_exp_ext_atomics_params ext_atom = dev_attr->ext_atom;
    return (ext_atom.log_max_atomic_inline >= ucs_ilog2(atomic_size)) &&
           (ext_atom.log_atomic_arg_sizes & atomic_size);
#else
      return 0;
#endif
}

void uct_rc_verbs_iface_common_query(uct_rc_verbs_iface_common_t *verbs_iface,
                                     uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    struct ibv_exp_device_attr *dev_attr = &uct_ib_iface_device(&iface->super)->dev_attr;

    /* PUT */
    iface_attr->cap.put.max_short = verbs_iface->config.max_inline;
    iface_attr->cap.put.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.put.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;

    /* GET */
    iface_attr->cap.get.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.get.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;

    /* AM */
    iface_attr->cap.am.max_short  = verbs_iface->config.max_inline - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_bcopy  = iface->super.config.seg_size
                                    - sizeof(uct_rc_hdr_t);

    iface_attr->cap.am.max_zcopy  = iface->super.config.seg_size
                                    - sizeof(uct_rc_hdr_t);
    /* TODO: may need to change for dc/rc */
    iface_attr->cap.am.max_hdr    = verbs_iface->config.short_desc_size - sizeof(uct_rc_hdr_t);

    /*
     * Atomics.
     * Need to make sure device support at least one kind of atomics.
     */
    if (IBV_EXP_HAVE_ATOMIC_HCA(dev_attr) ||
        IBV_EXP_HAVE_ATOMIC_GLOB(dev_attr) ||
        IBV_EXP_HAVE_ATOMIC_HCA_REPLY_BE(dev_attr))
    {
        iface_attr->cap.flags |= UCT_IFACE_FLAG_ATOMIC_ADD64 |
                                 UCT_IFACE_FLAG_ATOMIC_FADD64 |
                                 UCT_IFACE_FLAG_ATOMIC_CSWAP64;

        if (uct_rc_verbs_is_ext_atomic_supported(dev_attr, sizeof(uint32_t))) {
            iface_attr->cap.flags |= UCT_IFACE_FLAG_ATOMIC_ADD32 |
                                     UCT_IFACE_FLAG_ATOMIC_FADD32 |
                                     UCT_IFACE_FLAG_ATOMIC_SWAP32 |
                                     UCT_IFACE_FLAG_ATOMIC_CSWAP32;
        }

        if (uct_rc_verbs_is_ext_atomic_supported(dev_attr, sizeof(uint64_t))) {
            iface_attr->cap.flags |= UCT_IFACE_FLAG_ATOMIC_SWAP64;
        }
    }

    /* Software overhead */
    iface_attr->overhead = 75e-9;
}


unsigned uct_rc_verbs_iface_post_recv_always(uct_rc_iface_t *iface, unsigned max)
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

    UCT_IB_INSTRUMENT_RECORD_RECV_WR_LEN("uct_rc_iface_post_recv_always",
                                         &wrs[0].ibwr);
    ret = ibv_post_srq_recv(iface->rx.srq, &wrs[0].ibwr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_srq_recv() returned %d: %m", ret);
    }
    iface->rx.available -= count;

    return count;
}

ucs_status_t uct_rc_verbs_iface_prepost_recvs_common(uct_rc_iface_t *iface)
{
    while (iface->rx.available > 0) {
        if (uct_rc_verbs_iface_post_recv_common(iface, 1) == 0) {
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


ucs_status_t uct_rc_verbs_iface_common_init(uct_rc_verbs_iface_common_t *self,
                                            uct_rc_iface_t *rc_iface,
                                            uct_rc_verbs_iface_config_t *config)
{
    ucs_status_t status;
    size_t am_hdr_size;
    struct ibv_exp_device_attr *dev_attr;

    memset(self->inl_sge, 0, sizeof(self->inl_sge));

    /* Configuration */
    am_hdr_size = ucs_max(config->max_am_hdr, sizeof(uct_rc_hdr_t));
    self->config.short_desc_size = ucs_max(UCT_RC_MAX_ATOMIC_SIZE, am_hdr_size);
    dev_attr = &uct_ib_iface_device(&rc_iface->super)->dev_attr;
    if (IBV_EXP_HAVE_ATOMIC_HCA(dev_attr) || IBV_EXP_HAVE_ATOMIC_GLOB(dev_attr)) {
        self->config.atomic32_handler = uct_rc_ep_atomic_handler_32_be0;
        self->config.atomic64_handler = uct_rc_ep_atomic_handler_64_be0;
    } else if (IBV_EXP_HAVE_ATOMIC_HCA_REPLY_BE(dev_attr)) {
        self->config.atomic32_handler = uct_rc_ep_atomic_handler_32_be1;
        self->config.atomic64_handler = uct_rc_ep_atomic_handler_64_be1;
    }

    /* Create AM headers and Atomic mempool */
    status = uct_iface_mpool_init(&rc_iface->super.super,
                                  &self->short_desc_mp,
                                  sizeof(uct_rc_iface_send_desc_t) +
                                      self->config.short_desc_size,
                                  sizeof(uct_rc_iface_send_desc_t),
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->super.super.tx.mp,
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

