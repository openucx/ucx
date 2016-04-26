/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "rc_verbs_common.h"

#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <uct/ib/rc/base/rc_iface.h>


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

void uct_rc_verbs_iface_query_common(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr, 
                                     int max_inline, int short_desc_size)
{
    struct ibv_exp_device_attr *dev_attr = &uct_ib_iface_device(&iface->super)->dev_attr;

    /* PUT */
    iface_attr->cap.put.max_short = max_inline;
    iface_attr->cap.put.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.put.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;

    /* GET */
    iface_attr->cap.get.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.get.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;

    /* AM */
    iface_attr->cap.am.max_short  = max_inline - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_bcopy  = iface->super.config.seg_size
                                    - sizeof(uct_rc_hdr_t);

    iface_attr->cap.am.max_zcopy  = iface->super.config.seg_size
                                    - sizeof(uct_rc_hdr_t);
    /* TODO: may need to change for dc/rc */
    iface_attr->cap.am.max_hdr    = short_desc_size - sizeof(uct_rc_hdr_t);

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

void uct_rc_verbs_iface_common_init(uct_rc_verbs_iface_common_t *self)
{
    memset(self->inl_sge, 0, sizeof(self->inl_sge));
}

void uct_rc_verbs_txcnt_init(uct_rc_verbs_txcnt_t *txcnt)
{
    txcnt->pi = txcnt->ci = 0;
}
