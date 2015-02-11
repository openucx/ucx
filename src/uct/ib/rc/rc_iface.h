/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_IFACE_H
#define UCT_RC_IFACE_H

#include "rc_ep.h"

#include <uct/tl/tl_base.h>
#include <uct/ib/base/ib_iface.h>
#include <ucs/debug/log.h>


#define UCT_RC_QP_TABLE_ORDER       12
#define UCT_RC_QP_TABLE_SIZE        UCS_BIT(UCT_RC_QP_TABLE_ORDER)
#define UCT_RC_QP_TABLE_MEMB_ORDER  (UCT_IB_QPN_ORDER - UCT_RC_QP_TABLE_ORDER)
#define UCT_RC_MAX_ATOMIC_SIZE      sizeof(uint64_t)


struct uct_rc_iface {
    uct_ib_iface_t           super;

    struct {
        ucs_mpool_h          mp;
        unsigned             cq_available;
    } tx;

    struct {
        ucs_mpool_h          mp;
        struct ibv_srq       *srq;
        unsigned             available;
    } rx;

    struct {
        unsigned             tx_qp_len;
        unsigned             tx_min_sge;
        unsigned             tx_min_inline;
        unsigned             tx_moderation;
        unsigned             rx_max_batch;
        unsigned             rx_inline;
    } config;

    uct_rc_ep_t              **eps[UCT_RC_QP_TABLE_SIZE];
};


struct uct_rc_iface_config {
    uct_ib_iface_config_t    super;

    struct {
        unsigned             cq_len;
    } tx;
};


struct uct_rc_iface_send_desc {
    ucs_callbackq_elem_t     queue;
    uint32_t                 lkey;
    union {
        struct {
            ucs_callback_t            *cb;
        } callback;

        struct {
            uct_bcopy_recv_callback_t cb;
            void                      *arg;
            size_t                    length;
        } bcopy_recv;

        struct {
            uct_imm_recv_callback_t   cb;
            void                      *arg;
        } imm_recv;
    };
};


extern ucs_config_field_t uct_rc_iface_config_table[];

void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_rc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

void uct_rc_iface_add_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep);
void uct_rc_iface_remove_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep);

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface);
void uct_rc_iface_send_desc_init(uct_iface_h tl_iface, void *obj, uct_lkey_t lkey);

/**
 * Creates an RC QP and fills 'cap' with QP capabilities;
 */
ucs_status_t uct_rc_iface_qp_create(uct_rc_iface_t *iface, struct ibv_qp **qp_p,
                                    struct ibv_qp_cap *cap);


static inline uct_rc_ep_t *uct_rc_iface_lookup_ep(uct_rc_iface_t *iface,
                                                  unsigned qp_num)
{
    ucs_assert(qp_num < UCS_BIT(UCT_IB_QPN_ORDER));
    return iface->eps[qp_num >> UCT_RC_QP_TABLE_ORDER]
                     [qp_num &  UCS_MASK(UCT_RC_QP_TABLE_MEMB_ORDER)];
}


static UCS_F_ALWAYS_INLINE uint8_t
uct_rc_iface_tx_moderation(uct_rc_iface_t* iface, uct_rc_ep_t* ep, uint8_t flag)
{
    return (ep->tx.unsignaled >= iface->config.tx_moderation) ? flag : 0;
}

static UCS_F_ALWAYS_INLINE int
uct_rc_iface_have_tx_cqe_avail(uct_rc_iface_t* iface)
{
    return iface->tx.cq_available > 0;
}

static inline ucs_status_t
uct_rc_iface_invoke_am(uct_rc_iface_t *iface, uct_ib_iface_recv_desc_t *desc,
                       uct_rc_hdr_t *hdr, unsigned byte_len)
{
    ucs_trace_data("RX: AM [%d]", hdr->am_id);
    return uct_iface_invoke_am(&iface->super.super, hdr->am_id, desc, hdr + 1,
                               byte_len - sizeof(*hdr));
}

#endif
