/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_RC_IFACE_H
#define UCT_RC_IFACE_H

#include "rc_def.h"

#include <uct/tl/tl_base.h>
#include <uct/ib/base/ib_iface.h>
#include <ucs/datastruct/queue.h>
#include <ucs/debug/log.h>


#define UCT_RC_QP_TABLE_ORDER       12
#define UCT_RC_QP_TABLE_SIZE        UCS_BIT(UCT_RC_QP_TABLE_ORDER)
#define UCT_RC_QP_TABLE_MEMB_ORDER  (UCT_IB_QPN_ORDER - UCT_RC_QP_TABLE_ORDER)
#define UCT_RC_MAX_ATOMIC_SIZE      sizeof(uint64_t)
#define UCR_RC_QP_MAX_RETRY_COUNT   7


#define UCT_RC_IFACE_GET_TX_DESC(_iface, _mp, _desc) \
    UCT_TL_IFACE_GET_TX_DESC(&(_iface)->super.super, _mp, _desc, \
                             return UCS_ERR_NO_RESOURCE);


enum {
    UCT_RC_IFACE_STAT_RX_COMPLETION,
    UCT_RC_IFACE_STAT_TX_COMPLETION,
    UCT_RC_IFACE_STAT_NO_CQE,
    UCT_RC_IFACE_STAT_LAST
};


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
        uint8_t              min_rnr_timer;
        uint8_t              timeout;
        uint8_t              rnr_retry;
        uint8_t              retry_cnt;
        uint8_t              max_rd_atomic;
        uint8_t              sl;
        enum ibv_mtu         path_mtu;
    } config;

    UCS_STATS_NODE_DECLARE(stats);

    uct_rc_ep_t              **eps[UCT_RC_QP_TABLE_SIZE];
};


struct uct_rc_iface_config {
    uct_ib_iface_config_t    super;
    uct_ib_mtu_t             path_mtu;
    unsigned                 max_rd_atomic;

    struct {
        double               timeout;
        unsigned             retry_count;
        double               rnr_timeout;
        unsigned             rnr_retry_count;
        unsigned             cq_len;
    } tx;
};


struct uct_rc_completion {
    uct_completion_t         super;
    ucs_queue_elem_t         queue;
    uint16_t                 sn;
#if ! NVALGRIND
    unsigned                 length;
#endif
};


struct uct_rc_iface_send_desc {
    uct_rc_completion_t      super;
    uint32_t                 lkey;
    uct_completion_t         *comp;
};


/**
 * RC network header.
 */
typedef struct uct_rc_hdr {
    uint8_t           am_id;  /* Active message ID */
} UCS_S_PACKED uct_rc_hdr_t;


/*
 * Short active message header (active message header is always 64 bit).
 */
typedef struct uct_rc_am_short_hdr {
    uct_rc_hdr_t      rc_hdr;
    uint64_t          am_hdr;
} UCS_S_PACKED uct_rc_am_short_hdr_t;


extern ucs_config_field_t uct_rc_iface_config_table[];

void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr);

ucs_status_t uct_rc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr);

void uct_rc_iface_add_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep);
void uct_rc_iface_remove_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep);

ucs_status_t uct_rc_iface_flush(uct_iface_h tl_iface);
void uct_rc_iface_send_desc_init(uct_iface_h tl_iface, void *obj, uct_mem_h memh);

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


static UCS_F_ALWAYS_INLINE int
uct_rc_iface_have_tx_cqe_avail(uct_rc_iface_t* iface)
{
    return iface->tx.cq_available > 0;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_iface_invoke_am(uct_rc_iface_t *iface, uct_rc_hdr_t *hdr, unsigned length,
                       uct_ib_iface_recv_desc_t *desc)
{
    return uct_iface_invoke_am(&iface->super.super, hdr->am_id, hdr + 1,
                               length - sizeof(*hdr), desc);
}

#endif
