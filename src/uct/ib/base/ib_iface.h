/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_IFACE_H
#define UCT_IB_IFACE_H

#include "ib_pd.h"

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/sys/compiler.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/mpool.inl>


/* Forward declarations */
typedef struct uct_ib_iface_config   uct_ib_iface_config_t;
typedef struct uct_ib_iface_ops      uct_ib_iface_ops_t;
typedef struct uct_ib_iface         uct_ib_iface_t;


/**
 * IB port/path MTU.
 */
typedef enum uct_ib_mtu {
    UCT_IB_MTU_DEFAULT = 0,
    UCT_IB_MTU_512     = 1,
    UCT_IB_MTU_1024    = 2,
    UCT_IB_MTU_2048    = 3,
    UCT_IB_MTU_4096    = 4,
    UCT_IB_MTU_LAST
} uct_ib_mtu_t;


struct uct_ib_iface_config {
    uct_iface_config_t      super;

    struct {
        unsigned            queue_len;       /* Queue length */
        unsigned            max_batch;       /* How many fragments can be batched to one post send */
        unsigned            max_poll;        /* How many wcs can be picked when polling tx cq */
        size_t              min_inline;      /* Inline space to reserve */
        unsigned            min_sge;         /* How many SG entries to support */
        unsigned            cq_moderation;   /* How many TX messages are batched to one CQE */
        uct_iface_mpool_config_t mp;
    } tx;

    struct {
        unsigned            queue_len;       /* Queue length */
        unsigned            max_batch;       /* How many buffers can be batched to one post receuive */
        unsigned            max_poll;        /* How many wcs can be picked when polling rx cq */
        size_t              inl;             /* Inline space to reserve in CQ/QP */
        uct_iface_mpool_config_t mp;
    } rx;

    /* IB GID index to use  */
    unsigned                gid_index;

    /* IB SL to use */
    unsigned                sl;

    /* Ranges of path bits */
    UCS_CONFIG_ARRAY_FIELD(ucs_range_spec_t, ranges) lid_path_bits;

    /* IB PKEY to use */
    unsigned                pkey_value;

};


struct uct_ib_iface_ops {
    uct_iface_ops_t         super;
    ucs_status_t            (*arm_tx_cq)(uct_ib_iface_t *iface);
    ucs_status_t            (*arm_rx_cq)(uct_ib_iface_t *iface, int solicited);
};


struct uct_ib_iface {
    uct_base_iface_t        super;
    uint8_t                 *path_bits;
    unsigned                path_bits_count;
    union ibv_gid           gid;
    uint16_t                pkey_index;
    uint16_t                pkey_value;
    uint8_t                 port_num;
    uint8_t                 sl;
    uct_ib_address_scope_t  addr_scope;
    uint8_t                 addr_size;

    struct ibv_cq           *send_cq;
    struct ibv_cq           *recv_cq;
    struct ibv_comp_channel *comp_channel;

    struct {
        unsigned            rx_payload_offset;   /* offset from desc to payload */
        unsigned            rx_hdr_offset;       /* offset from desc to network header */
        unsigned            rx_headroom_offset;  /* offset from desc to user headroom */
        unsigned            rx_max_batch;
        unsigned            rx_max_poll;
        unsigned            tx_max_poll;
        unsigned            seg_size;
    } config;

    uct_ib_iface_ops_t      *ops;

};
UCS_CLASS_DECLARE(uct_ib_iface_t, uct_ib_iface_ops_t*, uct_pd_h, uct_worker_h, const char*,
                  unsigned, unsigned, unsigned, unsigned, size_t, uct_ib_iface_config_t*)


/*
 * The offset to the payload is the maximum between user-requested headroom
 * and transport-specific data/header. When the active message callback is invoked,
 * it gets a pointer to the beginning of the headroom.
 * The headroom can be either smaller (1) or larger (2) than the transport data.
 *
 * (1)
 *
 * <rx_headroom_offset>
 *                   |
 *                   |
 * uct_recv_desc_t   |
 *               |   |
 *               |   am_callback
 *               |   |
 * +------+------+---+-----------+---------+
 * | LKey |  ??? | D | Head Room | Payload |
 * +------+------+---+--+--------+---------+
 * | LKey |     TL data | TL hdr | Payload |
 * +------+-------------+--------+---------+
 *                      |
 *                      post_receive
 *
 * (2)
 *            am_callback
 *            |
 * +------+---+------------------+---------+
 * | LKey | D |     Head Room    | Payload |
 * +------+---+-----+---+--------+---------+
 * | LKey | TL data | ? | TL hdr | Payload |
 * +------+---------+---+--------+---------+
 *                      |
 *                      post_receive
 *        <dsc>
 *            <--- rx_headroom -->
 * <------- rx_payload_offset --->
 * <--- rx_hdr_offset -->
 *
 */
typedef struct uct_ib_iface_recv_desc {
    uint32_t                lkey;
} UCS_S_PACKED uct_ib_iface_recv_desc_t;



extern ucs_config_field_t uct_ib_iface_config_table[];
extern const char *uct_ib_mtu_values[];


/**
 * Create memory pool of receive descriptors.
 */
ucs_status_t uct_ib_iface_recv_mpool_init(uct_ib_iface_t *iface,
                                            uct_ib_iface_config_t *config,
                                            const char *name, ucs_mpool_t *mp);

void uct_ib_iface_release_am_desc(uct_iface_t *tl_iface, void *desc);


static UCS_F_ALWAYS_INLINE void
uct_ib_iface_invoke_am(uct_ib_iface_t *iface, uint8_t am_id, void *data,
                       unsigned length, uct_ib_iface_recv_desc_t *ib_desc)
{
    void *desc = (char*)ib_desc + iface->config.rx_headroom_offset;
    ucs_status_t status;

    status = uct_iface_invoke_am(&iface->super, am_id, data, length, desc);
    if (status == UCS_OK) {
        ucs_mpool_put_inline(ib_desc);
    } else {
        uct_recv_desc_iface(desc) = &iface->super.super;
    }
}

ucs_status_t uct_ib_iface_get_device_address(uct_iface_h tl_iface,
                                             uct_device_addr_t *dev_addr);

int uct_ib_iface_is_reachable(uct_iface_h tl_iface, const uct_device_addr_t *addr);

/*
 * @param xport_hdr_len       How many bytes this transport adds on top of IB header (LRH+BTH+iCRC+vCRC)
 */
ucs_status_t uct_ib_iface_query(uct_ib_iface_t *iface, size_t xport_hdr_len,
                                uct_iface_attr_t *iface_attr);

static inline uct_ib_pd_t* uct_ib_iface_pd(uct_ib_iface_t *iface)
{
    return ucs_derived_of(iface->super.pd, uct_ib_pd_t);
}

static inline uct_ib_device_t* uct_ib_iface_device(uct_ib_iface_t *iface)
{
    return &uct_ib_iface_pd(iface)->dev;
}

static inline struct ibv_exp_port_attr* uct_ib_iface_port_attr(uct_ib_iface_t *iface)
{
    return uct_ib_device_port_attr(uct_ib_iface_device(iface), iface->port_num);
}

static inline void* uct_ib_iface_recv_desc_hdr(uct_ib_iface_t *iface,
                                               uct_ib_iface_recv_desc_t *desc)
{
    return (void*)((char *)desc + iface->config.rx_hdr_offset);
}

typedef struct uct_ib_recv_wr {
    struct ibv_recv_wr ibwr;
    struct ibv_sge     sg;
} uct_ib_recv_wr_t;

/**
 * prepare a list of n work requests that can be passed to
 * ibv_post_recv()
 *
 * @return number of prepared wrs
 */
int uct_ib_iface_prepare_rx_wrs(uct_ib_iface_t *iface, ucs_mpool_t *mp,
                                uct_ib_recv_wr_t *wrs, unsigned n);

void uct_ib_iface_fill_ah_attr(uct_ib_iface_t *iface, const uct_ib_address_t *ib_addr,
                               uint8_t src_path_bits, struct ibv_ah_attr *ah_attr);

ucs_status_t uct_ib_iface_create_ah(uct_ib_iface_t *iface,
                                    const uct_ib_address_t *ib_addr,
                                    uint8_t src_path_bits,
                                    struct ibv_ah **ah_p);

ucs_status_t uct_ib_iface_wakeup_open(uct_iface_h iface, unsigned events,
                                      uct_wakeup_h wakeup);

ucs_status_t uct_ib_iface_wakeup_get_fd(uct_wakeup_h wakeup, int *fd_p);

ucs_status_t uct_ib_iface_wakeup_arm(uct_wakeup_h wakeup);

ucs_status_t uct_ib_iface_wakeup_wait(uct_wakeup_h wakeup);

ucs_status_t uct_ib_iface_wakeup_signal(uct_wakeup_h wakeup);

void uct_ib_iface_wakeup_close(uct_wakeup_h wakeup);

ucs_status_t uct_ib_iface_arm_tx_cq(uct_ib_iface_t *iface);

ucs_status_t uct_ib_iface_arm_rx_cq(uct_ib_iface_t *iface, int solicited);

#define UCT_IB_IFACE_VERBS_FOREACH_RXWQE(_iface, _i, _hdr, _wc, _wc_count) \
    for (_i = 0; _i < _wc_count && ({ \
        if (ucs_unlikely(_wc[i].status != IBV_WC_SUCCESS)) { \
            ucs_fatal("Receive completion with error: %s", ibv_wc_status_str(_wc[i].status)); \
        } \
        _hdr = (typeof(_hdr))uct_ib_iface_recv_desc_hdr(_iface, \
                                                      (uct_ib_iface_recv_desc_t *)(uintptr_t)_wc[i].wr_id); \
        VALGRIND_MAKE_MEM_DEFINED(_hdr, _wc[i].byte_len); \
        UCS_INSTRUMENT_RECORD(UCS_INSTRUMENT_TYPE_IB_RX, __FUNCTION__, \
                              _wc[i].wr_id, _wc[i].status); \
               1; }); ++_i) 

#define UCT_IB_IFACE_VERBS_FOREACH_TXWQE(_iface, _i, _wc, _wc_count) \
    for (_i = 0; _i < _wc_count && ({ \
        if (ucs_unlikely(_wc[i].status != IBV_WC_SUCCESS)) { \
            ucs_fatal("iface=%p: send completion %d with error: %s wqe: %p wr_id: %llu", \
                      _iface, _i, ibv_wc_status_str(_wc[i].status), \
                      &_wc[i], (unsigned long long)_wc[i].wr_id); \
        } \
               1; }); ++_i) 

#endif
