/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_IFACE_H
#define UCT_IB_IFACE_H

#include "ib_md.h"

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/sys/compiler.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/sys/math.h>

#define UCT_IB_MAX_IOV                     8UL
#define UCT_IB_IFACE_NULL_RES_DOMAIN_KEY   0u
#define UCT_IB_MAX_ATOMIC_SIZE             sizeof(uint64_t)


/* Forward declarations */
typedef struct uct_ib_iface_config   uct_ib_iface_config_t;
typedef struct uct_ib_iface_ops      uct_ib_iface_ops_t;
typedef struct uct_ib_iface          uct_ib_iface_t;


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


/**
 * Traffic direction.
 */
typedef enum {
    UCT_IB_DIR_RX,
    UCT_IB_DIR_TX,
    UCT_IB_DIR_NUM
} uct_ib_dir_t;

enum {
    UCT_IB_QPT_UNKNOWN,
#ifdef HAVE_DC_EXP
    UCT_IB_QPT_DCI = IBV_EXP_QPT_DC_INI,
#elif defined(HAVE_DC_DV)
    UCT_IB_QPT_DCI = IBV_QPT_DRIVER,
#endif
};

struct uct_ib_iface_config {
    uct_iface_config_t      super;

    struct {
        unsigned            queue_len;       /* Queue length */
        unsigned            max_batch;       /* How many fragments can be batched to one post send */
        unsigned            max_poll;        /* How many wcs can be picked when polling tx cq */
        size_t              min_inline;      /* Inline space to reserve for sends */
        size_t              inl_resp;        /* Inline space to reserve for responses */
        unsigned            min_sge;         /* How many SG entries to support */
        unsigned            cq_moderation;   /* How many TX messages are batched to one CQE */
        uct_iface_mpool_config_t mp;

        /* Event moderation parameters */
        unsigned            cq_moderation_count;
        double              cq_moderation_period;
    } tx;

    struct {
        unsigned            queue_len;       /* Queue length */
        unsigned            max_batch;       /* How many buffers can be batched to one post receive */
        unsigned            max_poll;        /* How many wcs can be picked when polling rx cq */
        size_t              inl;             /* Inline space to reserve in CQ/QP */
        uct_iface_mpool_config_t mp;

        /* Event moderation parameters */
        unsigned            cq_moderation_count;
        double              cq_moderation_period;
    } rx;

    /* Change the address type */
    int                     addr_type;

    /* Forice global routing */
    int                     is_global;

    /* IB SL to use */
    unsigned                sl;

    /* IB Traffic Class to use */
    unsigned long           traffic_class;

    /* IB hop limit / TTL */
    unsigned                hop_limit;

    /* Ranges of path bits */
    UCS_CONFIG_ARRAY_FIELD(ucs_range_spec_t, ranges) lid_path_bits;

    /* IB PKEY to use */
    unsigned                pkey_value;

    /* Multiple resource domains */
    int                     enable_res_domain;
};


typedef struct uct_ib_qp_attr {
    int                         qp_type;
    struct ibv_qp_cap           cap;
    struct ibv_srq              *srq;
    unsigned                    sq_sig_all;
    unsigned                    max_inl_recv;
#if HAVE_DECL_IBV_EXP_CREATE_QP
    struct ibv_exp_qp_init_attr ibv;
#elif HAVE_DECL_IBV_CREATE_QP_EX
    struct ibv_qp_init_attr_ex  ibv;
#else
    struct ibv_qp_init_attr     ibv;
#endif
} uct_ib_qp_attr_t;


struct uct_ib_iface_ops {
    uct_iface_ops_t         super;
    ucs_status_t            (*create_cq)(struct ibv_context *context, int cqe,
                                         struct ibv_comp_channel *channel,
                                         int comp_vector, int ignore_overrun,
                                         size_t *inl, struct ibv_cq **cq_p);
    ucs_status_t            (*arm_cq)(uct_ib_iface_t *iface,
                                      uct_ib_dir_t dir,
                                      int solicited_only);
    void                    (*event_cq)(uct_ib_iface_t *iface,
                                        uct_ib_dir_t dir);
    void                    (*handle_failure)(uct_ib_iface_t *iface, void *arg,
                                              ucs_status_t status);
    ucs_status_t            (*set_ep_failed)(uct_ib_iface_t *iface, uct_ep_h ep,
                                             ucs_status_t status);
    ucs_status_t            (*create_qp)(uct_ib_iface_t *iface, uct_ib_qp_attr_t *attr,
                                         struct ibv_qp **qp_p);
    ucs_status_t            (*init_res_domain)(uct_ib_iface_t *iface);
    void                    (*cleanup_res_domain)(uct_ib_iface_t *iface);
};


struct uct_ib_iface {
    uct_base_iface_t        super;

    struct ibv_cq           *cq[UCT_IB_DIR_NUM];
    struct ibv_comp_channel *comp_channel;
    uct_recv_desc_t         release_desc;

    uint8_t                 *path_bits;
    unsigned                path_bits_count;
    uint16_t                pkey_index;
    uint16_t                pkey_value;
    uint8_t                 is_global_addr;
    uint8_t                 addr_size;
    union ibv_gid           gid;

    struct {
        unsigned            rx_payload_offset;   /* offset from desc to payload */
        unsigned            rx_hdr_offset;       /* offset from desc to network header */
        unsigned            rx_headroom_offset;  /* offset from desc to user headroom */
        unsigned            rx_max_batch;
        unsigned            rx_max_poll;
        unsigned            tx_max_poll;
        unsigned            seg_size;
        uint8_t             max_inl_resp;
        uint8_t             port_num;
        uint8_t             sl;
        uint8_t             traffic_class;
        uint8_t             hop_limit;
        uint8_t             gid_index;           /* IB GID index to use  */
        uint8_t             enable_res_domain;   /* Disable multiple resource domains */
        uint8_t             qp_type;
        size_t              max_iov;             /* Maximum buffers in IOV array */
    } config;

    uct_ib_iface_ops_t      *ops;
};

enum {
    UCT_IB_CQ_IGNORE_OVERRUN         = UCS_BIT(0),
};

typedef struct uct_ib_iface_init_attr {
    unsigned    rx_priv_len;     /* Length of transport private data to reserve */
    unsigned    rx_hdr_len;      /* Length of transport network header */
    unsigned    tx_cq_len;       /* Send CQ length */
    unsigned    rx_cq_len;       /* Receive CQ length */
    size_t      seg_size;        /* Transport segment size */
    int         tm_cap_bit;      /* Required HW tag-matching capabilities */
    unsigned    fc_req_size;     /* Flow control request size */
    int         qp_type;         /* IB QP type */
    int         flags;           /* Various flags (see enum) */
} uct_ib_iface_init_attr_t;

UCS_CLASS_DECLARE(uct_ib_iface_t);
UCS_CLASS_DECLARE_INIT_FUNC(uct_ib_iface_t, uct_ib_iface_ops_t*, uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, const uct_ib_iface_config_t*,
                  const uct_ib_iface_init_attr_t*);

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
 *               |   am_callback/tag_unexp_callback
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
 *            am_callback/tag_unexp_callback
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
                                          const uct_ib_iface_config_t *config,
                                          const char *name, ucs_mpool_t *mp);

void uct_ib_iface_release_desc(uct_recv_desc_t *self, void *desc);


static UCS_F_ALWAYS_INLINE void
uct_ib_iface_invoke_am_desc(uct_ib_iface_t *iface, uint8_t am_id, void *data,
                            unsigned length, uct_ib_iface_recv_desc_t *ib_desc)
{
    void *desc = (char*)ib_desc + iface->config.rx_headroom_offset;
    ucs_status_t status;

    status = uct_iface_invoke_am(&iface->super, am_id, data, length,
                                 UCT_CB_PARAM_FLAG_DESC);
    if (status == UCS_OK) {
        ucs_mpool_put_inline(ib_desc);
    } else {
        uct_recv_desc(desc) = &iface->release_desc;
    }
}


/**
 * @return Whether the port used by this interface is RoCE
 */
int uct_ib_iface_is_roce(uct_ib_iface_t *iface);


/**
 * @return IB address size of the given link scope.
 */
size_t uct_ib_address_size(uct_ib_iface_t *iface);


/**
 * Pack IB address.
 *
 * @param [in]  dev        IB device. TODO remove this.
 * @param [in]  gid        GID address to pack.
 * @param [in]  lid        LID address to pack.
 * @param [out] ib_addr    Filled with packed ib address. Size of the structure
 *                         must be at least what @ref uct_ib_address_size() returns
 *                         for the given scope.
 */
void uct_ib_address_pack(uct_ib_iface_t *iface,
                         const union ibv_gid *gid, uint16_t lid,
                         uct_ib_address_t *ib_addr);


/**
 * Unpack IB address.
 *
 * @param [in]  ib_addr    IB address to unpack.
 * @param [out] lid        Filled with address LID, or 0 if not present.
 */
void uct_ib_address_unpack(const uct_ib_address_t *ib_addr, uint16_t *lid,
                           union ibv_gid *gid);


/**
 * Convert IB address to a human-readable string.
 */
const char *uct_ib_address_str(const uct_ib_address_t *ib_addr, char *buf,
                               size_t max);

ucs_status_t uct_ib_iface_get_device_address(uct_iface_h tl_iface,
                                             uct_device_addr_t *dev_addr);

int uct_ib_iface_is_reachable(const uct_iface_h tl_iface, const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr);

/*
 * @param xport_hdr_len       How many bytes this transport adds on top of IB header (LRH+BTH+iCRC+vCRC)
 */
ucs_status_t uct_ib_iface_query(uct_ib_iface_t *iface, size_t xport_hdr_len,
                                uct_iface_attr_t *iface_attr);

static inline uct_ib_md_t* uct_ib_iface_md(uct_ib_iface_t *iface)
{
    return ucs_derived_of(iface->super.md, uct_ib_md_t);
}

static inline uct_ib_device_t* uct_ib_iface_device(uct_ib_iface_t *iface)
{
    return &uct_ib_iface_md(iface)->dev;
}

static inline struct ibv_exp_port_attr* uct_ib_iface_port_attr(uct_ib_iface_t *iface)
{
    return uct_ib_device_port_attr(uct_ib_iface_device(iface), iface->config.port_num);
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

ucs_status_t uct_ib_iface_create_ah(uct_ib_iface_t *iface,
                                    struct ibv_ah_attr *ah_attr,
                                    struct ibv_ah **ah_p);

ucs_status_t uct_ib_iface_pre_arm(uct_ib_iface_t *iface);

ucs_status_t uct_ib_iface_event_fd_get(uct_iface_h iface, int *fd_p);

ucs_status_t uct_ib_iface_arm_cq(uct_ib_iface_t *iface,
                                 uct_ib_dir_t dir,
                                 int solicited_only);

static inline uint8_t uct_ib_iface_get_atomic_mr_id(uct_ib_iface_t *iface)
{
    return uct_ib_md_get_atomic_mr_id(ucs_derived_of(iface->super.md, uct_ib_md_t));
}

ucs_status_t uct_ib_verbs_create_cq(struct ibv_context *context, int cqe,
                                    struct ibv_comp_channel *channel,
                                    int comp_vector, int ignore_overrun,
                                    size_t *inl, struct ibv_cq **cq_p);

ucs_status_t uct_ib_iface_create_qp(uct_ib_iface_t *iface,
                                    uct_ib_qp_attr_t *attr,
                                    struct ibv_qp **qp_p);

void uct_ib_iface_fill_attr(uct_ib_iface_t *iface,
                            uct_ib_qp_attr_t *attr);


#define UCT_IB_IFACE_FMT \
    "%s:%d"
#define UCT_IB_IFACE_ARG(_iface) \
    uct_ib_device_name(uct_ib_iface_device(_iface)), (_iface)->config.port_num


#define UCT_IB_IFACE_VERBS_COMPLETION_ERR(_type, _iface, _i,  _wc) \
    ucs_fatal("%s completion[%d] with error on %s/%p: %s, vendor_err 0x%x wr_id 0x%lx", \
              _type, _i, uct_ib_device_name(uct_ib_iface_device(_iface)), _iface, \
              uct_ib_wc_status_str(_wc[i].status), _wc[i].vendor_err, \
              _wc[i].wr_id);

#define UCT_IB_IFACE_VERBS_FOREACH_RXWQE(_iface, _i, _hdr, _wc, _wc_count) \
    for (_i = 0; _i < _wc_count && ({ \
        if (ucs_unlikely(_wc[i].status != IBV_WC_SUCCESS)) { \
            UCT_IB_IFACE_VERBS_COMPLETION_ERR("receive", _iface, _i, _wc); \
        } \
        _hdr = (typeof(_hdr))uct_ib_iface_recv_desc_hdr(_iface, \
                                                      (uct_ib_iface_recv_desc_t *)(uintptr_t)_wc[i].wr_id); \
        VALGRIND_MAKE_MEM_DEFINED(_hdr, _wc[i].byte_len); \
               1; }); ++_i)

#define UCT_IB_MAX_ZCOPY_LOG_SGE(_iface) \
    (uct_ib_iface_device(_iface)->max_zcopy_log_sge)

/**
 * Fill ibv_sge data structure by data provided in uct_iov_t
 * The function avoids copying IOVs with zero length
 *
 * @return Number of elements in sge[]
 */
static UCS_F_ALWAYS_INLINE
size_t uct_ib_verbs_sge_fill_iov(struct ibv_sge *sge, const uct_iov_t *iov,
                                 size_t iovcnt)
{
    size_t iov_it, sge_it = 0;

    for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
        sge[sge_it].length = uct_iov_get_length(&iov[iov_it]);
        if (sge[sge_it].length > 0) {
            sge[sge_it].addr   = (uintptr_t)(iov[iov_it].buffer);
        } else {
            continue; /* to avoid zero length elements in sge */
        }

        if (iov[sge_it].memh == UCT_MEM_HANDLE_NULL) {
            sge[sge_it].lkey = 0;
        } else {
            sge[sge_it].lkey = ((uct_ib_mem_t *)(iov[iov_it].memh))->lkey;
        }
        ++sge_it;
    }

    return sge_it;
}


static UCS_F_ALWAYS_INLINE
size_t uct_ib_iface_get_max_iov(uct_ib_iface_t *iface)
{
    return iface->config.max_iov;
}


static UCS_F_ALWAYS_INLINE
void uct_ib_iface_set_max_iov(uct_ib_iface_t *iface, size_t max_iov)
{
    size_t min_iov_requested;

    ucs_assert((ssize_t)max_iov > 0);

    min_iov_requested = ucs_max(max_iov, 1UL); /* max_iov mustn't be 0 */
    iface->config.max_iov = ucs_min(UCT_IB_MAX_IOV, min_iov_requested);
}


static UCS_F_ALWAYS_INLINE
void uct_ib_iface_fill_ah_attr_from_gid_lid(uct_ib_iface_t *iface, uint16_t lid,
                                            const union ibv_gid *gid,
                                            uint8_t path_bits,
                                            struct ibv_ah_attr *ah_attr)
{
    memset(ah_attr, 0, sizeof(*ah_attr));

    ah_attr->sl                = iface->config.sl;
    ah_attr->src_path_bits     = path_bits;
    ah_attr->dlid              = lid | path_bits;
    ah_attr->port_num          = iface->config.port_num;
    ah_attr->grh.traffic_class = iface->config.traffic_class;

    if (iface->is_global_addr ||
        (iface->gid.global.subnet_prefix != gid->global.subnet_prefix)) {
        ucs_assert_always(gid->global.interface_id != 0);
        ah_attr->is_global      = 1;
        ah_attr->grh.dgid       = *gid;
        ah_attr->grh.sgid_index = iface->config.gid_index;
        ah_attr->grh.hop_limit  = iface->config.hop_limit;
    } else {
        ah_attr->is_global      = 0;
    }
}

static UCS_F_ALWAYS_INLINE
void uct_ib_iface_fill_ah_attr_from_addr(uct_ib_iface_t *iface,
                                         const uct_ib_address_t *ib_addr,
                                         uint8_t path_bits,
                                         struct ibv_ah_attr *ah_attr)
{
    union ibv_gid  gid;
    uint16_t       lid;

    uct_ib_address_unpack(ib_addr, &lid, &gid);

    uct_ib_iface_fill_ah_attr_from_gid_lid(iface, lid, &gid, path_bits, ah_attr);
}

static UCS_F_ALWAYS_INLINE
size_t uct_ib_iface_hdr_size(size_t max_inline, size_t min_size)
{
    return (size_t)ucs_max((ssize_t)(max_inline - min_size), 0);
}

#endif
