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
#include <uct/base/uct_iov.inl>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <ucs/sys/math.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/datastruct/string_buffer.h>


#define UCT_IB_MAX_IOV                     8UL
#define UCT_IB_IFACE_NULL_RES_DOMAIN_KEY   0u
#define UCT_IB_MAX_ATOMIC_SIZE             sizeof(uint64_t)
#define UCT_IB_ADDRESS_INVALID_GID_INDEX   UINT8_MAX
#define UCT_IB_ADDRESS_INVALID_PATH_MTU    ((enum ibv_mtu)0)
#define UCT_IB_ADDRESS_INVALID_PKEY        0
#define UCT_IB_ADDRESS_DEFAULT_PKEY        0xffff
#define UCT_IB_SL_MAX                      15
#define UCT_IB_SL_INVALID                  (UCT_IB_SL_MAX + 1)

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
#elif HAVE_DC_DV
    UCT_IB_QPT_DCI = IBV_QPT_DRIVER,
#endif
};


/**
 * IB address packing flags
 */
enum {
    UCT_IB_ADDRESS_PACK_FLAG_ETH           = UCS_BIT(0),
    UCT_IB_ADDRESS_PACK_FLAG_INTERFACE_ID  = UCS_BIT(1),
    UCT_IB_ADDRESS_PACK_FLAG_SUBNET_PREFIX = UCS_BIT(2),
    UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU      = UCS_BIT(3),
    UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX     = UCS_BIT(4),
    UCT_IB_ADDRESS_PACK_FLAG_PKEY          = UCS_BIT(5)
};


typedef struct uct_ib_address_pack_params {
    /* Packing flags, UCT_IB_ADDRESS_PACK_FLAG_xx. */
    uint64_t                          flags;
    /* GID address to pack/unpack. */
    union ibv_gid                     gid;
    /* LID address to pack/unpack. */
    uint16_t                          lid;
    /* RoCE version to pack/unpack in case of an Ethernet link layer,
       must be valid if @ref UCT_IB_ADDRESS_PACK_FLAG_ETH is set. */
    uct_ib_roce_version_info_t        roce_info;
    /* path MTU size as defined in enum ibv_mtu,
       must be valid if @ref UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU is set. */
    enum ibv_mtu                      path_mtu;
    /* GID index,
       must be valid if @ref UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX is set. */
    uint8_t                           gid_index;
    /* PKEY value,
       must be valid if @ref UCT_IB_ADDRESS_PACK_FLAG_PKEY is set. */
    uint16_t                          pkey;
} uct_ib_address_pack_params_t;


struct uct_ib_iface_config {
    uct_iface_config_t      super;

    size_t                  seg_size;      /* Maximal size of copy-out sends */

    struct {
        unsigned            queue_len;       /* Queue length */
        unsigned            max_batch;       /* How many fragments can be batched to one post send */
        unsigned            max_poll;        /* How many wcs can be picked when polling tx cq */
        size_t              min_inline;      /* Inline space to reserve for sends */
        unsigned            min_sge;         /* How many SG entries to support */
        uct_iface_mpool_config_t mp;

        /* Event moderation parameters */
        unsigned            cq_moderation_count;
        double              cq_moderation_period;
    } tx;

    struct {
        unsigned            queue_len;       /* Queue length */
        unsigned            max_batch;       /* How many buffers can be batched to one post receive */
        unsigned            max_poll;        /* How many wcs can be picked when polling rx cq */
        uct_iface_mpool_config_t mp;

        /* Event moderation parameters */
        unsigned            cq_moderation_count;
        double              cq_moderation_period;
    } rx;

    /* Inline space to reserve in CQ */
    size_t                  inl[UCT_IB_DIR_NUM];

    /* Change the address type */
    int                     addr_type;

    /* Force global routing */
    int                     is_global;

    /* IB SL to use (default: AUTO) */
    unsigned long           sl;

    /* IB Traffic Class to use */
    unsigned long           traffic_class;

    /* IB hop limit / TTL */
    unsigned                hop_limit;

    /* Number of paths to expose for the interface  */
    unsigned long           num_paths;

    /* Multiplier for RoCE LAG UDP source port calculation */
    unsigned                roce_path_factor;

    /* Ranges of path bits */
    UCS_CONFIG_ARRAY_FIELD(ucs_range_spec_t, ranges) lid_path_bits;

    /* IB PKEY to use */
    unsigned                pkey;

    /* Multiple resource domains */
    int                     enable_res_domain;

    /* Path MTU size */
    uct_ib_mtu_t            path_mtu;
};


enum {
    UCT_IB_CQ_IGNORE_OVERRUN         = UCS_BIT(0),
    UCT_IB_TM_SUPPORTED              = UCS_BIT(1)
};


typedef struct uct_ib_iface_init_attr {
    unsigned    rx_priv_len;            /* Length of transport private data to reserve */
    unsigned    rx_hdr_len;             /* Length of transport network header */
    unsigned    cq_len[UCT_IB_DIR_NUM]; /* CQ length */
    size_t      seg_size;               /* Transport segment size */
    unsigned    fc_req_size;            /* Flow control request size */
    int         qp_type;                /* IB QP type */
    int         flags;                  /* Various flags (see enum) */
} uct_ib_iface_init_attr_t;


typedef struct uct_ib_qp_attr {
    int                         qp_type;
    struct ibv_qp_cap           cap;
    int                         port;
    struct ibv_srq              *srq;
    uint32_t                    srq_num;
    unsigned                    sq_sig_all;
    unsigned                    max_inl_cqe[UCT_IB_DIR_NUM];
#if HAVE_DECL_IBV_EXP_CREATE_QP
    struct ibv_exp_qp_init_attr ibv;
#elif HAVE_DECL_IBV_CREATE_QP_EX
    struct ibv_qp_init_attr_ex  ibv;
#else
    struct ibv_qp_init_attr     ibv;
#endif
} uct_ib_qp_attr_t;


typedef ucs_status_t (*uct_ib_iface_create_cq_func_t)(uct_ib_iface_t *iface,
                                                      uct_ib_dir_t dir,
                                                      const uct_ib_iface_init_attr_t *init_attr,
                                                      int preferred_cpu,
                                                      size_t inl);

typedef ucs_status_t (*uct_ib_iface_arm_cq_func_t)(uct_ib_iface_t *iface,
                                                   uct_ib_dir_t dir,
                                                   int solicited_only);

typedef void (*uct_ib_iface_event_cq_func_t)(uct_ib_iface_t *iface,
                                             uct_ib_dir_t dir);

typedef void (*uct_ib_iface_handle_failure_func_t)(uct_ib_iface_t *iface, void *arg,
                                                   ucs_status_t status);

typedef ucs_status_t (*uct_ib_iface_set_ep_failed_func_t)(uct_ib_iface_t *iface, uct_ep_h ep,
                                                          ucs_status_t status);


struct uct_ib_iface_ops {
    uct_iface_ops_t                    super;
    uct_ib_iface_create_cq_func_t      create_cq;
    uct_ib_iface_arm_cq_func_t         arm_cq;
    uct_ib_iface_event_cq_func_t       event_cq;
    uct_ib_iface_handle_failure_func_t handle_failure;
    uct_ib_iface_set_ep_failed_func_t  set_ep_failed;
};


struct uct_ib_iface {
    uct_base_iface_t          super;

    struct ibv_cq             *cq[UCT_IB_DIR_NUM];
    struct ibv_comp_channel   *comp_channel;
    uct_recv_desc_t           release_desc;

    uint8_t                   *path_bits;
    unsigned                  path_bits_count;
    unsigned                  num_paths;
    uint16_t                  pkey_index;
    uint16_t                  pkey;
    uint8_t                   addr_size;
    uct_ib_device_gid_info_t  gid_info;

    struct {
        unsigned              rx_payload_offset;   /* offset from desc to payload */
        unsigned              rx_hdr_offset;       /* offset from desc to network header */
        unsigned              rx_headroom_offset;  /* offset from desc to user headroom */
        unsigned              rx_max_batch;
        unsigned              rx_max_poll;
        unsigned              tx_max_poll;
        unsigned              seg_size;
        unsigned              roce_path_factor;
        uint8_t               max_inl_cqe[UCT_IB_DIR_NUM];
        uint8_t               port_num;
        uint8_t               sl;
        uint8_t               traffic_class;
        uint8_t               hop_limit;
        uint8_t               enable_res_domain;   /* Disable multiple resource domains */
        uint8_t               qp_type;
        uint8_t               force_global_addr;
        enum ibv_mtu          path_mtu;
    } config;

    uct_ib_iface_ops_t        *ops;
};


typedef struct uct_ib_fence_info {
    uint16_t                    fence_beat; /* 16bit is enough because if it wraps around,
                                             * it means the older ops are already completed
                                             * because QP size is less than 64k */
} uct_ib_fence_info_t;


UCS_CLASS_DECLARE(uct_ib_iface_t, uct_ib_iface_ops_t*, uct_md_h, uct_worker_h,
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
 * @return Whether the port used by this interface is IB
 */
int uct_ib_iface_is_ib(uct_ib_iface_t *iface);


/**
 * Get the expected size of IB packed address.
 *
 * @param [in]  params   Address parameters as defined in
 *                       @ref uct_ib_address_pack_params_t.
 *
 * @return IB address size of the given link scope.
 */
size_t uct_ib_address_size(const uct_ib_address_pack_params_t *params);


/**
 * @return IB address packing flags of the given iface.
 */
unsigned uct_ib_iface_address_pack_flags(uct_ib_iface_t *iface);


/**
 * @return IB address size of the given iface.
 */
size_t uct_ib_iface_address_size(uct_ib_iface_t *iface);


/**
 * Pack IB address.
 *
 * @param [in]     params   Address parameters as defined in
 *                          @ref uct_ib_address_pack_params_t.
 * @param [in/out] ib_addr  Filled with packed ib address. Size of the structure
 *                          must be at least what @ref uct_ib_address_size()
 *                          returns for the given scope.
 */
void uct_ib_address_pack(const uct_ib_address_pack_params_t *params,
                         uct_ib_address_t *ib_addr);



/**
 * Pack the IB address of the given iface.
 *
 * @param [in]  iface      Iface whose IB address to pack.
 * @param [in/out] ib_addr Filled with packed ib address. Size of the structure
 *                         must be at least what @ref uct_ib_address_size()
 *                         returns for the given scope.
 */
void uct_ib_iface_address_pack(uct_ib_iface_t *iface, uct_ib_address_t *ib_addr);


/**
 * Unpack IB address.
 *
 * @param [in]  ib_addr    IB address to unpack.
 * @param [out] params_p   Filled with address attributes as in
 *                         @ref uct_ib_address_pack_params_t.
 */
void uct_ib_address_unpack(const uct_ib_address_t *ib_addr,
                           uct_ib_address_pack_params_t *params_p);


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


int uct_ib_iface_is_roce_v2(uct_ib_iface_t *iface, uct_ib_device_t *dev);


/**
 * Select the IB gid index and RoCE version to use for a RoCE port.
 *
 * @param iface                 IB interface
 * @param md_config_index       Gid index from the md configuration.
 */
ucs_status_t uct_ib_iface_init_roce_gid_info(uct_ib_iface_t *iface,
                                             size_t md_config_index);


static inline uct_ib_md_t* uct_ib_iface_md(uct_ib_iface_t *iface)
{
    return ucs_derived_of(iface->super.md, uct_ib_md_t);
}

static inline uct_ib_device_t* uct_ib_iface_device(uct_ib_iface_t *iface)
{
    return &uct_ib_iface_md(iface)->dev;
}

static inline struct ibv_port_attr* uct_ib_iface_port_attr(uct_ib_iface_t *iface)
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

void uct_ib_iface_fill_ah_attr_from_gid_lid(uct_ib_iface_t *iface, uint16_t lid,
                                            const union ibv_gid *gid,
                                            uint8_t gid_index,
                                            unsigned path_index,
                                            struct ibv_ah_attr *ah_attr);

void uct_ib_iface_fill_ah_attr_from_addr(uct_ib_iface_t *iface,
                                         const uct_ib_address_t *ib_addr,
                                         unsigned path_index,
                                         struct ibv_ah_attr *ah_attr,
                                         enum ibv_mtu *path_mtu);

ucs_status_t uct_ib_iface_pre_arm(uct_ib_iface_t *iface);

ucs_status_t uct_ib_iface_event_fd_get(uct_iface_h iface, int *fd_p);

ucs_status_t uct_ib_iface_arm_cq(uct_ib_iface_t *iface,
                                 uct_ib_dir_t dir,
                                 int solicited_only);

ucs_status_t uct_ib_verbs_create_cq(uct_ib_iface_t *iface, uct_ib_dir_t dir,
                                    const uct_ib_iface_init_attr_t *init_attr,
                                    int preferred_cpu, size_t inl);

ucs_status_t uct_ib_iface_create_qp(uct_ib_iface_t *iface,
                                    uct_ib_qp_attr_t *attr,
                                    struct ibv_qp **qp_p);

void uct_ib_iface_fill_attr(uct_ib_iface_t *iface,
                            uct_ib_qp_attr_t *attr);

uint8_t uct_ib_iface_config_select_sl(const uct_ib_iface_config_t *ib_config);


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
            sge[sge_it].lkey = uct_ib_memh_get_lkey(iov[iov_it].memh);
        }
        ++sge_it;
    }

    return sge_it;
}

static UCS_F_ALWAYS_INLINE
size_t uct_ib_iface_hdr_size(size_t max_inline, size_t min_size)
{
    return (size_t)ucs_max((ssize_t)(max_inline - min_size), 0);
}

static UCS_F_ALWAYS_INLINE void
uct_ib_fence_info_init(uct_ib_fence_info_t* fence)
{
    fence->fence_beat = 0;
}

static UCS_F_ALWAYS_INLINE
ucs_log_level_t uct_ib_iface_failure_log_level(uct_ib_iface_t *ib_iface,
                                               ucs_status_t err_handler_status,
                                               ucs_status_t status)
{
    if (err_handler_status != UCS_OK) {
        return UCS_LOG_LEVEL_FATAL;
    } else if ((status == UCS_ERR_ENDPOINT_TIMEOUT) ||
               (status == UCS_ERR_CONNECTION_RESET)) {
        return ib_iface->super.config.failure_level;
    } else {
        return UCS_LOG_LEVEL_ERROR;
    }
}

#endif
