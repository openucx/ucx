/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_MLX5_H_
#define UCT_IB_MLX5_H_


#include <uct/base/uct_worker.h>
#include <uct/ib/base/ib_log.h>
#include <uct/ib/base/ib_device.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>
#include <ucs/type/status.h>

/**
 * When using a clang version that is higher than 3.0, the GNUC_MINOR is set
 * to 2, which affects the offset of several fields that are used by UCX from
 * the liblmlx5 library (from the mlx5_qp struct).
 * According to libmlx5, resetting the GNUC_MINOR version to 3, will make the
 * offset of these fields inside libmlx5 (when compiled with GCC) the same as
 * the one used by UCX (when compiled with clang).
 */
#ifdef __clang__
#  define CLANG_VERSION ( __clang_major__ * 100 + __clang_minor__)
#  if CLANG_VERSION >= 300
#    undef __GNUC_MINOR__
#    define __GNUC_MINOR__ 3
#  endif
#endif

#if HAVE_INFINIBAND_MLX5DV_H
#  include <infiniband/mlx5dv.h>
#else
#  include <infiniband/mlx5_hw.h>
#  include <uct/ib/mlx5/exp/ib_mlx5_hw.h>
#endif
#include <uct/ib/mlx5/dv/ib_mlx5_dv.h>

#include <netinet/in.h>
#include <endian.h>
#include <string.h>


#define UCT_IB_MLX5_WQE_SEG_SIZE        16 /* Size of a segment in a WQE */
#define UCT_IB_MLX5_CQE64_MAX_INL       32 /* Inline scatter size in 64-byte CQE */
#define UCT_IB_MLX5_CQE128_MAX_INL      64 /* Inline scatter size in 128-byte CQE */
#define UCT_IB_MLX5_CQE64_SIZE_LOG      6
#define UCT_IB_MLX5_CQE128_SIZE_LOG     7
#define UCT_IB_MLX5_MAX_BB              4
#define UCT_IB_MLX5_WORKER_BF_KEY       0x00c1b7e8u
#define UCT_IB_MLX5_DEVX_UAR_KEY        0xdea1ab1eU
#define UCT_IB_MLX5_RES_DOMAIN_KEY      0x1b1bda7aU
#define UCT_IB_MLX5_WORKER_DM_KEY       0xacdf1245u
#define UCT_IB_MLX5_EXTENDED_UD_AV      0x80 /* htonl(0x80000000) */
#define UCT_IB_MLX5_AV_GRH_PRESENT      0x40 /* htonl(UCS_BIT(30)) */
#define UCT_IB_MLX5_BF_REG_SIZE         256
#define UCT_IB_MLX5_CQE_VENDOR_SYND_ODP 0x93
#define UCT_IB_MLX5_CQE_VENDOR_SYND_PSN 0x99
#define UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK 0x80
#define UCT_IB_MLX5_MAX_SEND_WQE_SIZE   (UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB)
#define UCT_IB_MLX5_CQ_SET_CI           0
#define UCT_IB_MLX5_CQ_ARM_DB           1
#define UCT_IB_MLX5_LOG_MAX_MSG_SIZE    30
#define UCT_IB_MLX5_ATOMIC_MODE         3
#define UCT_IB_MLX5_CQE_FLAG_L3_IN_DATA UCS_BIT(28) /* GRH/IP in the receive buffer */
#define UCT_IB_MLX5_CQE_FLAG_L3_IN_CQE  UCS_BIT(29) /* GRH/IP in the CQE */
#define UCT_IB_MLX5_MP_RQ_BYTE_CNT_MASK 0x0000FFFF  /* Byte count mask for multi-packet RQs */
#define UCT_IB_MLX5_MP_RQ_LAST_MSG_FLAG UCS_BIT(30) /* MP last packet indication */
#define UCT_IB_MLX5_MP_RQ_FILLER_FLAG   UCS_BIT(31) /* Filler CQE indicator */

#if HAVE_DECL_MLX5DV_UAR_ALLOC_TYPE_BF
#  define UCT_IB_MLX5_UAR_ALLOC_TYPE_WC MLX5DV_UAR_ALLOC_TYPE_BF
#else
#  define UCT_IB_MLX5_UAR_ALLOC_TYPE_WC 0
#endif

#if HAVE_DECL_MLX5DV_UAR_ALLOC_TYPE_NC
#  define UCT_IB_MLX5_UAR_ALLOC_TYPE_NC MLX5DV_UAR_ALLOC_TYPE_NC
#endif

#define UCT_IB_MLX5_OPMOD_EXT_ATOMIC(_log_arg_size) \
    ((8) | ((_log_arg_size) - 2))

#ifdef HAVE_STRUCT_MLX5_WQE_AV_BASE

#  define mlx5_av_base(_av)         (&(_av)->base)
#  define mlx5_av_grh(_av)          (&(_av)->grh_sec)
#  define UCT_IB_MLX5_AV_BASE_SIZE  sizeof(struct mlx5_base_av)
#  define UCT_IB_MLX5_AV_FULL_SIZE  sizeof(struct mlx5_wqe_av)

#else

#  define mlx5_av_base(_av)         (_av)
/* do not use direct cast from address of reserved0 to avoid compilation warnings */
#  define mlx5_av_grh(_av)          ((struct mlx5_grh_av *)(((char*)(_av)) + \
                                     ucs_offsetof(struct mlx5_wqe_av, reserved0[0])))
#  define UCT_IB_MLX5_AV_BASE_SIZE  ucs_offsetof(struct mlx5_wqe_av, reserved0[0])
#  define UCT_IB_MLX5_AV_FULL_SIZE  sizeof(struct mlx5_wqe_av)

#  define mlx5_base_av              mlx5_wqe_av

struct mlx5_grh_av {
        uint8_t         reserved0[4];
        uint8_t         rmac[6];
        uint8_t         tclass;
        uint8_t         hop_limit;
        uint32_t        grh_gid_fl;
        uint8_t         rgid[16];
};

#  define HAVE_STRUCT_MLX5_GRH_AV_RMAC 1

#endif

#ifndef MLX5_WQE_CTRL_SOLICITED
#  define MLX5_WQE_CTRL_SOLICITED  (1<<1)
#endif

#define UCT_IB_MLX5_WQE_CTRL_FLAG_FENCE        (2<<5)
#define UCT_IB_MLX5_WQE_CTRL_FLAG_STRONG_ORDER (3<<5)

#define UCT_IB_MLX5_AM_ZCOPY_MAX_IOV  3UL

#define UCT_IB_MLX5_AM_MAX_SHORT(_av_size) \
    (UCT_IB_MLX5_MAX_SEND_WQE_SIZE - \
     (sizeof(struct mlx5_wqe_ctrl_seg) + \
      (_av_size) + \
      sizeof(struct mlx5_wqe_inl_data_seg)))

#define UCT_IB_MLX5_SET_BASE_AV(to_base_av, from_base_av) \
    do { \
        (to_base_av)->dqp_dct      = (from_base_av)->dqp_dct; \
        (to_base_av)->stat_rate_sl = (from_base_av)->stat_rate_sl; \
        (to_base_av)->fl_mlid      = (from_base_av)->fl_mlid; \
        (to_base_av)->rlid         = (from_base_av)->rlid; \
    } while (0)

#define UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(_av_size) \
    (UCT_IB_MLX5_AM_MAX_SHORT(_av_size) - \
     UCT_IB_MLX5_AM_ZCOPY_MAX_IOV * sizeof(struct mlx5_wqe_data_seg))

#define UCT_IB_MLX5_PUT_MAX_SHORT(_av_size) \
    (UCT_IB_MLX5_AM_MAX_SHORT(_av_size) - sizeof(struct mlx5_wqe_raddr_seg))

#define UCT_IB_MLX5_XRQ_MIN_UWQ_POST 33

#define UCT_IB_MLX5_MD_FLAGS_DEVX_OBJS(_devx_objs) \
    ((_devx_objs) << UCT_IB_MLX5_MD_FLAG_DEVX_OBJS_SHIFT)

#define UCT_IB_MLX5_MD_FLAG_DEVX_OBJS(_obj) \
    UCT_IB_MLX5_MD_FLAGS_DEVX_OBJS(UCS_BIT(UCT_IB_DEVX_OBJ_ ## _obj))

#define UCT_IB_MLX5_DEVX_EVENT_TYPE_MASK  0xffff
#define UCT_IB_MLX5_DEVX_EVENT_DATA_SHIFT 16

enum {
    /* Device supports KSM */
    UCT_IB_MLX5_MD_FLAG_KSM              = UCS_BIT(0),
    /* Device supports DEVX */
    UCT_IB_MLX5_MD_FLAG_DEVX             = UCS_BIT(1),
    /* Device supports TM DC */
    UCT_IB_MLX5_MD_FLAG_DC_TM            = UCS_BIT(2),
    /* Device supports MP RQ */
    UCT_IB_MLX5_MD_FLAG_MP_RQ            = UCS_BIT(3),
    /* Device supports creation of indirect MR with atomics access rights */
    UCT_IB_MLX5_MD_FLAG_INDIRECT_ATOMICS = UCS_BIT(4),
    /* Device supports RMP to create SRQ for AM */
    UCT_IB_MLX5_MD_FLAG_RMP              = UCS_BIT(5),
    /* Device supports querying bitmask of OOO (AR) states per SL */
    UCT_IB_MLX5_MD_FLAG_OOO_SL_MASK      = UCS_BIT(6),

    /* Object to be created by DevX */
    UCT_IB_MLX5_MD_FLAG_DEVX_OBJS_SHIFT  = 7,
    UCT_IB_MLX5_MD_FLAG_DEVX_RC_QP       = UCT_IB_MLX5_MD_FLAG_DEVX_OBJS(RCQP),
    UCT_IB_MLX5_MD_FLAG_DEVX_RC_SRQ      = UCT_IB_MLX5_MD_FLAG_DEVX_OBJS(RCSRQ),
    UCT_IB_MLX5_MD_FLAG_DEVX_DCT         = UCT_IB_MLX5_MD_FLAG_DEVX_OBJS(DCT),
    UCT_IB_MLX5_MD_FLAG_DEVX_DC_SRQ      = UCT_IB_MLX5_MD_FLAG_DEVX_OBJS(DCSRQ),
};


enum {
    UCT_IB_MLX5_SRQ_TOPO_LIST         = 0x0,
    UCT_IB_MLX5_SRQ_TOPO_CYCLIC       = 0x1,
    UCT_IB_MLX5_SRQ_TOPO_LIST_MP_RQ   = 0x2,
    UCT_IB_MLX5_SRQ_TOPO_CYCLIC_MP_RQ = 0x3
};

#if HAVE_DEVX
typedef struct uct_ib_mlx5_devx_umem {
    struct mlx5dv_devx_umem  *mem;
    size_t                   size;
} uct_ib_mlx5_devx_umem_t;
#endif

/**
 * MLX5 IB memory domain.
 */
typedef struct uct_ib_mlx5_md {
    uct_ib_md_t              super;
    uint32_t                 flags;
    ucs_mpool_t              dbrec_pool;
    ucs_recursive_spinlock_t dbrec_lock;
#if HAVE_EXP_UMR
    struct ibv_qp            *umr_qp;   /* special QP for creating UMR */
    struct ibv_cq            *umr_cq;   /* special CQ for creating UMR */
#endif

#if HAVE_DEVX
    void                     *zero_buf;
    uct_ib_mlx5_devx_umem_t  zero_mem;
#endif
} uct_ib_mlx5_md_t;


typedef enum {
    UCT_IB_MLX5_MMIO_MODE_BF_POST,    /* BF without flush, can be used only from
                                         one thread */
    UCT_IB_MLX5_MMIO_MODE_BF_POST_MT, /* BF with order, can be used by multiple
                                         serialized threads */
    UCT_IB_MLX5_MMIO_MODE_DB,         /* 8-byte doorbell (with the mandatory flush) */
    UCT_IB_MLX5_MMIO_MODE_AUTO,       /* Auto-select according to driver/HW capabilities
                                         and multi-thread support level */
    UCT_IB_MLX5_MMIO_MODE_LAST
} uct_ib_mlx5_mmio_mode_t;


typedef struct uct_ib_mlx5_iface_config {
#if HAVE_IBV_DM
    struct {
        size_t               seg_len;
        unsigned             count;
    } dm;
#endif
    uct_ib_mlx5_mmio_mode_t  mmio_mode;
    ucs_ternary_auto_value_t ar_enable;
} uct_ib_mlx5_iface_config_t;


/**
 * MLX5 DoorBell record
 */
typedef struct uct_ib_mlx5_dbrec {
   volatile uint32_t  db[2];
   uint32_t           mem_id;
   size_t             offset;
   uct_ib_mlx5_md_t   *md;
} uct_ib_mlx5_dbrec_t;


typedef enum {
    UCT_IB_MLX5_OBJ_TYPE_VERBS,
    UCT_IB_MLX5_OBJ_TYPE_DEVX,
    UCT_IB_MLX5_OBJ_TYPE_LAST
} uct_ib_mlx5_obj_type_t;


/* Shared receive queue */
typedef struct uct_ib_mlx5_srq {
    uct_ib_mlx5_obj_type_t             type;
    uint32_t                           srq_num;
    void                               *buf;
    volatile uint32_t                  *db;
    uint16_t                           free_idx;   /* what is completed contiguously */
    uint16_t                           ready_idx;  /* what is ready to be posted to hw */
    uint16_t                           sw_pi;      /* what is posted to hw */
    uint16_t                           mask;
    uint16_t                           tail;       /* tail in the driver */
    uint16_t                           stride;
    union {
        struct {
            struct ibv_srq             *srq;
        } verbs;
#if HAVE_DEVX
        struct {
            uct_ib_mlx5_dbrec_t        *dbrec;
            uct_ib_mlx5_devx_umem_t    mem;
            struct mlx5dv_devx_obj     *obj;
        } devx;
#endif
    };
} uct_ib_mlx5_srq_t;


/* Completion queue */
typedef struct uct_ib_mlx5_cq {
    void               *cq_buf;
    unsigned           cq_ci;
    unsigned           cq_sn;
    unsigned           cq_length;
    unsigned           cqe_size_log;
    unsigned           cq_num;
    void               *uar;
    volatile uint32_t  *dbrec;
} uct_ib_mlx5_cq_t;


/* Blue flame register */
typedef struct uct_ib_mlx5_mmio_reg {
    uct_worker_tl_data_t        super;
    union {
        void                    *ptr;
        uintptr_t               uint;
    } addr;
    uct_ib_mlx5_mmio_mode_t     mode;
} uct_ib_mlx5_mmio_reg_t;


typedef struct uct_ib_mlx5_devx_uar {
    uct_ib_mlx5_mmio_reg_t      super;
#if HAVE_DEVX
    struct mlx5dv_devx_uar      *uar;
#endif
    struct ibv_context          *ctx;
} uct_ib_mlx5_devx_uar_t;


/* resource domain */
typedef struct uct_ib_mlx5_res_domain {
    uct_worker_tl_data_t        super;
#ifdef HAVE_IBV_EXP_RES_DOMAIN
    struct ibv_exp_res_domain   *ibv_domain;
#elif HAVE_DECL_IBV_ALLOC_TD
    struct ibv_td               *td;
    struct ibv_pd               *pd;
#endif
} uct_ib_mlx5_res_domain_t;


typedef struct uct_ib_mlx5_qp_attr {
    uct_ib_qp_attr_t            super;
    uct_ib_mlx5_mmio_mode_t     mmio_mode;
} uct_ib_mlx5_qp_attr_t;


/* MLX5 QP wrapper */
typedef struct uct_ib_mlx5_qp {
    uct_ib_mlx5_obj_type_t             type;
    uint32_t                           qp_num;
    union {
        struct {
            union {
                struct ibv_qp          *qp;
#ifdef HAVE_DC_EXP
                struct ibv_exp_dct     *dct;
#endif
            };
            uct_ib_mlx5_res_domain_t   *rd;
        } verbs;
#if HAVE_DEVX
        struct {
            void                       *wq_buf;
            uct_ib_mlx5_dbrec_t        *dbrec;
            uct_ib_mlx5_devx_umem_t    mem;
            struct mlx5dv_devx_obj     *obj;
        } devx;
#endif
    };
} uct_ib_mlx5_qp_t;

/* Send work-queue */
typedef struct uct_ib_mlx5_txwq {
    uct_ib_mlx5_qp_t            super;
    uint16_t                    sw_pi;      /* PI for next WQE */
    uint16_t                    prev_sw_pi; /* PI where last WQE *started*  */
    uct_ib_mlx5_mmio_reg_t      *reg;
    void                        *curr;
    volatile uint32_t           *dbrec;
    void                        *qstart;
    void                        *qend;
    uint16_t                    bb_max;
    uint16_t                    sig_pi;     /* PI for last signaled WQE */
#if UCS_ENABLE_ASSERT
    uint16_t                    hw_ci;
#endif
    uct_ib_fence_info_t         fi;
} uct_ib_mlx5_txwq_t;


/* Receive work-queue */
typedef struct uct_ib_mlx5_rxwq {
    /* producer index. It updated when new receive wqe is posted */
    uint16_t                    rq_wqe_counter;
    /* consumer index. It is better to track it ourselves than to do ntohs()
     * on the index in the cqe
     */
    uint16_t                    cq_wqe_counter;
    uint16_t                    mask;
    volatile uint32_t           *dbrec;
    struct mlx5_wqe_data_seg    *wqes;
} uct_ib_mlx5_rxwq_t;


/* Address-vector for link-local scope */
typedef struct uct_ib_mlx5_base_av {
    uint32_t                    dqp_dct;
    uint8_t                     stat_rate_sl;
    uint8_t                     fl_mlid;
    uint16_t                    rlid;
} UCS_S_PACKED uct_ib_mlx5_base_av_t;


typedef struct uct_ib_mlx5_err_cqe {
    uint8_t                     rsvd0[32];
    uint32_t                    srqn;
    uint8_t                     rsvd1[16];
    uint8_t                     hw_err_synd;
    uint8_t                     hw_synd_type;
    uint8_t                     vendor_err_synd;
    uint8_t                     syndrome;
    uint32_t                    s_wqe_opcode_qpn;
    uint16_t                    wqe_counter;
    uint8_t                     signature;
    uint8_t                     op_own;
} UCS_S_PACKED uct_ib_mlx5_err_cqe_t;


/**
 * SRQ segment
 *
 * We add some SW book-keeping information in the unused HW fields:
 *  - desc           - the receive descriptor.
 *  - strides        - Number of available strides in this WQE. When it is 0,
 *                     this segment can be reposted to the HW. Relevant for
 *                     Multi-Packet SRQ only.
 *  - free           - points to the next out-of-order completed segment.
 */
typedef struct uct_rc_mlx5_srq_seg {
    union {
        struct mlx5_wqe_srq_next_seg   mlx5_srq;
        struct {
            uint16_t                   ptr_mask;
            uint16_t                   next_wqe_index; /* Network byte order */
            uint8_t                    signature;
            uint8_t                    rsvd1[1];
            uint8_t                    strides;
            uint8_t                    free;           /* Released but not posted */
            uct_ib_iface_recv_desc_t   *desc;          /* Host byte order */
        } srq;
    };
    struct mlx5_wqe_data_seg           dptr[0];
} uct_ib_mlx5_srq_seg_t;


struct uct_ib_mlx5_atomic_masked_cswap32_seg {
    uint32_t           swap;
    uint32_t           compare;
    uint32_t           swap_mask;
    uint32_t           compare_mask;
} UCS_S_PACKED;


struct uct_ib_mlx5_atomic_masked_fadd32_seg {
    uint32_t           add;
    uint32_t           filed_boundary;
    uint32_t           reserved[2];
} UCS_S_PACKED;


struct uct_ib_mlx5_atomic_masked_cswap64_seg {
    uint64_t           swap;
    uint64_t           compare;
} UCS_S_PACKED;


struct uct_ib_mlx5_atomic_masked_fadd64_seg {
    uint64_t           add;
    uint64_t           filed_boundary;
} UCS_S_PACKED;

ucs_status_t uct_ib_mlx5_md_get_atomic_mr_id(uct_ib_md_t *md, uint8_t *mr_id);

ucs_status_t uct_ib_mlx5_iface_get_res_domain(uct_ib_iface_t *iface,
                                              uct_ib_mlx5_qp_t *txwq);

void uct_ib_mlx5_iface_put_res_domain(uct_ib_mlx5_qp_t *qp);

ucs_status_t uct_ib_mlx5_iface_create_qp(uct_ib_iface_t *iface,
                                         uct_ib_mlx5_qp_t *qp,
                                         uct_ib_mlx5_qp_attr_t *attr);

ucs_status_t uct_ib_mlx5_modify_qp_state(uct_ib_mlx5_md_t *md,
                                         uct_ib_mlx5_qp_t *qp,
                                         enum ibv_qp_state state);

void uct_ib_mlx5_destroy_qp(uct_ib_mlx5_md_t *md, uct_ib_mlx5_qp_t *qp);

/**
 * Create CQ with DV
 */
ucs_status_t uct_ib_mlx5_create_cq(uct_ib_iface_t *iface, uct_ib_dir_t dir,
                                   const uct_ib_iface_init_attr_t *init_attr,
                                   int preferred_cpu, size_t inl);

extern ucs_config_field_t uct_ib_mlx5_iface_config_table[];

/**
 * Get internal CQ information.
 */
ucs_status_t uct_ib_mlx5_get_cq(struct ibv_cq *cq, uct_ib_mlx5_cq_t *mlx5_cq);

/**
 * Get flag indicating compact AV support.
 */
ucs_status_t uct_ib_mlx5_get_compact_av(uct_ib_iface_t *iface, int *compact_av);

/**
 * Requests completion notification.
 */
ucs_status_t uct_ib_mlx5dv_arm_cq(uct_ib_mlx5_cq_t *cq, int solicited);

/**
 * Check for completion with error.
 */
void uct_ib_mlx5_check_completion(uct_ib_iface_t *iface, uct_ib_mlx5_cq_t *cq,
                                  struct mlx5_cqe64 *cqe);

ucs_status_t
uct_ib_mlx5_get_mmio_mode(uct_priv_worker_t *worker,
                          uct_ib_mlx5_mmio_mode_t cfg_mmio_mode,
                          unsigned bf_size,
                          uct_ib_mlx5_mmio_mode_t *mmio_mode);

/**
 * Initialize txwq structure.
 */
ucs_status_t uct_ib_mlx5_txwq_init(uct_priv_worker_t *worker,
                                   uct_ib_mlx5_mmio_mode_t cfg_mmio_mode,
                                   uct_ib_mlx5_txwq_t *txwq, struct ibv_qp *verbs_qp);

void uct_ib_mlx5_qp_mmio_cleanup(uct_ib_mlx5_qp_t *qp,
                                 uct_ib_mlx5_mmio_reg_t *reg);

/**
 * Reset txwq contents and posting indices.
 */
void uct_ib_mlx5_txwq_reset(uct_ib_mlx5_txwq_t *txwq);

/**
 * Initialize rxwq structure.
 */
ucs_status_t uct_ib_mlx5_get_rxwq(struct ibv_qp *qp, uct_ib_mlx5_rxwq_t *wq);

/**
 * Initialize srq structure.
 */
ucs_status_t
uct_ib_mlx5_verbs_srq_init(uct_ib_mlx5_srq_t *srq, struct ibv_srq *verbs_srq,
                           size_t sg_byte_count, int num_sge);

void uct_ib_mlx5_srq_buff_init(uct_ib_mlx5_srq_t *srq, uint32_t head,
                               uint32_t tail, size_t sg_byte_count, int num_sge);

void uct_ib_mlx5_verbs_srq_cleanup(uct_ib_mlx5_srq_t *srq, struct ibv_srq *verbs_srq);

/**
 * DEVX UAR API
 */
int uct_ib_mlx5_devx_uar_cmp(uct_ib_mlx5_devx_uar_t *uar,
                             uct_ib_mlx5_md_t *md,
                             uct_ib_mlx5_mmio_mode_t mmio_mode);

ucs_status_t uct_ib_mlx5_devx_uar_init(uct_ib_mlx5_devx_uar_t *uar,
                                       uct_ib_mlx5_md_t *md,
                                       uct_ib_mlx5_mmio_mode_t mmio_mode);

void uct_ib_mlx5_devx_uar_cleanup(uct_ib_mlx5_devx_uar_t *uar);

/**
 * DEVX QP API
 */

#if HAVE_DEVX

ucs_status_t uct_ib_mlx5_devx_create_qp(uct_ib_iface_t *iface,
                                        uct_ib_mlx5_qp_t *qp,
                                        uct_ib_mlx5_txwq_t *tx,
                                        uct_ib_mlx5_qp_attr_t *attr);

ucs_status_t uct_ib_mlx5_devx_modify_qp(uct_ib_mlx5_qp_t *qp,
                                        const void *in, size_t inlen,
                                        void *out, size_t outlen);

ucs_status_t uct_ib_mlx5_devx_modify_qp_state(uct_ib_mlx5_qp_t *qp,
                                              enum ibv_qp_state state);

void uct_ib_mlx5_devx_destroy_qp(uct_ib_mlx5_md_t *md, uct_ib_mlx5_qp_t *qp);

ucs_status_t uct_ib_mlx5_devx_query_ooo_sl_mask(uct_ib_mlx5_md_t *md,
                                                uint8_t port_num,
                                                uint16_t *ooo_sl_mask_p);

static inline ucs_status_t
uct_ib_mlx5_md_buf_alloc(uct_ib_mlx5_md_t *md, size_t size, int silent,
                         void **buf_p, uct_ib_mlx5_devx_umem_t *mem,
                         char *name)
{
    ucs_log_level_t level = silent ? UCS_LOG_LEVEL_DEBUG : UCS_LOG_LEVEL_ERROR;
    ucs_status_t status;
    void *buf;
    int ret;

    ret = ucs_posix_memalign(&buf, ucs_get_page_size(), size, name);
    if (ret != 0) {
        ucs_log(level, "failed to allocate buffer of %zu bytes: %m", size);
        return UCS_ERR_NO_MEMORY;
    }

    if (md->super.fork_init) {
        ret = madvise(buf, size, MADV_DONTFORK);
        if (ret != 0) {
            ucs_log(level, "madvise(DONTFORK, buf=%p, len=%zu) failed: %m", buf, size);
            status = UCS_ERR_IO_ERROR;
            goto err_free;
        }
    }

    mem->size = size;
    mem->mem  = mlx5dv_devx_umem_reg(md->super.dev.ibv_context, buf, size, 0);
    if (mem->mem == NULL) {
        ucs_log(level, "mlx5dv_devx_umem_reg() failed: %m");
        status = UCS_ERR_NO_MEMORY;
        goto err_dofork;
    }

    *buf_p = buf;
    return UCS_OK;

err_dofork:
    if (md->super.fork_init) {
        madvise(buf, size, MADV_DOFORK);
    }
err_free:
    ucs_free(buf);

    return status;
}

static inline void
uct_ib_mlx5_md_buf_free(uct_ib_mlx5_md_t *md, void *buf, uct_ib_mlx5_devx_umem_t *mem)
{
    int ret;

    if (buf == NULL) {
        return;
    }

    mlx5dv_devx_umem_dereg(mem->mem);
    if (md->super.fork_init) {
        ret = madvise(buf, mem->size, MADV_DOFORK);
        if (ret != 0) {
            ucs_warn("madvise(DOFORK, buf=%p, len=%zu) failed: %m", buf, mem->size);
        }
    }
    ucs_free(buf);
}

#else

static inline ucs_status_t
uct_ib_mlx5_devx_create_qp(uct_ib_iface_t *iface,
                           uct_ib_mlx5_qp_t *qp,
                           uct_ib_mlx5_txwq_t *tx,
                           uct_ib_mlx5_qp_attr_t *attr)
{
    return UCS_ERR_UNSUPPORTED;
}

static inline ucs_status_t
uct_ib_mlx5_devx_modify_qp(uct_ib_mlx5_qp_t *qp,
                           enum ibv_qp_state state)
{
    return UCS_ERR_UNSUPPORTED;
}

static inline ucs_status_t
uct_ib_mlx5_devx_modify_qp_state(uct_ib_mlx5_qp_t *qp, enum ibv_qp_state state)
{
    return UCS_ERR_UNSUPPORTED;
}

static inline void uct_ib_mlx5_devx_destroy_qp(uct_ib_mlx5_md_t *md, uct_ib_mlx5_qp_t *qp) { }

#endif

ucs_status_t
uct_ib_mlx5_select_sl(const uct_ib_iface_config_t *ib_config,
                      ucs_ternary_auto_value_t ar_enable,
                      uint16_t hw_sl_mask, int have_sl_mask_cap,
                      const char *dev_name, uint8_t port_num,
                      uint8_t *sl_p);

ucs_status_t
uct_ib_mlx5_iface_select_sl(uct_ib_iface_t *iface,
                            const uct_ib_mlx5_iface_config_t *ib_mlx5_config,
                            const uct_ib_iface_config_t *ib_config);

static inline uct_ib_mlx5_dbrec_t *uct_ib_mlx5_get_dbrec(uct_ib_mlx5_md_t *md)
{
    uct_ib_mlx5_dbrec_t *dbrec;

    ucs_recursive_spin_lock(&md->dbrec_lock);
    dbrec = (uct_ib_mlx5_dbrec_t *)ucs_mpool_get_inline(&md->dbrec_pool);
    ucs_recursive_spin_unlock(&md->dbrec_lock);
    if (dbrec != NULL) {
        dbrec->db[MLX5_SND_DBR] = 0;
        dbrec->db[MLX5_RCV_DBR] = 0;
        dbrec->md               = md;
    }

    return dbrec;
}

static inline void uct_ib_mlx5_put_dbrec(uct_ib_mlx5_dbrec_t *dbrec)
{
    uct_ib_mlx5_md_t *md = dbrec->md;

    ucs_recursive_spin_lock(&md->dbrec_lock);
    ucs_mpool_put_inline(dbrec);
    ucs_recursive_spin_unlock(&md->dbrec_lock);
}

#endif
