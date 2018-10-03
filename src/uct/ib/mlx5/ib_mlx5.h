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
#  include "ib_mlx5_hw.h"
#endif
#include "ib_mlx5_dv.h"

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
#define UCT_IB_MLX5_RES_DOMAIN_KEY      0x1b1bda7aU
#define UCT_IB_MLX5_WORKER_DM_KEY       0xacdf1245u
#define UCT_IB_MLX5_EXTENDED_UD_AV      0x80 /* htonl(0x80000000) */
#define UCT_IB_MLX5_AV_GRH_PRESENT      0x40 /* htonl(UCS_BIT(30)) */
#define UCT_IB_MLX5_BF_REG_SIZE         256
#define UCT_IB_MLX5_CQE_VENDOR_SYND_ODP 0x93
#define UCT_IB_MLX5_CQE_OP_OWN_ERR_MASK 0x80
#define UCT_IB_MLX5_MAX_SEND_WQE_SIZE   (UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB)
#define UCT_IB_MLX5_CQ_SET_CI           0
#define UCT_IB_MLX5_CQ_ARM_DB           1


#define UCT_IB_MLX5_OPMOD_EXT_ATOMIC(_log_arg_size) \
    ((8) | ((_log_arg_size) - 2))

#if HAVE_STRUCT_MLX5_WQE_AV_BASE

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

#endif

#if !(HAVE_MLX5_WQE_CTRL_SOLICITED)
#  define MLX5_WQE_CTRL_SOLICITED  (1<<1)
#endif

#define UCT_IB_MLX5_AM_ZCOPY_MAX_IOV  3UL

#define UCT_IB_MLX5_AM_MAX_SHORT(_av_size) \
    (UCT_IB_MLX5_MAX_SEND_WQE_SIZE - \
     (sizeof(struct mlx5_wqe_ctrl_seg) + \
      (_av_size) + \
      sizeof(struct mlx5_wqe_inl_data_seg)))

#define UCT_IB_MLX5_AM_ZCOPY_MAX_HDR(_av_size) \
    (UCT_IB_MLX5_AM_MAX_SHORT(_av_size) - \
     UCT_IB_MLX5_AM_ZCOPY_MAX_IOV * sizeof(struct mlx5_wqe_data_seg))

#define UCT_IB_MLX5_PUT_MAX_SHORT(_av_size) \
    (UCT_IB_MLX5_AM_MAX_SHORT(_av_size) - sizeof(struct mlx5_wqe_raddr_seg))

#define UCT_IB_MLX5_SRQ_STRIDE   (sizeof(struct mlx5_wqe_srq_next_seg) + \
                                  sizeof(struct mlx5_wqe_data_seg))

typedef enum {
    UCT_IB_MLX5_MMIO_MODE_BF_POST,    /* BF without flush, can be used only from
                                         one thread */
    UCT_IB_MLX5_MMIO_MODE_BF_POST_MT, /* BF with order, can be used by multiple
                                         serialized threads */
    UCT_IB_MLX5_MMIO_MODE_DB,         /* 8-byte doorbell (with the mandatory flush) */
    UCT_IB_MLX5_MMIO_MODE_LAST
} uct_ib_mlx5_mmio_mode_t;


/* Shared receive queue */
typedef struct uct_ib_mlx5_srq {
    void               *buf;
    volatile uint32_t  *db;
    uint16_t           free_idx;   /* what is completed contiguously */
    uint16_t           ready_idx;  /* what is ready to be posted to hw */
    uint16_t           sw_pi;      /* what is posted to hw */
    uint16_t           mask;
    uint16_t           tail;       /* tail in the driver */
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


/* Send work-queue */
typedef struct uct_ib_mlx5_txwq {
    uint16_t                    sw_pi;      /* PI for next WQE */
    uint16_t                    prev_sw_pi; /* PI where last WQE *started*  */
    uct_ib_mlx5_mmio_reg_t      *reg;
    void                        *curr;
    volatile uint32_t           *dbrec;
    void                        *qstart;
    void                        *qend;
    uint16_t                    bb_max;
    uint16_t                    sig_pi;     /* PI for last signaled WQE */
#if ENABLE_ASSERT
    uint16_t                    hw_ci;
#endif
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


/**
 * SRQ segment
 *
 * We add some SW book-keeping information in the unused HW fields:
 *  - next_hole - points to the next out-of-order completed segment
 *  - desc      - the receive descriptor.
 *
 */
typedef struct uct_rc_mlx5_srq_seg {
    union {
        struct mlx5_wqe_srq_next_seg   mlx5_srq;
        struct {
            uint8_t                    rsvd0[2];
            uint16_t                   next_wqe_index; /* Network byte order */
            uint8_t                    signature;
            uint8_t                    rsvd1[2];
            uint8_t                    free;           /* Released but not posted */
            uct_ib_iface_recv_desc_t   *desc;          /* Host byte order */
        } srq;
    };
    struct mlx5_wqe_data_seg           dptr;
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
int uct_ib_mlx5dv_arm_cq(uct_ib_mlx5_cq_t *cq, int solicited);

/**
 * Check for completion with error.
 */
void uct_ib_mlx5_check_completion(uct_ib_iface_t *iface, uct_ib_mlx5_cq_t *cq,
                                  struct mlx5_cqe64 *cqe);

/**
 * Initialize txwq structure.
 */
ucs_status_t uct_ib_mlx5_txwq_init(uct_priv_worker_t *worker, uct_ib_mlx5_txwq_t *txwq,
                                   struct ibv_qp *verbs_qp);
void uct_ib_mlx5_txwq_cleanup(uct_ib_mlx5_txwq_t* txwq);

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
ucs_status_t uct_ib_mlx5_srq_init(uct_ib_mlx5_srq_t *srq, struct ibv_srq *verbs_srq,
                                  size_t sg_byte_count);
void uct_ib_mlx5_srq_cleanup(uct_ib_mlx5_srq_t *srq, struct ibv_srq *verbs_srq);


#endif
