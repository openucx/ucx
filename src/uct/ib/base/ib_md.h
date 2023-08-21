/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
 * Copyright (C) The University of Tennessee and The University
 *               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_MD_H_
#define UCT_IB_MD_H_

#include "ib_device.h"

#include <uct/base/uct_md.h>
#include <ucs/stats/stats.h>

#define UCT_IB_MD_MAX_MR_SIZE        0x80000000UL
#define UCT_IB_MD_PACKED_RKEY_SIZE   sizeof(uint64_t)
#define UCT_IB_MD_INVALID_FLUSH_RKEY 0xff

#define UCT_IB_MD_DEFAULT_GID_INDEX 0   /**< The gid index used by default for an IB/RoCE port */

#define UCT_IB_MEM_ACCESS_FLAGS  (IBV_ACCESS_LOCAL_WRITE | \
                                  IBV_ACCESS_REMOTE_WRITE | \
                                  IBV_ACCESS_REMOTE_READ | \
                                  IBV_ACCESS_REMOTE_ATOMIC)

#define UCT_IB_CONFIG_PREFIX      "IB_"

#define UCT_IB_MD_NAME(_x)        "ib_" UCS_PP_QUOTE(_x)

#define UCT_IB_MD_FLUSH_REMOTE_LENGTH 8


#define uct_ib_md_log_mem_reg_error(_md, _flags, _fmt, ...) \
    uct_md_log_mem_reg_error(_flags, "md %p (%s): " _fmt, _md, \
                             uct_ib_device_name(&(_md)->dev), ## __VA_ARGS__)


/**
 * IB MD statistics counters
 */
enum {
    UCT_IB_MD_STAT_MEM_ALLOC,
    UCT_IB_MD_STAT_MEM_REG,
    UCT_IB_MD_STAT_LAST
};


enum {
    UCT_IB_MEM_FLAG_ODP              = UCS_BIT(0), /**< The memory region has on
                                                        demand paging enabled */
    UCT_IB_MEM_FLAG_ATOMIC_MR        = UCS_BIT(1), /**< The memory region has UMR
                                                        for the atomic access */
    UCT_IB_MEM_ACCESS_REMOTE_ATOMIC  = UCS_BIT(2), /**< An atomic access was
                                                        requested for the memory
                                                        region */
    UCT_IB_MEM_MULTITHREADED         = UCS_BIT(3), /**< The memory region registration
                                                        handled by chunks in parallel
                                                        threads */
    UCT_IB_MEM_FLAG_RELAXED_ORDERING = UCS_BIT(4), /**< The memory region will issue
                                                        PCIe writes with relaxed order
                                                        attribute */
    UCT_IB_MEM_FLAG_IMPORTED         = UCS_BIT(5), /**< The memory handle was
                                                        created by mem_attach */
#if ENABLE_PARAMS_CHECK
    UCT_IB_MEM_ACCESS_REMOTE_RMA     = UCS_BIT(6) /**< RMA access was requested
                                                        for the memory region */
#else
    UCT_IB_MEM_ACCESS_REMOTE_RMA     = 0
#endif
};

enum {
    UCT_IB_DEVX_OBJ_RCQP,
    UCT_IB_DEVX_OBJ_RCSRQ,
    UCT_IB_DEVX_OBJ_DCT,
    UCT_IB_DEVX_OBJ_DCSRQ,
    UCT_IB_DEVX_OBJ_DCI,
    UCT_IB_DEVX_OBJ_CQ
};

typedef struct uct_ib_md_ext_config {
    int                      eth_pause;    /**< Whether or not Pause Frame is
                                                enabled on the Ethernet network */
    int                      prefer_nearest_device; /**< Give priority for near
                                                         device */
    int                      enable_indirect_atomic; /** Enable indirect atomic */

    struct {
        int                  prefetch;     /**< Auto-prefetch non-blocking memory
                                                registrations / allocations */
    } odp;

    unsigned long            gid_index;    /**< IB GID index to use */

    size_t                   min_mt_reg;   /**< Multi-threaded registration threshold */
    size_t                   mt_reg_chunk; /**< Multi-threaded registration chunk */
    int                      mt_reg_bind;  /**< Multi-threaded registration bind to core */
    unsigned                 max_idle_rkey_count; /**< Maximal number of
                                                       invalidated memory keys
                                                       that are kept idle before
                                                       reuse*/
    unsigned                 reg_retry_cnt; /**< Memory registration retry count */
} uct_ib_md_ext_config_t;


typedef struct {
    uint32_t                lkey;
    uint32_t                exported_lkey;
    uint32_t                rkey;
    uint32_t                atomic_rkey;
    uint32_t                indirect_rkey;
    uint32_t                flags;
} uct_ib_mem_t;


typedef struct {
    struct ibv_mr           *ib;
} uct_ib_mr_t;


typedef enum {
    /* Default memory region with either strict or relaxed order */
    UCT_IB_MR_DEFAULT,
    /* Additional memory region with strict order,
     * if the default region is relaxed order */
    UCT_IB_MR_STRICT_ORDER,
    UCT_IB_MR_LAST
} uct_ib_mr_type_t;


/**
 * IB memory domain.
 */
typedef struct uct_ib_md {
    uct_md_t                 super;
    struct ibv_pd            *pd;       /**< IB memory domain */
    uct_ib_device_t          dev;       /**< IB device */
    ucs_linear_func_t        reg_cost;  /**< Memory registration cost */
    UCS_STATS_NODE_DECLARE(stats)
    uct_ib_md_ext_config_t   config;    /* IB external configuration */
    struct {
        uct_ib_device_spec_t *specs;    /* Custom device specifications */
        unsigned             count;     /* Number of custom devices */
    } custom_devices;
    int                      ece_enable;
    int                      check_subnet_filter;
    uint64_t                 subnet_filter;
    double                   pci_bw;
    int                      relaxed_order;
    int                      fork_init;
    uint64_t                 reg_mem_types;
    uint64_t                 reg_nonblock_mem_types;
    uint64_t                 cap_flags;
    char                     *name;
    /* flush_remote rkey is used as atomic_mr_id value (8-16 bits of rkey)
     * when UMR regions can be created. Bits 0-7 must be zero always (assuming
     * that lowest byte is mkey tag which is not used). Non-zero bits 0-7
     * means that flush_rkey is invalid and flush_remote operation could not
     * be initiated.  */
    uint32_t                 flush_rkey;
    uint16_t                 vhca_id;
} uct_ib_md_t;


typedef struct uct_ib_md_packed_mkey {
    uint32_t lkey;
    uint16_t vhca_id;
} UCS_S_PACKED uct_ib_md_packed_mkey_t;


/**
 * IB memory domain configuration.
 */
typedef struct uct_ib_md_config {
    uct_md_config_t          super;

    ucs_linear_func_t        reg_cost;     /**< Memory registration cost estimation */
    unsigned                 fork_init;    /**< Use ibv_fork_init() */
    int                      async_events; /**< Whether async events should be delivered */

    uct_ib_md_ext_config_t   ext;          /**< External configuration */

    UCS_CONFIG_STRING_ARRAY_FIELD(spec) custom_devices; /**< Custom device specifications */

    char                     *subnet_prefix; /**< Filter of subnet_prefix for IB ports */

    UCS_CONFIG_ARRAY_FIELD(ucs_config_bw_spec_t, device) pci_bw; /**< List of PCI BW for devices */

    int                      mlx5dv; /**< mlx5 support */
    int                      devx; /**< DEVX support */
    unsigned                 devx_objs;    /**< Objects to be created by DevX */
    ucs_ternary_auto_value_t mr_relaxed_order; /**< Allow reorder memory accesses */
    int                      enable_gpudirect_rdma; /**< Enable GPUDirect RDMA */
} uct_ib_md_config_t;

/**
 * Memory domain constructor.
 *
 * @param [in]  ibv_device    IB device.
 *
 * @param [in]  md_config     Memory domain configuration parameters.
 *
 * @param [out] md_p          Handle to memory domain.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_open_func_t)(struct ibv_device *ibv_device,
                                              const uct_ib_md_config_t *md_config,
                                              struct uct_ib_md **md_p);

/**
 * Memory domain method to register memory area.
 *
 * @param [in]  md             Memory domain.
 *
 * @param [in]  address        Memory area start address.
 *
 * @param [in]  length         Memory area length.
 *
 * @param [in]  access         IB verbs registration access flags.
 *
 * @param [in]  dmabuf_fd      dmabuf file descriptor.
 *
 * @param [in]  dmabuf_offset  Offset of the registered memory region within the
 *                             dmabuf backing region.
 *
 * @param [in]  memh           Memory region handle.
 *                             The method must initialize lkey & rkey.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_reg_key_func_t)(struct uct_ib_md *md,
                                                 void *address, size_t length,
                                                 uint64_t access, int dmabuf_fd,
                                                 size_t dmabuf_offset,
                                                 uct_ib_mem_t *memh,
                                                 uct_ib_mr_type_t mr_type,
                                                 int silent);

/**
 * Memory domain method to deregister memory area.
 *
 * @param [in]  md      Memory domain.
 *
 * @param [in]  memh    Memory region handle registered with
 *                      uct_ib_md_reg_key_func_t.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_dereg_key_func_t)(struct uct_ib_md *md,
                                                   uct_ib_mem_t *memh,
                                                   uct_ib_mr_type_t mr_type);

/**
 * Memory domain method to register memory area optimized for atomic ops.
 *
 * @param [in]  md      Memory domain.
 *
 * @param [in]  memh    Memory region handle registered for regular ops.
 *                      Method should initialize atomic_rkey
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_reg_atomic_key_func_t)(struct uct_ib_md *md,
                                                        uct_ib_mem_t *memh);


/**
 * Memory domain method to register indirect memory key which supports
 * @ref UCT_MD_MKEY_PACK_FLAG_INVALIDATE.
 *
 * @param [in]  md      Memory domain.
 *
 * @param [in]  memh    Memory region handle registered for regular ops.
 *                      Method should initialize indirect_rkey
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_reg_indirect_key_func_t)(struct uct_ib_md *md,
                                                          uct_ib_mem_t *memh);


/**
 * Memory domain method to release resources registered for atomic ops.
 *
 * @param [in]  md      Memory domain.
 *
 * @param [in]  memh    Memory region handle registered with
 *                      uct_ib_md_reg_atomic_key_func_t.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_dereg_atomic_key_func_t)(struct uct_ib_md *md,
                                                          uct_ib_mem_t *memh);

/**
 * Memory domain method to register memory area using multiple threads.
 *
 * @param [in]  md      Memory domain.
 *
 * @param [in]  address Memory area start address.
 *
 * @param [in]  length  Memory area length.
 *
 * @param [in]  access  IB verbs registration access flags
 *
 * @param [in]  memh    Memory region handle.
 *                      Method should initialize lkey & rkey.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_reg_multithreaded_func_t)(uct_ib_md_t *md,
                                                           void *address,
                                                           size_t length,
                                                           uint64_t access,
                                                           uct_ib_mem_t *memh,
                                                           uct_ib_mr_type_t mr_type,
                                                           int silent);

/**
 * Memory domain method to deregister memory area.
 *
 * @param [in]  md      Memory domain.
 *
 * @param [in]  memh    Memory region handle registered with
 *                      uct_ib_md_reg_key_func_t.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_dereg_multithreaded_func_t)(uct_ib_md_t *md,
                                                             uct_ib_mem_t *memh,
                                                             uct_ib_mr_type_t mr_type);

/**
 * Memory domain method to get unique atomic mr id.
 *
 * @param [in]  md      Memory domain.
 *
 * @param [out] mr_id   id to access atomic MR.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_get_atomic_mr_id_func_t)(uct_ib_md_t *md,
                                                          uint8_t *mr_id);


/**
 * Memory domain method to register crossed mkey for memory area.
 *
 * @param [in]  ib_md           Memory domain.
 * @param [out] ib_memh         Memory region handle.
 *                              Method should initialize lkey & rkey.
 *
 * @return UCS_OK on success or error code in case of failure.
 */
typedef ucs_status_t (*uct_ib_md_reg_exported_key_func_t)(
        uct_ib_md_t *ib_md, uct_ib_mem_t *ib_memh);


typedef struct uct_ib_md_ops {
    uct_md_ops_t                         super;
    uct_ib_md_open_func_t                open;
    uct_ib_md_reg_key_func_t             reg_key;
    uct_ib_md_reg_indirect_key_func_t    reg_indirect_key;
    uct_ib_md_dereg_key_func_t           dereg_key;
    uct_ib_md_reg_atomic_key_func_t      reg_atomic_key;
    uct_ib_md_dereg_atomic_key_func_t    dereg_atomic_key;
    uct_ib_md_reg_multithreaded_func_t   reg_multithreaded;
    uct_ib_md_dereg_multithreaded_func_t dereg_multithreaded;
    uct_ib_md_get_atomic_mr_id_func_t    get_atomic_mr_id;
    uct_ib_md_reg_exported_key_func_t    reg_exported_key;
} uct_ib_md_ops_t;


/**
 * IB memory domain constructor. Should have following logic:
 * - probe provided IB device, may return UCS_ERR_UNSUPPORTED
 * - allocate MD and IB context
 * - setup atomic MR ops
 * - determine device attributes and flags
 */
typedef struct uct_ib_md_ops_entry {
    const char                  *name;
    uct_ib_md_ops_t             *ops;
} uct_ib_md_ops_entry_t;


#define UCT_IB_MD_OPS_NAME(_name) uct_ib_md_ops_##_name##_entry

#define UCT_IB_MD_DEFINE_ENTRY(_name, _md_ops) \
    uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(_name) = { \
        .name = UCS_PP_MAKE_STRING(_md_ops), \
        .ops  = &_md_ops, \
    }

extern uct_component_t uct_ib_component;


static UCS_F_ALWAYS_INLINE uint32_t uct_ib_md_direct_rkey(uct_rkey_t uct_rkey)
{
    return (uint32_t)uct_rkey;
}


static UCS_F_ALWAYS_INLINE uint32_t
uct_ib_md_indirect_rkey(uct_rkey_t uct_rkey)
{
    return uct_rkey >> 32;
}


static UCS_F_ALWAYS_INLINE void
uct_ib_md_pack_rkey(uint32_t rkey, uint32_t atomic_rkey, void *rkey_buffer)
{
    uint64_t *rkey_p = (uint64_t*)rkey_buffer;

    *rkey_p = (((uint64_t)atomic_rkey) << 32) | rkey;
    ucs_trace("packed rkey: direct 0x%x indirect 0x%x", rkey, atomic_rkey);
}


static UCS_F_ALWAYS_INLINE void
uct_ib_md_pack_exported_mkey(uct_ib_md_t *md, uint32_t lkey, void *buffer)
{
    uct_ib_md_packed_mkey_t *mkey = (uct_ib_md_packed_mkey_t*)buffer;

    mkey->lkey    = lkey;
    mkey->vhca_id = md->vhca_id;

    ucs_trace("packed exported mkey on %s: lkey 0x%x",
              uct_ib_device_name(&md->dev), lkey);
}


/**
 * rkey is packed/unpacked is such a way that:
 * low  32 bits contain a direct key
 * high 32 bits contain either UCT_IB_INVALID_RKEY or a valid indirect key.
 */
static inline uint32_t uct_ib_resolve_atomic_rkey(uct_rkey_t uct_rkey,
                                                  uint16_t atomic_mr_offset,
                                                  uint64_t *remote_addr_p)
{
    uint32_t atomic_rkey = uct_ib_md_indirect_rkey(uct_rkey);
    if (atomic_rkey == UCT_IB_INVALID_MKEY) {
        return uct_ib_md_direct_rkey(uct_rkey);
    } else {
        *remote_addr_p += atomic_mr_offset;
        return atomic_rkey;
    }
}


static inline uint16_t uct_ib_md_atomic_offset(uint8_t atomic_mr_id)
{
    return 8 * atomic_mr_id;
}

static inline void
uct_ib_memh_init_keys(uct_ib_mem_t *memh, uint32_t lkey, uint32_t rkey)
{
    memh->lkey = lkey;
    memh->rkey = rkey;
}

static UCS_F_ALWAYS_INLINE uct_ib_mr_type_t
uct_ib_md_get_atomic_mr_type(uct_ib_md_t *md)
{
    return md->relaxed_order ? UCT_IB_MR_STRICT_ORDER : UCT_IB_MR_DEFAULT;
}

static UCS_F_ALWAYS_INLINE uint32_t uct_ib_memh_get_lkey(uct_mem_h memh)
{
    ucs_assert(memh != UCT_MEM_HANDLE_NULL);
    return ((uct_ib_mem_t*)memh)->lkey;
}


static UCS_F_ALWAYS_INLINE UCS_F_MAYBE_UNUSED ucs_status_t
uct_ib_md_rkey_mem_dereg_invalidate_check(uint32_t rkey, uint32_t access_mask,
                                          uint64_t cap_mask)
{
    if (!access_mask) {
        return UCS_OK;
    }

    if (!cap_mask) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (rkey == UCT_IB_INVALID_MKEY) {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_ib_md_mem_dereg_params_invalidate_check(
        const uct_ib_md_t *md, const uct_md_mem_dereg_params_t *params)
{
    uct_ib_mem_t *ib_memh;
    unsigned flags;
    ucs_status_t status;

    if (!ENABLE_PARAMS_CHECK) {
        return UCS_OK;
    }

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params,
                                  md->cap_flags & (UCT_MD_FLAG_INVALIDATE_RMA |
                                                   UCT_MD_FLAG_INVALIDATE_AMO));

    ib_memh = (uct_ib_mem_t*)UCT_MD_MEM_DEREG_FIELD_VALUE(params, memh,
                                                          FIELD_MEMH, NULL);
    flags   = UCT_MD_MEM_DEREG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0);
    if (!(flags & UCT_MD_MEM_DEREG_FLAG_INVALIDATE)) {
        return UCS_OK;
    }

    status = uct_ib_md_rkey_mem_dereg_invalidate_check(
            ib_memh->indirect_rkey,
            ib_memh->flags & UCT_IB_MEM_ACCESS_REMOTE_RMA,
            md->cap_flags & UCT_MD_FLAG_INVALIDATE_RMA);
    if (status != UCS_OK) {
        return status;
    }

    return uct_ib_md_rkey_mem_dereg_invalidate_check(
            ib_memh->atomic_rkey,
            ib_memh->flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC,
            md->cap_flags & UCT_MD_FLAG_INVALIDATE_AMO);
}

static UCS_F_ALWAYS_INLINE const uct_ib_md_ops_t *uct_ib_md_ops(uct_ib_md_t *md)
{
    return ucs_derived_of(md->super.ops, uct_ib_md_ops_t);
}

static UCS_F_ALWAYS_INLINE int
uct_ib_md_is_flush_rkey_valid(uint32_t flush_rkey) {
    /* Valid flush_rkey should have 0 in the LSB */
    return (flush_rkey & UCT_IB_MD_INVALID_FLUSH_RKEY) == 0;
}

static UCS_F_ALWAYS_INLINE uint8_t uct_ib_md_get_atomic_mr_id(uct_ib_md_t *md)
{
    uint8_t mr_id;

    if (uct_ib_md_ops(md)->get_atomic_mr_id(md, &mr_id) == UCS_OK) {
        return mr_id;
    }

    return 0;
}

ucs_status_t uct_ib_md_open(uct_component_t *component, const char *md_name,
                            const uct_md_config_t *uct_md_config, uct_md_h *md_p);

void uct_ib_md_parse_relaxed_order(uct_ib_md_t *md,
                                   const uct_ib_md_config_t *md_config,
                                   int is_supported);

ucs_status_t uct_ib_md_query(uct_md_h uct_md, uct_md_attr_v2_t *md_attr);

ucs_status_t uct_ib_mem_reg(uct_md_h uct_md, void *address, size_t length,
                            const uct_md_mem_reg_params_t *params,
                            size_t memh_base_size, size_t mr_size,
                            uct_mem_h *memh_p);

ucs_status_t
uct_ib_mem_dereg(uct_md_h uct_md, const uct_md_mem_dereg_params_t *params);

ucs_status_t uct_ib_mem_advise(uct_md_h uct_md, uct_mem_h memh, void *addr,
                               size_t length, unsigned advice);

ucs_status_t uct_ib_mkey_pack(uct_md_h uct_md, uct_mem_h uct_memh,
                              const uct_md_mkey_pack_params_t *params,
                              void *mkey_buffer);

int uct_ib_device_is_accessible(struct ibv_device *device);

ucs_status_t uct_ib_md_open_common(uct_ib_md_t *md,
                                   struct ibv_device *ib_device,
                                   const uct_ib_md_config_t *md_config);

void uct_ib_md_close_common(uct_ib_md_t *md);

void uct_ib_md_device_context_close(struct ibv_context *ctx);

uct_ib_md_t* uct_ib_md_alloc(size_t size, const char *name,
                             struct ibv_context *ctx);

void uct_ib_md_free(uct_ib_md_t *md);

void uct_ib_md_close(uct_md_h tl_md);

ucs_status_t uct_ib_reg_mr_params(uct_ib_md_t *md, void *address, size_t length,
                                  const uct_md_mem_reg_params_t *params,
                                  uint64_t access_flags, struct ibv_mr **mr_p);
ucs_status_t uct_ib_dereg_mr(struct ibv_mr *mr);


/**
 * Check if IB md device has ECE capability
 */
void uct_ib_md_ece_check(uct_ib_md_t *md);


ucs_status_t
uct_ib_md_handle_mr_list_mt(uct_ib_md_t *md, void *address, size_t length,
                            const uct_md_mem_reg_params_t *params,
                            uint64_t access_flags, struct ibv_mr **mrs);

ucs_status_t uct_ib_reg_key_impl(uct_ib_md_t *md, void *address, size_t length,
                                 uint64_t access_flags, int dmabuf_fd,
                                 size_t dmabuf_offset, uct_ib_mem_t *memh,
                                 uct_ib_mr_t *mr, uct_ib_mr_type_t mr_type,
                                 int silent);

uint64_t uct_ib_memh_access_flags(uct_ib_md_t *md, uct_ib_mem_t *memh);

ucs_status_t uct_ib_verbs_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                  const uct_md_mem_reg_params_t *params,
                                  uct_mem_h *memh_p);

ucs_status_t uct_ib_verbs_mem_dereg(uct_md_h uct_md,
                                    const uct_md_mem_dereg_params_t *params);

ucs_status_t uct_ib_verbs_mkey_pack(uct_md_h uct_md, uct_mem_h uct_memh,
                                    const uct_md_mkey_pack_params_t *params,
                                    void *mkey_buffer);

ucs_status_t uct_ib_memh_alloc(uct_ib_md_t *md, size_t length,
                               unsigned mem_flags, size_t memh_base_size,
                               size_t mr_size, uct_ib_mem_t **memh_p);

#endif
