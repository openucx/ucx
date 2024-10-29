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
    UCT_IB_MEM_ACCESS_REMOTE_ATOMIC  = UCS_BIT(1), /**< An atomic access was
                                                        requested for the memory
                                                        region */
    UCT_IB_MEM_MULTITHREADED         = UCS_BIT(2), /**< The memory region registration
                                                        handled by chunks in parallel
                                                        threads */
    UCT_IB_MEM_IMPORTED              = UCS_BIT(3), /**< The memory handle was
                                                        created by mem_attach */
#if ENABLE_PARAMS_CHECK
    UCT_IB_MEM_ACCESS_REMOTE_RMA     = UCS_BIT(4), /**< RMA access was requested
                                                        for the memory region */
#else
    UCT_IB_MEM_ACCESS_REMOTE_RMA     = 0,
#endif
    UCT_IB_MEM_FLAG_GVA              = UCS_BIT(5), /**< The memory handle is a
                                                        GVA region */
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
        uint64_t             mem_types;    /**< Supported mem types for ODP */
    } odp;

    unsigned long            gid_index;    /**< IB GID index to use */

    size_t                   min_mt_reg;   /**< Multi-threaded registration threshold */
    size_t                   mt_reg_chunk; /**< Multi-threaded registration chunk */
    int                      mt_reg_bind;  /**< Multi-threaded registration bind to core */
    unsigned                 max_idle_rkey_count; /**< Maximal number of
                                                       invalidated memory keys
                                                       that are kept idle before
                                                       reuse*/
    unsigned long            reg_retry_cnt; /**< Memory registration retry count */
    unsigned                 smkey_block_size; /**< Mkey indexes in a symmetric block */
} uct_ib_md_ext_config_t;


typedef struct {
    uint32_t lkey;
    uint32_t rkey;
    uint32_t flags;
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
    uint64_t                 gva_mem_types;
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
    struct {
        uint32_t             base;
        uint32_t             size;
    } mkey_by_name_reserve;
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
    uint64_t                 devx_objs;    /**< Objects to be created by DevX */
    ucs_ternary_auto_value_t mr_relaxed_order; /**< Allow reorder memory accesses */
    int                      enable_gpudirect_rdma; /**< Enable GPUDirect RDMA */
    int                      xgvmi_umr_enable; /**< Enable UMR workflow for XGVMI */
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


typedef struct uct_ib_md_ops {
    uct_md_ops_t          super;
    uct_ib_md_open_func_t open;
} uct_ib_md_ops_t;


/**
 * IB memory domain constructor. Should have following logic:
 * - probe provided IB device, may return UCS_ERR_UNSUPPORTED
 * - allocate MD and IB context
 * - setup atomic MR ops
 * - determine device attributes and flags
 */
typedef struct uct_ib_md_ops_entry {
    ucs_list_link_t             list;
    const char                  *name;
    uct_ib_md_ops_t             *ops;
} uct_ib_md_ops_entry_t;


#define UCT_IB_MD_OPS_NAME(_name) uct_ib_md_ops_##_name##_entry

#define UCT_IB_MD_DEFINE_ENTRY(_name, _md_ops) \
    uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(_name) = { \
        .name = UCS_PP_MAKE_STRING(_md_ops), \
        .ops  = &_md_ops, \
    }

/* Used by IB module and IB sub-modules */
extern uct_component_t uct_ib_component;
extern ucs_list_link_t uct_ib_ops;


static UCS_F_ALWAYS_INLINE uint32_t uct_ib_md_direct_rkey(uct_rkey_t uct_rkey)
{
    return (uint32_t)uct_rkey;
}


static UCS_F_ALWAYS_INLINE uint32_t uct_ib_md_atomic_rkey(uct_rkey_t uct_rkey)
{
    return uct_rkey >> 32;
}


static UCS_F_ALWAYS_INLINE void
uct_ib_md_pack_rkey(uint32_t rkey, uint32_t atomic_rkey, void *rkey_buffer)
{
    uint64_t *rkey_p = (uint64_t*)rkey_buffer;

    *rkey_p = (((uint64_t)atomic_rkey) << 32) | rkey;
    ucs_trace("packed rkey: direct 0x%x atomic 0x%x", rkey, atomic_rkey);
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
    uint32_t atomic_rkey = uct_ib_md_atomic_rkey(uct_rkey);
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
    return md->flush_rkey >> 8;
}

void uct_ib_md_parse_relaxed_order(uct_ib_md_t *md,
                                   const uct_ib_md_config_t *md_config,
                                   int is_supported);

ucs_status_t uct_ib_md_query(uct_md_h uct_md, uct_md_attr_v2_t *md_attr);

ucs_status_t uct_ib_mem_advise(uct_md_h uct_md, uct_mem_h memh, void *addr,
                               size_t length, unsigned advice);

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

ucs_status_t uct_ib_reg_mr(uct_ib_md_t *md, void *address, size_t length,
                           const uct_md_mem_reg_params_t *params,
                           uint64_t access_flags, struct ibv_dm *dm,
                           struct ibv_mr **mr_p);

ucs_status_t uct_ib_dereg_mr(struct ibv_mr *mr);

ucs_status_t uct_ib_mem_prefetch(uct_ib_md_t *md, uct_ib_mem_t *ib_memh,
                                 void *addr, size_t length);

/**
 * Check if IB md device has ECE capability
 */
void uct_ib_md_ece_check(uct_ib_md_t *md);

/* Check if IB MD supports nonblocking registration */
void uct_ib_md_check_odp(uct_ib_md_t *md);

int uct_ib_md_check_odp_common(uct_ib_md_t *md, const char **reason_ptr);

ucs_status_t
uct_ib_md_handle_mr_list_mt(uct_ib_md_t *md, void *address, size_t length,
                            const uct_md_mem_reg_params_t *params,
                            uint64_t access_flags, size_t mr_num,
                            struct ibv_mr **mrs);

uint64_t uct_ib_memh_access_flags(uct_ib_mem_t *memh, int relaxed_order);

ucs_status_t uct_ib_verbs_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                  const uct_md_mem_reg_params_t *params,
                                  uct_mem_h *memh_p);

ucs_status_t uct_ib_verbs_mem_dereg(uct_md_h uct_md,
                                    const uct_md_mem_dereg_params_t *params);

ucs_status_t uct_ib_verbs_mkey_pack(uct_md_h uct_md, uct_mem_h uct_memh,
                                    void *address, size_t length,
                                    const uct_md_mkey_pack_params_t *params,
                                    void *mkey_buffer);

ucs_status_t uct_ib_rkey_unpack(uct_component_t *component,
                                const void *rkey_buffer, uct_rkey_t *rkey_p,
                                void **handle_p);

ucs_status_t uct_ib_query_md_resources(uct_component_t *component,
                                       uct_md_resource_desc_t **resources_p,
                                       unsigned *num_resources_p);

ucs_status_t uct_ib_get_device_by_name(struct ibv_device **ib_device_list,
                                       int num_devices, const char *md_name,
                                       struct ibv_device** ibv_device_p);

ucs_status_t uct_ib_fork_init(const uct_ib_md_config_t *md_config,
                              int *fork_init_p);

ucs_status_t uct_ib_memh_alloc(uct_ib_md_t *md, size_t length,
                               unsigned mem_flags, size_t memh_base_size,
                               size_t mr_size, uct_ib_mem_t **memh_p);

#endif
