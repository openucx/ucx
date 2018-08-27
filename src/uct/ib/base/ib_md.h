/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
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
#include <ucs/sys/numa.h>
#include <ucs/sys/rcache.h>

#define UCT_IB_MD_MAX_MR_SIZE       0x80000000UL
#define UCT_IB_MD_PACKED_RKEY_SIZE  sizeof(uint64_t)

#define UCT_IB_MD_DEFAULT_GID_INDEX 0   /**< The gid index used by default for an IB/RoCE port */

/**
 * IB MD statistics counters
 */
enum {
    UCT_IB_MD_STAT_MEM_ALLOC,
    UCT_IB_MD_STAT_MEM_REG,
    UCT_IB_MD_STAT_LAST
};


enum {
    UCT_IB_MEM_FLAG_ODP             = UCS_BIT(0), /**< The memory region has on
                                                       demand paging enabled */
    UCT_IB_MEM_FLAG_ATOMIC_MR       = UCS_BIT(1), /**< The memory region has UMR
                                                       for the atomic access */
    UCT_IB_MEM_ACCESS_REMOTE_ATOMIC = UCS_BIT(2)  /**< An atomic access was 
                                                       requested for the memory
                                                       region */
};


typedef struct uct_ib_md_ext_config {
    int                      eth_pause;    /**< Whether or not Pause Frame is
                                                enabled on the Ethernet network */
    int                      prefer_nearest_device; /**< Give priority for near
                                                         device */
    int                      enable_umr_pack;
    int                      enable_contig_pages; /** Enable contiguous pages */
    int                      enable_indirect_atomic; /** Enable indirect atomic */
    int                      enable_gpudirect_rdma; /** Enable GPUDirect RDMA */

    struct {
        ucs_numa_policy_t    numa_policy;  /**< NUMA policy flags for ODP */
        int                  prefetch;     /**< Auto-prefetch non-blocking memory
                                                registrations / allocations */
        size_t               max_size;     /**< Maximal memory region size for ODP */
    } odp;

    size_t                   gid_index;    /**< IB GID index to use  */
} uct_ib_md_ext_config_t;


typedef struct uct_ib_mem {
    uint32_t                lkey;
    uint32_t                flags;
    struct ibv_mr           *mr;
    struct ibv_mr           *atomic_mr;
} uct_ib_mem_t;

/**
 * IB memory domain.
 */
typedef struct uct_ib_md {
    uct_md_t                 super;
    ucs_rcache_t             *rcache;   /**< Registration cache (can be NULL) */
    uct_ib_mem_t             global_odp;/**< Implicit ODP memory handle */
    struct ibv_pd            *pd;       /**< IB memory domain */
    uct_ib_device_t          dev;       /**< IB device */
    uct_linear_growth_t      reg_cost;  /**< Memory registration cost */
    /* keep it in md because pd is needed to create umr_qp/cq */
    struct ibv_qp            *umr_qp;   /* special QP for creating UMR */
    struct ibv_cq            *umr_cq;   /* special CQ for creating UMR */
    UCS_STATS_NODE_DECLARE(stats);
    uct_ib_md_ext_config_t   config;    /* IB external configuration */
    struct {
        uct_ib_device_spec_t *specs;    /* Custom device specifications */
        unsigned             count;     /* Number of custom devices */
    } custom_devices;
    int                      check_subnet_filter;
    uint64_t                 subnet_filter;
} uct_ib_md_t;


/**
 * IB memory domain configuration.
 */
typedef struct uct_ib_md_config {
    uct_md_config_t          super;

    /** List of registration methods in order of preference */
    UCS_CONFIG_STRING_ARRAY_FIELD(rmtd) reg_methods;

    uct_md_rcache_config_t   rcache;       /**< Registration cache config */
    uct_linear_growth_t      uc_reg_cost;  /**< Memory registration cost estimation
                                                without using the cache */
    unsigned                 fork_init;    /**< Use ibv_fork_init() */
    int                      async_events; /**< Whether async events should be delivered */

    uct_ib_md_ext_config_t   ext;          /**< External configuration */

    UCS_CONFIG_STRING_ARRAY_FIELD(spec) custom_devices; /**< Custom device specifications */

    char                     *subnet_prefix; /**< Filter of subnet_prefix for IB ports */
} uct_ib_md_config_t;


/**
 * IB memory region in the registration cache.
 */
typedef struct uct_ib_rcache_region {
    ucs_rcache_region_t  super;
    uct_ib_mem_t         memh;      /**<  mr exposed to the user as the memh */
} uct_ib_rcache_region_t;


extern uct_md_component_t uct_ib_mdc;


/**
 * Calculate unique id for atomic
 */
uint8_t uct_ib_md_get_atomic_mr_id(uct_ib_md_t *md);


static inline uint32_t uct_ib_md_direct_rkey(uct_rkey_t uct_rkey)
{
    return (uint32_t)uct_rkey;
}


static uint32_t uct_ib_md_indirect_rkey(uct_rkey_t uct_rkey)
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
    if (atomic_rkey == UCT_IB_INVALID_RKEY) {
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


void uct_ib_make_md_name(char md_name[UCT_MD_NAME_MAX], struct ibv_device *device);

ucs_status_t
uct_ib_md_open(const char *md_name, const uct_md_config_t *uct_md_config, uct_md_h *md_p);

void uct_ib_md_close(uct_md_h uct_md);

#endif
