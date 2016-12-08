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
#include <ucs/sys/rcache.h>


/**
 * IB MD statistics counters
 */
enum {
    UCT_IB_MD_STAT_MEM_ALLOC,
    UCT_IB_MD_STAT_MEM_REG,
    UCT_IB_MD_STAT_LAST
};


/*
 * NUMA policy
 */
typedef enum {
    UCT_IB_NUMA_POLICY_DEFAULT,
    UCT_IB_NUMA_POLICY_BIND,
    UCT_IB_NUMA_POLICY_PREFERRED,
    UCT_IB_NUMA_POLICY_LAST
} uct_ib_numa_policy_t;


enum {
    UCT_IB_MEM_FLAG_ODP       = UCS_BIT(0),
    UCT_IB_MEM_FLAG_ATOMIC_MR = UCS_BIT(1)
};


typedef struct uct_ib_odp_config {
    uct_ib_numa_policy_t numa_policy;/**< NUMA policy flags for ODP */
    int                  prefetch;   /**< Auto-prefetch non-blocking memory
                                          registrations / allocations */
    size_t               max_size;   /**< Maximal memory region size for ODP */
} uct_ib_odp_config_t;


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
    struct ibv_pd            *pd;       /**< IB memory domain */
    uct_ib_device_t          dev;       /**< IB device */
    uct_linear_growth_t      reg_cost;  /**< Memory registration cost */
    int                      eth_pause; /**< Pause Frame on an Ethernet network */
    uct_ib_odp_config_t      odp;       /**< ODP configuration */
    /* keep it in md because pd is needed to create umr_qp/cq */
    struct ibv_qp            *umr_qp;   /* special QP for creating UMR */
    struct ibv_cq            *umr_cq;   /* special CQ for creating UMR */
    UCS_STATS_NODE_DECLARE(stats);
} uct_ib_md_t;


/**
 * IB memory domain configuration.
 */
typedef struct uct_ib_md_config {
    uct_md_config_t          super;

    struct {
        ucs_ternary_value_t  enable;       /**< Enable registration cache */
        size_t               alignment;    /**< Force address alignment */
        unsigned             event_prio;   /**< Memory events priority */
        double               overhead;     /**< Lookup overhead estimation */
    } rcache;

    uct_linear_growth_t      uc_reg_cost;  /**< Memory registration cost estimation
                                                without using the cache */

    unsigned                fork_init;     /**< Use ibv_fork_init() */
    int                     eth_pause;     /**< Whether or not Pause Frame is
                                                enabled on the Ethernet network */
    uct_ib_odp_config_t      odp;          /**< ODP configuration */
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

#endif
