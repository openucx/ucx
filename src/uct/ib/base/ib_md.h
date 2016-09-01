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

typedef struct uct_ib_memh {
    uint32_t                lkey;
    struct ibv_mr           *mr;
    struct ibv_mr           *umr;
} uct_ib_memh_t;

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
#if HAVE_EXP_UMR
    /* keep it in md because pd is needed to create umr_qp/cq */
    struct ibv_qp            *umr_qp;   /* special QP for creating UMR */
    struct ibv_cq            *umr_cq;   /* special CQ for creating UMR */
#endif
    UCS_STATS_NODE_DECLARE(stats);
} uct_ib_md_t;


/**
 * IB memory domain configuration.
 */
typedef struct uct_ib_md_config {
    uct_md_config_t          super;

    struct {
        ucs_ternary_value_t  enable;       /**< Enable registration cache */
        unsigned             event_prio;   /**< Memory events priority */
        double               overhead;     /**< Lookup overhead estimation */
    } rcache;

    uct_linear_growth_t      uc_reg_cost;  /**< Memory registration cost estimation
                                                without using the cache */

    unsigned                fork_init;     /**< Use ibv_fork_init() */
    int                     eth_pause;     /**< Whether or not Pause Frame is
                                                enabled on the Ethernet network */

} uct_ib_md_config_t;


/**
 * IB memory region in the registration cache.
 */
typedef struct uct_ib_rcache_region {
    ucs_rcache_region_t  super;
    uct_ib_memh_t        memh;      /**<  mr exposed to the user as the memh */
} uct_ib_rcache_region_t;


extern uct_md_component_t uct_ib_mdc;

/**
 * rkey is packed/unpacked is such a way that
 * high 32 bits always contain a valid key. Either a umr key
 * or a regualar one. 
 */
static inline uint32_t uct_ib_md_umr_rkey(uct_rkey_t rkey)
{
    return (uint32_t)(rkey >> 32);
}

uint8_t  uct_ib_md_umr_id(uct_ib_md_t *md);

static inline uint16_t uct_ib_md_umr_offset(uint8_t umr_id)
{
    return umr_id<<3;
}

#endif
