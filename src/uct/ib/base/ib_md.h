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


/**
 * IB memory domain.
 */
typedef struct uct_ib_md {
    uct_md_t                 super;
    ucs_rcache_t             *rcache;   /**< Registration cache (can be NULL) */
    struct ibv_pd            *pd;       /**< IB memory domain */
    uct_ib_device_t          dev;       /**< IB device */
    uct_linear_growth_t      reg_cost;  /**< Memory registration cost */
    int                      pfc_enabled;
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
    int                     pfc_enabled;   /**< Whether or not PFC is enabled on the switch
                                                (Pause Frame Control) */

} uct_ib_md_config_t;


/**
 * IB memory region in the registration cache.
 */
typedef struct uct_ib_rcache_region {
    union {
        ucs_rcache_region_t  super;
        struct {
            /* Overlap the unused fields of stub_mr with the superclass */
            char             pad[sizeof(ucs_rcache_region_t) -
                                 __builtin_offsetof(struct ibv_mr, lkey)];
            struct ibv_mr    stub_mr;    /**< stub mr exposed to the user as the memh */
        };
    };
    struct ibv_mr            *mr __attribute__((may_alias));        /**< IB region handle */
} uct_ib_rcache_region_t;


extern uct_md_component_t uct_ib_mdc;

#endif
