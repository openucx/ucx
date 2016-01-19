/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_PD_H_
#define UCT_IB_PD_H_

#include "ib_device.h"

#include <uct/base/uct_pd.h>
#include <ucs/stats/stats.h>
#include <ucs/sys/rcache.h>


enum {
    UCT_IB_PD_STAT_MEM_ALLOC,
    UCT_IB_PD_STAT_MEM_REG,
    UCT_IB_PD_STAT_LAST
};


/**
 * IB memory domain.
 */
typedef struct uct_ib_pd {
    uct_pd_t                 super;
    ucs_rcache_t             *rcache;   /**< Registration cache (can be NULL) */
    struct ibv_pd            *pd;       /**< IB protection domain */
    uct_ib_device_t          dev;       /**< IB device */
    UCS_STATS_NODE_DECLARE(stats);
} uct_ib_pd_t;


/**
 * IB memory domain configuration.
 */
typedef struct uct_ib_pd_config {
    uct_pd_config_t          super;

    struct {
        ucs_ternary_value_t  enable;       /**< Enable registration cache */
        unsigned             event_prio;   /**< Memory events priority */
        double               overhead;     /**< Lookup overhead estimation */
    } rcache;

} uct_ib_pd_config_t;


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


extern uct_pd_component_t uct_ib_pdc;

#endif
