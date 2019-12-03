/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCG_BUILTIN_PLAN_H
#define UCG_BUILTIN_PLAN_H

#include <ucg/api/ucg_plan_component.h>
#include <ucs/datastruct/mpool.inl>
#include <uct/api/uct.h>
//#include <ucg/api/ucg_mpi.h>


//TODO: global variable of Binomial tree selection(should be moved to algorithm selection module!!!)
unsigned BMTREE;     /* BMTREE     0: builtin tree    1: binomial tree        */
unsigned RECURSIVE;  /* RECURSIVE  0: recursive       1: topo-aware recursive */
unsigned BRUCK;      /* RECURSIVE  0: recursive       1: allgather bruck */
unsigned PIPELINE;   /* PIPELINE   0: normal send     1: pipelining send for waypoint */

enum ucg_builtin_plan_topology_type {
    UCG_PLAN_RECURSIVE,
    UCG_PLAN_TREE_FANIN,
    UCG_PLAN_TREE_FANOUT,
    UCG_PLAN_TREE_FANIN_FANOUT,
    UCG_PLAN_ALLTOALL_AGGREGATION,
    UCG_PLAN_ALLTOALL_BRCUK,
    UCG_PLAN_BRUCK,
    UCG_PLAN_LAST,
};

enum UCS_S_PACKED ucg_builtin_plan_method_type {
    UCG_PLAN_METHOD_SEND_TERMINAL,     /* Send the message(s), nothing fancy */
    UCG_PLAN_METHOD_RECV_TERMINAL,     /* Final stop for incoming messages */
    UCG_PLAN_METHOD_BCAST_WAYPOINT,    /* receive and send on to all peers */
    UCG_PLAN_METHOD_GATHER_WAYPOINT,   /* gather from all peers, and pass on */
    UCG_PLAN_METHOD_SCATTER_TERMINAL,  /* scatter to all peers in the map */
    UCG_PLAN_METHOD_SCATTER_WAYPOINT,  /* scatter and send "downwards" */
    UCG_PLAN_METHOD_REDUCE_TERMINAL,   /* receive and reduce from each peer */
    UCG_PLAN_METHOD_REDUCE_WAYPOINT,   /* receive, reduce, and pass onwards */
    UCG_PLAN_METHOD_REDUCE_RECURSIVE,  /* send+receive and reduce (RD) */
    UCG_PLAN_METHOD_NEIGHBOR,          /* "halo exchange", for neighborhood ops */

    UCG_PLAN_METHOD_ALLGATHER_BRUCK,   /* send+receive for allgather  (BRUCK) */
    UCG_PLAN_METHOD_ALLGATHER_RECURSIVE,
    UCG_PLAN_METHOD_ALLTOALL_BRUCK,    /* send+receive for alltoall   (BRUCK) */
};

typedef struct ucg_builtin_plan_phase {
    /* Parameters for buffer send/recv action */
    union {
        uct_ep_h                     *multi_eps;     /* endpoint pointer array */
        uct_ep_h                      single_ep;     /* single endpoint handle */
    };
    uint32_t                          ep_cnt;        /* Number of endpoints (below) */
    enum ucg_builtin_plan_method_type method;        /* how to apply this map */
    ucg_step_idx_t                    step_index;    /* determines step index */

    /* threshold for sender */
    size_t                            max_short_one; /* max single short message */
    size_t                            max_short_max; /* max length to use short */
    size_t                            max_bcopy_one; /* max single bcopy message */
    size_t                            max_bcopy_max; /* max length to use bcopy */
    size_t                            max_zcopy_one; /* max single zcopy message */

    /* threshold for receiver */
    size_t                            max_short_one_recv; /* max single short message */
    size_t                            max_short_max_recv; /* max length to use short */
    size_t                            max_bcopy_one_recv; /* max single bcopy message */
    size_t                            max_bcopy_max_recv; /* max length to use bcopy */
    size_t                            max_zcopy_one_recv; /* max single zcopy message */
    size_t                            md_attr_cap_max_reg_recv; /* phase->md_attr->cap.max_reg */

    uct_md_h                          md;            /* memory (registration) domain */
    const uct_md_attr_t              *md_attr;       /* memory domain attributes */
    const uct_iface_attr_t           *ep_attr;       /* endpoint attributes */

#if ENABLE_DEBUG_DATA || ENABLE_FAULT_TOLERANCE
    ucg_group_member_index_t         *indexes;       /* array corresponding to EPs */
#endif
} ucg_builtin_plan_phase_t;

typedef struct ucg_builtin_group_ctx ucg_builtin_group_ctx_t;
typedef struct ucg_builtin_plan {
    ucg_plan_t               super;
    void                    *slots;   /* slots for builtin operations */
    ucs_list_link_t         *resend;  /* per-group list of requests to resend */
    ucs_list_link_t          list;    /* member of a per-group list of plans */
    ucs_list_link_t          by_root; /* extra phases for non-zero root */
    ucs_mpool_t              op_mp;   /* memory pool for (builtin_)operations */
    ucg_step_idx_t           phs_cnt; /* number of phases in the normal flow */
    uint8_t                  ep_cnt;  /* total endpoint count */
    uint16_t                 am_id;   /* active message ID */
    size_t                   non_power_of_two; /* number of processes is power of two or not */
    size_t                   extra_indexs; /* extra_indexs for non power of two processes */
    ucg_builtin_plan_phase_t phss[];  /* topology's phases */
/*  uct_ep_h                 eps[];    * logically located here */
} ucg_builtin_plan_t;

#define UCG_BUILTIN_CONNECT_SINGLE_EP ((unsigned)-1)
ucs_status_t ucg_builtin_connect(ucg_builtin_group_ctx_t *ctx,
        ucg_group_member_index_t idx, ucg_builtin_plan_phase_t *phase,
        unsigned phase_ep_index);

typedef struct ucg_builtin_config ucg_builtin_config_t;

typedef struct ucg_builtin_tree_config {
    unsigned radix;
    unsigned sock_thresh;
} ucg_builtin_tree_config_t;
extern ucs_config_field_t ucg_builtin_tree_config_table[];
ucs_status_t ucg_builtin_tree_create(ucg_builtin_group_ctx_t *ctx,
        enum ucg_builtin_plan_topology_type plan_topo_type,
        const ucg_builtin_config_t *config,
        const ucg_group_params_t *group_params,
        const ucg_collective_type_t *coll_type,
        ucg_builtin_plan_t **plan_p);
ucs_status_t ucg_builtin_topo_tree_set_root(ucg_group_member_index_t root,
        ucg_group_member_index_t my_index,
        ucg_builtin_plan_t *plan,
        ucg_builtin_plan_phase_t **first_phase_p,
        unsigned *phase_count_p);

typedef struct ucg_builtin_binomial_tree_config {
    unsigned degree;
} ucg_builtin_binomial_tree_config_t;
extern ucs_config_field_t ucg_builtin_binomial_tree_config_table[];
ucs_status_t ucg_builtin_binomial_tree_create(ucg_builtin_group_ctx_t *ctx,
        enum ucg_builtin_plan_topology_type plan_topo_type,
        const ucg_builtin_config_t *config,
        const ucg_group_params_t *group_params,
        const ucg_collective_type_t *coll_type,
        ucg_builtin_plan_t **plan_p);

typedef struct ucg_builtin_recursive_config {
    unsigned factor;
} ucg_builtin_recursive_config_t;
extern ucs_config_field_t ucg_builtin_recursive_config_table[];
ucs_status_t ucg_builtin_recursive_create(ucg_builtin_group_ctx_t *ctx,
        enum ucg_builtin_plan_topology_type plan_topo_type,
        const ucg_builtin_config_t *config,
        const ucg_group_params_t *group_params,
        const ucg_collective_type_t *coll_type,
        ucg_builtin_plan_t **plan_p);

typedef struct ucg_builtin_bruck_config {
    unsigned factor;
} ucg_builtin_bruck_config_t;
ucs_status_t ucg_builtin_bruck_create(ucg_builtin_group_ctx_t *ctx,
    enum ucg_builtin_plan_topology_type plan_topo_type,
    const ucg_builtin_config_t *config,
    const ucg_group_params_t *group_params,
    const ucg_collective_type_t *coll_type,
    ucg_builtin_plan_t **plan_p);

typedef struct ucg_builtin_neighbor_config {
    unsigned dimension;
} ucg_builtin_neighbor_config_t;
extern ucs_config_field_t ucg_builtin_neighbor_config_table[];
ucs_status_t ucg_topo_neighbor_create(ucg_builtin_group_ctx_t *ctx,
        enum ucg_builtin_plan_topology_type plan_topo_type,
        const ucg_builtin_config_t *config,
        const ucg_group_params_t *group_params,
        const ucg_collective_type_t *coll_type,
        ucg_builtin_plan_t **plan_p);

struct ucg_builtin_config {
    ucg_plan_config_t    super;

    ucg_builtin_tree_config_t          tree;
    ucg_builtin_binomial_tree_config_t bmtree;
    ucg_builtin_recursive_config_t     recursive;
    ucg_builtin_neighbor_config_t      neighbor;
    ucg_builtin_bruck_config_t         bruck;

    unsigned                       cache_size;
    size_t                         short_max_tx;
    size_t                         bcopy_max_tx;
    unsigned                       mem_reg_opt_cnt;
    
    unsigned                       pipelining;
};

#endif
