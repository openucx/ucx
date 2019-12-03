/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCG_PLAN_COMPONENT_H_
#define UCG_PLAN_COMPONENT_H_

#include <string.h>

#include "ucg.h"

#include <uct/api/uct.h>
#include <ucs/config/parser.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/list_types.h>
#include <ucs/datastruct/queue_types.h>

BEGIN_C_DECLS


typedef uint16_t ucg_group_id_t; /* unique */
typedef uint8_t  ucg_coll_id_t;  /* cyclic */
typedef uint8_t  ucg_step_idx_t;
typedef uint32_t ucg_offset_t;
typedef struct ucg_plan_component ucg_plan_component_t;
extern ucs_list_link_t ucg_plan_components_list;

/**
 * @ingroup UCG_RESOURCE
 * @brief Collectives' estimation of latency.
 *
 * This structure describes ...
 */
typedef struct ucg_plan_plogp_params {
    /* overhead time - per message and per byte, in seconds */
    struct {
        double sec_per_message;
        double sec_per_byte;
    } send, recv, gap;

    /* p2p latency, in seconds, by distance (assumes uniform network) */
    double latency_in_sec[UCG_GROUP_MEMBER_DISTANCE_LAST];

    /* number of peers on each level */
    ucg_group_member_index_t peer_count[UCG_GROUP_MEMBER_DISTANCE_LAST];
} ucg_plan_plogp_params_t;
typedef double (*ucg_plan_estimator_f)(ucg_plan_plogp_params_t plogp,
                                       ucg_collective_params_t *coll);

/*
 * Error-handling modes, which can be employed for fault-tolerance.
 */
enum ucg_plan_ft_mode {
    UCG_PLAN_FT_IGNORE = 0,
};

/**
 * @ingroup UCG_RESOURCE
 * @brief Collective planning resource descriptor.
 *
 * This structure describes a collective operation planning resource.
 */
enum ucg_plan_flags {
    UCG_PLAN_FLAG_PLOGP_LATENCY_ESTIMATOR = 0, /*< Supports PlogP latency estimation */
    UCG_PLAN_FLAG_FAULT_TOLERANCE_SUPPORT = 1, /*< Supported custom fault tolerance */
};

/**
 * @ingroup UCG_RESOURCE
 * @brief Collective planning resource descriptor.
 *
 * This structure describes a collective operation planning resource.
 */
#define UCG_PLAN_COMPONENT_NAME_MAX (16)
typedef struct ucg_plan_desc {
    char                  plan_name[UCG_PLAN_COMPONENT_NAME_MAX]; /**< Name */
    ucg_plan_component_t *plan_component;            /*< Component object */
    unsigned              modifiers_supported;       /*< @ref enum ucg_collective_modifiers */
    unsigned              flags;                     /*< @ref enum ucg_plan_flags */

    /* Optional parameters - depending on flags */
    ucg_plan_estimator_f  latency_estimator;         /*< @ref ucg_plan_estimator_f */
    unsigned              fault_tolerance_supported; /*< @ref enum ucg_plan_ft_mode */
} ucg_plan_desc_t;

/**
 * "Base" structure which defines planning configuration options.
 * Specific planning components extend this structure.
 *
 * Note: which components are actualy enabled/used is a configuration for UCG.
 */
typedef struct ucg_plan_config {
    enum ucg_plan_ft_mode ft;
} ucg_plan_config_t;
extern ucs_config_field_t ucg_plan_config_table[];

typedef struct ucg_plan {
    /* Plan lookup - caching mechanism */
    ucg_collective_type_t    type;
    ucs_list_link_t          op_head;   /**< List of requests following this plan */

    /* Plan progress */
    ucg_plan_component_t    *planner;
    ucg_group_id_t           group_id;
    ucg_group_member_index_t my_index;
    ucg_group_h              group;
    ucs_mpool_t             *am_mp;
    //char                     priv[0];
} ucg_plan_t;

enum ucg_request_common_flags {
    UCG_REQUEST_COMMON_FLAG_COMPLETED = UCS_BIT(0),

    UCG_REQUEST_COMMON_FLAG_MASK = UCS_MASK(1)
};

typedef struct ucg_request {
    volatile uint32_t        flags;      /**< @ref enum ucg_request_common_flags */
    volatile ucs_status_t    status;     /**< Operation status */
} ucg_request_t;

typedef struct ucg_op {
    /* Collective-specific request content */
    union {
        ucs_list_link_t      list;        /**< cache list member */
        struct {
            ucs_queue_elem_t queue;       /**< pending queue member */
            ucg_request_t  **pending_req; /**< original invocation request */
        };
    };

    ucg_plan_t              *plan;        /**< The group this belongs to */
    ucg_collective_params_t  params;      /**< original parameters for it */

    /* Component-specific request content */
    //char                     priv[0];
} ucg_op_t;

struct ucg_plan_component {
    /* test for support and other attribures of this component */
    ucs_status_t           (*query)   (unsigned ucg_api_version,
                                       ucg_plan_desc_t **resources_p,
                                       unsigned *nums_p);
    /* create a new planner context for a group */
    ucs_status_t           (*create)  (ucg_plan_component_t *plan_component,
                                       ucg_worker_h worker,
                                       ucg_group_h group,
                                       unsigned base_am_id,
                                       ucg_group_id_t group_id,
                                       ucs_mpool_t *group_am_mp,
                                       const ucg_group_params_t *group_params);
    /* destroy a group context, along with all its operations and requests */
    void                   (*destroy) (ucg_group_h group);
    /* destroy a group context, along with all its operations and requests */
    unsigned               (*progress)(ucg_group_h group);

    /* plan a collective operation with this component */
    ucs_status_t           (*plan)    (ucg_plan_component_t *plan_component,
                                       const ucg_collective_type_t *coll_type,
                                       ucg_group_h group,
                                       ucg_plan_t **plan_p);
    /* Prepare an operation to follow the given plan */
    ucs_status_t           (*prepare) (ucg_plan_t *plan,
                                       const ucg_collective_params_t *coll_params,
                                       ucg_op_t **op);
    /* Trigger an operation to start, generate a request handle for updates */
    ucs_status_t           (*trigger) (ucg_op_t *op,
                                       ucg_coll_id_t coll_id,
                                       ucg_request_t **request);
    /* Discard an operation previously prepared */
    void                   (*discard) (ucg_op_t *op);

    /* print a plan object, for debugging purposes */
    void                   (*print)   (ucg_plan_t *plan,
                                       const ucg_collective_params_t *coll_params);

    const char               name[UCG_PLAN_COMPONENT_NAME_MAX];
    const char              *cfg_prefix;        /**< prefix for configuration environment vars */
    ucs_config_field_t      *plan_config_table; /**< defines plan configuration options */
    void                    *plan_config;       /**< component configuration values */
    size_t                   plan_config_size;  /**< plan configuration structure size */
    size_t                   group_context_size;/**< size to be allocated with each group */
    ucs_list_link_t          list;

    /* Filled By UCG core, not by the component itself */
    size_t                   worker_ctx_offset; /**< offset between ucg_worker_h and my context */
    size_t                   group_ctx_offset;  /**< offset between ucg_group_h and my context */
};

/**
 * For Active-Message handlers - the Macros below allow translation from
 * ucp_worker_h (the AM argument) to the component's context.
 */
#define UCG_WORKER_TO_COMPONENT_CTX(comp, worker) \
    ((void*)((char*)(worker) + (comp).worker_ctx_offset))
#define UCG_GROUP_TO_COMPONENT_CTX(comp, group) \
    ((void*)((char*)(group) + (comp).group_ctx_offset))


/**
 * Define a planning component.
 *
 * @param _planc         Planning component structure to initialize.
 * @param _name          Planning component name.
 * @param _query         Function to query planning resources.
 * @param _prepare       Function to prepare an operation according to a plan.
 * @param _trigger       Function to start a prepared collective operation.
 * @param _destroy       Function to release a plan and all related objects.
 * @param _priv          Custom private data.
 * @param _cfg_prefix    Prefix for configuration environment variables.
 * @param _cfg_table     Defines the planning component's configuration values.
 * @param _cfg_struct    Planning component configuration structure.
 */
#define UCG_PLAN_COMPONENT_DEFINE(_planc, _name, _sz, _query, _create, _destroy,\
                                  _progress, _plan, _prepare, _trigger, _discard,\
                                  _print, _cfg_prefix, _cfg_table, _cfg_struct)\
                                                                               \
    ucg_plan_component_t _planc = {                                            \
        .group_context_size = _sz,                                             \
        .query              = _query,                                          \
        .create             = _create,                                         \
        .destroy            = _destroy,                                        \
        .progress           = _progress,                                       \
        .plan               = _plan,                                           \
        .prepare            = _prepare,                                        \
        .trigger            = _trigger,                                        \
        .discard            = _discard,                                        \
        .print              = _print,                                          \
        .cfg_prefix         = _cfg_prefix,                                     \
        .plan_config_table  = _cfg_table,                                      \
        .plan_config_size   = sizeof(_cfg_struct),                             \
        .name               = _name                                            \
    };                                                                         \
    UCS_STATIC_INIT {                                                          \
        ucs_list_add_tail(&ucg_plan_components_list, &_planc.list);            \
    }                                                                          \
    UCS_CONFIG_REGISTER_TABLE(_cfg_table, _name" planner", _cfg_prefix,        \
                              _cfg_struct)

/* Helper function to generate a simple planner description */
ucs_status_t ucg_plan_single(ucg_plan_component_t *planc,
                             ucg_plan_desc_t **resources_p,
                             unsigned *nums_p);

/* Helper function for connecting to other group members - by their index */
typedef ucs_status_t (*ucg_plan_reg_handler_cb)(uct_iface_h iface, void *arg);
ucs_status_t ucg_plan_connect(ucg_group_h group, ucg_group_member_index_t idx,
                              uct_ep_h *ep_p, const uct_iface_attr_t **ep_attr_p,
                              uct_md_h* md_p, const uct_md_attr_t** md_attr_p);

/* Helper function for selecting other planners - to be used as fall-back */
ucs_status_t ucg_plan_select(ucg_group_h group, const char* planner_name,
                             const ucg_collective_params_t *params,
                             ucg_plan_component_t **planc_p);

/* Start pending operations after a barrier has been completed */
ucs_status_t ucg_collective_release_barrier(ucg_group_h group);

END_C_DECLS

#endif
