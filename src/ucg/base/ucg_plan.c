/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucg_plan.h"

#include <ucg/api/ucg_mpi.h>
#include <ucs/config/parser.h>
#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <ucs/sys/module.h>
#include <ucs/sys/string.h>
#include <ucs/arch/cpu.h>

UCS_LIST_HEAD(ucg_plan_components_list);

ucs_config_field_t ucg_plan_config_table[] = {
  {NULL}
};

/**
 * Keeps information about allocated configuration structure, to be used when
 * releasing the options.
 */
typedef struct ucg_config_bundle {
    ucs_config_field_t *table;
    const char         *table_prefix;
    char               data[];
} ucg_config_bundle_t;

static ucs_status_t ucg_plan_config_read(ucg_config_bundle_t **bundle,
                                         ucs_config_field_t *config_table,
                                         size_t config_size, const char *env_prefix,
                                         const char *cfg_prefix)
{
    ucg_config_bundle_t *config_bundle;
    ucs_status_t status;

    config_bundle = ucs_calloc(1, sizeof(*config_bundle) + config_size, "uct_config");
    if (config_bundle == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = ucs_config_parser_fill_opts(config_bundle->data, config_table,
                                         env_prefix, cfg_prefix, 0);
    if (status != UCS_OK) {
        goto err_free_bundle;
    }

    config_bundle->table = config_table;
    config_bundle->table_prefix = ucs_strdup(cfg_prefix, "uct_config");
    if (config_bundle->table_prefix == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_bundle;
    }

    *bundle = config_bundle;
    return UCS_OK;

err_free_bundle:
    ucs_free(config_bundle);
err:
    return status;
}

ucs_status_t ucg_plan_query(ucg_plan_desc_t **resources_p, unsigned *nums_p)
{
    UCS_MODULE_FRAMEWORK_DECLARE(ucg);
    ucg_plan_desc_t *resources=NULL, *planners=NULL, *tmp=NULL;
    unsigned i, nums, num_plans;
    ucg_plan_component_t *planc=NULL;
    ucs_status_t status;

    UCS_MODULE_FRAMEWORK_LOAD(ucg, 0);

    nums = 0;

    ucs_list_for_each(planc, &ucg_plan_components_list, list) {
        status = planc->query(UCG_API_VERSION, &planners, &num_plans);
        if (status != UCS_OK) {
            ucs_debug("Failed to query %s* resources: %s", planc->name,
                      ucs_status_string(status));
            continue;
        }

        if (num_plans == 0) {
            ucs_free(planners);
            continue;
        }

        tmp = ucs_realloc(resources,
                          sizeof(*resources) * (nums + num_plans),
                          "planners");
        if (tmp == NULL) {
            ucs_free(planners);
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }

        for (i = 0; i < num_plans; ++i) {
            ucs_assertv_always(!strncmp(planc->name, planners[i].plan_name,
                                       strlen(planc->name)),
                               "Planner name must begin with topology component name."
                               "Planner name: %s Plan component name: %s ",
                               planners[i].plan_name, planc->name);

            /* read component's configuration */
            ucg_config_bundle_t *bundle = NULL;
            status = ucg_plan_config_read(&bundle, planc->plan_config_table,
                                          planc->plan_config_size, NULL,
                                          planc->cfg_prefix);
            ucs_assert(bundle != NULL);
            planc->plan_config = bundle->data; // TODO:
        }
        resources = tmp;
        memcpy(resources + nums, planners,
               sizeof(*planners) * num_plans);
        nums += num_plans;
        ucs_free(planners);
    }

    *resources_p     = resources;
    *nums_p = nums;
    return UCS_OK;

err:
    ucs_free(resources);
    return status;
}

void ucg_plan_release_list(ucg_plan_desc_t *resources, unsigned resource_cnt)
{
    unsigned i;
    for (i = 0; i < resource_cnt; i++) {
        ucg_plan_desc_t *plan_desc = &resources[i];
        ucg_config_bundle_t *bundle =
                ucs_container_of(plan_desc->plan_component->plan_config,
                        ucg_config_bundle_t, data);

        ucs_config_parser_release_opts(bundle->data, bundle->table);
        ucs_free((void*)(bundle->table_prefix));
        ucs_free(bundle);
    }

    ucs_free(resources);
}

ucs_status_t ucg_plan_single(ucg_plan_component_t *planc,
                             ucg_plan_desc_t **resources_p,
                             unsigned *nums_p)
{
    ucg_plan_desc_t *resource;

    resource = ucs_malloc(sizeof(*resource), "planner description");
    if (resource == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_snprintf_zero(resource->plan_name, UCG_PLAN_COMPONENT_NAME_MAX, "%s", planc->name);

    resource->plan_component = planc;
    *resources_p             = resource;
    *nums_p                  = 1;

    return UCS_OK;
}

ucs_status_t ucg_plan_select_component(ucg_plan_desc_t *planners,
                                       unsigned num_planners,
                                       const char* planner_name,
                                       const ucg_group_params_t *group_params,
                                       const ucg_collective_params_t *coll_params,
                                       ucg_plan_component_t **planc_p)
{
    if (planner_name && strcmp(planner_name, planners[0].plan_name)) {
        ucs_error("Unknown planner component name: \"%s\"", planner_name);
        return UCS_ERR_INVALID_PARAM;
    }

    *planc_p = planners[0].plan_component;
    return UCS_OK;
}
