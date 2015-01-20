/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * $COPYRIGHT$
 * $HEADER$
 */

#include "ucs/debug/memtrack.h"
#include "ucs/type/class.h"

#include "uct/tl/context.h"
#include "mmp_iface.h"
#include "mmp_context.h"

ucs_status_t uct_mmp_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p);

ucs_config_field_t uct_mmp_iface_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_mmp_iface_config_t, super),
    UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},
    UCT_IFACE_MPOOL_CONFIG_FIELDS("FMA", -1, "fma",
                                  ucs_offsetof(uct_mmp_iface_config_t, mpool),
                                  "\nAttention: Setting this param with value
                                  != -1 is a dangerous thing\n" "and could
                                  cause deadlock or performance degradation."),

    {NULL}
};

static ucs_status_t get_cookie(uint32_t *cookie)
{
}

static ucs_status_t get_ptag(uint8_t *ptag)
{
}

ucs_status_t uct_mmp_query_resources(uct_context_h context,
                                      uct_resource_desc_t **resources_p,
                                      unsigned *num_resources_p)
{
}

ucs_status_t mmp_activate_domain(uct_mmp_context_t *mmp_ctx)
{
}

ucs_status_t uct_mmp_init(uct_context_h context)
{
}

void uct_mmp_cleanup(uct_context_t *context)
{
}
UCS_COMPONENT_DEFINE(uct_context_t, mmp, uct_mmp_init, uct_mmp_cleanup, 
                     sizeof(uct_mmp_context_t))

uct_mmp_device_t * uct_mmp_device_by_name(uct_mmp_context_t *mmp_ctx,
                                            const char *dev_name)
{
}
