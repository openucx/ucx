/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2023. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCX_PERFTEST_MAD_H
#define UCX_PERFTEST_MAD_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucs/type/status.h>
#include <ucs/debug/log.h>

struct perftest_context;

#ifdef HAVE_MAD
ucs_status_t setup_mad_rte(struct perftest_context *ctx);
ucs_status_t cleanup_mad_rte(struct perftest_context *ctx);
#else
static inline ucs_status_t setup_mad_rte(struct perftest_context *ctx)
{
    ucs_error("Infiniband MAD RTE transport is not supported");
    return UCS_ERR_UNSUPPORTED;
}

static inline ucs_status_t cleanup_mad_rte(struct perftest_context *ctx)
{
    return UCS_ERR_UNSUPPORTED;
}
#endif /* HAVE_MAD */
#endif /* UCX_PERFTEST_MAD_H */
