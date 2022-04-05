/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/efa/ib_efa.h>


int uct_ib_efadv_check(struct ibv_device *ibv_device)
{
    struct efadv_device_attr efadv_attr;
    struct ibv_context *ctx = ibv_open_device(ibv_device);
    if (ctx == NULL) {
        ucs_diag("ibv_open_device(%s) failed: %m",
                 ibv_get_device_name(ibv_device));
        return 0;
    }

    if (efadv_query_device(ctx, &efadv_attr, sizeof(efadv_attr))) {
        return 0;
    }

    ibv_close_device(ctx);
    return 1;
}

ucs_status_t uct_ib_efadv_query(struct ibv_context *ctx,
                                struct efadv_device_attr *efadv_attr)
{
    if (efadv_query_device(ctx, efadv_attr, sizeof(*efadv_attr))) {
        ucs_error("efadv_query_device failed for EFA device %s %m",
                  ibv_get_device_name(ctx->device));
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}
