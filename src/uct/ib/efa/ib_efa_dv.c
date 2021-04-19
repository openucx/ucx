/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/efa/ib_efa.h>


ucs_status_t uct_ib_efadv_query(struct ibv_context *ctx,
                                uct_ib_efadv_attr_t *efadv_attr)
{
    if (efadv_query_device(ctx, efadv_attr, sizeof(*efadv_attr))) {
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}
