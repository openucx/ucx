/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "uct_cm.h"

#include <ucs/sys/math.h>
#include <uct/base/uct_md.h>


ucs_status_t uct_cm_open(const uct_cm_params_t *params, uct_cm_h *cm_p)
{
    uct_md_component_t *mdc;
    ucs_status_t status;

    if (!ucs_test_all_flags(params->field_mask,
                            UCT_CM_PARAM_FIELD_MD_NAME |
                            UCT_CM_PARAM_FIELD_WORKER)) {
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_find_md_component(params->md_name, &mdc);
    if (status != UCS_OK) {
        return status;
    }
    return mdc->cm_open(params, cm_p);
}

void uct_cm_close(uct_cm_h cm)
{
    cm->ops->close(cm);
}

ucs_status_t uct_cm_query(uct_cm_h cm, uct_cm_attr_t *cm_attr)
{
    return cm->ops->cm_query(cm, cm_attr);
}
