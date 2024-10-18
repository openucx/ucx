/**
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/ib/efa/ib_efa.h>
#include <ucs/sys/compiler_def.h>


void UCS_F_CTOR uct_efa_init(void)
{
    ucs_list_add_head(&uct_ib_ops, &UCT_IB_MD_OPS_NAME(efa).list);
}

void UCS_F_DTOR uct_efa_cleanup(void)
{
    ucs_list_del(&UCT_IB_MD_OPS_NAME(efa).list);
}
