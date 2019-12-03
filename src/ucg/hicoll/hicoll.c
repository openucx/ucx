/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

/*
 * This file is a stub, calling the Huawei Collective Operations library, if
 * available. To ensure compatibility, the built UCX version must match the one
 * used when building the HiColl library.
 */
#if WIP
UCG_PLAN_COMPONENT_DEFINE(ucg_hicoll_component, "HiColl", 0, hicoll_ucx_query,
                          hicoll_ucx_create,   hicoll_ucx_destroy,
                          hicoll_ucx_progress, hicoll_ucx_plan,
                          hicoll_ucx_prepare,  hicoll_ucx_trigger,
                          hicoll_ucx_discard,  hicoll_ucx_print, "HICOLL_",
                          hicoll_ucx_topo_config_table,
                          hicoll_ucx_topo_config_t);
#endif
