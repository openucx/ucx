/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <dlfcn.h>

#include <ucs/debug/log.h>
#include "uct_ib_plugin.h"


#define UCT_IB_PLUGIN_STUB(_ret_type, _name, _default_ret, _params, _args) \
    typedef _ret_type (*_name##_fn_t) _params;                              \
    _ret_type _name _params                                                 \
    {                                                                       \
        _name##_fn_t real = (_name##_fn_t)dlsym(RTLD_DEFAULT, #_name);      \
        if (real == _name) {                                                \
            return _default_ret;                                            \
        }                                                                   \
        return real _args;                                                  \
    }

UCT_IB_PLUGIN_STUB(uint64_t, uct_ib_plugin_iface_flags, 0,
                   (void), ())

UCT_IB_PLUGIN_STUB(ucs_status_t, uct_ib_plugin_qp_query, UCS_ERR_UNSUPPORTED,
                   (const uct_ib_plugin_qp_query_params_t *params,
                    uct_ib_plugin_qp_query_attr_t *attr),
                   (params, attr))
