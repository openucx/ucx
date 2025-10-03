/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "device_common.h"

#include <ucs/config/global_opts.h>

const char *ucs_device_level_names[] = {"thread", "warp", "block", "grid"};

void ucs_device_log_config_init(ucs_device_log_config_t *config)
{
    config->level = ucs_global_opts.device_log_level;
}
