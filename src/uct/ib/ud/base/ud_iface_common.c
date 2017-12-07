/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ud_iface_common.h"

#include <ucs/sys/compiler.h>


ucs_config_field_t uct_ud_iface_common_config_table[] = {
  {"RX_QUEUE_LEN_INIT", "128",
   "Initial length of receive queue, before the interface is activated.",
   ucs_offsetof(uct_ud_iface_common_config_t, rx_queue_len_init),
   UCS_CONFIG_TYPE_UINT},

  {NULL}
};
