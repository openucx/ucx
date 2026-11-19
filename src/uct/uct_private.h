/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_PRIVATE_H_
#define UCT_PRIVATE_H_

#include <uct/api/uct.h>

BEGIN_C_DECLS

/* Internal operations used by UCP. */
int uct_iface_has_put_short_comp(uct_iface_h iface);

ucs_status_t uct_ep_put_short_comp(uct_ep_h ep, const void *buffer,
                                  unsigned length, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp);

END_C_DECLS

#endif
