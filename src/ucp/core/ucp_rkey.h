/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RKEY_H_
#define UCP_RKEY_H_

#include <ucp/api/ucp_def.h>
#include <ucp/core/ucp_ep.h>
#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/log.h>

#include <inttypes.h>

ucs_status_t ucp_rkey_write(ucp_context_h context,
                            ucp_md_map_t md_map, uct_mem_h *memh,
                            void *rkey_buffer, size_t *size_p);

#endif
