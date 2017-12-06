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

typedef void (ucp_ep_rkey_read_cb_t)(unsigned remote_md_index, unsigned rkey_index,
                                     uct_rkey_bundle_t *rkey, void *data);

ucs_status_t ucp_ep_rkey_read(ucp_ep_h ep, void *rkey_buffer,
                              ucp_ep_rkey_read_cb_t cb, void *data);

ucs_status_t ucp_rkey_write(ucp_context_h context, ucp_mem_h memh,
                            void *rkey_buffer, size_t *size_p);

size_t ucp_rkey_packed_rkey_size(size_t key_size);

#endif
