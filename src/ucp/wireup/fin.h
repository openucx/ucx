/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef FIN_H_
#define FIN_H_

#include <ucp/core/ucp_ep.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>

/**
 * Packet structure for FIN requests.
 */
typedef struct ucp_fin_msg {
    uint64_t        ep_id;     /* EP address or worker uuid */
    uint8_t         is_ptr;    /* Flag if ep_id is EP address */
} UCS_S_PACKED ucp_fin_msg_t;

ucs_status_t ucp_fin_msg_send(ucp_ep_h ep);

#endif /* FIN_H_ */

