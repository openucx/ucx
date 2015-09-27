/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_UD_LOG_H
#define UCT_UD_LOG_H

#include "ud_def.h"

#include <uct/tl/tl_log.h>
#include <ucs/debug/log.h>


void uct_ud_log_packet(const char *file, int line, const char *function,
                       uct_ud_iface_t *iface, uct_ud_ep_t *ep,
                       uct_am_trace_type_t type, uct_ud_neth_t *neth, uint32_t len);


#define uct_ud_ep_log_tx(_iface, _ep, _skb) \
    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        uct_ud_log_packet(__FILE__, __LINE__, __FUNCTION__, _iface, _ep, \
                          UCT_AM_TRACE_TYPE_SEND, (_skb)->neth, (_skb)->len); \
    }

#define uct_ud_iface_log_rx(_iface, _ep, _neth, _len) \
    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        uct_ud_log_packet(__FILE__, __LINE__, __FUNCTION__, _iface, _ep, \
                          UCT_AM_TRACE_TYPE_RECV, _neth, _len); \
    }


#endif 
