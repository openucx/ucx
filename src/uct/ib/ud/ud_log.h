/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_UD_LOG_H
#define UCT_UD_LOG_H

#include "ud_def.h"

#include <uct/tl/tl_log.h>
#include <ucs/debug/log.h>


void __uct_ud_log_packet(const char *file, int line, const char *function,
                         char *tag, 
                         uct_ud_iface_t *iface, uct_ud_ep_t *ep, 
                         uct_ud_neth_t *neth, uint32_t pkt_len);

#define uct_ud_ep_log_tx_tag(tag, ep, neth, len) \
    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ud_log_packet(__FILE__, __LINE__, __FUNCTION__, tag, ucs_derived_of(&(ep)->super.super.iface, uct_ud_iface_t), ep, neth, len); \
    }

#define uct_ud_ep_log_tx(ep, skb) uct_ud_ep_log_tx_tag("TX", ep, skb->neth, skb->len) 

#define uct_ud_iface_log_rx(iface, ep, neth, len) \
    if (ucs_log_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        __uct_ud_log_packet(__FILE__, __LINE__, __FUNCTION__, "RX", iface, ep, neth, len); \
    }


#endif 
