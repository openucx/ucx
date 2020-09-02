/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "tcp_base.h"
#include <uct/base/uct_cm.h>


typedef struct uct_tcp_sockcm_ep   uct_tcp_sockcm_ep_t;


/**
 * A TCP connection manager
 */
typedef struct uct_tcp_sockcm {
    uct_cm_t            super;
    size_t              priv_data_len;
    size_t              sockopt_sndbuf;  /** SO_SNDBUF */
    size_t              sockopt_rcvbuf;  /** SO_RCVBUF */
    unsigned            syn_cnt;         /** TCP_SYNCNT */
    ucs_list_link_t     ep_list;         /** List of endpoints */
    int                 allow_addr_inuse;
} uct_tcp_sockcm_t;

/**
 * TCP SOCKCM configuration.
 */
typedef struct uct_tcp_sockcm_config {
    uct_cm_config_t                 super;
    size_t                          priv_data_len;
    uct_tcp_send_recv_buf_config_t  sockopt;
    unsigned                        syn_cnt;
    int                             allow_addr_inuse;
} uct_tcp_sockcm_config_t;


typedef struct uct_tcp_sockcm_priv_data_hdr {
    size_t             length;       /** Length of the private data */
    uint8_t            status;
} uct_tcp_sockcm_priv_data_hdr_t;

extern ucs_config_field_t uct_tcp_sockcm_config_table[];

UCS_CLASS_DECLARE_NEW_FUNC(uct_tcp_sockcm_t, uct_cm_t, uct_component_h,
                           uct_worker_h, const uct_cm_config_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_tcp_sockcm_t, uct_cm_t);

void uct_tcp_sa_data_handler(int fd, int events, void *arg);
