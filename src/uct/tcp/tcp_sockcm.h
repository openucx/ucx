/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/base/uct_cm.h>


typedef struct uct_tcp_sockcm_ep   uct_tcp_sockcm_ep_t;


typedef struct uct_tcp_sa_arg {
    int                 fd;
    ucs_list_link_t     list;
    uct_tcp_sockcm_ep_t *ep;
} uct_tcp_sa_arg_t;


/**
 * A TCP connection manager
 */
typedef struct uct_tcp_sockcm {
    uct_cm_t            super;
    size_t              priv_data_len;
    /** List of async handler contexes. One per client */
    ucs_list_link_t     sa_arg_list;
} uct_tcp_sockcm_t;

/**
 * TCP SOCKCM configuration.
 */
typedef struct uct_tcp_sockcm_config {
    uct_cm_config_t     super;
    size_t              priv_data_len;
} uct_tcp_sockcm_config_t;


typedef struct uct_tcp_sockcm_priv_data_hdr {
    uint8_t             length;       /* length of the private data */
} uct_tcp_sockcm_priv_data_hdr_t;

extern ucs_config_field_t uct_tcp_sockcm_config_table[];

UCS_CLASS_DECLARE_NEW_FUNC(uct_tcp_sockcm_t, uct_cm_t, uct_component_h,
                           uct_worker_h, const uct_cm_config_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_tcp_sockcm_t, uct_cm_t);

void uct_tcp_sa_data_handler(int fd, void *arg);
