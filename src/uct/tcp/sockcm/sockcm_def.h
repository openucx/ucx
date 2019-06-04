/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SOCKCM_H
#define UCT_SOCKCM_H

#include <uct/api/uct.h>
#include <uct/api/uct_def.h>
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <ucs/type/class.h>
#include <ucs/time/time.h>
#include <ucs/async/async.h>
#include <sys/poll.h>
#include <ucs/sys/sock.h>
#include <net/if.h>

#define UCT_SOCKCM_TL_NAME              "sockcm"
#define UCT_SOCKCM_PRIV_DATA_LEN        1024

typedef struct uct_sockcm_iface   uct_sockcm_iface_t;
typedef struct uct_sockcm_ep      uct_sockcm_ep_t;

typedef struct uct_sockcm_priv_data_hdr {
    unsigned int length;     /* length of the private data */
    int8_t       status;
} uct_sockcm_priv_data_hdr_t;

typedef struct uct_sockcm_conn_param {
    uct_sockcm_priv_data_hdr_t hdr;
    char                       private_data[UCT_SOCKCM_PRIV_DATA_LEN];
    unsigned int               private_data_len;
} uct_sockcm_conn_param_t;

typedef struct uct_sockcm_ctx {
    int               sock_id;
    ucs_list_link_t   list;
} uct_sockcm_ctx_t;

ucs_status_t uct_sockcm_ep_set_sock_id(uct_sockcm_iface_t *iface, uct_sockcm_ep_t *ep);

#endif /* UCT_SOCKCM_H */
