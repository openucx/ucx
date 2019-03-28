/*
 *  * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 *  * Copyright (C) NVIDIA Corporation. 2019.  ALL RIGHTS RESERVED.
 *   * See file LICENSE for terms.
 *    */

#ifndef UCT_TCPCM_H
#define UCT_TCPCM_H

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

#define UCT_TCPCM_TL_NAME              "tcpcm"
#define UCT_TCPCM_UDP_PRIV_DATA_LEN    1024   /** FIXME */

typedef struct uct_tcpcm_iface   uct_tcpcm_iface_t;
typedef struct uct_tcpcm_ep      uct_tcpcm_ep_t;

typedef struct uct_tcpcm_priv_data_hdr {
    unsigned int length;     /* length of the private data */
    int8_t       status;
} uct_tcpcm_priv_data_hdr_t;

typedef struct uct_tcpcm_conn_param {
    uct_tcpcm_priv_data_hdr_t hdr;
    char                      private_data[UCT_TCPCM_UDP_PRIV_DATA_LEN];
    unsigned int              private_data_len;
} uct_tcpcm_conn_param_t;

typedef struct uct_tcpcm_ctx {
    int               sock_id;
    uct_tcpcm_ep_t    *ep;
    ucs_list_link_t   list;    /* for list of used sock_ids *FIXME* */
} uct_tcpcm_ctx_t;

ucs_status_t uct_tcpcm_ep_set_sock_id(uct_tcpcm_iface_t *iface, uct_tcpcm_ep_t *ep);

#endif /* UCT_TCPCM_H */
