/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/base/uct_cm.h>

#define UCT_SOCKCM_CM_PRIV_DATA_LEN  1024  /** Closest power of two to common MTU value in TCP (1500) */

/**
 * A sockcm connection manager
 */
typedef struct uct_sockcm_cm {
    uct_cm_t super;
} uct_sockcm_cm_t;

typedef struct uct_sockcm_priv_data_hdr {
    uint8_t  length;     /* length of the private data */
} uct_sockcm_priv_data_hdr_t;

UCS_CLASS_DECLARE(uct_sockcm_cm_t, uct_component_h, uct_worker_h);
UCS_CLASS_DECLARE_NEW_FUNC(uct_sockcm_cm_t, uct_cm_t, uct_component_h, uct_worker_h);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_sockcm_cm_t, uct_cm_t);

