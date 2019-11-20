/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/base/uct_cm.h>


/**
 * SOCKCM CM configuration.
 */
typedef struct uct_sockcm_cm_config {
    uct_cm_config_t  super;
    int              priv_data_len;
} uct_sockcm_cm_config_t;


/**
 * A sockcm connection manager
 */
typedef struct uct_sockcm_cm {
    uct_cm_t        super;
    int             priv_data_len;
} uct_sockcm_cm_t;


typedef struct uct_sockcm_priv_data_hdr {
    uint8_t         length;       /* length of the private data */
} uct_sockcm_priv_data_hdr_t;


extern uct_component_t uct_sockcm_component;
extern ucs_config_field_t uct_sockcm_cm_config_table[];

UCS_CLASS_DECLARE(uct_sockcm_cm_t, uct_component_h, uct_worker_h, const uct_cm_config_t*);
UCS_CLASS_DECLARE_NEW_FUNC(uct_sockcm_cm_t, uct_cm_t, uct_component_h,
                           uct_worker_h, const uct_cm_config_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_sockcm_cm_t, uct_cm_t);

