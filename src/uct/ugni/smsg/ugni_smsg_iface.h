/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_IFACE_H
#define UCT_UGNI_IFACE_H

#include <gni_pub.h>
#include <uct/ugni/base/ugni_md.h>
#include <uct/ugni/base/ugni_device.h>
#include <uct/ugni/base/ugni_iface.h>
#include "ugni_smsg_ep.h"

#define SMSG_MAX_SIZE 65535
#define UCT_UGNI_LOCAL_CQ 8192

typedef struct uct_ugni_smsg_iface {
    uct_ugni_iface_t        super;        /**< Super type */
    gni_cq_handle_t         remote_cq;    /**< Remote completion queue */
    ucs_mpool_t             free_desc;    /**< Pool of FMA descriptors for
                                               requests without bouncing buffers */
    ucs_mpool_t             free_mbox;    /**< Pool of mboxes for use with smsg */
    uint32_t                smsg_id;      /**< Id number to uniquely identify smsgs in the cq */
    struct {
        unsigned            smsg_seg_size; /**< Max SMSG size */
        size_t              rx_headroom;   /**< The size of user defined header for am */
        uint16_t            smsg_max_retransmit;
        uint16_t            smsg_max_credit; /**< Max credits for smsg boxes */
    } config;

    size_t bytes_per_mbox;
    uct_ugni_smsg_desc_t *user_desc;

    uct_ugni_smsg_desc_t *       smsg_list[UCT_UGNI_HASH_SIZE]; /**< A list of descriptors currently outstanding */
} uct_ugni_smsg_iface_t;

typedef struct uct_ugni_smsg_header {
    uint32_t length;
} uct_ugni_smsg_header_t;

ucs_status_t progress_remote_cq(uct_ugni_smsg_iface_t *iface);

#endif
