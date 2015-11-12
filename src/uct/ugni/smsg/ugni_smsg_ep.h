/**
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_SMSG_EP_H
#define UCT_UGNI_SMSG_EP_H

#include <gni_pub.h>
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>
#include <uct/ugni/base/ugni_ep.h>

#define UCT_UGNI_SMSG_ANY 0

typedef struct uct_sockaddr_smsg_ugni {
    uct_sockaddr_ugni_t super;
    uint64_t ep_hash;
    gni_smsg_attr_t smsg_attr;
} uct_sockaddr_smsg_ugni_t;

typedef struct uct_ugni_mbox_handle {
    gni_mem_handle_t gni_mem;
    uintptr_t base_address;
    gni_smsg_attr_t mbox_attr;
} uct_ugni_smsg_mbox_t;

typedef struct uct_ugni_smsg_ep {
    uct_ugni_ep_t super;
    uct_ugni_smsg_mbox_t *smsg_attr;
} uct_ugni_smsg_ep_t;

typedef struct uct_ugni_smsg_desc {
    uint32_t msg_id;
    uct_ugni_ep_t  *ep;
    struct uct_ugni_smsg_desc *next;
} uct_ugni_smsg_desc_t;

SGLIB_DEFINE_LIST_PROTOTYPES(uct_ugni_smsg_desc_t, uct_ugni_smsg_desc_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_ugni_smsg_desc_t, UCT_UGNI_HASH_SIZE, uct_ugni_smsg_desc_hash);

static inline uint32_t uct_ugni_smsg_desc_compare(uct_ugni_smsg_desc_t *smsg1, uct_ugni_smsg_desc_t *smsg2)
{
    return smsg1->msg_id - smsg2->msg_id;
}

static inline unsigned uct_ugni_smsg_desc_hash(uct_ugni_smsg_desc_t *smsg)
{
    return smsg->msg_id;
}

ucs_status_t uct_ugni_smsg_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                       const void *payload, unsigned length);
ssize_t uct_ugni_smsg_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                  uct_pack_callback_t pack_cb, void *arg);
ucs_status_t uct_ugni_smsg_ep_get_address(uct_ep_h tl_ep, struct sockaddr *addr);
ucs_status_t uct_ugni_smsg_ep_connect_to_ep(uct_ep_h tl_ep, const struct sockaddr *addr);
UCS_CLASS_DECLARE_NEW_FUNC(uct_ugni_smsg_ep_t, uct_ep_t, uct_iface_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ugni_smsg_ep_t, uct_ep_t);
#endif
