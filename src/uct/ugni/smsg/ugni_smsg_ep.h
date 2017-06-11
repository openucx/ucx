/**
 * Copyright (C) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_SMSG_EP_H
#define UCT_UGNI_SMSG_EP_H

#include <uct/ugni/base/ugni_types.h>
#include <uct/ugni/base/ugni_ep.h>
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <gni_pub.h>

#define UCT_UGNI_SMSG_ANY 0

typedef struct uct_ugni_compact_smsg_attr {
    gni_mem_handle_t mem_hndl;
    void *msg_buffer;
    uint32_t mbox_offset;
} UCS_S_PACKED uct_ugni_compact_smsg_attr_t;

typedef struct uct_sockaddr_smsg_ugni {
    uct_sockaddr_ugni_t super;
    uct_ugni_compact_smsg_attr_t smsg_compact_attr;
    uint32_t ep_hash;
} UCS_S_PACKED uct_sockaddr_smsg_ugni_t;

typedef struct uct_ugni_mbox_handle {
    gni_mem_handle_t gni_mem;
    uintptr_t base_address;
    gni_smsg_attr_t mbox_attr;
} uct_ugni_smsg_mbox_t;

typedef struct uct_ugni_smsg_ep {
    uct_ugni_ep_t super;
    uct_ugni_smsg_mbox_t *smsg_attr;
    uct_ugni_compact_smsg_attr_t smsg_compact_attr;
} uct_ugni_smsg_ep_t;

typedef struct uct_ugni_smsg_desc {
    uint32_t msg_id;
    uct_ugni_flush_group_t *flush_group;
    struct uct_ugni_smsg_desc *next;
} uct_ugni_smsg_desc_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_ugni_smsg_ep_t, uct_ep_t, uct_iface_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ugni_smsg_ep_t, uct_ep_t);

ucs_status_t uct_ugni_smsg_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                       const void *payload, unsigned length);
ssize_t uct_ugni_smsg_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                  uct_pack_callback_t pack_cb, void *arg,
                                  unsigned flags);
ucs_status_t uct_ugni_smsg_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr);
ucs_status_t uct_ugni_smsg_ep_connect_to_ep(uct_ep_h tl_ep,
                                            const uct_device_addr_t *dev_addr,
                                            const uct_ep_addr_t *ep_addr);

static inline uint32_t uct_ugni_smsg_desc_compare(uct_ugni_smsg_desc_t *smsg1, uct_ugni_smsg_desc_t *smsg2)
{
    return smsg1->msg_id - smsg2->msg_id;
}

static inline unsigned uct_ugni_smsg_desc_hash(uct_ugni_smsg_desc_t *smsg)
{
    return smsg->msg_id;
}

SGLIB_DEFINE_LIST_PROTOTYPES(uct_ugni_smsg_desc_t, uct_ugni_smsg_desc_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_ugni_smsg_desc_t, UCT_UGNI_HASH_SIZE, uct_ugni_smsg_desc_hash);

#endif
