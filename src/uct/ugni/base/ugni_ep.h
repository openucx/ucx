#ifndef UCT_UGNI_EP_H
#define UCT_UGNI_EP_H

#include <gni_pub.h>
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/datastruct/arbiter.h>

#define UCT_UGNI_HASH_SIZE   (256)

#define UCT_UGNI_ZERO_LENGTH_POST(len)              \
if (0 == len) {                                     \
    ucs_trace_data("Zero length request: skip it"); \
    return UCS_OK;                                  \
}

typedef struct uct_ugni_ep {
  uct_base_ep_t     super;
  gni_ep_handle_t   ep;
  unsigned          outstanding;
  uint32_t          hash_key;
  ucs_arbiter_group_t arb_group;
  struct uct_ugni_ep *next;
} uct_ugni_ep_t;

static inline int32_t uct_ugni_ep_compare(uct_ugni_ep_t *ep1, uct_ugni_ep_t *ep2)
{
    return ep1->hash_key - ep2->hash_key;
}

static inline unsigned uct_ugni_ep_hash(uct_ugni_ep_t *ep)
{
    return ep->hash_key;
}

SGLIB_DEFINE_LIST_PROTOTYPES(uct_ugni_ep_t, uct_ugni_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_ugni_ep_t, UCT_UGNI_HASH_SIZE, uct_ugni_ep_hash);

UCS_CLASS_DECLARE(uct_ugni_ep_t, uct_iface_t*, const struct sockaddr*);
UCS_CLASS_DECLARE_NEW_FUNC(uct_ugni_ep_t, uct_ep_t, uct_iface_t*, const struct sockaddr*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ugni_ep_t, uct_ep_t);

struct uct_ugni_iface;
uct_ugni_ep_t *uct_ugni_iface_lookup_ep(struct uct_ugni_iface *iface, uintptr_t hash_key);
ucs_status_t ugni_connect_ep(struct uct_ugni_iface *iface, const uct_sockaddr_ugni_t *iface_addr, uct_ugni_ep_t *ep);
ucs_status_t uct_ugni_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n);
void uct_ugni_ep_pending_purge(uct_ep_h tl_ep, uct_pending_callback_t cb);
ucs_arbiter_cb_result_t uct_ugni_ep_process_pending(ucs_arbiter_t *arbiter,
                                                    ucs_arbiter_elem_t *elem,
                                                    void *arg);
#endif
