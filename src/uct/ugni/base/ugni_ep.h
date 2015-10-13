#ifndef UCT_UGNI_EP_H
#define UCT_UGNI_EP_H

#include <gni_pub.h>
#include <uct/api/uct.h>
#include <uct/tl/tl_base.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/sglib_wrapper.h>

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
  uint64_t          hash_key;
  struct uct_ugni_ep *next;
} uct_ugni_ep_t;

static inline ptrdiff_t uct_ugni_ep_compare(uct_ugni_ep_t *ep1, uct_ugni_ep_t *ep2)
{
    return ep1->hash_key - ep2->hash_key;
}

static inline unsigned uct_ugni_ep_hash(uct_ugni_ep_t *ep)
{
    return ep->hash_key;
}

UCS_CLASS_DECLARE(uct_ugni_ep_t, uct_iface_t*, const struct sockaddr*);
UCS_CLASS_DECLARE_NEW_FUNC(uct_ugni_ep_t, uct_ep_t, uct_iface_t*, const struct sockaddr*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ugni_ep_t, uct_ep_t);

struct uct_ugni_iface;
uct_ugni_ep_t *uct_ugni_iface_lookup_ep(struct uct_ugni_iface *iface, uintptr_t hash_key);

#endif
