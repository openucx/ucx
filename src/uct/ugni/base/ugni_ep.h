#ifndef UCT_UGNI_EP_H
#define UCT_UGNI_EP_H

#include <gni_pub.h>
#include <uct/api/uct.h>
#include <uct/tl/tl_base.h>
#include <ucs/type/class.h>


typedef struct uct_ugni_ep {
  uct_base_ep_t     super;
  gni_ep_handle_t   ep;
  unsigned          outstanding;
  uintptr_t         hash_key;
  struct uct_ugni_ep *next;
} uct_ugni_ep_t;

#endif
