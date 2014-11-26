/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_iface.h"
#include "rc_ep.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>


void uct_rc_iface_query(uct_rc_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    iface_attr->max_short      = 0;
    iface_attr->max_bcopy      = 0;
    iface_attr->max_zcopy      = 0;
    iface_attr->iface_addr_len = sizeof(uct_ib_iface_addr_t);
    iface_attr->ep_addr_len    = sizeof(uct_rc_ep_addr_t);
    iface_attr->flags          = 0;
}

ucs_status_t uct_rc_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *iface_addr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);

    *(uct_ib_iface_addr_t*)iface_addr = iface->super.addr;
    return UCS_OK;
}

static inline int uct_rc_ep_compare(uct_rc_ep_t *ep1, uct_rc_ep_t *ep2)
{
    return (int32_t)ep1->qp_num - (int32_t)ep2->qp_num;
}

static inline unsigned uct_rc_ep_hash(uct_rc_ep_t *ep)
{
    return ep->qp_num;
}

SGLIB_DEFINE_LIST_PROTOTYPES(uct_rc_ep_t, mxm_rc_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_rc_ep_t, UCT_RC_QP_HASH_SIZE, mxm_rc_ep_hash);
SGLIB_DEFINE_LIST_FUNCTIONS(uct_rc_ep_t, uct_rc_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_rc_ep_t, UCT_RC_QP_HASH_SIZE, uct_rc_ep_hash);

uct_rc_ep_t *uct_rc_iface_lookup_ep(uct_rc_iface_t *iface, unsigned qp_num)
{
    uct_rc_ep_t tmp;
    tmp.qp_num = qp_num;
    return sglib_hashed_uct_rc_ep_t_find_member(iface->eps, &tmp);
}

void uct_rc_iface_add_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep)
{
    sglib_hashed_uct_rc_ep_t_add(iface->eps, ep);
}

void uct_rc_iface_remove_ep(uct_rc_iface_t *iface, uct_rc_ep_t *ep)
{
    sglib_hashed_uct_rc_ep_t_delete(iface->eps, ep);
}


static UCS_CLASS_INIT_FUNC(uct_rc_iface_t, uct_context_h context, const char *dev_name)
{
    UCS_CLASS_CALL_SUPER_INIT(context, dev_name);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_iface_t)
{
}

UCS_CLASS_DEFINE(uct_rc_iface_t, uct_ib_iface_t);

ucs_config_field_t uct_rc_iface_config_table[] = {
  {"", "", NULL,
   ucs_offsetof(uct_rc_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_ib_iface_config_table)},

  {NULL}
};
