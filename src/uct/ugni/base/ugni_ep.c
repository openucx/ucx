/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include <uct/ugni/base/ugni_ep.h>
#include <uct/ugni/base/ugni_iface.h>

SGLIB_DEFINE_LIST_PROTOTYPES(uct_ugni_ep_t, uct_ugni_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_ugni_ep_t, UCT_UGNI_HASH_SIZE, uct_ugni_ep_hash);
SGLIB_DEFINE_LIST_FUNCTIONS(uct_ugni_ep_t, uct_ugni_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_ugni_ep_t, UCT_UGNI_HASH_SIZE, uct_ugni_ep_hash);

/* Endpoint definition */
UCS_CLASS_INIT_FUNC(uct_ugni_ep_t, uct_iface_t *tl_iface,
                    const struct sockaddr *addr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    const uct_sockaddr_ugni_t *iface_addr = (const uct_sockaddr_ugni_t*)addr;
    gni_return_t ugni_rc;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)

    ugni_rc = GNI_EpCreate(iface->nic_handle, iface->local_cq, &self->ep);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    ugni_rc = GNI_EpBind(self->ep, iface_addr->nic_addr, iface_addr->domain_id);
    if (GNI_RC_SUCCESS != ugni_rc) {
        (void)GNI_EpDestroy(self->ep);
        ucs_error("GNI_EpBind failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_UNREACHABLE;
    }

    ucs_debug("Binding ep %p to address (%d %d)", self, iface_addr->nic_addr,
              iface_addr->domain_id);

    self->outstanding = 0;
    self->hash_key = (uintptr_t)&self->ep;
    sglib_hashed_uct_ugni_ep_t_add(iface->eps, self);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_ep_t)
{
    uct_ugni_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                             uct_ugni_iface_t);
    gni_return_t ugni_rc;

    ugni_rc = GNI_EpDestroy(self->ep);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_EpDestroy failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
    }
    sglib_hashed_uct_ugni_ep_t_delete(iface->eps, self);
}
UCS_CLASS_DEFINE(uct_ugni_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_ep_t, uct_ep_t, uct_iface_t*, const struct sockaddr*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_ep_t, uct_ep_t);

uct_ugni_ep_t *uct_ugni_iface_lookup_ep(uct_ugni_iface_t *iface, uintptr_t hash_key)
{
    uct_ugni_ep_t tmp;
    tmp.hash_key = hash_key;
    return sglib_hashed_uct_ugni_ep_t_find_member(iface->eps, &tmp);
}
