/**
 * Copyright (C) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ugni_ep.h"
#include "ugni_iface.h"

#include <ucs/arch/atomic.h>

SGLIB_DEFINE_LIST_FUNCTIONS(uct_ugni_ep_t, uct_ugni_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_ugni_ep_t, UCT_UGNI_HASH_SIZE, uct_ugni_ep_hash);

ucs_status_t uct_ugni_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n){
    uct_ugni_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_iface_t);
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);

    UCS_STATIC_ASSERT(sizeof(ucs_arbiter_elem_t) <= UCT_PENDING_REQ_PRIV_LEN);
    uct_ugni_enter_async(iface);
    ucs_arbiter_elem_init((ucs_arbiter_elem_t *)n->priv);
    ucs_arbiter_group_push_elem(&ep->arb_group, (ucs_arbiter_elem_t*) n->priv);
    ucs_arbiter_group_schedule(&iface->arbiter, &ep->arb_group);
    uct_ugni_leave_async(iface);
    return UCS_OK;
}

ucs_arbiter_cb_result_t uct_ugni_ep_process_pending(ucs_arbiter_t *arbiter,
                                                    ucs_arbiter_elem_t *elem,
                                                    void *arg){
    uct_ugni_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), uct_ugni_ep_t, arb_group);
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    ucs_status_t rc;

    ep->arb_sched = 1;
    rc = req->func(req);
    ep->arb_sched = 0;
    ucs_trace_data("progress pending request %p returned %s", req,
                   ucs_status_string(rc));

    if (UCS_OK == rc) {
        /* sent successfully. remove from the arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    } else if (UCS_INPROGRESS == rc) {
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    } else {
        /* couldn't send. keep this request in the arbiter until the next time
         * this function is called */
        return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
    }
}

ucs_arbiter_cb_result_t uct_ugni_ep_abriter_purge_cb(ucs_arbiter_t *arbiter,
                                                     ucs_arbiter_elem_t *elem,
                                                     void *arg)
{
    uct_ugni_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), uct_ugni_ep_t, arb_group);
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_purge_cb_args_t *cb_args = arg;

    if (NULL != arg) {
        cb_args->cb(req, cb_args->arg);
    } else {
        ucs_warn("ep=%p cancelling user pending request %p", ep, req);
    }

    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

void uct_ugni_ep_pending_purge(uct_ep_h tl_ep,
                               uct_pending_purge_callback_t cb,
                               void *arg){
    uct_ugni_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_iface_t);
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_purge_cb_args_t args = {cb, arg};

    ucs_arbiter_group_purge(&iface->arbiter, &ep->arb_group,
                            uct_ugni_ep_abriter_purge_cb, &args);
}


static uct_ugni_flush_group_t *uct_ugni_new_flush_group(uct_ugni_iface_t *iface)
{
    return ucs_mpool_get(&iface->flush_pool);
}

static void uct_ugni_put_flush_group(uct_ugni_flush_group_t *group)
{
    ucs_mpool_put(group);
}

static void uct_ugni_flush_cb(uct_completion_t *self, ucs_status_t status)
{
    uct_ugni_flush_group_t *group = ucs_container_of(self, uct_ugni_flush_group_t, flush_comp);

    ucs_trace("group=%p, parent=%p, user_comp=%p", group, group->parent, group->user_comp);
    uct_invoke_completion(group->user_comp, UCS_OK);
    uct_ugni_check_flush(group->parent);
    uct_ugni_put_flush_group(group);
}

static uintptr_t uct_ugni_safe_swap_pointers(void *address, uintptr_t new_value)
{
    if (sizeof(void*) == 4) {
        return ucs_atomic_swap32(address, new_value);
    } else {
        return ucs_atomic_swap64(address, new_value);
    }
}

static ucs_status_t uct_ugni_add_flush_comp(uct_ugni_ep_t *ep,  unsigned flags,
                                            uct_completion_t *comp)
{
    uct_ugni_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_ugni_iface_t);
    uct_ugni_flush_group_t *new_group, *present_group;

    if (!uct_ugni_ep_can_send(ep)) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (NULL == comp) {
        return UCS_INPROGRESS;
    }

    new_group = uct_ugni_new_flush_group(iface);
    new_group->flush_comp.count = UCT_UGNI_INIT_FLUSH_REQ;
#ifdef DEBUG
    new_group->flush_comp.func = NULL;
    new_group->parent = NULL;
#endif
    present_group = (uct_ugni_flush_group_t*)uct_ugni_safe_swap_pointers(&ep->flush_group,
                                                                         (uintptr_t)new_group);
    present_group->flush_comp.func = uct_ugni_flush_cb;
    present_group->user_comp = comp;
    present_group->parent = new_group;
    uct_invoke_completion(&present_group->flush_comp, UCS_OK);
    return UCS_INPROGRESS;
}

ucs_status_t uct_ugni_ep_flush(uct_ep_h tl_ep, unsigned flags,
                               uct_completion_t *comp)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    ucs_status_t status = UCS_INPROGRESS;

    ucs_trace_func("tl_ep=%p, flags=%x, comp=%p", tl_ep, flags, comp);

    if (uct_ugni_ep_can_flush(ep)) {
        UCT_TL_EP_STAT_FLUSH(ucs_derived_of(tl_ep, uct_base_ep_t));
        return UCS_OK;
    }
    status = uct_ugni_add_flush_comp(ep, flags, comp);
    if (UCS_INPROGRESS == status) {
        UCT_TL_EP_STAT_FLUSH_WAIT(ucs_derived_of(tl_ep, uct_base_ep_t));
    }
    return status;
}

ucs_status_t ugni_connect_ep(uct_ugni_iface_t *iface,
                             const uct_devaddr_ugni_t *dev_addr,
                             const uct_sockaddr_ugni_t *iface_addr,
                             uct_ugni_ep_t *ep){
    gni_return_t ugni_rc;

    uct_ugni_cdm_lock(&iface->cdm);
    ugni_rc = GNI_EpBind(ep->ep, dev_addr->nic_addr, iface_addr->domain_id);
    uct_ugni_cdm_unlock(&iface->cdm);
    if (GNI_RC_SUCCESS != ugni_rc) {
        uct_ugni_cdm_lock(&iface->cdm);
        (void)GNI_EpDestroy(ep->ep);
        uct_ugni_cdm_unlock(&iface->cdm);
        ucs_error("GNI_EpBind failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_UNREACHABLE;
    }

    ucs_debug("Binding ep %p to address (%d %d)", ep, dev_addr->nic_addr,
              iface_addr->domain_id);

    ep->flush_group->flush_comp.count = UCT_UGNI_INIT_FLUSH;

    return UCS_OK;
}

/* Endpoint definition */
UCS_CLASS_INIT_FUNC(uct_ugni_ep_t, uct_iface_t *tl_iface,
                    const uct_device_addr_t *dev_addr, const uct_iface_addr_t *addr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    const uct_sockaddr_ugni_t *iface_addr = (const uct_sockaddr_ugni_t*)addr;
    const uct_devaddr_ugni_t *ugni_dev_addr = (const uct_devaddr_ugni_t *)dev_addr;
    ucs_status_t rc = UCS_OK;
    gni_return_t ugni_rc;
    uint32_t *big_hash;

    self->arb_sched = 0;
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);
    self->flush_group = uct_ugni_new_flush_group(iface);
#ifdef DEBUG
    self->flush_group->flush_comp.func = NULL;
    self->flush_group->parent = NULL;
#endif
    uct_ugni_cdm_lock(&iface->cdm);
    ugni_rc = GNI_EpCreate(uct_ugni_iface_nic_handle(iface), iface->local_cq, &self->ep);
    uct_ugni_cdm_unlock(&iface->cdm);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    if(NULL != addr){
        rc = ugni_connect_ep(iface, ugni_dev_addr, iface_addr, self);
    }

    ucs_arbiter_group_init(&self->arb_group);
    big_hash = (void *)&self->ep;
    self->hash_key = big_hash[0];
    if (uct_ugni_check_device_type(iface, GNI_DEVICE_ARIES)) {
        self->hash_key &= 0x00FFFFFF;
    }
    ucs_debug("Adding ep hash %x to iface %p", self->hash_key, iface);
    sglib_hashed_uct_ugni_ep_t_add(iface->eps, self);

    return rc;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_ep_t)
{
    uct_ugni_iface_t *iface = ucs_derived_of(self->super.super.iface,
                                             uct_ugni_iface_t);
    gni_return_t ugni_rc;

    ucs_debug("Removinig ep hash %x from iface %p", self->hash_key, iface);

    ucs_arbiter_group_purge(&iface->arbiter, &self->arb_group,
                            uct_ugni_ep_abriter_purge_cb, NULL);
    uct_ugni_cdm_lock(&iface->cdm);
    ugni_rc = GNI_EpDestroy(self->ep);
    uct_ugni_cdm_unlock(&iface->cdm);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_EpDestroy failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
    }
    sglib_hashed_uct_ugni_ep_t_delete(iface->eps, self);
    uct_ugni_ep_pending_purge(&self->super.super, NULL, NULL);
    uct_ugni_put_flush_group(self->flush_group);
}

UCS_CLASS_DEFINE(uct_ugni_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_ep_t, uct_ep_t);

uct_ugni_ep_t *uct_ugni_iface_lookup_ep(uct_ugni_iface_t *iface, uintptr_t hash_key)
{
    uct_ugni_ep_t tmp;
    tmp.hash_key = hash_key;
    return sglib_hashed_uct_ugni_ep_t_find_member(iface->eps, &tmp);
}
