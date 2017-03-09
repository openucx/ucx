/**
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include <uct/ugni/base/ugni_ep.h>
#include <uct/ugni/base/ugni_iface.h>

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
    ep->arb_size++;
    uct_ugni_leave_async(iface);
    return UCS_OK;
}

ucs_arbiter_cb_result_t uct_ugni_ep_process_pending(ucs_arbiter_t *arbiter,
                                                    ucs_arbiter_elem_t *elem,
                                                    void *arg){
    uct_ugni_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), uct_ugni_ep_t, arb_group);
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    ucs_status_t rc;

    /* If a flush is in progress, and we are done with the pending queue, skip us. */
    if(ep->flush_flag && !ep->arb_flush){
        return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
    }

    ep->arb_sched = 1;
    rc = req->func(req);
    ep->arb_sched = 0;
    ucs_trace_data("progress pending request %p returned %s", req,
                   ucs_status_string(rc));

    if (UCS_OK == rc) {
        ep->arb_size--;
        if(ep->flush_flag) {
            ep->arb_flush--;
        }
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

static ucs_arbiter_cb_result_t uct_ugni_ep_abriter_purge_cb(ucs_arbiter_t *arbiter,
                                                            ucs_arbiter_elem_t *elem,
                                                            void *arg){
    uct_ugni_ep_t *ep = ucs_container_of(ucs_arbiter_elem_group(elem), uct_ugni_ep_t, arb_group);
    uct_pending_req_t *req = ucs_container_of(elem, uct_pending_req_t, priv);
    uct_purge_cb_args_t *cb_args = arg;

    if (NULL != arg) {
        cb_args->cb(req, cb_args->arg);
    } else {
        ucs_warn("ep=%p cancelling user pending request %p", ep, req);
    }

    ep->arb_size--;

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

ucs_status_t ugni_connect_ep(uct_ugni_iface_t *iface,
                             const uct_devaddr_ugni_t *dev_addr,
                             const uct_sockaddr_ugni_t *iface_addr,
                             uct_ugni_ep_t *ep){
    gni_return_t ugni_rc;

    ugni_rc = GNI_EpBind(ep->ep, dev_addr->nic_addr, iface_addr->domain_id);
    if (GNI_RC_SUCCESS != ugni_rc) {
        (void)GNI_EpDestroy(ep->ep);
        ucs_error("GNI_EpBind failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_UNREACHABLE;
    }

    ucs_debug("Binding ep %p to address (%d %d)", ep, dev_addr->nic_addr,
              iface_addr->domain_id);

    ep->outstanding = 0;

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
    self->arb_size = 0;
    self->arb_sched = 0;
    self->flush_flag = 0;
    self->arb_flush = 0;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    ugni_rc = GNI_EpCreate(iface->nic_handle, iface->local_cq, &self->ep);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    if(NULL != addr){
        rc = ugni_connect_ep(iface, ugni_dev_addr, iface_addr, self);
    }

    ucs_arbiter_group_init(&self->arb_group);

    uint32_t *big_hash;
    big_hash = (void *)&self->ep;
    self->hash_key = big_hash[0];
    if (GNI_DEVICE_ARIES == iface->dev->type) {
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

    ugni_rc = GNI_EpDestroy(self->ep);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_EpDestroy failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
    }
    sglib_hashed_uct_ugni_ep_t_delete(iface->eps, self);
    uct_ugni_ep_pending_purge(&self->super.super, NULL, NULL);
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
