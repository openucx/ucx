#include "ofi_iface.h"
#include "ofi_ep.h"

static uint64_t uct_ofi_gen_id()
{
    return ucs_generate_uuid(0);
}

UCS_CLASS_INIT_FUNC(uct_ofi_ep_t, const uct_ep_params_t *params)
{
    uct_ofi_iface_t *iface;
    uct_ofi_name_t *address;
    int ret;

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);

    iface = ucs_derived_of(params->iface, uct_ofi_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);
    address = (uct_ofi_name_t *)params->iface_addr;
    self->id = uct_ofi_gen_id();
    self->av_index = uct_ofi_get_next_av(iface->av);

    ret = fi_av_insert(iface->av->av, address, 1, &iface->av->table[self->av_index], 0, NULL);
    if (ret) {
        ucs_debug("Created OFI EP, id=%lu", self->id);
        return UCS_OK;
    } else {
        ucs_error("Failed to create OFI EP");
        return UCS_ERR_NO_DEVICE;
    }
}

static UCS_CLASS_CLEANUP_FUNC(uct_ofi_ep_t)
{
    uct_ofi_iface_t *iface =  ucs_derived_of(self->super.super.iface, uct_ofi_iface_t);
    int status;

    status = fi_av_remove(iface->av->av, &iface->av->table[self->av_index], 1, 0);
    if (status) {
        ucs_error("OFI: fi_av_remove() failed: %s", fi_strerror(status));
    }
}

UCS_CLASS_DEFINE(uct_ofi_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ofi_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ofi_ep_t, uct_ep_t);

ucs_status_t uct_ofi_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    ucs_debug("Getting OFI 'EP' address");
    /* The underlying fi_getname() function exposes the same name for both */
    return uct_ofi_iface_get_address(tl_ep->iface, (uct_iface_addr_t*)addr);
}

ucs_status_t uct_ofi_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                       const void *payload, unsigned length)
{
    return UCS_ERR_NO_MEMORY;
}

ssize_t uct_ofi_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                  uct_pack_callback_t pack_cb,
                                  void *arg, unsigned flags)
{
    return 0;
}

ucs_status_t uct_ofi_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n,
                                     unsigned flags)
{
    return UCS_ERR_NO_MEMORY;
}

void uct_ofi_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                               void *arg)
{
}

ucs_arbiter_cb_result_t uct_ofi_ep_process_pending(ucs_arbiter_t *arbiter,
                                                    ucs_arbiter_group_t *group,
                                                    ucs_arbiter_elem_t *elem,
                                                    void *arg)
{
    return UCS_ARBITER_CB_RESULT_STOP;
}

ucs_arbiter_cb_result_t uct_ofi_ep_arbiter_purge_cb(ucs_arbiter_t *arbiter,
                                                     ucs_arbiter_group_t *group,
                                                     ucs_arbiter_elem_t *elem,
                                                     void *arg)
{
    return UCS_ARBITER_CB_RESULT_STOP;
}

ucs_status_t uct_ofi_ep_flush(uct_ep_h tl_ep, unsigned flags,
                               uct_completion_t *comp)
{
    return UCS_ERR_NO_MEMORY;
}
