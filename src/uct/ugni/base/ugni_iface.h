#ifndef UCT_UGNI_IFACE
#define UCT_UGNI_IFACE

#include <gni_pub.h>
#include <uct/base/uct_iface.h>
#include <ucs/datastruct/arbiter.h>
#include <ucs/async/async.h>
#include "uct/api/uct.h"
#include "ugni_ep.h"
#include "ugni_device.h"
#include "ugni_md.h"

#define UCT_UGNI_RDMA_TL_NAME   "ugni_rdma"

typedef struct uct_ugni_iface {
    uct_base_iface_t        super;
    uct_ugni_device_t       *dev;
    gni_cdm_handle_t        cdm_handle;                  /**< Ugni communication domain */
    gni_nic_handle_t        nic_handle;                  /**< Ugni NIC handle */
    gni_cq_handle_t         local_cq;                    /**< Completion queue */
    uint16_t                domain_id;                   /**< Id for UGNI domain creation */
    uct_ugni_ep_t           *eps[UCT_UGNI_HASH_SIZE];    /**< Array of QPs */
    unsigned                outstanding;                 /**< Counter for outstanding packets
                                                              on the interface */
    bool                    activated;                   /**< nic status */
    ucs_arbiter_t           arbiter;
} uct_ugni_iface_t;

UCS_CLASS_DECLARE(uct_ugni_iface_t, uct_md_h, uct_worker_h,
                  const uct_iface_params_t*, uct_iface_ops_t*,
                  const uct_iface_config_t* UCS_STATS_ARG(ucs_stats_node_t*))

ucs_status_t uct_ugni_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                  uct_completion_t *comp);
ucs_status_t uct_ugni_ep_flush(uct_ep_h tl_ep, unsigned flags,
                               uct_completion_t *comp);
ucs_status_t uct_ugni_iface_get_address(uct_iface_h tl_iface, uct_iface_addr_t *addr);
int uct_ugni_iface_is_reachable(uct_iface_h tl_iface, const uct_device_addr_t *dev_addr, const uct_iface_addr_t *iface_addr);
void uct_ugni_progress(void *arg);

typedef struct uct_ugni_base_desc {
    gni_post_descriptor_t desc;
    uct_completion_t *comp_cb;
    uct_unpack_callback_t unpack_cb;
    uct_ugni_ep_t  *ep;
    int not_ready_to_free;
} uct_ugni_base_desc_t;

ucs_status_t ugni_activate_iface(uct_ugni_iface_t *iface);
ucs_status_t ugni_deactivate_iface(uct_ugni_iface_t *iface);

ucs_status_t uct_ugni_init_nic(int device_index,
                               uint16_t *domain_id,
                               gni_cdm_handle_t *cdm_handle,
                               gni_nic_handle_t *nic_handle,
                               uint32_t *address);

static inline uct_ugni_device_t * uct_ugni_iface_device(uct_ugni_iface_t *iface)
{
    return iface->dev;
}

void uct_ugni_base_desc_init(ucs_mpool_t *mp, void *obj, void *chunk);
void uct_ugni_base_desc_key_init(uct_iface_h iface, void *obj, uct_mem_h memh);
ucs_status_t uct_ugni_query_tl_resources(uct_md_h md, const char *tl_name,
                                         uct_tl_resource_desc_t **resource_p,
                                         unsigned *num_resources_p);

typedef struct uct_ugni_iface_config {
    uct_iface_config_t       super;
    uct_iface_mpool_config_t mpool;
} uct_ugni_iface_config_t;

#define uct_ugni_enter_async(x) do {\
    ucs_trace_async("Taking lock on worker %p", (x)->super.worker); \
    UCS_ASYNC_BLOCK((x)->super.worker->async);                      \
    }while(0)

#define uct_ugni_leave_async(x) do {\
    ucs_trace_async("Releasing lock on worker %p",    (x)->super.worker); \
    UCS_ASYNC_UNBLOCK((x)->super.worker->async);                        \
    }while(0)

#endif
