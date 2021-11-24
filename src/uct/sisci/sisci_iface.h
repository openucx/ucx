
//#include <uct/api/uct.h>
//#include <uct/base/uct_iface.h>
//#include <ucs/sys/math.h>
//#include <ucs/sys/iovec.h>

#ifndef SISCI_IFACE_H_
#define SISCI_IFACE_H_

#define UCT_SISCI_MAX_IOV                  16  //TODO what is iov? is it 16?
#define UCT_SISCI_DEVICE_NAME              "sisci rdma"

#endif



extern ucs_config_field_t uct_sisci_iface_config_table[];

typedef struct uct_sisci_iface_common_config {
    uct_iface_config_t     super;
    double                 bandwidth; /* Memory bandwidth in bytes per second */
} uct_sisci_iface_config_t;

typedef struct uct_sisci_iface {
    uct_base_iface_t       super;
    struct {
        double             bandwidth; /* Memory bandwidth in bytes per second */
    } config;
} uct_sisci_iface_t;

ucs_status_t
uct_sisci_base_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p);

ucs_status_t uct_sisci_iface_get_device_address(uct_iface_t *tl_iface,
                                             uct_device_addr_t *addr);

int uct_sisci_iface_is_reachable(const uct_iface_h tl_iface, const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr);

ucs_status_t uct_sisci_iface_fence(uct_iface_t *tl_iface, unsigned flags);

size_t uct_sisci_iface_get_device_addr_len();

ucs_status_t uct_sisci_ep_fence(uct_ep_t *tl_ep, unsigned flags);

UCS_CLASS_DECLARE(uct_sisci_iface_t, uct_iface_ops_t*, uct_iface_internal_ops_t*,
                  uct_md_h, uct_worker_h, const uct_iface_params_t*,
                  const uct_iface_config_t*);

