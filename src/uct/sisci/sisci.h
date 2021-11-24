
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <sisci_error.h> //TODO
#include <sisci_api.h>


#define UCT_SISCI_NAME "sisci"
#define UCT_SISCI_CONFIG_PREFIX "SISCI_"


typedef uint64_t uct_sisci_iface_addr_t;


void sisci_testing();

// iface file contents

//extern ucs_config_field_t uct_sisci_iface_config_table[];

typedef struct uct_sisci_iface_config {
    uct_iface_config_t    super;
    size_t                seg_size;      /* Maximal send size */
} uct_sisci_iface_config_t;


typedef struct uct_sisci_iface {
    uct_base_iface_t      super;
    uct_sisci_iface_addr_t id;           /* Unique identifier for the instance */
    size_t                send_size;    /* Maximum size for payload */
    ucs_mpool_t           msg_mp;       /* Messages memory pool */
} uct_sisci_iface_t;

ucs_status_t
uct_sisci_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p);

ucs_status_t uct_sisci_iface_get_device_address(uct_iface_t *tl_iface,
                                             uct_device_addr_t *addr);

static int uct_sisci_iface_is_reachable(const uct_iface_h tl_iface, const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr);

ucs_status_t uct_sisci_iface_fence(uct_iface_t *tl_iface, unsigned flags);

size_t uct_sisci_iface_get_device_addr_len();

ucs_status_t uct_sisci_ep_fence(uct_ep_t *tl_ep, unsigned flags);

/*
UCS_CLASS_DECLARE(uct_sisci_iface_t, uct_iface_ops_t*, uct_iface_internal_ops_t*,
                  uct_md_h, uct_worker_h, const uct_iface_params_t*,
                  const uct_iface_config_t*);

*/

/**
 * @brief self device MD descriptor
 */
typedef struct uct_sisci_md {
    uct_md_t super;
    size_t   num_devices; /* Number of devices to create */

    unsigned int segment_id;
    size_t segment_size;
    unsigned int localAdapterNo;

    sci_desc_t sisci_virtual_device;
    sci_local_segment_t local_segment;
    
} uct_sisci_md_t;


/**
 * @brief self device MD configuration
 */
typedef struct uct_sisci_md_config {
    uct_md_config_t super;
    size_t          num_devices; /* Number of devices to create */
    size_t          segment_size;
    size_t          segment_id;
} uct_sisci_md_config_t;




typedef struct uct_sisci_ep {
    uct_base_ep_t         super;
} uct_sisci_ep_t;




ucs_status_t uct_sisci_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                 unsigned length, uint64_t remote_addr,
                                 uct_rkey_t rkey);
ssize_t uct_sisci_ep_put_bcopy(uct_ep_h ep, uct_pack_callback_t pack_cb,
                            void *arg, uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_sisci_ep_get_bcopy(uct_ep_h ep, uct_unpack_callback_t unpack_cb,
                                 void *arg, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp);

ucs_status_t uct_sisci_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare,
                                      uint64_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint64_t *result,
                                      uct_completion_t *comp);
ucs_status_t uct_sisci_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare,
                                      uint32_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint32_t *result,
                                      uct_completion_t *comp);
ucs_status_t uct_sisci_ep_atomic64_post(uct_ep_h ep, unsigned opcode, uint64_t value,
                                     uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_sisci_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                      uint64_t value, uint64_t *result,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp);
ucs_status_t uct_sisci_ep_atomic32_post(uct_ep_h ep, unsigned opcode, uint32_t value,
                                     uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_sisci_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                      uint32_t value, uint32_t *result,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uct_completion_t *comp);
