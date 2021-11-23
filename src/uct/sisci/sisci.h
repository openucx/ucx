
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <sisci_error.h>
#include <sisci_api.h>

#define UCT_SISCI_NAME "sisci"
#define UCT_SISCI_CONFIG_PREFIX "SISCI_"


typedef uint64_t uct_sisci_iface_addr_t;


void sisci_testing();

typedef struct uct_sisci_iface_config {
    uct_iface_config_t    super;
    size_t                seg_size;      /* Maximal send size */
} uct_sisci_iface_config_t;


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


typedef struct uct_sisci_iface {
    uct_base_iface_t      super;
    uct_sisci_iface_addr_t id;           /* Unique identifier for the instance */
    size_t                send_size;    /* Maximum size for payload */
    ucs_mpool_t           msg_mp;       /* Messages memory pool */
} uct_sisci_iface_t;


typedef struct uct_sisci_ep {
    uct_base_ep_t         super;
} uct_sisci_ep_t;

