#ifndef UCT_SISCI_H
#define UCT_SISCI_H



#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <ucs/sys/iovec.h>
#include <uct/base/uct_iov.inl>


#include <sisci_error.h> //TODO
#include <sisci_api.h>

#define UCT_sci_NAME "sci"
#define UCT_sci_CONFIG_PREFIX "sci_"

#define ADAPTER_NO 0
#define SCI_NO_FLAGS 0
#define SISCI_STATUS_WRITING_DONE 1


// https://stackoverflow.com/questions/1941307/debug-print-macro-in-c by Tom Kuschel
#define DEBUG 0

#if defined(DEBUG) && DEBUG > 0
 #define DEBUG_PRINT(fmt, args...) fprintf(stdout, "DEBUG: %s:%d:%s(): " fmt, \
    __FILE__, __LINE__, __func__, ##args)
#else
 #define DEBUG_PRINT(fmt, args...) /* Don't do anything in release builds */
#endif

typedef struct uct_sci_iface_addr {
    unsigned int segment_id; /* Listening port of iface */
} UCS_S_PACKED uct_sci_iface_addr_t;

typedef struct uct_sci_device_addr {
    unsigned int node_id;
} UCS_S_PACKED uct_sci_device_addr_t;


typedef struct sci_map_holder {
    volatile unsigned int* mapped;
} sci_map_holder_t;

void sci_testing();

// iface file contents

//extern ucs_config_field_t uct_sci_iface_config_table[];

typedef struct uct_sci_iface_config {
    uct_iface_config_t    super;
    size_t                send_size;      /* Maximal send size */
} uct_sci_iface_config_t;


typedef struct sisci_packet {
    uint8_t     status;
    uint8_t     am_id;
    unsigned    length;
    //void        data;
} UCS_S_PACKED sisci_packet_t;

typedef struct uct_sci_ep_zcopy_tx {
    sisci_packet_t                super;     /* UCT TCP AM header */
    uct_completion_t              *comp;     /* Local UCT completion object */
    size_t                        iov_index; /* Current IOV index */
    size_t                        iov_cnt;   /* Number of IOVs that should be sent */
    struct iovec                  iov[0];    /* IOVs that should be sent */
} uct_sci_ep_zcopy_tx_t;

typedef struct uct_sci_iface {
    uct_base_iface_t      super;
    unsigned int          segment_id;           /* Unique identifier for the instance */
    unsigned int          device_addr; //nodeID
    size_t                send_size;    /* Maximum size for payload */
    ucs_mpool_t           msg_mp;       /* Messages memory pool */
    void*                 recv_buffer;
    sci_local_segment_t   local_segment;
    sci_map_t             local_map;


} uct_sci_iface_t;

ucs_status_t uct_sci_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p);

int uct_sci_iface_is_reachable(const uct_iface_h tl_iface, const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr);

ucs_status_t uct_sci_iface_fence(uct_iface_t *tl_iface, unsigned flags);

size_t uct_sci_iface_get_device_addr_len();

ucs_status_t uct_sci_ep_fence(uct_ep_t *tl_ep, unsigned flags);


/**
 * @brief self device MD descriptor
 */
typedef struct uct_sci_md {
    uct_md_t super;
    size_t   num_devices; /* Number of devices to create */

    unsigned int segment_id;
    size_t segment_size;
    unsigned int localAdapterNo;

    sci_desc_t sci_virtual_device;
    sci_local_segment_t local_segment;
    
} uct_sci_md_t;


/**
 * @brief self device MD configuration
 */
typedef struct uct_sci_md_config {
    uct_md_config_t super;
    size_t          num_devices; /* Number of devices to create */
    size_t          segment_size;
    size_t          segment_id;
    
} uct_sci_md_config_t;



#endif