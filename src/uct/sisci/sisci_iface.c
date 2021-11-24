#include "sisci_iface.h"

#include <uct/base/uct_md.h>


ucs_config_field_t uct_sisci_iface_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_sisci_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"BW", "A LOT OF MBs",
     "Effective memory bandwidth",
     ucs_offsetof(uct_sisci_iface_config_t, bandwidth), UCS_CONFIG_TYPE_BW},

    {NULL}
};


ucs_status_t
uct_sisci_base_query_tl_devices(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                             unsigned *num_tl_devices_p)
{
    return uct_single_device_resource(md, UCT_SISCI_DEVICE_NAME,
                                      UCT_DEVICE_TYPE_SHM,
                                      UCS_SYS_DEVICE_ID_UNKNOWN, tl_devices_p,
                                      num_tl_devices_p);
}

//uct_sisci_iface_get_device_address
//uct_sisci_iface_is_reachable
//uct_sisci_iface_fence
//uct_sisci_iface_get_device_addr_len
//uct_sisci_ep_fence
