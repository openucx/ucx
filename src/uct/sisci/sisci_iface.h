
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/sys/math.h>
#include <ucs/sys/iovec.h>

#ifndef SISCI_IFACE_H_
#define SISCI_IFACE_H_

#define UCT_SISCI_MAX_IOV                  16  //TODO what is iov? is it 16?
#define UCT_SISCI_DEVICE_NAME              "sisci rdma"

#endif



extern ucs_config_field_t uct_sisci_iface_config_table[];

/*typedef struct uct_sisci_iface_common_config {
    uct_iface_config_t     super;
    double                 bandwidth; // Memory bandwidth in bytes per second
} uct_sisci_iface_config_t; */

/*typedef struct uct_sisci_iface {
    uct_base_iface_t       super;
    struct {
        double             bandwidth; // Memory bandwidth in bytes per second
    } config;
} uct_sisci_iface_t;*/



