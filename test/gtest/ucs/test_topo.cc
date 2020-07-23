/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/sys/topo.h>
}

class test_topo : public ucs::test {
};

UCS_TEST_F(test_topo, find_device_by_bus_id) {
    ucs_status_t status;
    ucs_sys_device_t dev1;
    ucs_sys_device_t dev2;
    ucs_sys_bus_id_t dummy_bus_id;

    dummy_bus_id.domain   = 0xffff;
    dummy_bus_id.bus      = 0xff;
    dummy_bus_id.slot     = 0xff;
    dummy_bus_id.function = 1; 

    status = ucs_topo_find_device_by_bus_id(&dummy_bus_id, &dev1);
    ASSERT_UCS_OK(status);

    dummy_bus_id.function = 2; 

    status = ucs_topo_find_device_by_bus_id(&dummy_bus_id, &dev2);
    ASSERT_UCS_OK(status);
    ASSERT_EQ(dev2, ((unsigned)dev1 + 1));
}

UCS_TEST_F(test_topo, get_distance) {
    ucs_status_t status;
    ucs_sys_dev_distance_t distance;

    status = ucs_topo_get_distance(UCS_SYS_DEVICE_ID_UNKNOWN,
                                   UCS_SYS_DEVICE_ID_UNKNOWN, &distance);
    ASSERT_EQ(UCS_ERR_IO_ERROR, status);
}

UCS_TEST_F(test_topo, print_info) {
    ucs_topo_print_info(NULL);
}
