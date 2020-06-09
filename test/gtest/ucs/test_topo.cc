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
    ucs_sys_device_t unknown_dev1;
    ucs_sys_device_t unknown_dev2;
    ucs_sys_bus_id_t dummy_bus_id1;

    status = ucs_topo_find_device_by_bus_id(&ucs_sys_bus_id_unknown, &unknown_dev1);
    ASSERT_UCS_OK(status);
    ASSERT_EQ(unknown_dev1, UINT_MAX);

    dummy_bus_id1.domain   = ucs_sys_bus_id_unknown.domain;
    dummy_bus_id1.bus      = ucs_sys_bus_id_unknown.bus;
    dummy_bus_id1.slot     = ucs_sys_bus_id_unknown.slot;
    dummy_bus_id1.function = 1; 

    status = ucs_topo_find_device_by_bus_id(&dummy_bus_id1, &unknown_dev2);
    ASSERT_UCS_OK(status);
    ASSERT_EQ(unknown_dev2, 0);

    dummy_bus_id1.function = 2; 

    status = ucs_topo_find_device_by_bus_id(&dummy_bus_id1, &unknown_dev2);
    ASSERT_UCS_OK(status);
    ASSERT_EQ(unknown_dev2, 1);
}

UCS_TEST_F(test_topo, get_distance) {
    ucs_status_t status;
    ucs_sys_device_t unknown_dev1;
    ucs_sys_device_t unknown_dev2;
    ucs_sys_dev_distance_t distance;

    status = ucs_topo_find_device_by_bus_id(&ucs_sys_bus_id_unknown, &unknown_dev1);
    ASSERT_UCS_OK(status);

    status = ucs_topo_find_device_by_bus_id(&ucs_sys_bus_id_unknown, &unknown_dev2);
    ASSERT_UCS_OK(status);

    status = ucs_topo_get_distance(unknown_dev1, unknown_dev2, &distance);
    ASSERT_UCS_OK(status);
}

UCS_TEST_F(test_topo, print_info) {
    ucs_topo_print_info(NULL);
}
