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

    status = ucs_topo_find_device_by_bus_id(NULL, NULL);
    ASSERT_UCS_OK(status);
}

UCS_TEST_F(test_topo, get_distance) {
    ucs_status_t status;

    status = ucs_topo_get_distance(NULL, NULL, NULL);
    ASSERT_UCS_OK(status);
}

UCS_TEST_F(test_topo, print_info) {
    ucs_topo_print_info(NULL);
}
