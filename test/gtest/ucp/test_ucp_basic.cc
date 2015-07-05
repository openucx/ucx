/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_test.h"

class test_ucp_basic : public ucp_test {
};

UCS_TEST_F(test_ucp_basic, entity) {
    create_entity();
}
