/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <ucs/time/time.h>
}
#include <ucs/gtest/test.h>
#include <uct/ib/base/ib_iface.h>
#include "uct_test.h"

class test_uct_ib : public uct_test {
public:
    void initialize() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_e2 = uct_test::create_entity(0);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        m_entities.push_back(m_e1);
        m_entities.push_back(m_e2);
    }

    typedef struct {
        unsigned length;
        /* data follows */
    } recv_desc_t;

    static ucs_status_t ib_am_handler(void *arg, void *data, size_t length, void *desc) {
        recv_desc_t *my_desc  = (recv_desc_t *) arg;
        uint64_t *test_ib_hdr = (uint64_t *) data;
        uint64_t *actual_data = (uint64_t *) test_ib_hdr + 1;
        unsigned data_length  = length - sizeof(test_ib_hdr);

        my_desc->length = data_length;
        if (*test_ib_hdr == 0xbeef) {
            memcpy(my_desc + 1, actual_data , data_length);
        }

        return UCS_OK;
    }

    ucs_status_t pkey_test(const char *dev_name, unsigned port_num) {
        struct ibv_device **device_list;
        struct ibv_context *ibctx = NULL;
        struct ibv_port_attr port_attr;
        uct_ib_iface_config_t *ib_config = ucs_derived_of(m_iface_config, uct_ib_iface_config_t);
        int num_devices, i, found = 0;
        uint16_t table_idx, pkey, pkey_partition;

        /* get device list */
        device_list = ibv_get_device_list(&num_devices);
        if (device_list == NULL) {
            UCS_TEST_ABORT("Failed to get the device list.");
        }

        /* search for the given device in the device list */
        for (i = 0; i < num_devices; ++i) {
            if (strcmp(device_list[i]->name, dev_name)) {
                continue;
            }
            /* found this dev_name on the host - open it */
            ibctx = ibv_open_device(device_list[i]);
            if (ibctx == NULL) {
                UCS_TEST_ABORT("Failed to open the device.");
            }
            found = 1;
            break;
        }
        if (found != 1) {
            UCS_TEST_ABORT("The requested device: " << dev_name << ", wasn't found in the device list.");
        }

        found = 0;
        /* check if the configured pkey exists in the port's pkey table */
        if (ibv_query_port(ibctx, port_num, &port_attr) != 0) {
            UCS_TEST_ABORT("Failed to query port " << port_num << " on device: " << dev_name);
        }
        for (table_idx = 0; table_idx < port_attr.pkey_tbl_len; table_idx++) {
            if(ibv_query_pkey(ibctx, port_num, table_idx, &pkey)) {
                UCS_TEST_ABORT("Failed to query pkey on port " << port_num << " on device: " << dev_name);
            }
            pkey_partition = ntohs(pkey) & UCT_IB_PKEY_PARTITION_MASK;
            if (pkey_partition == (ib_config->pkey_value & UCT_IB_PKEY_PARTITION_MASK)) {
                found = 1;
                break;
            }
        }
        ibv_close_device(ibctx);
        ibv_free_device_list(device_list);

        if (found) {
            return UCS_OK;
        } else {
            return UCS_ERR_NO_ELEM;
        }
    }

    ucs_status_t test_pkey_avail() {
        char *p, *dev_name;
        unsigned port_num;
        ucs_status_t ret;

        dev_name = strdup(GetParam()->dev_name.c_str()); /* device name and port number */
        /* split dev_name */
        p = strchr(dev_name, ':');
        EXPECT_TRUE(p != NULL);
        *p = 0;

        /* dev_name holds the device name */
        /* port number */
        if (sscanf(p + 1, "%d", &port_num) != 1) {
            UCS_TEST_ABORT("Failed to get the port number on device: " << dev_name);
        }
        ret = pkey_test(dev_name, port_num);

        free(dev_name);
        return ret;
    }

    void short_progress_loop() {
        ucs_time_t end_time = ucs_get_time() +
                        ucs_time_from_msec(100.0) * ucs::test_time_multiplier();
        while (ucs_get_time() < end_time) {
            progress();
        }
    }

    void cleanup() {
        uct_test::cleanup();
    }

protected:
    entity *m_e1, *m_e2;
};

UCS_TEST_P(test_uct_ib, non_default_pkey, "IB_PKEY=0x2")
{
    uint64_t send_data   = 0xdeadbeef;
    uint64_t test_ib_hdr = 0xbeef;
    recv_desc_t *recv_buffer;
    ucs_status_t ret;

    /* check if the configured pkey exists in the port's pkey table.
     * skip this test if it doesn't. */
    ret = test_pkey_avail();

    if (ret == UCS_OK) {
        initialize();
    } else {
        UCS_TEST_SKIP_R("pkey not found");
    }

    check_caps(UCT_IFACE_FLAG_AM_SHORT);

    recv_buffer = (recv_desc_t *) malloc(sizeof(*recv_buffer) + sizeof(uint64_t));
    recv_buffer->length = 0; /* Initialize length to 0 */

    /* set a callback for the uct to invoke for receiving the data */
    uct_iface_set_am_handler(m_e2->iface(), 0, ib_am_handler , recv_buffer);

    /* send the data */
    uct_ep_am_short(m_e1->ep(0), 0, test_ib_hdr, &send_data, sizeof(send_data));

    short_progress_loop();

    ASSERT_EQ(sizeof(send_data), recv_buffer->length);
    EXPECT_EQ(send_data, *(uint64_t*)(recv_buffer+1));

    free(recv_buffer);
}

UCT_INSTANTIATE_IB_TEST_CASE(test_uct_ib);
