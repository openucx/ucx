/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

extern "C" {
#include <poll.h>
#include <uct/api/uct.h>
#include <ucs/time/time.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_iface.h>
}
#include <common/test.h>
#include "uct_test.h"

class test_uct_ib : public uct_test {
public:
    void initialize() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);
    }

    typedef struct {
        unsigned length;
        /* data follows */
    } recv_desc_t;

    typedef struct {
        unsigned have_pkey; /* if 1 - means that the configured pkey was found */
        unsigned have_lmc;  /* if 1 - means that the lmc is higher than zero */
#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
        unsigned have_valid_gid_idx;
#endif
    } ib_port_desc_t;

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

    void pkey_find(const char *dev_name, unsigned port_num, struct ibv_port_attr port_attr,
                  struct ibv_context *ibctx, ib_port_desc_t *port_desc) {
         uint16_t table_idx, pkey, pkey_partition;
         uct_ib_iface_config_t *ib_config = ucs_derived_of(m_iface_config, uct_ib_iface_config_t);

         /* check if the configured pkey exists in the port's pkey table */
           for (table_idx = 0; table_idx < port_attr.pkey_tbl_len; table_idx++) {
               if(ibv_query_pkey(ibctx, port_num, table_idx, &pkey)) {
                   UCS_TEST_ABORT("Failed to query pkey on port " << port_num << " on device: " << dev_name);
               }
               pkey_partition = ntohs(pkey) & UCT_IB_PKEY_PARTITION_MASK;
               if (pkey_partition == (ib_config->pkey_value & UCT_IB_PKEY_PARTITION_MASK)) {
                   port_desc->have_pkey = 1;
                   break;
               }
           }
    }

#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
    void test_eth_port(struct ibv_port_attr port_attr, struct ibv_context *ibctx,
                       unsigned port_num, uct_ib_iface_config_t *ib_config,
                       ib_port_desc_t *port_desc) {

        union ibv_gid gid;

        /* no pkeys for Ethernet */
        port_desc->have_pkey = 0;

        /* check the gid index */
        if (ibv_query_gid(ibctx, port_num, ib_config->gid_index, &gid) != 0) {
            UCS_TEST_ABORT("Failed to query gid (index=" << ib_config->gid_index << ")");
        }
        if (uct_ib_device_is_gid_raw_empty(gid.raw)) {
            port_desc->have_valid_gid_idx = 0;
        } else {
            port_desc->have_valid_gid_idx = 1;
        }

    }
#endif

    void lmc_find(struct ibv_port_attr port_attr, ib_port_desc_t *port_desc) {

         if (port_attr.lmc > 0) {
             port_desc->have_lmc = 1;
         }
    }

    void port_attr_test(const char *dev_name, unsigned port_num, ib_port_desc_t *port_desc) {
        struct ibv_device **device_list;
        struct ibv_context *ibctx = NULL;
        struct ibv_port_attr port_attr;
        uct_ib_iface_config_t *ib_config = ucs_derived_of(m_iface_config, uct_ib_iface_config_t);
        int num_devices, i, found = 0;

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

        if (ibv_query_port(ibctx, port_num, &port_attr) != 0) {
            UCS_TEST_ABORT("Failed to query port " << port_num << " on device: " << dev_name);
        }

        /* check the lmc value in the port */
        lmc_find(port_attr, port_desc);

#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
        if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
                test_eth_port(port_attr, ibctx, port_num, ib_config, port_desc);
                goto out;
            }
#endif

        /* find the configured pkey */
        pkey_find(dev_name, port_num, port_attr, ibctx, port_desc);

#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
out:
#endif
        ibv_close_device(ibctx);
        ibv_free_device_list(device_list);
    }

    void test_port_avail(ib_port_desc_t *port_desc) {
        char *p, *dev_name;
        unsigned port_num;

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
        port_attr_test(dev_name, port_num, port_desc);

        free(dev_name);
    }

    void test_address_pack(uct_ib_address_type_t scope, uint64_t subnet_prefix) {
        static const uint16_t lid_in = 0x1ee7;
        union ibv_gid gid_in, gid_out;
        uct_ib_address_t *ib_addr;
        uint16_t lid_out;
        uint8_t is_global;

        ib_addr = (uct_ib_address_t*)malloc(uct_ib_address_size(scope));

        gid_in.global.subnet_prefix = subnet_prefix;
        gid_in.global.interface_id  = 0xdeadbeef;
        uct_ib_address_pack(ib_device(m_e1), scope, &gid_in, lid_in, ib_addr);

        uct_ib_address_unpack(ib_addr, &lid_out, &is_global, &gid_out);

        EXPECT_EQ((scope != UCT_IB_ADDRESS_TYPE_LINK_LOCAL), is_global);
        EXPECT_EQ(lid_in, lid_out);

        if (is_global) {
            EXPECT_EQ(gid_in.global.subnet_prefix, gid_out.global.subnet_prefix);
            EXPECT_EQ(gid_in.global.interface_id,  gid_out.global.interface_id);
        }

        free(ib_addr);
    }

    void send_recv_short() {
        uint64_t send_data   = 0xdeadbeef;
        uint64_t test_ib_hdr = 0xbeef;
        recv_desc_t *recv_buffer;

        initialize();
        check_caps(UCT_IFACE_FLAG_AM_SHORT);

        recv_buffer = (recv_desc_t *) malloc(sizeof(*recv_buffer) + sizeof(uint64_t));
        recv_buffer->length = 0; /* Initialize length to 0 */

        /* set a callback for the uct to invoke for receiving the data */
        uct_iface_set_am_handler(m_e2->iface(), 0, ib_am_handler , recv_buffer, UCT_AM_CB_FLAG_SYNC);

        /* send the data */
        uct_ep_am_short(m_e1->ep(0), 0, test_ib_hdr, &send_data, sizeof(send_data));

        short_progress_loop(100.0);

        ASSERT_EQ(sizeof(send_data), recv_buffer->length);
        EXPECT_EQ(send_data, *(uint64_t*)(recv_buffer+1));

        free(recv_buffer);
    }

    uct_ib_device_t *ib_device(entity *entity) {
        uct_ib_iface_t *iface = ucs_derived_of(entity->iface(), uct_ib_iface_t);
        return uct_ib_iface_device(iface);
    }

protected:
    entity *m_e1, *m_e2;
};


UCS_TEST_P(test_uct_ib, non_default_pkey, "IB_PKEY=0x2")
{
    ib_port_desc_t *port_desc;

    /* check if the configured pkey exists in the port's pkey table.
     * skip this test if it doesn't. */
    port_desc = (ib_port_desc_t *) calloc(1, sizeof(*port_desc));
    test_port_avail(port_desc);

    if (port_desc->have_pkey) {
        free(port_desc);
    } else {
        free(port_desc);
        UCS_TEST_SKIP_R("pkey not found or not an IB port");
    }

    send_recv_short();
}

UCS_TEST_P(test_uct_ib, non_default_lmc, "IB_LID_PATH_BITS=1")
{
    ib_port_desc_t *port_desc;

    /* check if a non zero lmc is set on the port.
     * skip this test if it isn't. */
    port_desc = (ib_port_desc_t *) calloc(1, sizeof(*port_desc));
    test_port_avail(port_desc);

    if (port_desc->have_lmc) {
        free(port_desc);
    } else {
        free(port_desc);
        UCS_TEST_SKIP_R("lmc is set to zero on an IB port");
    }

    send_recv_short();
}

#if HAVE_DECL_IBV_LINK_LAYER_ETHERNET
UCS_TEST_P(test_uct_ib, non_default_gid_idx, "IB_GID_INDEX=1")
{
    ib_port_desc_t *port_desc;

    /* check if a non zero gid index can be used on the port.
     * skip this test if it isn't. */
    port_desc = (ib_port_desc_t *) calloc(1, sizeof(*port_desc));
    test_port_avail(port_desc);

    if (port_desc->have_valid_gid_idx) {
        free(port_desc);
    } else {
        free(port_desc);
        UCS_TEST_SKIP_R("the configured gid index (1) cannot be used on the port");
    }

    send_recv_short();
}
#endif

UCS_TEST_P(test_uct_ib, address_pack) {
    initialize();
    test_address_pack(UCT_IB_ADDRESS_TYPE_LINK_LOCAL, UCT_IB_LINK_LOCAL_PREFIX);
    test_address_pack(UCT_IB_ADDRESS_TYPE_SITE_LOCAL, UCT_IB_SITE_LOCAL_PREFIX | htonll(0x7200));
    test_address_pack(UCT_IB_ADDRESS_TYPE_GLOBAL,     0xdeadfeedbeefa880ul);
}


UCT_INSTANTIATE_IB_TEST_CASE(test_uct_ib);
