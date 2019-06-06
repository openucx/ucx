/**
* Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <uct/ib/test_ib.h>


class test_uct_ib_pkey : public test_uct_ib {
public:
    void init() {
        test_uct_ib::init();

        if (IBV_PORT_IS_LINK_LAYER_ETHERNET(&m_port_attr)) {
            test_uct_ib::cleanup();
            /* no pkeys for Ethernet */
            UCS_TEST_SKIP_R("skip pkey test for port with Ethernet link type");
        }
    }

    uint16_t query_pkey(uint16_t pkey_idx) {
        uint16_t pkey;

        if (ibv_query_pkey(m_ibctx, m_port, pkey_idx, &pkey)) {
            UCS_TEST_ABORT("Failed to query pkey on port " << m_port <<
                           " on device: " << m_dev_name);
        }
        return ntohs(pkey) & UCT_IB_PKEY_PARTITION_MASK;
    }

    bool pkey_find() {
        uct_ib_iface_config_t *ib_config =
            ucs_derived_of(m_iface_config, uct_ib_iface_config_t);

        /* check if the configured pkey exists in the port's pkey table */
        for (uint16_t table_idx = 0; table_idx < m_port_attr.pkey_tbl_len; table_idx++) {
            uint16_t pkey = query_pkey(table_idx);
            if (pkey == ib_config->pkey_value) {
                return true;
            }
        }

        return false;
    }
};

UCS_TEST_P(test_uct_ib_pkey, non_default_pkey) {
    /* test with invalid pkey set (0 - trival case, start from 1) */
    for (unsigned pkey = 1; pkey <= UCT_IB_PKEY_PARTITION_MASK; pkey++) {
        modify_config("IB_PKEY", "0x" + ucs::to_hex_string(pkey));

        if (!pkey_find()) {
            send_recv_short();
            break;
        }
    }
}

UCS_TEST_P(test_uct_ib_pkey, all_avail_pkeys) {
    /* test all pkeys that are configured for the device */
    for (uint16_t table_idx = 0; table_idx < m_port_attr.pkey_tbl_len; table_idx++) {
        uint16_t pkey = query_pkey(table_idx);
        if (!(pkey & UCT_IB_PKEY_MEMBERSHIP_MASK)) {
            continue;
        }
        modify_config("IB_PKEY", "0x" + ucs::to_hex_string(pkey));
        send_recv_short();
    }
}


UCT_INSTANTIATE_IB_TEST_CASE(test_uct_ib_pkey);
