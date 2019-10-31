/**
* Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <uct/ib/test_ib.h>


class test_uct_ib_pkey : public test_uct_ib_with_specific_port {
public:
    test_uct_ib_pkey() {
        m_pkey_value = 0;
        m_pkey_index = 0;
    }

    void check_port_attr() {
        if (IBV_PORT_IS_LINK_LAYER_ETHERNET(&m_port_attr)) {
            /* no pkeys for Ethernet */
            UCS_TEST_SKIP_R("skip pkey test for port with Ethernet link type");
        }
    }

    void send_recv_short() {
        create_connected_entities();

        EXPECT_TRUE(check_pkey(m_e1->iface(), m_pkey_value, m_pkey_index));
        EXPECT_TRUE(check_pkey(m_e2->iface(), m_pkey_value, m_pkey_index));

        test_uct_ib::send_recv_short();

        m_e1->destroy_eps();
        m_e2->destroy_eps();
        m_entities.remove(m_e1);
        m_entities.remove(m_e2);
    }

    uint16_t query_pkey(uint16_t pkey_idx) const {
        uint16_t pkey;

        if (ibv_query_pkey(m_ibctx, m_port, pkey_idx, &pkey)) {
            UCS_TEST_ABORT("Failed to query pkey on port " << m_port <<
                           " on device: " << m_dev_name);
        }
        return ntohs(pkey);
    }

    bool check_pkey(const uct_iface_t *iface, uint16_t pkey_value,
                    uint16_t pkey_index) const {
        const uct_ib_iface_t *ib_iface = ucs_derived_of(iface, uct_ib_iface_t);
        return ((pkey_value == ib_iface->pkey_value) &&
                (pkey_index == ib_iface->pkey_index));
    }

    bool find_default_pkey(uint16_t &pkey_value, uint16_t &pkey_index) const {
        for (uint16_t table_idx = 0; table_idx < m_port_attr.pkey_tbl_len; table_idx++) {
            uint16_t pkey = query_pkey(table_idx);
            if (can_use_pkey(pkey)) {
                /* found the first valid pkey with full membership */
                pkey_value = pkey;
                pkey_index = table_idx;
                return true;
            }
        }

        return false;
    }

    bool can_use_pkey(uint16_t pkey_value) const {
        return (pkey_value && (pkey_value & UCT_IB_PKEY_MEMBERSHIP_MASK));
    }

public:
    uint16_t m_pkey_value;
    uint16_t m_pkey_index;
};

UCS_TEST_P(test_uct_ib_pkey, default_pkey) {
    if (!find_default_pkey(m_pkey_value, m_pkey_index)) {
        UCS_TEST_SKIP_R("unable to find a valid pkey with full membership");
    }

    send_recv_short();
}

UCS_TEST_P(test_uct_ib_pkey, all_avail_pkeys) {
    /* test all pkeys that are configured for the device */
    for (uint16_t table_idx = 0; table_idx < m_port_attr.pkey_tbl_len; table_idx++) {
        m_pkey_value = query_pkey(table_idx);
        if (!can_use_pkey(m_pkey_value)) {
            continue;
        }
        modify_config("IB_PKEY", "0x" +
                      ucs::to_hex_string(m_pkey_value &
                                         UCT_IB_PKEY_PARTITION_MASK));
        m_pkey_index = table_idx;
        send_recv_short();
    }
}


UCT_INSTANTIATE_IB_TEST_CASE(test_uct_ib_pkey);
