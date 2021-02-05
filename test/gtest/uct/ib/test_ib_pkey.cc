/**
* Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <uct/ib/test_ib.h>


class test_uct_ib_pkey : public test_uct_ib_with_specific_port {
protected:
    test_uct_ib_pkey() {
        m_pkey[0]       = UCT_IB_ADDRESS_INVALID_PKEY;
        m_pkey[1]       = UCT_IB_ADDRESS_INVALID_PKEY;
        m_pkey_index[0] = 0;
        m_pkey_index[1] = 0;
    }

    void check_port_attr() {
        if (IBV_PORT_IS_LINK_LAYER_ETHERNET(&m_port_attr)) {
            /* no pkeys for Ethernet */
            UCS_TEST_SKIP_R("skip pkey test for port with Ethernet link type");
        }
    }

    void check_pkeys() {
        EXPECT_TRUE(check_pkey(m_e1->iface(), m_pkey[0], m_pkey_index[0]));
        EXPECT_TRUE(check_pkey(m_e2->iface(), m_pkey[1], m_pkey_index[1]));
    }

    void cleanup_entities() {
        m_e1->destroy_eps();
        m_e2->destroy_eps();
        m_entities.remove(m_e1);
        m_entities.remove(m_e2);
        m_e1 = NULL;
        m_e2 = NULL;
    }

    void send_recv_short() {
        create_connected_entities();
        check_pkeys();

        test_uct_ib::send_recv_short();

        cleanup_entities();
    }

    uint16_t query_pkey(uint16_t pkey_idx) const {
        uint16_t pkey;

        if (ibv_query_pkey(m_ibctx, m_port, pkey_idx, &pkey)) {
            UCS_TEST_ABORT("Failed to query pkey on port " << m_port <<
                           " on device: " << m_dev_name);
        }
        return ntohs(pkey);
    }

    bool check_pkey(const uct_iface_t *iface, uint16_t pkey,
                    uint16_t pkey_index) const {
        const uct_ib_iface_t *ib_iface = ucs_derived_of(iface, uct_ib_iface_t);
        return ((pkey == ib_iface->pkey) &&
                (pkey_index == ib_iface->pkey_index));
    }

    bool find_default_pkey(uint16_t &pkey, uint16_t &pkey_index) const {
        for (uint16_t table_idx = 0; table_idx < m_port_attr.pkey_tbl_len; table_idx++) {
            uint16_t pkey_value = query_pkey(table_idx);
            if (can_use_pkey(pkey_value)) {
                /* found the first valid pkey with full membership */
                pkey       = pkey_value;
                pkey_index = table_idx;
                return true;
            }
        }

        return false;
    }

    bool can_use_pkey(uint16_t pkey) const {
        return ((pkey != UCT_IB_ADDRESS_INVALID_PKEY) &&
                ((pkey & UCT_IB_PKEY_MEMBERSHIP_MASK) != 0));
    }

    typedef std::pair<
        /* PKEY values */
        std::vector<std::vector<uint16_t> >,
        /* PKEY indices */
        std::vector<std::vector<uint16_t> > > ib_pkey_pairs_t;

    ib_pkey_pairs_t supported_pkey_pairs(bool full_membership_only = true) {
        static std::vector<std::vector<uint16_t> > supported_pkey_pairs;
        static std::vector<std::vector<uint16_t> > supported_pkey_idx_pairs;
        static ib_pkey_pairs_t result;

        if (result.first.empty()) {
            std::vector<uint16_t> supported_pkeys;
            std::vector<uint16_t> supported_pkeys_idx;
            for (uint16_t table_idx = 0;
                 table_idx < m_port_attr.pkey_tbl_len; table_idx++) {
                uint16_t pkey = query_pkey(table_idx);
                if (pkey == UCT_IB_ADDRESS_INVALID_PKEY) {
                    continue;
                }

                supported_pkeys.push_back(pkey);
                supported_pkeys_idx.push_back(table_idx);
            }

            supported_pkey_pairs = ucs::make_pairs(supported_pkeys);
            supported_pkey_idx_pairs = ucs::make_pairs(supported_pkeys_idx);

            result = std::make_pair(supported_pkey_pairs,
                                    supported_pkey_idx_pairs);
        }

        return result;
    }

    uint16_t test_pack_unpack_ib_address(uct_ib_iface_t *iface,
                                         uct_ib_address_t *ib_addr) {
        uct_ib_address_pack_params_t params;

        uct_ib_iface_address_pack(iface, ib_addr);
        uct_ib_address_unpack(ib_addr, &params);
        EXPECT_TRUE(params.flags & UCT_IB_ADDRESS_PACK_FLAG_PKEY);
        EXPECT_EQ(m_pkey[0], params.pkey);

        return params.pkey;
    }

public:
    uint16_t m_pkey[2];
    uint16_t m_pkey_index[2];
};

UCS_TEST_P(test_uct_ib_pkey, default_pkey) {
    if (!find_default_pkey(m_pkey[0], m_pkey_index[0])) {
        UCS_TEST_SKIP_R("unable to find a valid pkey with full membership");
    }

    m_pkey[1]       = m_pkey[0];
    m_pkey_index[1] = m_pkey_index[0];

    send_recv_short();
}

UCS_TEST_P(test_uct_ib_pkey, all_avail_pkeys) {
    /* test all pkeys that are configured for the device */
    for (uint16_t table_idx = 0; table_idx < m_port_attr.pkey_tbl_len; table_idx++) {
        m_pkey[0] = m_pkey[1] = query_pkey(table_idx);
        if (!can_use_pkey(m_pkey[0])) {
            continue;
        }
        modify_config("IB_PKEY", "0x" +
                      ucs::to_hex_string(m_pkey[0] &
                                         UCT_IB_PKEY_PARTITION_MASK));
        m_pkey_index[0] = m_pkey_index[1] = table_idx;
        send_recv_short();
    }
}

UCS_TEST_P(test_uct_ib_pkey, test_pkey_pairs) {
    /* test all pkeys (even with limited membership) that are configured
     * for the device */
    ib_pkey_pairs_t pairs = supported_pkey_pairs(false);

    for (size_t i = 0; i < pairs.first.size(); i++) {
        m_pkey[0]       = pairs.first[i][0];
        m_pkey[1]       = pairs.first[i][1];
        m_pkey_index[0] = pairs.second[i][0];
        m_pkey_index[1] = pairs.second[i][1];

        modify_config("IB_PKEY", "0x" +
                      ucs::to_hex_string(m_pkey[0] &
                                         UCT_IB_PKEY_PARTITION_MASK));
        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        modify_config("IB_PKEY", "0x" +
                      ucs::to_hex_string(m_pkey[1] &
                                         UCT_IB_PKEY_PARTITION_MASK));
        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        m_e1->connect(0, *m_e2, 0);
        m_e2->connect(0, *m_e1, 0);

        check_pkeys();

        /* pack-unpack the first IB iface address */
        uct_ib_iface_t *iface1      = ucs_derived_of(m_e1->iface(),
                                                     uct_ib_iface_t);
         uct_ib_address_t *ib_addr1 =
             (uct_ib_address_t*)ucs_alloca(uct_ib_iface_address_size(iface1));
        uint16_t pkey1              = test_pack_unpack_ib_address(iface1,
                                                                  ib_addr1);

        /* pack-unpack the second IB iface address */
        uct_ib_iface_t *iface2     = ucs_derived_of(m_e2->iface(),
                                                    uct_ib_iface_t);
        uct_ib_address_t *ib_addr2 =
            (uct_ib_address_t*)ucs_alloca(uct_ib_iface_address_size(iface2));
        uint16_t pkey2             = test_pack_unpack_ib_address(iface2,
                                                                 ib_addr2);

        int res = !(/* both PKEYs are with limited membership */
                    !((pkey1 | pkey2) & UCT_IB_PKEY_MEMBERSHIP_MASK) ||
                    /* the PKEYs are not equal */
                    ((pkey1 ^ pkey2) & UCT_IB_PKEY_PARTITION_MASK));
        EXPECT_EQ(res, uct_ib_iface_is_reachable(m_e1->iface(),
                                                 (uct_device_addr_t*)ib_addr2,
                                                 NULL));
        EXPECT_EQ(res, uct_ib_iface_is_reachable(m_e2->iface(),
                                                 (uct_device_addr_t*)ib_addr1,
                                                 NULL));

        if (res) {
            test_uct_ib::send_recv_short();
        }

        cleanup_entities();
    }
}


UCT_INSTANTIATE_IB_TEST_CASE(test_uct_ib_pkey);
