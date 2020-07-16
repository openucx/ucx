/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ud_base.h"

extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include <uct/ib/ud/base/ud_def.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_iface.h>
}


/* test ud connect data structures */
class test_ud_ds : public uct_test {
public:
    virtual void init() {
        uct_test::init();

        m_e1 = create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = create_entity(0);
        m_entities.push_back(m_e2);

        uct_iface_get_address(m_e1->iface(), (uct_iface_addr_t*)(void *)&if_adr1);
        uct_iface_get_address(m_e2->iface(), (uct_iface_addr_t*)(void *)&if_adr2);

        ib_adr1 = (uct_ib_address_t*)malloc(ucs_derived_of(m_e1->iface(), uct_ib_iface_t)->addr_size);
        ib_adr2 = (uct_ib_address_t*)malloc(ucs_derived_of(m_e2->iface(), uct_ib_iface_t)->addr_size);

        uct_iface_get_device_address(m_e1->iface(), (uct_device_addr_t*)ib_adr1);
        uct_iface_get_device_address(m_e2->iface(), (uct_device_addr_t*)ib_adr2);
    }

    uct_ud_iface_t *iface(entity *e) {
        return ucs_derived_of(e->iface(), uct_ud_iface_t);
    }

    uct_ud_ep_t *ep(entity *e, int i) {
        return ucs_derived_of(e->ep(i), uct_ud_ep_t);
    }

    void cleanup() {
        free(ib_adr2);
        free(ib_adr1);
        uct_test::cleanup();
    }

    void test_cep_insert(entity *e, uct_ib_address_t *ib_addr,
                         uct_ud_iface_addr_t *if_addr, unsigned base);

    void create_and_insert_ep(entity *e, uct_ib_address_t *ib_addr,
                              uct_ud_iface_addr_t *if_addr,
                              uct_ud_ep_conn_sn_t conn_sn, unsigned ep_index,
                              bool ep_private);

    void retrieve_ep(entity *e, uct_ib_address_t *ib_addr,
                     uct_ud_iface_addr_t *if_addr,
                     uct_ud_ep_conn_sn_t conn_sn, bool ep_private);

protected:
    entity *m_e1, *m_e2;
    uct_ib_address_t *ib_adr1, *ib_adr2;
    uct_ud_iface_addr_t if_adr1, if_adr2;
    static unsigned N;
};

unsigned test_ud_ds::N = 1000;

UCS_TEST_P(test_ud_ds, if_addr) {
    uct_ib_address_pack_params_t unpack_params1, unpack_params2;

    uct_ib_address_unpack(ib_adr1, &unpack_params1);
    uct_ib_address_unpack(ib_adr2, &unpack_params2);
    EXPECT_EQ(unpack_params1.lid, unpack_params2.lid);
    EXPECT_EQ(unpack_params1.gid.global.subnet_prefix,
              unpack_params2.gid.global.subnet_prefix);
    EXPECT_EQ(unpack_params1.gid.global.interface_id,
              unpack_params2.gid.global.interface_id);
    EXPECT_NE(uct_ib_unpack_uint24(if_adr1.qp_num),
              uct_ib_unpack_uint24(if_adr2.qp_num));

    EXPECT_TRUE(!(unpack_params1.flags & UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU));
    EXPECT_EQ(UCT_IB_ADDRESS_INVALID_PATH_MTU, unpack_params1.path_mtu);
    EXPECT_TRUE(!(unpack_params2.flags & UCT_IB_ADDRESS_PACK_FLAG_PATH_MTU));
    EXPECT_EQ(UCT_IB_ADDRESS_INVALID_PATH_MTU, unpack_params2.path_mtu);

    EXPECT_TRUE(!(unpack_params1.flags & UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX));
    EXPECT_EQ(UCT_IB_ADDRESS_INVALID_GID_INDEX, unpack_params1.gid_index);
    EXPECT_TRUE(!(unpack_params2.flags & UCT_IB_ADDRESS_PACK_FLAG_GID_INDEX));
    EXPECT_EQ(UCT_IB_ADDRESS_INVALID_GID_INDEX, unpack_params2.gid_index);

    EXPECT_TRUE((unpack_params1.flags & UCT_IB_ADDRESS_PACK_FLAG_PKEY) != 0);
    EXPECT_EQ(UCT_IB_ADDRESS_DEFAULT_PKEY, unpack_params1.pkey);
    EXPECT_TRUE((unpack_params2.flags & UCT_IB_ADDRESS_PACK_FLAG_PKEY) != 0);
    EXPECT_EQ(UCT_IB_ADDRESS_DEFAULT_PKEY, unpack_params2.pkey);
}

void test_ud_ds::test_cep_insert(entity *e, uct_ib_address_t *ib_addr,
                                 uct_ud_iface_addr_t *if_addr, unsigned base)
{
    unsigned i;
    uct_ud_ep_t *my_ep;

    for (i = 0; i < N; i++) {
        uct_ud_ep_conn_sn_t conn_sn =
            uct_ud_iface_cep_get_conn_sn(iface(e), ib_addr, if_addr, 0);
        EXPECT_EQ(i, conn_sn);

        e->create_ep(i + base);
        ep(e, i + base)->conn_sn = conn_sn;
        EXPECT_EQ(i + base, ep(e, i + base)->ep_id);
        EXPECT_EQ((unsigned)UCT_UD_EP_NULL_ID, ep(e, i + base)->dest_ep_id);

        uct_ud_iface_cep_insert(iface(e), ib_addr, if_addr, 0,
                                conn_sn, ep(e, i + base));
    }

    /* lookup non existing ep */
    my_ep = uct_ud_iface_cep_retrieve(iface(e), ib_addr, if_addr, 0, base + 3333, 0);
    EXPECT_TRUE(my_ep == NULL);

    for (i = 0; i < N; i++) {
        my_ep = uct_ud_iface_cep_retrieve(iface(e), ib_addr, if_addr, 0,
                                          i, 0);
        EXPECT_TRUE(my_ep != NULL);
        EXPECT_EQ(i + base, ep(e, i + base)->ep_id);
        EXPECT_EQ(i, ep(e, i + base)->conn_sn);
    }
}

void test_ud_ds::create_and_insert_ep(entity *e, uct_ib_address_t *ib_addr,
                                      uct_ud_iface_addr_t *if_addr,
                                      uct_ud_ep_conn_sn_t conn_sn,
                                      unsigned ep_index, bool ep_private)
{
    uct_ud_ep_t *check_ep;

    check_ep = uct_ud_iface_cep_retrieve(iface(e), ib_addr, if_addr, 0,
                                         conn_sn, !ep_private);
    EXPECT_TRUE(check_ep == NULL);

    e->create_ep(ep_index);
    ep(e, ep_index)->conn_sn    = conn_sn;
    if (ep_private) {
        ep(e, ep_index)->flags |= UCT_UD_EP_FLAG_PRIVATE;
    }
    uct_ud_iface_cep_insert(iface(e), ib_addr, if_addr, 0, conn_sn,
                            ep(e, ep_index));
}

void test_ud_ds::retrieve_ep(entity *e, uct_ib_address_t *ib_addr,
                             uct_ud_iface_addr_t *if_addr,
                             uct_ud_ep_conn_sn_t conn_sn, bool ep_private)
{
    uct_ud_ep_t *check_ep;

    check_ep = uct_ud_iface_cep_retrieve(iface(e), ib_addr, if_addr, 0,
                                         conn_sn, ep_private);
    EXPECT_TRUE(check_ep != NULL);
    EXPECT_EQ(conn_sn, check_ep->conn_sn);
}

/* simulate creq send */
UCS_TEST_P(test_ud_ds, cep_insert) {
    test_cep_insert(m_e1, ib_adr1, &if_adr1, 0);
    test_cep_insert(m_e1, ib_adr2, &if_adr2, N);
}

UCS_TEST_P(test_ud_ds, cep_replace) {
    uct_ud_ep_conn_sn_t conn_sn;

    /* add N connections */
    test_cep_insert(m_e1, ib_adr1, &if_adr1, 0);

    /* Assume that 3 CREQs received */
    create_and_insert_ep(m_e1, ib_adr1, &if_adr1, N + 1, N, true);
    create_and_insert_ep(m_e1, ib_adr1, &if_adr1, N + 4, N + 1, true);
    create_and_insert_ep(m_e1, ib_adr1, &if_adr1, N + 5, N + 2, true);

    /* slot N is free */
    conn_sn = uct_ud_iface_cep_get_conn_sn(iface(m_e1), ib_adr1, &if_adr1, 0);
    EXPECT_EQ(N, conn_sn);
    create_and_insert_ep(m_e1, ib_adr1, &if_adr1, conn_sn, N + 3, false);

    /* slot N + 1 already occupied */
    conn_sn = uct_ud_iface_cep_get_conn_sn(iface(m_e1), ib_adr1, &if_adr1, 0);
    EXPECT_EQ(N + 1, conn_sn);
    retrieve_ep(m_e1, ib_adr1, &if_adr1, conn_sn, true);

    /* slot N + 2 is free */
    conn_sn = uct_ud_iface_cep_get_conn_sn(iface(m_e1), ib_adr1, &if_adr1, 0);
    EXPECT_EQ(N + 2, conn_sn);
    create_and_insert_ep(m_e1, ib_adr1, &if_adr1, conn_sn, N + 4, false);

    /* slot N + 3 is free */
    conn_sn = uct_ud_iface_cep_get_conn_sn(iface(m_e1), ib_adr1, &if_adr1, 0);
    EXPECT_EQ(N + 3, conn_sn);
    create_and_insert_ep(m_e1, ib_adr1, &if_adr1, conn_sn, N + 5, false);

    /* slot N + 4 already occupied */
    conn_sn = uct_ud_iface_cep_get_conn_sn(iface(m_e1), ib_adr1, &if_adr1, 0);
    EXPECT_EQ(N + 4, conn_sn);
    retrieve_ep(m_e1, ib_adr1, &if_adr1, conn_sn, true);

    /* slot N + 5 already occupied */
    conn_sn = uct_ud_iface_cep_get_conn_sn(iface(m_e1), ib_adr1, &if_adr1, 0);
    EXPECT_EQ(N + 5, conn_sn);
    retrieve_ep(m_e1, ib_adr1, &if_adr1, conn_sn, true);

    /* slot N already occupied */
    retrieve_ep(m_e1, ib_adr1, &if_adr1, N, false);

    /* slot N + 2 already occupied */
    retrieve_ep(m_e1, ib_adr1, &if_adr1, N + 2, false);

    /* slot N + 3 already occupied */
    retrieve_ep(m_e1, ib_adr1, &if_adr1, N + 3, false);
}

UCT_INSTANTIATE_UD_TEST_CASE(test_ud_ds)
