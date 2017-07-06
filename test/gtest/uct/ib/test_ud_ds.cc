/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/uct_test.h>

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

    void test_cep_insert(entity *e, uct_ib_address_t *ib_addr, uct_ud_iface_addr_t *if_addr, unsigned base);

protected:
    entity *m_e1, *m_e2;
    uct_ib_address_t *ib_adr1, *ib_adr2;
    uct_ud_iface_addr_t if_adr1, if_adr2;
    static unsigned N;
};

unsigned test_ud_ds::N = 1000;

UCS_TEST_P(test_ud_ds, if_addr) {
    union ibv_gid gid1, gid2;
    uint16_t lid1, lid2;
    uint8_t is_global;
    uct_ib_address_unpack(ib_adr1, &lid1, &is_global, &gid1);
    uct_ib_address_unpack(ib_adr2, &lid2, &is_global, &gid2);
    EXPECT_EQ(lid1, lid2);
    EXPECT_EQ(gid1.global.subnet_prefix, gid2.global.subnet_prefix);
    EXPECT_EQ(gid1.global.interface_id, gid2.global.interface_id);
    EXPECT_NE(uct_ib_unpack_uint24(if_adr1.qp_num),
              uct_ib_unpack_uint24(if_adr2.qp_num));
}

void test_ud_ds::test_cep_insert(entity *e, uct_ib_address_t *ib_addr,
                                 uct_ud_iface_addr_t *if_addr, unsigned base)
{
    unsigned i;
    uct_ud_ep_t *my_ep;

    for (i = 0; i < N; i++) {
        e->create_ep(i + base);
        EXPECT_EQ(i+base, ep(e, i + base)->ep_id);
        EXPECT_EQ((unsigned)UCT_UD_EP_NULL_ID, ep(e, i + base)->dest_ep_id);
        EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(e), ib_addr, if_addr, ep(e, i + base), UCT_UD_EP_CONN_ID_MAX));
        EXPECT_EQ(i, ep(e, i + base)->conn_id);
    }
    /* lookup non existing ep */
    my_ep = uct_ud_iface_cep_lookup(iface(e), ib_addr, if_addr, 3333);
    EXPECT_TRUE(my_ep == NULL);
    for (i = 0; i < N; i++) {
        my_ep = uct_ud_iface_cep_lookup(iface(e), ib_addr, if_addr, i);
        EXPECT_TRUE(my_ep != NULL);
        EXPECT_EQ(i+base, ep(e, i + base)->ep_id);
        EXPECT_EQ(i, ep(e, i + base)->conn_id);
    }
}

/* simulate creq send */
UCS_TEST_P(test_ud_ds, cep_insert) {
    test_cep_insert(m_e1, ib_adr1, &if_adr1, 0);
    test_cep_insert(m_e1, ib_adr2, &if_adr2, N);
}

UCS_TEST_P(test_ud_ds, cep_rollback) {

    m_e1->create_ep(0);
    EXPECT_EQ(0U, ep(m_e1, 0)->ep_id);
    EXPECT_EQ((unsigned)UCT_UD_EP_NULL_ID, ep(m_e1, 0)->dest_ep_id);
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), ib_adr1, &if_adr1, ep(m_e1, 0), UCT_UD_EP_CONN_ID_MAX));
    EXPECT_EQ(0U, ep(m_e1, 0)->conn_id);

    uct_ud_iface_cep_rollback(iface(m_e1), ib_adr1, &if_adr1, ep(m_e1, 0));

    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), ib_adr1, &if_adr1, ep(m_e1, 0), UCT_UD_EP_CONN_ID_MAX));
    EXPECT_EQ(0U, ep(m_e1, 0)->conn_id);
}

UCS_TEST_P(test_ud_ds, cep_replace) {

    uct_ud_ep_t *my_ep;

    /* add N connections */
    test_cep_insert(m_e1, ib_adr1, &if_adr1, 0);

    /* Assume that we have 5 connections pending and 3 CREQs received */
    m_e1->create_ep(N);
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), ib_adr1, &if_adr1, ep(m_e1, N), N+1));
    EXPECT_EQ(N+1, ep(m_e1, N)->conn_id);

    m_e1->create_ep(N+1);
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), ib_adr1, &if_adr1, ep(m_e1, N+1), N+4));
    EXPECT_EQ(N+4, ep(m_e1, N+1)->conn_id);

    m_e1->create_ep(N+2);
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), ib_adr1, &if_adr1, ep(m_e1, N+2), N+5));
    EXPECT_EQ(N+5, ep(m_e1, N+2)->conn_id);

    /* we initiate 2 connections */
    my_ep = uct_ud_iface_cep_lookup(iface(m_e1), ib_adr1, &if_adr1, UCT_UD_EP_CONN_ID_MAX);
    EXPECT_TRUE(my_ep == NULL);
    m_e1->create_ep(N+3);
    /* slot N must be free. conn_id will be N+1 when inserting ep with no id */
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), ib_adr1, &if_adr1, ep(m_e1, N+3), UCT_UD_EP_CONN_ID_MAX));
    EXPECT_EQ(N, ep(m_e1, N+3)->conn_id);

    /* slot N+1 already occupied */
    my_ep = uct_ud_iface_cep_lookup(iface(m_e1), ib_adr1, &if_adr1, UCT_UD_EP_CONN_ID_MAX);
    EXPECT_TRUE(my_ep != NULL);
    EXPECT_EQ(N+1, my_ep->conn_id);
}

_UCT_INSTANTIATE_TEST_CASE(test_ud_ds, ud)
_UCT_INSTANTIATE_TEST_CASE(test_ud_ds, ud_mlx5)

