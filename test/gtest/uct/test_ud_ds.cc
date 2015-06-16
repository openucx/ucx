/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_test.h"
extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include "uct/ib/ud/ud_def.h"
#include "uct/ib/ud/ud_ep.h"
#include "uct/ib/ud/ud_iface.h"
};
/* test ud connect data structures */
class test_ud_ds : public uct_test {
public:
    virtual void init() {
        uct_test::init();

        m_e1 = create_entity(0);
        m_e2 = create_entity(0);

        m_entities.push_back(m_e1);
        m_entities.push_back(m_e2);

        uct_iface_get_address(m_e1->iface(), (struct sockaddr *)&adr1);
        uct_iface_get_address(m_e2->iface(), (struct sockaddr *)&adr2);
    }
    uct_ud_iface_t *iface(entity *e) {
        return ucs_derived_of(e->iface(), uct_ud_iface_t);
    }

    uct_ud_ep_t *ep(entity *e, int i) {
        return ucs_derived_of(e->ep(i), uct_ud_ep_t);
    }

    void cleanup() {
        uct_test::cleanup();
    }

    void test_cep_insert(entity *e, uct_sockaddr_ib_t *adr, unsigned base);

protected:
    entity *m_e1, *m_e2;
    uct_sockaddr_ib_t adr1, adr2;
    static unsigned N;
};

unsigned test_ud_ds::N = 1000;

UCS_TEST_P(test_ud_ds, if_addr) {
    EXPECT_EQ(adr1.lid, adr2.lid);
    EXPECT_NE(adr1.qp_num, adr2.qp_num);
}

void test_ud_ds::test_cep_insert(entity *e, uct_sockaddr_ib_t *adr, unsigned base)
{
    unsigned i;
    uct_ud_ep_t *my_ep;

    for (i = 0; i < N; i++) {
        e->create_ep(i + base);
        EXPECT_EQ(i+base, ep(e, i + base)->ep_id);
        EXPECT_EQ((unsigned)UCT_UD_EP_NULL_ID, ep(e, i + base)->dest_ep_id);
        EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(e), adr, ep(e, i + base), UCT_UD_EP_CONN_ID_MAX));
        EXPECT_EQ(i, ep(e, i + base)->conn_id);
    }
    /* lookup non existing ep */
    my_ep = uct_ud_iface_cep_lookup(iface(e), adr, 3333);
    EXPECT_TRUE(my_ep == NULL);
    for (i = 0; i < N; i++) {
        my_ep = uct_ud_iface_cep_lookup(iface(e), adr, i);
        EXPECT_TRUE(my_ep != NULL);
        EXPECT_EQ(i+base, ep(e, i + base)->ep_id);
        EXPECT_EQ(i, ep(e, i + base)->conn_id);
    }
}

/* simulate creq send */
UCS_TEST_P(test_ud_ds, cep_insert) {
    test_cep_insert(m_e1, &adr1, 0);
    test_cep_insert(m_e1, &adr2, N);
}

UCS_TEST_P(test_ud_ds, cep_rollback) {

    m_e1->create_ep(0);
    EXPECT_EQ(0U, ep(m_e1, 0)->ep_id);
    EXPECT_EQ((unsigned)UCT_UD_EP_NULL_ID, ep(m_e1, 0)->dest_ep_id);
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, 0), UCT_UD_EP_CONN_ID_MAX));
    EXPECT_EQ(0U, ep(m_e1, 0)->conn_id);

    uct_ud_iface_cep_rollback(iface(m_e1), &adr1, ep(m_e1, 0));

    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, 0), UCT_UD_EP_CONN_ID_MAX));
    EXPECT_EQ(0U, ep(m_e1, 0)->conn_id);
}

UCS_TEST_P(test_ud_ds, cep_replace) {

    uct_ud_ep_t *my_ep;

    /* add N connections */
    test_cep_insert(m_e1, &adr1, 0);

    /* Assume that we have 5 connections pending and 3 CREQs received */
    m_e1->create_ep(N);
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, N), N+1));
    EXPECT_EQ(N+1, ep(m_e1, N)->conn_id);

    m_e1->create_ep(N+1);
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, N+1), N+4));
    EXPECT_EQ(N+4, ep(m_e1, N+1)->conn_id);

    m_e1->create_ep(N+2);
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, N+2), N+5));
    EXPECT_EQ(N+5, ep(m_e1, N+2)->conn_id);

    /* we initiate 2 connections */
    my_ep = uct_ud_iface_cep_lookup(iface(m_e1), &adr1, UCT_UD_EP_CONN_ID_MAX);
    EXPECT_TRUE(my_ep == NULL);
    m_e1->create_ep(N+3);
    /* slot N must be free. conn_id will be N+1 when inserting ep with no id */
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, N+3), UCT_UD_EP_CONN_ID_MAX));
    EXPECT_EQ(N, ep(m_e1, N+3)->conn_id);

    /* slot N+1 already occupied */
    my_ep = uct_ud_iface_cep_lookup(iface(m_e1), &adr1, UCT_UD_EP_CONN_ID_MAX);
    EXPECT_TRUE(my_ep != NULL);
    EXPECT_EQ(N+1, my_ep->conn_id);
}

_UCT_INSTANTIATE_TEST_CASE(test_ud_ds, ud)

