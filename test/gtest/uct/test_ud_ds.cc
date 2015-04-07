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

        m_e1 = new entity(GetParam(), 0);
        m_e2 = new entity(GetParam(), 0);

        uct_iface_get_address(m_e1->iface(), &adr1.super);
        uct_iface_get_address(m_e2->iface(), &adr2.super);
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

    void test_cep_insert(entity *e, uct_ud_iface_addr_t *adr);

protected:
    entity *m_e1, *m_e2;
    uct_ud_iface_addr_t adr1, adr2;
    static int N;
};

int test_ud_ds::N = 1000;

UCS_TEST_P(test_ud_ds, if_addr) {
    EXPECT_EQ(adr1.lid, adr2.lid);
    EXPECT_NE(adr1.qp_num, adr2.qp_num);
}

void test_ud_ds::test_cep_insert(entity *e, uct_ud_iface_addr_t *adr)
{
    int i;
    uct_ud_ep_t *my_ep;

    for (i = 0; i < N; i++) {
        e->add_ep();
        //printf("ep id: %d", ep(e, i)->ep_id);
        EXPECT_EQ(ep(e, i)->ep_id, i);
        EXPECT_EQ(ep(e, i)->dest_ep_id, UCT_UD_EP_NULL_ID);
        EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(e), adr, ep(e, i), UCT_UD_EP_CONN_ID_MAX));
        EXPECT_EQ(ep(e, i)->conn_id, i);
    }
    /* lookup non existing ep */
    my_ep = uct_ud_iface_cep_lookup(iface(e), adr, 3333);
    EXPECT_TRUE(my_ep == NULL);
    for (i = 0; i < N; i++) {
        my_ep = uct_ud_iface_cep_lookup(iface(e), adr, i);
        EXPECT_TRUE(my_ep != NULL);
        EXPECT_EQ(ep(e, i)->ep_id, i);
        EXPECT_EQ(ep(e, i)->conn_id, i);
    }
}

/* simulate creq send */
UCS_TEST_P(test_ud_ds, cep_insert) {
    test_cep_insert(m_e1, &adr1);
    test_cep_insert(m_e1, &adr2);
}

UCS_TEST_P(test_ud_ds, cep_replace) {

    uct_ud_ep_t *my_ep;

    /* add N connections */
    test_cep_insert(m_e1, &adr1);

    /* Assume that we have 5 connections pending and 3 CREQs received */
    m_e1->add_ep();
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, N), N+1));
    EXPECT_EQ(ep(m_e1, N)->conn_id, N+1);

    m_e1->add_ep();
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, N+1), N+4));
    EXPECT_EQ(ep(m_e1, N+1)->conn_id, N+4);

    m_e1->add_ep();
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, N+2), N+5));
    EXPECT_EQ(ep(m_e1, N+2)->conn_id, N+5);

    /* we initiate 2 connections */
    my_ep = uct_ud_iface_cep_lookup(iface(m_e1), &adr1, UCT_UD_EP_CONN_ID_MAX);
    EXPECT_TRUE(my_ep == NULL);
    m_e1->add_ep();
    /* slot N must be free. conn_id will be N+1 when inserting ep with no id */
    EXPECT_UCS_OK(uct_ud_iface_cep_insert(iface(m_e1), &adr1, ep(m_e1, N+3), UCT_UD_EP_CONN_ID_MAX));
    EXPECT_EQ(ep(m_e1, N+3)->conn_id, N);

    /* slot N+1 already occupied */
    my_ep = uct_ud_iface_cep_lookup(iface(m_e1), &adr1, UCT_UD_EP_CONN_ID_MAX);
    EXPECT_TRUE(my_ep != NULL);
    EXPECT_EQ(my_ep->conn_id, N+1);

    /* replace ep */
    m_e1->add_ep();
    ep(m_e1, N+4)->conn_id = my_ep->conn_id;
    uct_ud_iface_cep_replace(my_ep, ep(m_e1, N+4), uct_ud_ep_cp);
    EXPECT_EQ(ep(m_e1, N+4)->dest_if->conn_id_last, N+2);
    my_ep = uct_ud_iface_cep_lookup(iface(m_e1), &adr1, N+1);
    EXPECT_TRUE(my_ep != NULL);
    EXPECT_EQ(my_ep, ep(m_e1, N+4));
}



_UCT_INSTANTIATE_TEST_CASE(test_ud_ds, ud)

