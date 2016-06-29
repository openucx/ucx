/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.All rights reserved.
* See file LICENSE for terms.
*/

extern "C" {
#include <uct/api/uct.h>
#include <uct/ib/dc/base/dc_iface.h>
#include <uct/ib/dc/base/dc_ep.h>
}
#include <common/test.h>
#include "uct_test.h"

class test_dc : public uct_test {
public:
    virtual void init() {
        uct_test::init();

        m_e1 = uct_test::create_entity(0);
        m_entities.push_back(m_e1);

        m_e2 = uct_test::create_entity(0);
        m_entities.push_back(m_e2);

        uct_iface_set_am_handler(m_e1->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
        uct_iface_set_am_handler(m_e2->iface(), 0, am_dummy_handler,
                                 NULL, UCT_AM_CB_FLAG_SYNC);
    }

    uct_dc_iface_t* dc_iface(entity *e) {
        return ucs_derived_of(e->iface(), uct_dc_iface_t);
    }

    uct_dc_ep_t* dc_ep(entity *e, int idx) {
        return ucs_derived_of(e->ep(idx), uct_dc_ep_t);
    }

    static ucs_status_t am_dummy_handler(void *arg, void *data, size_t length, void *desc) {
        return UCS_OK;
    }

    virtual void cleanup() {
        uct_test::cleanup();
    }

protected:
    entity *m_e1, *m_e2;
};

UCS_TEST_P(test_dc, dcs_single) {
    ucs_status_t status;
    uct_dc_ep_t *ep;
    uct_dc_iface_t *iface;

    m_e1->connect_to_iface(0, *m_e2);
    ep = dc_ep(m_e1, 0);
    iface = dc_iface(m_e1);
    EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
    status = uct_ep_am_short(m_e1->ep(0), 0, 0, NULL, 0);
    EXPECT_UCS_OK(status);
    /* dci 0 must be assigned to the ep */
    EXPECT_EQ(iface->tx.dcis_stack[0], ep->dci);
    EXPECT_EQ(1, iface->tx.stack_top);
    EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);

    flush();

    /* after the flush dci must be released */
    EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
    EXPECT_EQ(0, iface->tx.stack_top);
    EXPECT_EQ(0, iface->tx.dcis_stack[0]);
}

UCS_TEST_P(test_dc, dcs_multi) {
    ucs_status_t status;
    uct_dc_ep_t *ep;
    uct_dc_iface_t *iface;
    unsigned i;

    iface = dc_iface(m_e1);
    for (i = 0; i <= iface->tx.ndci; i++) {
        m_e1->connect_to_iface(i, *m_e2);
    }

    for (i = 0; i < iface->tx.ndci; i++) {
        ep = dc_ep(m_e1, i);
        EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
        status = uct_ep_am_short(m_e1->ep(i), 0, 0, NULL, 0);
        EXPECT_UCS_OK(status);

        /* dci on free LIFO must be assigned to the ep */
        EXPECT_EQ(iface->tx.dcis_stack[i], ep->dci);
        EXPECT_EQ(i+1, iface->tx.stack_top);
        EXPECT_EQ(ep, iface->tx.dcis[ep->dci].ep);
    }

    /* this should fail because there are no free dci */
    status = uct_ep_am_short(m_e1->ep(i), 0, 0, NULL, 0);
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, status);

    flush();

    /* after the flush dci must be released */
    
    EXPECT_EQ(0, iface->tx.stack_top);
    for (i = 0; i < iface->tx.ndci; i++) {
        ep = dc_ep(m_e1, i);
        EXPECT_EQ(UCT_DC_EP_NO_DCI, ep->dci);
    }
}

_UCT_INSTANTIATE_TEST_CASE(test_dc, dc)
