/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_test.h"
#include "ud_base.h"
extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_iface.h>
};

class test_ud_pending : public ud_base_test {
public:
    void dispatch_req(uct_pending_req_t *r) {
        EXPECT_UCS_OK(tx(m_e1));
    } 

    static const int N; 
    static const int W; 
    static int req_count;
    static test_ud_pending *me;

    static ucs_status_t pending_cb_dispatch(uct_pending_req_t *r)
    {
        req_count++;
        me->dispatch_req(r);
        return UCS_OK;
    }

    static ucs_status_t pending_cb(uct_pending_req_t *r)
    {
        req_count++;
        return UCS_OK;
    }

    static ucs_status_t pending_cb_busy(uct_pending_req_t *r)
    {
        return UCS_ERR_BUSY;
    }

};

const int test_ud_pending::N = 13; 
const int test_ud_pending::W = 6; 
int test_ud_pending::req_count = 0;
test_ud_pending *test_ud_pending::me = 0;

/* add/purge requests */
UCS_TEST_P(test_ud_pending, async_progress) {
    uct_pending_req_t r[N];
    int i;

    req_count = 0;
    connect();

    set_tx_win(m_e1, 2);
    EXPECT_UCS_OK(tx(m_e1));

    for(i = 0; i < N; i++) {
        EXPECT_EQ(UCS_OK, uct_ep_pending_add(m_e1->ep(0), &r[i]));
    }
    twait(300);
    /* requests must not be dispatched from async progress */
    EXPECT_EQ(0, req_count);
    uct_ep_pending_purge(m_e1->ep(0), pending_cb);
    EXPECT_EQ(N, req_count);
}

UCS_TEST_P(test_ud_pending, sync_progress) {
    uct_pending_req_t r[N];
    int i;

    req_count = 0;
    connect();

    set_tx_win(m_e1, 2);
    EXPECT_UCS_OK(tx(m_e1));

    for(i = 0; i < N; i++) {
        r[i].func = pending_cb;
        EXPECT_EQ(UCS_OK, uct_ep_pending_add(m_e1->ep(0), &r[i]));
    }
    short_progress_loop();
    /* requests must be dispatched from progress */
    EXPECT_EQ(N, req_count);
    uct_ep_pending_purge(m_e1->ep(0), pending_cb);
    EXPECT_EQ(N, req_count);
}

UCS_TEST_P(test_ud_pending, err_busy) {
    uct_pending_req_t r[N];
    int i;

    req_count = 0;
    connect();

    set_tx_win(m_e1, 2);
    EXPECT_UCS_OK(tx(m_e1));

    for(i = 0; i < N; i++) {
        r[i].func = pending_cb_busy;
        EXPECT_EQ(UCS_OK, uct_ep_pending_add(m_e1->ep(0), &r[i]));
    }
    short_progress_loop();
    /* requests will not be dispatched from progress */
    EXPECT_EQ(0, req_count);
    uct_ep_pending_purge(m_e1->ep(0), pending_cb);
    EXPECT_EQ(N, req_count);
}

UCS_TEST_P(test_ud_pending, connect)
{
    uct_pending_req_t r[N];
    int i;

    req_count = 0;
    me = this;
    m_e1->connect_to_iface(0, *m_e2);
    set_tx_win(m_e1, UCT_UD_CA_MAX_WINDOW);
    /* ep is not connected yet */
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(m_e1));
    
    /* queuee some work */
    for(i = 0; i < N; i++) {
        r[i].func = pending_cb_dispatch;
        EXPECT_EQ(UCS_OK, uct_ep_pending_add(m_e1->ep(0), &r[i]));
    }
    short_progress_loop();
    /* now all work should be complete */
    EXPECT_EQ(N, req_count);
    uct_ep_pending_purge(m_e1->ep(0), pending_cb);
}

UCS_TEST_P(test_ud_pending, window)
{
    int i;
    uct_pending_req_t r;
    req_count = 0;
    me = this;
    connect();
    set_tx_win(m_e1, W+1);
    for (i = 0; i < W; i ++) {
        EXPECT_UCS_OK(tx(m_e1));
    }
    EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(m_e1));
    r.func = pending_cb_dispatch;
    EXPECT_EQ(UCS_OK, uct_ep_pending_add(m_e1->ep(0), &r));
    short_progress_loop();
    EXPECT_EQ(1, req_count);
    uct_ep_pending_purge(m_e1->ep(0), pending_cb);
}

UCS_TEST_P(test_ud_pending, tx_wqe)
{
    int i;
    uct_pending_req_t r;
    ucs_status_t status;
    req_count = 0;
    me = this;
    disable_async(m_e1);
    disable_async(m_e2);
    connect();
    /* set big window */
    set_tx_win(m_e1, 8192);
    i = 0;
    do {
       status = tx(m_e1);
       i++;
    } while (status == UCS_OK);

    r.func = pending_cb_dispatch;
    EXPECT_EQ(UCS_OK, uct_ep_pending_add(m_e1->ep(0), &r));
    short_progress_loop();
    EXPECT_EQ(1, req_count);
    short_progress_loop();
    uct_ep_pending_purge(m_e1->ep(0), pending_cb);
}

_UCT_INSTANTIATE_TEST_CASE(test_ud_pending, ud)
_UCT_INSTANTIATE_TEST_CASE(test_ud_pending, ud_mlx5)

