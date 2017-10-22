/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ud_base.h"

#include <uct/uct_test.h>

extern "C" {
#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_iface.h>
}


class test_ud_pending : public ud_base_test {
public:
    uct_pending_req_t m_r[64];

    void dispatch_req(uct_pending_req_t *r) {
        EXPECT_UCS_OK(tx(m_e1));
    } 

    void post_pending_reqs(void) 
    {
        int i;

        req_count = 0;
        me = this;
        m_e1->connect_to_iface(0, *m_e2);
        set_tx_win(m_e1, UCT_UD_CA_MAX_WINDOW);
        /* ep is not connected yet */
        EXPECT_EQ(UCS_ERR_NO_RESOURCE, tx(m_e1));

        /* queuee some work */
        for(i = 0; i < N; i++) {
            m_r[i].func = pending_cb_dispatch;
            EXPECT_EQ(UCS_OK, uct_ep_pending_add(m_e1->ep(0), &m_r[i]));
        }
    }

    void check_pending_reqs(bool wait)
    {
        /* wait for all work to be complete */
        ucs_time_t start_time = ucs_get_time();
        while (wait && (req_count < N) &&
               (ucs_get_time() < start_time + ucs_time_from_sec(10.0)))
        {
            progress();
        }
        EXPECT_EQ(N, req_count);
        uct_ep_pending_purge(m_e1->ep(0), purge_cb, NULL);
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

    static void purge_cb(uct_pending_req_t *r, void *arg)
    {
        req_count++;
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
    uct_ep_pending_purge(m_e1->ep(0), purge_cb, NULL);
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
    wait_for_value(&req_count, N, true);
    /* requests must be dispatched from progress */
    EXPECT_EQ(N, req_count);
    uct_ep_pending_purge(m_e1->ep(0), purge_cb, NULL);
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
    uct_ep_pending_purge(m_e1->ep(0), purge_cb, NULL);
    EXPECT_EQ(N, req_count);
}

UCS_TEST_P(test_ud_pending, connect)
{
    disable_async(m_e1);
    disable_async(m_e2);
    post_pending_reqs();
    check_pending_reqs(true);
}

UCS_TEST_P(test_ud_pending, flush)
{
    disable_async(m_e1);
    disable_async(m_e2);
    post_pending_reqs();
    flush();
    check_pending_reqs(false);
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
    wait_for_value(&req_count, 1, true);
    EXPECT_EQ(1, req_count);
    uct_ep_pending_purge(m_e1->ep(0), purge_cb, NULL);
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
    wait_for_value(&req_count, 1, true);
    EXPECT_EQ(1, req_count);
    short_progress_loop();
    uct_ep_pending_purge(m_e1->ep(0), purge_cb, NULL);
}

_UCT_INSTANTIATE_TEST_CASE(test_ud_pending, ud)
_UCT_INSTANTIATE_TEST_CASE(test_ud_pending, ud_mlx5)

