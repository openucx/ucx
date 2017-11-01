/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#ifndef _UD_BASE_TEST
#define _UD_BASE_TEST

#include <uct/uct_test.h>

#include <ucs/time/time.h>
#include <ucs/datastruct/queue.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_iface.h>


#define TEST_UD_PROGRESS_TIMEOUT 300.0
#define TEST_UD_TIMEOUT_IN_SEC   10.0


class ud_base_test : public uct_test {
public:
    virtual void init();

    uct_ud_ep_t *ep(entity *e);

    uct_ud_ep_t *ep(entity *e, int i);

    uct_ud_iface_t *iface(entity *e);

    void connect();

    void cleanup();

    ucs_status_t tx(entity *e);

    ucs_status_t ep_flush_b(entity *e);

    ucs_status_t iface_flush_b(entity *e);

    void set_tx_win(entity *e, uct_ud_psn_t size);

    void disable_async(entity *e);

    virtual void short_progress_loop(double delta_ms=10.0) const;

protected:
    entity *m_e1, *m_e2;
    uint64_t m_dummy;
};

#endif
