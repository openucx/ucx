/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_test.h"


class test_connect : public uct_test {
public:
    test_connect() : m_done(false), m_desc(NULL) {
    }

    virtual void init() {
        uct_test::init();
        entity *e = new entity(GetParam(), 0);
        e->add_ep();
        m_done = false;
        m_entities.push_back(e);
    }

    /*
     * Note: this can be invoked from another thread!
     */
    static ucs_status_t am_handler(void *arg, void *data, size_t length, void *desc)
    {
        test_connect *self = reinterpret_cast<test_connect*>(arg);
        if (desc != data) {
            memcpy(desc, data, length);
        }

        self->m_desc = desc;
        ucs_compiler_fence();
        self->m_done = true;
        return UCS_INPROGRESS;
    }

protected:
    const entity& e() {
        return ent(0);
    }

    bool  m_done;
    void* m_desc;
};

enum {
    AM_ID = 7,
};

UCS_TEST_P(test_connect, to_iface) {
    ucs_status_t status;
    uint64_t header = 0xdeadbeef;
    uint64_t data   = 1337;

    check_caps(UCT_IFACE_FLAG_CONNECT_TO_IFACE);

    uct_iface_addr_t *addr = (uct_iface_addr_t*)malloc(e().iface_attr().iface_addr_len);

    status = uct_iface_get_address(e().iface(), addr);
    ASSERT_UCS_OK(status);

    status = uct_iface_set_am_handler(e().iface(), AM_ID, am_handler, this);
    ASSERT_UCS_OK(status);

    status = uct_ep_connect_to_iface(e().ep(0), addr);
    ASSERT_UCS_OK(status);

    free(addr);

    status = uct_ep_am_short(e().ep(0), AM_ID, header, &data, sizeof(data));
    ASSERT_UCS_OK(status);

    while (!m_done) {
        sched_yield();
    }

    ucs_compiler_fence();

    EXPECT_EQ(header, *((uint64_t*)m_desc + 0));
    EXPECT_EQ(data,   *((uint64_t*)m_desc + 1));

    uct_iface_release_am_desc(e().iface(), m_desc);
    m_desc = NULL;
}

UCT_INSTANTIATE_TEST_CASE(test_connect)


