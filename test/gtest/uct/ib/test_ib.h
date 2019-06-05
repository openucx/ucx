/**
* Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
#include <uct/uct_test.h>

extern "C" {
#include <poll.h>
#include <uct/api/uct.h>
#include <ucs/time/time.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/ib/base/ib_md.h>
}


class test_uct_ib : public uct_test {
public:
    typedef struct {
        unsigned length;
        /* data follows */
    } recv_desc_t;

    test_uct_ib();
    void init();
    void cleanup();
    static ucs_status_t ib_am_handler(void *arg, void *data,
                                      size_t length, unsigned flags);
    bool test_eth_port();
    bool lmc_find();
    void test_address_pack(uint64_t subnet_prefix);
    void send_recv_short();
    uct_ib_device_t *ib_device(entity *entity);

protected:
    entity *m_e1, *m_e2;
    static size_t m_ib_am_handler_counter;
    std::string m_dev_name;
    unsigned m_port;
    struct ibv_context *m_ibctx;
    struct ibv_port_attr m_port_attr;
};
