/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/gtest/test.h>
extern "C" {
#include <uct/api/uct.h>
}


class test_uct : public ucs::test {
};


UCS_TEST_F(test_uct, query_resources) {
    ucs_status_t status;
    uct_context_h ucth;
    uct_resource_desc_t *resources;
    unsigned num_resources;

    ucth = NULL;
    status = uct_init(&ucth);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(ucth != NULL);

    status = uct_query_resources(ucth, &resources, &num_resources);
    ASSERT_UCS_OK(status);

    for (unsigned i = 0; i < num_resources; ++i) {
        uct_resource_desc_t *res = &resources[i];
        EXPECT_TRUE(strcmp(res->tl_name, ""));
        EXPECT_TRUE(strcmp(res->hw_name, ""));
        EXPECT_GT(res->latency, (uint64_t)0);
        EXPECT_GT(res->bandwidth, (size_t)0);
        UCS_TEST_MESSAGE << i << ": " << res->tl_name <<
                        " on " << res->hw_name <<
                        " at " << (res->bandwidth / 1024.0 / 1024.0) << " MB/sec";
    }

    uct_release_resource_list(resources);

    uct_cleanup(ucth);
}

UCS_TEST_F(test_uct, open_iface) {
    ucs_status_t status;
    uct_context_h ucth;
    uct_resource_desc_t *resources;
    unsigned num_resources;

    ucth = NULL;
    status = uct_init(&ucth);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(ucth != NULL);

    status = uct_query_resources(ucth, &resources, &num_resources);
    ASSERT_UCS_OK(status);

    for (unsigned i = 0; i < num_resources; ++i) {
        uct_iface_h iface = NULL;
        status = uct_iface_open(ucth, resources[i].tl_name, resources[i].hw_name,
                                &iface);
        ASSERT_TRUE(iface != NULL);
        ASSERT_UCS_OK(status);

        uct_iface_close(iface);
    }

    uct_release_resource_list(resources);

    uct_cleanup(ucth);
}

class entity {
public:
    entity() {
        ucs_status_t status;

        status = uct_init(&m_ucth);
        ASSERT_UCS_OK(status);

        status = uct_iface_open(m_ucth, "rc_mlx5", "mlx5_0:1", &m_iface);
        ASSERT_UCS_OK(status);

        status = uct_ep_create(m_iface, &m_ep);
        ASSERT_UCS_OK(status);
    }

    ~entity() {
        uct_ep_destroy(m_ep);
        uct_iface_close(m_iface);
        uct_cleanup(m_ucth);
    }

    void connect(const entity& other) {
        ucs_status_t status;

        uct_iface_attr_t iface_attr;
        status = uct_iface_query(other.m_iface, &iface_attr);
        ASSERT_UCS_OK(status);

        uct_iface_addr_t *iface_addr = (uct_iface_addr_t*)malloc(iface_attr.iface_addr_len);
        uct_ep_addr_t *ep_addr       = (uct_ep_addr_t*)malloc(iface_attr.ep_addr_len);

        status = uct_iface_get_address(other.m_iface, iface_addr);
        ASSERT_UCS_OK(status);

        status = uct_ep_get_address(other.m_ep, ep_addr);
        ASSERT_UCS_OK(status);

        status = uct_ep_connect_to_ep(m_ep, iface_addr, ep_addr);
        ASSERT_UCS_OK(status);

        free(ep_addr);
        free(iface_addr);
    }

    void mem_map(void *address, size_t length, uct_lkey_t *lkey_p,
                 uct_rkey_bundle_t *rkey_p) {
        ucs_status_t status;
        void *rkey_buffer;
        uct_pd_attr_t pd_attr;

        status = uct_mem_map(m_iface->pd, address, length, 0, lkey_p);
        ASSERT_UCS_OK(status);

        status = uct_pd_query(m_iface->pd, &pd_attr);
        ASSERT_UCS_OK(status);

        rkey_buffer = malloc(pd_attr.rkey_packed_size);
        ASSERT_TRUE(rkey_buffer != NULL);

        status = uct_rkey_pack(m_iface->pd, *lkey_p, rkey_buffer);
        ASSERT_UCS_OK(status);

        status = uct_rkey_unpack(m_ucth, rkey_buffer, rkey_p);
        ASSERT_UCS_OK(status);

        free(rkey_buffer);
    }

    void mem_unmap(uct_lkey_t lkey, uct_rkey_bundle_t *rkey) {
        ucs_status_t status;
        uct_rkey_release(m_ucth, rkey);
        status = uct_mem_unmap(m_iface->pd, lkey);
        ASSERT_UCS_OK(status);
    }

    void put8(uint64_t value, uint64_t address, uct_rkey_t rkey) {
        ucs_status_t status;
        status = uct_ep_put_short(m_ep, &value, sizeof(value), address, rkey,
                                  NULL, NULL);
        ASSERT_UCS_OK(status);
    }

    void flush() {
        ucs_status_t status;
        status = uct_iface_flush(m_iface, NULL, NULL);
        ASSERT_UCS_OK(status);
    }

    uct_context_h m_ucth;
    uct_iface_h m_iface;
    uct_ep_h m_ep;

};

UCS_TEST_F(test_uct, connect_ep) {

    const uint64_t magic = 0xdeadbeed1ee7a880;
    uct_rkey_bundle_t rkey;
    uct_lkey_t lkey;
    entity e1, e2;
    uint64_t val8;

    e2.mem_map(&val8, sizeof(val8), &lkey, &rkey);

    e1.connect(e2);
    e2.connect(e1);

    val8 = 0;
    e1.put8(magic, (uintptr_t)&val8, rkey.rkey);

    usleep(100000);

    e1.flush();

    EXPECT_EQ(magic, val8);

    e2.mem_unmap(lkey, &rkey);

}
