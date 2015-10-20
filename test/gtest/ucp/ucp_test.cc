/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <ucs/gtest/test_helpers.h>
extern "C" {
#include <ucs/arch/atomic.h>
}


const ucs::ptr_vector<ucp_test::entity>& ucp_test::entities() const {
    return m_entities;
}

void ucp_test::cleanup() {
    m_entities.clear();
}

ucp_test::entity* ucp_test::create_entity() {
    entity *e = new entity(*this);
    m_entities.push_back(e);
    return e;
}

void ucp_test::get_params(ucp_params_t& params) const {
    params.features        = 0;
    params.request_size    = 0;
    params.request_init    = NULL;
    params.request_cleanup = NULL;
}

void ucp_test::progress() const {
    for (ucs::ptr_vector<entity>::const_iterator iter = entities().begin();
         iter != entities().end(); ++iter)
    {
        (*iter)->progress();
    }
}

void ucp_test::short_progress_loop() const {
    for (unsigned i = 0; i < 100; ++i) {
        progress();
        usleep(100);
    }
}

ucp_test::entity::entity(const ucp_test& test) : m_test(test), m_inprogress(0) {
    ucs::handle<ucp_config_t*> config;

    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);

    ucp_params_t params;
    test.get_params(params);

    UCS_TEST_CREATE_HANDLE(ucp_context_h, m_ucph, ucp_cleanup, ucp_init,
                           &params, config);

    UCS_TEST_CREATE_HANDLE(ucp_worker_h, m_worker, ucp_worker_destroy,
                           ucp_worker_create, m_ucph, UCS_THREAD_MODE_MULTI);

    ucp_worker_progress_register(m_worker, progress_cb, this);
}

ucp_test::entity::~entity() {
    ucp_worker_progress_unregister(m_worker, progress_cb, this);
}

void ucp_test::entity::connect(const ucp_test::entity* other) {
    ucs_status_t status;
    ucp_address_t *address;
    size_t address_length;

    status = ucp_worker_get_address(other->worker(), &address, &address_length);
    ASSERT_UCS_OK(status);

    ucp_ep_h ep;
    status = ucp_ep_create(m_worker, address, &ep);
    if (status == UCS_ERR_UNREACHABLE) {
        UCS_TEST_SKIP_R("could not find a valid transport");
    }

    ASSERT_UCS_OK(status);
    m_ep.reset(ep, ucp_ep_destroy);

    ucp_worker_release_address(other->worker(), address);
}

void ucp_test::entity::flush() const {
    ucs_status_t status = ucp_worker_flush(worker());
    ASSERT_UCS_OK(status);
}

void ucp_test::entity::disconnect() {
    m_ep.reset();
}

ucp_ep_h ucp_test::entity::ep() const {
    return m_ep;
}

ucp_worker_h ucp_test::entity::worker() const {
    return m_worker;
}

ucp_context_h ucp_test::entity::ucph() const {
    return m_ucph;
}

void ucp_test::entity::progress()
{
    if (ucs_atomic_cswap32(&m_inprogress, 0, 1) != 0) {
        return;
    }

    ucp_worker_progress(m_worker);
    m_inprogress = 0;
}

void ucp_test::entity::progress_cb(void *arg)
{
    entity *e = reinterpret_cast<entity*>(arg);
    e->progress_test();
}

void ucp_test::entity::progress_test()
{
    if (ucs_atomic_cswap32(&m_inprogress, 0, 1) != 0) {
        return;
    }
    m_test.progress();
    m_inprogress = 0;
}
