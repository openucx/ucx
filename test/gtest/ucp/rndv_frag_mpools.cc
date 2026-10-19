/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "rndv_frag_mpools.h"

extern "C" {
#include <ucp/core/ucp_worker.h>
#include <ucp/rndv/rndv.h>
#include <ucs/datastruct/mpool.inl>
}


unsigned rndv_frag_mpools::s_num_alloc_failures = 0;

rndv_frag_mpools::rndv_frag_mpools(ucp_worker_h worker,
                                   ucs_memory_type_t mem_type) :
    m_worker(worker)
{
    std::vector<ucs_sys_device_t> sys_devs;
    ucp_mem_desc_t *mdesc;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    for (khiter_t iter = kh_begin(&worker->mpool_hash);
         iter != kh_end(&worker->mpool_hash); ++iter) {
        if (kh_exist(&worker->mpool_hash, iter) &&
            (kh_key(&worker->mpool_hash, iter).mem_type == mem_type)) {
            sys_devs.push_back(kh_key(&worker->mpool_hash, iter).sys_dev);
        }
    }

    /* A pool is created by the first allocation from it, so the other pool
     * of the same device may not exist yet */
    for (ucs_sys_device_t sys_dev : sys_devs) {
        for (unsigned pool : {UCP_WORKER_RNDV_FRAG_POOL_SHARED,
                              UCP_WORKER_RNDV_FRAG_POOL_RESERVED}) {
            if (ucp_rndv_mpool_get(worker, mem_type, sys_dev, pool, &mdesc) ==
                UCS_OK) {
                ucs_mpool_put_inline(mdesc);
            }
        }
    }

    for (khiter_t iter = kh_begin(&worker->mpool_hash);
         iter != kh_end(&worker->mpool_hash); ++iter) {
        if (kh_exist(&worker->mpool_hash, iter) &&
            (kh_key(&worker->mpool_hash, iter).mem_type == mem_type)) {
            m_mpools.push_back(&kh_val(&worker->mpool_hash, iter));
        }
    }

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
}

rndv_frag_mpools::~rndv_frag_mpools()
{
    restore_alloc();
    release();
}

void rndv_frag_mpools::hold()
{
    ucp_mem_desc_t *mdesc;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(m_worker);

    for (ucs_mpool_t *mpool : m_mpools) {
        /* Terminates only because the mpool quota is finite, which requires
         * proto v2 */
        while ((mdesc = static_cast<ucp_mem_desc_t*>(
                        ucs_mpool_get_inline(mpool))) != nullptr) {
            m_held.push_back(mdesc);
        }
    }

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(m_worker);
}

void rndv_frag_mpools::release()
{
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(m_worker);

    for (ucp_mem_desc_t *mdesc : m_held) {
        /* Return the fragment without the flow-control wakeup: no request
         * may be left waiting for it when the test completes */
        ucs_mpool_put_inline(mdesc);
    }

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(m_worker);

    m_held.clear();
}

ucs_status_t rndv_frag_mpools::fail_chunk_alloc(ucs_mpool_t *mp, size_t *size_p,
                                                void **chunk_p)
{
    ++s_num_alloc_failures;
    return UCS_ERR_NO_MEMORY;
}

void rndv_frag_mpools::fail_alloc()
{
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(m_worker);

    for (ucs_mpool_t *mpool : m_mpools) {
        m_orig_ops.push_back(mpool->data->ops);
        m_fail_ops             = *mpool->data->ops;
        m_fail_ops.chunk_alloc = fail_chunk_alloc;
        mpool->data->ops       = &m_fail_ops;
    }

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(m_worker);
}

void rndv_frag_mpools::restore_alloc()
{
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(m_worker);

    for (size_t i = 0; i < m_orig_ops.size(); ++i) {
        m_mpools[i]->data->ops = m_orig_ops[i];
    }

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(m_worker);

    m_orig_ops.clear();
}

bool rndv_frag_mpools::can_grow() const
{
    bool can_grow = false;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(m_worker);

    for (ucs_mpool_t *mpool : m_mpools) {
        /* With no free fragment, empty means the quota is exhausted */
        can_grow = can_grow || !ucs_mpool_is_empty(mpool);
    }

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(m_worker);

    return can_grow;
}
