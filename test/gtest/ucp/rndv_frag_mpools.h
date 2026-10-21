/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TEST_RNDV_FRAG_MPOOLS_H_
#define UCP_TEST_RNDV_FRAG_MPOOLS_H_

#include <ucp/api/ucp.h>

extern "C" {
#include <ucp/core/ucp_mm.h>
#include <ucs/datastruct/mpool.h>
}

#include <vector>


/**
 * Controls the rendezvous fragment memory pools of one memory type on a
 * worker. Requires a pool to exist, i.e. a transfer which used fragments of
 * that memory type must have completed on the worker; the other pool of the
 * same device is created if needed.
 */
class rndv_frag_mpools {
public:
    rndv_frag_mpools(ucp_worker_h worker, ucs_memory_type_t mem_type);
    ~rndv_frag_mpools();

    /* Take every free fragment. Grows the pools up to their quota, unless
     * growing was made to fail. */
    void hold();

    /* Return the held fragments without waking up throttled requests */
    void release();

    /* Make the pools fail to grow with UCS_ERR_NO_MEMORY */
    void fail_alloc();
    void restore_alloc();

    /* Whether some pool can still grow, so that fail_alloc() takes effect
     * for the next allocation which finds no free fragment: a request which
     * finds the quota of a pool exhausted tries the next one */
    bool can_grow() const;

    size_t num_pools() const
    {
        return m_mpools.size();
    }

    size_t num_held() const
    {
        return m_held.size();
    }

    /* Number of failed growth attempts since the last reset */
    static unsigned num_alloc_failures()
    {
        return s_num_alloc_failures;
    }

    static void reset_alloc_failures()
    {
        s_num_alloc_failures = 0;
    }

private:
    static ucs_status_t fail_chunk_alloc(ucs_mpool_t *mp, size_t *size_p,
                                         void **chunk_p);

    static unsigned                     s_num_alloc_failures;
    ucp_worker_h                        m_worker;
    std::vector<ucs_mpool_t*>           m_mpools;
    std::vector<ucp_mem_desc_t*>        m_held;
    std::vector<const ucs_mpool_ops_t*> m_orig_ops;
    ucs_mpool_ops_t                     m_fail_ops;
};

#endif
