/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University
*               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#pragma once

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "libperf_int.h"

class ucp_perf_test_runner_base {
public:
    ucp_perf_test_runner_base(ucx_perf_context_t &perf) :
        m_perf(perf)
    {}

    virtual ~ucp_perf_test_runner_base() {}

    virtual ucs_status_t run() = 0;

    void request_wait(ucs_status_ptr_t request, ucs_memory_type_t mem_type,
                      const char *operation_name)
    {
        ucs_status_t status;

        if (UCS_PTR_IS_PTR(request)) {
            do {
                ucp_worker_progress(m_perf.ucp.worker);
                status = ucp_request_check_status(request);
            } while (status == UCS_INPROGRESS);
            ucp_request_free(request);
        } else {
            status = UCS_PTR_STATUS(request);
        }

        if (status != UCS_OK) {
            ucs_warn("failed to %s(memory_type=%s): %s", operation_name,
                     ucs_memory_type_names[mem_type], ucs_status_string(status));
        }
    }

protected:
    ucx_perf_context_t &m_perf;
};

template<typename PSN>
class ucp_perf_test_runner_base_psn : public ucp_perf_test_runner_base {
public:
    ucp_perf_test_runner_base_psn(ucx_perf_context_t &perf) :
        ucp_perf_test_runner_base(perf)
    {}

    UCS_F_ALWAYS_INLINE static void *sn_ptr(void *buffer, size_t length)
    {
        return UCS_PTR_BYTE_OFFSET(buffer, length - sizeof(PSN));
    }

    UCS_F_ALWAYS_INLINE static PSN
    get_sn(const volatile void *sn, ucs_memory_type_t mem_type,
           const ucx_perf_allocator_t *allocator)
    {
        if (ucs_likely(mem_type == UCS_MEMORY_TYPE_HOST)) {
            return *reinterpret_cast<const volatile PSN*>(sn);
        }

        PSN host_sn;
        allocator->memcpy(&host_sn, UCS_MEMORY_TYPE_HOST,
                          const_cast<const void*>(sn), mem_type, sizeof(PSN));
        return host_sn;
    }

    UCS_F_ALWAYS_INLINE PSN read_sn(void *buffer, size_t length)
    {
        ucs_memory_type_t mem_type = m_perf.params.recv_mem_type;
        const void *ptr            = sn_ptr(buffer, length);
        ucp_request_param_t param  = {0};
        ucs_status_ptr_t request;
        PSN sn;

        if (mem_type == UCS_MEMORY_TYPE_HOST) {
            return *(const volatile PSN*)ptr;
        } else {
            request = ucp_get_nbx(m_perf.ucp.self_ep, &sn, sizeof(sn),
                                  (uint64_t)ptr, m_perf.ucp.self_recv_rkey,
                                  &param);
            request_wait(request, mem_type, "read_sn");
            request = ucp_ep_flush_nbx(m_perf.ucp.self_ep, &param);
            request_wait(request, mem_type, "flush read_sn");
            return sn;
        }
    }

    UCS_F_ALWAYS_INLINE void write_sn(void *buffer, ucs_memory_type_t mem_type,
                                      size_t length, PSN sn, ucp_rkey_h rkey)
    {
        void *ptr                 = sn_ptr(buffer, length);
        ucp_request_param_t param = {0};
        ucs_status_ptr_t request;

        if (mem_type == UCS_MEMORY_TYPE_HOST) {
            *(volatile PSN*)ptr = sn;
        } else {
            request = ucp_put_nbx(m_perf.ucp.self_ep, &sn, sizeof(sn),
                                  (uint64_t)ptr, rkey, &param);
            request_wait(request, mem_type, "write_sn");
            request = ucp_ep_flush_nbx(m_perf.ucp.self_ep, &param);
            request_wait(request, mem_type, "flush write_sn");
        }
    }
};
