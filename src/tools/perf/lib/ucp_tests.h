/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TESTS_H
#define UCP_TESTS_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "libperf_int.h"

template<typename PSN>
class ucp_perf_test_runner_base {
public:
    ucp_perf_test_runner_base(ucx_perf_context_t &perf) :
        m_perf(perf)
    {}

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

    UCS_F_ALWAYS_INLINE static PSN *sn_ptr(void *buffer, size_t length)
    {
        return (PSN*)UCS_PTR_BYTE_OFFSET(buffer, length - sizeof(PSN));
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
        const PSN *ptr             = sn_ptr(buffer, length);
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
        PSN *ptr                  = sn_ptr(buffer, length);
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

    void fill_sn(void *buffer, ucp_perf_datatype_t datatype,
                 const ucx_perf_allocator_t *allocator, size_t length, PSN sn)
    {
        if (datatype == UCP_PERF_DATATYPE_IOV) {
            const ucp_dt_iov_t *iov = reinterpret_cast<const ucp_dt_iov_t*>(buffer);
            for (size_t i = 0; i < length; ++i) {
                fill_sn_region(iov[i].buffer, iov[i].length, allocator, sn);
            }
        } else {
            fill_sn_region(buffer, length, allocator, sn);
        }
    }

    ucs_status_t validate_sn(const void *buffer, ucp_perf_datatype_t datatype,
                             ucs_memory_type_t mem_type,
                             const ucx_perf_allocator_t *allocator,
                             size_t length, PSN sn, void *host_buffer)
    {
        if (datatype == UCP_PERF_DATATYPE_IOV) {
            const ucp_dt_iov_t *iov = reinterpret_cast<const ucp_dt_iov_t*>(buffer);
            for (size_t i = 0; i < length; ++i) {
                ucs_status_t status = validate_sn_region(iov[i].buffer,
                                                         mem_type, allocator,
                                                         iov[i].length, sn,
                                                         host_buffer);
                if (status != UCS_OK) {
                    ucs_error("data validation failed in iov element %zu", i);
                    return status;
                }
            }
            return UCS_OK;
        }

        return validate_sn_region(buffer, mem_type, allocator, length, sn,
                                  host_buffer);
    }

protected:
    ucx_perf_context_t &m_perf;

private:
    void fill_sn_region(void *buffer, size_t length, const ucx_perf_allocator_t *allocator,
                        PSN sn)
    {
        allocator->memset(buffer, sn, length);
    }

    ucs_status_t validate_sn_region(const void *buffer,
                                    ucs_memory_type_t mem_type,
                                    const ucx_perf_allocator_t *allocator,
                                    size_t length, PSN sn, void *host_buffer)
    {
        const uint8_t *data;

        if (mem_type == UCS_MEMORY_TYPE_HOST) {
            data = reinterpret_cast<const uint8_t*>(buffer);
        } else {
            allocator->memcpy(host_buffer, UCS_MEMORY_TYPE_HOST, buffer,
                              mem_type, length);
            data = reinterpret_cast<const uint8_t*>(host_buffer);
        }

        for (size_t i = 0; i < length; ++i) {
            if (data[i] != static_cast<uint8_t>(sn)) {
                ucs_error("data validation failed at offset %zu: "
                          "got 0x%x expected 0x%x",
                          i, data[i], static_cast<uint8_t>(sn));
                return UCS_ERR_IO_ERROR;
            }
        }

        return UCS_OK;
    }
};

#endif /* UCP_TESTS_H */
