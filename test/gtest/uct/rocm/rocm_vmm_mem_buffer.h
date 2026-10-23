/**
 * See file LICENSE for terms.
 */

#ifndef UCT_TEST_ROCM_VMM_MEM_BUFFER_H
#define UCT_TEST_ROCM_VMM_MEM_BUFFER_H

#include <common/test_helpers.h>
#include <hip/hip_runtime.h>
#include <vector>

extern "C" {
#include <ucs/sys/ptr_arith.h>
}

/* Device memory mapped with hipMemMap: one reserved virtual range backed by
 * @a num_chunks separate allocations, as frameworks that grow buffers in place
 * map them */
class rocm_vmm_mem_buffer {
public:
    rocm_vmm_mem_buffer(size_t size, size_t num_chunks = 1,
                        hipMemAllocationHandleType handle_type =
                                hipMemHandleTypeNone)
    {
        if (!alloc(size, num_chunks, handle_type)) {
            cleanup();
            UCS_TEST_SKIP_R("failed to allocate HIP VMM memory");
        }
    }

    ~rocm_vmm_mem_buffer()
    {
        cleanup();
    }

    void *ptr() const
    {
        return m_ptr;
    }

    size_t size() const
    {
        return m_size;
    }

    void *chunk(size_t index) const
    {
        return UCS_PTR_BYTE_OFFSET(m_ptr, index * m_chunk_size);
    }

    size_t chunk_size() const
    {
        return m_chunk_size;
    }

private:
    bool alloc(size_t size, size_t num_chunks,
               hipMemAllocationHandleType handle_type)
    {
        hipMemAllocationProp prop = {};
        hipMemAccessDesc access   = {};
        size_t granularity;
        int device;

        if (hipGetDevice(&device) != hipSuccess) {
            return false;
        }

        prop.type                 = hipMemAllocationTypePinned;
        prop.location.type        = hipMemLocationTypeDevice;
        prop.location.id          = device;
        prop.requestedHandleTypes = handle_type;
        if (hipMemGetAllocationGranularity(
                    &granularity, &prop, hipMemAllocationGranularityMinimum) !=
            hipSuccess) {
            return false;
        }

        m_chunk_size = ucs_align_up(size, granularity);
        if (hipMemAddressReserve(&m_ptr, m_chunk_size * num_chunks, 0, nullptr,
                                 0) != hipSuccess) {
            m_ptr = nullptr;
            return false;
        }

        m_size = m_chunk_size * num_chunks;
        for (size_t i = 0; i < num_chunks; ++i) {
            hipMemGenericAllocationHandle_t handle;

            if (hipMemCreate(&handle, m_chunk_size, &prop, 0) != hipSuccess) {
                return false;
            }

            if (hipMemMap(chunk(i), m_chunk_size, 0, handle, 0) != hipSuccess) {
                hipMemRelease(handle);
                return false;
            }

            m_handles.push_back(handle);
        }

        access.location = prop.location;
        access.flags    = hipMemAccessFlagsProtReadWrite;
        return hipMemSetAccess(m_ptr, m_size, &access, 1) == hipSuccess;
    }

    /* Also the unwind path of a partially completed alloc() */
    void cleanup()
    {
        for (size_t i = 0; i < m_handles.size(); ++i) {
            hipMemUnmap(chunk(i), m_chunk_size);
            hipMemRelease(m_handles[i]);
        }

        if (m_ptr != nullptr) {
            hipMemAddressFree(m_ptr, m_size);
        }

        m_handles.clear();
        m_ptr = nullptr;
    }

    std::vector<hipMemGenericAllocationHandle_t> m_handles;
    size_t m_chunk_size = 0;
    size_t m_size       = 0;
    void *m_ptr         = nullptr;
};

#endif
