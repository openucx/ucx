/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "mem_buffer.h"

#include <sys/types.h>
#include <ucp/core/ucp_mm.h>
#include <ucs/debug/assert.h>
#include <common/test_helpers.h>

#if HAVE_CUDA
#  include <cuda.h>
#  include <cuda_runtime.h>

#define CUDA_CALL(_code, _details) \
    do { \
        cudaError_t cerr = _code; \
        if (cerr != cudaSuccess) { \
            UCS_TEST_ABORT(#_code << " failed: " \
                    << cudaGetErrorString(cerr) \
                    << _details); \
        } \
    } while (0)

#endif

#if HAVE_ROCM
#  include <hip_runtime.h>

#define ROCM_CALL(_code) \
    do { \
        hipError_t cerr = _code; \
        if (cerr != hipSuccess) { \
            UCS_TEST_ABORT(# _code << " failed"); \
        } \
    } while (0)

#endif


bool mem_buffer::is_cuda_supported()
{
#if HAVE_CUDA
    int num_gpus        = 0;
    cudaError_t cudaErr = cudaGetDeviceCount(&num_gpus);
    return (cudaErr == cudaSuccess) && (num_gpus > 0);
#else
    return false;
#endif
}

bool mem_buffer::is_rocm_supported()
{
#if HAVE_ROCM
    int num_gpus;
    hipError_t hipErr = hipGetDeviceCount(&num_gpus);
    return (hipErr == hipSuccess) && (num_gpus > 0);
#else
    return false;
#endif
}

bool mem_buffer::is_gpu_supported()
{
    return is_cuda_supported() || is_rocm_supported();
}

bool mem_buffer::is_rocm_managed_supported()
{
#if HAVE_ROCM
    hipError_t ret;
    void *dptr;
    hipPointerAttribute_t attr;

    ret = hipMallocManaged(&dptr, 64);
    if (ret != hipSuccess) {
        return false;
    }

    ret = hipPointerGetAttributes(&attr, dptr);
    if (ret != hipSuccess) {
        return false;
    }

    hipFree(dptr);
    return attr.memoryType == hipMemoryTypeUnified;
#else
    return false;
#endif
}

const std::vector<ucs_memory_type_t>&  mem_buffer::supported_mem_types()
{
    static std::vector<ucs_memory_type_t> vec;

    if (vec.empty()) {
        vec.push_back(UCS_MEMORY_TYPE_HOST);
        if (is_cuda_supported()) {
            vec.push_back(UCS_MEMORY_TYPE_CUDA);
            vec.push_back(UCS_MEMORY_TYPE_CUDA_MANAGED);
        }
        if (is_rocm_supported()) {
            vec.push_back(UCS_MEMORY_TYPE_ROCM);
        }
        if (is_rocm_managed_supported()) {
            vec.push_back(UCS_MEMORY_TYPE_ROCM_MANAGED);
        }
    }

    return vec;
}

void mem_buffer::set_device_context()
{
    static __thread bool device_set = false;

    if (device_set) {
        return;
    }

#if HAVE_CUDA
    if (is_cuda_supported()) {
        cudaSetDevice(0);
        /* need to call free as context maybe lazily initialized when calling
         * cudaSetDevice(0) but calling cudaFree(0) should guarantee context
         * creation upon return */
        cudaFree(0);
    }
#endif

#if HAVE_ROCM
    if (is_rocm_supported()) {
        hipSetDevice(0);
    }
#endif

    device_set = true;
}

void *mem_buffer::allocate(size_t size, ucs_memory_type_t mem_type)
{
    void *ptr;

    if (size == 0) {
        return NULL;
    }

    switch (mem_type) {
    case UCS_MEMORY_TYPE_HOST:
        ptr = malloc(size);
        if (ptr == NULL) {
            UCS_TEST_ABORT("malloc(size=" << size << ") failed");
        }
        return ptr;
#if HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
        CUDA_CALL(cudaMalloc(&ptr, size), ": size=" << size);
        return ptr;
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        CUDA_CALL(cudaMallocManaged(&ptr, size), ": size=" << size);
        return ptr;
#endif
#if HAVE_ROCM
    case UCS_MEMORY_TYPE_ROCM:
        ROCM_CALL(hipMalloc(&ptr, size));
        return ptr;
    case UCS_MEMORY_TYPE_ROCM_MANAGED:
        ROCM_CALL(hipMallocManaged(&ptr, size));
        return ptr;
#endif
    default:
        UCS_TEST_SKIP_R(std::string(ucs_memory_type_names[mem_type]) +
                        " memory is not supported");
    }
}

void mem_buffer::release(void *ptr, ucs_memory_type_t mem_type)
{
    try {
        switch (mem_type) {
        case UCS_MEMORY_TYPE_HOST:
            free(ptr);
            break;
#if HAVE_CUDA
        case UCS_MEMORY_TYPE_CUDA:
        case UCS_MEMORY_TYPE_CUDA_MANAGED:
            CUDA_CALL(cudaFree(ptr), ": ptr=" << ptr);
            break;
#endif
#if HAVE_ROCM
        case UCS_MEMORY_TYPE_ROCM:
        case UCS_MEMORY_TYPE_ROCM_MANAGED:
            ROCM_CALL(hipFree(ptr));
            break;
#endif
        default:
            break;
        }
    } catch (const std::exception &e) {
        UCS_TEST_MESSAGE << "got \"" << e.what() << "\" exception when"
                << " destroying memory "
                << mem_type_name(mem_type) << " buffer";
    }
}

void mem_buffer::pattern_fill(void *buffer, size_t length, uint64_t seed)
{
    size_t word_length = ucs_align_down_pow2(length, sizeof(uint64_t));
    uint64_t *end      = (uint64_t*)UCS_PTR_BYTE_OFFSET(buffer, word_length);

    for (uint64_t *ptr = (uint64_t*)buffer; ptr < end; ++ptr) {
        *ptr = seed;
        seed = pat(seed);
    }

    memcpy(end, &seed, length - UCS_PTR_BYTE_DIFF(buffer, end));
}

void mem_buffer::pattern_check_failed(uint64_t expected, uint64_t actual,
                                      size_t length, uint64_t mask,
                                      size_t offset, const void *orig_ptr)
{

    std::stringstream ss;
    ss << "Pattern check failed at " << UCS_PTR_BYTE_OFFSET(orig_ptr, offset)
       << " offset " << offset;

    ucs_assert(length <= sizeof(actual));
    if (length != sizeof(actual)) {
        // If mask is partial, print it as well
        ss << " (length " << length << " mask: 0x" << std::hex << mask << ")";
    }

    ss << ": Expected: 0x" << std::hex << (expected & mask) << " Actual: 0x"
       << std::hex << actual << std::dec;

    UCS_TEST_ABORT(ss.str());
}

void mem_buffer::pattern_check(const void *buffer, size_t length, uint64_t seed,
                               const void *orig_ptr)
{
    const char *end     = (const char*)buffer + length;
    const uint64_t *ptr = (const uint64_t*)buffer;

    if (orig_ptr == NULL) {
        orig_ptr = buffer;
    }

    while ((const char*)(ptr + 1) <= end) {
        pattern_check(seed, *ptr, sizeof(*ptr), UCS_PTR_BYTE_DIFF(buffer, ptr),
                      buffer, orig_ptr);
        seed = pat(seed);
        ++ptr;
    }

    size_t remainder = (end - (const char*)ptr);
    if (remainder > 0) {
        ucs_assert(remainder < sizeof(*ptr));
        uint64_t value = 0;
        memcpy(&value, ptr, remainder);
        pattern_check(seed, value, remainder, UCS_PTR_BYTE_DIFF(buffer, ptr),
                      buffer, orig_ptr);
    }
}

void mem_buffer::pattern_check(const void *buffer, size_t length,
                               const void *orig_ptr)
{
    if (length > sizeof(uint64_t)) {
        pattern_check(buffer, length, *(const uint64_t*)buffer, orig_ptr);
    }
}

void mem_buffer::pattern_fill(void *buffer, size_t length, uint64_t seed,
                              ucs_memory_type_t mem_type)
{
    if (UCP_MEM_IS_HOST(mem_type)) {
        pattern_fill(buffer, length, seed);
    } else {
        ucs::auto_buffer temp(length);
        pattern_fill(*temp, length, seed);
        copy_to(buffer, *temp, length, mem_type);
    }
}

void mem_buffer::pattern_check(const void *buffer, size_t length, uint64_t seed,
                               ucs_memory_type_t mem_type)
{
    if (UCP_MEM_IS_HOST(mem_type)) {
        pattern_check(buffer, length, seed, buffer);
    } else {
        ucs::auto_buffer temp(length);
        copy_from(*temp, buffer, length, mem_type);
        pattern_check(*temp, length, seed, buffer);
    }
}

void mem_buffer::memset(void *buffer, size_t length, int c,
                        ucs_memory_type_t mem_type)
{
    switch (mem_type) {
    case UCS_MEMORY_TYPE_HOST:
    case UCS_MEMORY_TYPE_ROCM_MANAGED:
        ::memset(buffer, c, length);
        break;
#if HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        CUDA_CALL(cudaMemset(buffer, c, length),
                  ": ptr=" << buffer << " value=" << c << " count=" << length);
        CUDA_CALL(cudaDeviceSynchronize(), "");
        break;
#endif
#if HAVE_ROCM
    case UCS_MEMORY_TYPE_ROCM:
        ROCM_CALL(hipMemset(buffer, c, length));
        ROCM_CALL(hipDeviceSynchronize());
        break;
#endif
    default:
        UCS_TEST_ABORT("Wrong buffer memory type " + mem_type_name(mem_type));
    }
}

void mem_buffer::copy_to(void *dst, const void *src, size_t length,
                         ucs_memory_type_t dst_mem_type)
{
    copy_between(dst, src, length, dst_mem_type, UCS_MEMORY_TYPE_HOST);
}

void mem_buffer::copy_from(void *dst, const void *src, size_t length,
                           ucs_memory_type_t src_mem_type)
{
    copy_between(dst, src, length, UCS_MEMORY_TYPE_HOST, src_mem_type);
}

/* check both mem types are in the given set */
bool mem_buffer::check_mem_types(ucs_memory_type_t dst_mem_type,
                                 ucs_memory_type_t src_mem_type,
                                 const uint64_t mem_types)
{
    return (UCS_BIT(dst_mem_type) & mem_types) &&
           (UCS_BIT(src_mem_type) & mem_types);
}

void mem_buffer::copy_between(void *dst, const void *src, size_t length,
                              ucs_memory_type_t dst_mem_type,
                              ucs_memory_type_t src_mem_type)
{
    const uint64_t host_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
#if HAVE_CUDA
    const uint64_t cuda_mem_types = host_mem_types |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA) |
                                    UCS_BIT(UCS_MEMORY_TYPE_CUDA_MANAGED);
#endif
#if HAVE_ROCM
    const uint64_t rocm_mem_types = host_mem_types |
                                    UCS_BIT(UCS_MEMORY_TYPE_ROCM) |
                                    UCS_BIT(UCS_MEMORY_TYPE_ROCM_MANAGED);
#endif

    if (check_mem_types(dst_mem_type, src_mem_type, host_mem_types)) {
        memcpy(dst, src, length);
#if HAVE_CUDA
    } else if (check_mem_types(dst_mem_type, src_mem_type, cuda_mem_types)) {
        CUDA_CALL(cudaMemcpy(dst, src, length, cudaMemcpyDefault),
                  ": dst=" << dst << " src=" << src << " length=" << length);
        CUDA_CALL(cudaDeviceSynchronize(), "");
#endif
#if HAVE_ROCM
    } else if (check_mem_types(dst_mem_type, src_mem_type, rocm_mem_types)) {
        ROCM_CALL(hipMemcpy(dst, src, length, hipMemcpyDefault));
        ROCM_CALL(hipDeviceSynchronize());
#endif
    } else {
        UCS_TEST_ABORT("Wrong buffer memory type pair " +
                       mem_type_name(src_mem_type) + "/" +
                       mem_type_name(dst_mem_type));
    }
}

bool mem_buffer::compare(const void *expected, const void *buffer,
                         size_t length, ucs_memory_type_t mem_type)
{
    /* don't access managed memory from CPU to avoid moving the pages
     * from GPU to CPU during the test
     */
    if ((mem_type == UCS_MEMORY_TYPE_HOST) ||
        (mem_type == UCS_MEMORY_TYPE_ROCM_MANAGED)) {
        return memcmp(expected, buffer, length) == 0;
    } else {
        ucs::auto_buffer temp(length);
        copy_from(*temp, buffer, length, mem_type);
        return memcmp(expected, *temp, length) == 0;
    }
}

bool mem_buffer::compare(const void *expected, const void *buffer,
                         size_t length, ucs_memory_type_t mem_type_expected,
                         ucs_memory_type_t mem_type_buffer)
{
    ucs::handle<void*> expected_copy, buffer_copy;
    const void *expected_host, *buffer_host;

    if (UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type_expected)) {
        expected_host = expected;
    } else {
        expected_copy.reset(malloc(length), free);
        copy_from(expected_copy.get(), expected, length, mem_type_expected);
        expected_host = expected_copy.get();
    }

    if (UCP_MEM_IS_ACCESSIBLE_FROM_CPU(mem_type_buffer)) {
        buffer_host = buffer;
    } else {
        buffer_copy.reset(malloc(length), free);
        copy_from(buffer_copy.get(), buffer, length, mem_type_buffer);
        buffer_host = buffer_copy.get();
    }

    return memcmp(expected_host, buffer_host, length) == 0;
}

std::string mem_buffer::mem_type_name(ucs_memory_type_t mem_type)
{
    return ucs_memory_type_names[mem_type];
}

mem_buffer::mem_buffer(size_t size, ucs_memory_type_t mem_type) :
    m_mem_type(mem_type), m_ptr(allocate(size, mem_type)), m_size(size) {
}

mem_buffer::mem_buffer(size_t size, ucs_memory_type_t mem_type, uint64_t seed) :
    m_mem_type(mem_type), m_ptr(allocate(size, mem_type)), m_size(size) {
    pattern_fill(seed);
}

mem_buffer::~mem_buffer() {
    release(ptr(), mem_type());
}

ucs_memory_type_t mem_buffer::mem_type() const {
    return m_mem_type;
}

void *mem_buffer::ptr() const {
    return m_ptr;
}

size_t mem_buffer::size() const {
    return m_size;
}

void mem_buffer::pattern_fill(uint64_t seed, size_t length) {
    pattern_fill(ptr(), std::min(length, size()), seed, mem_type());
}

void mem_buffer::pattern_check(uint64_t seed, size_t length) const {
    pattern_check(ptr(), std::min(length, size()), seed, mem_type());
}

void mem_buffer::memset(int c) {
    memset(ptr(), size(), c, mem_type());
}
