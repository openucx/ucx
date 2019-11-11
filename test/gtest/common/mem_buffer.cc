/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "mem_buffer.h"

#include <ucs/debug/assert.h>
#include <common/test_helpers.h>

#if HAVE_CUDA
#  include <cuda.h>
#  include <cuda_runtime.h>

#define CUDA_CALL(_code) \
    do { \
        cudaError_t cerr = _code; \
        if (cerr != cudaSuccess) { \
            UCS_TEST_ABORT(# _code << " failed"); \
        } \
    } while (0)

#endif


std::vector<ucs_memory_type_t> mem_buffer::supported_mem_types()
{
    std::vector<ucs_memory_type_t> vec;
    vec.push_back(UCS_MEMORY_TYPE_HOST);
#if HAVE_CUDA
    vec.push_back(UCS_MEMORY_TYPE_CUDA);
    vec.push_back(UCS_MEMORY_TYPE_CUDA_MANAGED);
#endif
    return vec;
}

void *mem_buffer::allocate(size_t size, ucs_memory_type_t mem_type)
{
    void *ptr;

    switch (mem_type) {
    case UCS_MEMORY_TYPE_HOST:
        ptr = malloc(size);
        if (ptr == NULL) {
            UCS_TEST_ABORT("malloc() failed");
        }
        return ptr;
#if HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
        CUDA_CALL(cudaMalloc(&ptr, size));
        return ptr;
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        CUDA_CALL(cudaMallocManaged(&ptr, size));
        return ptr;
#endif
    default:
        UCS_TEST_SKIP_R(std::string(ucs_memory_type_names[mem_type]) +
                        " memory is not supported");
    }
}

void mem_buffer::release(void *ptr, ucs_memory_type_t mem_type)
{
    switch (mem_type) {
    case UCS_MEMORY_TYPE_HOST:
        free(ptr);
        break;
#if HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        CUDA_CALL(cudaFree(ptr));
        break;
#endif
    default:
        break;
    }
}

void mem_buffer::pattern_fill(void *buffer, size_t length, uint64_t seed)
{
    uint64_t *ptr = (uint64_t*)buffer;
    char *end = (char *)buffer + length;

    while ((char*)(ptr + 1) <= end) {
        *ptr = seed;
        seed = pat(seed);
        ++ptr;
    }
    memcpy(ptr, &seed, end - (char*)ptr);
}

void mem_buffer::pattern_check(const void *buffer, size_t length, uint64_t seed)
{
    const char* end = (const char*)buffer + length;
    const uint64_t *ptr = (const uint64_t*)buffer;

    while ((const char*)(ptr + 1) <= end) {
       if (*ptr != seed) {
            UCS_TEST_ABORT("At offset " << ((const char*)ptr - (const char*)buffer) << ": " <<
                           "Expected: 0x" << std::hex << seed << " " <<
                           "Got: 0x" << std::hex << (*ptr) << std::dec);
        }
        seed = pat(seed);
        ++ptr;
    }

    size_t remainder = (end - (const char*)ptr);
    if (remainder > 0) {
        ucs_assert(remainder < sizeof(*ptr));
        uint64_t mask = UCS_MASK_SAFE(remainder * 8 * sizeof(char));
        uint64_t value = 0;
        memcpy(&value, ptr, remainder);
        if (value != (seed & mask)) {
             UCS_TEST_ABORT("At offset " << ((const char*)ptr - (const char*)buffer) <<
                            " (remainder " << remainder << ") : " <<
                            "Expected: 0x" << std::hex << (seed & mask) << " " <<
                            "Mask: 0x" << std::hex << mask << " " <<
                            "Got: 0x" << std::hex << value << std::dec);
         }
    }
}

void mem_buffer::pattern_check(const void *buffer, size_t length)
{
    if (length > sizeof(uint64_t)) {
        pattern_check(buffer, length, *(const uint64_t*)buffer);
    }
}

void mem_buffer::pattern_fill(void *buffer, size_t length, uint64_t seed,
                              ucs_memory_type_t mem_type)
{
    if ((mem_type == UCS_MEMORY_TYPE_HOST) ||
        (mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED)) {
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
    if ((mem_type == UCS_MEMORY_TYPE_HOST) ||
        (mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED)) {
        pattern_check(buffer, length, seed);
    } else {
        ucs::auto_buffer temp(length);
        copy_from(*temp, buffer, length, mem_type);
        pattern_check(*temp, length, seed);
    }
}

void mem_buffer::copy_to(void *dst, const void *src, size_t length,
                         ucs_memory_type_t dst_mem_type)
{
    switch (dst_mem_type) {
    case UCS_MEMORY_TYPE_HOST:
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        memcpy(dst, src, length);
        break;
#if HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
        CUDA_CALL(cudaMemcpy(dst, src, length, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaDeviceSynchronize());
        break;
#endif
    default:
        abort_wrong_mem_type(dst_mem_type);
    }
}

void mem_buffer::copy_from(void *dst, const void *src, size_t length,
                           ucs_memory_type_t src_mem_type)
{
    switch (src_mem_type) {
    case UCS_MEMORY_TYPE_HOST:
    case UCS_MEMORY_TYPE_CUDA_MANAGED:
        memcpy(dst, src, length);
        break;
#if HAVE_CUDA
    case UCS_MEMORY_TYPE_CUDA:
        CUDA_CALL(cudaMemcpy(dst, src, length, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());
        break;
#endif
    default:
        abort_wrong_mem_type(src_mem_type);
    }
}

bool mem_buffer::compare(const void *expected, const void *buffer,
                         size_t length, ucs_memory_type_t mem_type)
{
    if ((mem_type == UCS_MEMORY_TYPE_HOST) ||
        (mem_type == UCS_MEMORY_TYPE_CUDA_MANAGED)) {
        return memcmp(expected, buffer, length) == 0;
    } else {
        ucs::auto_buffer temp(length);
        copy_from(*temp, buffer, length, mem_type);
        return memcmp(expected, *temp, length) == 0;
    }
}

std::string mem_buffer::mem_type_name(ucs_memory_type_t mem_type)
{
    if (mem_type < UCS_MEMORY_TYPE_LAST) {
        return ucs_memory_type_names[mem_type];
    } else {
        return "unknown";
    }
}

void mem_buffer::abort_wrong_mem_type(ucs_memory_type_t mem_type) {
    UCS_TEST_ABORT("Wrong buffer memory type " + mem_type_name(mem_type));
}

uint64_t mem_buffer::pat(uint64_t prev) {
    /* LFSR pattern */
    static const uint64_t polynom = 1337;
    return (prev << 1) | (__builtin_parityl(prev & polynom) & 1);
}

mem_buffer::mem_buffer(size_t size, ucs_memory_type_t mem_type) :
    m_mem_type(mem_type), m_ptr(allocate(size, mem_type)), m_size(size) {
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
