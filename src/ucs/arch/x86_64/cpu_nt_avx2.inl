/**
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

/* AVX2 NT buffer-transfer implementation.  Defines the inline helpers and
 * forward kernels used when AVX is enabled without AVX512BW: NT_DEST streams
 * destination stores and issues an sfence, while NT_SOURCE uses regular stores
 * without a fence.  The bridge macros at the end bind these kernels to the
 * generic dispatcher hooks in cpu.c. */

/* Copy a sub-cache-line remainder with overlapping stores.
 * len in [1,63]; dst is 64B-aligned; src may be unaligned. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx2_nt_tail_copy(void *dst, const void *src, size_t len)
{
    __m256i y0, y1;
    dst = __builtin_assume_aligned(dst, 64);
    if (len >= 32) {
        y0 = _mm256_loadu_si256(src);
        y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, len - 32));
        _mm256_store_si256(dst, y0);
        _mm256_storeu_si256((__m256i *)UCS_PTR_BYTE_OFFSET(dst, len - 32), y1);
    } else if (len >= 16) {
        *(uint64_t *)dst = *(uint64_t *)src;
        *(uint64_t *)UCS_PTR_BYTE_OFFSET(dst, 8) =
                *(uint64_t *)UCS_PTR_BYTE_OFFSET(src, 8);
        *(uint64_t *)UCS_PTR_BYTE_OFFSET(dst, len - 16) =
                *(uint64_t *)UCS_PTR_BYTE_OFFSET(src, len - 16);
        *(uint64_t *)UCS_PTR_BYTE_OFFSET(dst, len - 8) =
                *(uint64_t *)UCS_PTR_BYTE_OFFSET(src, len - 8);
    } else if (len >= 8) {
        *(uint64_t *)dst = *(uint64_t *)src;
        *(uint64_t *)UCS_PTR_BYTE_OFFSET(dst, len - 8) =
                *(uint64_t *)UCS_PTR_BYTE_OFFSET(src, len - 8);
    } else if (len >= 4) {
        *(uint32_t *)dst = *(uint32_t *)src;
        *(uint32_t *)UCS_PTR_BYTE_OFFSET(dst, len - 4) =
                *(uint32_t *)UCS_PTR_BYTE_OFFSET(src, len - 4);
    } else if (len >= 2) {
        *(uint16_t *)dst = *(uint16_t *)src;
        *(uint16_t *)UCS_PTR_BYTE_OFFSET(dst, len - 2) =
                *(uint16_t *)UCS_PTR_BYTE_OFFSET(src, len - 2);
    } else {
        *(uint8_t *)dst = *(uint8_t *)src;
    }
}

/* 32B store, unaligned dst (overlap edge). */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx2_loadu_storeu_32(void *dst, const void *src)
{
    __m256i y0 = _mm256_loadu_si256(src);
    _mm256_storeu_si256((__m256i *)dst, y0);
}

/* 32B store. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx2_loadu_store_32(void *dst, const void *src)
{
    __m256i y0 = _mm256_loadu_si256(src);
    _mm256_store_si256((__m256i *)dst, y0);
}

/* Prefix fill up to the next 64B cache-line boundary; returns the body-start
 * offset (dst + offset is 64B-aligned). */
static UCS_F_ALWAYS_INLINE size_t
ucs_x86_avx2_nt_prefix_to_line(void *dst, const void *src)
{
    const uintptr_t addr = (uintptr_t)dst;
    size_t prefix_offset;

    ucs_x86_avx2_loadu_storeu_32(dst, src);
    if ((addr & 63u) < 32u) {
        prefix_offset = (size_t)(ucs_align_down_pow2(addr + 32, 32) - addr);
        ucs_x86_avx2_loadu_store_32(UCS_PTR_BYTE_OFFSET(dst, prefix_offset),
                                    UCS_PTR_BYTE_OFFSET(src, prefix_offset));
    }

    return (size_t)(ucs_align_down_pow2(addr + 64, 64) - addr);
}

/* YMM copy helpers.  Except the _storeu_ edge helper, dst is 32B-aligned; src
 * is aligned only in the _loada_ variants.  _stream_ helpers issue NT
 * stores. */

/* 64B NT store (2x YMM). */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx2_loadu_stream_64(void *dst, const void *src)
{
    const __m256i *sa = src;
    __m256i *da       = dst;
    __m256i y0        = _mm256_loadu_si256(sa);
    __m256i y1        = _mm256_loadu_si256(sa + 1);
    _mm256_stream_si256(da, y0);
    _mm256_stream_si256(da + 1, y1);
}

/* 256B NT store (8x YMM). */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx2_loadu_stream_256(void *dst, const void *src)
{
    const __m256i *sa = src;
    __m256i *da       = dst;
    __m256i y0        = _mm256_loadu_si256(sa);
    __m256i y1        = _mm256_loadu_si256(sa + 1);
    __m256i y2        = _mm256_loadu_si256(sa + 2);
    __m256i y3        = _mm256_loadu_si256(sa + 3);
    __m256i y4        = _mm256_loadu_si256(sa + 4);
    __m256i y5        = _mm256_loadu_si256(sa + 5);
    __m256i y6        = _mm256_loadu_si256(sa + 6);
    __m256i y7        = _mm256_loadu_si256(sa + 7);
    _mm256_stream_si256(da, y0);
    _mm256_stream_si256(da + 1, y1);
    _mm256_stream_si256(da + 2, y2);
    _mm256_stream_si256(da + 3, y3);
    _mm256_stream_si256(da + 4, y4);
    _mm256_stream_si256(da + 5, y5);
    _mm256_stream_si256(da + 6, y6);
    _mm256_stream_si256(da + 7, y7);
}

/* 256B NT store (8x YMM), aligned load. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx2_loada_stream_256(void *dst, const void *src)
{
    const __m256i *sa = src;
    __m256i *da       = dst;
    __m256i y0        = _mm256_load_si256(sa);
    __m256i y1        = _mm256_load_si256(sa + 1);
    __m256i y2        = _mm256_load_si256(sa + 2);
    __m256i y3        = _mm256_load_si256(sa + 3);
    __m256i y4        = _mm256_load_si256(sa + 4);
    __m256i y5        = _mm256_load_si256(sa + 5);
    __m256i y6        = _mm256_load_si256(sa + 6);
    __m256i y7        = _mm256_load_si256(sa + 7);
    _mm256_stream_si256(da, y0);
    _mm256_stream_si256(da + 1, y1);
    _mm256_stream_si256(da + 2, y2);
    _mm256_stream_si256(da + 3, y3);
    _mm256_stream_si256(da + 4, y4);
    _mm256_stream_si256(da + 5, y5);
    _mm256_stream_si256(da + 6, y6);
    _mm256_stream_si256(da + 7, y7);
}

/* 64B store (2x YMM). */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx2_loadu_store_64(void *dst, const void *src)
{
    const __m256i *sa = src;
    __m256i *da       = dst;
    __m256i y0        = _mm256_loadu_si256(sa);
    __m256i y1        = _mm256_loadu_si256(sa + 1);
    _mm256_store_si256(da, y0);
    _mm256_store_si256(da + 1, y1);
}

/* 256B store (8x YMM). */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx2_loadu_store_256(void *dst, const void *src)
{
    const __m256i *sa = src;
    __m256i *da       = dst;
    __m256i y0        = _mm256_loadu_si256(sa);
    __m256i y1        = _mm256_loadu_si256(sa + 1);
    __m256i y2        = _mm256_loadu_si256(sa + 2);
    __m256i y3        = _mm256_loadu_si256(sa + 3);
    __m256i y4        = _mm256_loadu_si256(sa + 4);
    __m256i y5        = _mm256_loadu_si256(sa + 5);
    __m256i y6        = _mm256_loadu_si256(sa + 6);
    __m256i y7        = _mm256_loadu_si256(sa + 7);
    _mm256_store_si256(da, y0);
    _mm256_store_si256(da + 1, y1);
    _mm256_store_si256(da + 2, y2);
    _mm256_store_si256(da + 3, y3);
    _mm256_store_si256(da + 4, y4);
    _mm256_store_si256(da + 5, y5);
    _mm256_store_si256(da + 6, y6);
    _mm256_store_si256(da + 7, y7);
}

/* 256B store (8x YMM), aligned load. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx2_loada_store_256(void *dst, const void *src)
{
    const __m256i *sa = src;
    __m256i *da       = dst;
    __m256i y0        = _mm256_load_si256(sa);
    __m256i y1        = _mm256_load_si256(sa + 1);
    __m256i y2        = _mm256_load_si256(sa + 2);
    __m256i y3        = _mm256_load_si256(sa + 3);
    __m256i y4        = _mm256_load_si256(sa + 4);
    __m256i y5        = _mm256_load_si256(sa + 5);
    __m256i y6        = _mm256_load_si256(sa + 6);
    __m256i y7        = _mm256_load_si256(sa + 7);
    _mm256_store_si256(da, y0);
    _mm256_store_si256(da + 1, y1);
    _mm256_store_si256(da + 2, y2);
    _mm256_store_si256(da + 3, y3);
    _mm256_store_si256(da + 4, y4);
    _mm256_store_si256(da + 5, y5);
    _mm256_store_si256(da + 6, y6);
    _mm256_store_si256(da + 7, y7);
}

/* AVX2 NT_DEST copy-in: NT body stores + trailing sfence; ascending.  A
 * prefix first fills [dp, dp+offset) up to the next 64B line so dp+offset is
 * 64B-aligned for the body; the sub-cache-line tail (1..63 B) is copied by
 * an overlapping-store if-chain.  Precondition: len >= 64. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_nt_dst_avx2_buffer_transfer(void *dst, const void *src, size_t len)
{
    char *dp       = dst;
    const char *sp = src;
    size_t offset;
    int src_aligned;

    /* (1) copy prefix to the next 64B cache-line boundary. */
    offset = ucs_x86_avx2_nt_prefix_to_line(dp, sp);

    /* (2) ascending 256B NT-stream body to last full 256B. */
    src_aligned = ((UCS_PTR_BYTE_DIFF(dp, sp) & 31u) == 0u);
    if (ucs_unlikely(src_aligned)) {
        while (offset + 256 <= len) {
            ucs_x86_avx2_loada_stream_256(UCS_PTR_BYTE_OFFSET(dp, offset),
                                          UCS_PTR_BYTE_OFFSET(sp, offset));
            offset += 256;
        }
    } else {
        while (offset + 256 <= len) {
            ucs_x86_avx2_loadu_stream_256(UCS_PTR_BYTE_OFFSET(dp, offset),
                                          UCS_PTR_BYTE_OFFSET(sp, offset));
            offset += 256;
        }
    }

    /* (3) ascending 64B NT drain to the last full cache line. */
    while (offset + 64 <= len) {
        ucs_x86_avx2_loadu_stream_64(UCS_PTR_BYTE_OFFSET(dp, offset),
                                     UCS_PTR_BYTE_OFFSET(sp, offset));
        offset += 64;
    }

    /* (4) if [1,63] remainder copy it.  dp+offset is 64B-aligned. */
    if (offset != len) {
        ucs_x86_avx2_nt_tail_copy(UCS_PTR_BYTE_OFFSET(dp, offset),
                                  UCS_PTR_BYTE_OFFSET(sp, offset),
                                  len - offset);
    }

    /* make the streaming writes visible to the other core */
    ucs_memory_bus_store_fence();
}

/* AVX2 NT_SOURCE copy-out: body stores, ascending.  A prefix
 * first fills [dp, dp+offset) up to the next 64B line so dp+offset is
 * 64B-aligned for the body; the sub-cache-line tail (1..63 B) is copied by
 * an overlapping-store if-chain.  Precondition: len >= 64. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_nt_src_avx2_buffer_transfer(void *dst, const void *src, size_t len)
{
    char *dp       = dst;
    const char *sp = src;
    size_t offset;
    int src_aligned;

    /* (1) copy prefix to the next 64B cache-line boundary. */
    offset = ucs_x86_avx2_nt_prefix_to_line(dp, sp);

    /* (2) ascending 256B body to the last full 256B. */
    src_aligned = ((UCS_PTR_BYTE_DIFF(dp, sp) & 31u) == 0u);
    if (src_aligned) {
        while (offset + 256 <= len) {
            ucs_x86_avx2_loada_store_256(UCS_PTR_BYTE_OFFSET(dp, offset),
                                         UCS_PTR_BYTE_OFFSET(sp, offset));
            offset += 256;
        }
    } else {
        while (offset + 256 <= len) {
            ucs_x86_avx2_loadu_store_256(UCS_PTR_BYTE_OFFSET(dp, offset),
                                         UCS_PTR_BYTE_OFFSET(sp, offset));
            offset += 256;
        }
    }

    /* (3) ascending 64B drain to the last full cache line. */
    while (offset + 64 <= len) {
        ucs_x86_avx2_loadu_store_64(UCS_PTR_BYTE_OFFSET(dp, offset),
                                    UCS_PTR_BYTE_OFFSET(sp, offset));
        offset += 64;
    }

    /* (4) if [1,63] remainder copy it.  dp+offset is 64B-aligned. */
    if (offset != len) {
        ucs_x86_avx2_nt_tail_copy(UCS_PTR_BYTE_OFFSET(dp, offset),
                                  UCS_PTR_BYTE_OFFSET(sp, offset),
                                  len - offset);
    }
}

#define ucs_x86_nt_dst_buffer_transfer ucs_x86_nt_dst_avx2_buffer_transfer
#define ucs_x86_nt_src_buffer_transfer ucs_x86_nt_src_avx2_buffer_transfer
