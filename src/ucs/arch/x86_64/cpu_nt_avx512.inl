/**
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

/* AVX-512 NT buffer-transfer implementation.  Defines the inline helpers and
 * masked-edge kernels used when AVX512BW is enabled: NT_DEST streams forward
 * destination stores and issues an sfence, while NT_SOURCE uses descending
 * regular stores without a fence.  The bridge macros at the end bind these
 * kernels to the generic dispatcher hooks in cpu.c. */

/* Tail edge length for the last (partial) 64B line of [dst, dst+len):
 * ((dst+len) & 63) mapped 0 -> 64 (full line), else the remainder in [1,64]. */
static UCS_F_ALWAYS_INLINE size_t
ucs_x86_avx512_nt_tail_edge_len(const void *dst, size_t len)
{
    const size_t rem = ((uintptr_t)dst + len) & 63u;
    return ((rem - 1u) & 63u) + 1u;
}

/* Forward body-start offset: dst to the next 64B-aligned line (head edge). */
static UCS_F_ALWAYS_INLINE size_t
ucs_x86_avx512_nt_body_off_fwd(const void *dst)
{
    const uintptr_t addr = (uintptr_t)dst;
    return (size_t)(ucs_align_down_pow2(addr + 64, 64) - addr);
}

/* Byte-masked head edge: copy [dst, dst+edge_len), edge_len in [1,64], onto
 * the 64B-aligned line containing dst (high edge_len lanes). */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx512_nt_masked_head_64(void *dst, const void *src, size_t edge_len)
{
    const size_t head_pad = 64u - edge_len;
    const __mmask64 k     = (__mmask64)(~0ull << head_pad);
    __m512i v;

    v = _mm512_maskz_loadu_epi8(k, UCS_PTR_BYTE_OFFSET(src, -head_pad));
    _mm512_mask_storeu_epi8(UCS_PTR_BYTE_OFFSET(dst, -head_pad), k, v);
}

/* Byte-masked tail edge: copy [dst+offset, dst+offset+edge_len), edge_len in
 * [1,64], onto the 64B-aligned base dst+offset (low edge_len lanes). */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx512_nt_masked_tail_64(void *dst, const void *src, size_t offset,
                                 size_t edge_len)
{
    const __mmask64 k = (__mmask64)(~0ull >> (64u - edge_len));
    __m512i v;

    v = _mm512_maskz_loadu_epi8(k, UCS_PTR_BYTE_OFFSET(src, offset));
    _mm512_mask_storeu_epi8(UCS_PTR_BYTE_OFFSET(dst, offset), k, v);
}

/* Vector copy helpers.  Ascending helpers store from the base [dst, dst+n);
 * descending (_bwd) helpers store from the top [dst-n, dst).  The
 * ucs_compiler_fence() between the four stores pins their order (they are
 * independent, so the compiler would otherwise reschedule them). */

/* 64B NT store, ascending. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx512_loadu_stream_64(void *dst, const void *src)
{
    __m512i z = _mm512_loadu_si512(src);
    _mm512_stream_si512((__m512i *)dst, z);
}

/* 256B NT store (4x), ascending. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx512_loadu_stream_256(void *dst, const void *src)
{
    const __m512i *sa = src;
    __m512i *da       = dst;
    __m512i z0        = _mm512_loadu_si512(sa);
    __m512i z1        = _mm512_loadu_si512(sa + 1);
    __m512i z2        = _mm512_loadu_si512(sa + 2);
    __m512i z3        = _mm512_loadu_si512(sa + 3);
    _mm512_stream_si512(da, z0);
    ucs_compiler_fence();
    _mm512_stream_si512(da + 1, z1);
    ucs_compiler_fence();
    _mm512_stream_si512(da + 2, z2);
    ucs_compiler_fence();
    _mm512_stream_si512(da + 3, z3);
}

/* 256B NT store (4x), ascending, aligned load. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx512_loada_stream_256(void *dst, const void *src)
{
    const __m512i *sa = src;
    __m512i *da       = dst;
    __m512i z0        = _mm512_load_si512(sa);
    __m512i z1        = _mm512_load_si512(sa + 1);
    __m512i z2        = _mm512_load_si512(sa + 2);
    __m512i z3        = _mm512_load_si512(sa + 3);
    _mm512_stream_si512(da, z0);
    ucs_compiler_fence();
    _mm512_stream_si512(da + 1, z1);
    ucs_compiler_fence();
    _mm512_stream_si512(da + 2, z2);
    ucs_compiler_fence();
    _mm512_stream_si512(da + 3, z3);
}

/* 64B store, descending. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx512_loadu_store_64_bwd(void *dst, const void *src)
{
    const __m512i *sa = src;
    __m512i *da       = dst;
    __m512i z         = _mm512_loadu_si512(sa - 1);
    _mm512_store_si512(da - 1, z);
}

/* 256B store (4x), descending. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx512_loadu_store_256_bwd(void *dst, const void *src)
{
    const __m512i *sa = src;
    __m512i *da       = dst;
    __m512i z0        = _mm512_loadu_si512(sa - 1);
    __m512i z1        = _mm512_loadu_si512(sa - 2);
    __m512i z2        = _mm512_loadu_si512(sa - 3);
    __m512i z3        = _mm512_loadu_si512(sa - 4);
    _mm512_store_si512(da - 1, z0);
    ucs_compiler_fence();
    _mm512_store_si512(da - 2, z1);
    ucs_compiler_fence();
    _mm512_store_si512(da - 3, z2);
    ucs_compiler_fence();
    _mm512_store_si512(da - 4, z3);
}

/* 256B store (4x), descending, aligned load. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_avx512_loada_store_256_bwd(void *dst, const void *src)
{
    const __m512i *sa = src;
    __m512i *da       = dst;
    __m512i z0        = _mm512_load_si512(sa - 1);
    __m512i z1        = _mm512_load_si512(sa - 2);
    __m512i z2        = _mm512_load_si512(sa - 3);
    __m512i z3        = _mm512_load_si512(sa - 4);
    _mm512_store_si512(da - 1, z0);
    ucs_compiler_fence();
    _mm512_store_si512(da - 2, z1);
    ucs_compiler_fence();
    _mm512_store_si512(da - 3, z2);
    ucs_compiler_fence();
    _mm512_store_si512(da - 4, z3);
}

/* AVX-512 NT_DEST copy-in: NT body stores + trailing sfence; ascending.  An
 * unconditional byte-masked head edge covers [dp, dp+offset) up to the next 64B
 * line so dp+offset is 64B-aligned for the body; the sub-cache-line tail
 * (1..63 B) is covered by a byte-masked store.  Precondition: len >= 64. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_nt_dst_avx512_buffer_transfer(void *dst, const void *src, size_t len)
{
    char *dp       = dst;
    const char *sp = src;
    size_t offset  = ucs_x86_avx512_nt_body_off_fwd(dp);
    char *nt_store_ptr;
    int src_aligned;

    /* (1) unconditional byte-masked head edge [dp, dp+offset); aligned base,
     * offset==64 when dp is aligned (full first line). */
    ucs_x86_avx512_nt_masked_head_64(dp, sp, offset);

    /* (2) ascending 256B NT-stream body to last full 256B.  A dedicated
     * cursor keeps the hot NT stores in base+displacement form. */
    nt_store_ptr = UCS_PTR_BYTE_OFFSET(dp, offset);
    src_aligned  = ((UCS_PTR_BYTE_DIFF(dp, sp) & 63u) == 0u);
    if (ucs_unlikely(src_aligned)) {
        while (offset + 256 <= len) {
            ucs_x86_avx512_loada_stream_256(nt_store_ptr,
                                            UCS_PTR_BYTE_OFFSET(sp, offset));
            nt_store_ptr += 256;
            offset       += 256;
        }
    } else {
        while (offset + 256 <= len) {
            ucs_x86_avx512_loadu_stream_256(nt_store_ptr,
                                            UCS_PTR_BYTE_OFFSET(sp, offset));
            nt_store_ptr += 256;
            offset       += 256;
        }
    }

    /* (3) ascending 64B NT drain to the last full cache line. */
    while (offset + 64 <= len) {
        ucs_x86_avx512_loadu_stream_64(nt_store_ptr,
                                       UCS_PTR_BYTE_OFFSET(sp, offset));
        nt_store_ptr += 64;
        offset       += 64;
    }

    /* (4) tail masked edge [dp+offset, dp+len); dp+offset is 64B-aligned. */
    if (offset != len) {
        ucs_x86_avx512_nt_masked_tail_64(dp, sp, offset, len - offset);
    }

    /* make the streaming writes visible to the other core */
    ucs_memory_bus_store_fence();
}

/* AVX-512 NT_SOURCE copy-out: body stores, descending.  An unconditional
 * byte-masked tail edge covers the last (partial) 64B line so the descent
 * starts 64B-aligned; the sub-cache-line head (1..63 B) is covered by a
 * byte-masked store.  Precondition: len >= 64. */
static UCS_F_ALWAYS_INLINE void
ucs_x86_nt_src_avx512_buffer_transfer(void *dst, const void *src, size_t len)
{
    char *dp          = dst;
    const char *sp    = src;
    const size_t edge = ucs_x86_avx512_nt_tail_edge_len(dp, len);
    size_t offset     = len - edge; /* descent start; 64B-aligned */
    int src_aligned;

    /* (1) unconditional byte-masked tail edge [dp+len-edge, dp+len); aligned
     * base, edge==64 when the tail is aligned (full last line). */
    ucs_x86_avx512_nt_masked_tail_64(dp, sp, offset, edge);

    /* (2) descending 256B body to last full 256B. */
    src_aligned = ((UCS_PTR_BYTE_DIFF(dp, sp) & 63u) == 0u);
    if (src_aligned) {
        while (offset >= 256) {
            ucs_x86_avx512_loada_store_256_bwd(UCS_PTR_BYTE_OFFSET(dp, offset),
                                               UCS_PTR_BYTE_OFFSET(sp, offset));
            offset -= 256;
        }
    } else {
        while (offset >= 256) {
            ucs_x86_avx512_loadu_store_256_bwd(UCS_PTR_BYTE_OFFSET(dp, offset),
                                               UCS_PTR_BYTE_OFFSET(sp, offset));
            offset -= 256;
        }
    }

    /* (3) descending 64B drain to the last full cache line. */
    while (offset >= 64) {
        ucs_x86_avx512_loadu_store_64_bwd(UCS_PTR_BYTE_OFFSET(dp, offset),
                                          UCS_PTR_BYTE_OFFSET(sp, offset));
        offset -= 64;
    }

    /* (4) head masked edge [dp, dp+offset); offset == descent head
     * remainder. */
    if (offset != 0u) {
        ucs_x86_avx512_nt_masked_head_64(dp, sp, offset);
    }
}

#define ucs_x86_nt_dst_buffer_transfer ucs_x86_nt_dst_avx512_buffer_transfer
#define ucs_x86_nt_src_buffer_transfer ucs_x86_nt_src_avx512_buffer_transfer
