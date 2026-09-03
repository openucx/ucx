/**
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

static size_t ucs_x86_nt_all_buffer_transfer(void *dst, const void *src, size_t len)
{
    size_t offset;
    __m256i y0, y1, y2, y3, y4, y5, y6, y7;

    /* copy 64 bytes unconditionally */
    y0 = _mm256_loadu_si256(src);
    y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, 32));
    _mm256_storeu_si256(dst, y0);
    _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, 32), y1);

    offset = 64 - ((uintptr_t)dst & 0x1f);
    len   -= offset;

    if (ucs_likely((size_t)UCS_PTR_BYTE_OFFSET(src, offset) & 0x1f)) {
        /* src address is not aligned to 32 byte */
        while (len >= 256) {
            y4 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset));
            y5 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 32));
            y6 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 64));
            y7 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 96));
            y0 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 128));
            y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 160));
            y2 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 192));
            y3 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 224));
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset), y4);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 32), y5);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 64), y6);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 96), y7);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 128), y0);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 160), y1);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 192), y2);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 224), y3);

            if ((len > 1024) && (((offset >> 8) & 3) == 0)) {
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (8 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (9 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (10 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (11 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (12 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (13 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (14 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (15 * 64)));
            }

            offset += 256;
            len    -= 256;
        }
    } else {
        /* src address aligned to 32 byte */
        while (len >= 256) {
            y4 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset));
            y5 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 32));
            y6 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 64));
            y7 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 96));
            y0 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 128));
            y1 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 160));
            y2 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 192));
            y3 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 224));
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset), y4);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 32), y5);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 64), y6);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 96), y7);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 128), y0);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 160), y1);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 192), y2);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 224), y3);

            if ((len > 1024) && (((offset >> 8) & 3) == 0)) {
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (8 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (9 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (10 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (11 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (12 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (13 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (14 * 64)));
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (15 * 64)));
            }

            offset += 256;
            len    -= 256;
        }
    }

    while (len >= 64) {
        y4 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset));
        y5 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 32));
        _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset), y4);
        _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 32), y5);
        offset += 64;
        len    -= 64;
    }

    /* make the writes visible to the other core */
    ucs_memory_bus_store_fence();

    /* Handle the remaining bytes <= 63 */
    return len;
}

static UCS_F_ALWAYS_INLINE
size_t ucs_x86_nt_dst_buffer_transfer(void *dst, const void *src, size_t len,
                                      size_t total_len)
{
    const size_t switch_to_nt_store_size = 2048;
    size_t offset, prefetch_tail;
    __m256i y0, y1, y2, y3, y4, y5, y6, y7;

    ucs_nt_write_prefetch(dst);
    ucs_nt_write_prefetch(UCS_PTR_BYTE_OFFSET(dst, 64));
    ucs_nt_write_prefetch(UCS_PTR_BYTE_OFFSET(dst, 128));

    /* copy 64 bytes unconditionally */
    y0 = _mm256_loadu_si256(src);
    y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, 32));
    _mm256_storeu_si256(dst, y0);
    _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, 32), y1);

    if (ucs_unlikely(total_len > switch_to_nt_store_size)) {
        offset = 64 - ((uintptr_t)dst & 0x1f);
        len   -= offset;

        if (ucs_likely((size_t)UCS_PTR_BYTE_OFFSET(src, offset) & 0x1f)) {
            /* src address is not aligned to 32 byte */
            while (len >= 256) {
                y4 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset));
                y5 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 32));
                y6 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 64));
                y7 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 96));
                y0 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 128));
                y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 160));
                y2 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 192));
                y3 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 224));
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset), y4);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 32), y5);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 64), y6);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 96), y7);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 128), y0);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 160), y1);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 192), y2);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 224), y3);

                offset += 256;
                len    -= 256;
            }
        } else {
            /* src address aligned to 32 byte */
            while (len >= 256) {
                y4 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset));
                y5 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 32));
                y6 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 64));
                y7 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 96));
                y0 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 128));
                y1 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 160));
                y2 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 192));
                y3 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 224));
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset), y4);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 32), y5);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 64), y6);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 96), y7);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 128), y0);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 160), y1);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 192), y2);
                _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 224), y3);

                offset += 256;
                len    -= 256;
            }
        }

        while (len >= 64) {
            y4 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset));
            y5 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 32));
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset), y4);
            _mm256_stream_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 32), y5);
            offset += 64;
            len    -= 64;
        }

        if (len) {
            ucs_nt_write_prefetch(UCS_PTR_BYTE_OFFSET(dst, offset));
        }

        /* make the writes visible to the other core */
        ucs_memory_bus_store_fence();
    } else {
        /* copy next 64 bytes unconditionally */
        y2 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, 64));
        y3 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, 96));
        _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, 64), y2);
        _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, 96), y3);

        offset        = 128 - ((uintptr_t)dst & 0x1f);
        prefetch_tail = 192 - (offset + ((uintptr_t)dst & 0x3f));
        len          -= offset;

        if (len > prefetch_tail) {
            ucs_nt_write_prefetch(UCS_PTR_BYTE_OFFSET(dst, 192));
            if (len > (prefetch_tail + 64)) {
                ucs_nt_write_prefetch(UCS_PTR_BYTE_OFFSET(dst, 256));
            }
        }

        while (len >= 128) {
            if (len > (prefetch_tail + 128)) {
                ucs_nt_write_prefetch(UCS_PTR_BYTE_OFFSET(dst, offset + (3 * 64)));
                if (len > (prefetch_tail + 192)) {
                    ucs_nt_write_prefetch(UCS_PTR_BYTE_OFFSET(dst, offset + (4 * 64)));
                }
            }

            y0 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset));
            y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 32));
            y2 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 64));
            y3 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 96));

            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset), y0);
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 32), y1);
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 64), y2);
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 96), y3);

            offset += 128;
            len    -= 128;
        }
    }

    /* Handle the remaining bytes <= 127 */
    return len;
}

static UCS_F_ALWAYS_INLINE
size_t ucs_x86_nt_src_buffer_transfer(void *dst, const void *src, size_t len)
{
    __m256i y0, y1, y2, y3;
    size_t offset, prefetch_tail;

    ucs_nt_read_prefetch(src);
    ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, 64));
    ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, 128));

    /* copy 128 bytes unconditionally */
    y0 = _mm256_loadu_si256(src);
    y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, 32));
    y2 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, 64));
    y3 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, 96));
    _mm256_storeu_si256(dst, y0);
    _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, 32), y1);
    _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, 64), y2);
    _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, 96), y3);

    offset        = 128 - ((uintptr_t)dst & 0x1f);
    prefetch_tail = 192 - (offset + ((uintptr_t)src & 0x3f));
    len          -= offset;

    if (len > prefetch_tail) {
        ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, 192));
        if (len > (prefetch_tail + 64)) {
            ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, 256));
        }
    }

    if (ucs_likely((size_t)UCS_PTR_BYTE_OFFSET(src, offset) & 0x1f)) {
        if (len > (prefetch_tail + 128)) {
            ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, 320));
            if (len > (prefetch_tail + 192)) {
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, 384));
            }
        }

        while (len >= 128) {
            y0 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset));
            y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 32));
            y2 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 64));
            y3 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, offset + 96));
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset), y0);
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 32), y1);
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 64), y2);
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 96), y3);

            if (len > (prefetch_tail + 256)) {
                ucs_nt_read_prefetch(
                    UCS_PTR_BYTE_OFFSET(src, prefetch_tail + offset + (4 * 64)));
                if (len > (prefetch_tail + 320)) {
                    ucs_nt_read_prefetch(
                        UCS_PTR_BYTE_OFFSET(src, prefetch_tail + offset + (5 * 64)));
                }
            }

            offset += 128;
            len    -= 128;
        }
    } else {
        while (len >= 128) {
            if (len > (prefetch_tail + 128)) {
                ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (3 * 64)));
                if (len > (prefetch_tail + 192)) {
                    ucs_nt_read_prefetch(UCS_PTR_BYTE_OFFSET(src, offset + (4 * 64)));
                }
            }

            /* Can we use streaming loads on normal memory type? */
            y0 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset));
            y1 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 32));
            y2 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 64));
            y3 = _mm256_load_si256(UCS_PTR_BYTE_OFFSET(src, offset + 96));
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset), y0);
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 32), y1);
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 64), y2);
            _mm256_store_si256(UCS_PTR_BYTE_OFFSET(dst, offset + 96), y3);

            offset += 128;
            len    -= 128;
        }
    }

    /* Handle the remaining bytes <= 127 */
    return len;
}

static UCS_F_ALWAYS_INLINE void
ucs_x86_copy_bytes_le_128(void *dst, const void *src, uint32_t len)
{
    __m256i y0, y1, y2, y3;
    /* Handle lengths that fall usually within eager short range */
    switch (ucs_count_leading_zero_bits(len)) {
    /* 0 */
    case 32:
        break;
    /* 1 */
    case 31:
        *(uint8_t *)dst = *(uint8_t *)src;
        break;
    /* 2 - 3 */
    case 30:
        *(uint16_t *)dst = *(uint16_t *)src;
        *(uint16_t *)UCS_PTR_BYTE_OFFSET(dst, len - 2) = \
            *(uint16_t *)UCS_PTR_BYTE_OFFSET(src, len - 2);
        break;
    /* 4 - 7 */
    case 29:
        *(uint32_t *)dst = *(uint32_t *)src;
        *(uint32_t *)UCS_PTR_BYTE_OFFSET(dst, len - 4) = \
            *(uint32_t *)UCS_PTR_BYTE_OFFSET(src, len - 4);
        break;
    /* 8 - 15 */
    case 28:
        *(uint64_t *)dst = *(uint64_t *)src;
        *(uint64_t *)UCS_PTR_BYTE_OFFSET(dst, len - 8) = \
            *(uint64_t *)UCS_PTR_BYTE_OFFSET(src, len - 8);
        break;
    /* 16 - 31 */
    case 27:
        *(uint64_t *)dst = *(uint64_t *)src;
        *(uint64_t *)UCS_PTR_BYTE_OFFSET(dst, 8) = \
            *(uint64_t *)UCS_PTR_BYTE_OFFSET(src, 8);
        *(uint64_t *)UCS_PTR_BYTE_OFFSET(dst, len - 16) = \
            *(uint64_t *)UCS_PTR_BYTE_OFFSET(src, len - 16);
        *(uint64_t *)UCS_PTR_BYTE_OFFSET(dst, len - 8) = \
            *(uint64_t *)UCS_PTR_BYTE_OFFSET(src, len - 8);
        break;
    /* 32 - 63 */
    case 26:
        y0 = _mm256_loadu_si256(src);
        y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src,  len - 32));
        _mm256_storeu_si256(dst, y0);
        _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, len - 32), y1);
        break;
    /* 64 - 128 */
    default:
        y0 = _mm256_loadu_si256(src);
        y1 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, 32));
        y2 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, len - 64));
        y3 = _mm256_loadu_si256(UCS_PTR_BYTE_OFFSET(src, len - 32));
        _mm256_storeu_si256(dst, y0);
        _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, 32), y1);
        _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, len - 64), y2);
        _mm256_storeu_si256(UCS_PTR_BYTE_OFFSET(dst, len - 32), y3);
        break;
    }
}
