/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "libperf_int.h"


static void uct_test_am_pack_cb(void *dest, void *arg, size_t length)
{
    uct_perf_context_t *perf = (uct_perf_context_t*)arg;
    uint64_t *hdr = (uint64_t*)dest;

    *hdr = perf->am_hdr;
    memcpy(hdr + 1, perf->super.send_buffer, length);
}

template <ucx_perf_data_layout_t D>
static UCS_F_ALWAYS_INLINE
void uct_perf_am_short_b(uct_ep_h ep, uint64_t hdr, void *buffer, unsigned length,
                         uct_perf_context_t *perf)
{
    ucs_status_t status;

    do {
        switch (D) {
        case UCX_PERF_DATA_LAYOUT_SHORT:
            status = uct_ep_am_short(ep, UCT_PERF_TEST_AM_ID, hdr, buffer, length);
            break;
        case UCX_PERF_DATA_LAYOUT_BCOPY:
            perf->am_hdr = hdr;
            status = uct_ep_am_bcopy(ep, UCT_PERF_TEST_AM_ID, uct_test_am_pack_cb,
                                     perf, length);
            break;
        case UCX_PERF_DATA_LAYOUT_ZCOPY:
            status = uct_ep_am_zcopy(ep, UCT_PERF_TEST_AM_ID, &hdr, sizeof(hdr),
                                     buffer, length, perf->send_lkey, NULL);
            break;
        }
        if (status != UCS_ERR_WOULD_BLOCK) {
            break;
        }
        uct_progress(perf->context);
    } while (1);
}

template <ucx_perf_data_layout_t D>
static UCS_F_ALWAYS_INLINE void
uct_perf_put_short_b(uct_ep_h ep, void *buffer, unsigned length,
                     uint64_t remote_addr, uct_rkey_t rkey, uct_perf_context_t *perf)
{
    ucs_status_t status;

    do {
        switch (D) {
        case UCX_PERF_DATA_LAYOUT_SHORT:
            status = uct_ep_put_short(ep, buffer, length, remote_addr, rkey);
            break;
        case UCX_PERF_DATA_LAYOUT_BCOPY:
            status = uct_ep_put_bcopy(ep, (uct_pack_callback_t)memcpy,
                                      buffer, length, remote_addr, rkey);
            break;
        case UCX_PERF_DATA_LAYOUT_ZCOPY:
            status = uct_ep_put_zcopy(ep, buffer, length, perf->send_lkey,
                                      remote_addr, rkey, NULL);
            break;
        }
        if (status != UCS_ERR_WOULD_BLOCK) {
            break;
        }
        uct_progress(perf->context);
    } while (1);
}

template <ucx_perf_data_layout_t D>
static ucs_status_t uct_perf_run_put_lat(uct_perf_context_t *perf)
{
    volatile uint8_t *send_sn, *recv_sn;
    unsigned my_index;
    uct_ep_h ep;
    void *buffer;
    size_t length;
    unsigned long remote_addr;
    uct_rkey_t rkey;
    uint8_t sn;

    if (perf->super.params.message_size < 1) {
        return UCS_ERR_INVALID_PARAM;
    }

    recv_sn = (uint8_t*)perf->super.recv_buffer + perf->super.params.message_size - 1;
    send_sn = (uint8_t*)perf->super.send_buffer + perf->super.params.message_size - 1;

    *recv_sn = -1;
    rte_call(&perf->super, barrier);

    my_index = rte_call(&perf->super, group_index);

    buffer = perf->super.send_buffer;
    length = perf->super.params.message_size;

    ucx_perf_test_start_clock(&perf->super);

    *send_sn = sn = 0;
    if (my_index == 0) {
        ep          = perf->peers[1].ep;
        remote_addr = perf->peers[1].remote_addr;
        rkey        = perf->peers[1].rkey.rkey;
        UCX_PERF_TEST_FOREACH(&perf->super) {
            while (*recv_sn != sn);
            uct_perf_put_short_b<D>(ep, buffer, length, remote_addr, rkey, perf);
            uct_progress(perf->context);
            *send_sn = ++sn;
            ucx_perf_update(&perf->super, 1, length);
        }
    } else if (my_index == 1) {
        ep          = perf->peers[0].ep;
        remote_addr = perf->peers[0].remote_addr;
        rkey        = perf->peers[0].rkey.rkey;
        UCX_PERF_TEST_FOREACH(&perf->super) {
            uct_perf_put_short_b<D>(ep, buffer,length, remote_addr, rkey, perf);
            uct_progress(perf->context);
            while (*recv_sn != sn);
            *send_sn = ++sn;
            ucx_perf_update(&perf->super, 1, length);
        }
    }
    return UCS_OK;
}

template <ucx_perf_data_layout_t D>
static ucs_status_t uct_perf_run_am_lat(uct_perf_context_t *perf)
{
    uint64_t sn, *am_sn;
    unsigned my_index;
    uct_ep_h ep;
    void *buffer;
    size_t length;

    if (perf->super.params.message_size < 8) {
        return UCS_ERR_INVALID_PARAM;
    }

    am_sn = (uint64_t*)perf->super.recv_buffer;
    *am_sn = 0;
    rte_call(&perf->super, barrier);

    my_index = rte_call(&perf->super, group_index);
    buffer   = perf->super.send_buffer;
    length   = perf->super.params.message_size - 8;

    ucx_perf_test_start_clock(&perf->super);

    sn = 0;
    if (my_index == 0) {
        ep = perf->peers[1].ep;
        UCX_PERF_TEST_FOREACH(&perf->super) {
            uct_perf_am_short_b<D>(ep, sn + 1, buffer, length, perf);
            while (*am_sn == sn) {
                uct_progress(perf->context);
            }
            sn = *am_sn;
            ucx_perf_update(&perf->super, 1, length);
        }
    } else if (my_index == 1) {
        ep = perf->peers[0].ep;
        UCX_PERF_TEST_FOREACH(&perf->super) {
            while (*am_sn == sn) {
                uct_progress(perf->context);
            }
            sn = *am_sn;
            uct_perf_am_short_b<D>(ep, sn + 1, buffer, length, perf);
            ucx_perf_update(&perf->super, 1, length);
        }
    }

    return UCS_OK;
}

template <ucx_perf_data_layout_t D>
static ucs_status_t uct_perf_run_put_bw(uct_perf_context_t *perf)
{
    unsigned long remote_addr;
    uct_rkey_t rkey;
    void *buffer;
    volatile uint8_t *ptr;
    unsigned length;
    uct_ep_h ep;

    if (perf->super.params.message_size < 1) {
        return UCS_ERR_INVALID_PARAM;
    }

    *(uint8_t*)perf->super.recv_buffer = 0;

    rte_call(&perf->super, barrier);

    ucx_perf_test_start_clock(&perf->super);

    if (rte_call(&perf->super, group_index) == 0) {
        ep          = perf->peers[1].ep;
        buffer      = perf->super.send_buffer;
        length      = perf->super.params.message_size;
        remote_addr = perf->peers[1].remote_addr;
        rkey        = perf->peers[1].rkey.rkey;
        *(uint8_t*)buffer = 1;

        UCX_PERF_TEST_FOREACH(&perf->super) {
            uct_perf_put_short_b<D>(ep, buffer, length, remote_addr, rkey, perf);
            ucx_perf_update(&perf->super, 1, length);
        }

        *(uint8_t*)buffer = 2;
        uct_perf_put_short_b<D>(ep, buffer, length, remote_addr, rkey, perf);
    } else {
        ptr = (uint8_t*)perf->super.recv_buffer;
        while (*ptr != 2);
    }
    return UCS_OK;
}

template <ucx_perf_data_layout_t D>
static ucs_status_t uct_perf_run_am_bw(uct_perf_context_t *perf)
{
    void *buffer;
    unsigned length;
    uct_ep_h ep;
    uint64_t send_sn, *recv_sn;
    uint64_t window;

    if (perf->super.params.message_size < 8) {
        return UCS_ERR_INVALID_PARAM;
    }

    send_sn  = 0;
    recv_sn  = (uint64_t*)perf->super.recv_buffer;
    *recv_sn = 0;

    rte_call(&perf->super, barrier);

    ucx_perf_test_start_clock(&perf->super);

    buffer = perf->super.send_buffer;
    length = perf->super.params.message_size;
    window = perf->super.params.am_window;

    if (rte_call(&perf->super, group_index) == 0) {
        ep = perf->peers[1].ep;
        UCX_PERF_TEST_FOREACH(&perf->super) {
            ++send_sn;
            while (send_sn > (*recv_sn) + window) {
                uct_progress(perf->context);
            }
            uct_perf_am_short_b<D>(ep, send_sn, buffer, length, perf);
            ucx_perf_update(&perf->super, 1, length);
        }
        uct_perf_am_short_b<D>(ep, (uint64_t)-1, buffer, length, perf);
    } else {
        ep = perf->peers[0].ep;
        while (*recv_sn != (uint64_t)-1) {
            uct_progress(perf->context);
            if (*recv_sn > send_sn + (window / 2)) {
                send_sn = *recv_sn;
                uct_perf_am_short_b<D>(ep, send_sn, buffer, length, perf);
            }
        }
    }
    return UCS_OK;
}

#define uct_perf_test_dispatch_data(perf, func) \
    ({ \
        ucs_status_t status; \
        switch (perf->super.params.data_layout) { \
        case UCX_PERF_DATA_LAYOUT_SHORT: \
            status = func<UCX_PERF_DATA_LAYOUT_SHORT>(perf); \
            break; \
        case UCX_PERF_DATA_LAYOUT_BCOPY: \
            status = func<UCX_PERF_DATA_LAYOUT_BCOPY>(perf); \
            break; \
        case UCX_PERF_DATA_LAYOUT_ZCOPY: \
            status = func<UCX_PERF_DATA_LAYOUT_ZCOPY>(perf); \
            break; \
        default: \
            status = UCS_ERR_INVALID_PARAM; \
            break; \
        } \
        status; \
    })

ucs_status_t uct_perf_test_dispatch(uct_perf_context_t *perf)
{
    ucs_status_t status;

    if (perf->super.params.command == UCX_PERF_TEST_CMD_AM &&
        perf->super.params.test_type == UCX_PERF_TEST_TYPE_PINGPONG)
    {
        status = uct_perf_test_dispatch_data(perf, uct_perf_run_am_lat);
    } else if (perf->super.params.command == UCX_PERF_TEST_CMD_PUT &&
               perf->super.params.test_type == UCX_PERF_TEST_TYPE_PINGPONG)
    {
        status = uct_perf_test_dispatch_data(perf, uct_perf_run_put_lat);
    } else if (perf->super.params.command == UCX_PERF_TEST_CMD_AM &&
               perf->super.params.test_type == UCX_PERF_TEST_TYPE_STREAM_UNI)
    {
        status = uct_perf_test_dispatch_data(perf, uct_perf_run_am_bw);
    } else if (perf->super.params.command == UCX_PERF_TEST_CMD_PUT &&
               perf->super.params.test_type == UCX_PERF_TEST_TYPE_STREAM_UNI)
    {
        status = uct_perf_test_dispatch_data(perf, uct_perf_run_put_bw);
    } else {
        return UCS_ERR_INVALID_PARAM;
    }

    return status;
}
