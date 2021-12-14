/**
* Copyright (C) NVIDIA 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/debug/log.h>
#include <ucs/arch/bitops.h>
#include <ucs/sys/module.h>
#include <ucs/sys/string.h>

#include <tools/perf/lib/libperf_int.h>

#include <string.h>
#include <unistd.h>

#if _OPENMP
#   include <omp.h>
#endif /* _OPENMP */


static ucs_status_t ucp_perf_test_alloc_iov_mem(ucp_perf_datatype_t datatype,
                                                size_t iovcnt, unsigned thread_count,
                                                ucp_dt_iov_t **iov_p)
{
    ucp_dt_iov_t *iov;

    if (UCP_PERF_DATATYPE_IOV == datatype) {
        iov = malloc(sizeof(*iov) * iovcnt * thread_count);
        if (NULL == iov) {
            ucs_error("Failed allocate IOV buffer with iovcnt=%lu", iovcnt);
            return UCS_ERR_NO_MEMORY;
        }
        *iov_p = iov;
    }

    return UCS_OK;
}

static ucs_status_t ucp_perf_mem_alloc(const ucx_perf_context_t *perf,
                                       size_t length,
                                       ucs_memory_type_t mem_type,
                                       void **address_p, ucp_mem_h *memh_p)
{
    ucp_mem_map_params_t params;
    ucp_mem_attr_t attr;
    ucs_status_t status;

    params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                         UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                         UCP_MEM_MAP_PARAM_FIELD_FLAGS |
                         UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
    params.address     = NULL;
    params.memory_type = mem_type;
    params.length      = length;
    params.flags       = UCP_MEM_MAP_ALLOCATE;
    if (perf->params.flags & UCX_PERF_TEST_FLAG_MAP_NONBLOCK) {
        params.flags |= UCP_MEM_MAP_NONBLOCK;
    }

    status = ucp_mem_map(perf->ucp.context, &params, memh_p);
    if (status != UCS_OK) {
        return status;
    }

    attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status          = ucp_mem_query(*memh_p, &attr);
    if (status != UCS_OK) {
        ucp_mem_unmap(perf->ucp.context, *memh_p);
        return status;
    }

    *address_p = attr.address;
    return UCS_OK;
}

static void ucp_perf_mem_free(const ucx_perf_context_t *perf, ucp_mem_h memh)
{
    ucs_status_t status;

    status = ucp_mem_unmap(perf->ucp.context, memh);
    if (status != UCS_OK) {
        ucs_warn("ucp_mem_unmap() failed: %s", ucs_status_string(status));
    }
}

ucs_status_t ucp_perf_test_alloc_mem(ucx_perf_context_t *perf)
{
    ucx_perf_params_t *params = &perf->params;
    ucs_status_t status;
    size_t buffer_size;

    if (params->iov_stride) {
        buffer_size = params->msg_size_cnt * params->iov_stride;
    } else {
        buffer_size = ucx_perf_get_message_size(params);
    }

    /* Allocate send buffer memory */
    status = ucp_perf_mem_alloc(perf, buffer_size * params->thread_count,
                                params->send_mem_type, &perf->send_buffer,
                                &perf->ucp.send_memh);
    if (status != UCS_OK) {
        goto err;
    }

    /* Allocate receive buffer memory */
    status = ucp_perf_mem_alloc(perf, buffer_size * params->thread_count,
                                params->recv_mem_type, &perf->recv_buffer,
                                &perf->ucp.recv_memh);
    if (status != UCS_OK) {
        goto err_free_send_buffer;
    }

    /* Allocate AM header */
    if (params->ucp.am_hdr_size != 0) {
        perf->ucp.am_hdr = malloc(params->ucp.am_hdr_size);
        if (perf->ucp.am_hdr == NULL) {
            goto err_free_buffers;
        }
    } else {
        perf->ucp.am_hdr = NULL;
    }

    /* Allocate IOV datatype memory */
    perf->ucp.send_iov = NULL;
    status = ucp_perf_test_alloc_iov_mem(params->ucp.send_datatype,
                                         perf->params.msg_size_cnt,
                                         params->thread_count,
                                         &perf->ucp.send_iov);
    if (UCS_OK != status) {
        goto err_free_am_hdr;
    }

    perf->ucp.recv_iov = NULL;
    status = ucp_perf_test_alloc_iov_mem(params->ucp.recv_datatype,
                                         perf->params.msg_size_cnt,
                                         params->thread_count,
                                         &perf->ucp.recv_iov);
    if (UCS_OK != status) {
        goto err_free_send_iov_buffers;
    }

    return UCS_OK;

err_free_send_iov_buffers:
    free(perf->ucp.send_iov);
err_free_am_hdr:
    free(perf->ucp.am_hdr);
err_free_buffers:
    ucp_perf_mem_free(perf, perf->ucp.recv_memh);
err_free_send_buffer:
    ucp_perf_mem_free(perf, perf->ucp.send_memh);
err:
    return UCS_ERR_NO_MEMORY;
}

void ucp_perf_test_free_mem(ucx_perf_context_t *perf)
{
    free(perf->ucp.recv_iov);
    free(perf->ucp.send_iov);
    free(perf->ucp.am_hdr);
    ucp_perf_mem_free(perf, perf->ucp.recv_memh);
    ucp_perf_mem_free(perf, perf->ucp.send_memh);
}

static void
ucx_perf_test_memcpy_host(void *dst, ucs_memory_type_t dst_mem_type,
                          const void *src, ucs_memory_type_t src_mem_type,
                          size_t count)
{
    if ((dst_mem_type != UCS_MEMORY_TYPE_HOST) ||
        (src_mem_type != UCS_MEMORY_TYPE_HOST)) {
        ucs_error("wrong memory type passed src - %d, dst - %d",
                  src_mem_type, dst_mem_type);
    } else {
        memcpy(dst, src, count);
    }
}

static ucs_status_t
uct_perf_test_alloc_host(const ucx_perf_context_t *perf, size_t length,
                         unsigned flags, uct_allocated_memory_t *alloc_mem)
{
    ucs_status_t status;

    status = uct_iface_mem_alloc(perf->uct.iface, length,
                                 flags, "perftest", alloc_mem);
    if (status != UCS_OK) {
        ucs_error("failed to allocate memory: %s", ucs_status_string(status));
        return status;
    }

    ucs_assert(alloc_mem->md == perf->uct.md);

    return UCS_OK;
}

static void uct_perf_test_free_host(const ucx_perf_context_t *perf,
                                    uct_allocated_memory_t *alloc_mem)
{
    uct_iface_mem_free(alloc_mem);
}

ucs_status_t uct_perf_test_alloc_mem(ucx_perf_context_t *perf)
{
    ucx_perf_params_t *params = &perf->params;
    ucs_status_t status;
    unsigned flags;
    size_t buffer_size;

    if ((UCT_PERF_DATA_LAYOUT_ZCOPY == params->uct.data_layout) && params->iov_stride) {
        buffer_size = params->msg_size_cnt * params->iov_stride;
    } else {
        buffer_size = ucx_perf_get_message_size(params);
    }

    /* TODO use params->alignment  */

    flags = (params->flags & UCX_PERF_TEST_FLAG_MAP_NONBLOCK) ?
             UCT_MD_MEM_FLAG_NONBLOCK : 0;
    flags |= UCT_MD_MEM_ACCESS_ALL;

    /* Allocate send buffer memory */
    status = perf->send_allocator->uct_alloc(perf,
                                             buffer_size * params->thread_count,
                                             flags, &perf->uct.send_mem);

    if (status != UCS_OK) {
        goto err;
    }

    perf->send_buffer = perf->uct.send_mem.address;

    /* Allocate receive buffer memory */
    status = perf->recv_allocator->uct_alloc(perf,
                                             buffer_size * params->thread_count,
                                             flags, &perf->uct.recv_mem);
    if (status != UCS_OK) {
        goto err_free_send;
    }

    perf->recv_buffer = perf->uct.recv_mem.address;

    /* Allocate IOV datatype memory */
    perf->params.msg_size_cnt = params->msg_size_cnt;
    perf->uct.iov             = malloc(sizeof(*perf->uct.iov) *
                                       perf->params.msg_size_cnt *
                                       params->thread_count);
    if (NULL == perf->uct.iov) {
        status = UCS_ERR_NO_MEMORY;
        ucs_error("Failed allocate send IOV(%lu) buffer: %s",
                  perf->params.msg_size_cnt, ucs_status_string(status));
        goto err_free_recv;
    }

    ucs_debug("allocated memory. Send buffer %p, Recv buffer %p",
              perf->send_buffer, perf->recv_buffer);
    return UCS_OK;

err_free_recv:
    perf->recv_allocator->uct_free(perf, &perf->uct.recv_mem);
err_free_send:
    perf->send_allocator->uct_free(perf, &perf->uct.send_mem);
err:
    return status;
}

void uct_perf_test_free_mem(ucx_perf_context_t *perf)
{
    perf->send_allocator->uct_free(perf, &perf->uct.send_mem);
    perf->recv_allocator->uct_free(perf, &perf->uct.recv_mem);
    free(perf->uct.iov);
}

void ucx_perf_global_init()
{
    static ucx_perf_allocator_t host_allocator = {
        .mem_type  = UCS_MEMORY_TYPE_HOST,
        .init      = ucs_empty_function_return_success,
        .uct_alloc = uct_perf_test_alloc_host,
        .uct_free  = uct_perf_test_free_host,
        .memcpy    = ucx_perf_test_memcpy_host,
        .memset    = memset
    };
    UCS_MODULE_FRAMEWORK_DECLARE(ucx_perftest);

    ucx_perf_mem_type_allocators[UCS_MEMORY_TYPE_HOST] = &host_allocator;

    /* FIXME Memtype allocator modules must be loaded to global scope, otherwise
     * alloc hooks, which are using dlsym() to get pointer to original function,
     * do not work. Need to use bistro for memtype hooks to fix it.
     */
    UCS_MODULE_FRAMEWORK_LOAD(ucx_perftest, UCS_MODULE_LOAD_FLAG_GLOBAL);
}
