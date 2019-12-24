/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

#include <common/test_helpers.h>
#include <ucs/sys/sys.h>


std::vector<ucp_test_param>
test_ucp_memheap::enum_test_params(const ucp_params_t& ctx_params,
                                   const std::string& name,
                                   const std::string& test_case_name,
                                   const std::string& tls)
{
    std::vector<ucp_test_param> result;
    generate_test_params_variant(ctx_params, name,
                                 test_case_name, tls, 0, result);
    generate_test_params_variant(ctx_params, name,
                                 test_case_name + "/map_nb",
                                 tls, UCP_MEM_MAP_NONBLOCK, result);
    return result;
}

void test_ucp_memheap::mem_map_and_rkey_exchange(ucp_test_base::entity &receiver,
                                                 ucp_test_base::entity &sender,
                                                 const ucp_mem_map_params_t &params,
                                                 ucp_mem_h &receiver_memh,
                                                 ucp_rkey_h &sender_rkey,
                                                 void **memheap_addr_p)
{
    ucs_status_t status;
    ucp_mem_attr_t mem_attr;

    status = ucp_mem_map(receiver.ucph(), &params, &receiver_memh);
    ASSERT_UCS_OK(status);

    mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                          UCP_MEM_ATTR_FIELD_LENGTH;
    status = ucp_mem_query(receiver_memh, &mem_attr);
    ASSERT_UCS_OK(status);
    EXPECT_GE(mem_attr.length, params.length);
    if (memheap_addr_p != NULL) {
        *memheap_addr_p = mem_attr.address;
    }

    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(receiver.ucph(), receiver_memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    status = ucp_ep_rkey_unpack(sender.ep(), rkey_buffer, &sender_rkey);
    ASSERT_UCS_OK(status);

    ucp_rkey_buffer_release(rkey_buffer);
}

void test_ucp_memheap::test_nonblocking_implicit_stream_xfer(nonblocking_send_func_t send,
                                                             size_t size, int max_iter,
                                                             size_t alignment,
                                                             bool malloc_allocate,
                                                             bool is_ep_flush)
{
    void *memheap;
    size_t memheap_size;
    ucp_mem_map_params_t params;
    ucs_status_t status;

    memheap = NULL;
    memheap_size = max_iter * size + alignment;

    if (max_iter == DEFAULT_ITERS) {
        max_iter = 300 / ucs::test_time_multiplier();
    }

    if (size == DEFAULT_SIZE) {
        size = ucs_max((size_t)ucs::rand() % (12 * UCS_KBYTE), alignment);
    }
    memheap_size = max_iter * size + alignment;

    sender().connect(&receiver(), get_ep_params());

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.length     = memheap_size;
    params.flags      = GetParam().variant;
    if (malloc_allocate) {
        memheap = malloc(memheap_size);
        params.address = memheap;
        params.flags   = params.flags & (~(UCP_MEM_MAP_ALLOCATE|UCP_MEM_MAP_FIXED));
    } else if (params.flags & UCP_MEM_MAP_FIXED) {
        params.address = ucs::mmap_fixed_address();
    } else {
        params.address = NULL;
        params.flags  |= UCP_MEM_MAP_ALLOCATE;
    }

    ucp_mem_h memh;
    ucp_rkey_h rkey;
    mem_map_and_rkey_exchange(receiver(), sender(), params, memh, rkey,
                              (!malloc_allocate ? &memheap : NULL));
    memset(memheap, 0, memheap_size);

    std::string expected_data[300];
    assert (max_iter <= 300);

    for (int i = 0; i < max_iter; ++i) {
        expected_data[i].resize(size);

        ucs::fill_random(expected_data[i]);

        ucs_assert(size * i + alignment <= memheap_size);

        char *ptr = (char*)memheap + alignment + i * size;
        (this->*send)(&sender(), size, (void*)ptr, rkey, expected_data[i]);
    }

    if (is_ep_flush) {
        flush_ep(sender());
    } else {
        flush_worker(sender());
    }

    for (int i = 0; i < max_iter; ++i) {
        char *ptr = (char*)memheap + alignment + i * size;
        EXPECT_EQ(expected_data[i].substr(0, 20),
                  std::string(ptr, expected_data[i].length()).substr(0, 20)) <<
                        ((void*)ptr);
    }

    ucp_rkey_destroy(rkey);

    disconnect(sender());

    status = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(status);

    if (malloc_allocate) {
        free(memheap);
    }
}

/* NOTE: alignment is ignored if memheap_size is not default */
void test_ucp_memheap::test_blocking_xfer(blocking_send_func_t send,
                                          size_t memheap_size, int max_iter,
                                          size_t alignment,
                                          bool malloc_allocate, 
                                          bool is_ep_flush)
{
    ucp_mem_map_params_t params;
    ucs_status_t status;
    size_t size;
    int zero_offset = 0;

    if (max_iter == DEFAULT_ITERS) {
        max_iter = 300 / ucs::test_time_multiplier();
    }

    if (memheap_size == DEFAULT_SIZE) {
        memheap_size = 3 * UCS_KBYTE;
        zero_offset = 1;
    }

    sender().connect(&receiver(), get_ep_params());

    /* avoid deadlock for blocking rma/amo */
    flush_worker(sender());

    void *memheap = NULL;

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.length     = memheap_size;
    params.flags      = GetParam().variant;
    if (malloc_allocate) {
        memheap = malloc(memheap_size);
        params.address = memheap;
        params.flags   = params.flags & (~(UCP_MEM_MAP_ALLOCATE|UCP_MEM_MAP_FIXED));
    } else if (params.flags & UCP_MEM_MAP_FIXED) {
        params.address = ucs::mmap_fixed_address();
        params.flags |= UCP_MEM_MAP_ALLOCATE;
    } else {
        params.address = NULL;
        params.flags |= UCP_MEM_MAP_ALLOCATE;
    }

    ucp_mem_h memh;
    ucp_rkey_h rkey;
    mem_map_and_rkey_exchange(receiver(), sender(), params, memh, rkey,
                              (!malloc_allocate ? &memheap : NULL));
    memset(memheap, 0, memheap_size);

    for (int i = 0; i < max_iter; ++i) {
        size_t offset;

        if (!zero_offset) {
            size = ucs_max(ucs::rand() % (memheap_size - alignment - 1), alignment);
            offset = ucs::rand() % (memheap_size - size - alignment);
        } else {
            size = memheap_size;
            offset = 0;
        }

        offset = ucs_align_up(offset, alignment);

        ucs_assert(((((uintptr_t)memheap + offset)) % alignment) == 0);
        ucs_assert(size + offset <= memheap_size);

        std::string expected_data;
        expected_data.resize(size);

        ucs::fill_random(expected_data);
        (this->*send)(&sender(), size, (void*)((uintptr_t)memheap + offset),
                      rkey, expected_data);

        if (is_ep_flush) {
            flush_ep(sender());
        } else {
            flush_worker(sender());
        }

        EXPECT_EQ(expected_data,
                  std::string((char*)memheap + offset, expected_data.length()));

        expected_data.clear();
    }

    ucp_rkey_destroy(rkey);

    disconnect(sender());

    status = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(status);

    if (malloc_allocate) {
        free(memheap);
    }
}

void test_ucp_memheap_check_mem_type::init() {
    m_local_mem_type  = mem_type_pairs[GetParam().variant][0];
    m_remote_mem_type = mem_type_pairs[GetParam().variant][1];

    test_ucp_memheap::init();
    m_remote_mem_buf = new mem_buffer(get_data_size(), m_remote_mem_type);
    m_remote_mem_buf->pattern_fill(m_remote_mem_buf->ptr(),
                                   m_remote_mem_buf->size(), 0,
                                   m_remote_mem_buf->mem_type());
    m_local_mem_buf = new mem_buffer(get_data_size(), m_local_mem_type);
    m_local_mem_buf->pattern_fill(m_local_mem_buf->ptr(),
                                  m_local_mem_buf->size(), 0,
                                  m_local_mem_buf->mem_type());
    sender().connect(&receiver(), get_ep_params());

    ucp_mem_map_params_t params;
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
        UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = m_remote_mem_buf->ptr();
    params.length  = m_remote_mem_buf->size();
    mem_map_and_rkey_exchange(receiver(), sender(), params,
                              m_remote_mem_buf_memh,
                              m_remote_mem_buf_rkey);
}

void test_ucp_memheap_check_mem_type::cleanup() {
        ucs_status_t status;
        ucp_rkey_destroy(m_remote_mem_buf_rkey);
        status = ucp_mem_unmap(receiver().ucph(), m_remote_mem_buf_memh);
        ASSERT_UCS_OK(status);
        disconnect(sender());
        delete m_local_mem_buf;
        delete m_remote_mem_buf;
        test_ucp_memheap::cleanup();
    }

std::vector<ucp_test_param> test_ucp_memheap_check_mem_type::
enum_test_params(const ucp_params_t& ctx_params, const std::string& name,
                 const std::string& test_case_name, const std::string& tls) {
    std::vector<ucp_test_param> result;
    int count = 0;

    for (std::vector<std::vector<ucs_memory_type_t> >::const_iterator iter =
             mem_type_pairs.begin(); iter != mem_type_pairs.end(); ++iter) {
        generate_test_params_variant(ctx_params, name, test_case_name + "/" +
                                     std::string(ucs_memory_type_names[(*iter)[0]]) +
                                     "<->" + std::string(ucs_memory_type_names[(*iter)[1]]),
                                     tls, count++, result);
    }

    return result;
}

ucs_log_func_rc_t test_ucp_memheap_check_mem_type::
error_handler(const char *file, unsigned line, const char *function,
              ucs_log_level_t level, const char *message, va_list ap) {
    // Ignore errors that invalid input parameters as it is expected
    if (level == UCS_LOG_LEVEL_ERROR) {
        std::string err_str = format_message(message, ap);

        if ((err_str.find(err_exp_str) != std::string::npos) ||
            /* the error below occurs when RMA lane can be configured for
             * current TL (i.e. RMA emulation over AM lane is not used) and
             * UCT is unable to do registration for the current memory type
             * (e.g. SHM TLs or IB TLs w/o GPUDirect support)*/
            (err_str.find("remote memory is unreachable") != std::string::npos)) {
            UCS_TEST_MESSAGE << err_str;
            return UCS_LOG_FUNC_RC_STOP;
        }
    }

    return UCS_LOG_FUNC_RC_CONTINUE;
}

std::string test_ucp_memheap_check_mem_type::
get_err_exp_str(const std::string &op_type,
                bool check_local,
                bool check_remote) {
    return "UCP doesn't support " + op_type +  " for \"" +
        (check_local ?
         std::string(ucs_memory_type_names[m_local_mem_type]) :
         std::string(ucs_memory_type_names[UCS_MEMORY_TYPE_HOST])) +
        "\"<->\"" +
        (check_remote ?
         std::string(ucs_memory_type_names[m_remote_mem_type]) :
         std::string(ucs_memory_type_names[UCS_MEMORY_TYPE_HOST])) +
        "\" memory types";
}

void test_ucp_memheap_check_mem_type::
check_mem_type_op_status(ucs_status_t status,
                         bool check_local,
                         bool check_remote,
                         bool allow_gpu_direct_local,
                         bool allow_gpu_direct_remote) {
    if ((check_local &&
         !UCP_MEM_IS_ACCESSIBLE_FROM_CPU(m_local_mem_type) &&
         (!check_gpu_direct_support(m_local_mem_type) ||
          !allow_gpu_direct_local)) ||
        (check_remote &&
         !UCP_MEM_IS_ACCESSIBLE_FROM_CPU(m_remote_mem_type) &&
         (!check_gpu_direct_support(m_remote_mem_type) ||
          !allow_gpu_direct_remote))) {
        EXPECT_TRUE((status == UCS_ERR_UNSUPPORTED) ||
                    (status == UCS_ERR_UNREACHABLE));
    } else {
        ASSERT_UCS_OK_OR_INPROGRESS(status);
    }
}

std::string test_ucp_memheap_check_mem_type::err_exp_str = "";

std::vector<std::vector<ucs_memory_type_t> >
test_ucp_memheap_check_mem_type::mem_type_pairs = ucs::supported_mem_type_pairs();
