/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University
*               of Tennessee Research Foundation. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2020.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "test_perf.h"

extern "C" {
#include <ucs/async/async.h>
#include <ucs/sys/string.h>
}
#include <pthread.h>
#include <string>
#include <vector>


#define UCP_ARM_PERF_TEST_MULTIPLIER  2
#define UCT_ARM_PERF_TEST_MULTIPLIER 15
#define UCT_PERF_TEST_MULTIPLIER      5


test_perf::rte_comm::rte_comm() {
    pthread_mutex_init(&m_mutex, NULL);
}

void test_perf::rte_comm::push(const void *data, size_t size) {
    pthread_mutex_lock(&m_mutex);
    m_queue.append((const char *)data, size);
    pthread_mutex_unlock(&m_mutex);
}

void test_perf::rte_comm::pop(void *data, size_t size,
                              void (*progress)(void *arg), void *arg) {
    bool done = false;
    do {
        pthread_mutex_lock(&m_mutex);
        if (m_queue.length() >= size) {
            memcpy(data, &m_queue[0], size);
            m_queue.erase(0, size);
            done = true;
        }
        pthread_mutex_unlock(&m_mutex);
        if (!done) {
            progress(arg);
        }
    } while (!done);
}


test_perf::rte::rte(unsigned index, unsigned group_size, unsigned peer,
                    rte_comm& send, rte_comm& recv) :
    m_index(index), m_gsize(group_size), m_peer(peer), m_send(send),
    m_recv(recv) {
}

unsigned test_perf::rte::index() const {
    return m_index;
}

unsigned test_perf::rte::group_index(void *rte_group) {
    rte *self = reinterpret_cast<rte*>(rte_group);
    return self->index();
}

unsigned test_perf::rte::gsize() const {
    return m_gsize;
}

unsigned test_perf::rte::group_size(void *rte_group) {
    rte *self = reinterpret_cast<rte*>(rte_group);
    return self->gsize();
}

void test_perf::rte::barrier(void *rte_group, void (*progress)(void *arg),
                             void *arg) {
    static const uint32_t magic = 0xdeadbeed;
    rte *self = reinterpret_cast<rte*>(rte_group);
    uint32_t dummy = magic;
    self->m_send.push(&dummy, sizeof(dummy));
    dummy = 0;
    self->m_recv.pop(&dummy, sizeof(dummy), progress, arg);
    ucs_assert_always(dummy == magic);
}

void test_perf::rte::post_vec(void *rte_group, const struct iovec *iovec,
                              int iovcnt, void **req)
{
    rte *self = reinterpret_cast<rte*>(rte_group);
    size_t size;
    int i;

    size = 0;
    for (i = 0; i < iovcnt; ++i) {
        size += iovec[i].iov_len;
    }

    self->m_send.push(&size, sizeof(size));
    for (i = 0; i < iovcnt; ++i) {
        self->m_send.push(iovec[i].iov_base, iovec[i].iov_len);
    }
}

void test_perf::rte::recv(void *rte_group, unsigned src, void *buffer,
                          size_t max, void *req)
{
    rte *self = reinterpret_cast<rte*>(rte_group);
    size_t size;

    if (src != self->m_peer) {
        return;
    }

    self->m_recv.pop(&size, sizeof(size), (void(*)(void*))ucs_empty_function, NULL);
    ucs_assert_always(size <= max);
    self->m_recv.pop(buffer, size, (void(*)(void*))ucs_empty_function, NULL);
}

void test_perf::rte::exchange_vec(void *rte_group, void * req)
{
}

void test_perf::rte::report(void *rte_group, const ucx_perf_result_t *result,
                            void *arg, const char *extra_info, int is_final,
                            int is_multi_thread)
{
}

ucx_perf_rte_t test_perf::rte::test_rte = {
    rte::group_size,
    rte::group_index,
    rte::barrier,
    rte::post_vec,
    rte::recv,
    rte::exchange_vec,
    rte::report,
};

std::vector<int> test_perf::get_affinity() {
    std::vector<int> cpus;
    cpu_set_t affinity;
    int ret, nr_cpus;

    CPU_ZERO(&affinity);
    ret = sched_getaffinity(getpid(), sizeof(affinity), &affinity);
    if (ret != 0) {
        ucs_error("Failed to get CPU affinity: %m");
        throw ucs::test_abort_exception();
    }

    nr_cpus = ucs_sys_get_num_cpus();
    if (nr_cpus < 0) {
        throw ucs::test_abort_exception();
    }

    for (int cpu = 0; cpu < nr_cpus; ++cpu) {
        if (CPU_ISSET(cpu, &affinity)) {
            cpus.push_back(cpu);
        }
    }

    return cpus;
}

void test_perf::set_affinity(int cpu)
{
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(cpu , &affinity);
    sched_setaffinity(ucs_get_tid(), sizeof(affinity), &affinity);
}

void* test_perf::test_func(void *arg)
{
    thread_arg *a = (thread_arg*)arg;
    rte *r        = reinterpret_cast<rte*>(a->params.rte_group);
    test_result *result;

    ucs_log_set_thread_name("p-%d", r->index());
    set_affinity(a->cpu);
    result = new test_result();
    result->status = ucx_perf_run(&a->params, &result->result);
    return result;
}

void test_perf::test_params_init(const test_spec &test,
                                 ucx_perf_params_t &params,
                                 unsigned flags,
                                 const std::string &tl_name,
                                 const std::string &dev_name)
{
    params.api             = test.api;
    params.command         = test.command;
    params.test_type       = test.test_type;
    params.thread_mode     = UCS_THREAD_MODE_SINGLE;
    params.async_mode      = UCS_ASYNC_THREAD_LOCK_TYPE;
    params.thread_count    = 1;
    params.wait_mode       = test.wait_mode;
    params.flags           = test.test_flags | flags;
    params.uct.am_hdr_size = 8;
    params.alignment       = ucs_get_page_size();
    params.max_outstanding = test.max_outstanding;
    params.send_mem_type   = test.send_mem_type;
    params.recv_mem_type   = test.recv_mem_type;
    params.percentile_rank = 50.0;

    memset(params.uct.md_name, 0, sizeof(params.uct.md_name));

    if (ucs::test_time_multiplier() == 1) {
        params.warmup_iter = ucs_max(1, test.iters / 100);
        params.max_iter    = test.iters;
    } else {
        params.warmup_iter = 0;
        params.max_iter    = ucs_min(20u, test.iters /
                                          ucs::test_time_multiplier() /
                                          ucs::test_time_multiplier());
    }

    params.warmup_time     = 100e-3;
    params.max_time        = 0.0;
    params.report_interval = 1.0;
    params.rte_group       = NULL;
    params.rte             = &rte::test_rte;
    params.report_arg      = NULL;

    ucs_strncpy_zero(params.uct.dev_name, dev_name.c_str(), sizeof(params.uct.dev_name));
    ucs_strncpy_zero(params.uct.tl_name , tl_name.c_str(), sizeof(params.uct.tl_name));

    params.uct.data_layout      = (uct_perf_data_layout_t)test.data_layout;
    params.uct.fc_window        = UCT_PERF_TEST_MAX_FC_WINDOW;
    params.msg_size_cnt         = test.msglencnt;
    params.msg_size_list        = (size_t *)test.msglen;
    params.iov_stride           = test.msg_stride;
    params.ucp.send_datatype    = (ucp_perf_datatype_t)test.data_layout;
    params.ucp.recv_datatype    = (ucp_perf_datatype_t)test.data_layout;
    params.ucp.nonblocking_mode = 0;
    params.ucp.am_hdr_size      = 0;
}

test_perf::test_result test_perf::run_multi_threaded(const test_spec &test, unsigned flags,
                                                     const std::string &tl_name,
                                                     const std::string &dev_name,
                                                     const std::vector<int> &cpus)
{
    rte_comm c0to1, c1to0;
    ucx_perf_params_t params;

    test_params_init(test, params, flags, tl_name, dev_name);

    rte rte0(0, 2, 1, c0to1, c1to0);
    rte rte1(1, 2, 0, c1to0, c0to1);
    rte *rtes[2] = {&rte0, &rte1};

    /* Run 2 test threads */
    thread_arg args[2];
    pthread_t threads[2];

    for (unsigned i = 0; i < 2; ++i) {
        args[i].params           = params;
        args[i].cpu              = cpus[i];
        args[i].params.rte_group = rtes[i];

        ucs_status_t status = ucs_pthread_create(&threads[i], test_func,
                                                 &args[i], "perf%d", i);
        if (status != UCS_OK) {
            throw ucs::test_abort_exception();
        }
    }

    /* Collect results */
    test_result *results[2], result;
    for (unsigned i = 0; i < 2; ++i) {
        void *ptr;

        pthread_join(threads[i], &ptr);
        results[i] = reinterpret_cast<test_result*>(ptr);
        if (i == 1) {
            result = *results[i];
        }

        delete results[i];
    }

    return result;
}

test_perf::test_result
test_perf::run_single_threaded(const test_spec &test, unsigned flags,
                               const std::string &tl_name,
                               const std::string &dev_name)
{
    rte_comm c0to0;
    ucx_perf_params_t params;

    test_params_init(test, params, flags, tl_name, dev_name);

    rte rte0(0, 1, 0, c0to0, c0to0);
    params.rte_group = &rte0;

    test_result result;
    result.status = ucx_perf_run(&params, &result.result);
    return result;
}

double test_perf::run_test(const test_spec& test, unsigned flags, bool check_perf,
                           const std::string &tl_name, const std::string &dev_name)
{
    std::vector<int> cpus;

    if (!(flags & UCX_PERF_TEST_FLAG_LOOPBACK)) {
        cpus = get_affinity();
        if (cpus.size() < 2) {
            UCS_TEST_MESSAGE << "Need at least 2 CPUs (got: " << cpus.size()
                             << " )";
            throw ucs::test_abort_exception();
        }
        cpus.resize(2);
    }

    check_perf = check_perf &&
                 (ucs::test_time_multiplier() == 1) &&
                 (ucs::perf_retry_count > 0);
    for (int i = 0; i < (ucs::perf_retry_count + 1); ++i) {
        test_result result;

        if (flags & UCX_PERF_TEST_FLAG_LOOPBACK) {
            result = run_single_threaded(test, flags, tl_name, dev_name);
        } else {
            result = run_multi_threaded(test, flags, tl_name, dev_name, cpus);
        }

        if ((result.status == UCS_ERR_UNSUPPORTED) ||
            (result.status == UCS_ERR_UNREACHABLE))
        {
            return 0.0; /* Skipped */
        }

        ASSERT_UCS_OK(result.status);

        double value = *(double*)( ((char*)&result.result) + test.field_offset) *
                        test.norm;
        char result_str[200] = {0};
        snprintf(result_str, sizeof(result_str) - 1, "%s %25s : %.3f %s",
                 dev_name.c_str(), test.title, value, test.units);
        if (i == 0) {
            if (check_perf) {
                UCS_TEST_MESSAGE << result_str;
            } else {
                UCS_TEST_MESSAGE << result_str << " (performance not checked)";
            }
        } else {
            UCS_TEST_MESSAGE << result_str << " (attempt " << i << ")";
        }

        if (!check_perf) {
            return value; /* Skip */
        } else if ((value >= test.min) && (value <= test.max)) {
            return value; /* Success */
        } else {
            ucs::safe_sleep(ucs::perf_retry_interval);
        }
    }

    ADD_FAILURE() << "Invalid " << test.title << " performance, expected: "
                  << std::setprecision(3) << test.min << ".." << test.max;

    return 0.0;
}
