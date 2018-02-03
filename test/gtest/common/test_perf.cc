/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University 
*               of Tennessee Research Foundation. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "test_perf.h"

extern "C" {
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
}
#include <pthread.h>
#include <string>
#include <vector>


test_perf::rte_comm::rte_comm() {
    pthread_mutex_init(&m_mutex, NULL);
}

void test_perf::rte_comm::push(const void *data, size_t size) {
    pthread_mutex_lock(&m_mutex);
    m_queue.append((const char *)data, size);
    pthread_mutex_unlock(&m_mutex);
}

void test_perf::rte_comm::pop(void *data, size_t size) {
    bool done = false;
    do {
        pthread_mutex_lock(&m_mutex);
        if (m_queue.length() >= size) {
            memcpy(data, &m_queue[0], size);
            m_queue.erase(0, size);
            done = true;
        }
        pthread_mutex_unlock(&m_mutex);
    } while (!done);
}


test_perf::rte::rte(unsigned index, rte_comm& send, rte_comm& recv) :
    m_index(index), m_send(send), m_recv(recv) {
}

unsigned test_perf::rte::index() const {
    return m_index;
}

unsigned test_perf::rte::group_size(void *rte_group) {
    return 2;
}

unsigned test_perf::rte::group_index(void *rte_group) {
    rte *self = reinterpret_cast<rte*>(rte_group);
    return self->index();
}

void test_perf::rte::barrier(void *rte_group) {
    static const uint32_t magic = 0xdeadbeed;
    rte *self = reinterpret_cast<rte*>(rte_group);
    uint32_t dummy = magic;
    self->m_send.push(&dummy, sizeof(dummy));
    dummy = 0;
    self->m_recv.pop(&dummy, sizeof(dummy));
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

    if (src != 1 - self->m_index) {
        return;
    }

    self->m_recv.pop(&size, sizeof(size));
    ucs_assert_always(size <= max);
    self->m_recv.pop(buffer, size);
}

void test_perf::rte::exchange_vec(void *rte_group, void * req)
{
}

void test_perf::rte::report(void *rte_group, const ucx_perf_result_t *result,
                            void *arg, int is_final)
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

    ret = sched_getaffinity(getpid(), sizeof(affinity), &affinity);
    if (ret != 0) {
        ucs_error("Failed to get CPU affinity: %m");
        throw ucs::test_abort_exception();
    }

    nr_cpus = sysconf(_SC_NPROCESSORS_CONF);
    if (nr_cpus < 0) {
        ucs_error("Failed to get CPU count: %m");
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

void* test_perf::thread_func(void *arg)
{
    thread_arg *a = (thread_arg*)arg;
    test_result *result;

    set_affinity(a->cpu);
    result = new test_result();
    result->status = ucx_perf_run(&a->params, &result->result);
    return result;
}

test_perf::test_result test_perf::run_multi_threaded(const test_spec &test, unsigned flags,
                                                     const std::string &tl_name,
                                                     const std::string &dev_name,
                                                     const std::vector<int> &cpus)
{
    rte_comm c0to1, c1to0;

    ucx_perf_params_t params;
    memset(&params, 0, sizeof(params));
    params.api = test.api;
    params.command         = test.command;
    params.test_type       = test.test_type;
    params.thread_mode     = UCS_THREAD_MODE_SINGLE;
    params.async_mode      = UCS_ASYNC_MODE_THREAD;
    params.thread_count    = 1;
    params.wait_mode       = UCX_PERF_WAIT_MODE_LAST;
    params.flags           = test.test_flags | flags;
    params.am_hdr_size     = 8;
    params.alignment       = ucs_get_page_size();
    params.max_outstanding = test.max_outstanding;
    if (ucs::test_time_multiplier() == 1) {
        params.warmup_iter     = test.iters / 10;
        params.max_iter        = test.iters;
    } else {
        params.warmup_iter     = 0;
        params.max_iter        = ucs_min(20u,
                                         test.iters / ucs::test_time_multiplier());
    }
    params.max_time        = 0.0;
    params.report_interval = 1.0;
    params.rte_group       = NULL;
    params.rte             = &rte::test_rte;
    params.report_arg      = NULL;
    ucs_strncpy_zero(params.uct.dev_name, dev_name.c_str(), sizeof(params.uct.dev_name));
    ucs_strncpy_zero(params.uct.tl_name , tl_name.c_str(),  sizeof(params.uct.tl_name));
    params.uct.data_layout = (uct_perf_data_layout_t)test.data_layout;
    params.uct.fc_window   = UCT_PERF_TEST_MAX_FC_WINDOW;
    params.msg_size_cnt    = test.msglencnt;
    params.msg_size_list   = (size_t *)test.msglen;
    params.iov_stride      = test.msg_stride;
    params.ucp.send_datatype = (ucp_perf_datatype_t)test.data_layout;
    params.ucp.recv_datatype = (ucp_perf_datatype_t)test.data_layout;

    thread_arg arg0;
    arg0.params   = params;
    arg0.cpu      = cpus[0];

    rte rte0(0, c0to1, c1to0);
    arg0.params.rte_group = &rte0;

    pthread_t thread0, thread1;
    int ret = pthread_create(&thread0, NULL, thread_func, &arg0);
    if (ret) {
        UCS_TEST_MESSAGE << strerror(errno);
        throw ucs::test_abort_exception();
    }

    thread_arg arg1;
    arg1.params   = params;
    arg1.cpu      = cpus[1];

    rte rte1(1, c1to0, c0to1);
    arg1.params.rte_group = &rte1;

    ret = pthread_create(&thread1, NULL, thread_func, &arg1);
    if (ret) {
        UCS_TEST_MESSAGE << strerror(errno);
        throw ucs::test_abort_exception();
    }

    void *ptr0, *ptr1;
    pthread_join(thread0, &ptr0);
    pthread_join(thread1, &ptr1);

    test_result *result0 = reinterpret_cast<test_result*>(ptr0),
                *result1 = reinterpret_cast<test_result*>(ptr1);
    test_result result = *result1;
    delete result0;
    delete result1;
    return result;
}

void test_perf::run_test(const test_spec& test, unsigned flags, bool check_perf,
                         const std::string &tl_name, const std::string &dev_name)
{
    std::vector<int> cpus = get_affinity();
    if (cpus.size() < 2) {
        UCS_TEST_MESSAGE << "Need at least 2 CPUs (got: " << cpus.size() << " )";
        throw ucs::test_abort_exception();
    }
    cpus.resize(2);

    check_perf = check_perf &&
                 (ucs::test_time_multiplier() == 1) &&
                 (ucs::perf_retry_count > 0);
    for (int i = 0; i < (ucs::perf_retry_count + 1); ++i) {
        test_result result = run_multi_threaded(test, flags, tl_name, dev_name,
                                                cpus);
        if ((result.status == UCS_ERR_UNSUPPORTED) ||
            (result.status == UCS_ERR_UNREACHABLE))
        {
            return; /* Skipped */
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
            return; /* Skip */
        } else if ((value >= test.min) && (value <= test.max)) {
            return; /* Success */
        } else {
            ucs::safe_sleep(ucs::perf_retry_interval);
        }
    }

     ADD_FAILURE() << "Invalid " << test.title << " performance, expected: " <<
                      std::setprecision(3) << test.min << ".." << test.max;
}

