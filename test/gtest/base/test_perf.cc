/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include "test_perf.h"

extern "C" {
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

void test_perf::rte::send(void *rte_group, unsigned dest, void *value,
                                      size_t size)
{
    rte *self = reinterpret_cast<rte*>(rte_group);
    if (dest == self->m_index) {
        self->m_self.push(value, size);
    } else if (dest == 1 - self->m_index) {
        self->m_send.push(value, size);
    }
}

void test_perf::rte::recv(void *rte_group, unsigned src, void *value,
                                      size_t size)
{
    rte *self = reinterpret_cast<rte*>(rte_group);
    if (src == self->m_index) {
        self->m_self.pop(value, size);
    } else if (src == 1 - self->m_index) {
        self->m_recv.pop(value, size);
    }
}

void test_perf::rte::post_vec(void *rte_group, struct iovec *iovec,
                                          size_t num, void **req)
{
    int i;
    size_t j;
    int group_size;
    int group_index;
    rte *self = reinterpret_cast<rte*>(rte_group);

    group_size = self->group_size(rte_group);
    group_index = self->group_index(rte_group);

    for (i = 0; i < group_size; ++i) {
        if (i != group_index) {
            for (j = 0; j < num; ++j) {
                self->send(rte_group, i, iovec[j].iov_base, iovec[j].iov_len);
            }
        }
    }
}

void test_perf::rte::recv_vec(void *rte_group, unsigned dest,
                                          struct iovec *iovec, size_t num, void* req)
{
    int group_index;
    size_t i;
    rte *self = reinterpret_cast<rte*>(rte_group);

    group_index = self->group_index(rte_group);
    if (dest != (unsigned)group_index) {
        for (i = 0; i < num; ++i) {
            self->recv(rte_group, dest, iovec[i].iov_base, iovec[i].iov_len);
        }
    }
}

void test_perf::rte::exchange_vec(void *rte_group, void * req) {
}

void test_perf::rte::report(void *rte_group, ucx_perf_result_t *result,
                                        int is_final)
{
}

ucx_perf_rte_t test_perf::rte::test_rte = {
    rte::group_size,
    rte::group_index,
    rte::barrier,
    rte::post_vec,
    rte::recv_vec,
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
    params.api             = test.api;
    params.command         = test.command;
    params.test_type       = test.test_type;
    params.thread_mode     = UCS_THREAD_MODE_SINGLE;
    params.wait_mode       = UCX_PERF_WAIT_MODE_LAST;
    params.flags           = flags;
    params.message_size    = test.msglen;
    params.am_hdr_size     = 8;
    params.alignment       = ucs_get_page_size();
    params.max_outstanding = test.max_outstanding;
    params.warmup_iter     = test.iters / 10;
    params.max_iter        = test.iters;
    params.max_time        = 0.0;
    params.report_interval = 1.0;
    params.rte_group       = NULL;
    params.rte             = &rte::test_rte;
    strncpy(params.uct.dev_name, dev_name.c_str(), sizeof(params.uct.dev_name));
    strncpy(params.uct.tl_name , tl_name.c_str(),  sizeof(params.uct.tl_name));
    params.uct.data_layout = test.data_layout;
    params.uct.fc_window   = UCT_PERF_TEST_MAX_FC_WINDOW;

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

void test_perf::run_test(const test_spec& test, unsigned flags, double min, double max,
                         const std::string &tl_name, const std::string &dev_name)
{
    if (ucs::test_time_multiplier() > 1) {
        UCS_TEST_SKIP;
    }

    std::vector<int> cpus = get_affinity();
    if (cpus.size() < 2) {
        UCS_TEST_MESSAGE << "Need at least 2 CPUs (got: " << cpus.size() << " )";
        throw ucs::test_abort_exception();
    }
    cpus.resize(2);

    char result_str[200] = {0};
    test_result result = run_multi_threaded(test, flags, tl_name, dev_name, cpus);
    if ((result.status == UCS_ERR_UNSUPPORTED) ||
        (result.status == UCS_ERR_UNREACHABLE))
    {
        return;
    }

    ASSERT_UCS_OK(result.status);

    double value = *(double*)( ((char*)&result.result) + test.field_offset) * test.norm;
    snprintf(result_str, sizeof(result_str) - 1, "%s %25s : %.3f %s",
             dev_name.c_str(), test.title, value, test.units);
    UCS_TEST_MESSAGE << result_str;

    EXPECT_GE(value, min);
    EXPECT_LT(value, max);
}

