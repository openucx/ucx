/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_test.h"

#include <tools/perf/libperf.h>
#include <pthread.h>
#include <string>
#include <vector>


class test_rte_comm {
public:
    test_rte_comm() {
        pthread_mutex_init(&m_mutex, NULL);
    }

    void push(const void *data, size_t size) {
        pthread_mutex_lock(&m_mutex);
        m_queue.append((const char *)data, size);
        pthread_mutex_unlock(&m_mutex);
    }

    void pop(void *data, size_t size) {
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

    pthread_mutex_t  m_mutex;
    std::string      m_queue;
};


class test_rte {
public:
    /* RTE functions */
    test_rte(unsigned index, test_rte_comm& send, test_rte_comm& recv) :
        m_index(index), m_send(send), m_recv(recv) {
    }

    unsigned index() const {
        return m_index;
    }

    static unsigned group_size(void *rte_group)
    {
        return 2;
    }

    static unsigned group_index(void *rte_group)
    {
        test_rte *self = reinterpret_cast<test_rte*>(rte_group);
        return self->index();
    }

    static void barrier(void *rte_group)
    {
        static const uint32_t magic = 0xdeadbeed;
        test_rte *self = reinterpret_cast<test_rte*>(rte_group);
        uint32_t dummy = magic;
        self->m_send.push(&dummy, sizeof(dummy));
        dummy = 0;
        self->m_recv.pop(&dummy, sizeof(dummy));
        ucs_assert_always(dummy == magic);
    }

    static void send(void *rte_group, unsigned dest, void *value, size_t size)
    {
        test_rte *self = reinterpret_cast<test_rte*>(rte_group);
        if (dest == self->m_index) {
            self->m_self.push(value, size);
        } else if (dest == 1 - self->m_index) {
            self->m_send.push(value, size);
        }
    }

    static void recv(void *rte_group, unsigned src,  void *value, size_t size)
    {
        test_rte *self = reinterpret_cast<test_rte*>(rte_group);
        if (src == self->m_index) {
            self->m_self.pop(value, size);
        } else if (src == 1 - self->m_index) {
            self->m_recv.pop(value, size);
        }
    }

    static void post_vec(void *rte_group, struct iovec *iovec, size_t num, void **req)
    {
        int i;
        size_t j;
        int group_size;
        int group_index;
        test_rte *self = reinterpret_cast<test_rte*>(rte_group);

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

    static void recv_vec(void *rte_group, unsigned dest, struct iovec *iovec, size_t num, void * req)
    {
        int group_index;
        size_t i;
        test_rte *self = reinterpret_cast<test_rte*>(rte_group);

        group_index = self->group_index(rte_group);
        if (dest != (unsigned)group_index) {
            for (i = 0; i < num; ++i) {
                self->recv(rte_group, dest, iovec[i].iov_base, iovec[i].iov_len);
            }
        }
    }

    static void exchange_vec(void *rte_group, void * req)
    {
    }

    static void report(void *rte_group, ucx_perf_result_t *result, int is_final)
    {
    }

    static ucx_perf_test_rte_t rte;

private:
    const unsigned m_index;
    test_rte_comm &m_send;
    test_rte_comm &m_recv;
    test_rte_comm m_self;
};

ucx_perf_test_rte_t test_rte::rte = {
    test_rte::group_size,
    test_rte::group_index,
    test_rte::barrier,
    test_rte::post_vec,
    test_rte::recv_vec,
    test_rte::exchange_vec,
    test_rte::report,
};


class test_uct_perf : public uct_test {
protected:
    struct test_spec {
        const char           *title;
        const char           *units;
        double               min;
        double               max;
        ucx_perf_cmd_t       command;
        ucx_perf_data_layout_t data_layout;
        ucx_perf_test_type_t test_type;
        size_t               msglen;
        unsigned             max_outstanding;
        size_t               iters;
        size_t               field_offset;
        double               norm;
    };

    struct thread_arg {
        ucx_perf_test_params_t   params;
        std::string              tl_name;
        std::string              dev_name;
        int                      cpu;
    };

    struct test_result {
        ucs_status_t        status;
        ucx_perf_result_t   result;
    };

    static std::vector<int> get_affinity() {
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

    static void set_affinity(int cpu)
    {
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpu , &affinity);
        sched_setaffinity(ucs_get_tid(), sizeof(affinity), &affinity);
    }

    static void* thread_func(void *arg)
    {
        thread_arg *a = (thread_arg*)arg;
        test_result *result;
        ucs_status_t status;
        uct_context_h ucth;

        status = uct_init(&ucth);
        ASSERT_UCS_OK(status);
        set_affinity(a->cpu);

        uct_iface_config_t *iface_config;
        status = uct_iface_config_read(ucth, a->tl_name.c_str(), NULL, NULL,
                                       &iface_config);
        ASSERT_UCS_OK(status);

        result = new test_result();
        status = uct_perf_test_run(ucth, &a->params, a->tl_name.c_str(),
                                   a->dev_name.c_str(), iface_config,
                                   &result->result);
        result->status = status;

        uct_iface_config_release(iface_config);
        uct_cleanup(ucth);
        return result;
    }

    test_result run_multi_threaded(const test_spec &test,
                                   const std::string &tl_name,
                                   const std::string &dev_name,
                                   const std::vector<int> &cpus)
    {
        test_rte_comm c0to1, c1to0;

        ucx_perf_test_params_t params;
        params.command         = test.command;
        params.test_type       = test.test_type;
        params.data_layout     = test.data_layout;
        params.thread_mode     = UCT_THREAD_MODE_SINGLE;
        params.wait_mode       = UCX_PERF_WAIT_MODE_LAST;
        params.flags           = 0;
        params.message_size    = test.msglen;
        params.hdr_size        = 8;
        params.alignment       = ucs_get_page_size();
        params.fc_window       = UCX_PERF_TEST_MAX_FC_WINDOW;
        params.max_outstanding = test.max_outstanding;
        params.warmup_iter     = test.iters / 10;
        params.max_iter        = test.iters;
        params.max_time        = 0.0;
        params.report_interval = 1.0;
        params.rte_group       = NULL;
        params.rte             = &test_rte::rte;

        thread_arg arg0;
        arg0.params   = params;
        arg0.tl_name  = tl_name;
        arg0.dev_name = dev_name;
        arg0.cpu      = cpus[0];

        test_rte rte0(0, c0to1, c1to0);
        arg0.params.rte_group = &rte0;

        pthread_t thread0, thread1;
        int ret = pthread_create(&thread0, NULL, thread_func, &arg0);
        if (ret) {
            UCS_TEST_MESSAGE << strerror(errno);
            throw ucs::test_abort_exception();
        }

        thread_arg arg1;
        arg1.params   = params;
        arg1.tl_name  = tl_name;
        arg1.dev_name = dev_name;
        arg1.cpu      = cpus[1];

        test_rte rte1(1, c1to0, c0to1);
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
        test_result result = *result0;
        delete result0;
        delete result1;
        return result;
    }

    static test_spec tests[];
    static int can_run_test(const test_spec &test, uint32_t cap_flags);
};


test_uct_perf::test_spec test_uct_perf::tests[] =
{
  { "am latency", "usec", 0.1, 1.3,
    UCX_PERF_TEST_CMD_AM,  UCX_PERF_DATA_LAYOUT_SHORT, UCX_PERF_TEST_TYPE_PINGPONG,
    8, 1, 100000l, ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6 },

  { "am rate", "Mpps", 3.0, 20.0,
    UCX_PERF_TEST_CMD_AM, UCX_PERF_DATA_LAYOUT_SHORT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    8, 1, 2000000l, ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6 },

  { "am bcopy bw", "MB/sec", 700.0, 10000.0,
    UCX_PERF_TEST_CMD_AM, UCX_PERF_DATA_LAYOUT_BCOPY, UCX_PERF_TEST_TYPE_STREAM_UNI,
    2048, 1, 100000l, ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), pow(1024.0, -2) },

  { "am zcopy bw", "MB/sec", 700.0, 10000.0,
    UCX_PERF_TEST_CMD_AM, UCX_PERF_DATA_LAYOUT_ZCOPY, UCX_PERF_TEST_TYPE_STREAM_UNI,
    2048, 32, 100000l, ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), pow(1024.0, -2) },

  { "put latency", "usec", 0.01, 1.5,
    UCX_PERF_TEST_CMD_PUT, UCX_PERF_DATA_LAYOUT_SHORT, UCX_PERF_TEST_TYPE_PINGPONG,
    8, 1, 100000l, ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6 },

  { "put rate", "Mpps", 1.5, 20.0,
    UCX_PERF_TEST_CMD_PUT, UCX_PERF_DATA_LAYOUT_SHORT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    8, 1, 2000000l, ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6 },

  { "put bcopy bw", "MB/sec", 700.0, 10000.0,
    UCX_PERF_TEST_CMD_PUT, UCX_PERF_DATA_LAYOUT_BCOPY, UCX_PERF_TEST_TYPE_STREAM_UNI,
    2048, 1, 100000l, ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), pow(1024.0, -2) },

  { "put zcopy bw", "MB/sec", 700.0, 10000.0,
    UCX_PERF_TEST_CMD_PUT, UCX_PERF_DATA_LAYOUT_ZCOPY, UCX_PERF_TEST_TYPE_STREAM_UNI,
    2048, 32, 100000l, ucs_offsetof(ucx_perf_result_t, bandwidth.total_average), pow(1024.0, -2) },

  { "get latency", "usec", 0.1, 2.5,
    UCX_PERF_TEST_CMD_GET,  UCX_PERF_DATA_LAYOUT_ZCOPY, UCX_PERF_TEST_TYPE_STREAM_UNI,
    8, 1, 100000l, ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6 },

  { "atomic add latency", "usec", 0.1, 2.5,
    UCX_PERF_TEST_CMD_ADD,  UCX_PERF_DATA_LAYOUT_SHORT, UCX_PERF_TEST_TYPE_PINGPONG,
    8, 1, 100000l, ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6 },

  { "atomic add rate", "Mpps", 1.0, 5.0,
    UCX_PERF_TEST_CMD_ADD,  UCX_PERF_DATA_LAYOUT_SHORT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    8, 1, 2000000l, ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6 },

  { "atomic fadd latency", "usec", 0.1, 2.5,
    UCX_PERF_TEST_CMD_FADD,  UCX_PERF_DATA_LAYOUT_SHORT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    8, 1, 100000l, ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6 },

  { "atomic cswap latency", "usec", 0.1, 2.5,
    UCX_PERF_TEST_CMD_CSWAP,  UCX_PERF_DATA_LAYOUT_SHORT, UCX_PERF_TEST_TYPE_STREAM_UNI,
    8, 1, 100000l, ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6 },

  { NULL }
};

UCS_TEST_P(test_uct_perf, envelope) {
    uct_resource_desc_t resource = GetParam();
    entity tl(resource, 0);
    bool check_perf;

    if (ucs::test_time_multiplier() > 1) {
        UCS_TEST_SKIP;
    }

    std::vector<int> cpus = get_affinity();
    if (cpus.size() < 2) {
        UCS_TEST_MESSAGE << "Need at least 2 CPUs (got: " << cpus.size() << " )";
        throw ucs::test_abort_exception();
    }
    cpus.resize(2);

    /* For SandyBridge CPUs, don't check performance of far-socket devices */
    check_perf = true;
    if (ucs_get_cpu_model() == UCS_CPU_MODEL_INTEL_SANDYBRIDGE) {
        for (std::vector<int>::iterator iter = cpus.begin(); iter != cpus.end(); ++iter) {
            if (!CPU_ISSET(*iter, &resource.local_cpus)) {
                UCS_TEST_MESSAGE << "Not enforcing performance on SandyBridge far socket";
                check_perf = false;
                break;
            }
        }
    }

    /* Run all tests */
    for (test_uct_perf::test_spec *test = tests; test->title != NULL; ++test) {
        char result_str[200] = {0};
        test_result result = run_multi_threaded(*test,
                                                resource.tl_name,
                                                resource.dev_name,
                                                cpus);
        if (result.status == UCS_ERR_UNSUPPORTED) {
            continue;
        }

        ASSERT_UCS_OK(result.status);

        double value = *(double*)( ((char*)&result.result) + test->field_offset) * test->norm;
        snprintf(result_str, sizeof(result_str) - 1, "%s/%s %25s : %.3f %s",
                 resource.tl_name, resource.dev_name, test->title, value, test->units);
        UCS_TEST_MESSAGE << result_str;
        if (check_perf) {
            /* TODO take expected values from resource advertised capabilities */
            EXPECT_GE(value, test->min);
            EXPECT_LT(value, test->max);
        }
    }
}

UCT_INSTANTIATE_TEST_CASE(test_uct_perf);
