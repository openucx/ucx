/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/gtest/test.h>
#include <perf/libperf.h>
extern "C" {
#include <uct/api/uct.h>
}
#include <linux/sched.h>
#include <pthread.h>
#include <boost/format.hpp>
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

    static void report(void *rte_group, ucx_perf_result_t *result)
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
    test_rte::send,
    test_rte::recv,
    test_rte::report,
};


class test_uct_perf : public ucs::test {
protected:
    struct test_spec {
        const char           *title;
        const char           *units;
        double               min;
        double               max;
        ucx_perf_cmd_t       command;
        ucx_perf_test_type_t test_type;
        size_t               msglen;
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
        ucx_perf_result_t *result;
        ucs_status_t status;
        uct_context_h ucth;

        status = uct_init(&ucth);
        ASSERT_UCS_OK(status);
        set_affinity(a->cpu);

        result = new ucx_perf_result_t();

        status = uct_perf_test_run(ucth, &a->params, a->tl_name.c_str(),
                                   a->dev_name.c_str(), result);
        ASSERT_UCS_OK(status);

        uct_cleanup(ucth);
        return result;
    }

    ucx_perf_result_t run_multi_threaded(const test_spec &test,
                                         const std::string &tl_name,
                                         const std::string &dev_name,
                                         const std::vector<int> &cpus)
    {
        test_rte_comm c0to1, c1to0;

        ucx_perf_test_params_t params;
        params.command         = test.command;
        params.test_type       = test.test_type;
        params.data_layout     = UCX_PERF_DATA_LAYOUT_BUFFER;
        params.wait_mode       = UCX_PERF_WAIT_MODE_LAST;
        params.flags           = UCX_PERF_TEST_FLAG_WARMUP;
        params.message_size    = test.msglen;
        params.alignment       = ucs_get_page_size();
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

        ucx_perf_result_t *result0, *result1, result;
        pthread_join(thread0, (void**)&result0);
        pthread_join(thread1, (void**)&result1);

        result = *result0;
        delete result0;
        delete result1;
        return result;
    }

    static test_spec tests[];
};


test_uct_perf::test_spec test_uct_perf::tests[] =
{
  { "put latency", "usec", 0.0, 1.0,
    UCX_PERF_TEST_CMD_PUT_SHORT, UCX_PERF_TEST_TYPE_PINGPONG,   8, 100000l,
    ucs_offsetof(ucx_perf_result_t, latency.total_average), 1e6 },

  { "put msgrate", "Mpps", 6.0, 20.0,
    UCX_PERF_TEST_CMD_PUT_SHORT, UCX_PERF_TEST_TYPE_STREAM_UNI, 8, 2000000l,
    ucs_offsetof(ucx_perf_result_t, msgrate.total_average), 1e-6 },

  { NULL }
};

UCS_TEST_F(test_uct_perf, envelope) {
    ucs_status_t status;
    uct_context_h ucth;
    uct_resource_desc_t *resources, resource;
    unsigned i, num_resources;
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

    status = uct_init(&ucth);
    ASSERT_UCS_OK(status);

    status = uct_query_resources(ucth, &resources, &num_resources);
    ASSERT_UCS_OK(status);

    bool found = false;
    for (i = 0; i < num_resources; ++i) {
        if (!strcmp(resources[i].tl_name, "rc_mlx5") && !strcmp(resources[i].dev_name, "mlx5_0:1")) {
            /* TODO take resource dev/tl name from test env */
            resource = resources[i];
            found = true;
            break;
        }
    }

    uct_release_resource_list(resources);
    uct_cleanup(ucth);

    if (!found) {
        UCS_TEST_SKIP;
    }

    /* For SandyBridge CPUs, don't check performance of far-socket devices */
    check_perf = true;
    if (ucs_get_cpu_model() == UCS_CPU_MODEL_INTEL_SANDYBRIDGE) {
        BOOST_FOREACH(int cpu, cpus) {
            if (!CPU_ISSET(cpu, &resources[i].local_cpus)) {
                UCS_TEST_MESSAGE << "Not enforcing performance on SandyBridge far socket";
                check_perf = false;
                break;
            }
        }
    }

    /* Run all tests */
    for (test_uct_perf::test_spec *test = tests; test->title != NULL; ++test) {
        ucx_perf_result_t result = run_multi_threaded(*test,
                                                      resource.tl_name,
                                                      resource.dev_name,
                                                      cpus);
        double value = *(double*)( (char*)&result + test->field_offset) * test->norm;
        UCS_TEST_MESSAGE << boost::format("%s/%s %15s : %.3f %s")
                            % resource.tl_name
                            % resource.dev_name
                            % test->title
                            % value
                            % test->units;
        if (check_perf) {
            EXPECT_GE(value, test->min);
            EXPECT_LT(value, test->max);
        }
    }

}
