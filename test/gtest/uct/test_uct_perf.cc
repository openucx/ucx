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

    static unsigned group_size(void *rte_group)
    {
        return 2;
    }

    static unsigned group_index(void *rte_group)
    {
        test_rte *self = reinterpret_cast<test_rte*>(rte_group);
        return self->m_index;
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
        ucx_perf_cmd_t       command;
        ucx_perf_test_type_t test_type;
        size_t               msglen;
        double               norm;
        size_t               field_offset;
    };

    test_uct_perf() {
        memset(&m_orig_affinity, 0, sizeof(m_orig_affinity));
    }

    void init() {
        ucs::test::init();

        sched_getaffinity(getpid(), sizeof(m_orig_affinity), &m_orig_affinity);

        const int max_cpus = sysconf(_SC_NPROCESSORS_CONF);
        int num_cpus = 0, first_cpu = -1;

        for (int cpu = 0; cpu < sysconf(_SC_NPROCESSORS_CONF); ++cpu) {
            if (CPU_ISSET(cpu, &m_orig_affinity)) {
                ++num_cpus;
                first_cpu = cpu;
            }
        }

        ucs_assert_always(num_cpus > 0 && first_cpu != -1);

        if (num_cpus < 2) {
            unsigned next_cpu = (first_cpu + 1) % max_cpus;
            UCS_TEST_MESSAGE << "Changing CPU affinity to " << first_cpu <<
                            "," << next_cpu;

            cpu_set_t affinity = m_orig_affinity;
            CPU_SET(next_cpu , &affinity);
            sched_setaffinity(getpid(), sizeof(affinity), &affinity);
        }
    }

    void cleanup() {
        sched_setaffinity(getpid(), sizeof(m_orig_affinity), &m_orig_affinity);
        ucs::test::cleanup();
    }

    static void* thread_func(void *arg)
    {
        ucx_perf_test_params_t *params = (ucx_perf_test_params_t*)arg;
        ucx_perf_result_t *result;
        ucs_status_t status;
        uct_context_h ucth;

        status = uct_init(&ucth);
        ASSERT_UCS_OK(status);

        result = new ucx_perf_result_t();

        status = uct_perf_test_run(ucth, params, "mlx5_0:1", "rc_mlx5", result);
        ASSERT_UCS_OK(status);

        uct_cleanup(ucth);
        return result;
    }

    ucx_perf_result_t run_multi_threaded(const test_spec &test, const char *hw_name,
                                         const char *tl_name)
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
        params.max_iter        = 400000l;
        params.max_time        = 0.0;
        params.report_interval = 1.0;
        params.rte_group       = NULL;
        params.rte             = &test_rte::rte;

        ucx_perf_test_params_t params0 = params;
        test_rte rte0(0, c0to1, c1to0);
        params0.rte_group = &rte0;

        pthread_t thread0, thread1;
        int ret = pthread_create(&thread0, NULL, thread_func, &params0);
        if (ret) {
            UCS_TEST_MESSAGE << strerror(errno);
            throw ucs::test_abort_exception();
        }

        ucx_perf_test_params_t params1 = params;
        test_rte rte1(1, c1to0, c0to1);
        params1.rte_group = &rte1;

        ret = pthread_create(&thread1, NULL, thread_func, &params1);
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

    cpu_set_t m_orig_affinity;

};


test_uct_perf::test_spec test_uct_perf::tests[] =
{
  { "put latency", "usec", UCX_PERF_TEST_CMD_PUT_SHORT, UCX_PERF_TEST_TYPE_PINGPONG,
    8, 1e6, ucs_offsetof(ucx_perf_result_t, latency.total_average) },

  { NULL }
};

UCS_TEST_F(test_uct_perf, envelope) {
    if (ucs::test_time_multiplier() > 1) {
        UCS_TEST_SKIP;
    }

    const char *hw_name = "mlx5_0:1";
    const char *tl_name = "rc_mlx5";
    for (test_uct_perf::test_spec *test = tests; test->title != NULL; ++test) {
        ucx_perf_result_t result = run_multi_threaded(*test, hw_name, tl_name);
        double value = *(double*)( (char*)&result + test->field_offset);
        UCS_TEST_MESSAGE << boost::format("%s/%s %15s : %.3f %s")
                            % tl_name
                            % hw_name
                            % test->title
                            % (value * test->norm)
                            % test->units;
    }
}
