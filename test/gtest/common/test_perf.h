/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef TEST_PERF_H_
#define TEST_PERF_H_

#include <common/test.h>

#include <tools/perf/libperf.h>


class test_perf {
protected:
    struct test_spec {
        const char             *title;
        const char             *units;
        ucx_perf_api_t         api;
        ucx_perf_cmd_t         command;
        ucx_perf_test_type_t   test_type;
        int                    data_layout;
        size_t                 msg_stride;
        size_t                 msglencnt;
        size_t                 msglen[3];
        unsigned               max_outstanding;
        size_t                 iters;
        size_t                 field_offset;
        double                 norm;
        double                 min; /* TODO remove this field */
        double                 max; /* TODO remove this field */
        unsigned               test_flags;
    };

    static std::vector<int> get_affinity();

    void run_test(const test_spec& test, unsigned flags, bool check_perf,
                  const std::string &tl_name, const std::string &dev_name);

private:
    class rte_comm {
    public:
        rte_comm();

        void push(const void *data, size_t size);

        void pop(void *data, size_t size);

    private:
        pthread_mutex_t  m_mutex;
        std::string      m_queue;
    };

    class rte {
    public:
        /* RTE functions */
        rte(unsigned index, rte_comm& send, rte_comm& recv);

        unsigned index() const;

        static unsigned group_size(void *rte_group);

        static unsigned group_index(void *rte_group);

        static void barrier(void *rte_group);

        static void post_vec(void *rte_group, const struct iovec *iovec,
                             int iovcnt, void **req);

        static void recv(void *rte_group, unsigned src, void *buffer,
                         size_t max, void *req);

        static void exchange_vec(void *rte_group, void * req);

        static void report(void *rte_group, const ucx_perf_result_t *result,
                           void *arg, int is_final);

        static ucx_perf_rte_t test_rte;

    private:
        const unsigned m_index;
        rte_comm       &m_send;
        rte_comm       &m_recv;
    };

    struct thread_arg {
        ucx_perf_params_t   params;
        int                 cpu;
    };

    struct test_result {
        ucs_status_t        status;
        ucx_perf_result_t   result;
    };

    static void set_affinity(int cpu);

    static void* thread_func(void *arg);

    test_result run_multi_threaded(const test_spec &test, unsigned flags,
                                   const std::string &tl_name,
                                   const std::string &dev_name,
                                   const std::vector<int> &cpus);
};

#endif
