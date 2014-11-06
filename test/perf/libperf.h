/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCX_LIBPERF_H
#define UCX_LIBPERF_H

#include <ucs/sys/compiler.h>

BEGIN_C_DECLS

#include <uct/api/uct.h>
#include <ucs/sys/math.h>
#include <ucs/type/status.h>


typedef enum {
    UCX_PERF_TEST_CMD_PUT_SHORT,
    UCX_PERF_TEST_CMD_LAST
} ucx_perf_cmd_t;


typedef enum {
    UCX_PERF_TEST_TYPE_PINGPONG,
    UCX_PERF_TEST_TYPE_STREAM_UNI,
    UCX_PERF_TEST_TYPE_STREAM_BI,
    UCX_PERF_TEST_TYPE_LAST
} ucx_perf_test_type_t;


typedef enum {
    UCX_PERF_DATA_LAYOUT_BUFFER,
    UCX_PERF_DATA_LAYOUT_LAST
} ucx_perf_data_layout_t;


typedef enum {
    UCX_PERF_WAIT_MODE_PROGRESS,     /* Repeatedly call progress */
    UCX_PERF_WAIT_MODE_SLEEP,        /* Go to sleep */
    UCX_PERF_WAIT_MODE_SPIN,         /* Spin without calling progress */
    UCX_PERF_WAIT_MODE_LAST
} ucx_perf_wait_mode_t;


enum {
    UCX_PERF_TEST_FLAG_WARMUP     = UCS_BIT(1), /* Run some warm-up iterations */
    UCX_PERF_TEST_FLAG_VALIDATE   = UCS_BIT(2)  /* Validate data. Affects performance. */
};


/**
 * Performance counter type.
 */
typedef uint64_t ucx_perf_counter_t;


/*
 * Performance test result.
 *
 * Time values are in seconds.
 * Size values are in bytes.
 */
typedef struct ucx_perf_result {
    ucx_perf_counter_t      iters;
    double                  elapsed_time;
    ucx_perf_counter_t      bytes;
    struct {
        double              typical;
        double              moment_average; /* Average since last report */
        double              total_average;  /* Average of the whole test */
    }
    latency, bandwidth, msgrate;
} ucx_perf_result_t;


/**
 * RTE used to bring-up the test
 */
typedef struct ucx_perf_test_rte {
    /* @return Group size */
    unsigned   (*group_size)(void *rte_group);

    /* @return My index within the group */
    unsigned   (*group_index)(void *rte_group);

    /* Barrier */
    void        (*barrier)(void *rte_group);

    /* Direct modex */
    void        (*send)(void *rte_group, unsigned dest, void *value, size_t size);
    void        (*recv)(void *rte_group, unsigned src,  void *value, size_t size);

    /* Handle results */
    void        (*report)(void *rte_group, ucx_perf_result_t *result);

} ucx_perf_test_rte_t;


/**
 * Describes a performance test.
 */
typedef struct ucx_perf_test_params {
    ucx_perf_cmd_t         command;         /* Command to perform */
    ucx_perf_test_type_t   test_type;       /* Test communication type */
    ucx_perf_data_layout_t data_layout;     /* Data layout to use */
    ucx_perf_wait_mode_t   wait_mode;       /* How to wait */
    unsigned               flags;           /* Additional flags */

    size_t                 message_size;    /* Test message size */
    size_t                 alignment;       /* Message buffer alignment */
    ucx_perf_counter_t     max_iter;        /* Iterations limit, 0 - unlimited */
    double                 max_time;        /* Time limit (seconds), 0 - unlimited */
    double                 report_interval; /* Interval at which to call the report callback */

    void                   *rte_group;      /* Opaque RTE group handle */
    ucx_perf_test_rte_t    *rte;            /* RTE functions used to exchange data */
} ucx_perf_test_params_t;


/**
 * Run a performance test.
 */
ucs_status_t uct_perf_test_run(uct_context_h context,
                               ucx_perf_test_params_t *params, const char *hw_name,
                               const char *tl_name, ucx_perf_result_t *result);


END_C_DECLS

#endif /* UCX_PERF_H_ */
