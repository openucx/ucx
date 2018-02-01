/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University 
*               of Tennessee Research Foundation. 2015. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCX_LIBPERF_H
#define UCX_LIBPERF_H

#include <ucs/sys/compiler.h>
#if HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

BEGIN_C_DECLS

#include <sys/uio.h>
#include <uct/api/uct.h>
#include <ucp/api/ucp.h>
#include <ucs/sys/math.h>
#include <ucs/type/status.h>


typedef enum {
    UCX_PERF_API_UCT,
    UCX_PERF_API_UCP,
    UCX_PERF_API_LAST
} ucx_perf_api_t;


typedef enum {
    UCX_PERF_CMD_AM,
    UCX_PERF_CMD_PUT,
    UCX_PERF_CMD_GET,
    UCX_PERF_CMD_ADD,
    UCX_PERF_CMD_FADD,
    UCX_PERF_CMD_SWAP,
    UCX_PERF_CMD_CSWAP,
    UCX_PERF_CMD_TAG,
    UCX_PERF_CMD_TAG_SYNC,
    UCX_PERF_CMD_STREAM,
    UCX_PERF_CMD_LAST
} ucx_perf_cmd_t;


typedef enum {
    UCX_PERF_TEST_TYPE_PINGPONG,         /* Ping-pong mode */
    UCX_PERF_TEST_TYPE_STREAM_UNI,       /* Unidirectional stream */
    UCX_PERF_TEST_TYPE_STREAM_BI,        /* Bidirectional stream */
    UCX_PERF_TEST_TYPE_LAST
} ucx_perf_test_type_t;


typedef enum {
    UCP_PERF_DATATYPE_CONTIG,
    UCP_PERF_DATATYPE_IOV,
} ucp_perf_datatype_t;


typedef enum {
    UCT_PERF_DATA_LAYOUT_SHORT,
    UCT_PERF_DATA_LAYOUT_BCOPY,
    UCT_PERF_DATA_LAYOUT_ZCOPY,
    UCT_PERF_DATA_LAYOUT_LAST
} uct_perf_data_layout_t;


typedef enum {
    UCX_PERF_WAIT_MODE_PROGRESS,     /* Repeatedly call progress */
    UCX_PERF_WAIT_MODE_SLEEP,        /* Go to sleep */
    UCX_PERF_WAIT_MODE_SPIN,         /* Spin without calling progress */
    UCX_PERF_WAIT_MODE_LAST
} ucx_perf_wait_mode_t;


enum ucx_perf_test_flags {
    UCX_PERF_TEST_FLAG_VALIDATE         = UCS_BIT(1), /* Validate data. Affects performance. */
    UCX_PERF_TEST_FLAG_ONE_SIDED        = UCS_BIT(2), /* For test which involve only one side,
                                                         the responder would not call progress(). */
    UCX_PERF_TEST_FLAG_MAP_NONBLOCK     = UCS_BIT(3), /* Map memory in non-blocking mode */
    UCX_PERF_TEST_FLAG_TAG_WILDCARD     = UCS_BIT(4), /* For tag tests, use wildcard mask */
    UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE  = UCS_BIT(5), /* For tag tests, use probe to get unexpected receive */
    UCX_PERF_TEST_FLAG_VERBOSE          = UCS_BIT(7), /* Print error messages */
    UCX_PERF_TEST_FLAG_STREAM_RECV_DATA = UCS_BIT(8)  /* For stream tests, use recv data API */
};


enum {
    UCT_PERF_TEST_MAX_FC_WINDOW   = 127         /* Maximal flow-control window */
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
typedef struct ucx_perf_rte {
    /* @return Group size */
    unsigned   (*group_size)(void *rte_group);

    /* @return My index within the group */
    unsigned   (*group_index)(void *rte_group);

    /* Barrier */
    void        (*barrier)(void *rte_group);

    /* Direct modex */
    void        (*post_vec)(void *rte_group, const struct iovec *iovec,
                            int iovcnt, void **req);
    void        (*recv)(void *rte_group, unsigned src, void *buffer, size_t max,
                        void *req);
    void        (*exchange_vec)(void *rte_group, void *req);

    /* Handle results */
    void        (*report)(void *rte_group, const ucx_perf_result_t *result,
                          void *arg, int is_final);

} ucx_perf_rte_t;


/**
 * Describes a performance test.
 */
typedef struct ucx_perf_params {
    ucx_perf_api_t         api;             /* Which API to test */
    ucx_perf_cmd_t         command;         /* Command to perform */
    ucx_perf_test_type_t   test_type;       /* Test communication type */
    ucs_thread_mode_t      thread_mode;     /* Thread mode for communication objects */
    unsigned               thread_count;    /* Number of threads in the test program */
    ucs_async_mode_t       async_mode;      /* how async progress and locking is done */
    ucx_perf_wait_mode_t   wait_mode;       /* How to wait */
    uct_memory_type_t      mem_type;        /* memory type */
    unsigned               flags;           /* See ucx_perf_test_flags. */

    size_t                 *msg_size_list;  /* Test message sizes list. The size
                                               of the array is in msg_size_cnt */
    size_t                 msg_size_cnt;    /* Number of message sizes in
                                               message sizes list */
    size_t                 iov_stride;      /* Distance between starting address
                                               of consecutive IOV entries. It is
                                               similar to UCT uct_iov_t type stride */
    size_t                 am_hdr_size;     /* Active message header size (included in message size) */
    size_t                 alignment;       /* Message buffer alignment */
    unsigned               max_outstanding; /* Maximal number of outstanding sends */
    ucx_perf_counter_t     warmup_iter;     /* Number of warm-up iterations */
    ucx_perf_counter_t     max_iter;        /* Iterations limit, 0 - unlimited */
    double                 max_time;        /* Time limit (seconds), 0 - unlimited */
    double                 report_interval; /* Interval at which to call the report callback */

    void                   *rte_group;      /* Opaque RTE group handle */
    ucx_perf_rte_t         *rte;            /* RTE functions used to exchange data */
    void                   *report_arg;     /* Custom argument for report function */

    struct {
        char                   dev_name[UCT_DEVICE_NAME_MAX]; /* Device name to use */
        char                   tl_name[UCT_TL_NAME_MAX];      /* Transport to use */
        uct_perf_data_layout_t data_layout; /* Data layout to use */
        unsigned               fc_window;   /* Window size for flow control <= UCX_PERF_TEST_MAX_FC_WINDOW */
    } uct;

    struct {
        unsigned               nonblocking_mode; /* TBD */
        ucp_perf_datatype_t    send_datatype;
        ucp_perf_datatype_t    recv_datatype;
    } ucp;

} ucx_perf_params_t;


/**
 * Run a UCT performance test.
 */
ucs_status_t ucx_perf_run(ucx_perf_params_t *params, ucx_perf_result_t *result);


END_C_DECLS

#endif /* UCX_PERF_H_ */
