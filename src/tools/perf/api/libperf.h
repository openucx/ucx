/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University
*               of Tennessee Research Foundation. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2020-2021.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCX_LIBPERF_H
#define UCX_LIBPERF_H

#include <ucs/sys/compiler.h>

BEGIN_C_DECLS

/** @file libperf.h */

#include <uct/api/uct.h>
#include <ucp/api/ucp.h>


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
    UCX_PERF_TEST_TYPE_PINGPONG_WAIT_MEM,/* Ping-pong mode with
                                            ucp_worker_wait_mem() */
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
    UCT_PERF_DATA_LAYOUT_SHORT_IOV,
    UCT_PERF_DATA_LAYOUT_BCOPY,
    UCT_PERF_DATA_LAYOUT_ZCOPY,
    UCT_PERF_DATA_LAYOUT_LAST
} uct_perf_data_layout_t;


typedef enum {
    UCX_PERF_WAIT_MODE_POLL,         /* Repeatedly call progress */
    UCX_PERF_WAIT_MODE_SLEEP,        /* Go to sleep */
    UCX_PERF_WAIT_MODE_SPIN,         /* Spin without calling progress */
    UCX_PERF_WAIT_MODE_LAST
} ucx_perf_wait_mode_t;


enum ucx_perf_test_flags {
    UCX_PERF_TEST_FLAG_VALIDATE         = UCS_BIT(1), /* Validate data. Affects performance. */
    UCX_PERF_TEST_FLAG_ONE_SIDED        = UCS_BIT(2), /* For tests which involves only one side,
                                                         the responder should not call progress(). */
    UCX_PERF_TEST_FLAG_MAP_NONBLOCK     = UCS_BIT(3), /* Map memory in non-blocking mode */
    UCX_PERF_TEST_FLAG_TAG_WILDCARD     = UCS_BIT(4), /* For tag tests, use wildcard mask */
    UCX_PERF_TEST_FLAG_TAG_UNEXP_PROBE  = UCS_BIT(5), /* For tag tests, use probe to get unexpected receive */
    UCX_PERF_TEST_FLAG_VERBOSE          = UCS_BIT(7), /* Print error messages */
    UCX_PERF_TEST_FLAG_STREAM_RECV_DATA = UCS_BIT(8), /* For stream tests, use recv data API */
    UCX_PERF_TEST_FLAG_FLUSH_EP         = UCS_BIT(9), /* Issue flush on endpoint instead of worker */
    UCX_PERF_TEST_FLAG_WAKEUP           = UCS_BIT(10), /* Create context with wakeup feature enabled */
    UCX_PERF_TEST_FLAG_ERR_HANDLING     = UCS_BIT(11), /* Create UCP eps with error handling support */
    UCX_PERF_TEST_FLAG_LOOPBACK         = UCS_BIT(12), /* Use loopback connection */
    UCX_PERF_TEST_FLAG_PREREG           = UCS_BIT(13), /* Pass pre-registered memory handle */
    UCX_PERF_TEST_FLAG_AM_RECV_COPY     = UCS_BIT(14)  /* Do additional memcopy during AM receive */
};


enum {
    UCT_PERF_TEST_MAX_FC_WINDOW   = 127         /* Maximal flow-control window */
};


/*
 * Active message ID used for performance tests.
 */
#define UCP_PERF_AM_ID 1


/*
 *   +------+                                                        +------+
 *   |HOST 0|-------------------UCP_PERF_AM_ID(AM)------------------>|HOST 1|
 *   +------+        +---------+                  +---------+        +------+
 *      |            |DPU DMN 0|                  |DPU DMN 1|            |
 *      |            +---------+                  +---------+            |
 *      |                 |                            |                 |
 *      |------INIT------>|                            |<------INIT------|
 *      |                 |----------PEER_INIT-------->|                 |
 *      |                 |                            |                 |
 *      |----SEND_REQ---->|                            |<--RECV_REQ(TM)--|
 *      |                 |---------PEER_TX(AM)------->|                 |
 *      |                 |----------(TM msg)--------->|                 |
 *      |<---SEND_CMPL----|                            |---RECV_CMPL---->|
 *      |                 |                            |                 |
 *      |-------FIN------>|                            |<------FIN-------|
 *
 * (AM) used only when AM communication is configured
 * (TM) used only when tag matching communication is configured
 */
#define UCP_FOREACH_PERF_DAEMON_AM_ID(_macro) \
    /* 1) No DPU offloading case: this message is used to transfer data \
     *    between hosts \
     * 2) DPU offloading case: Sender DPU->Receiver DPU, (optional) AM message \
     *    to transfer data between DPU peers when AM communication is used. \
     *    For tag matching this message isn't used */ \
    _macro(UCP_PERF_DAEMON_AM_ID_PEER_TX,   UCP_PERF_AM_ID) \
    /* Host->DPU daemon initialization message */ \
    _macro(UCP_PERF_DAEMON_AM_ID_INIT,      UCP_PERF_AM_ID + 1) \
    /* Host->DPU daemon, initiate data transfer to the remote peer */ \
    _macro(UCP_PERF_DAEMON_AM_ID_SEND_REQ,  UCP_PERF_AM_ID + 2) \
    /* DPU daemon->Host, ack the data tx operation initiated by SEND_REQ */ \
    _macro(UCP_PERF_DAEMON_AM_ID_SEND_CMPL, UCP_PERF_AM_ID + 3) \
    /* Host->DPU daemon, (optional) initiate data receive. Not used when \
     * DPU peers communicate by AM. Will be used by tag matching */ \
    _macro(UCP_PERF_DAEMON_AM_ID_RECV_REQ,  UCP_PERF_AM_ID + 4) \
    /* DPU daemon->Host, ack the receive completion. This message is sent \
     * unsolicited in case of AM communication. For tag matching transport it \
     * confirms the completion of RECV_REQ request */ \
    _macro(UCP_PERF_DAEMON_AM_ID_RECV_CMPL, UCP_PERF_AM_ID + 5) \
    /* Host->DPU daemon finalize message */ \
    _macro(UCP_PERF_DAEMON_AM_ID_FIN,       UCP_PERF_AM_ID + 6) \
    /* Sender DPU->Receiver DPU, message to establish communication between \
     * DPU peers */ \
    _macro(UCP_PERF_DAEMON_AM_ID_PEER_INIT, UCP_PERF_AM_ID + 7)


#define UCP_PERF_DAEMON_ENUMIFY(ID, VALUE) ID = VALUE,


typedef enum {
    UCP_FOREACH_PERF_DAEMON_AM_ID(UCP_PERF_DAEMON_ENUMIFY)
} ucp_perf_daemon_am_id_t;


#define UCT_PERF_TEST_PARAMS_FMT             "%s/%s"
#define UCT_PERF_TEST_PARAMS_ARG(_params)    (_params)->uct.tl_name, \
                                             (_params)->uct.dev_name


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
        double              percentile;
        double              moment_average; /* Average since last report */
        double              total_average;  /* Average of the whole test */
    }
    latency, bandwidth, msgrate;
} ucx_perf_result_t;


typedef void (*ucx_perf_rte_progress_cb_t)(void *arg);

typedef ucs_status_t (*ucx_perf_rte_setup_func_t)(void *arg);
typedef void (*ucx_perf_rte_cleanup_func_t)(void *arg);
typedef unsigned (*ucx_perf_rte_group_size_func_t)(void *rte_group);
typedef unsigned (*ucx_perf_rte_group_index_func_t)(void *rte_group);
typedef void (*ucx_perf_rte_barrier_func_t)(void *rte_group,
                                            ucx_perf_rte_progress_cb_t progress,
                                            void *arg);
typedef void (*ucx_perf_rte_post_vec_func_t)(void *rte_group,
                                             const struct iovec *iovec,
                                             int iovcnt, void **req);
typedef void (*ucx_perf_rte_recv_func_t)(void *rte_group, unsigned src,
                                         void *buffer, size_t max, void *req);
typedef void (*ucx_perf_rte_exchange_vec_func_t)(void *rte_group, void *req);


/**
 * RTE used to bring-up the test
 */
typedef struct ucx_perf_rte {
    /* @return UCS_OK, UCS_ERR_UNSUPPORTED or actual error */
    ucx_perf_rte_setup_func_t        setup;

    /* Cleanup on successfully setup RTE */
    ucx_perf_rte_cleanup_func_t      cleanup;

    /* @return Group size */
    ucx_perf_rte_group_size_func_t   group_size;

    /* @return My index within the group */
    ucx_perf_rte_group_index_func_t  group_index;

    /* Barrier */
    ucx_perf_rte_barrier_func_t      barrier;

    /* Direct modex */
    ucx_perf_rte_post_vec_func_t     post_vec;
    ucx_perf_rte_recv_func_t         recv;
    ucx_perf_rte_exchange_vec_func_t exchange_vec;

    /* List of supported RTE */
    ucs_list_link_t                  list;

} ucx_perf_rte_t;


/**
 * Common report function
 */
typedef void (*ucx_perf_report_func_t)(void *rte_group,
                                       const ucx_perf_result_t *result,
                                       void *arg, const char *extra_info,
                                       int is_final, int is_multi_thread);

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
    ucs_memory_type_t      send_mem_type;   /* Send memory type */
    ucs_memory_type_t      recv_mem_type;   /* Recv memory type */
    unsigned               flags;           /* See ucx_perf_test_flags. */

    size_t                 *msg_size_list;  /* Test message sizes list. The size
                                               of the array is in msg_size_cnt */
    size_t                 msg_size_cnt;    /* Number of message sizes in
                                               message sizes list */
    size_t                 iov_stride;      /* Distance between starting address
                                               of consecutive IOV entries. It is
                                               similar to UCT uct_iov_t type stride */
    size_t                 alignment;       /* Message buffer alignment */
    unsigned               max_outstanding; /* Maximal number of outstanding sends */
    ucx_perf_counter_t     warmup_iter;     /* Number of warm-up iterations */
    double                 warmup_time;     /* Approximately how long to warm-up */
    ucx_perf_counter_t     max_iter;        /* Iterations limit, 0 - unlimited */
    double                 max_time;        /* Time limit (seconds), 0 - unlimited */
    double                 report_interval; /* Interval at which to call the report callback */
    double                 percentile_rank; /* The percentile rank of the percentile reported
                                               in latency tests */

    void                   *rte_group;      /* Opaque RTE group handle */
    ucx_perf_rte_t         *rte;            /* RTE functions used to exchange data */

    ucx_perf_report_func_t report_func;     /* Report function callback */
    void                   *report_arg;     /* Custom argument for report function */

    struct {
        char                   dev_name[UCT_DEVICE_NAME_MAX]; /* Device name to use */
        char                   tl_name[UCT_TL_NAME_MAX];      /* Transport to use */
        char                   md_name[UCT_MD_NAME_MAX];      /* Memory domain name to use */
        uct_perf_data_layout_t data_layout; /* Data layout to use */
        unsigned               fc_window;   /* Window size for flow control <= UCT_PERF_TEST_MAX_FC_WINDOW */
        size_t                 am_hdr_size; /* UCT Active Message header size
                                               (included in message size) */
    } uct;

    struct {
        ucp_perf_datatype_t     send_datatype;
        ucp_perf_datatype_t     recv_datatype;
        size_t                  am_hdr_size; /* UCP Active Message header size
                                                (not included in message size) */
        int                     is_daemon_mode;  /* Whether DPU offloading daemon
                                                    is configured */
        struct sockaddr_storage dmn_local_addr;  /* IP and port of local daemon,
                                                    used to offload communication */
        struct sockaddr_storage dmn_remote_addr; /* IP and port of remote daemon,
                                                    used to offload communication */
    } ucp;

} ucx_perf_params_t;


typedef struct {
    uint64_t addr;
    uint64_t length;
} ucp_perf_daemon_req_t;


/* Allocators for each memory type */
typedef struct ucx_perf_allocator ucx_perf_allocator_t;
extern const ucx_perf_allocator_t* ucx_perf_mem_type_allocators[];


const char *ucp_perf_daemon_am_id_name(ucp_perf_daemon_am_id_t id);


/**
 * Initialize performance testing framework. May be called multiple times.
 */
void ucx_perf_global_init();


/**
 * Run a UCT performance test.
 */
ucs_status_t ucx_perf_run(const ucx_perf_params_t *params,
                          ucx_perf_result_t *result);


END_C_DECLS

#endif /* UCX_PERF_H_ */
