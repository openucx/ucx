/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University 
*               of Tennessee Research Foundation. 2015. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2017-2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "perftest.h"

#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/sock.h>
#include <ucs/debug/log.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/poll.h>


test_type_t tests[] = {
    {"am_lat", UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
     "active message latency", "latency", 1},

    {"put_lat", UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
     "put latency", "latency", 1},

    {"add_lat", UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_PINGPONG,
     "atomic add latency", "latency", 1},

    {"get", UCX_PERF_API_UCT, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "get latency / bandwidth / message rate", "latency", 1},

    {"fadd", UCX_PERF_API_UCT, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic fetch-and-add latency / rate", "latency", 1},

    {"swap", UCX_PERF_API_UCT, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic swap latency / rate", "latency", 1},

    {"cswap", UCX_PERF_API_UCT, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic compare-and-swap latency / rate", "latency", 1},

    {"am_bw", UCX_PERF_API_UCT, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "active message bandwidth / message rate", "overhead", 1},

    {"put_bw", UCX_PERF_API_UCT, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "put bandwidth / message rate", "overhead", 1},

    {"add_mr", UCX_PERF_API_UCT, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic add message rate", "overhead", 1},

    {"tag_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_PINGPONG,
     "tag match latency", "latency", 1},

    {"tag_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "tag match bandwidth", "overhead", 32},

    {"tag_sync_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG_SYNC, UCX_PERF_TEST_TYPE_PINGPONG,
     "tag sync match latency", "latency", 1},

    {"tag_sync_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_TAG_SYNC, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "tag sync match bandwidth", "overhead", 32},

    {"ucp_put_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_PINGPONG,
     "put latency", "latency", 1},

    {"ucp_put_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_PUT, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "put bandwidth", "overhead", 32},

    {"ucp_get", UCX_PERF_API_UCP, UCX_PERF_CMD_GET, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "get latency / bandwidth / message rate", "latency", 1},

    {"ucp_add", UCX_PERF_API_UCP, UCX_PERF_CMD_ADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic add bandwidth / message rate", "overhead", 1},

    {"ucp_fadd", UCX_PERF_API_UCP, UCX_PERF_CMD_FADD, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic fetch-and-add latency / bandwidth / rate", "latency", 1},

    {"ucp_swap", UCX_PERF_API_UCP, UCX_PERF_CMD_SWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic swap latency / bandwidth / rate", "latency", 1},

    {"ucp_cswap", UCX_PERF_API_UCP, UCX_PERF_CMD_CSWAP, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "atomic compare-and-swap latency / bandwidth / rate", "latency", 1},

    {"stream_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "stream bandwidth", "overhead", 1},

    {"stream_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_STREAM, UCX_PERF_TEST_TYPE_PINGPONG,
     "stream latency", "latency", 1},

    {"ucp_am_lat", UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_PINGPONG,
     "am latency", "latency", 1},

    {"ucp_am_bw", UCX_PERF_API_UCP, UCX_PERF_CMD_AM, UCX_PERF_TEST_TYPE_STREAM_UNI,
     "am bandwidth / message rate", "overhead", 32},

    {NULL}
};

static int sock_io(int sock, ssize_t (*sock_call)(int, void *, size_t, int),
                   int poll_events, void *data, size_t size,
                   void (*progress)(void *arg), void *arg, const char *name)
{
    size_t total = 0;
    struct pollfd pfd;
    int ret;

    while (total < size) {
        pfd.fd      = sock;
        pfd.events  = poll_events;
        pfd.revents = 0;

        ret = poll(&pfd, 1, 1); /* poll for 1ms */
        if (ret > 0) {
            ucs_assert(ret == 1);
            ucs_assert(pfd.revents & poll_events);

            ret = sock_call(sock, (char*)data + total, size - total, 0);
            if (ret < 0) {
                ucs_error("%s() failed: %m", name);
                return -1;
            }
            total += ret;
        } else if ((ret < 0) && (errno != EINTR)) {
            ucs_error("poll(fd=%d) failed: %m", sock);
            return -1;
        }

        /* progress user context */
        if (progress != NULL) {
            progress(arg);
        }
    }
    return 0;
}

static int safe_send(int sock, void *data, size_t size,
                     void (*progress)(void *arg), void *arg)
{
    typedef ssize_t (*sock_call)(int, void *, size_t, int);

    return sock_io(sock, (sock_call)send, POLLOUT, data, size, progress, arg, "send");
}

static int safe_recv(int sock, void *data, size_t size,
                     void (*progress)(void *arg), void *arg)
{
    return sock_io(sock, recv, POLLIN, data, size, progress, arg, "recv");
}

ucs_status_t init_test_params(perftest_params_t *params)
{
    memset(params, 0, sizeof(*params));
    params->super.api               = UCX_PERF_API_LAST;
    params->super.command           = UCX_PERF_CMD_LAST;
    params->super.test_type         = UCX_PERF_TEST_TYPE_LAST;
    params->super.thread_mode       = UCS_THREAD_MODE_SINGLE;
    params->super.thread_count      = 1;
    params->super.async_mode        = UCS_ASYNC_THREAD_LOCK_TYPE;
    params->super.wait_mode         = UCX_PERF_WAIT_MODE_LAST;
    params->super.max_outstanding   = 0;
    params->super.warmup_iter       = 10000;
    params->super.alignment         = ucs_get_page_size();
    params->super.max_iter          = 1000000l;
    params->super.max_time          = 0.0;
    params->super.report_interval   = 1.0;
    params->super.percentile_rank   = 50.0;
    params->super.flags             = UCX_PERF_TEST_FLAG_VERBOSE;
    params->super.uct.fc_window     = UCT_PERF_TEST_MAX_FC_WINDOW;
    params->super.uct.data_layout   = UCT_PERF_DATA_LAYOUT_SHORT;
    params->super.uct.am_hdr_size   = 8;
    params->super.send_mem_type     = UCS_MEMORY_TYPE_HOST;
    params->super.recv_mem_type     = UCS_MEMORY_TYPE_HOST;
    params->super.msg_size_cnt      = 1;
    params->super.iov_stride        = 0;
    params->super.ucp.send_datatype = UCP_PERF_DATATYPE_CONTIG;
    params->super.ucp.recv_datatype = UCP_PERF_DATATYPE_CONTIG;
    params->super.ucp.am_hdr_size   = 0;
    strcpy(params->super.uct.dev_name, TL_RESOURCE_NAME_NONE);
    strcpy(params->super.uct.tl_name,  TL_RESOURCE_NAME_NONE);

    params->super.msg_size_list = calloc(params->super.msg_size_cnt,
                                         sizeof(*params->super.msg_size_list));
    if (params->super.msg_size_list == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    params->super.msg_size_list[0] = 8;
    params->test_id                = TEST_ID_UNDEFINED;

    return UCS_OK;
}

static unsigned sock_rte_group_size(void *rte_group)
{
    return 2;
}

static unsigned sock_rte_group_index(void *rte_group)
{
    sock_rte_group_t *group = rte_group;
    return group->is_server ? 0 : 1;
}

static void sock_rte_barrier(void *rte_group, void (*progress)(void *arg),
                             void *arg)
{
#pragma omp barrier

#pragma omp master
  {
    sock_rte_group_t *group = rte_group;
    const unsigned magic = 0xdeadbeef;
    unsigned snc;

    snc = magic;
    safe_send(group->connfd, &snc, sizeof(unsigned), progress, arg);

    snc = 0;
    
    if (safe_recv(group->connfd, &snc, sizeof(unsigned), progress, arg) == 0) {
        ucs_assert(snc == magic);
    }
  }
#pragma omp barrier
}

static void sock_rte_post_vec(void *rte_group, const struct iovec *iovec,
                              int iovcnt, void **req)
{
    sock_rte_group_t *group = rte_group;
    size_t size;
    int i;

    size = 0;
    for (i = 0; i < iovcnt; ++i) {
        size += iovec[i].iov_len;
    }

    safe_send(group->connfd, &size, sizeof(size), NULL, NULL);
    for (i = 0; i < iovcnt; ++i) {
        safe_send(group->connfd, iovec[i].iov_base, iovec[i].iov_len, NULL,
                  NULL);
    }
}

static void sock_rte_recv(void *rte_group, unsigned src, void *buffer,
                          size_t max, void *req)
{
    sock_rte_group_t *group = rte_group;
    int group_index;
    size_t size;

    group_index = sock_rte_group_index(rte_group);
    if (src == group_index) {
        return;
    }

    ucs_assert_always(src == (1 - group_index));
    safe_recv(group->connfd, &size, sizeof(size), NULL, NULL);
    ucs_assert_always(size <= max);
    safe_recv(group->connfd, buffer, size, NULL, NULL);
}

static void sock_rte_report(void *rte_group, const ucx_perf_result_t *result,
                            void *arg, int is_final, int is_multi_thread)
{
    struct perftest_context *ctx = arg;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags,
                   is_final, ctx->server_addr == NULL, is_multi_thread);
}

static ucx_perf_rte_t sock_rte = {
    .group_size    = sock_rte_group_size,
    .group_index   = sock_rte_group_index,
    .barrier       = sock_rte_barrier,
    .post_vec      = sock_rte_post_vec,
    .recv          = sock_rte_recv,
    .exchange_vec  = (ucx_perf_rte_exchange_vec_func_t)ucs_empty_function,
    .report        = sock_rte_report,
};

static ucs_status_t setup_sock_rte(struct perftest_context *ctx)
{
    struct sockaddr_in inaddr;
    struct hostent *he;
    ucs_status_t status;
    int optval = 1;
    int sockfd, connfd;
    int ret;

    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        ucs_error("socket() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    if (ctx->server_addr == NULL) {
        optval = 1;
        status = ucs_socket_setopt(sockfd, SOL_SOCKET, SO_REUSEADDR,
                                   &optval, sizeof(optval));
        if (status != UCS_OK) {
            goto err_close_sockfd;
        }

        inaddr.sin_family      = AF_INET;
        inaddr.sin_port        = htons(ctx->port);
        inaddr.sin_addr.s_addr = INADDR_ANY;
        memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));
        ret = bind(sockfd, (struct sockaddr*)&inaddr, sizeof(inaddr));
        if (ret < 0) {
            ucs_error("bind() failed: %m");
            status = UCS_ERR_INVALID_ADDR;
            goto err_close_sockfd;
        }

        ret = listen(sockfd, 10);
        if (ret < 0) {
            ucs_error("listen() failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_close_sockfd;
        }

        printf("Waiting for connection...\n");

        /* Accept next connection */
        connfd = accept(sockfd, NULL, NULL);
        if (connfd < 0) {
            ucs_error("accept() failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_close_sockfd;
        }

        close(sockfd);

        /* release the memory for the list of the message sizes allocated
         * during the initialization of the default testing parameters */
        free(ctx->params.super.msg_size_list);
        ctx->params.super.msg_size_list = NULL;

        ret = safe_recv(connfd, &ctx->params, sizeof(ctx->params), NULL, NULL);
        if (ret) {
            status = UCS_ERR_IO_ERROR;
            goto err_close_connfd;
        }

        if (ctx->params.super.msg_size_cnt != 0) {
            ctx->params.super.msg_size_list =
                    calloc(ctx->params.super.msg_size_cnt,
                           sizeof(*ctx->params.super.msg_size_list));
            if (NULL == ctx->params.super.msg_size_list) {
                status = UCS_ERR_NO_MEMORY;
                goto err_close_connfd;
            }

            ret = safe_recv(connfd, ctx->params.super.msg_size_list,
                            sizeof(*ctx->params.super.msg_size_list) *
                            ctx->params.super.msg_size_cnt,
                            NULL, NULL);
            if (ret) {
                status = UCS_ERR_IO_ERROR;
                goto err_close_connfd;
            }
        }

        ctx->sock_rte_group.connfd    = connfd;
        ctx->sock_rte_group.is_server = 1;
    } else {
        he = gethostbyname(ctx->server_addr);
        if (he == NULL || he->h_addr_list == NULL) {
            ucs_error("host %s not found: %s", ctx->server_addr,
                      hstrerror(h_errno));
            status = UCS_ERR_INVALID_ADDR;
            goto err_close_sockfd;
        }

        inaddr.sin_family = he->h_addrtype;
        inaddr.sin_port   = htons(ctx->port);
        ucs_assert(he->h_length == sizeof(inaddr.sin_addr));
        memcpy(&inaddr.sin_addr, he->h_addr_list[0], he->h_length);
        memset(inaddr.sin_zero, 0, sizeof(inaddr.sin_zero));

        ret = connect(sockfd, (struct sockaddr*)&inaddr, sizeof(inaddr));
        if (ret < 0) {
            ucs_error("connect() failed: %m");
            status = UCS_ERR_UNREACHABLE;
            goto err_close_sockfd;
        }

        safe_send(sockfd, &ctx->params, sizeof(ctx->params), NULL, NULL);
        if (ctx->params.super.msg_size_cnt != 0) {
            safe_send(sockfd, ctx->params.super.msg_size_list,
                      sizeof(*ctx->params.super.msg_size_list) *
                      ctx->params.super.msg_size_cnt,
                      NULL, NULL);
        }

        ctx->sock_rte_group.connfd    = sockfd;
        ctx->sock_rte_group.is_server = 0;
    }

    if (ctx->sock_rte_group.is_server) {
        ctx->flags |= TEST_FLAG_PRINT_TEST;
    } else {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->params.super.rte_group  = &ctx->sock_rte_group;
    ctx->params.super.rte        = &sock_rte;
    ctx->params.super.report_arg = ctx;
    return UCS_OK;

err_close_connfd:
    close(connfd);
    goto err;
err_close_sockfd:
    close(sockfd);
err:
    return status;
}

static ucs_status_t cleanup_sock_rte(struct perftest_context *ctx)
{
    close(ctx->sock_rte_group.connfd);
    return UCS_OK;
}

#if defined (HAVE_MPI)
static unsigned mpi_rte_group_size(void *rte_group)
{
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

static unsigned mpi_rte_group_index(void *rte_group)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

static void mpi_rte_barrier(void *rte_group, void (*progress)(void *arg),
                            void *arg)
{
    int group_size, my_rank, i;
    MPI_Request *reqs;
    int nreqs = 0;
    int dummy;
    int flag;

#pragma omp barrier

#pragma omp master
  {
    /*
     * Naive non-blocking barrier implementation over send/recv, to call user
     * progress while waiting for completion.
     * Not using MPI_Ibarrier to be compatible with MPI-1.
     */

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    /* allocate maximal possible number of requests */
    reqs = (MPI_Request*)alloca(sizeof(*reqs) * group_size);

    if (my_rank == 0) {
        /* root gathers "ping" from all other ranks */
        for (i = 1; i < group_size; ++i) {
            MPI_Irecv(&dummy, 0, MPI_INT,
                      i /* source */,
                      1 /* tag */,
                      MPI_COMM_WORLD,
                      &reqs[nreqs++]);
        }
    } else {
        /* every non-root rank sends "ping" and waits for "pong" */
        MPI_Send(&dummy, 0, MPI_INT,
                 0 /* dest */,
                 1 /* tag */,
                 MPI_COMM_WORLD);
        MPI_Irecv(&dummy, 0, MPI_INT,
                  0 /* source */,
                  2 /* tag */,
                  MPI_COMM_WORLD,
                  &reqs[nreqs++]);
    }

    /* Waiting for receive requests */
    do {
        MPI_Testall(nreqs, reqs, &flag, MPI_STATUSES_IGNORE);
        progress(arg);
    } while (!flag);

    if (my_rank == 0) {
        /* root sends "pong" to all ranks */
        for (i = 1; i < group_size; ++i) {
            MPI_Send(&dummy, 0, MPI_INT,
                     i /* dest */,
                     2 /* tag */,
                     MPI_COMM_WORLD);
       }
    }
  }
#pragma omp barrier
}

static void mpi_rte_post_vec(void *rte_group, const struct iovec *iovec,
                             int iovcnt, void **req)
{
    int group_size;
    int my_rank;
    int dest, i;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &group_size);

    for (dest = 0; dest < group_size; ++dest) {
        if (dest == my_rank) {
            continue;
        }

        for (i = 0; i < iovcnt; ++i) {
            MPI_Send(iovec[i].iov_base, iovec[i].iov_len, MPI_BYTE, dest,
                     i == (iovcnt - 1), /* Send last iov with tag == 1 */
                     MPI_COMM_WORLD);
        }
    }

    *req = (void*)(uintptr_t)1;
}

static void mpi_rte_recv(void *rte_group, unsigned src, void *buffer, size_t max,
                         void *req)
{
    MPI_Status status;
    size_t offset;
    int my_rank;
    int count;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    if (src == my_rank) {
        return;
    }

    offset = 0;
    do {
        ucs_assert_always(offset < max);
        MPI_Recv(buffer + offset, max - offset, MPI_BYTE, src, MPI_ANY_TAG,
                 MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_BYTE, &count);
        offset += count;
    } while (status.MPI_TAG != 1);
}

static void mpi_rte_report(void *rte_group, const ucx_perf_result_t *result,
                           void *arg, int is_final, int is_multi_thread)
{
    struct perftest_context *ctx = arg;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags,
                   is_final, ctx->server_addr == NULL, is_multi_thread);
}
#elif defined (HAVE_RTE)
static unsigned ext_rte_group_size(void *rte_group)
{
    rte_group_t group = (rte_group_t)rte_group;
    return rte_group_size(group);
}

static unsigned ext_rte_group_index(void *rte_group)
{
    rte_group_t group = (rte_group_t)rte_group;
    return rte_group_rank(group);
}

static void ext_rte_barrier(void *rte_group, void (*progress)(void *arg),
                            void *arg)
{
#pragma omp barrier

#pragma omp master
  {
    rte_group_t group = (rte_group_t)rte_group;
    int rc;

    rc = rte_barrier(group);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_barrier");
    }
  }
#pragma omp barrier
}

static void ext_rte_post_vec(void *rte_group, const struct iovec* iovec,
                             int iovcnt, void **req)
{
    rte_group_t group = (rte_group_t)rte_group;
    rte_srs_session_t session;
    rte_iovec_t *r_vec;
    int i, rc;

    rc = rte_srs_session_create(group, 0, &session);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_session_create");
    }

    r_vec = calloc(iovcnt, sizeof(rte_iovec_t));
    if (r_vec == NULL) {
        return;
    }
    for (i = 0; i < iovcnt; ++i) {
        r_vec[i].iov_base = iovec[i].iov_base;
        r_vec[i].type     = rte_datatype_uint8_t;
        r_vec[i].count    = iovec[i].iov_len;
    }
    rc = rte_srs_set_data(session, "KEY_PERF", r_vec, iovcnt);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_set_data");
    }
    *req = session;
    free(r_vec);
}

static void ext_rte_recv(void *rte_group, unsigned src, void *buffer,
                         size_t max, void *req)
{
    rte_group_t group         = (rte_group_t)rte_group;
    rte_srs_session_t session = (rte_srs_session_t)req;
    void *rte_buffer = NULL;
    rte_iovec_t r_vec;
    uint32_t offset;
    int size;
    int rc;

    rc = rte_srs_get_data(session, rte_group_index_to_ec(group, src),
                          "KEY_PERF", &rte_buffer, &size);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_get_data");
        return;
    }

    r_vec.iov_base = buffer;
    r_vec.type     = rte_datatype_uint8_t;
    r_vec.count    = max;

    offset = 0;
    rte_unpack(&r_vec, rte_buffer, &offset);

    rc = rte_srs_session_destroy(session);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_session_destroy");
    }
    free(rte_buffer);
}

static void ext_rte_exchange_vec(void *rte_group, void * req)
{
    rte_srs_session_t session = (rte_srs_session_t)req;
    int rc;

    rc = rte_srs_exchange_data(session);
    if (RTE_SUCCESS != rc) {
        ucs_error("Failed to rte_srs_exchange_data");
    }
}

static void ext_rte_report(void *rte_group, const ucx_perf_result_t *result,
                           void *arg, int is_final, int is_multi_thread)
{
    struct perftest_context *ctx = arg;
    print_progress(ctx->test_names, ctx->num_batch_files, result, ctx->flags,
                   is_final, ctx->server_addr == NULL, is_multi_thread);
}

static ucx_perf_rte_t ext_rte = {
    .group_size    = ext_rte_group_size,
    .group_index   = ext_rte_group_index,
    .barrier       = ext_rte_barrier,
    .report        = ext_rte_report,
    .post_vec      = ext_rte_post_vec,
    .recv          = ext_rte_recv,
    .exchange_vec  = ext_rte_exchange_vec,
};
#endif

static ucs_status_t setup_mpi_rte(struct perftest_context *ctx)
{
#if defined (HAVE_MPI)
    static ucx_perf_rte_t mpi_rte = {
        .group_size    = mpi_rte_group_size,
        .group_index   = mpi_rte_group_index,
        .barrier       = mpi_rte_barrier,
        .post_vec      = mpi_rte_post_vec,
        .recv          = mpi_rte_recv,
        .exchange_vec  = (void*)ucs_empty_function,
        .report        = mpi_rte_report,
    };

    int size, rank;

    ucs_trace_func("");

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        ucs_error("This test should run with exactly 2 processes (actual: %d)", size);
        return UCS_ERR_INVALID_PARAM;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 1) {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->params.super.rte_group  = NULL;
    ctx->params.super.rte        = &mpi_rte;
    ctx->params.super.report_arg = ctx;
#elif defined (HAVE_RTE)
    ucs_trace_func("");
    
    ctx->params.rte_group         = NULL;
    ctx->params.rte               = &mpi_rte;
    ctx->params.report_arg        = ctx;
    rte_group_t group;

    rte_init(NULL, NULL, &group);
    if (1 == rte_group_rank(group)) {
        ctx->flags |= TEST_FLAG_PRINT_RESULTS;
    }

    ctx->params.super.rte_group  = group;
    ctx->params.super.rte        = &ext_rte;
    ctx->params.super.report_arg = ctx;
#endif
    return UCS_OK;
}

static ucs_status_t cleanup_mpi_rte(struct perftest_context *ctx)
{
#ifdef HAVE_RTE
    rte_finalize();
#endif
    return UCS_OK;
}

static ucs_status_t check_system(struct perftest_context *ctx)
{
    ucs_sys_cpuset_t cpuset;
    unsigned i, count, nr_cpus;
    int ret;

    ucs_trace_func("");

    ret = ucs_sys_get_num_cpus();
    if (ret < 0) {
        return UCS_ERR_INVALID_PARAM;
    }
    nr_cpus = ret;

    memset(&cpuset, 0, sizeof(cpuset));
    if (ctx->flags & TEST_FLAG_SET_AFFINITY) {
        for (i = 0; i < ctx->num_cpus; i++) {
            if (ctx->cpus[i] >= nr_cpus) {
                ucs_error("cpu (%u) out of range (0..%u)", ctx->cpus[i], nr_cpus - 1);
                return UCS_ERR_INVALID_PARAM;
            }
        }

        for (i = 0; i < ctx->num_cpus; i++) {
            CPU_SET(ctx->cpus[i], &cpuset);
        }

        ret = ucs_sys_setaffinity(&cpuset);
        if (ret) {
            ucs_warn("sched_setaffinity() failed: %m");
            return UCS_ERR_INVALID_PARAM;
        }
    } else {
        ret = ucs_sys_getaffinity(&cpuset);
        if (ret) {
            ucs_warn("sched_getaffinity() failed: %m");
            return UCS_ERR_INVALID_PARAM;
        }

        count = 0;
        for (i = 0; i < CPU_SETSIZE; ++i) {
            if (CPU_ISSET(i, &cpuset)) {
                ++count;
            }
        }
        if (count > 2) {
            ucs_warn("CPU affinity is not set (bound to %u cpus)."
                     " Performance may be impacted.", count);
        }
    }

    return UCS_OK;
}

int main(int argc, char **argv)
{
    struct perftest_context ctx;
    ucs_status_t status;
    int mpi_initialized;
    int mpi_rte;
    int ret;

#ifdef HAVE_MPI
    int provided;

    mpi_initialized = !isatty(0) &&
                      /* Using MPI_THREAD_FUNNELED since ucx_perftest supports
                       * using multiple threads when only the main one makes
                       * MPI calls (which is also suitable for a single threaded
                       * run).
                       * MPI_THREAD_FUNNELED:
                       * The process may be multi-threaded, but only the main
                       * thread will make MPI calls (all MPI calls are funneled
                       * to the main thread). */
                      (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) == 0);

    if (mpi_initialized && (provided != MPI_THREAD_FUNNELED)) {
        printf("MPI_Init_thread failed to set MPI_THREAD_FUNNELED. (provided = %d)\n",
               provided);
        ret = -1;
        goto out;
    }
#else
    mpi_initialized = 0;
#endif

    /* Parse command line */
    status = parse_opts(&ctx, mpi_initialized, argc, argv);
    if (status != UCS_OK) {
        ret = (status == UCS_ERR_CANCELED) ? 0 : -127;
        goto out_msg_size_list;
    }

#ifdef __COVERITY__
    /* coverity[dont_call] */
    mpi_rte = rand(); /* Shut up deadcode error */
#endif

    if (ctx.mpi) {
        mpi_rte = 1;
    } else {
#ifdef HAVE_RTE
        mpi_rte = 1;
#else
        mpi_rte = 0;
#endif
    }

    status = check_system(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out_msg_size_list;
    }

    /* Create RTE */
    status = (mpi_rte) ? setup_mpi_rte(&ctx) : setup_sock_rte(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out_msg_size_list;
    }

    /* Run the test */
    status = run_test(&ctx);
    if (status != UCS_OK) {
        ret = -1;
        goto out_cleanup_rte;
    }

    ret = 0;

out_cleanup_rte:
    (mpi_rte) ? cleanup_mpi_rte(&ctx) : cleanup_sock_rte(&ctx);
out_msg_size_list:
    free(ctx.params.super.msg_size_list);
#if HAVE_MPI
out:
#endif
    if (mpi_initialized) {
#ifdef HAVE_MPI
        MPI_Finalize();
#endif
    }
    return ret;
}
