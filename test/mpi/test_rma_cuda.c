/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_cuda_common.h"
#include "test_mpi_tags_def.h"
#include "test_ucp.h"
#include "test_ucx_check_def.h"

#include <ucp/api/ucp.h>
#include <ucs/sys/string.h>

#include <stdio.h>
#include <mpi.h>

typedef enum {
    OP_TYPE_GET,
    OP_TYPE_PUT
} op_type_t;

typedef struct {
    const char *name;
    op_type_t  type;
} operation_t;

typedef struct {
    const operation_t *operation;
    ucp_t             ucp;
} xfer_args_t;

const operation_t operations[] = {{"get", OP_TYPE_GET}, {"put", OP_TYPE_PUT}};

static int is_initiator(int sender, op_type_t op_type)
{
    return (sender && (op_type == OP_TYPE_PUT)) ||
           (!sender && (op_type == OP_TYPE_GET));
}

static void request_wait(ucp_worker_h ucp_worker, void *request)
{
    ucs_status_t ucs_status;

    if (UCS_PTR_IS_PTR(request)) {
        do {
            ucp_worker_progress(ucp_worker);
            ucs_status = ucp_request_check_status(request);
        } while (ucs_status == UCS_INPROGRESS);

        ucp_request_free(request);
    } else {
        ucs_status = UCS_PTR_STATUS(request);
    }

    if (ucs_status != UCS_OK) {
        fprintf(stderr, "failed to wait for request: %s\n",
                ucs_status_string(ucs_status));
    }
}

void initiate_rma(int peer_rank, op_type_t op_type, ucp_worker_h ucp_worker,
                  ucp_ep_h ucp_ep, void *buffer, size_t size)
{
    ucp_request_param_t ucp_request_param = {0};
    int ack                               = 0;
    rkey_t rkey;

    rkey = recv_rkey(peer_rank, ucp_ep);

    if (op_type == OP_TYPE_GET) {
        request_wait(ucp_worker,
                     ucp_get_nbx(ucp_ep, buffer, size, rkey.remote_address,
                                 rkey.rkey, &ucp_request_param));
    } else {
        request_wait(ucp_worker,
                     ucp_put_nbx(ucp_ep, buffer, size, rkey.remote_address,
                                 rkey.rkey, &ucp_request_param));

        request_wait(ucp_worker, ucp_ep_flush_nbx(ucp_ep, &ucp_request_param));
    }

    MPI_Ssend(&ack, 1, MPI_INT, peer_rank, MPI_TAG_ACK, MPI_COMM_WORLD);

    ucp_rkey_destroy(rkey.rkey);
}

void complete_rma(int peer_rank, op_type_t op_type, ucp_context_h ucp_context,
                  ucp_worker_h ucp_worker, void *buffer, size_t size)
{
    int request_complete = 0;
    ucp_mem_h ucp_mem;
    int ack;
    MPI_Request request;

    ucp_mem = send_rkey(peer_rank, buffer, size, ucp_context);

    MPI_Irecv(&ack, 1, MPI_INT, peer_rank, MPI_TAG_ACK, MPI_COMM_WORLD,
              &request);

    while (!request_complete) {
        ucp_worker_progress(ucp_worker);
        MPI_Test(&request, &request_complete, MPI_STATUS_IGNORE);
    }

    ucp_mem_unmap(ucp_context, ucp_mem);
}

static const xfer_args_t *get_xfer_args(const xfer_params_t *params)
{
    return (const xfer_args_t*)params->xfer_args;
}

static const operation_t *get_operation(const xfer_params_t *params)
{
    return get_xfer_args(params)->operation;
}

static ucp_context_h get_ucp_context(const xfer_params_t *params)
{
    return get_xfer_args(params)->ucp.context;
}

static ucp_worker_h get_ucp_worker(const xfer_params_t *params)
{
    return get_xfer_args(params)->ucp.worker;
}

static ucp_ep_h get_ucp_ep(const xfer_params_t *params)
{
    return get_xfer_args(params)->ucp.ep;
}

void ucp_rma(int sender, const xfer_params_t *params)
{
    int peer_rank = (params->rank + 1) % 2;
    int is_sender = params->rank == sender;
    void *buffer  = (void*)(is_sender ? params->send_ptr : params->recv_ptr);
    op_type_t op_type = get_operation(params)->type;

    if (is_initiator(is_sender, op_type)) {
        initiate_rma(peer_rank, op_type, get_ucp_worker(params),
                     get_ucp_ep(params), buffer, params->size);
    } else {
        complete_rma(peer_rank, op_type, get_ucp_context(params),
                     get_ucp_worker(params), buffer, params->size);
    }
}

static void ucp_rma_pingpong(const xfer_params_t *params)
{
    ucp_rma(0, params);
    ucp_rma(1, params);
}

int main(int argc, char **argv)
{
    init_params_h init_params;
    xfer_args_t xfer_args;
    int operation_idx;

    if (test_cuda_init(argc, argv, &init_params) != 0) {
        return -1;
    }

    xfer_args.ucp = create_ucp();

    for (operation_idx = 0; operation_idx < ucs_static_array_size(operations);
         ++operation_idx) {
        xfer_args.operation = operations + operation_idx;
        PRINT_ROOT("\nTesting operation: %s\n", xfer_args.operation->name);
        test_cuda(init_params, ucp_rma_pingpong, &xfer_args);
    }

    destroy_ucp(xfer_args.ucp);
    test_cuda_cleanup(init_params);
    return 0;
}
