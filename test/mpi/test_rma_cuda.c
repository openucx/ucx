/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_cuda_common.h"

#include <ucp/api/ucp.h>
#include <ucs/sys/string.h>

#include <stdio.h>
#include <mpi.h>

#define MPI_TAG_WORKER_ADDRESS_LENGTH 777
#define MPI_TAG_WORKER_ADDRESS        778
#define MPI_TAG_ADDRESS               779
#define MPI_TAG_PACKED_RKEY_SIZE      780
#define MPI_TAG_PACKED_RKEY           781
#define MPI_TAG_ACK                   782

#define UCX_CHECK(_func) \
    LIB_CHECK(ucs_status_t, _func, UCS_OK, ucs_status_string)

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
    ucp_context_h     ucp_context;
    ucp_worker_h      ucp_worker;
    ucp_ep_h          ucp_ep;
} xfer_args_t;

const operation_t operations[] = {{"get", OP_TYPE_GET}, {"put", OP_TYPE_PUT}};

static ucp_context_h create_ucp_context()
{
    ucp_params_t ucp_params;
    ucp_config_t *ucp_config;
    ucp_context_h ucp_context;

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_NAME;
    ucp_params.features   = UCP_FEATURE_RMA | UCP_FEATURE_WAKEUP;
    ucp_params.name       = "test_rma_cuda_ctx";
    UCX_CHECK(ucp_config_read(NULL, NULL, &ucp_config));
    UCX_CHECK(ucp_init(&ucp_params, ucp_config, &ucp_context));
    return ucp_context;
}

static ucp_worker_h create_ucp_worker(ucp_context_h ucp_context)
{
    ucp_worker_params_t ucp_worker_params;
    ucp_worker_h ucp_worker;

    ucp_worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE |
                                    UCP_WORKER_PARAM_FIELD_NAME;
    ucp_worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    ucp_worker_params.name        = "test_rma_cuda_worker";
    UCX_CHECK(ucp_worker_create(ucp_context, &ucp_worker_params, &ucp_worker));
    return ucp_worker;
}

static void send_worker_address(int receiver_rank, ucp_worker_h ucp_worker)
{
    ucp_worker_attr_t ucp_worker_attr;

    ucp_worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
    UCX_CHECK(ucp_worker_query(ucp_worker, &ucp_worker_attr));

    MPI_Send(&ucp_worker_attr.address_length,
             sizeof(ucp_worker_attr.address_length), MPI_BYTE, receiver_rank,
             MPI_TAG_WORKER_ADDRESS_LENGTH, MPI_COMM_WORLD);

    MPI_Send(ucp_worker_attr.address, ucp_worker_attr.address_length, MPI_BYTE,
             receiver_rank, MPI_TAG_WORKER_ADDRESS, MPI_COMM_WORLD);

    ucp_worker_release_address(ucp_worker, ucp_worker_attr.address);
}

static void *recv_worker_address(int sender_rank)
{
    size_t ucp_worker_address_length;
    void *ucp_worker_address;

    MPI_Recv(&ucp_worker_address_length, sizeof(ucp_worker_address_length),
             MPI_BYTE, sender_rank, MPI_TAG_WORKER_ADDRESS_LENGTH,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    ucp_worker_address = malloc(ucp_worker_address_length);
    MPI_Recv(ucp_worker_address, ucp_worker_address_length, MPI_BYTE,
             sender_rank, MPI_TAG_WORKER_ADDRESS, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    return ucp_worker_address;
}

static ucp_ep_h create_ucp_ep(ucp_worker_h ucp_worker)
{
    int rank;
    void *ucp_worker_address;
    ucp_ep_params_t ucp_ep_params;
    ucp_ep_h ucp_ep;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        send_worker_address(1, ucp_worker);
        ucp_worker_address = recv_worker_address(1);
    } else {
        ucp_worker_address = recv_worker_address(0);
        send_worker_address(0, ucp_worker);
    }

    ucp_ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ucp_ep_params.address    = (ucp_address_t*)ucp_worker_address;

    UCX_CHECK(ucp_ep_create(ucp_worker, &ucp_ep_params, &ucp_ep));
    free(ucp_worker_address);
    return ucp_ep;
}

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
    uint64_t remote_address;
    size_t ucp_packed_rkey_size;
    void *ucp_packed_rkey;
    ucp_rkey_h ucp_rkey;

    MPI_Recv(&remote_address, sizeof(remote_address), MPI_BYTE, peer_rank,
             MPI_TAG_ADDRESS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Recv(&ucp_packed_rkey_size, sizeof(ucp_packed_rkey_size), MPI_BYTE,
             peer_rank, MPI_TAG_PACKED_RKEY_SIZE, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    ucp_packed_rkey = malloc(ucp_packed_rkey_size);
    MPI_Recv(ucp_packed_rkey, ucp_packed_rkey_size, MPI_BYTE, peer_rank,
             MPI_TAG_PACKED_RKEY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    UCX_CHECK(ucp_ep_rkey_unpack(ucp_ep, ucp_packed_rkey, &ucp_rkey));
    free(ucp_packed_rkey);

    if (op_type == OP_TYPE_GET) {
        request_wait(ucp_worker,
                     ucp_get_nbx(ucp_ep, buffer, size, remote_address, ucp_rkey,
                                 &ucp_request_param));
    } else {
        request_wait(ucp_worker,
                     ucp_put_nbx(ucp_ep, buffer, size, remote_address, ucp_rkey,
                                 &ucp_request_param));

        request_wait(ucp_worker, ucp_ep_flush_nbx(ucp_ep, &ucp_request_param));
    }

    MPI_Ssend(&ack, 1, MPI_INT, peer_rank, MPI_TAG_ACK, MPI_COMM_WORLD);

    ucp_rkey_destroy(ucp_rkey);
}

static ucp_mem_h
create_ucp_mem(void *buffer, size_t size, ucp_context_h ucp_context)
{
    ucp_mem_map_params_t ucp_mem_map_params;
    ucp_mem_h ucp_mem;

    ucp_mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                    UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    ucp_mem_map_params.address    = buffer;
    ucp_mem_map_params.length     = size;
    UCX_CHECK(ucp_mem_map(ucp_context, &ucp_mem_map_params, &ucp_mem));
    return ucp_mem;
}

void complete_rma(int peer_rank, op_type_t op_type, ucp_context_h ucp_context,
                  ucp_worker_h ucp_worker, void *buffer, size_t size)
{
    uint64_t local_address = (uint64_t)buffer;
    ucp_memh_buffer_release_params_t ucp_memh_buffer_release_params = {};
    int request_complete                                            = 0;
    ucp_mem_h ucp_mem;
    void *ucp_packed_rkey;
    size_t ucp_packed_rkey_size;
    int ack;
    MPI_Request request;

    MPI_Send(&local_address, sizeof(local_address), MPI_BYTE, peer_rank,
             MPI_TAG_ADDRESS, MPI_COMM_WORLD);

    ucp_mem = create_ucp_mem(buffer, size, ucp_context);
    UCX_CHECK(ucp_rkey_pack(ucp_context, ucp_mem, &ucp_packed_rkey,
                            &ucp_packed_rkey_size));

    MPI_Send(&ucp_packed_rkey_size, sizeof(ucp_packed_rkey_size), MPI_BYTE,
             peer_rank, MPI_TAG_PACKED_RKEY_SIZE, MPI_COMM_WORLD);

    MPI_Send(ucp_packed_rkey, ucp_packed_rkey_size, MPI_BYTE, peer_rank,
             MPI_TAG_PACKED_RKEY, MPI_COMM_WORLD);

    ucp_memh_buffer_release(ucp_packed_rkey, &ucp_memh_buffer_release_params);

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
    return get_xfer_args(params)->ucp_context;
}

static ucp_worker_h get_ucp_worker(const xfer_params_t *params)
{
    return get_xfer_args(params)->ucp_worker;
}

static ucp_ep_h get_ucp_ep(const xfer_params_t *params)
{
    return get_xfer_args(params)->ucp_ep;
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

static void destroy_ucp_ep(ucp_ep_h ucp_ep)
{
    ucp_request_param_t ucp_request_param;

    ucp_request_param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    ucp_request_param.flags        = UCP_EP_CLOSE_FLAG_FORCE;
    ucp_ep_close_nbx(ucp_ep, &ucp_request_param);
}

int main(int argc, char **argv)
{
    init_params_h init_params;
    xfer_args_t xfer_args;
    int operation_idx;

    if (test_cuda_init(argc, argv, &init_params) != 0) {
        return -1;
    }

    xfer_args.ucp_context = create_ucp_context();
    xfer_args.ucp_worker  = create_ucp_worker(xfer_args.ucp_context);
    xfer_args.ucp_ep      = create_ucp_ep(xfer_args.ucp_worker);

    for (operation_idx = 0; operation_idx < ucs_static_array_size(operations);
         ++operation_idx) {
        xfer_args.operation = operations + operation_idx;
        PRINT_ROOT("\nTesting operation: %s\n", xfer_args.operation->name);
        test_cuda(init_params, ucp_rma_pingpong, &xfer_args);
    }

    destroy_ucp_ep(xfer_args.ucp_ep);
    ucp_worker_destroy(xfer_args.ucp_worker);
    ucp_cleanup(xfer_args.ucp_context);
    test_cuda_cleanup(init_params);
    return 0;
}
