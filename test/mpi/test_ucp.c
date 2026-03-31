/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_ucp.h"

#include "test_mpi_tags_def.h"
#include "test_ucx_check_def.h"

#include <mpi.h>

static ucp_context_h create_ucp_context()
{
    ucp_params_t params;
    ucp_config_t *config;
    ucp_context_h context;

    params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_NAME;
    params.features   = UCP_FEATURE_RMA | UCP_FEATURE_WAKEUP;
    params.name       = "test_context";
    UCX_CHECK(ucp_config_read(NULL, NULL, &config));
    UCX_CHECK(ucp_init(&params, config, &context));
    ucp_config_release(config);
    return context;
}

static ucp_worker_h create_ucp_worker(ucp_context_h context)
{
    ucp_worker_params_t worker_params;
    ucp_worker_h worker;

    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE |
                                UCP_WORKER_PARAM_FIELD_NAME;
    worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
    worker_params.name        = "test_worker";
    UCX_CHECK(ucp_worker_create(context, &worker_params, &worker));
    return worker;
}

static void send_worker_address(int receiver_rank, ucp_worker_h worker)
{
    ucp_worker_attr_t worker_attr;

    worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_ADDRESS;
    UCX_CHECK(ucp_worker_query(worker, &worker_attr));

    MPI_Send(&worker_attr.address_length, sizeof(worker_attr.address_length),
             MPI_BYTE, receiver_rank, MPI_TAG_WORKER_ADDRESS_LENGTH,
             MPI_COMM_WORLD);

    MPI_Send(worker_attr.address, worker_attr.address_length, MPI_BYTE,
             receiver_rank, MPI_TAG_WORKER_ADDRESS, MPI_COMM_WORLD);

    ucp_worker_release_address(worker, worker_attr.address);
}

static void *recv_worker_address(int sender_rank)
{
    size_t worker_address_length;
    void *worker_address;

    MPI_Recv(&worker_address_length, sizeof(worker_address_length), MPI_BYTE,
             sender_rank, MPI_TAG_WORKER_ADDRESS_LENGTH, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    worker_address = malloc(worker_address_length);
    if (worker_address == NULL) {
        fprintf(stderr, "malloc worker address failed\n");
        exit(EXIT_FAILURE);
    }

    MPI_Recv(worker_address, worker_address_length, MPI_BYTE, sender_rank,
             MPI_TAG_WORKER_ADDRESS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    return worker_address;
}

static ucp_ep_h create_ucp_ep(ucp_worker_h worker)
{
    int rank;
    void *worker_address;
    ucp_ep_params_t ep_params;
    ucp_ep_h ep;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        send_worker_address(1, worker);
        worker_address = recv_worker_address(1);
    } else {
        worker_address = recv_worker_address(0);
        send_worker_address(0, worker);
    }

    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    ep_params.address    = (ucp_address_t*)worker_address;

    UCX_CHECK(ucp_ep_create(worker, &ep_params, &ep));
    free(worker_address);
    return ep;
}

static void destroy_ucp_ep(ucp_ep_h ep)
{
    ucp_request_param_t request_param;

    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    request_param.flags        = UCP_EP_CLOSE_FLAG_FORCE;
    ucp_ep_close_nbx(ep, &request_param);
}

static ucp_mem_h
create_ucp_mem(void *buffer, size_t size, ucp_context_h context)
{
    ucp_mem_map_params_t mem_map_params;
    ucp_mem_h mem;

    mem_map_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    mem_map_params.address    = buffer;
    mem_map_params.length     = size;
    UCX_CHECK(ucp_mem_map(context, &mem_map_params, &mem));
    return mem;
}

ucp_t create_ucp()
{
    ucp_t ucp;
    ucp.context = create_ucp_context();
    ucp.worker  = create_ucp_worker(ucp.context);
    ucp.ep      = create_ucp_ep(ucp.worker);
    return ucp;
}

void destroy_ucp(ucp_t ucp)
{
    destroy_ucp_ep(ucp.ep);
    ucp_worker_destroy(ucp.worker);
    ucp_cleanup(ucp.context);
}

ucp_mem_h
send_rkey(int peer_rank, void *buffer, size_t size, ucp_context_h context)
{
    ucp_memh_buffer_release_params_t memh_buffer_release_params = {};
    uint64_t local_address;
    ucp_mem_h mem;
    void *packed_rkey;
    size_t packed_rkey_size;

    local_address = (uint64_t)buffer;
    MPI_Send(&local_address, sizeof(local_address), MPI_BYTE, peer_rank,
             MPI_TAG_ADDRESS, MPI_COMM_WORLD);

    mem = create_ucp_mem(buffer, size, context);
    UCX_CHECK(ucp_rkey_pack(context, mem, &packed_rkey, &packed_rkey_size));

    MPI_Send(&packed_rkey_size, sizeof(packed_rkey_size), MPI_BYTE, peer_rank,
             MPI_TAG_PACKED_RKEY_SIZE, MPI_COMM_WORLD);

    MPI_Send(packed_rkey, packed_rkey_size, MPI_BYTE, peer_rank,
             MPI_TAG_PACKED_RKEY, MPI_COMM_WORLD);

    ucp_memh_buffer_release(packed_rkey, &memh_buffer_release_params);
    return mem;
}

rkey_t recv_rkey(int peer_rank, ucp_ep_h ep)
{
    rkey_t rkey;
    size_t packed_rkey_size;
    void *packed_rkey;

    MPI_Recv(&rkey.remote_address, sizeof(rkey.remote_address), MPI_BYTE,
             peer_rank, MPI_TAG_ADDRESS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Recv(&packed_rkey_size, sizeof(packed_rkey_size), MPI_BYTE, peer_rank,
             MPI_TAG_PACKED_RKEY_SIZE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    packed_rkey = malloc(packed_rkey_size);
    MPI_Recv(packed_rkey, packed_rkey_size, MPI_BYTE, peer_rank,
             MPI_TAG_PACKED_RKEY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    UCX_CHECK(ucp_ep_rkey_unpack(ep, packed_rkey, &rkey.rkey));
    free(packed_rkey);

    return rkey;
}
