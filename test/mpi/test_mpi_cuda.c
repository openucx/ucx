/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_cuda_common.h"

#include <mpi.h>

static void mpi_send_recv(int rank, int sender, CUdeviceptr d_send,
                          CUdeviceptr d_recv, size_t size)
{
    int receiver = (sender + 1) % 2;

    if (rank == sender) {
        MPI_Send((void*)d_send, size, MPI_BYTE, receiver, 0, MPI_COMM_WORLD);
    } else {
        MPI_Recv((void*)d_recv, size, MPI_BYTE, sender, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

static void mpi_pingpong(const xfer_params_t *params)
{
    mpi_send_recv(params->rank, 0, params->send_ptr, params->recv_ptr,
                  params->size);
    mpi_send_recv(params->rank, 1, params->send_ptr, params->recv_ptr,
                  params->size);
}

int main(int argc, char **argv)
{
    init_params_h init_params;

    if (test_cuda_init(argc, argv, &init_params) != 0) {
        return -1;
    }

    test_cuda(init_params, mpi_pingpong, NULL);
    test_cuda_cleanup(init_params);

    return 0;
}
