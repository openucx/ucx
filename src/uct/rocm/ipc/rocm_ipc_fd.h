/*
 * See file LICENSE for terms.
 */

#ifndef UCT_ROCM_IPC_FD_H
#define UCT_ROCM_IPC_FD_H

#include <ucs/type/status.h>

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>


/*
 * Passes the dma-buf file descriptors of exported VMM allocations to peer
 * processes with SCM_RIGHTS, over an abstract unix socket served from the UCS
 * async thread. Unlike pidfd_getfd, this needs no ptrace access to the
 * exporting process, which Yama restricts to its ancestors by default.
 */


/**
 * Make @a fd, which shares the allocation mapped at [address, address+length),
 * available to peer processes. Takes ownership of @a fd. Exports of the same
 * allocation share one identifier, so that peers map it once; @a fd is closed
 * right away if the allocation is already exported.
 *
 * @param [in]  fd           File descriptor to export.
 * @param [in]  address      Base address of the allocation.
 * @param [in]  length       Length of the allocation.
 * @param [out] server_id_p  Identifies this process' fd server.
 * @param [out] fd_id_p      Identifies the exported allocation.
 */
ucs_status_t uct_rocm_ipc_fd_export(int fd, uintptr_t address, size_t length,
                                    uint64_t *server_id_p, uint64_t *fd_id_p);


/**
 * Release an export from @ref uct_rocm_ipc_fd_export, closing its file
 * descriptor once no registration uses it.
 */
void uct_rocm_ipc_fd_unexport(uint64_t fd_id);


/**
 * Obtain a duplicate of a file descriptor exported by process @a pid.
 *
 * @param [in]  pid        Exporting process.
 * @param [in]  server_id  Fd server of the exporting process.
 * @param [in]  fd_id      Exported file descriptor.
 * @param [out] fd_p       Local file descriptor, to be closed by the caller.
 *
 * @return UCS_ERR_UNREACHABLE if the exporting process cannot be contacted.
 */
ucs_status_t uct_rocm_ipc_fd_import(pid_t pid, uint64_t server_id,
                                    uint64_t fd_id, int *fd_p);

#endif
