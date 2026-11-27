/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCS_FD_EXCHANGE_H
#define UCS_FD_EXCHANGE_H

#include <ucs/sys/compiler_def.h>
#include <ucs/time/time_def.h>
#include <ucs/type/status.h>

#include <stdint.h>

BEGIN_C_DECLS

/**
 * A file descriptor exchange server hands out descriptors of the current
 * process to other processes of the same user on the same machine, over a Unix
 * domain datagram socket which it creates in a private directory under a shared
 * directory. Unlike pidfd_getfd(), this works between PID namespaces and does
 * not need ptrace permission: the importer only has to see the directory, for
 * example through a volume mounted into several containers. The processes must
 * have the same user and group IDs as the kernel sees them, so containers must
 * not remap them with user namespaces.
 */
typedef struct ucs_fd_exchange_server ucs_fd_exchange_server_t;


/* Wire protocol, visible for tests. A request which carries the magic number
 * but another version or size is answered with an unsupported reply, so that an
 * importer of another version fails at once instead of timing out. */
#define UCS_FD_EXCHANGE_MAGIC   0x78666475u
#define UCS_FD_EXCHANGE_VERSION 1u


typedef struct {
    uint32_t magic;   /* UCS_FD_EXCHANGE_MAGIC */
    uint32_t version; /* UCS_FD_EXCHANGE_VERSION */
    uint64_t key;     /* Key the descriptor was added with */
} ucs_fd_exchange_request_t;


typedef enum {
    UCS_FD_EXCHANGE_REPLY_NOT_FOUND   = 0, /* The key is not served */
    UCS_FD_EXCHANGE_REPLY_FOUND       = 1, /* The descriptor is attached */
    UCS_FD_EXCHANGE_REPLY_UNSUPPORTED = 2  /* Unknown request version or size */
} ucs_fd_exchange_reply_t;


/**
 * Create a file descriptor exchange server.
 *
 * @param [in]  dir       Absolute path of an existing directory, which the
 *                        importing processes see at the same path. The server
 *                        creates a private subdirectory in it for its socket.
 * @param [out] server_p  Filled with the server handle.
 *
 * @return UCS_OK, or an error if the directory is unusable.
 */
ucs_status_t ucs_fd_exchange_server_create(const char *dir,
                                           ucs_fd_exchange_server_t **server_p);


/**
 * Destroy a file descriptor exchange server and remove its socket and
 * directory. All descriptors must have been removed from it.
 *
 * @param [in] server  Server to destroy.
 */
void ucs_fd_exchange_server_destroy(ucs_fd_exchange_server_t *server);


/**
 * @param [in] server  Server handle.
 *
 * @return Socket path which importers pass to @ref ucs_fd_exchange_import.
 */
const char *
ucs_fd_exchange_server_path(const ucs_fd_exchange_server_t *server);


/**
 * Let importers get a copy of a file descriptor. The caller keeps the
 * descriptor open until it is removed from the server.
 *
 * @param [in] server  Server handle.
 * @param [in] key     Key the importers ask for, unique in the server.
 * @param [in] fd      File descriptor to serve.
 *
 * @return UCS_OK, UCS_ERR_ALREADY_EXISTS if the key is already served, or
 *         UCS_ERR_NO_MEMORY.
 */
ucs_status_t ucs_fd_exchange_server_add_fd(ucs_fd_exchange_server_t *server,
                                           uint64_t key, int fd);


/**
 * Stop serving a file descriptor.
 *
 * @param [in] server  Server handle.
 * @param [in] key     Key the descriptor was added with.
 */
void ucs_fd_exchange_server_remove_fd(ucs_fd_exchange_server_t *server,
                                      uint64_t key);


/**
 * Import a file descriptor from a server of another process.
 *
 * @param [in]  path     Server socket path.
 * @param [in]  key      Key the descriptor was added with.
 * @param [in]  timeout  Maximal time to wait for the reply, or
 *                       UCS_TIME_INFINITY.
 * @param [out] fd_p     Filled with a new close-on-exec descriptor, which the
 *                       caller owns.
 *
 * @return UCS_OK, UCS_ERR_UNREACHABLE if the server does not exist,
 *         UCS_ERR_NO_ELEM if it does not serve the key, UCS_ERR_UNSUPPORTED
 *         if it speaks another protocol version, UCS_ERR_TIMED_OUT if it did
 *         not reply in time, or another error.
 */
ucs_status_t ucs_fd_exchange_import(const char *path, uint64_t key,
                                    ucs_time_t timeout, int *fd_p);

END_C_DECLS

#endif
