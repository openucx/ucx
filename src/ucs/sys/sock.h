/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_SOCKET_H
#define UCS_SOCKET_H

#include <ucs/type/status.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <dirent.h>

BEGIN_C_DECLS


/* A string to hold the IP address and port from a sockaddr */
#define UCS_SOCKADDR_STRING_LEN      60

#define UCS_SOCKET_INET_ADDR(_addr)  (((struct sockaddr_in*)(_addr))->sin_addr)
#define UCS_SOCKET_INET_PORT(_addr)  (((struct sockaddr_in*)(_addr))->sin_port)

#define UCS_SOCKET_INET6_ADDR(_addr) (((struct sockaddr_in6*)(_addr))->sin6_addr)
#define UCS_SOCKET_INET6_PORT(_addr) (((struct sockaddr_in6*)(_addr))->sin6_port)


/* Returns an error if the default error handling should be
 * done (the error value will be returned to a caller),
 * otherwise `UCS_OK` */
typedef ucs_status_t (*ucs_socket_io_err_cb_t)(void *arg, int io_errno);


/**
 * Check if the given (interface) flags represent an active interface.
 *
 * @param [in] flags  Interface flags (Can be obtained using getifaddrs
 *                    or from SIOCGIFFLAGS ioctl).
 *
 * @return 1 if true, otherwise 0
 */
int ucs_netif_flags_is_active(unsigned int flags);


/**
 * Perform an ioctl call on the given interface with the given request.
 * Set the result in the ifreq struct.
 *
 * @param [in]  if_name      Interface name to test.
 * @param [in]  request      The request to fulfill.
 * @param [out] if_req       Filled with the requested information.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_netif_ioctl(const char *if_name, unsigned long request,
                             struct ifreq *if_req);


/**
 * Check if the given interface is in an active state.
 *
 * @param [in]  if_name      Interface name to check.
 *
 * @return 1 if true, otherwise 0
 */
int ucs_netif_is_active(const char *if_name);


/**
 * Create a socket.
 *
 * @param [in]   domain     Communication domain (AF_INET/AF_INET6/etc).
 * @param [in]   type       Communication semantics (SOCK_STREAM/SOCK_DGRAM/etc).
 * @param [out]  fd_p       Pointer to created fd.
 *
 * @return UCS_OK on success or UCS_ERR_IO_ERROR on failure.
 */
ucs_status_t ucs_socket_create(int domain, int type, int *fd_p);


/**
 * Set options on socket.
 *
 * @param [in]   fd          Socket fd.
 * @param [in]   level       The level at which the option is defined.
 * @param [in]   optname     The socket option for which the value is to be set.
 * @param [in]   optval      A pointer to the buffer in which the value for the
 *                           requested option is specified.
 * @param [in]   optlen      The size, in bytes, of the buffer pointed to by the
 *                           optval and def_optval parameters.
 *
 * @return UCS_OK on success or UCS_ERR_IO_ERROR on failure
 */
ucs_status_t ucs_socket_setopt(int fd, int level, int optname,
                               const void *optval, socklen_t optlen);


/**
 * Connect the socket referred to by the file descriptor `fd`
 * to the address specified by `dest_addr`.
 *
 * @param [in]  fd                Socket fd.
 * @param [in]  dest_addr         Pointer to destination address.
 *
 * @return UCS_OK on success or UCS_ERR_UNREACHABLE on failure or
 *         UCS_INPROGRESS if operation is in progress.
 */
ucs_status_t ucs_socket_connect(int fd, const struct sockaddr *dest_addr);


/**
 * Check whether the socket referred to by the file descriptor `fd`
 * is connected to a peer or not.
 *
 * @param [in]  fd          Socket fd.
 *
 * @return 1 - connected, 0 - not connected.
 */
int ucs_socket_is_connected(int fd);


/**
 * Returns the maximum possible value for the number of sockets that
 * are ready to be accepted. It maybe either value from the system path
 * or SOMAXCONN value.
 *
 * @return The queue length for completely established sockets
 * waiting to be accepted.
 */
int ucs_socket_max_conn();


/**
 * Non-blocking send operation sends data on the connected (or bound
 * connectionless) socket referred to by the file descriptor `fd`.
 *
 * @param [in]      fd              Socket fd.
 * @param [in]      data            A pointer to a buffer containing the data to
 *                                  be transmitted.
 * @param [in/out]  length_p        The length, in bytes, of the data in buffer
 *                                  pointed to by the `data` parameter. The amount of
 *                                  data transmitted is written to this argument.
 * @param [in]      err_cb          Error callback.
 * @param [in]      err_cb_arg      User's argument for the error callback.
 *
 * @return UCS_OK on success, UCS_ERR_CANCELED if connection closed,
 *         UCS_ERR_NO_PROGRESS if system call was interrupted or
 *         would block, UCS_ERR_IO_ERROR on failure.
 */
ucs_status_t ucs_socket_send_nb(int fd, const void *data, size_t *length_p,
                                ucs_socket_io_err_cb_t err_cb,
                                void *err_cb_arg);


/**
 * Non-blocking receive operation receives data from the connected (or bound
 * connectionless) socket referred to by the file descriptor `fd`.
 *
 * @param [in]      fd              Socket fd.
 * @param [in]      data            A pointer to a buffer to receive the incoming
 *                                  data.
 * @param [in/out]  length_p        The length, in bytes, of the data in buffer
 *                                  pointed to by the `data` parameter. The amount of
 *                                  data received is written to this argument.
 * @param [in]      err_cb          Error callback.
 * @param [in]      err_cb_arg      User's argument for the error callback.
 *
 * @return UCS_OK on success, UCS_ERR_CANCELED if connection closed,
 *         UCS_ERR_NO_PROGRESS if system call was interrupted or
 *         would block, UCS_ERR_IO_ERROR on failure.
 */
ucs_status_t ucs_socket_recv_nb(int fd, void *data, size_t *length_p,
                                ucs_socket_io_err_cb_t err_cb,
                                void *err_cb_arg);


/**
 * Blocking send operation sends data on the connected (or bound connectionless)
 * socket referred to by the file descriptor `fd`.
 *
 * @param [in]      fd              Socket fd.
 * @param [in]      data            A pointer to a buffer containing the data to
 *                                  be transmitted.
 * @param [in/out]  length          The length, in bytes, of the data in buffer
 *                                  pointed to by the `data` parameter.
 * @param [in]      err_cb          Error callback.
 * @param [in]      err_cb_arg      User's argument for the error callback.
 *
 * @return UCS_OK on success, UCS_ERR_CANCELED if connection closed,
 *         UCS_ERR_IO_ERROR on failure.
 */
ucs_status_t ucs_socket_send(int fd, const void *data, size_t length,
                             ucs_socket_io_err_cb_t err_cb,
                             void *err_cb_arg);


/**
 * Non-blocking send operation sends I/O vector on the connected (or bound
 * connectionless) socket referred to by the file descriptor `fd`.
 *
 * @param [in]      fd              Socket fd.
 * @param [in]      iov             A pointer to an array of iovec buffers.
 * @param [in]      iov_cnt         The number of buffers pointed to by
 *                                  the iov parameter.
 * @param [out]     length_p        The amount of data transmitted is written to
 *                                  this argument.
 * @param [in]      err_cb          Error callback.
 * @param [in]      err_cb_arg      User's argument for the error callback.
 *
 * @return UCS_OK on success, UCS_ERR_CANCELED if connection closed,
 *         UCS_ERR_NO_PROGRESS if system call was interrupted or
 *         would block, UCS_ERR_IO_ERROR on failure.
 */
ucs_status_t ucs_socket_sendv_nb(int fd, struct iovec *iov, size_t iov_cnt,
                                 size_t *length_p, ucs_socket_io_err_cb_t err_cb,
                                 void *err_cb_arg);


/**
 * Blocking receive operation receives data from the connected (or bound
 * connectionless) socket referred to by the file descriptor `fd`.
 *
 * @param [in]      fd              Socket fd.
 * @param [in]      data            A pointer to a buffer to receive the incoming
 *                                  data.
 * @param [in/out]  length          The length, in bytes, of the data in buffer
 *                                  pointed to by the `data` paramete.
 * @param [in]      err_cb          Error callback.
 * @param [in]      err_cb_arg      User's argument for the error callback.
 *
 * @return UCS_OK on success, UCS_ERR_CANCELED if connection closed,
 *         UCS_ERR_IO_ERROR on failure.
 */
ucs_status_t ucs_socket_recv(int fd, void *data, size_t length,
                             ucs_socket_io_err_cb_t err_cb,
                             void *err_cb_arg);


/**
 * Return size of a given sockaddr structure.
 * 
 * @param [in]   addr       Pointer to sockaddr structure.
 * @param [out]  size_p     Pointer to variable where size of
 *                          sockaddr_in/sockaddr_in6 structure will be written
 *
 * @return UCS_OK on success or UCS_ERR_INVALID_PARAM on failure.
 */
ucs_status_t ucs_sockaddr_sizeof(const struct sockaddr *addr, size_t *size_p);


/**
 * Return port of a given sockaddr structure.
 * 
 * @param [in]   addr       Pointer to sockaddr structure.
 * @param [out]  port_p     Pointer to variable where port (host notation)
 *                          of sockaddr_in/sockaddr_in6 structure will be written
 *
 * @return UCS_OK on success or UCS_ERR_INVALID_PARAM on failure.
 */
ucs_status_t ucs_sockaddr_get_port(const struct sockaddr *addr, uint16_t *port_p);


/**
 * Set port to a given sockaddr structure.
 * 
 * @param [in]   addr       Pointer to sockaddr structure.
 * @param [in]   port       Port (host notation) that will be written
 *
 * @return UCS_OK on success or UCS_ERR_INVALID_PARAM on failure.
 */
ucs_status_t ucs_sockaddr_set_port(struct sockaddr *addr, uint16_t port);


/**
 * Return IP addr of a given sockaddr structure.
 * 
 * @param [in]   addr       Pointer to sockaddr structure.
 *
 * @return IP address of sockaddr_in/sockaddr_in6 structure
 *         on success or NULL on failure.
 */
const void *ucs_sockaddr_get_inet_addr(const struct sockaddr *addr);


/**
 * Extract the IP address from a given sockaddr and return it as a string.
 *
 * @param [in]   sock_addr   Sockaddr to take IP address from.
 * @param [out]  str         A string filled with the IP address.
 * @param [in]   max_size    Size of a string (considering '\0'-terminated symbol)
 *
 * @return ip_str if the sock_addr has a valid IP address or 'Invalid address'
 *         otherwise.
 */
const char* ucs_sockaddr_str(const struct sockaddr *sock_addr,
                             char *str, size_t max_size);


/**
 * Return a value indicating the relationships between passed sockaddr structures.
 *
 * @param [in]     sa1        Pointer to sockaddr structure #1.
 * @param [in]     sa2        Pointer to sockaddr structure #2.
 * @param [in/out] status_p   Pointer (can be NULL) to a status: UCS_OK on success
 *                            or UCS_ERR_INVALID_PARAM on failure.
 *
 * @return Returns an integral value indicating the relationship between the
 *         socket addresses:
 *         > 0 - the first socket address is greater than the second
 *               socket address;
 *         < 0 - the first socket address is lower than the second
 *               socket address;
 *         = 0 - the socket addresses are equal.
 *         Note: it returns a positive integer value in case of error occured
 *               during comparison.
 */
int ucs_sockaddr_cmp(const struct sockaddr *sa1,
                     const struct sockaddr *sa2,
                     ucs_status_t *status_p);

/**
 * Indicate if given IP addr is INADDR_ANY (IPV4) or in6addr_any (IPV6)
 * 
 * @param [in]   addr       Pointer to sockaddr structure.
 *
 * @return 1 if input is INADDR_ANY or in6addr_any
 *         0 if not
 */
int ucs_sockaddr_is_inaddr_any(struct sockaddr *addr);


/**
 * Copy the src_addr sockaddr to dst_addr sockaddr. The length to copy is
 * the size of the src_addr sockaddr.
 *
 * @param [in] dst_addr  Pointer to destination sockaddr (to copy to).
 * @param [in] src_addr  Pointer to source sockaddr (to copy from).
 *
 * @return UCS_OK on success or UCS_ERR_INVALID_PARAM on failure.
 */
ucs_status_t ucs_sockaddr_copy(struct sockaddr *dst_addr,
                               const struct sockaddr *src_addr);


/**
 * Copy into ifname_name the interface associated the IP on which the socket
 * file descriptor fd is bound on. IPv4 and IPv6 addresses are handled.
 *
 * @param [in]   fd          Socket fd.
 * @param [out]  if_str      A string filled with the interface name.
 * @param [in]   max_strlen  Maximum length of the if_str.
 */
ucs_status_t ucs_sockaddr_get_ifname(int fd, char *ifname_str, size_t max_strlen);

END_C_DECLS

#endif
