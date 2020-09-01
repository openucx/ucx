/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_SOCKET_H
#define UCS_SOCKET_H

#include <ucs/type/status.h>
#include <ucs/config/parser.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <dirent.h>

BEGIN_C_DECLS


#define UCS_IPV4_ADDR_LEN            sizeof(struct in_addr)
#define UCS_IPV6_ADDR_LEN            sizeof(struct in6_addr)

/* A string to hold the IP address and port from a sockaddr */
#define UCS_SOCKADDR_STRING_LEN      60

#define UCS_SOCKET_INET_ADDR(_addr)  (((struct sockaddr_in*)(_addr))->sin_addr)
#define UCS_SOCKET_INET_PORT(_addr)  (((struct sockaddr_in*)(_addr))->sin_port)

#define UCS_SOCKET_INET6_ADDR(_addr) (((struct sockaddr_in6*)(_addr))->sin6_addr)
#define UCS_SOCKET_INET6_PORT(_addr) (((struct sockaddr_in6*)(_addr))->sin6_port)


/**
 * Close the given file descriptor.
 *
 * @param [in] fd_p   pointer to the file descriptor to close.
 */
void ucs_close_fd(int *fd_p);


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
 * Get number of active 802.3ad ports for a bond device. If the device is not
 * a bond device, or 802.3ad is not enabled, return 1.
 *
 * @param [in]  if_name      Name of network interface to check.
 *
 * @return Number of active 802.3ad ports on @a if_name.
 */
unsigned ucs_netif_bond_ad_num_ports(const char *if_name);


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
 * Get options of a socket.
 *
 * @param [in]   fd          Socket fd.
 * @param [in]   level       The level at which the option is defined.
 * @param [in]   optname     The socket option for which the value is fetched.
 * @param [in]   optval      A pointer to the buffer in which the value for the
 *                           requested option is stored.
 * @param [in]   optlen      The size, in bytes, of optval.
 *
 * @return UCS_OK on success or UCS_ERR_IO_ERROR on failure
 */
ucs_status_t ucs_socket_getopt(int fd, int level, int optname,
                               void *optval, socklen_t optlen);


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
 * Accept a connection request on the given socket fd.
 *
 * @param [in]  fd                Socket fd.
 * @param [out] addr              Client socket address that initiated the connection
 * @param [out] length_ptr        Client address socket's length
 * @param [out] accept_fd         Upon success, a non-negative file descriptor
 *                                of the accepted socket. Otherwise, -1.
 *
 * @return UCS_OK on success or UCS_ERR_NO_PROGRESS to indicate that no progress
 *         was made or UCS_ERR_IO_ERROR on failure.
 */
ucs_status_t ucs_socket_accept(int fd, struct sockaddr *addr, socklen_t *length_ptr,
                               int *accept_fd);


/**
 * Get the address of the peer's socket that the given fd is connected to
 *
 * @param [in]  fd                Socket fd.
 * @param [out] peer_addr         Address of the remote peer.
 * @param [out] peer_addr_len     Length of the remote peer's address.
 *
 * @return UCS_OK on success or UCS_ERR_IO_ERROR on failure
 */
ucs_status_t ucs_socket_getpeername(int fd, struct sockaddr_storage *peer_addr,
                                    socklen_t *peer_addr_len);


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
 * Set options on a socket for its send and receive buffers.
 * Set the options only if the given buffers sizes are not set to UCS_MEMUNITS_AUTO.
 *
 * @param [in]  fd                 Socket fd.
 * @param [in]  sockopt_sndbuf     Send buffer in which the value for the
 *                                 option is specified.
 * @param [in]  sockopt_rcvbuf     Receive buffer in which the value for the
 *                                 option is specified.
 *
 * @return UCS_OK on success or UCS_ERR_IO_ERROR on failure.
 */
ucs_status_t ucs_socket_set_buffer_size(int fd, size_t sockopt_sndbuf,
                                        size_t sockopt_rcvbuf);


/**
 * Initialize a TCP server.
 * Open a socket, bind a sockadrr to that socket and start listening on it for
 * incoming connection requests.
 *
 * @param [in]  saddr             Sockaddr for the server to listen on.
 *                                If the port number inside is set to zero -
 *                                use a random port.
 * @param [in]  socklen           Size of saddr.
 * @param [in]  backlog           Length of the queue for pending connections -
 *                                for the listen() call.
 * @param [in]  silent_bind       Whether or not to print error message on bind
 *                                failure with EADDRINUSE.
 * @param [in]  allow_addr_inuse  Whether or not to allow the socket to use an
 *                                address that is already in use and was not
 *                                released by another socket yet.
 * @param [out] listen_fd         The fd that belongs to the server.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_socket_server_init(const struct sockaddr *saddr, socklen_t socklen,
                                    int backlog, int silent_bind, int allow_addr_inuse,
                                    int *listen_fd);


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
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_socket_send_nb(int fd, const void *data, size_t *length_p);


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
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_socket_recv_nb(int fd, void *data, size_t *length_p);


/**
 * Blocking send operation sends data on the connected (or bound connectionless)
 * socket referred to by the file descriptor `fd`.
 *
 * @param [in]      fd              Socket fd.
 * @param [in]      data            A pointer to a buffer containing the data to
 *                                  be transmitted.
 * @param [in/out]  length          The length, in bytes, of the data in buffer
 *                                  pointed to by the `data` parameter.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_socket_send(int fd, const void *data, size_t length);


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
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_socket_sendv_nb(int fd, struct iovec *iov, size_t iov_cnt,
                                 size_t *length_p);


/**
 * Blocking receive operation receives data from the connected (or bound
 * connectionless) socket referred to by the file descriptor `fd`.
 *
 * @param [in]      fd              Socket fd.
 * @param [in]      data            A pointer to a buffer to receive the incoming
 *                                  data.
 * @param [in/out]  length          The length, in bytes, of the data in buffer
 *                                  pointed to by the `data` paramete.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_socket_recv(int fd, void *data, size_t length);


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
 * Extract the IP address from a given socket fd and return it as a string.
 *
 * @param [in]   fd          Socket fd.
 * @param [out]  str         A string filled with the IP address.
 * @param [in]   max_size    Size of a string (considering '\0'-terminated symbol)
 *
 * @return ip_str if the sock_addr has a valid IP address or 'Invalid address'
 *         otherwise.
 */
const char *ucs_socket_getname_str(int fd, char *str, size_t max_size);


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
 * Check if the IP addresses of the given sockaddrs are the same.
 *
 * @param [in] sa1        Pointer to sockaddr structure #1.
 * @param [in] sa2        Pointer to sockaddr structure #2.
 *
 * @return Return 0 if the IP addresses are the same and a non-zero value
 *         otherwise.
 */
int ucs_sockaddr_ip_cmp(const struct sockaddr *sa1, const struct sockaddr *sa2);


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


/**
 * Convert the given address family to a string containing its value.
 *
 * @param [in]   af          Address family to convert.
 *
 * Only IPv4 and IPv6 conversions are supported.
 */
const char *ucs_sockaddr_address_family_str(sa_family_t af);


/**
 * Get the local ports range.
 *
 * @param [out]  port_range     Pointer to a struct holding the ports range.
 *
 * @return UCS_OK or error in case of failure.
 */
ucs_status_t ucs_sockaddr_get_ip_local_port_range(ucs_range_spec_t *port_range);

END_C_DECLS

#endif
