/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_SOCKET_H
#define UCS_SOCKET_H

#include <ucs/type/status.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#ifdef __linux__
#include <linux/if.h>
#else
#include <net/if.h>
#endif
#include <arpa/inet.h>


BEGIN_C_DECLS


/* A string to hold the IP address and port from a sockaddr */
#define UCS_SOCKADDR_STRING_LEN          60

#define UCS_SOCKET_INET_ADDR(_addr)      (((struct sockaddr_in*)(_addr))->sin_addr)
#define UCS_SOCKET_INET_PORT(_addr)      (((struct sockaddr_in*)(_addr))->sin_port)

#define UCS_SOCKET_INET6_ADDR(_addr)     (((struct sockaddr_in6*)(_addr))->sin6_addr)
#define UCS_SOCKET_INET6_PORT(_addr)     (((struct sockaddr_in6*)(_addr))->sin6_port)


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
 * Connects the socket referred to by the file descriptor `fd`
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
 * Reports information about non-blocking connection status for
 * the socket referred to by the file descriptor `fd`.
 *
 * @param [in]  fd          Socket fd.
 *
 * @return UCS_OK on success or UCS_ERR_UNREACHABLE on failure or
 *         UCS_INPROGRESS if operation is still in progress.
 */
ucs_status_t ucs_socket_connect_nb_get_status(int fd);


/**
 * Returns size of a given sockaddr structure.
 * 
 * @param [in]   addr       Pointer to sockaddr structure.
 * @param [out]  size_p     Pointer to variable where size of
 *                          sockaddr_in/sockaddr_in6 structure will be written
 *
 * @return UCS_OK on success or UCS_ERR_INVALID_PARAM on failure.
 */
ucs_status_t ucs_sockaddr_sizeof(const struct sockaddr *addr, size_t *size_p);


/**
 * Returns port of a given sockaddr structure.
 * 
 * @param [in]   addr       Pointer to sockaddr structure.
 * @param [out]  port_p     Pointer to variable where port (host notation)
 *                          of sockaddr_in/sockaddr_in6 structure will be written
 *
 * @return UCS_OK on success or UCS_ERR_INVALID_PARAM on failure.
 */
ucs_status_t ucs_sockaddr_get_port(const struct sockaddr *addr, unsigned *port_p);


/**
 * Returns IP addr of a given sockaddr structure.
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


END_C_DECLS

#endif
