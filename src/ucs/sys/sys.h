/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_SYS_H
#define UCS_SYS_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler.h>
#include <ucs/type/status.h>
#include <ucs/debug/memtrack.h>
#include <ucs/config/types.h>

#include <errno.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/fcntl.h>
#include <sys/epoll.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/param.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <net/if_arp.h>
#include <net/if.h>
#include <netdb.h>

BEGIN_C_DECLS

/**
 * @return Host name.
 */
const char *ucs_get_host_name();


/**
 * @return user name.
 */
const char *ucs_get_user_name();


/**
 * Expand a partial path to full path.
 *
 * @param path       Path to expand.
 * @param fullpath   Filled with full path.
 * @param max        Room in "fullpath"
 */
void ucs_expand_path(const char *path, char *fullpath, size_t max);


/**
 * @return Path to the main executable.
 */
const char *ucs_get_exe();


/**
 * Calculate checksum of a file.
 */
uint32_t ucs_file_checksum(const char *filename);


/**
 * Get a globally unique identifier of the machine running the current process.
 */
uint64_t ucs_machine_guid();


/**
 * Get the first processor number we are bound to.
 */
int ucs_get_first_cpu();


/**
 * Generate a world-wide unique ID
 *
 * @param seed Additional seed to mix in.
 *
 * @note All bits of the returned number have the same randomness.
 */
uint64_t ucs_generate_uuid(uint64_t seed);


/**
 * Open an output stream according to user configuration:
 *   - file:<name> - file name, %p, %h, %c are substituted.
 *   - stdout
 *   - stderr
 *
 * *p_fstream is filled with the stream handle, *p_need_close is set to whether
 * fclose() should be called to release resources, *p_next_token to the remainder
 * of config_str.
 */
ucs_status_t
ucs_open_output_stream(const char *config_str, ucs_log_level_t err_log_level,
                       FILE **p_fstream, int *p_need_close,
                       const char **p_next_token);


/**
 * Read file contents into a string. If the size of the data is smaller than the
 * supplied upper limit (max), a null terminator is appended to the data.
 *
 * @param buffer        Buffer to fill with file contents.
 * @param max           Maximal buffer size.
 * @param filename_fmt  File name printf-like format string.
 *
 * @return Number of bytes read, or -1 in case of error.
 */
ssize_t ucs_read_file(char *buffer, size_t max, int silent,
                      const char *filename_fmt, ...)
    UCS_F_PRINTF(4, 5);


/**
 * Read file contents as a numeric value.
 *
 * @param value         Filled with the number read from the file.
 * @param filename_fmt  File name printf-like format string.
 *
 * @return UCS_OK if successful, or error code otherwise.
 */
ucs_status_t ucs_read_file_number(long *value, int silent,
                                  const char *filename_fmt, ...)
    UCS_F_PRINTF(3, 4);


/**
 * @return Regular _SC_IOV_MAX on the system.
 */
size_t ucs_get_max_iov();


/**
 * @return Regular page size on the system.
 */
size_t ucs_get_page_size();


/**
 * @return Huge page size on the system.
 */
size_t ucs_get_huge_page_size();


/**
 * @return free mem size on the system.
 */
size_t ucs_get_memfree_size();


/**
 * @return Physical memory size on the system.
 */
size_t ucs_get_phys_mem_size();


/**
 * Allocate shared memory using SystemV API.
 *
 * @param size      Pointer to memory size to allocate, updated with actual size
 *                  (rounded up to huge page size or to regular page size).
 * @param max_size  maximal size to allocate. If need to allocate more than this,
 *                  the function fails and returns UCS_ERR_EXCEEDS_LIMIT.
 * @param address_p Filled with allocated memory address.
 * @param flags     Flags to indicate the permissions for the allocate memory.
 *                  (also, whether or not to allocate memory with huge pages).
 * @param shmid     Filled with the shmid from the shmget call in the function.
 */
ucs_status_t ucs_sysv_alloc(size_t *size, size_t max_size, void **address_p,
                            int flags, int *shimd UCS_MEMTRACK_ARG);


/**
 * Release memory allocated via SystemV API.
 *
 * @param address   Memory to release (returned from @ref ucs_sysv_alloc).
 */
ucs_status_t ucs_sysv_free(void *address);


/**
 * Allocate private memory using mmap API.
 *
 * @param size      Pointer to memory size to allocate, updated with actual size
 *                  (rounded up to huge page size or to regular page size).
 * @param address_p Filled with allocated memory address.
 * @param flags     Flags to pass to the mmap() system call
 */
ucs_status_t ucs_mmap_alloc(size_t *size, void **address_p,
                            int flags UCS_MEMTRACK_ARG);

/**
 * Release memory allocated via mmap API.
 *
 * @param address   Address of memory to release as returned from @ref ucs_mmap_alloc.
 * @param length    Length of memory to release as returned from @ref ucs_mmap_alloc.
 */
ucs_status_t ucs_mmap_free(void *address, size_t length);

/**
 * Retrieve memory access flags for a given region of memory.
 * If the specified memory region has multiple different access flags, the AND
 * of them is returned. If any part of the region is not mapped, PROT_NONE will
 * be returned.
 *
 * @param start Region start.
 * @param end   Region end.
 * @return Memory protection flags (PROT_xxx).
 */
int ucs_get_mem_prot(unsigned long start, unsigned long end);


/**
 * Modify file descriptor flags via fcntl().
 *
 * @param fd     File descriptor to modify.
 * @param add    Flags to add.
 * @param remove Flags to remove.
 *
 * Note: if a flags is specified in both add and remove, it will be removed.
 */
ucs_status_t ucs_sys_fcntl_modfl(int fd, int add, int remove);


/**
 * Get process command line
 */
const char* ucs_get_process_cmdline();


/**
 * Get current thread (LWP) id.
 */
pid_t ucs_get_tid(void);


/**
 * Send signal to a thread.
 */
int ucs_tgkill(int tgid, int tid, int sig);


/**
 * Get CPU frequency from /proc/cpuinfo. Return value is clocks-per-second.
 *
 * @param header String in /proc/cpuinfo which precedes the clock speed number.
 * @param scale  Frequency value units.
 */
double ucs_get_cpuinfo_clock_freq(const char *mhz_header, double scale);


/**
 * Check if transparent huge-pages are enabled .
 *
 * @return 1 for true and 0 for false
 */
int ucs_is_thp_enabled();


/**
 * Get shmmax size from /proc/sys/kernel/shmmax.
 *
 * @return shmmax size
 */
size_t ucs_get_shmmax();


/**
 * Allocate or re-allocate memory from the operating system.
 *
 * @param [in]  old_ptr     Pointer to existing block, may be NULL. If non-NULL,
 *                          this block will be resized and potentially moved.
 * @param [in]  old_length  Length of the block pointed by old_ptr.
 * @param [in]  new_length  Length to allocate for the new block.
 *
 * @return New allocated block, with size 'new_length'.
 * @note Actual allocation size is rounded up to system page size.
 */
void *ucs_sys_realloc(void *old_ptr, size_t old_length, size_t new_length);


/**
 * Release memory previously allocated by @ref ucs_sys_realloc().
 *
 * @param [in]  ptr         Pointer to memory block to release.
 * @param [in]  length      Length of the memory block.
 */
void ucs_sys_free(void *ptr, size_t length);


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
 */
int ucs_netif_is_active(const char *if_name);


/**
 * Create a socket.
 *
 * @param [out]  fd_p       Pointer to created fd.
 */
ucs_status_t ucs_tcpip_socket_create(int *fd_p);


/**
 * Empty function which can be casted to a no-operation callback in various situations.
 */
void ucs_empty_function();
unsigned ucs_empty_function_return_zero();
ucs_status_t ucs_empty_function_return_success();
ucs_status_t ucs_empty_function_return_unsupported();
ucs_status_t ucs_empty_function_return_inprogress();
ucs_status_t ucs_empty_function_return_no_resource();
ucs_status_ptr_t ucs_empty_function_return_ptr_no_resource();
ucs_status_t ucs_empty_function_return_ep_timeout();
ssize_t ucs_empty_function_return_bc_ep_timeout();
ucs_status_t ucs_empty_function_return_busy();

END_C_DECLS

#endif
