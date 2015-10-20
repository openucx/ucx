/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
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

/*
 * Valgrind support
 */
#ifndef NVALGRIND
#  include <valgrind/memcheck.h>
#  ifndef VALGRIND_MAKE_MEM_DEFINED
#    define VALGRIND_MAKE_MEM_DEFINED(p, n)   VALGRIND_MAKE_READABLE(p, n)
#  endif
#  ifndef VALGRIND_MAKE_MEM_UNDEFINED
#    define VALGRIND_MAKE_MEM_UNDEFINED(p, n) VALGRIND_MAKE_WRITABLE(p, n)
#  endif
#else
#  define VALGRIND_MAKE_MEM_DEFINED(p, n)
#  define VALGRIND_MAKE_MEM_UNDEFINED(p, n)
#  define VALGRIND_MAKE_MEM_NOACCESS(p, n)
#  define VALGRIND_CREATE_MEMPOOL(n,p,x)
#  define VALGRIND_DESTROY_MEMPOOL(p)
#  define VALGRIND_MEMPOOL_ALLOC(n,p,x)
#  define VALGRIND_MEMPOOL_FREE(n,p)
#  define VALGRIND_COUNT_ERRORS              0
#  define VALGRIND_COUNT_LEAKS(a,b,c,d)      { a = b = c = d = 0; }
#  define RUNNING_ON_VALGRIND                0
#  define VALGRIND_PRINTF(...)
#endif


/*
 * BullsEye Code Coverage tool
 */
#if _BullseyeCoverage
#define BULLSEYE_ON                          1
#define BULLSEYE_EXCLUDE_START               #pragma BullseyeCoverage off
#define BULLSEYE_EXCLUDE_END                 #pragma BullseyeCoverage on
#define BULLSEYE_EXCLUDE_BLOCK_START         "BullseyeCoverage save off";
#define BULLSEYE_EXCLUDE_BLOCK_END           "BullseyeCoverage restore";
#else
#define BULLSEYE_ON                          0
#define BULLSEYE_EXCLUDE_START
#define BULLSEYE_EXCLUDE_END
#define BULLSEYE_EXCLUDE_BLOCK_START
#define BULLSEYE_EXCLUDE_BLOCK_END
#endif


/**
 * @return Host name.
 */
const char *ucs_get_host_name();


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
 * Fill a filename template. The following values in the string are replaced:
 *  %p - replaced by process id
 *  %h - replaced by host name
 *
 * @param tmpl   File name template (possibly containing formatting sequences)
 * @param buf    Filled with resulting file name
 * @param max    Maximal size of destination buffer.
 */
void ucs_fill_filename_template(const char *tmpl, char *buf, size_t max);


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
ucs_open_output_stream(const char *config_str, FILE **p_fstream, int *p_need_close,
                       const char **p_next_token);


/**
 * Return a number filled with the first characters of the string.
 */
uint64_t ucs_string_to_id(const char *str);


/**
 * Format a string to a buffer of given size, and fill the rest of the buffer
 * with '\0'. Also, guarantee that the last char in the buffer is '\0'.
 *
 * @param buf  Buffer to format the string to.
 * @param size Buffer size.
 * @param fmt  Format string.
 */
void ucs_snprintf_zero(char *buf, size_t size, const char *fmt, ...)
    UCS_F_PRINTF(3, 4);


/**
 * Read file contents into a string. If the size of the data is smaller than the
 * supplied upper limit (max), a null terminator is appended to the data.
 *
 * @param buffer        Buffer to fill with file contents.
 * @param max           Maximal buffer size.
 * @param filename_fmt  File name printf-like format string.
 *
 * @return Number of ytes read, or -1 in case of error.
 */
ssize_t ucs_read_file(char *buffer, size_t max, int silent,
                      const char *filename_fmt, ...)
    UCS_F_PRINTF(4, 5);


/**
 * @return Regular page size on the system.
 */
size_t ucs_get_page_size();


/**
 * @return Huge page size on the system.
 */
size_t ucs_get_huge_page_size();


/**
 * @return Physical memory size on the system.
 */
size_t ucs_get_phys_mem_size();


/**
 * Allocate shared memory using SystemV API.
 *
 * @param size      Pointer to memory size to allocate, updated with actual size
 *                  (rounded up to huge page size or to regular page size).
 * @param address_p Filled with allocated memory address.
 * @param flags     Flags to indicate the permissions for the allocate memory.
 *                  (also, whether or not to allocate memory with huge pages).
 * @param shmid     Filled with the shmid from the shmget call in the function.
 */
ucs_status_t ucs_sysv_alloc(size_t *size, void **address_p, int flags, int *shimd
                            UCS_MEMTRACK_ARG);


/**
 * Release memory allocated via hugetlb.
 *
 * @param address   Memory to release (retuned from ucs_hugetlb_alloc).
 */
ucs_status_t ucs_sysv_free(void *address);


/**
 * Retrieve memory access flags for a given region of memory.
 * If the specified memory region has multiple different access flags, the AND
 * of them is returned. If any part of the region is not mapped, PROT_NONE will
 * be returned.
 *
 * @param address Region start.
 * @param length  Region length.
 * @return Memory protection flags (PROT_xxx).
 */
unsigned ucs_get_mem_prot(void *address, size_t length);


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
 * Get CPU frequency from /proc/cpuinfo. Return value is clocks-per-second.
 *
 * @param mhz_header String in /proc/cpuinfo which precedes the clock speed number.
 */
double ucs_get_cpuinfo_clock_freq(const char *mhz_header);


/**
 * Empty function which can be casted to a no-operation callback in various situations.
 */
void ucs_empty_function();
ucs_status_t ucs_empty_function_return_success();
ucs_status_t uct_empty_function_return_unsupported();

#endif
