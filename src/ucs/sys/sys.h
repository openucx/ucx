/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 * Copyright (c) UT-Battelle, LLC. 2014-2019. ALL RIGHTS RESERVED.
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
#include <ucs/type/cpu_set.h>
#include <ucs/time/time_def.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/config/types.h>
#include <ucs/config/parser.h>

#include <errno.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/fcntl.h>
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
#include <dirent.h>


#include <sys/types.h>
#if defined(__linux__) || defined(HAVE_CPU_SET_T)
#include <sched.h>
typedef cpu_set_t ucs_sys_cpuset_t;
#elif defined(__FreeBSD__) || defined(HAVE_CPUSET_T)
#include <sys/cpuset.h>
typedef cpuset_t ucs_sys_cpuset_t;
#else
#error "Port me"
#endif


BEGIN_C_DECLS

/** @file sys.h */


typedef ino_t ucs_sys_ns_t;


/* namespace type used in @ref ucs_sys_get_ns and @ref ucs_sys_ns_is_default */
typedef enum {
    UCS_SYS_NS_TYPE_IPC,
    UCS_SYS_NS_TYPE_MNT,
    UCS_SYS_NS_TYPE_NET,
    UCS_SYS_NS_TYPE_PID,
    UCS_SYS_NS_TYPE_USER,
    UCS_SYS_NS_TYPE_UTS,
    UCS_SYS_NS_TYPE_LAST
} ucs_sys_namespace_type_t;


/* virtual memory area flags */
typedef enum {
    UCS_SYS_VMA_FLAG_DONTCOPY = UCS_BIT(0),
} ucs_sys_vma_info_flags_t;


/* information about virtual memory area */
typedef struct {
    unsigned long start;
    unsigned long end;
    size_t        page_size;
    uint32_t      flags;
} ucs_sys_vma_info_t;


/**
 * Callback function type used in ucs_sys_iterate_vm.
 */
typedef void (*ucs_sys_vma_cb_t)(ucs_sys_vma_info_t *info, void *ctx);


/**
 * Callback function type used in ucs_sys_readdir.
 */
typedef ucs_status_t (*ucs_sys_readdir_cb_t)(struct dirent *entry, void *ctx);


/**
 * Callback function type used in ucs_sys_enum_threads.
 */
typedef ucs_status_t (*ucs_sys_enum_threads_cb_t)(pid_t pid, void *ctx);


/**
 * Callback function type used in ucs_sys_enum_pfn.
 */
typedef void (*ucs_sys_enum_pfn_cb_t)(unsigned page_number, unsigned long pfn,
                                      void *ctx);


/**
 * @return TMPDIR environment variable if set. Otherwise, return "/tmp".
 */
const char *ucs_get_tmpdir();

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
 * Returns a value one greater than the maximum file descriptor
 * number that can be opened by this process. Attempts to exceed
 * this limit yield the error EMFILE.
 *
 * @return The maximal number of files that could be opened simultaneously.
 */
int ucs_sys_max_open_files();


/**
 * Open an output stream according to user configuration:
 *   - file:<name> - file name, %p, %h, %c are substituted.
 *   - stdout
 *   - stderr
 *
 * @param [in]  config_str     The file name or name of the output stream
 *                             (stdout/stderr).
 * @param [in]  err_log_level  Logging level that should be used for printing
 *                             errors.
 * @param [out] p_fstream      Pointer that is filled with the stream handle.
 *                             User is responsible to close tha stream handle then.
 * @param [out] p_need_close   Pointer to the variable that is set to whether
 *                             fclose() should be called to release resources (1)
 *                             or not (0).
 * @param [out] p_next_token   Pointer that is set to remainder of @config_str.
 * @param [out] p_filename     Pointer to the variable that is filled with the
 *                             resulted name of the log file (if it is not NULL).
 *                             Caller is responsible to release memory then.
 *
 * @return UCS_OK if successful, or error code otherwise.
 */
ucs_status_t
ucs_open_output_stream(const char *config_str, ucs_log_level_t err_log_level,
                       FILE **p_fstream, int *p_need_close,
                       const char **p_next_token, char **p_filename);


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
 * Read file contents into a string closed by null terminator.
 *
 * @param buffer        Buffer to fill with file contents.
 * @param max           Maximal buffer size.
 * @param filename_fmt  File name printf-like format string.
 *
 * @return Number of bytes read, or -1 in case of error.
 */
ssize_t ucs_read_file_str(char *buffer, size_t max, int silent,
                          const char *filename_fmt, ...)
    UCS_F_PRINTF(4, 5);


/**
 * @return Regular page size on the system.
 */
size_t ucs_get_page_size();


/**
 * Get info for memory range.
 *
 * @param address     Memory range start address,
 * @param size        Memory range size.
 * @param cb          Callback function which is called for every vm area.
 * @param ctx         Context argument passed to @a cb call.
 */
void ucs_sys_iterate_vm(void *address, size_t size, ucs_sys_vma_cb_t cb,
                        void *ctx);


/**
 * Get page size of a memory range.
 *
 * @param [in]  address          Memory range start address,
 * @param [in]  size             Memory range size.
 * @param [out] min_page_size_p  Set to the minimal page size in the memory range.
 * @param [out] max_page_size_p  Set to the maximal page size in the memory range.
 */
void ucs_get_mem_page_size(void *address, size_t size, size_t *min_page_size_p,
                           size_t *max_page_size_p);


/**
 * @return Huge page size on the system, or -1 if unsupported.
 */
ssize_t ucs_get_huge_page_size();


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
 * @param size       Pointer to memory size to allocate, updated with actual size
 *                   (rounded up to huge page size or to regular page size).
 * @param max_size   maximal size to allocate. If need to allocate more than this,
 *                   the function fails and returns UCS_ERR_EXCEEDS_LIMIT.
 * @param address_p  Filled with allocated memory address.
 * @param flags      Flags to indicate the permissions for the allocate memory.
 *                   (also, whether or not to allocate memory with huge pages).
 * @param alloc_name Name of memory allocation, for debug/error reporting purposes.
 * @param shmid      Filled with the shmid from the shmget call in the function.
 */
ucs_status_t ucs_sysv_alloc(size_t *size, size_t max_size, void **address_p,
                            int flags, const char *alloc_name, int *shimd);


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
                            int flags, const char *alloc_name);

/**
 * Release memory allocated via mmap API.
 *
 * @param address   Address of memory to release as returned from @ref ucs_mmap_alloc.
 * @param length    Length of memory to release passed to @ref ucs_mmap_alloc.
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
 * Returns the physical page frame number of a given virtual page address.
 * If the page map file is non-readable (for example, due to permissions), or
 * the page is not present, this function returns 0.
 *
 * @param address    Virtual address to get the. PFN for.
 * @param page_count Number of pages to process
 * @param data       Result buffer.
 * @return UCS_OK if all pages are processed, else error code.
 */
ucs_status_t ucs_sys_get_pfn(uintptr_t address, unsigned page_count,
                             unsigned long *data);


/**
 * Enums the physical page frame numbers of a given virtual address range.
 * If the page map file is non-readable (for example, due to permissions), or
 * the page is not present, this function returns error.
 *
 * @param address     Virtual address to get the PFN for.
 * @param page_number Number of pages to process.
 * @param cb          Callback function which is called for every page.
 * @param ctx         Context argument passed to @a cb call.
 * @return error code if failed to enumerate or UCS_OK.
 */
ucs_status_t ucs_sys_enum_pfn(uintptr_t address, unsigned page_count,
                              ucs_sys_enum_pfn_cb_t cb, void *ctx);


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
 * Return effective capabilities of the current thread.
 *
 * @param effective  Filled with thread's effective capabilities.
 * @return UCS_OK or error in case of failure.
 */
ucs_status_t ucs_sys_get_proc_cap(uint32_t *effective);


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
 * Fill human readable cpu set representation
 *
 * @param [in]  cpuset      Set of CPUs
 * @param [in]  str         String to fill
 * @param [in]  len         String length
 *
 * @return Filled string
 */
char *ucs_make_affinity_str(const ucs_sys_cpuset_t *cpuset, char *str, size_t len);

/**
 * Sets affinity for the current process.
 *
 * @param [in] cpuset      Pointer to the cpuset to assign
 *
 * @return -1 on error with errno set, 0 on success
 */
int ucs_sys_setaffinity(ucs_sys_cpuset_t *cpuset);

/**
 * Queries affinity for the current process.
 *
 * @param [out] cpuset      Pointer to the cpuset to return result
 *
 * @return -1 on error with errno set, 0 on success
 */
int ucs_sys_getaffinity(ucs_sys_cpuset_t *cpuset);

/**
 * Copies ucs_sys_cpuset_t to ucs_cpu_set_t.
 *
 * @param [in]  src         Source
 * @param [out] dst         Destination
 */
void ucs_sys_cpuset_copy(ucs_cpu_set_t *dst, const ucs_sys_cpuset_t *src);

/**
 * Get namespace id for resource.
 *
 * @param [in]  name        Namespace to get value
 *
 * @return namespace value or 0 if namespaces are not supported
 */
ucs_sys_ns_t ucs_sys_get_ns(ucs_sys_namespace_type_t name);


/**
 * Check if namespace is namespace of host system.
 *
 * @param [in]  name        Namespace to evaluate
 *
 * @return 1 in case if namespace is root, 0 - in other cases
 */
int ucs_sys_ns_is_default(ucs_sys_namespace_type_t name);


/**
 * Get 128-bit boot ID value.
 *
 * @param [out]  high       Pointer to high 64 bit of 128 boot ID
 * @param [out]  low        Pointer to low 64 bit of 128 boot ID
 *
 * @return UCS_OK or error in case of failure.
 */
ucs_status_t ucs_sys_get_boot_id(uint64_t *high, uint64_t *low);

/**
 * Read directory
 *
 * @param [in]  path       Path to directory to read
 * @param [in]  cb         Callback function, see NOTES
 * @param [in]  ctx        Context pointer passed to callback
 *
 * @return UCS_OK if directory is found and successfully iterated thought all
 *         entries, error code in all other cases, see NOTES.
 *
 * @note ucs_sys_readdir function reads directory pointed by @a path argument
 *       and calls @a cb function for every entry in directory, including
 *       '.' and '..'. In case if @a cb function returns value different from
 *       UCS_OK then function breaks immediately and this value is returned
 *       from ucs_sys_readdir.
 */
ucs_status_t ucs_sys_readdir(const char *path, ucs_sys_readdir_cb_t cb, void *ctx);

/**
 * Enumerate process threads
 *
 * @param [in]  cb         Callback function, see NOTES
 * @param [in]  ctx        Context pointer passed to callback
 *
 * @return UCS_OK if directory is found and successfully iterated thought all
 *         entries, error code in all other cases, see NOTES.
 *
 * @note ucs_sys_enum_threads function enumerates current process threads
 *       and calls @a cb function for every thread. In case if @a cb function
 *       returns value different from UCS_OK then function breaks
 *       immediately and this value is returned from ucs_sys_enum_threads.
 */
ucs_status_t ucs_sys_enum_threads(ucs_sys_enum_threads_cb_t cb, void *ctx);


/**
 * Check the per-process limit on the number of open file descriptors.
 *
 * @return UCS_OK if the limit has not been reached. UCS_ERR_EXCEEDS_LIMIT,
 *         otherwise.
 */
ucs_status_t ucs_sys_check_fd_limit_per_process();


/*
 * Create a named thread.
 *
 * @param [out] thread_id_p     If successful, set to the new thread id.
 * @param [in]  start_routine   Thread function.
 * @param [in]  arg             Custom argument for start_routine.
 * @param [in]  fmt             Thread name format string.
 *
 * @return UCS_OK if successful, or error status if failed.
 */
ucs_status_t ucs_pthread_create(pthread_t *thread_id_p,
                                void *(*start_routine)(void*), void *arg,
                                const char *fmt, ...) UCS_F_PRINTF(4, 5);

/*
 * Get number of CPUs.
 *
 * @return number of CPUs, or -1 in case of error.
 */
long ucs_sys_get_num_cpus();


/*
 * Get process creation time.
 *
 * @param [in] pid             Process id to get start time.
 *
 * @return The time the process started after system boot or 0 in case of error.
 */
unsigned long ucs_sys_get_proc_create_time(pid_t pid);


/*
 * Get the current max locked memory limit.
 *
 * @param [out] rlimit_value If successful, set to the current limit value.
 */
ucs_status_t ucs_sys_get_memlock_rlimit(size_t *rlimit_value);


/*
 * Check if library is built dynamically (.so module)
 *
 * @return 1 if built dynamically, 0 if statically.
 */
int ucs_sys_is_dynamic_lib(void);

END_C_DECLS

#endif
