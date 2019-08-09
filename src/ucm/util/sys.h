/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_UTIL_SYS_H_
#define UCM_UTIL_SYS_H_

#include <stddef.h>


/*
 * Substitutes for glibc memory allocation routines, which take memory
 * directly from the operating system, and therefore are safe to use from
 * malloc hooks.
 */
void *ucm_sys_malloc(size_t size);
void *ucm_sys_calloc(size_t nmemb, size_t size);
void ucm_sys_free(void *ptr);
void *ucm_sys_realloc(void *oldptr, size_t newsize);


/**
 * Callback function for processing entries in /proc/self/maps.
 *
 * @param [in] arg      User-defined argument.
 * @param [in] addr     Mapping start address.
 * @param [in] length   Mapping length.
 * @param [in] prot     Mapping memory protection flags (PROT_xx).
 * @param [in] path     Backing file path, or NULL for anonymous mapping.
 *
 * @return 0 to continue iteration, nonzero - stop iteration.
 */
typedef int (*ucm_proc_maps_cb_t)(void *arg, void *addr, size_t length,
                                  int prot, const char *path);


/**
 * @return Page size on the system.
 */
size_t ucm_get_page_size();


/**
 * Read and process entries from /proc/self/maps.
 *
 * @param [in]  cb      Callback function that would be called for each entry
 *                      found in /proc/self/maps.
 * @param [in]  arg     User-defined argument for the function.
 */
void ucm_parse_proc_self_maps(ucm_proc_maps_cb_t cb, void *arg);


/**
 * @brief Get the size of a shared memory segment, attached with shmat()
 *
 * @param [in]  shmaddr  Segment pointer.
 * @return Segment size, or 0 if not found.
 */
size_t ucm_get_shm_seg_size(const void *shmaddr);


/**
 * @brief Convert a errno number to error string
 *
 *  @param [in]  en    errno value
 *  @param [out] buf   Buffer to put the error string in
 *  @param [in]  max   Size of the buffer
 */
void ucm_strerror(int eno, char *buf, size_t max);


void ucm_prevent_dl_unload();


/*
 * Concatenate directory and file names into full path.
 *
 * @param buffer        Filled with the result path.
 * @param max           Maximal buffer size.
 * @param dir           Directory name.
 * @param file          File name.
 *
 * @return Result buffer.
 */
char *ucm_concat_path(char *buffer, size_t max, const char *dir, const char *file);


#endif
