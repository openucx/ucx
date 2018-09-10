/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sys.h"

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucm/api/ucm.h>
#include <ucm/util/log.h>
#include <ucs/sys/math.h>
#include <linux/mman.h>
#include <sys/mman.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>


#define UCM_PROC_SELF_MAPS "/proc/self/maps"

ucm_global_config_t ucm_global_opts = {
    .log_level                  = UCS_LOG_LEVEL_WARN,
    .enable_events              = 1,
    .enable_mmap_reloc          = 1,
    .enable_malloc_hooks        = 1,
    .enable_malloc_reloc        = 0,
    .enable_cuda_reloc          = 1,
    .enable_dynamic_mmap_thresh = 1,
    .alloc_alignment            = 16,
    .enable_syscall             = 0
};

size_t ucm_get_page_size()
{
    static long page_size = -1;
    long value;

    if (page_size == -1) {
        value = sysconf(_SC_PAGESIZE);
        if (value < 0) {
            page_size = 4096;
        } else {
            page_size = value;
        }
    }
    return page_size;
}

static void *ucm_sys_complete_alloc(void *ptr, size_t size)
{
    *(size_t*)ptr = size;
    return ptr + sizeof(size_t);
}

void *ucm_sys_malloc(size_t size)
{
    size_t sys_size;
    void *ptr;

    sys_size = ucs_align_up_pow2(size + sizeof(size_t), ucm_get_page_size());
    ptr = ucm_orig_mmap(NULL, sys_size, PROT_READ|PROT_WRITE,
                        MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        return NULL;
    }

    return ucm_sys_complete_alloc(ptr, sys_size);
}

void *ucm_sys_calloc(size_t nmemb, size_t size)
{
    size_t total_size = size * nmemb;
    void *ptr;

    ptr = ucm_sys_malloc(total_size);
    if (ptr == NULL) {
        return NULL;
    }

    memset(ptr, 0, total_size);
    return ptr;
}

void ucm_sys_free(void *ptr)
{
    size_t size;

    if (ptr == NULL) {
        return;
    }

    ptr -= sizeof(size_t);
    size = *(size_t*)ptr;
    munmap(ptr, size);
}

void *ucm_sys_realloc(void *ptr, size_t size)
{
    size_t oldsize, sys_size;
    void *oldptr, *newptr;

    if (ptr == NULL) {
        return ucm_sys_malloc(size);
    }

    oldptr   = ptr - sizeof(size_t);
    oldsize  = *(size_t*)oldptr;
    sys_size = ucs_align_up_pow2(size + sizeof(size_t), ucm_get_page_size());

    if (sys_size == oldsize) {
        return ptr;
    }

    newptr = ucm_orig_mremap(oldptr, oldsize, sys_size, MREMAP_MAYMOVE);
    if (newptr == MAP_FAILED) {
        return NULL;
    }

    return ucm_sys_complete_alloc(newptr, sys_size);
}

void ucm_parse_proc_self_maps(ucm_proc_maps_cb_t cb, void *arg)
{
    static char  *buffer         = MAP_FAILED;
    static size_t buffer_size    = 32768;
    static pthread_rwlock_t lock = PTHREAD_RWLOCK_INITIALIZER;
    ssize_t read_size, offset;
    unsigned long start, end;
    char prot_c[4];
    int prot;
    char *ptr, *newline;
    int maps_fd;
    int ret;

    maps_fd = open(UCM_PROC_SELF_MAPS, O_RDONLY);
    if (maps_fd < 0) {
        ucm_fatal("cannot open %s for reading: %m", UCM_PROC_SELF_MAPS);
    }

    /* read /proc/self/maps fully into the buffer */
    pthread_rwlock_wrlock(&lock);

    if (buffer == MAP_FAILED) {
        buffer = ucm_orig_mmap(NULL, buffer_size, PROT_READ|PROT_WRITE,
                               MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (buffer == MAP_FAILED) {
            ucm_fatal("failed to allocate maps buffer(size=%zu): %m", buffer_size);
        }
    }

    offset = 0;
    for (;;) {
        read_size = read(maps_fd, buffer + offset, buffer_size - offset);
        if (read_size < 0) {
            /* error */
            if (errno != EINTR) {
                ucm_fatal("failed to read from %s: %m", UCM_PROC_SELF_MAPS);
            }
        } else if (read_size == buffer_size - offset) {
            /* enlarge buffer */
            buffer = ucm_orig_mremap(buffer, buffer_size, buffer_size * 2,
                                     MREMAP_MAYMOVE);
            if (buffer == MAP_FAILED) {
                ucm_fatal("failed to allocate maps buffer(size=%zu)", buffer_size);
            }
            buffer_size *= 2;

            /* read again from the beginning of the file */
            ret = lseek(maps_fd, 0, SEEK_SET);
            if (ret < 0) {
               ucm_fatal("failed to lseek(0): %m");
            }
            offset = 0;
        } else if (read_size == 0) {
            /* finished reading */
            buffer[offset] = '\0';
            break;
        } else {
            /* more data could be available even if the buffer is not full */
            offset += read_size;
        }
    }
    pthread_rwlock_unlock(&lock);

    close(maps_fd);

    pthread_rwlock_rdlock(&lock);

    ptr    = buffer;
    while ( (newline = strchr(ptr, '\n')) != NULL ) {
        /* 00400000-0040b000 r-xp ... \n */
        ret = sscanf(ptr, "%lx-%lx %4c", &start, &end, prot_c);
        if (ret != 3) {
            ucm_fatal("failed to parse %s error at offset %zd",
                      UCM_PROC_SELF_MAPS, ptr - buffer);
        }

        prot = 0;
        if (prot_c[0] == 'r') {
            prot |= PROT_READ;
        }
        if (prot_c[1] == 'w') {
            prot |= PROT_WRITE;
        }
        if (prot_c[2] == 'x') {
            prot |= PROT_EXEC;
        }

        if (cb(arg, (void*)start, end - start, prot)) {
            goto out;
        }

        ptr = newline + 1;
    }

out:
    pthread_rwlock_unlock(&lock);
}

typedef struct {
    const void   *shmaddr;
    size_t       seg_size;
} ucm_get_shm_seg_size_ctx_t;

static int ucm_get_shm_seg_size_cb(void *arg, void *addr, size_t length, int prot)
{
    ucm_get_shm_seg_size_ctx_t *ctx = arg;
    if (addr == ctx->shmaddr) {
        ctx->seg_size = length;
        return 1;
    }
    return 0;
}

size_t ucm_get_shm_seg_size(const void *shmaddr)
{
    ucm_get_shm_seg_size_ctx_t ctx = { shmaddr, 0 };
    ucm_parse_proc_self_maps(ucm_get_shm_seg_size_cb, &ctx);
    return ctx.seg_size;
}

void ucm_strerror(int eno, char *buf, size_t max)
{
#if STRERROR_R_CHAR_P
    char *ret = strerror_r(eno, buf, max);
    if (ret != buf) {
        strncpy(buf, ret, max);
    }
#else
    (void)strerror_r(eno, buf, max);
#endif
}
