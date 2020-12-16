/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE /* for dladdr */
#endif

#include "sys.h"

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucm/api/ucm.h>
#include <ucm/util/log.h>
#include <ucm/mmap/mmap.h>
#include <ucs/type/init_once.h>
#include <ucs/sys/math.h>
#include <linux/mman.h>
#include <sys/mman.h>
#include <pthread.h>
#include <syscall.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <dlfcn.h>


#define UCM_PROC_SELF_MAPS "/proc/self/maps"

ucm_global_config_t ucm_global_opts = {
    .log_level                  = UCS_LOG_LEVEL_WARN,
    .enable_events              = 1,
    .mmap_hook_mode             = UCM_DEFAULT_HOOK_MODE,
    .enable_malloc_hooks        = 1,
    .enable_malloc_reloc        = 0,
    .cuda_hook_mode             = UCM_DEFAULT_HOOK_MODE,
    .enable_dynamic_mmap_thresh = 1,
    .alloc_alignment            = 16,
    .dlopen_process_rpath       = 1
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
    return UCS_PTR_BYTE_OFFSET(ptr, sizeof(size_t));
}

void *ucm_sys_malloc(size_t size)
{
    size_t sys_size;
    void *ptr;

    sys_size = ucs_align_up_pow2(size + sizeof(size_t), ucm_get_page_size());
    ptr = ucm_orig_mmap(NULL, sys_size, PROT_READ|PROT_WRITE,
                        MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        ucm_error("mmap(size=%zu) failed: %m", sys_size);
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

    /* Do not use UCS_PTR_BYTE_OFFSET macro here due to coverity
     * false positive.
     * TODO: check for false positive on newer coverity. */
    ptr  = (char*)ptr - sizeof(size_t);
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

    oldptr   = UCS_PTR_BYTE_OFFSET(ptr, -sizeof(size_t));
    oldsize  = *(size_t*)oldptr;
    sys_size = ucs_align_up_pow2(size + sizeof(size_t), ucm_get_page_size());

    if (sys_size == oldsize) {
        return ptr;
    }

    newptr = ucm_orig_mremap(oldptr, oldsize, sys_size, MREMAP_MAYMOVE);
    if (newptr == MAP_FAILED) {
        ucm_error("mremap(oldptr=%p oldsize=%zu, newsize=%zu) failed: %m",
                  oldptr, oldsize, sys_size);
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
    int line_num;
    int prot;
    char *ptr, *newline;
    int maps_fd;
    int ret;
    int n;

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

    ptr      = buffer;
    line_num = 1;
    while ( (newline = strchr(ptr, '\n')) != NULL ) {
        /* address           perms offset   dev   inode   pathname
         * 00400000-0040b000 r-xp  00001a00 0a:0b 12345   /dev/mydev
         */
        *newline = '\0';
        ret = sscanf(ptr, "%lx-%lx %4c %*x %*x:%*x %*d %n",
                     &start, &end, prot_c,
                     /* ignore offset, dev, inode */
                     &n /* save number of chars before path begins */);
        if (ret < 3) {
            ucm_warn("failed to parse %s line %d: '%s'",
                     UCM_PROC_SELF_MAPS, line_num, ptr);
        } else {
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

            if (cb(arg, (void*)start, end - start, prot, ptr + n)) {
                goto out;
            }
        }

        ptr = newline + 1;
        ++line_num;
    }

out:
    pthread_rwlock_unlock(&lock);
}

typedef struct {
    const void   *shmaddr;
    size_t       seg_size;
} ucm_get_shm_seg_size_ctx_t;

static int ucm_get_shm_seg_size_cb(void *arg, void *addr, size_t length,
                                   int prot, const char *path)
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

void ucm_prevent_dl_unload()
{
    static ucs_init_once_t init_once = UCS_INIT_ONCE_INITIALIZER;
    int flags                        = RTLD_LOCAL | RTLD_NODELETE;
    Dl_info info;
    void *dl;
    int ret;

    if (ucm_global_opts.module_unload_prevent_mode ==
        UCM_UNLOAD_PREVENT_MODE_NONE) {
        return;
    }

    UCS_INIT_ONCE(&init_once) {
        flags |= (ucm_global_opts.module_unload_prevent_mode ==
                  UCM_UNLOAD_PREVENT_MODE_NOW) ? RTLD_NOW : RTLD_LAZY;

        /* Get the path to current library by current function pointer */
        (void)dlerror();
        ret = dladdr(ucm_prevent_dl_unload, &info);
        if (ret == 0) {
            ucm_warn("could not find address of current library: %s", dlerror());
            return;
        }

        /* Load the current library with NODELETE flag, to prevent it from being
         * unloaded. This will create extra reference to the library, but also add
         * NODELETE flag to the dynamic link map.
         */
        (void)dlerror();
        dl = dlopen(info.dli_fname, flags);
        if (dl == NULL) {
            ucm_warn("failed to load '%s': %s", info.dli_fname, dlerror());
            return;
        }

        ucm_debug("loaded '%s' at %p with NODELETE flag", info.dli_fname, dl);

        /* coverity[overwrite_var] */
        dl = NULL;
    }
}

char *ucm_concat_path(char *buffer, size_t max, const char *dir, const char *file)
{
    size_t len;

    len = strlen(dir);
    while (len && (dir[len - 1] == '/')) {
        len--; /* trim closing '/' */
    }

    len = ucs_min(len, max);
    memcpy(buffer, dir, len);
    max -= len;
    if (max < 2) { /* buffer is shorter than dir - copy dir only */
        buffer[len - 1] = '\0';
        return buffer;
    }

    buffer[len] = '/';
    max--;

    while (file[0] == '/') {
        file++; /* trim beginning '/' */
    }

    strncpy(buffer + len + 1, file, max);
    buffer[max + len] = '\0'; /* force close string */

    return buffer;
}

void *ucm_brk_syscall(void *addr)
{
    void *result;

#ifdef __x86_64__
    asm volatile("mov %1, %%rdi\n\t"
                 "mov $0xc, %%eax\n\t"
                 "syscall\n\t"
                 : "=a"(result)
                 : "m"(addr));
#else
    /* TODO implement 64-bit syscall for aarch64, ppc64le */
    result = (void*)syscall(SYS_brk, addr);
#endif
    return result;
}

pid_t ucm_get_tid()
{
    return syscall(SYS_gettid);
}
