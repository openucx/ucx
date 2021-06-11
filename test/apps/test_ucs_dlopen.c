/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <sys/mman.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <ucm/api/ucm.h>

#define _QUOTE(x) #x
#define QUOTE(x) _QUOTE(x)


static void vm_unmap_cb(ucm_event_type_t event_type, ucm_event_t *event,
                        void *arg)
{
}

int test_ucm_set_event_handler(void *handle)
{
    typedef ucs_status_t (*ucm_set_event_handler_func_t)(int events,
                                                         int priority,
                                                         ucm_event_callback_t cb,
                                                         void *arg);

    ucm_set_event_handler_func_t ucm_set_event_handler_f;
    ucs_status_t status;

    dlerror();
    ucm_set_event_handler_f = (ucm_set_event_handler_func_t)dlsym(handle,
                                                                  "ucm_set_event_handler");
    if (ucm_set_event_handler_f == NULL) {
        fprintf(stderr, "failed to resolve ucm_set_event_handler(): %s\n",
                dlerror());
        return -1;
    }

    status = ucm_set_event_handler_f(UCM_EVENT_VM_UNMAPPED, 0, vm_unmap_cb,
                                     NULL);
    if (status != UCS_OK) {
        fprintf(stderr, "ucm_set_event_handler() failed\n");
        return -1;
    }

    return 0;
}

int main(int argc, char **argv)
{
    const char *filename = QUOTE(LIB_PATH);
    void *handle;
    void *ptr1, *ptr2;
    size_t alloc_size;
    long ret;

    /* get page size */
    ret = sysconf(_SC_PAGESIZE);
    if (ret < 0) {
        fprintf(stderr, "sysconf(_SC_PAGESIZE) failed: %m\n");
        return -1;
    }
    alloc_size = ret;

    /* allocate some memory */
    ptr1 = malloc(alloc_size);
    if (!ptr1) {
        fprintf(stderr, "malloc() failed\n");
        return -1;
    }

    ptr2 = mmap(NULL, alloc_size, PROT_READ|PROT_WRITE,
                MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (ptr2 == MAP_FAILED) {
        fprintf(stderr, "mmap() failed: %m\n");
        ret = -1;
        goto failed_mmap;
    }

    /* load ucm */
    printf("opening '%s'\n", filename);
    dlerror();
    handle = dlopen(filename, RTLD_NOW);
    if (handle == NULL) {
        fprintf(stderr, "failed to open %s: %s\n", filename, dlerror());
        ret = -1;
        goto failed_dlopen;
    }

    /* init ucm */
    ret = test_ucm_set_event_handler(handle);

    /* unload ucp */
    dlclose(handle);

failed_dlopen:
    /* release the memory - could break if UCM is unloaded */
    munmap(ptr2, alloc_size);
failed_mmap:
    free(ptr1);

    printf("done\n");
    return ret;
}

