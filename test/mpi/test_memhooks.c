/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <mpi.h>

#include <ucs/sys/preprocessor.h>
#include <ucm/api/ucm.h>
#include <sys/mman.h>
#include <malloc.h>
#include <dlfcn.h>
#include <unistd.h>

static size_t total_mapped = 0;
static size_t total_unmapped = 0;

static void event_callback(ucm_event_type_t event_type, ucm_event_t *event, void *arg)
{
    if (event_type == UCM_EVENT_VM_MAPPED) {
        total_mapped += event->vm_mapped.size;
    } else if (event_type == UCM_EVENT_VM_UNMAPPED) {
        total_unmapped += event->vm_unmapped.size;
    }
}

static void* set_hooks()
{
    const char *libucm_path = UCS_PP_MAKE_STRING(UCM_LIB_DIR) "/" "libucm.so";
    const char *func_name   = "ucm_set_event_handler";
    ucs_status_t (*func)(int events, int priority,
                         ucm_event_callback_t cb, void *arg);
    void *dl;

    /* Load UCM dynamically, to simulate what an MPI could be doing */
    dl = dlopen(libucm_path, RTLD_LAZY);
    if (dl == NULL) {
        fprintf(stderr, "Failed to load '%s': %m\n", libucm_path);
        return NULL;
    }

    func = dlsym(dl, func_name);
    if (func == NULL) {
        fprintf(stderr, "Failed to resolve symbol '%s': %m\n", func_name);
        dlclose(dl);
        return NULL;
    }

    func(UCM_EVENT_VM_MAPPED | UCM_EVENT_VM_UNMAPPED, 0, event_callback, NULL);
    return dl;
}

static void* call_mmap_from_loaded_lib(size_t size)
{
    const char *lib_path = UCS_PP_MAKE_STRING(TEST_LIB_DIR) "/" "libtest_memhooks.so";
    const char *func_name   = "memhook_test_lib_call_mmap";
    void * (*func)(size_t size);
    void *dl;

    /* Load library dynamically, to simulate additional objects which could be
     * loaded at runtime. */
    dl = dlopen(lib_path, RTLD_LAZY);
    if (dl == NULL) {
        fprintf(stderr, "Failed to load '%s': %m\n", lib_path);
        return MAP_FAILED;
    }

    func = dlsym(dl, func_name);
    if (func == NULL) {
        fprintf(stderr, "Failed to resolve symbol '%s': %m\n", func_name);
        dlclose(dl);
        return MAP_FAILED;
    }

    return func(size);
}

/*
 * TODO add to jenkins
 */

int main(int argc, char **argv)
{
    const size_t size = 1024 * 1024;
    void *ptr_malloc_core;
    void *ptr_malloc_mmap;
    void *ptr_direct_mmap;
    void *dl;
    int ret;

    ret = MPI_Init(&argc, &argv);
    if (ret != 0) {
        return ret;
    }

    printf("MPI_Init() done\n");

    dl = set_hooks();
    if (dl == NULL) {
        goto fail;
    }

    printf("Allocating memory\n");

    /* Allocate using morecore */
    mallopt(M_MMAP_THRESHOLD, size * 2);
    mallopt(M_TRIM_THRESHOLD, size / 2);
    total_mapped = 0;
    ptr_malloc_core = malloc(1024 * 1024);
    if (total_mapped == 0) {
        printf("No callback for core malloc\n");
        goto fail;
    }
    printf("After core malloc: mapped=%zu\n", total_mapped);

    /* Allocate using mmap */
    mallopt(M_MMAP_THRESHOLD, size / 2);
    total_mapped = 0;
    ptr_malloc_mmap = malloc(2 * 1024 * 1024);
    if (total_mapped == 0) {
        printf("No callback for mmap malloc\n");
        goto fail;
    }
    printf("After mmap malloc: mapped=%zu\n", total_mapped);

    /* Allocate directly with mmap */
    total_mapped = 0;
    ptr_direct_mmap = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON,
                           -1, 0);
    if (total_mapped == 0) {
        printf("No callback for mmap\n");
        goto fail;
    }
    printf("After mmap: mapped=%zu\n", total_mapped);

    /* Call munmap directly */
    total_unmapped = 0;
    munmap(ptr_direct_mmap, size);
    if (total_unmapped == 0) {
        printf("No callback for munmap\n");
        goto fail;
    }
    printf("After munmap: unmapped=%zu\n", total_unmapped);

    /* Release indirectly */
    total_unmapped = 0;
    free(ptr_malloc_mmap);
    malloc_trim(0);
    if (total_unmapped == 0) {
        printf("No callback for munmap from malloc\n");
        goto fail;
    }
    printf("After mmap free + trim: unmapped=%zu\n", total_unmapped);

    /* Call mmap from a library we load after hooks are installed */
    total_mapped = 0;
    ptr_direct_mmap = call_mmap_from_loaded_lib(size);
    if (ptr_direct_mmap == MAP_FAILED) {
        printf("Failed to mmap from synamic lib\n");
        goto fail;
    }
    if (total_mapped == 0) {
        printf("No callback for mmap from dynamic lib\n");
        goto fail;
    }
    printf("After another mmap from dynamic lib: mapped=%zu\n", total_mapped);
    munmap(ptr_direct_mmap, size);

    /*
     * Test closing UCM.
     * The library should not really be unloaded, because the meory hooks still
     * point to functions inside it.
     */
    dlclose(dl);
    free(ptr_malloc_core); /* This should still work */

    printf("PASS\n");
    return MPI_Finalize();

fail:
    printf("FAIL\n");
    MPI_Finalize();
    return -1;

}
