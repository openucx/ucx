/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <mpi.h>

#include <ucs/sys/preprocessor.h>
#include <ucm/api/ucm.h>
#include <sys/mman.h>
#include <sys/shm.h>
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
#include <dlfcn.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

#define CHKERR_JUMP(cond, msg, label) \
    do { \
        if (cond) { \
            printf("%s:%d: %s\n", basename(__FILE__), __LINE__, msg); \
            goto label; \
        } \
    } while (0)

#define DL_FIND_FUNC(dl, func_name, func, err_action) \
    do { \
        char *error; \
        dlerror(); /* clear existing errors */ \
        func = dlsym(dl, func_name); \
        if (((error = dlerror()) != NULL) || (func == NULL)) { \
            error = error ? error : "not found"; \
            fprintf(stderr, "Failed to resolve symbol '%s': %s\n", \
                    func_name, error); \
            err_action; \
        } \
    } while (0);

#define SHMAT_FAILED ((void*)-1)

void *event_init(const char *path, ucm_mmap_hook_mode_t mmap_mode);
void *ext_event_init(const char *path, ucm_mmap_hook_mode_t mmap_mode);
void* flag_no_install_init(const char *path, ucm_mmap_hook_mode_t mmap_mode);
int malloc_hooks_run_all(void *dl);
int malloc_hooks_run_unmapped(void *dl);
int ext_event_run(void *dl);

typedef struct memtest_type {
    const char *name;
    void*      (*init)(const char *path, ucm_mmap_hook_mode_t mmap_mode);
    int        (*run) (void *arg);
} memtest_type_t;

memtest_type_t tests[] = {
    {"malloc_hooks",          event_init,           malloc_hooks_run_all},
    {"malloc_hooks_unmapped", event_init,           malloc_hooks_run_unmapped},
    {"external_events",       ext_event_init,       ext_event_run},
    {"flag_no_install",       flag_no_install_init, ext_event_run},
    {NULL}
};

static volatile size_t total_mapped = 0;
static volatile size_t total_unmapped = 0;

static void usage() {
    printf("Usage: test_memhooks [options]\n");
    printf("Options are:\n");
    printf("  -h         Print this info.\n");
    printf("  -t <name>  Test name to execute (malloc_hooks)\n");
    printf("                 malloc_hooks          : General UCM test for VM_MAPPED and VM_UNMAPPED\n");
    printf("                 malloc_hooks_unmapped : Test VM_UNMAPPED event only\n");
    printf("                 external_events       : Test of ucm_set_external_event() API\n");
    printf("                 flag_no_install       : Test of UCM_EVENT_FLAG_NO_INSTALL flag\n");
    printf(" -m <mode>   Memory hooks mode (bistro)\n");
    printf("                 reloc   : Change .plt/.got tables\n");
    printf("                 bistro  : Binary code patching\n");
    printf("\n");
}

static void event_callback(ucm_event_type_t event_type, ucm_event_t *event,
                           void *arg)
{
    if (event_type == UCM_EVENT_VM_MAPPED) {
        total_mapped += event->vm_mapped.size;
    } else if (event_type == UCM_EVENT_VM_UNMAPPED) {
        total_unmapped += event->vm_unmapped.size;
    }
}

static ucs_status_t set_event_handler(void *dl, int events)
{
    ucs_status_t (*set_handler)(int events, int priority,
                                ucm_event_callback_t cb, void *arg);

    DL_FIND_FUNC(dl, "ucm_set_event_handler", set_handler,
                 return UCS_ERR_UNSUPPORTED);

    return set_handler(events, 0, event_callback, NULL);
}

static ucs_status_t init_ucm_config(void *dl_ucm, int enable_hooks,
                                    ucm_mmap_hook_mode_t mmap_mode)
{
    void (*library_init)(const ucm_global_config_t *ucm_opts);
    ucm_global_config_t *ucm_opts;

    DL_FIND_FUNC(dl_ucm, "ucm_library_init", library_init,
                 return UCS_ERR_NO_ELEM);
    DL_FIND_FUNC(dl_ucm, "ucm_global_opts", ucm_opts,
                 return UCS_ERR_NO_ELEM);

    if (enable_hooks) {
        ucm_opts->mmap_hook_mode      = mmap_mode;
    } else {
        ucm_opts->enable_malloc_hooks = 0;
        ucm_opts->enable_malloc_reloc = 0;
        ucm_opts->mmap_hook_mode      = UCM_MMAP_HOOK_NONE;
    }

    library_init(NULL);

    return UCS_OK;
}

void* open_dyn_lib(const char *lib_path)
{
    void *dl = dlopen(lib_path, RTLD_LAZY);
    char *error;

    if (dl == NULL) {
        error = dlerror();
        error = error ? error : "unknown error";
        fprintf(stderr, "Failed to load '%s': %s\n", lib_path, error);
    }
    return dl;
}

void *event_init(const char *path, ucm_mmap_hook_mode_t mmap_mode)
{
    ucs_status_t status;
    void *dl_ucm;

    dl_ucm = open_dyn_lib(path);
    if (dl_ucm == NULL) {
        return NULL;
    }

    status = init_ucm_config(dl_ucm, 1, mmap_mode);
    CHKERR_JUMP(status != UCS_OK, "Failed to initialize UCM", fail);

    return dl_ucm;

fail:
    dlclose(dl_ucm);
    return NULL;
}

void *ext_event_init(const char *path, ucm_mmap_hook_mode_t mmap_mode)
{
    void (*set_ext_event)(int events);
    ucs_status_t status;
    void *dl_ucm;

    dl_ucm = open_dyn_lib(path);
    if (dl_ucm == NULL) {
        return NULL;
    }

    status = init_ucm_config(dl_ucm, 0, mmap_mode);
    CHKERR_JUMP(status != UCS_OK, "Failed to initialize UCM", fail);

    DL_FIND_FUNC(dl_ucm, "ucm_set_external_event", set_ext_event, goto fail);
    set_ext_event(UCM_EVENT_VM_MAPPED | UCM_EVENT_VM_UNMAPPED);

    status = set_event_handler(dl_ucm, UCM_EVENT_VM_MAPPED |
                                       UCM_EVENT_VM_UNMAPPED);
    CHKERR_JUMP(status != UCS_OK, "Failed to set event handler", fail);

    return dl_ucm;

fail:
    dlclose(dl_ucm);
    return NULL;
}

void* flag_no_install_init(const char *path, ucm_mmap_hook_mode_t mmap_mode)
{
    void *dl_ucm;
    ucs_status_t status;

    dl_ucm = open_dyn_lib(path);
    if (dl_ucm == NULL) {
        return NULL;
    }

    status = init_ucm_config(dl_ucm, 0, mmap_mode);
    CHKERR_JUMP(status != UCS_OK, "Failed to initialize UCM", fail);

    status = set_event_handler(dl_ucm, UCM_EVENT_VM_MAPPED   |
                                       UCM_EVENT_VM_UNMAPPED |
                                       UCM_EVENT_FLAG_NO_INSTALL);
    CHKERR_JUMP(status != UCS_OK, "Failed to set event handler", fail);
    return dl_ucm;

fail:
    dlclose(dl_ucm);
    return NULL;
}

int malloc_hooks_run_flags(void *dl, ucm_event_type_t events)
{
    ucs_status_t status;
    void *ptr_malloc_core = NULL;
    void *ptr_malloc_mmap = NULL;
    void *ptr_direct_mmap = MAP_FAILED;
    int  shmid            = -1;
    void *ptr_shmat       = SHMAT_FAILED;
    void *dl_test;
    const size_t size = 1024 * 1024;
    const char *lib_path = UCS_PP_MAKE_STRING(TEST_LIB_DIR) "/" "libtest_memhooks.so";
    const char *cust_mmap_name  = "memhook_test_lib_call_mmap";
    void * (*cust_mmap)(size_t size);

    status = set_event_handler(dl, events);
    CHKERR_JUMP(status != UCS_OK, "Failed to set event handler", fail_close_ucm);

    printf("Allocating memory\n");

    /* Create SysV segment */
    shmid = shmget(IPC_PRIVATE, size, IPC_CREAT|SHM_R|SHM_W);
    CHKERR_JUMP(shmid == -1, "Failed to create shared memory segment: %m",
                fail_close_ucm);

    /*
     * Test shmat/shmdt before malloc() because shmat() add entires to an internal
     * hash of pointers->size, which makes previous pointers un-releasable
     */

    /* Attach SysV segment */
    total_mapped = 0;
    ptr_shmat = shmat(shmid, NULL, 0);
    CHKERR_JUMP(ptr_shmat == SHMAT_FAILED, "Failed to attach shared memory segment",
                fail_close_ucm);
    if (events & UCM_EVENT_VM_MAPPED) {
        CHKERR_JUMP(total_mapped < size, "No callback for shmat", fail_close_ucm);
    }
    printf("After shmat: reported mapped=%zu\n", total_mapped);

    /* Detach SysV segment */
    total_unmapped = 0;
    shmdt(ptr_shmat);
    ptr_shmat = SHMAT_FAILED;
    if (events & UCM_EVENT_VM_UNMAPPED) {
        CHKERR_JUMP(total_unmapped < size, "No callback for shmdt", fail_close_ucm);
    }
    printf("After shmdt: reported unmapped=%zu\n", total_unmapped);

    /* Attach SysV segment at fixed address */
    total_mapped = 0;
    total_unmapped = 0;
    ptr_shmat = shmat(shmid, (void*)0xff000000, SHM_REMAP);
    CHKERR_JUMP(ptr_shmat == SHMAT_FAILED, "Failed to attach shared memory segment",
                fail_close_ucm);
    if (events & UCM_EVENT_VM_MAPPED) {
        CHKERR_JUMP(total_mapped < size, "No map callback for shmat(REMAP)", fail_close_ucm);
    }
    if (events & UCM_EVENT_VM_UNMAPPED) {
        CHKERR_JUMP(total_unmapped < size, "No unmap callback for shmat(REMAP)",
                    fail_close_ucm);
    }
    printf("After shmat(REMAP): reported mapped=%zu unmapped=%zu\n", total_mapped,
           total_unmapped);

    /* Detach SysV segment */
    total_unmapped = 0;
    shmdt(ptr_shmat);
    ptr_shmat = SHMAT_FAILED;
    if (events & UCM_EVENT_VM_UNMAPPED) {
        CHKERR_JUMP(total_unmapped < size, "No callback for shmdt", fail_close_ucm);
    }
    printf("After shmdt: reported unmapped=%zu\n", total_unmapped);

    /* Destroy SysV segment */
    shmctl(shmid, IPC_RMID, NULL);
    shmid = -1;

    /* Allocate using morecore */
    mallopt(M_MMAP_THRESHOLD, size * 2);
    mallopt(M_TRIM_THRESHOLD, size / 2);
    total_mapped = 0;
    ptr_malloc_core = malloc(1024 * 1024);
    if (events & UCM_EVENT_VM_MAPPED) {
        CHKERR_JUMP(total_mapped == 0, "No callback for core malloc",
                    fail_close_ucm);
    }
    printf("After core malloc: reported mapped=%zu\n", total_mapped);

    /* Allocate using mmap */
    mallopt(M_MMAP_THRESHOLD, size / 2);
    total_mapped = 0;
    ptr_malloc_mmap = malloc(2 * 1024 * 1024);
    if (events & UCM_EVENT_VM_MAPPED) {
        CHKERR_JUMP(total_mapped == 0, "No callback for mmap malloc",
                    fail_close_ucm);
    }
    printf("After mmap malloc: reported mapped=%zu\n", total_mapped);

    /* Allocate directly with mmap */
    total_mapped = 0;
    ptr_direct_mmap = mmap(NULL, size, PROT_READ|PROT_WRITE,
                           MAP_PRIVATE|MAP_ANON, -1, 0);
    if (events & UCM_EVENT_VM_MAPPED) {
        CHKERR_JUMP(total_mapped < size, "No callback for mmap", fail_close_ucm);
    }
    printf("After mmap: reported mapped=%zu\n", total_mapped);

    /* Remap */
    total_unmapped = 0;
    ptr_direct_mmap = mmap(ptr_direct_mmap, size, PROT_READ|PROT_WRITE,
                           MAP_PRIVATE|MAP_ANON|MAP_FIXED, -1, 0);
    if (events & UCM_EVENT_VM_UNMAPPED) {
        CHKERR_JUMP(total_unmapped < size, "No unmap callback for mmap(FIXED)",
                    fail_close_ucm);
    }
    printf("After mmap(FIXED): reported unmapped=%zu\n", total_unmapped);

    /* Call munmap directly */
    total_unmapped = 0;
    munmap(ptr_direct_mmap, size);
    if (events & UCM_EVENT_VM_UNMAPPED) {
        CHKERR_JUMP(total_unmapped == 0, "No callback for munmap", fail_close_ucm);
    }
    printf("After munmap: reported unmapped=%zu\n", total_unmapped);

    /* Release indirectly */
    total_unmapped = 0;
    free(ptr_malloc_mmap);
    ptr_malloc_mmap = NULL;
    malloc_trim(0);
    if (events & UCM_EVENT_VM_UNMAPPED) {
        CHKERR_JUMP(total_unmapped == 0, "No callback for munmap from free",
                    fail_close_ucm);
    }
    printf("After mmap free + trim: reported unmapped=%zu\n", total_unmapped);

    /* Call mmap from a library we load after hooks are installed */
    dl_test = open_dyn_lib(lib_path);
    CHKERR_JUMP(dl_test == NULL, "Failed to load test lib", fail_close_ucm);

    DL_FIND_FUNC(dl_test, cust_mmap_name, cust_mmap, goto fail_close_all);
    total_mapped = 0;
    ptr_direct_mmap = cust_mmap(size);
    CHKERR_JUMP(ptr_direct_mmap == MAP_FAILED, "Failed to mmap from dynamic lib",
                fail_close_all);
    if (events & UCM_EVENT_VM_MAPPED) {
        CHKERR_JUMP(total_mapped == 0,"No callback for mmap from dynamic lib",
                    fail_close_all);
    }
    printf("After another mmap from dynamic lib: reported mapped=%zu\n", total_mapped);
    munmap(ptr_direct_mmap, size);
    ptr_direct_mmap = MAP_FAILED;

    /*
     * Test closing UCM.
     * The library should not really be unloaded, because the memory hooks still
     * point to functions inside it.
     */
    total_unmapped = 0;
    dlclose(dl);
    dlclose(dl_test);
    free(ptr_malloc_core); /* This should still work */
    ptr_malloc_core = NULL;
    malloc_trim(0);
    if (events & UCM_EVENT_VM_UNMAPPED) {
        CHKERR_JUMP(total_unmapped == 0, "No callback for munmap from malloc", fail);
    }
    printf("After core malloc free: reported unmapped=%zu\n", total_unmapped);

    return 0;

fail_close_all:
    dlclose(dl_test);
fail_close_ucm:
    dlclose(dl);
fail:
    if (ptr_shmat != SHMAT_FAILED) {
        shmdt(ptr_shmat);
    }
    if (shmid != -1) {
        shmctl(shmid, IPC_RMID, NULL);
    }
    free(ptr_malloc_mmap);
    free(ptr_malloc_core);
    if (ptr_direct_mmap != MAP_FAILED) {
        munmap(ptr_direct_mmap, size);
    }

    return  -1;
}

int malloc_hooks_run_all(void *dl)
{
    return malloc_hooks_run_flags(dl, UCM_EVENT_VM_MAPPED | UCM_EVENT_VM_UNMAPPED);
}

int malloc_hooks_run_unmapped(void *dl)
{
    return malloc_hooks_run_flags(dl, UCM_EVENT_VM_UNMAPPED);
}

int ext_event_run(void *dl)
{
    void *ptr_direct_mmap;
    void (*ucm_event)(void *addr, size_t length);
    const size_t size = 1024 * 1024;
    int ret = -1;

    /* Allocate directly with mmap */
    total_mapped = 0;
    ptr_direct_mmap = mmap(NULL, size, PROT_READ|PROT_WRITE,
                           MAP_PRIVATE|MAP_ANON, -1, 0);
    printf("total_mapped=%lu\n", total_mapped);
    /* No callback should be called as we registered events to be external */
    CHKERR_JUMP(total_mapped != 0,
                "Callback for mmap invoked, while hooks were not set", fail);
    DL_FIND_FUNC(dl, "ucm_vm_mmap", ucm_event, goto fail);
    ucm_event(ptr_direct_mmap, size);
    CHKERR_JUMP(total_mapped == 0, "Callback for mmap is not called", fail);
    printf("After ucm_vm_mmap called: total_mapped=%zu\n", total_mapped);

    /* Call munmap directly */
    total_unmapped = 0;
    munmap(ptr_direct_mmap, size);
    CHKERR_JUMP(total_unmapped != 0,
                "Callback for munmap invoked, while hooks were not set\n", fail);

    DL_FIND_FUNC(dl, "ucm_vm_munmap", ucm_event, goto fail);
    ucm_event(ptr_direct_mmap, size);
    CHKERR_JUMP(total_unmapped == 0, "Callback for mmap is not called", fail);
    printf("After ucm_vm_munmap: total_unmapped=%zu\n", total_unmapped);

    ret = 0;

fail:
    dlclose(dl);
    return ret;
}

int main(int argc, char **argv)
{
    const char *ucm_path           = UCS_PP_MAKE_STRING(UCM_LIB_DIR)"/libucm.so";
    memtest_type_t *test           = tests;
    ucm_mmap_hook_mode_t mmap_mode = UCM_MMAP_HOOK_BISTRO;
    void *dl;
    int ret;
    int c;

    while ((c = getopt(argc, argv, "t:m:h")) != -1) {
        switch (c) {
        case 't':
            for (test = tests; test->name != NULL; ++test) {
                if (!strcmp(test->name, optarg)){
                    break;
                }
            }
            if (test->name == NULL) {
                fprintf(stderr, "Wrong test name %s\n", optarg);
                return -1;
            }
            break;
        case 'm':
            if (!strcasecmp(optarg, "bistro")) {
                mmap_mode = UCM_MMAP_HOOK_BISTRO;
            } else if (!strcasecmp(optarg, "reloc")) {
                mmap_mode = UCM_MMAP_HOOK_RELOC;
            } else {
                fprintf(stderr, "Wrong mmap mode %s\n", optarg);
                return -1;
            }
            break;
        case 'h':
        default:
            usage();
            return -1;
        }
    }

    /* Some tests need to modify UCM config before to call ucp_init,
     * which may be called by MPI_Init */
    dl = test->init(ucm_path, mmap_mode);
    if (dl == NULL) {
        return -1;
    }

    printf("%s: initialized\n", test->name);

    MPI_Init(&argc, &argv);

    ret = test->run(dl);

    printf("%s: %s\n", test->name, ret == 0 ? "PASS" : "FAIL");

    MPI_Finalize();
    return ret;
}



