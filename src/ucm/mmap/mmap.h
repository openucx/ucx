/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_MMAP_H_
#define UCM_MMAP_H_

#include <ucm/api/ucm.h>
#include <ucm/util/sys.h>
#include <ucs/sys/checker.h>

#define UCM_MMAP_HOOK_RELOC_STR  "reloc"
#define UCM_MMAP_HOOK_BISTRO_STR "bistro"

#if UCM_BISTRO_HOOKS
#  define UCM_DEFAULT_HOOK_MODE UCM_MMAP_HOOK_BISTRO
#  define UCM_DEFAULT_HOOK_MODE_STR UCM_MMAP_HOOK_BISTRO_STR
#else
#  define UCM_DEFAULT_HOOK_MODE UCM_MMAP_HOOK_RELOC
#  define UCM_DEFAULT_HOOK_MODE_STR UCM_MMAP_HOOK_RELOC_STR
#endif

/* Functional GPU (cuda/rocm) hook modes, i.e. excluding UCM_MMAP_HOOK_NONE,
 * which occupies bit 0 and is therefore never equal to a zero hook-modes
 * bitmap. */
#define UCM_GPU_HOOK_MODES_MASK \
    (UCS_BIT(UCM_MMAP_HOOK_BISTRO) | UCS_BIT(UCM_MMAP_HOOK_RELOC))

/* Reserve UCS_ERR_UNSUPPORTED for "disabled by configuration", so a real
 * patch failure on an enabled backend isn't mistaken for that and skipped. */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucm_gpu_hook_install_status(ucs_status_t status)
{
    return (status == UCS_ERR_UNSUPPORTED) ? UCS_ERR_IO_ERROR : status;
}

ucs_status_t ucm_mmap_install(int events, int exclusive);

void *ucm_override_mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
int ucm_override_munmap(void *addr, size_t length);
void *ucm_override_mremap(void *old_address, size_t old_size, size_t new_size,
                          int flags, void *new_address);
void *ucm_override_shmat(int shmid, const void *shmaddr, int shmflg);
int ucm_override_shmdt(const void *shmaddr);
void *ucm_override_sbrk(intptr_t increment);
void *ucm_sbrk_select(intptr_t increment);
int ucm_override_brk(void *addr);
int ucm_override_madvise(void *addr, size_t length, int advice);
void *ucm_get_current_brk();
void ucm_fire_mmap_events(int events);
ucs_status_t ucm_mmap_test_installed_events(int events);
ucs_status_t ucm_mmap_test_events(int events, const char *event_type);
void ucm_mmap_init();


/**
 * Memory hooks mode names.
 */
extern const char *ucm_mmap_hook_modes[];


static UCS_F_ALWAYS_INLINE ucm_mmap_hook_mode_t ucm_mmap_hook_mode(void)
{
    return ucm_get_hook_mode(ucm_global_opts.mmap_hook_mode);
}

#endif
