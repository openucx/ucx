/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_H_
#define UCM_H_

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

#include <ucs/config/types.h>
#include <ucs/memory/memory_type.h>
#include <ucs/type/status.h>

#include <sys/types.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>


/**
 * @brief Memory event types
 */
typedef enum ucm_event_type {
    /* Default initialization value */
    UCM_EVENT_NONE            = 0,
    /* Native events */
    UCM_EVENT_MMAP            = UCS_BIT(0),
    UCM_EVENT_MUNMAP          = UCS_BIT(1),
    UCM_EVENT_MREMAP          = UCS_BIT(2),
    UCM_EVENT_SHMAT           = UCS_BIT(3),
    UCM_EVENT_SHMDT           = UCS_BIT(4),
    UCM_EVENT_SBRK            = UCS_BIT(5),
    UCM_EVENT_MADVISE         = UCS_BIT(6),
    UCM_EVENT_BRK             = UCS_BIT(7),

    /* Aggregate events */
    UCM_EVENT_VM_MAPPED       = UCS_BIT(16),
    UCM_EVENT_VM_UNMAPPED     = UCS_BIT(17),

    /* Non-accessible memory alloc/free events */
    UCM_EVENT_MEM_TYPE_ALLOC  = UCS_BIT(20),
    UCM_EVENT_MEM_TYPE_FREE   = UCS_BIT(21),

    /* Add event handler, but don't install new hooks */
    UCM_EVENT_FLAG_NO_INSTALL = UCS_BIT(24),

    /* When the event handler is added, generate approximated events for
     * existing memory allocations.
     * Currently implemented only for @ref UCM_EVENT_MEM_TYPE_ALLOC.
     */
    UCM_EVENT_FLAG_EXISTING_ALLOC = UCS_BIT(25)

} ucm_event_type_t;


/**
 * @brief MMAP hook modes
 */
typedef enum ucm_mmap_hook_mode {
    UCM_MMAP_HOOK_NONE,
    UCM_MMAP_HOOK_RELOC,
    UCM_MMAP_HOOK_BISTRO,
    UCM_MMAP_HOOK_LAST
} ucm_mmap_hook_mode_t;


/**
 * @brief UCM module unload prevent mode
 */
typedef enum ucm_module_unload_prevent_mode {
    UCM_UNLOAD_PREVENT_MODE_LAZY,
    UCM_UNLOAD_PREVENT_MODE_NOW,
    UCM_UNLOAD_PREVENT_MODE_NONE,
    UCM_UNLOAD_PREVENT_MODE_LAST
} ucm_module_unload_prevent_mode_t;


/**
 * @brief Memory event parameters and result.
 */
typedef union ucm_event {
    /*
     * UCM_EVENT_MMAP
     * mmap() is called.
     * callbacks: pre, post
     */
    struct {
        void               *result;
        void               *address;
        size_t             size;
        int                prot;
        int                flags;
        int                fd;
        off_t              offset;
    } mmap;

    /*
     * UCM_EVENT_MUNMAP
     * munmap() is called.
     */
    struct {
        int                result;
        void               *address;
        size_t             size;
    } munmap;

    /*
     * UCM_EVENT_MREMAP
     * mremap() is called.
     */
    struct {
        void               *result;
        void               *address;
        size_t             old_size;
        size_t             new_size;
        int                flags;
    } mremap;

    /*
     * UCM_EVENT_SHMAT
     * shmat() is called.
     */
    struct {
        void               *result;
        int                shmid;
        const void         *shmaddr;
        int                shmflg;
    } shmat;

    /*
     * UCM_EVENT_SHMDT
     * shmdt() is called.
     */
    struct {
        int                result;
        const void         *shmaddr;
    } shmdt;

    /*
     * UCM_EVENT_SBRK
     * sbrk() is called.
     */
    struct {
        void               *result;
        intptr_t           increment;
    } sbrk;

    /*
     * UCM_EVENT_MADVISE
     * madvise() is called.
     */
    struct {
        int                result;
        void               *addr;
        size_t             length;
        int                advice;
    } madvise;

    /*
     * UCM_EVENT_BRK
     * brk() is called.
     */
    struct {
        int                result;
        void               *addr;
    } brk;

    /*
     * UCM_EVENT_VM_MAPPED, UCM_EVENT_VM_UNMAPPED
     *
     * This is a "read-only" event which is called whenever memory is mapped
     * or unmapped from process address space, in addition to the other events.
     * It can return only UCM_EVENT_STATUS_NEXT.
     *
     * For UCM_EVENT_VM_MAPPED, callbacks are post
     * For UCM_EVENT_VM_UNMAPPED, callbacks are pre
     */
    struct {
        void               *address;
        size_t             size;
    } vm_mapped, vm_unmapped;

    /*
     * UCM_EVENT_MEM_TYPE_ALLOC, UCM_EVENT_MEM_TYPE_FREE
     *
     * Memory type allocation and deallocation event.
     * If mem_type is @ref UCS_MEMORY_TYPE_LAST, the memory type is unknown, and
     * further memory type detection is required.
     */
    struct {
        void               *address;
        size_t             size;
        ucs_memory_type_t  mem_type;
    } mem_type;

} ucm_event_t;


/**
 * @brief Global UCM configuration.
 *
 * Can be safely modified before using UCM functions.
 */
typedef struct ucm_global_config {
    ucs_log_level_t      log_level;                   /* Logging level */
    int                  enable_events;               /* Enable memory events */
    ucm_mmap_hook_mode_t mmap_hook_mode;              /* MMAP hook mode */
    int                  enable_malloc_hooks;         /* Enable installing malloc hooks */
    int                  enable_malloc_reloc;         /* Enable installing malloc relocations */
    ucm_mmap_hook_mode_t cuda_hook_mode;              /* Cuda hooks mode */
    int                  enable_dynamic_mmap_thresh;  /* Enable adaptive mmap threshold */
    size_t               alloc_alignment;             /* Alignment for memory allocations */
    int                  dlopen_process_rpath;        /* Process RPATH section in dlopen hook */
    int                  module_unload_prevent_mode;  /* Module unload prevention mode */
} ucm_global_config_t;


/*
 * Global UCM configuration to be set externally.
 * @deprecated replaced by @ref ucm_library_init.
 */
extern ucm_global_config_t ucm_global_opts;


/**
 * @brief Memory event callback.
 *
 *  This type describes a callback which handles memory events in the current process.
 *
 * @param [in]     event_type  Type of the event being fired. see @ref ucm_event_type_t.
 * @param [inout]  event       Event information. This structure can be updated by
 *                               this callback, as described below.
 * @param [in]     arg         User-defined argument as passed to @ref ucm_set_event_handler.
 *
 *
 *  Events are dispatched in order of callback priority (low to high).
 *
 * The fields of the relevant part of the union are initialized as follows:
 *  - "result" - to an invalid erroneous return value (depends on the specific event).
 *  - the rest - to the input parameters of the event.
 *
 *  The callback is allowed to modify the fields, and those modifications will
 * be passed to the next callback. Also, the callback is allowed to modify the
 * result, but **only if it's currently invalid**. A valid result indicates that
 * a previous callback already performed the requested memory operation, so a
 * callback should **refrain from actions with side-effects** in this case.
 *
 *  If the result is still invalid after all callbacks are called, the parameters,
 * possibly modified by the callbacks, will be passed to the original handler.
 *
 *
 * Important Note: The callback must not call any memory allocation routines, or
 *       anything which may trigger or wait for memory allocation, because it
 *       may lead to deadlock or infinite recursion.
 *
 * @todo describe use cases
 *
 */
typedef void (*ucm_event_callback_t)(ucm_event_type_t event_type,
                                     ucm_event_t *event, void *arg);


/**
 * Initialize UCM library and set its configuration.
 *
 * @param [in]  ucm_opts   UCM library global configuration. If NULL, default
 *                         configuration is applied.
 *
 * @note Calling this function more than once in the same process has no effect.
 */
void ucm_library_init(const ucm_global_config_t *ucm_opts);


/**
 * @brief Install a handler for memory events.
 *
 * @param [in]  events     Bit-mask of events to handle.
 * @param [in]  priority   Priority value which defines the order in which event
 *                          callbacks are called.
 *                           <  0 - called before the original implementation,
 *                           >= 0 - called after the original implementation.
 * @param [in]  cb         Event-handling callback.
 * @param [in]  arg        User-defined argument for the callback.
 *
 * @note If UCM_EVENT_FLAG_NO_INSTALL flag is passed in @a events argument,
 *       only @cb handler will be registered for @a events. No memory
 *       events/hooks will be installed.
 *
 * @return Status code.
 */
ucs_status_t ucm_set_event_handler(int events, int priority,
                                   ucm_event_callback_t cb, void *arg);


/**
 * @brief Remove a handler for memory events.
 *
 * @param [in]  events     Which events to remove. The handler is removed
 *                          completely when all its events are removed.
 * @param [in]  cb         Event-handling callback.
 * @param [in]  arg        User-defined argument for the callback.
 */
void ucm_unset_event_handler(int events, ucm_event_callback_t cb, void *arg);


/**
 * @brief Add memory events to the external events list.
 *
 * When the event is set to be external, it means that user is responsible for
 * handling it. So, setting a handler for external event will not trigger
 * installing of UCM memory hooks (if they were not installed before). In this
 * case the corresponding UCM function needs to be invoked to trigger event
 * handlers.
 * Usage example is when the user disables UCM memory hooks (he may have its
 * own hooks, like Open MPI), but it wants to use some UCM based functionality,
 * e.g. IB registration cache. IB registration cache needs to be notified about
 * UCM_EVENT_VM_UNMAPPED events, therefore it adds specific handler for it.
 * In this case user needs to declare UCM_EVENT_VM_UNMAPPED event as external
 * and explicitly call ucm_vm_munmap() when some memory release operation
 * occurs.
 *
 * @param [in]  events    Bit-mask of events which are supposed to be handled
 *                        externally.
 *
 * @note To take an effect, the event should be set external prior to adding
 *       event handlers for it.
 */
void ucm_set_external_event(int events);


/**
 * @brief Remove memory events from the external events list.
 *
 * When the event is removed from the external events list, any subsequent call
 * to ucm_set_event_handler() for that event will trigger installing of UCM
 * memory hooks (if they are enabled and were not installed before).
 *
 * @param [in]  events     Which events to remove from the external events list.
 */
void ucm_unset_external_event(int events);


/**
 * @brief Test event handlers
 *
 * This routine checks if event handlers are called when corresponding system API
 * is invoked.
 *
 * @param [in]  events    Bit-mask of events which are supposed to be handled
 *                        externally.
 *
 * @return Status code.
 */
ucs_status_t ucm_test_events(int events);


/**
 * @brief Test event external handlers
 *
 * This routine checks if external events, as set by @ref ucm_set_external_event,
 * are actually being reported (by calling APIs such as @ref ucm_vm_munmap).
 *
 * @param [in]  events    Bit-mask of events which are supposed to be handled
 *                        externally.
 *
 * @return Status code.
 */
ucs_status_t ucm_test_external_events(int events);


/**
 * @brief Call the original implementation of @ref mmap without triggering events.
 */
void *ucm_orig_mmap(void *addr, size_t length, int prot, int flags, int fd,
                    off_t offset);


/**
 * @brief Call the original implementation of @ref munmap without triggering events.
 */
int ucm_orig_munmap(void *addr, size_t length);


/**
 * @brief Call the original implementation of @ref mremap without triggering events.
 */
void *ucm_orig_mremap(void *old_address, size_t old_size, size_t new_size,
                      int flags);


/**
 * @brief Call the original implementation of @ref shmat without triggering events.
 */
void *ucm_orig_shmat(int shmid, const void *shmaddr, int shmflg);


/**
 * @brief Call the original implementation of @ref shmdt without triggering events.
 */
int ucm_orig_shmdt(const void *shmaddr);


/**
 * @brief Call the original implementation of @ref sbrk without triggering events.
 */
void *ucm_orig_sbrk(intptr_t increment);


/**
 * @brief Call the original implementation of @ref brk without triggering events.
 */
int ucm_orig_brk(void *addr);


/**
 * @brief Call the original implementation of @ref madvise without triggering events.
 */
int ucm_orig_madvise(void *addr, size_t length, int advice);


/**
 * @brief Call the original implementation of @ref mmap and all handlers
 * associated with it.
 */
void *ucm_mmap(void *addr, size_t length, int prot, int flags, int fd,
               off_t offset);


/**
 * @brief Call the original implementation of @ref munmap and all handlers
 * associated with it.
 */
int ucm_munmap(void *addr, size_t length);


/**
 * @brief Call the handlers registered for aggregated VM_MMAP event.
 */
void ucm_vm_mmap(void *addr, size_t length);


/**
 * @brief Call the handlers registered for aggregated VM_MUNMAP event.
 */
void ucm_vm_munmap(void *addr, size_t length);


/**
 * @brief Call the original implementation of @ref mremap and all handlers
 * associated with it.
 */
void *ucm_mremap(void *old_address, size_t old_size, size_t new_size, int flags);


/**
 * @brief Call the original implementation of @ref shmat and all handlers
 * associated with it.
 */
void *ucm_shmat(int shmid, const void *shmaddr, int shmflg);


/**
 * @brief Call the original implementation of @ref shmdt and all handlers
 * associated with it.
 */
int ucm_shmdt(const void *shmaddr);


/**
 * @brief Call the original implementation of @ref sbrk and all handlers
 * associated with it.
 */
void *ucm_sbrk(intptr_t increment);


/**
 * @brief Call the original implementation of @ref brk and all handlers
 * associated with it.
 */
int ucm_brk(void *addr);


/**
 * @brief Call the original implementation of @ref madvise and all handlers
 * associated with it.
 */
int ucm_madvise(void *addr, size_t length, int advice);


/**
 * @brief Call the original implementation of @ref dlopen and all handlers
 * associated with it.
 */
void *ucm_dlopen(const char *filename, int flag);


END_C_DECLS

#endif
