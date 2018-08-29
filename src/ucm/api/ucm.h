/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCM_H_
#define UCM_H_

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

#include <ucs/config/types.h>
#include <ucs/type/status.h>

#include <sys/types.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>


/**
 * @brief Memory event types
 */
typedef enum ucm_event_type {
    /* Native events */
    UCM_EVENT_MMAP            = UCS_BIT(0),
    UCM_EVENT_MUNMAP          = UCS_BIT(1),
    UCM_EVENT_MREMAP          = UCS_BIT(2),
    UCM_EVENT_SHMAT           = UCS_BIT(3),
    UCM_EVENT_SHMDT           = UCS_BIT(4),
    UCM_EVENT_SBRK            = UCS_BIT(5),
    UCM_EVENT_MADVISE         = UCS_BIT(6),

    /* Aggregate events */
    UCM_EVENT_VM_MAPPED       = UCS_BIT(16),
    UCM_EVENT_VM_UNMAPPED     = UCS_BIT(17),

    /* Non-accessible memory alloc/free events */
    UCM_EVENT_MEM_TYPE_ALLOC  = UCS_BIT(20),
    UCM_EVENT_MEM_TYPE_FREE   = UCS_BIT(21),

    /* Auxiliary flags */
    UCM_EVENT_FLAG_NO_INSTALL = UCS_BIT(24)

} ucm_event_type_t;

/**
 * @brief Memory types for alloc and free events
 */
typedef enum ucm_mem_type {
    /*cuda memory */
    UCM_MEM_TYPE_CUDA         = UCS_BIT(0),
    UCM_MEM_TYPE_CUDA_MANAGED = UCS_BIT(1)
} ucm_mem_type_t;


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
        void           *result;
        void           *address;
        size_t         size;
        int            prot;
        int            flags;
        int            fd;
        off_t          offset;
    } mmap;

    /*
     * UCM_EVENT_MUNMAP
     * munmap() is called.
     */
    struct {
        int            result;
        void           *address;
        size_t         size;
    } munmap;

    /*
     * UCM_EVENT_MREMAP
     * mremap() is called.
     */
    struct {
        void           *result;
        void           *address;
        size_t         old_size;
        size_t         new_size;
        int            flags;
    } mremap;

    /*
     * UCM_EVENT_SHMAT
     * shmat() is called.
     */
    struct {
        void           *result;
        int            shmid;
        const void     *shmaddr;
        int            shmflg;
    } shmat;

    /*
     * UCM_EVENT_SHMDT
     * shmdt() is called.
     */
    struct {
        int            result;
        const void     *shmaddr;
    } shmdt;

    /*
     * UCM_EVENT_SBRK
     * sbrk() is called.
     */
    struct {
        void           *result;
        intptr_t       increment;
    } sbrk;

    /*
     * UCM_EVENT_MADVISE
     * madvise() is called.
     */
    struct {
        int            result;
        void           *addr;
        size_t         length;
        int            advice;
    } madvise;

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
        void           *address;
        size_t         size;
    } vm_mapped, vm_unmapped;

    /*
     * memory type allocation and deallocation event
     */
    struct {
        void           *address;
        size_t         size;
        ucm_mem_type_t mem_type;
    } mem_type;

} ucm_event_t;


/**
 * @brief Global UCM configuration.
 *
 * Can be safely modified before using UCM functions.
 */
typedef struct ucm_global_config {
    ucs_log_level_t log_level;                   /* Logging level */
    int             enable_events;               /* Enable memory events */
    int             enable_mmap_reloc;           /* Enable installing mmap relocations */
    int             enable_malloc_hooks;         /* Enable installing malloc hooks */
    int             enable_malloc_reloc;         /* Enable installing malloc relocations */
    int             enable_cuda_reloc;           /* Enable installing CUDA relocations */
    int             enable_dynamic_mmap_thresh;  /* Enable adaptive mmap threshold */
    size_t          alloc_alignment;             /* Alignment for memory allocations */
    int             enable_syscall;              /* Use syscalls when possible to implement
                                                    the functionality of replaced libc routines */
} ucm_global_config_t;


/* Global UCM configuration */
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
 * @brief Call the original implementation of @ref ucm_madvise and all handlers
 * associated with it.
 */
int ucm_madvise(void *addr, size_t length, int advice);


END_C_DECLS

#endif
