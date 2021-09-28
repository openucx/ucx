/**
 * Copyright (C) Hiroyuki Sato. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_EVENT_SET_H
#define UCS_EVENT_SET_H

#include <ucs/type/status.h>

#include <stdint.h>

BEGIN_C_DECLS

/**
 * ucs_sys_event_set_t structure used in ucs_event_set_XXX functions.
 *
 */
typedef struct ucs_sys_event_set ucs_sys_event_set_t;

/**
 * Bit set composed using the available event types
 */
typedef uint8_t ucs_event_set_types_t;

/**
 * ucs_event_set_handler call this handler for notifying event
 *
 * @param [in] callback_data  User data which set in ucs_event_set_add().
 * @param [in] events         Detection event. Sets of ucs_event_set_types_t.
 * @param [in] arg            User data which set in ucs_event_set_wait().
 *
 */
typedef void (*ucs_event_set_handler_t)(void *callback_data,
                                        ucs_event_set_types_t events,
                                        void *arg);

/**
 * Event types that could be requested to notify
 */
typedef enum {
    UCS_EVENT_SET_EVREAD         = UCS_BIT(0),
    UCS_EVENT_SET_EVWRITE        = UCS_BIT(1),
    UCS_EVENT_SET_EVERR          = UCS_BIT(2),
    UCS_EVENT_SET_EDGE_TRIGGERED = UCS_BIT(3)
} ucs_event_set_type_t;

/* The maximum possible number of events based on system constraints */
extern const unsigned ucs_sys_event_set_max_wait_events;

/**
 * Allocate ucs_sys_event_set_t structure and assign provided file
 * descriptor to wait for events on.
 *
 * @param [out] event_set_p  Event set pointer to initialize.
 * @param [in]  event_fd     File descriptor to wait for events on.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_create_from_fd(ucs_sys_event_set_t **event_set_p,
                                          int event_fd);

/**
 * Allocate ucs_sys_event_set_t structure.
 *
 * @param [out] event_set_p  Event set pointer to initialize.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_create(ucs_sys_event_set_t **event_set_p);

/**
 * Register the target event.
 *
 * @param [in] event_set_p   Event set pointer to initialize.
 * @param [in] fd            Register the target file descriptor fd.
 * @param [in] events        Operation events.
 * @param [in] callback_data ucs_event_set_handler_t accepts this data.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_add(ucs_sys_event_set_t *event_set, int fd,
                               ucs_event_set_types_t events,
                               void *callback_data);

/**
 * Modify the target event.
 *
 * @param [in] event_set     Event set created by ucs_event_set_create.
 * @param [in] fd            Register the target file descriptor fd.
 * @param [in] events        Operation events.
 * @param [in] callback_data ucs_event_set_handler_t accepts this data.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_mod(ucs_sys_event_set_t *event_set, int fd,
                               ucs_event_set_types_t events,
                               void *callback_data);

/**
 * Remove the target event.
 *
 * @param [in] event_set    Event set created by ucs_event_set_create.
 * @param [in] fd           Register the target file descriptor fd.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_del(ucs_sys_event_set_t *event_set, int fd);

/**
 * Wait for an I/O events
 *
 * @param [in]     event_set          Event set created by ucs_event_set_create.
 * @param [in/out] num_events         Number of expected/read events.
 * @param [in]     timeout_ms         Timeout period in ms.
 * @param [in]     event_set_handler  Callback functions.
 * @param [in]     arg                User data variables.
 *
 * @return return UCS_OK on success, UCS_INPROGRESS - call was interrupted by a
 *         signal handler, UCS_ERR_IO_ERROR - an error occurred during waiting
 *         for I/O events.
 */
ucs_status_t ucs_event_set_wait(ucs_sys_event_set_t *event_set,
                                unsigned *num_events, int timeout_ms,
                                ucs_event_set_handler_t event_set_handler,
                                void *arg);

/**
 * Cleanup event set
 *
 * @param [in] event_set    Event set created by ucs_event_set_create.
 *
 */
void ucs_event_set_cleanup(ucs_sys_event_set_t *event_set);

/**
 * Get file descriptor for watching events.
 *
 * @param [in]  event_set    Event set created by ucs_event_set_create.
 * @param [out] event_fd_p   File descriptor that is used by Event set to wait
 *                           for events on.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_fd_get(ucs_sys_event_set_t *event_set,
                                  int *event_fd_p);

END_C_DECLS

#endif
