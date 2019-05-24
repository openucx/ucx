/**
 * Copyright (C) Hiroyuki Sato. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_EVENT_SET_H
#define UCS_EVENT_SET_H

#include <ucs/type/status.h>

/**
 * ucs_sys_event_set structure used in ucs_event_set_XXX functions.
 *
 */
typedef struct ucs_sys_event_set ucs_sys_event_set_t;

/**
 * ucs_event_set_extra_arg used in ucs_event_set_wait callback function.
 *
 */
typedef void* ucs_event_set_extra_arg;

/**
 * ucs_event_set_handler call this handler for notifying event
 *
 */
typedef void (*ucs_event_set_handler_t)(int fd, int event,
                                        ucs_event_set_extra_arg *args);

/**
 * ucs_event_set_type_t member is a bit set composed using the following
 * available event types
 */
typedef enum {
    UCS_EVENT_SET_EVREAD  = UCS_BIT(0),
    UCS_EVENT_SET_EVWRITE = UCS_BIT(1),
    UCS_EVENT_SET_EVNONE =  UCS_BIT(2)
} ucs_event_set_type_t;

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
 * @param [in] event_set_p  Event set pointer to initialize.
 * @param [in] event_fd     Register the target file descriptor fd.
 * @param [in] events       Operation events.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_add(ucs_sys_event_set_t *event_set, int event_fd,
                               ucs_event_set_type_t events);

/**
 * Modify the target event.
 *
 * @param [in] event_set    Event set created by ucs_event_set_create.
 * @param [in] event_fd     Register the target file descriptor fd.
 * @param [in] events       Operation events.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_mod(ucs_sys_event_set_t *event_set, int event_fd,
                               ucs_event_set_type_t events);

/**
 * Remove the target event.
 *
 * @param [in] event_set    Event set created by ucs_event_set_create.
 * @param [in] event_fd     Register the target file descriptor fd.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_del(ucs_sys_event_set_t *event_set, int event_fd);

/**
 * Wait for an I/O events
 *
 * @param [in] event_set          Event set created by ucs_event_set_create.
 * @param [in] timeout_ms         Timeout period in ms.
 * @param [in] event_set_handler  Callback functions.
 *
 * @return UCS_OK on success or an error code on failure.
 */
ucs_status_t ucs_event_set_wait(ucs_sys_event_set_t *event_set, int timeout_ms,
                                ucs_event_set_handler_t event_set_handler,
                                ucs_event_set_extra_arg *args);

/**
 * Cleanup event set
 *
 * @param [in] event_set    Event set created by ucs_event_set_create.
 *
 */
void ucs_event_set_cleanup(ucs_sys_event_set_t *event_set);

#endif
