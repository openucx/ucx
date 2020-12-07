/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CONFIG_TYPES_H
#define UCS_CONFIG_TYPES_H

#include <ucs/sys/compiler_def.h>
#include <sys/socket.h>

/**
 * Logging levels.
 */
typedef enum {
    UCS_LOG_LEVEL_FATAL,        /* Immediate termination */
    UCS_LOG_LEVEL_ERROR,        /* Error is returned to the user */
    UCS_LOG_LEVEL_WARN,         /* Something's wrong, but we continue */
    UCS_LOG_LEVEL_DIAG,         /* Diagnostics, silent adjustments or internal error handling */
    UCS_LOG_LEVEL_INFO,         /* Information */
    UCS_LOG_LEVEL_DEBUG,        /* Low-volume debugging */
    UCS_LOG_LEVEL_TRACE,        /* High-volume debugging */
    UCS_LOG_LEVEL_TRACE_REQ,    /* Every send/receive request */
    UCS_LOG_LEVEL_TRACE_DATA,   /* Data sent/received on the transport */
    UCS_LOG_LEVEL_TRACE_ASYNC,  /* Asynchronous progress engine */
    UCS_LOG_LEVEL_TRACE_FUNC,   /* Function calls */
    UCS_LOG_LEVEL_TRACE_POLL,   /* Polling functions */
    UCS_LOG_LEVEL_LAST,
    UCS_LOG_LEVEL_PRINT         /* Temporary output */
} ucs_log_level_t;


/**
 * Async progress mode.
 */
typedef enum {
    UCS_ASYNC_MODE_SIGNAL,
    UCS_ASYNC_MODE_THREAD,     /* Deprecated, keep for backward compatibility */
    UCS_ASYNC_MODE_THREAD_SPINLOCK = UCS_ASYNC_MODE_THREAD,
    UCS_ASYNC_MODE_THREAD_MUTEX,
    UCS_ASYNC_MODE_POLL,       /* TODO keep only in debug version */
    UCS_ASYNC_MODE_LAST
} ucs_async_mode_t;


extern const char *ucs_async_mode_names[];


/**
 * Ternary logic or Auto value.
 */
typedef enum ucs_ternary_auto_value {
    UCS_NO   = 0,
    UCS_YES  = 1,
    UCS_TRY  = 2,
    UCS_AUTO = 3,
    UCS_TERNARY_LAST
} ucs_ternary_auto_value_t;


/**
 * On/Off/Auto logic value.
 */
typedef enum ucs_on_off_auto_value {
    UCS_CONFIG_OFF  = 0,
    UCS_CONFIG_ON   = 1,
    UCS_CONFIG_AUTO = 2,
    UCS_CONFIG_ON_OFF_LAST
} ucs_on_off_auto_value_t;


/**
 * Error handling modes
 */
typedef enum {
    UCS_HANDLE_ERROR_BACKTRACE, /* Print backtrace */
    UCS_HANDLE_ERROR_FREEZE,    /* Freeze and wait for a debugger */
    UCS_HANDLE_ERROR_DEBUG,     /* Attach debugger */
    UCS_HANDLE_ERROR_LAST
} ucs_handle_error_t;


/**
 * Configuration printing flags
 */
typedef enum {
    UCS_CONFIG_PRINT_CONFIG        = UCS_BIT(0),
    UCS_CONFIG_PRINT_HEADER        = UCS_BIT(1),
    UCS_CONFIG_PRINT_DOC           = UCS_BIT(2),
    UCS_CONFIG_PRINT_HIDDEN        = UCS_BIT(3)
} ucs_config_print_flags_t;


/**
 * Structure type for array configuration. Should be used inside the configuration
 * structure declaration.
 */
#define UCS_CONFIG_ARRAY_FIELD(_type, _array_name) \
    struct { \
        _type    *_array_name; \
        unsigned count; \
        unsigned pad; \
    }


/* Specific structure for an array of strings */
#define UCS_CONFIG_STRING_ARRAY_FIELD(_array_name) \
    UCS_CONFIG_ARRAY_FIELD(char*, _array_name)


typedef UCS_CONFIG_STRING_ARRAY_FIELD(names) ucs_config_names_array_t;


/**
 * Enum for representing possible modes of an "allow-list"
 */
typedef enum {
    UCS_CONFIG_ALLOW_LIST_ALLOW_ALL, /* Allow all possible options */
    UCS_CONFIG_ALLOW_LIST_ALLOW, /* Allow only the specified options */
    UCS_CONFIG_ALLOW_LIST_NEGATE /* Negate (forbid) the specified options */
} ucs_config_allow_list_mode_t;


typedef struct {
    ucs_config_names_array_t array;
    ucs_config_allow_list_mode_t mode;
} ucs_config_allow_list_t;


/**
 * @ingroup UCS_RESOURCE
 * BSD socket address specification.
 */
typedef struct ucs_sock_addr {
    const struct sockaddr   *addr;      /**< Pointer to socket address */
    socklen_t                addrlen;   /**< Address length */
} ucs_sock_addr_t;

/**
 * Logging component.
 */
typedef struct ucs_log_component_config {
    ucs_log_level_t log_level;
    char            name[16];
} ucs_log_component_config_t;

#endif /* TYPES_H_ */
