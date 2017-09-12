/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "stats.h"

#include <ucs/debug/log.h>
#include <ucs/time/time.h>
#include <ucs/config/global_opts.h>
#include <ucs/config/parser.h>
#include <ucs/type/status.h>
#include <ucs/sys/sys.h>

#include <sys/ioctl.h>
#include <linux/futex.h>

const char *ucs_stats_formats_names[] = {
    [UCS_STATS_FULL]        = "full",
    [UCS_STATS_FULL_AGG]    = "agg",
    [UCS_STATS_SUMMARY]     = "summary",
    [UCS_STATS_LAST]        = NULL
};

#if ENABLE_STATS

enum {
    UCS_STATS_FLAG_ON_EXIT        = UCS_BIT(0),
    UCS_STATS_FLAG_ON_TIMER       = UCS_BIT(1),
    UCS_STATS_FLAG_ON_SIGNAL      = UCS_BIT(2),

    UCS_STATS_FLAG_SOCKET         = UCS_BIT(8),
    UCS_STATS_FLAG_STREAM         = UCS_BIT(9),
    UCS_STATS_FLAG_STREAM_CLOSE   = UCS_BIT(10),
    UCS_STATS_FLAG_STREAM_BINARY  = UCS_BIT(11),
};

enum {
    UCS_ROOT_STATS_RUNTIME,
    UCS_ROOT_STATS_LAST
};

typedef struct {
    volatile unsigned    flags;

    ucs_time_t           start_time;
    ucs_stats_filter_node_t  root_filter_node;
    ucs_stats_node_t     root_node;
    ucs_stats_counter_t  root_counters[UCS_ROOT_STATS_LAST];

    union {
        FILE             *stream;         /* Output stream */
        ucs_stats_client_h client;       /* UDP client */
    };

    union {
        int              signo;
        double           interval;
    };

    pthread_mutex_t      lock;
    pthread_t            thread;
} ucs_stats_context_t;

static ucs_stats_context_t ucs_stats_context = {
    .flags            = 0,
    .root_node        = {},
    .root_filter_node = {},
    .lock             = PTHREAD_MUTEX_INITIALIZER,
    .thread           = 0xfffffffful
};

static ucs_stats_class_t ucs_stats_root_node_class = {
    .name          = "",
    .num_counters  = UCS_ROOT_STATS_LAST,
    .counter_names = {
        [UCS_ROOT_STATS_RUNTIME] = "runtime"
    }
};


static inline int
ucs_sys_futex(volatile void *addr1, int op, int val1, struct timespec *timeout,
              void *uaddr2, int val3)
{
    return syscall(SYS_futex, addr1, op, val1, timeout, uaddr2, val3);
}

static void ucs_stats_clean_node(ucs_stats_node_t *node) {
    ucs_stats_filter_node_t * temp_filter_node;
    ucs_stats_filter_node_t * filter_node;

    filter_node = node->filter_node;
    filter_node->type_list_len--;
    temp_filter_node = node->filter_node;

    if (temp_filter_node->ref_count) {
        while (temp_filter_node != NULL) {
            temp_filter_node->ref_count--;
            temp_filter_node = temp_filter_node->parent;
        }
    }

    if (!filter_node->type_list_len) {
        ucs_list_del(&filter_node->list);
    }
    ucs_list_del(&node->type_list);
}

static void ucs_stats_node_remove(ucs_stats_node_t *node, int make_inactive)
{
    ucs_assert(node != &ucs_stats_context.root_node);

    if (!ucs_list_is_empty(&node->children[UCS_STATS_ACTIVE_CHILDREN])) {
        ucs_warn("stats node "UCS_STATS_NODE_FMT" still has active children",
                 UCS_STATS_NODE_ARG(node));
    }

    pthread_mutex_lock(&ucs_stats_context.lock);

    ucs_list_del(&node->list);
    if (make_inactive) {
        ucs_list_add_tail(&node->parent->children[UCS_STATS_INACTIVE_CHILDREN], &node->list);
    } else {
        ucs_stats_clean_node(node); 
    }

    pthread_mutex_unlock(&ucs_stats_context.lock);

    if (!make_inactive) {
        if (!node->filter_node->type_list_len) {
            ucs_free(node->filter_node);
        }
        ucs_free(node);
    }
}   

static void ucs_stats_filter_node_init_root() {
    ucs_list_head_init(&ucs_stats_context.root_filter_node.list);
    ucs_stats_context.root_filter_node.parent = NULL;
    ucs_list_head_init(&ucs_stats_context.root_filter_node.type_list_head);
    ucs_list_add_tail(&ucs_stats_context.root_filter_node.type_list_head,
                      &ucs_stats_context.root_node.type_list);
    ucs_stats_context.root_filter_node.counters_bitmask = 0;
    ucs_stats_context.root_filter_node.ref_count = 0;
    ucs_stats_context.root_filter_node.type_list_len = 1;
    ucs_list_head_init(&ucs_stats_context.root_filter_node.children);
}

static void ucs_stats_node_init_root(const char *name, ...)
{
    ucs_status_t status;
    va_list ap;

    if (!ucs_stats_is_active()) {
        return;
    }

    va_start(ap, name);
    status = ucs_stats_node_initv(&ucs_stats_context.root_node,
                                 &ucs_stats_root_node_class, name, ap);
    ucs_assert_always(status == UCS_OK);
    va_end(ap);

    ucs_stats_context.root_node.parent = NULL;
    ucs_stats_context.root_node.filter_node = &ucs_stats_context.root_filter_node;

    ucs_stats_filter_node_init_root();
}

static ucs_status_t ucs_stats_node_new(ucs_stats_class_t *cls, ucs_stats_node_t **p_node)
{
    ucs_stats_node_t *node;

    node = ucs_malloc(sizeof(ucs_stats_node_t) +
                      sizeof(ucs_stats_counter_t) *
                      (cls->num_counters > 0 ? cls->num_counters - 1 : 0),
                      "stats node");
    if (node == NULL) {
        ucs_error("Failed to allocate stats node for %s", cls->name);
        return UCS_ERR_NO_MEMORY;
    }

    *p_node = node;
    return UCS_OK;
}

static ucs_status_t ucs_stats_filter_node_new(ucs_stats_class_t *cls, ucs_stats_filter_node_t **p_node)
{
    ucs_stats_filter_node_t *node;

    node = ucs_malloc(sizeof(ucs_stats_filter_node_t),
                      "stats filter node");
    if (node == NULL) {
        ucs_error("Failed to allocate stats filter node for %s", cls->name);
        return UCS_ERR_NO_MEMORY;
    }

    *p_node = node;
    return UCS_OK;
}

static ucs_stats_filter_node_t * ucs_stats_find_class(ucs_stats_filter_node_t *filter_parent,
                                                      const char *class_name) {
    ucs_stats_filter_node_t *filter_node;
    ucs_stats_node_t * node;

    ucs_list_for_each(filter_node, &filter_parent->children, list) {
        if (ucs_list_is_empty(&filter_node->type_list_head)) {
            ucs_error("type list is empty");
            return NULL;
        }
        node = ucs_list_head(&filter_node->type_list_head,
                             ucs_stats_node_t,
                             type_list);
        if (!strcmp(node->cls->name, class_name)) {
            return filter_node;
        }
    }
    return NULL;
}

static void ucs_stats_add_to_filter(ucs_stats_node_t *node,
                                    ucs_stats_filter_node_t * new_filter_node)
{
    ucs_stats_filter_node_t *temp_filter_node;
    ucs_stats_filter_node_t *filter_node = NULL;
    ucs_stats_filter_node_t *filter_parent;
    int found = 0;
    int filter_index = 0;
    int i;

    if (ucs_global_opts.stats_format == UCS_STATS_SUMMARY) {
        filter_parent = &ucs_stats_context.root_filter_node;
    } else {
        filter_parent = node->parent->filter_node;
    }

    if (ucs_global_opts.stats_format != UCS_STATS_FULL) {
        filter_node = ucs_stats_find_class(filter_parent, node->cls->name);
    }

    if (!filter_node) {
        filter_node = new_filter_node;

        filter_node->type_list_len = 0;
        filter_node->ref_count = 0;
        filter_node->counters_bitmask = 0;
        ucs_list_head_init(&filter_node->children);
        ucs_list_head_init(&filter_node->type_list_head);
        filter_node->parent = filter_parent;
        ucs_list_add_tail(&filter_parent->children, &filter_node->list);
    }

    filter_node->type_list_len++;
    ucs_list_add_tail(&filter_node->type_list_head, &node->type_list);
    node->filter_node = filter_node;   

    for (i = 0; (i < node->cls->num_counters) && (i < 64); ++i) {
        filter_index = ucs_config_names_search(ucs_global_opts.stats_filter,
                                               node->cls->counter_names[i]);
        if (filter_index >= 0) {
            filter_node->counters_bitmask |= UCS_BIT(i);
            found = 1; 
        }
    }

    if (found) {
        temp_filter_node = filter_node;
        while (temp_filter_node != NULL) {
            temp_filter_node->ref_count++;
            temp_filter_node = temp_filter_node->parent;
        }
    }
}

static int ucs_stats_node_add(ucs_stats_node_t *node,
                              ucs_stats_node_t *parent,
                              ucs_stats_filter_node_t *filter_node)
{
    ucs_assert(node != &ucs_stats_context.root_node);
    if (parent == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    /* Append node to existing tree */
    pthread_mutex_lock(&ucs_stats_context.lock);
    ucs_list_add_tail(&parent->children[UCS_STATS_ACTIVE_CHILDREN], &node->list);
    node->parent = parent;
    ucs_stats_add_to_filter(node, filter_node);

    pthread_mutex_unlock(&ucs_stats_context.lock);

    return UCS_OK;
}

ucs_status_t ucs_stats_node_alloc(ucs_stats_node_t** p_node, ucs_stats_class_t *cls,
                                 ucs_stats_node_t *parent, const char *name, ...)
{
    ucs_stats_node_t *node;
    ucs_stats_filter_node_t *filter_node;
    ucs_status_t status;
    va_list ap;

    if (!ucs_stats_is_active()) {
        *p_node = NULL;
        return UCS_OK;
    }

    status = ucs_stats_node_new(cls, &node);
    if (status != UCS_OK) {
        return status;
    }

    va_start(ap, name);
    status = ucs_stats_node_initv(node, cls, name, ap);
    va_end(ap);

    if (status != UCS_OK) {
        ucs_free(node);
        return status;
    }

    status = ucs_stats_filter_node_new(node->cls, &filter_node);
    if (status != UCS_OK) {
        ucs_free(node);
        return status;
    }

    ucs_trace("allocated stats node '"UCS_STATS_NODE_FMT"'", UCS_STATS_NODE_ARG(node));

    status = ucs_stats_node_add(node, parent, filter_node);
    if (status != UCS_OK) {
        ucs_free(node);
        ucs_free(filter_node);
        return status;
    }

    if (node->filter_node != filter_node) {
        ucs_free(filter_node);
    }

    *p_node = node;
    return UCS_OK;
}

void ucs_stats_node_free(ucs_stats_node_t *node)
{
    if (node == NULL) {
        return;
    }

    ucs_trace("releasing stats node '"UCS_STATS_NODE_FMT"'", UCS_STATS_NODE_ARG(node));

    /* If we would dump stats in exit, keep this data instead of releasing it */
    if (ucs_stats_context.flags & UCS_STATS_FLAG_ON_EXIT) {
        ucs_stats_node_remove(node, 1);
    } else {
        ucs_stats_node_remove(node, 0);
    }
}

static void __ucs_stats_dump(int inactive)
{
    ucs_status_t status = UCS_OK;
    int options;

    /* Assume locked */

    UCS_STATS_SET_TIME(&ucs_stats_context.root_node, UCS_ROOT_STATS_RUNTIME,
                       ucs_stats_context.start_time);

    if (ucs_stats_context.flags & UCS_STATS_FLAG_SOCKET) {
        status = ucs_stats_client_send(ucs_stats_context.client,
                                      &ucs_stats_context.root_node,
                                      ucs_get_time());
    }

    if (ucs_stats_context.flags & UCS_STATS_FLAG_STREAM) {
        options = 0;
        if (ucs_stats_context.flags & UCS_STATS_FLAG_STREAM_BINARY) {
            options |= UCS_STATS_SERIALIZE_BINARY;
        }
        if (inactive) {
            options |= UCS_STATS_SERIALIZE_INACTVIVE;
        }

        status = ucs_stats_serialize(ucs_stats_context.stream,
                                    &ucs_stats_context.root_node, options);
        fflush(ucs_stats_context.stream);
    }

    if (status != UCS_OK) {
        ucs_warn("Failed to dump statistics: %s", ucs_status_string(status));
    }
}

static void* ucs_stats_thread_func(void *arg)
{
    struct timespec timeout, *ptime;
    unsigned flags;
    long nsec;

    if (ucs_stats_context.interval > 0) {
        nsec = (long)(ucs_stats_context.interval * UCS_NSEC_PER_SEC + 0.5);
        timeout.tv_sec  = nsec / UCS_NSEC_PER_SEC;
        timeout.tv_nsec = nsec % UCS_NSEC_PER_SEC;
        ptime = &timeout;
    }
    else {
        ptime = NULL;
    }

    flags = ucs_stats_context.flags;
    while (flags & UCS_STATS_FLAG_ON_TIMER) {
        /* Wait for timeout/wakeup */
        ucs_sys_futex(&ucs_stats_context.flags, FUTEX_WAIT, flags, ptime, NULL, 0);
        ucs_stats_dump();
        flags = ucs_stats_context.flags;
    }

    return NULL;
}

static void ucs_stats_open_dest()
{
    ucs_status_t status;
    char *copy_str, *saveptr;
    const char *hostname, *port_str;
    const char *next_token;
    int need_close;

    if (!strncmp(ucs_global_opts.stats_dest, "udp:", 4)) {

        copy_str = strdupa(&ucs_global_opts.stats_dest[4]);
        saveptr  = NULL;
        hostname = strtok_r(copy_str, ":", &saveptr);
        port_str = strtok_r(NULL,     ":", &saveptr);

        if (hostname == NULL) {
           ucs_error("Invalid statistics destination format (%s)", ucs_global_opts.stats_dest);
           return;
        }

        status = ucs_stats_client_init(hostname,
                                      port_str ? atoi(port_str) : UCS_STATS_DEFAULT_UDP_PORT,
                                      &ucs_stats_context.client);
        if (status != UCS_OK) {
            return;
        }

        ucs_stats_context.flags |= UCS_STATS_FLAG_SOCKET;
    } else if (strcmp(ucs_global_opts.stats_dest, "") != 0) {
        status = ucs_open_output_stream(ucs_global_opts.stats_dest,
                                        UCS_LOG_LEVEL_ERROR,
                                        &ucs_stats_context.stream,
                                        &need_close, &next_token);
        if (status != UCS_OK) {
            return;
        }

        /* File flags */
        ucs_stats_context.flags |= UCS_STATS_FLAG_STREAM;
        if (need_close) {
            ucs_stats_context.flags |= UCS_STATS_FLAG_STREAM_CLOSE;
        }

        /* Optional: Binary mode */
        if (!strcmp(next_token, ":bin")) {
            ucs_stats_context.flags |= UCS_STATS_FLAG_STREAM_BINARY;
        }
    }
}

static void ucs_stats_close_dest()
{
    if (ucs_stats_context.flags & UCS_STATS_FLAG_SOCKET) {
        ucs_stats_context.flags &= ~UCS_STATS_FLAG_SOCKET;
        ucs_stats_client_cleanup(ucs_stats_context.client);
    }
    if (ucs_stats_context.flags & UCS_STATS_FLAG_STREAM) {
        fflush(ucs_stats_context.stream);
        if (ucs_stats_context.flags & UCS_STATS_FLAG_STREAM_CLOSE) {
            fclose(ucs_stats_context.stream);
        }
        ucs_stats_context.flags &= ~(UCS_STATS_FLAG_STREAM|
                                     UCS_STATS_FLAG_STREAM_BINARY|
                                     UCS_STATS_FLAG_STREAM_CLOSE);
    }
}

static void ucs_stats_dump_sighandler(int signo)
{
    ucs_stats_dump();
}

static void ucs_stats_set_trigger()
{
    char *p;

    if (!strcmp(ucs_global_opts.stats_trigger, "exit")) {
        ucs_stats_context.flags |= UCS_STATS_FLAG_ON_EXIT;
    } else if (!strncmp(ucs_global_opts.stats_trigger, "timer:", 6)) {
        p = ucs_global_opts.stats_trigger + 6;
        if (!ucs_config_sscanf_time(p, &ucs_stats_context.interval, NULL)) {
            ucs_error("Invalid statistics interval time format: %s", p);
            return;
        }

        ucs_stats_context.flags |= UCS_STATS_FLAG_ON_TIMER;
        pthread_create(&ucs_stats_context.thread, NULL, ucs_stats_thread_func, NULL);
   } else if (!strncmp(ucs_global_opts.stats_trigger, "signal:", 7)) {
        p = ucs_global_opts.stats_trigger + 7;
        if (!ucs_config_sscanf_signo(p, &ucs_stats_context.signo, NULL)) {
            ucs_error("Invalid statistics signal specification: %s", p);
            return;
        }

        signal(ucs_stats_context.signo, ucs_stats_dump_sighandler);
        ucs_stats_context.flags |= UCS_STATS_FLAG_ON_SIGNAL;
    } else if (!strcmp(ucs_global_opts.stats_trigger, "")) {
        /* No external trigger */
    } else {
        ucs_error("Invalid statistics trigger: %s", ucs_global_opts.stats_trigger);
    }
}

static void ucs_stats_unset_trigger()
{
    void *result;

    if (ucs_stats_context.flags & UCS_STATS_FLAG_ON_TIMER) {
        ucs_stats_context.flags &= ~UCS_STATS_FLAG_ON_TIMER;
        ucs_sys_futex(&ucs_stats_context.flags, FUTEX_WAKE, 1, NULL, NULL, 0);
        pthread_join(ucs_stats_context.thread, &result);
    }

    if (ucs_stats_context.flags & UCS_STATS_FLAG_ON_EXIT) {
        ucs_debug("dumping stats");
        __ucs_stats_dump(1);
        ucs_stats_context.flags &= ~UCS_STATS_FLAG_ON_EXIT;
    }

    if (ucs_stats_context.flags & UCS_STATS_FLAG_ON_SIGNAL) {
        ucs_stats_context.flags &= ~UCS_STATS_FLAG_ON_SIGNAL;
        signal(ucs_stats_context.signo, SIG_DFL);
    }
}

static void ucs_stats_clean_node_recurs(ucs_stats_node_t *node)
{
    ucs_stats_node_t *child, *tmp;

    if (!ucs_list_is_empty(&node->children[UCS_STATS_ACTIVE_CHILDREN])) {
        ucs_warn("stats node "UCS_STATS_NODE_FMT" still has active children",
                 UCS_STATS_NODE_ARG(node));
    }

    ucs_list_for_each_safe(child, tmp, &node->children[UCS_STATS_INACTIVE_CHILDREN], list) {
        ucs_stats_clean_node_recurs(child);
        ucs_stats_node_remove(child, 0);
    }
}

void ucs_stats_init()
{
    ucs_assert(ucs_stats_context.flags == 0);
    ucs_stats_open_dest();

    if (!ucs_stats_is_active()) {
        ucs_trace("statistics disabled");
        return;
    }

    UCS_STATS_START_TIME(ucs_stats_context.start_time);
    ucs_stats_node_init_root("%s:%d", ucs_get_host_name(), getpid());
    ucs_stats_set_trigger();

    ucs_debug("statistics enabled, flags: %c%c%c%c%c%c%c",
              (ucs_stats_context.flags & UCS_STATS_FLAG_ON_TIMER)      ? 't' : '-',
              (ucs_stats_context.flags & UCS_STATS_FLAG_ON_EXIT)       ? 'e' : '-',
              (ucs_stats_context.flags & UCS_STATS_FLAG_ON_SIGNAL)     ? 's' : '-',
              (ucs_stats_context.flags & UCS_STATS_FLAG_SOCKET)        ? 'u' : '-',
              (ucs_stats_context.flags & UCS_STATS_FLAG_STREAM)        ? 'f' : '-',
              (ucs_stats_context.flags & UCS_STATS_FLAG_STREAM_BINARY) ? 'b' : '-',
              (ucs_stats_context.flags & UCS_STATS_FLAG_STREAM_CLOSE)  ? 'c' : '-');
}

void ucs_stats_cleanup()
{
    if (!ucs_stats_is_active()) {
        return;
    }

    ucs_stats_unset_trigger();
    ucs_stats_clean_node_recurs(&ucs_stats_context.root_node);
    ucs_stats_close_dest();
    ucs_assert(ucs_stats_context.flags == 0);
}

void ucs_stats_dump()
{
    pthread_mutex_lock(&ucs_stats_context.lock);
    __ucs_stats_dump(0);
    pthread_mutex_unlock(&ucs_stats_context.lock);
}

int ucs_stats_is_active()
{
    return ucs_stats_context.flags & (UCS_STATS_FLAG_SOCKET|UCS_STATS_FLAG_STREAM);
}

ucs_stats_node_t * ucs_stats_get_root() {
    return &ucs_stats_context.root_node;
}

#else

void ucs_stats_init()
{
}

void ucs_stats_cleanup()
{
}

void ucs_stats_dump()
{
}

int ucs_stats_is_active()
{
    return 0;
}

ucs_stats_node_t *ucs_stats_get_root()
{
    return NULL;
}
#endif
