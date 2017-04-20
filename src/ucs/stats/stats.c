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
    [UCS_STATS_FULL]        = "FULL",
    [UCS_STATS_SUMMARY]     = "SUMMARY",
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
    ucs_stats_summary_t  sum_list;
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
    .sum_list         = {},
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

static void ucs_stats_node_add(ucs_stats_node_t *node, ucs_stats_node_t *parent)
{
    ucs_assert(node != &ucs_stats_context.root_node);

    /* Append node to existing tree */
    pthread_mutex_lock(&ucs_stats_context.lock);
    if (parent == NULL) {
        parent = &ucs_stats_context.root_node;
    }
    ucs_list_add_tail(&parent->children[UCS_STATS_ACTIVE_CHILDREN], &node->list);
    node->parent = parent;

    pthread_mutex_unlock(&ucs_stats_context.lock);
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
    }
    pthread_mutex_unlock(&ucs_stats_context.lock);
}

static void ucs_stats_node_init_root(const char *name, ...)
{
    ucs_status_t status;
    va_list ap;

    if (!ucs_stats_is_active()) {
        return;
    }

    va_start(ap, name);
    ucs_stats_context.root_node.counters_sum =
             ucs_malloc(sizeof(ucs_stats_counter_list_t *) *
                        ucs_stats_root_node_class.num_counters,
                        "root counter sum");
    status = ucs_stats_node_initv(&ucs_stats_context.root_node,
                                 &ucs_stats_root_node_class, name, ap);
    ucs_assert_always(status == UCS_OK);
    va_end(ap);

    ucs_stats_context.root_node.parent = NULL;
}

static ucs_status_t ucs_stats_summary_new(ucs_stats_summary_t **p_summary)
{
    ucs_stats_summary_t *stats_summary;

    stats_summary = ucs_malloc(sizeof(ucs_stats_summary_t), "stats summary");
    if (stats_summary == NULL) {
        ucs_error("Failed to allocate stats summary for");
        return UCS_ERR_NO_MEMORY;
    }

    *p_summary = stats_summary;
    return UCS_OK;
}

static ucs_status_t ucs_stats_counter_new(ucs_stats_counter_list_t **p_counter)
{
    ucs_stats_counter_list_t *stats_counter;

    stats_counter = ucs_malloc(sizeof(ucs_stats_counter_list_t), "stats counter");
    if (stats_counter == NULL) {
        ucs_error("Failed to allocate stats counter for");
        return UCS_ERR_NO_MEMORY;
    }

    *p_counter = stats_counter;
    return UCS_OK;
}

static ucs_status_t ucs_stats_node_new(ucs_stats_class_t *cls, ucs_stats_node_t **p_node)
{
    ucs_stats_node_t *node;

    node = ucs_malloc(sizeof(ucs_stats_node_t) +
                      sizeof(ucs_stats_counter_t) * cls->num_counters,
                      "stats node");

    if (node == NULL) {
        ucs_error("Failed to allocate stats node for %s", cls->name);
        return UCS_ERR_NO_MEMORY;
    }

    node->counters_sum = ucs_malloc(sizeof(ucs_stats_counter_list_t *) *
                                    cls->num_counters, "stats sum list");
    if (node->counters_sum == NULL) {
        ucs_free(node);
        ucs_error("Failed to allocate counter summary for %s", cls->name);
        return UCS_ERR_NO_MEMORY;
    }

    memset(node->counters_sum, 0, sizeof(ucs_stats_counter_list_t *) *
                                  cls->num_counters);

    *p_node = node;
    return UCS_OK;
}

static void ucs_stats_add_wildcard(ucs_stats_node_t *node,
                                   ucs_stats_summary_t* sum_list)
{
    unsigned i;
    int filter_index = 0;
    ucs_stats_summary_t *elem;
    ucs_stats_counter_list_t *counter_item;

    for (i = 0; i < node->cls->num_counters; ++i) {
        node->counters_sum[i] = (ucs_stats_counter_list_t *) NULL;
        filter_index = ucs_config_names_search(ucs_global_opts.stats_filter,
                                               node->cls->counter_names[i]);
        if (filter_index >= 0) {
            int found = 0;
            ucs_list_for_each(elem, &sum_list->list, list) {
                if (!strcmp(elem->class_name, node->cls->name) &&
                    !strcmp(elem->counter_name, node->cls->counter_names[i])) {
                    found = 1;
                    break;
                }
            }
            if (!found) {
                if (!ucs_stats_summary_new(&elem)) {
                    strncpy(elem->counter_name, node->cls->counter_names[i],
                            sizeof(elem->counter_name) - 1);
                    strncpy(elem->class_name, node->cls->name,
                            sizeof(elem->class_name) - 1);
                    ucs_list_add_tail(&sum_list->list, &elem->list);
                    ucs_list_head_init(&elem->counter_list);
                }
            }
            if (!ucs_stats_counter_new(&counter_item)) {
                counter_item->counter = &node->counters[i];
                node->counters_sum[i] = counter_item;
                ucs_list_add_tail(&elem->counter_list, &counter_item->list);
            }
        }
    }
}

ucs_status_t ucs_stats_node_alloc(ucs_stats_node_t** p_node, ucs_stats_class_t *cls,
                                 ucs_stats_node_t *parent, const char *name, ...)
{
    ucs_stats_node_t *node;
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

    ucs_trace("allocated stats node '"UCS_STATS_NODE_FMT"'", UCS_STATS_NODE_ARG(node));

    ucs_stats_node_add(node, parent);
    ucs_stats_add_wildcard(node, &ucs_stats_context.sum_list);
    *p_node = node;
    return UCS_OK;
}

void ucs_stats_node_free(ucs_stats_node_t *node)
{
    int i;

    if (node == NULL) {
        return;
    }

    ucs_trace("releasing stats node '"UCS_STATS_NODE_FMT"'", UCS_STATS_NODE_ARG(node));

    /* If we would dump stats in exit, keep this data instead of releasing it */
    if (ucs_stats_context.flags & UCS_STATS_FLAG_ON_EXIT) {
        ucs_stats_node_remove(node, 1);
    } else {
        for (i = 0; i < node->cls->num_counters; ++i) {
            if (node->counters_sum[i]) {
                ucs_list_del(&node->counters_sum[i]->list);
                ucs_free(node->counters_sum[i]);
            }
        }
        ucs_stats_node_remove(node, 0);
        ucs_free(node->counters_sum);
        ucs_free(node);
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

static void
ucs_stats_sum_clean()
{
    ucs_stats_summary_t *sum_item, *tmp_sum_item;
    ucs_stats_counter_list_t *counter_item, *tmp_counter_item;
    ucs_list_for_each_safe(sum_item, tmp_sum_item,
                           &ucs_stats_context.sum_list.list, list) {
        ucs_list_for_each_safe(counter_item, tmp_counter_item,
                               &sum_item->counter_list, list) {
            ucs_free(counter_item);
        }
        ucs_free(sum_item);
    }
}

static void
ucs_stats_traverse_sum_counters()
{
    ucs_stats_summary_t *sum_item;
    ucs_stats_counter_list_t *counter_item;
    char current_class[80] = "";
    int first_item = 1;

    fprintf(ucs_stats_context.stream, "%s:", ucs_stats_context.root_node.name);
    ucs_list_for_each(sum_item, &ucs_stats_context.sum_list.list, list) {
        sum_item->counter_sum = 0;
        ucs_list_for_each(counter_item, &sum_item->counter_list, list) {
            sum_item->counter_sum += *counter_item->counter;
        }
        if (sum_item->counter_sum < ucs_global_opts.stats_threshold) {
            continue;
        }
        if (strcmp(current_class, sum_item->class_name)) {
            if (current_class[0]) {
                first_item = 1;
                fprintf(ucs_stats_context.stream, "]");
            }
            fprintf(ucs_stats_context.stream, "%s{", sum_item->class_name);
            strcpy(current_class, sum_item->class_name);
        }
        if (!first_item) {
            fprintf(ucs_stats_context.stream, " ");
        }
        fprintf(ucs_stats_context.stream, "%s:%"PRIu64,
                sum_item->counter_name, sum_item->counter_sum);
        first_item = 0;
    }
    if (current_class[0]) {
        fprintf(ucs_stats_context.stream, "}");
    }
    fprintf(ucs_stats_context.stream, "\n");
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
        if (ucs_global_opts.stats_format == UCS_STATS_FULL) {
            __ucs_stats_dump(1);
        } else {
            ucs_stats_traverse_sum_counters();
        }
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
    int i;

    if (!ucs_list_is_empty(&node->children[UCS_STATS_ACTIVE_CHILDREN])) {
        ucs_warn("stats node "UCS_STATS_NODE_FMT" still has active children",
                 UCS_STATS_NODE_ARG(node));
    }

    ucs_list_for_each_safe(child, tmp, &node->children[UCS_STATS_INACTIVE_CHILDREN], list) {
        ucs_stats_clean_node_recurs(child);
        ucs_stats_node_remove(child, 0);
        for (i = 0; i < child->cls->num_counters; ++i) {
            if (child->counters_sum[i]) {
                ucs_list_del(&child->counters_sum[i]->list);
                ucs_free(child->counters_sum[i]);
            }
        }
        ucs_free(child);
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
    ucs_list_head_init(&ucs_stats_context.sum_list.list);
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
    ucs_stats_sum_clean();
    ucs_stats_close_dest();
    ucs_assert(ucs_stats_context.flags == 0);
}

void ucs_stats_dump()
{
    pthread_mutex_lock(&ucs_stats_context.lock);
    if (ucs_global_opts.stats_format == UCS_STATS_FULL) {
        __ucs_stats_dump(0);
    } else {
        ucs_stats_traverse_sum_counters();
        fflush(ucs_stats_context.stream);
    }
    pthread_mutex_unlock(&ucs_stats_context.lock);
}

int ucs_stats_is_active()
{
    return ucs_stats_context.flags & (UCS_STATS_FLAG_SOCKET|UCS_STATS_FLAG_STREAM);
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

#endif
