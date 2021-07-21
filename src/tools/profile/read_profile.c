/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/profile/profile.h>
#include <ucs/datastruct/khash.h>
#include <ucs/sys/string.h>

#include <sys/signal.h>
#include <sys/fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <errno.h>


#define INDENT             4
#define PAGER_LESS         "less"
#define PAGER_LESS_CMD     PAGER_LESS " -R"
#define FUNC_NAME_MAX_LEN  35
#define MAX_THREADS        256

#define TERM_COLOR_CLEAR   "\x1B[0m"
#define TERM_COLOR_RED     "\x1B[31m"
#define TERM_COLOR_GREEN   "\x1B[32m"
#define TERM_COLOR_YELLOW  "\x1B[33m"
#define TERM_COLOR_BLUE    "\x1B[34m"
#define TERM_COLOR_MAGENTA "\x1B[35m"
#define TERM_COLOR_CYAN    "\x1B[36m"
#define TERM_COLOR_WHITE   "\x1B[37m"
#define TERM_COLOR_GRAY    "\x1B[90m"

#define NAME_COLOR         (opts->raw ? "" : TERM_COLOR_CYAN)
#define HEAD_COLOR         (opts->raw ? "" : TERM_COLOR_RED)
#define TS_COLOR           (opts->raw ? "" : TERM_COLOR_WHITE)
#define LOC_COLOR          (opts->raw ? "" : TERM_COLOR_GRAY)
#define REQ_COLOR          (opts->raw ? "" : TERM_COLOR_YELLOW)
#define CLEAR_COLOR        (opts->raw ? "" : TERM_COLOR_CLEAR)

#define print_error(_fmt, ...) \
    fprintf(stderr, "Error: " _fmt "\n", ## __VA_ARGS__)


typedef enum {
    TIME_UNITS_NSEC,
    TIME_UNITS_USEC,
    TIME_UNITS_MSEC,
    TIME_UNITS_SEC,
    TIME_UNITS_LAST
} time_units_t;


typedef struct options {
    const char                   *filename;
    int                          raw;
    time_units_t                 time_units;
    int                          thread_list[MAX_THREADS + 1];
} options_t;


typedef struct {
    const ucs_profile_thread_header_t   *header;
    const ucs_profile_thread_location_t *locations;
    const ucs_profile_record_t          *records;
} profile_thread_data_t;


typedef struct {
    void                         *mem;
    size_t                       length;
    const ucs_profile_header_t   *header;
    const ucs_profile_location_t *locations;
    profile_thread_data_t        *threads;
} profile_data_t;


typedef struct {
    uint64_t                     total_time;
    size_t                       count;
    unsigned                     location_idx;
} profile_sorted_location_t;


/* Used to redirect output to a "less" command */
static int output_pipefds[2] = {-1, -1};


static const char* time_units_str[] = {
    [TIME_UNITS_NSEC] = "(nsec)",
    [TIME_UNITS_USEC] = "(usec)",
    [TIME_UNITS_MSEC] = "(msec)",
    [TIME_UNITS_SEC]  = "(sec)",
    [TIME_UNITS_LAST] = NULL
};


static int read_profile_data(const char *file_name, profile_data_t *data)
{
    uint32_t thread_idx;
    struct stat stt;
    const void *ptr;
    int ret, fd;

    fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        print_error("failed to open %s: %m", file_name);
        ret = fd;
        goto out;
    }

    ret = fstat(fd, &stt);
    if (ret < 0) {
        print_error("fstat(%s) failed: %m", file_name);
        goto out_close;
    }

    data->length = stt.st_size;
    data->mem    = mmap(NULL, stt.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data->mem == MAP_FAILED) {
        print_error("mmap(%s, length=%zd) failed: %m", file_name,
                    data->length);
        ret = -1;
        goto out_close;
    }

    ptr          = data->mem;
    data->header = ptr;
    ptr          = data->header + 1;

    if (data->header->version != UCS_PROFILE_FILE_VERSION) {
        print_error("invalid file version, expected: %u, actual: %u",
                    UCS_PROFILE_FILE_VERSION, data->header->version);
        ret = -EINVAL;
        goto err_munmap;
    }

    data->locations = ptr;
    ptr             = data->locations + data->header->num_locations;

    data->threads   = calloc(data->header->num_threads, sizeof(*data->threads));
    if (data->threads == NULL) {
        print_error("failed to allocate threads array");
        goto err_munmap;
    }

    for (thread_idx = 0; thread_idx < data->header->num_threads; ++thread_idx) {
        profile_thread_data_t *thread = &data->threads[thread_idx];
        thread->header    = ptr;
        ptr               = thread->header + 1;
        thread->locations = ptr;
        ptr               = thread->locations + data->header->num_locations;
        thread->records   = ptr;
        ptr               = thread->records + thread->header->num_records;
    }

    ret = 0;

out_close:
    close(fd);
out:
    return ret;
err_munmap:
    munmap(data->mem, data->length);
    goto out_close;
}

static void release_profile_data(profile_data_t *data)
{
    free(data->threads);
    munmap(data->mem, data->length);
}

static int parse_thread_list(int *thread_list, const char *str)
{
    char *str_dup, *p, *saveptr, *tailptr;
    int thread_idx;
    unsigned idx;
    int ret;

    str_dup = strdup(str);
    if (str_dup == NULL) {
        ret = -ENOMEM;
        print_error("failed to duplicate thread list string");
        goto out;
    }

    idx = 0;

    /* the special value 'all' will create an empty thread list, which means
     * use all threads
     */
    if (!strcasecmp(str_dup, "all")) {
        goto out_terminate;
    }

    p = strtok_r(str_dup, ",", &saveptr);
    while (p != NULL) {
        if (idx >= MAX_THREADS) {
            ret = -EINVAL;
            print_error("up to %d threads are supported", MAX_THREADS);
            goto out;
        }

        thread_idx = strtol(p, &tailptr, 0);
        if (*tailptr != '\0') {
            ret = -ENOMEM;
            print_error("failed to parse thread number '%s'", p);
            goto out;
        }

        if (thread_idx <= 0) {
            ret = -EINVAL;
            print_error("invalid thread index %d", thread_idx);
            goto out;
        }

        thread_list[idx++] = thread_idx;
        p = strtok_r(NULL, ",", &saveptr);
    }

    if (idx == 0) {
        ret = -EINVAL;
        print_error("empty thread list");
        goto out;
    }

out_terminate:
    ret                = 0;
    thread_list[idx] = -1; /* terminator */
out:
    free(str_dup);
    return ret;
}

static const char* thread_list_str(const int *thread_list, char *buf, size_t max)
{
    char *p, *endp;
    const int *t;
    int ret;

    p    = buf;
    endp = buf + max - 4; /* leave room for "...\0" */

    for (t = thread_list; *t != -1; ++t) {
        ret = snprintf(p, endp - p, "%d,", *t);
        if (ret >= endp - p) {
            /* truncated */
            strcat(p, "...");
            return buf;
        }

        p += strlen(p);
    }

    if (p > buf) {
        *(p - 1) = '\0';
    } else {
        *buf = '\0';
    }
    return buf;
}

static double time_to_units(profile_data_t *data, options_t *opts, uint64_t time)
{
    static const double time_units_val[] = {
        [TIME_UNITS_NSEC] = 1e9,
        [TIME_UNITS_USEC] = 1e6,
        [TIME_UNITS_MSEC] = 1e3,
        [TIME_UNITS_SEC]  = 1e0
    };

    return time * time_units_val[opts->time_units] / data->header->one_second;
}

static int compare_locations(const void *l1, const void *l2)
{
    const ucs_profile_thread_location_t *loc1 = l1;
    const ucs_profile_thread_location_t *loc2 = l2;
    return (loc1->total_time > loc2->total_time) ? -1 :
           (loc1->total_time < loc2->total_time) ? +1 :
           0;
}

static int show_profile_data_accum(profile_data_t *data, options_t *opts)
{
    typedef struct {
        long overall_time; /* overall threads runtime */
        int  thread_list[MAX_THREADS + 1];
        int  *last;
    } location_thread_info_t;

    const uint32_t            num_locations          = data->header->num_locations;
    profile_sorted_location_t *sorted_locations      = NULL;
    location_thread_info_t    *locations_thread_info = NULL;
    const ucs_profile_thread_location_t *thread_location;
    location_thread_info_t *loc_thread_info;
    profile_sorted_location_t *sorted_loc;
    const profile_thread_data_t *thread;
    const ucs_profile_location_t *loc;
    unsigned location_idx, thread_idx;
    char avg_buf[20], total_buf[20], overall_buf[20];
    char thread_list_buf[20];
    char *avg_str, *total_str, *overall_str;
    int ret;
    int *t;

    sorted_locations      = calloc(num_locations, sizeof(*sorted_locations));
    locations_thread_info = calloc(num_locations, sizeof(*locations_thread_info));
    if ((sorted_locations == NULL) || (locations_thread_info == NULL)) {
        print_error("failed to allocate locations info");
        ret = -ENOMEM;
        goto out;
    }

    /* Go over the list of threads provided by the user and accumulate the times
     * and counts from all threads. In addition, track which calls were made from
     * which threads.
     */
    for (location_idx = 0; location_idx < num_locations; ++location_idx) {
        sorted_loc                      = &sorted_locations[location_idx];
        loc_thread_info                 = &locations_thread_info[location_idx];
        sorted_loc->location_idx        = location_idx;
        loc_thread_info->thread_list[0] = -1;
        loc_thread_info->last           = loc_thread_info->thread_list;
        loc_thread_info->overall_time   = 0;

        for (t = opts->thread_list; *t != -1; ++t) {
            thread_idx              = *t - 1;
            thread                  = &data->threads[thread_idx];
            thread_location         = &thread->locations[location_idx];
            sorted_loc->count      += thread_location->count;
            sorted_loc->total_time += thread_location->total_time;

            if (thread_location->count > 0) {
                loc_thread_info->overall_time += thread->header->end_time -
                                                 thread->header->start_time;
                *(loc_thread_info->last++)     = thread_idx + 1;
            }
        }

        *loc_thread_info->last = -1;
    }

    /* Sort locations */
    qsort(sorted_locations, num_locations, sizeof(*sorted_locations),
          compare_locations);

    /* Print locations */
    printf("%s%*s %6s %-6s %6s %-6s %13s %12s %18s%-6s  %-*s %s%s\n",
           HEAD_COLOR,
           FUNC_NAME_MAX_LEN,
           "NAME",
           "AVG", time_units_str[opts->time_units],
           "TOTAL", time_units_str[opts->time_units],
           "%OVERALL",
           "COUNT",
           "FILE",
           ":LINE",
           FUNC_NAME_MAX_LEN,
           "FUNCTION",
           "THREADS",
           CLEAR_COLOR);

    for (sorted_loc = sorted_locations;
         sorted_loc < (sorted_locations + num_locations); ++sorted_loc) {

        if (sorted_loc->count == 0) {
            continue;
        }

        loc             = &data->locations[sorted_loc->location_idx];
        loc_thread_info = &locations_thread_info[sorted_loc->location_idx];

        switch (loc->type) {
        case UCS_PROFILE_TYPE_SCOPE_END:
            snprintf(avg_buf,     sizeof(avg_buf) - 1, "%.3f",
                     time_to_units(data, opts,
                                   sorted_loc->total_time / sorted_loc->count));
            snprintf(total_buf,   sizeof(total_buf) - 1, "%.2f",
                     time_to_units(data, opts, sorted_loc->total_time));
            snprintf(overall_buf, sizeof(overall_buf) - 1, "%.3f",
                     sorted_loc->total_time * 100.0 / loc_thread_info->overall_time);

            avg_str     = avg_buf;
            total_str   = total_buf;
            overall_str = overall_buf;
            break;
        case UCS_PROFILE_TYPE_SAMPLE:
        case UCS_PROFILE_TYPE_REQUEST_EVENT:
            avg_str = total_str = overall_str = "n/a";
            break;
        default:
            continue;
        }

        printf("%s%*.*s%s %13s %13s %13s %12zu %s%18s:%-6d %-*s %-13s%s\n",
               NAME_COLOR, FUNC_NAME_MAX_LEN, FUNC_NAME_MAX_LEN, loc->name, CLEAR_COLOR,
               avg_str,
               total_str,
               overall_str,
               sorted_loc->count,
               LOC_COLOR,
                   loc->file, loc->line,
                   FUNC_NAME_MAX_LEN, loc->function,
                   thread_list_str(loc_thread_info->thread_list, thread_list_buf,
                                   sizeof(thread_list_buf)),
               CLEAR_COLOR);
   }

    ret = 0;

out:
    free(locations_thread_info);
    free(sorted_locations);
    return ret;
}

KHASH_MAP_INIT_INT64(request_ids, size_t)

static void show_profile_data_log(profile_data_t *data, options_t *opts,
                                  int thread_idx)
{
    profile_thread_data_t *thread = &data->threads[thread_idx];
    size_t num_records            = thread->header->num_records;
    size_t reqid_ctr              = 1;
    const ucs_profile_record_t **stack[UCS_PROFILE_STACK_MAX * 2];
    const ucs_profile_record_t **scope_ends;
    const ucs_profile_location_t *loc;
    const ucs_profile_record_t *rec, *se, **sep;
    int nesting, min_nesting;
    uint64_t prev_time;
    const char *action;
    char buf[256];
    khash_t(request_ids) reqids;
    int hash_extra_status;
    khiter_t hash_it;
    size_t reqid;

#define RECORD_FMT       "%s%10.3f%s%*s"
#define RECORD_ARG(_ts)  TS_COLOR, time_to_units(data, opts, (_ts)), CLEAR_COLOR, \
                         INDENT * nesting, ""
#define PRINT_RECORD()   printf("%-*s %s%15s:%-4d %s()%s\n", \
                                (int)(60 + strlen(NAME_COLOR) + \
                                      2 * strlen(TS_COLOR) + \
                                      3 * strlen(CLEAR_COLOR)), \
                                buf, \
                                LOC_COLOR, \
                                ucs_basename(loc->file), loc->line, \
                                loc->function, CLEAR_COLOR)

    scope_ends = calloc(1, sizeof(*scope_ends) * num_records);
    if (scope_ends == NULL) {
        print_error("failed to allocate memory for scope ends");
        return;
    }

    printf("\n");
    printf("%sThread %d (tid %d%s)%s\n", HEAD_COLOR, thread_idx + 1,
           thread->header->tid,
           (thread->header->tid == data->header->pid) ? ", main" : "",
           CLEAR_COLOR);
    printf("\n");

    memset(stack, 0, sizeof(stack));

    /* Find the first record with minimal nesting level, which is the base of call stack */
    nesting         = 0;
    min_nesting     = 0;
    for (rec = thread->records; rec < thread->records + num_records; ++rec) {
        loc = &data->locations[rec->location];
        switch (loc->type) {
        case UCS_PROFILE_TYPE_SCOPE_BEGIN:
            stack[nesting + UCS_PROFILE_STACK_MAX] = &scope_ends[rec - thread->records];
            ++nesting;
            break;
        case UCS_PROFILE_TYPE_SCOPE_END:
            --nesting;
            if (nesting < min_nesting) {
                min_nesting     = nesting;
            }
            sep = stack[nesting + UCS_PROFILE_STACK_MAX];
            if (sep != NULL) {
                *sep = rec;
            }
            break;
        default:
            break;
        }
    }

    if (num_records > 0) {
        prev_time = thread->records[0].timestamp;
    } else {
        prev_time = 0;
    }

    kh_init_inplace(request_ids, &reqids);

    /* Display records */
    nesting = -min_nesting;
    for (rec = thread->records; rec < thread->records + num_records; ++rec) {
        loc = &data->locations[rec->location];
        switch (loc->type) {
        case UCS_PROFILE_TYPE_SCOPE_BEGIN:
            se = scope_ends[rec - thread->records];
            if (se != NULL) {
                snprintf(buf, sizeof(buf), RECORD_FMT"  %s%s%s %s%.3f%s {",
                         RECORD_ARG(rec->timestamp - prev_time),
                         NAME_COLOR, data->locations[se->location].name,
                         CLEAR_COLOR, TS_COLOR,
                         time_to_units(data, opts, se->timestamp - rec->timestamp),
                         CLEAR_COLOR);
            } else {
                snprintf(buf, sizeof(buf), "<unfinished>");
            }
            PRINT_RECORD();
            nesting++;
            break;
        case UCS_PROFILE_TYPE_SCOPE_END:
            --nesting;
            printf(RECORD_FMT"  }\n", RECORD_ARG(rec->timestamp - prev_time));
            break;
        case UCS_PROFILE_TYPE_SAMPLE:
            snprintf(buf, sizeof(buf), RECORD_FMT"  %s%s%s",
                     RECORD_ARG(rec->timestamp - prev_time),
                     NAME_COLOR, loc->name, CLEAR_COLOR);
            PRINT_RECORD();
            break;
        case UCS_PROFILE_TYPE_REQUEST_NEW:
        case UCS_PROFILE_TYPE_REQUEST_EVENT:
        case UCS_PROFILE_TYPE_REQUEST_FREE:
            if (loc->type == UCS_PROFILE_TYPE_REQUEST_NEW) {
                hash_it = kh_put(request_ids, &reqids, rec->param64,
                                 &hash_extra_status);
                if (hash_it == kh_end(&reqids)) {
                    if (hash_extra_status == 0) {
                        /* old request was not released, replace it */
                        hash_it = kh_get(request_ids, &reqids, rec->param64);
                        reqid = reqid_ctr++;
                        kh_value(&reqids, hash_it) = reqid;
                    } else {
                        reqid = 0; /* error inserting to hash */
                    }
                } else {
                    /* new request */
                    reqid = reqid_ctr++;
                    kh_value(&reqids, hash_it) = reqid;
                }
                action = "NEW ";
            } else {
                hash_it = kh_get(request_ids, &reqids, rec->param64);
                if (hash_it == kh_end(&reqids)) {
                    reqid = 0; /* could not find request */
                } else {
                    assert(reqid_ctr > 1);
                    reqid = kh_value(&reqids, hash_it);
                    if (loc->type == UCS_PROFILE_TYPE_REQUEST_FREE) {
                        kh_del(request_ids, &reqids, hash_it);
                    }
                }
                if (loc->type == UCS_PROFILE_TYPE_REQUEST_FREE) {
                    action = "FREE";
                } else {
                    action = "";
                }
            }
            snprintf(buf, sizeof(buf), RECORD_FMT"  %s%s%s%s %s{%zu}%s",
                     RECORD_ARG(rec->timestamp - prev_time),
                     REQ_COLOR, action, loc->name, CLEAR_COLOR,
                     REQ_COLOR, reqid, CLEAR_COLOR);
            PRINT_RECORD();
            break;
        default:
            break;
        }
        prev_time = rec->timestamp;
    }

    kh_destroy_inplace(request_ids, &reqids);
    free(scope_ends);
}

static void close_pipes()
{
    close(output_pipefds[0]);
    close(output_pipefds[1]);
}

static int redirect_output(const profile_data_t *data, options_t *opts)
{
    const char *shell_cmd = "sh";
    struct winsize wsz;
    uint64_t num_lines;
    const char *pager_cmd;
    pid_t pid;
    int ret;
    int *t;

    ret = ioctl(STDOUT_FILENO, TIOCGWINSZ, &wsz);
    if (ret < 0) {
        print_error("ioctl(TIOCGWINSZ) failed: %m");
        return ret;
    }

    num_lines = 6 + /* header */
                1; /* footer */

    if (data->header->mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        num_lines += 1 + /* locations title */
                     data->header->num_locations + /* locations data */
                     1; /* locations footer */
    }

    if (data->header->mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        for (t = opts->thread_list; *t != -1; ++t) {
            num_lines += 3 + /* thread header */
                         data->threads[*t - 1].header->num_records; /* thread records */
        }
    }

    if (num_lines <= wsz.ws_row) {
        return 0; /* no need to use 'less' */
    }

    ret = pipe(output_pipefds);
    if (ret < 0) {
        print_error("pipe() failed: %m");
        return ret;
    }

    pid = fork();
    if (pid < 0) {
        print_error("fork() failed: %m");
        close_pipes();
        return pid;
    }

    /* Parent replaces itself with 'less'
     * Child continues to dump log
     */
    if (pid == 0) {
        /* redirect output to pipe */
        ret = dup2(output_pipefds[1], fileno(stdout));
        if (ret < 0) {
            print_error("failed to redirect stdout: %m");
            close_pipes();
            return ret;
        }

        close_pipes();
        return 0;
    } else {
        /* redirect input from pipe */
        ret = dup2(output_pipefds[0], fileno(stdin));
        if (ret < 0) {
            print_error("failed to redirect stdin: %m");
            exit(ret);
        }

        close_pipes();

        /* If PAGER environment variable is set, use it. If it's not set, or it
         * is equal to "less", use "less -R" to show colors.
         */
        pager_cmd = getenv("PAGER");
        if ((pager_cmd == NULL) || !strcmp(pager_cmd, PAGER_LESS)) {
            pager_cmd = PAGER_LESS_CMD;
        }

        /* coverity[tainted_string] */
        ret = execlp(shell_cmd, shell_cmd, "-c", pager_cmd, NULL);
        if (ret) {
            print_error("failed to execute shell '%s': %m", shell_cmd);
        }
        return ret;
    }
}

static void show_header(profile_data_t *data, options_t *opts)
{
    char buf[80];

    printf("\n");
    printf("   ucs lib : %s\n", data->header->ucs_path);
    printf("   host    : %s\n", data->header->hostname);
    printf("   command : %s\n", data->header->cmdline);
    printf("   pid     : %d\n", data->header->pid);
    printf("   threads : %-3d", data->header->num_threads);
    if (opts->thread_list[0] != -1) {
        printf("(showing %s",
               (opts->thread_list[1] == -1) ? "thread" : "threads");
        printf(" %s)", thread_list_str(opts->thread_list, buf, sizeof(buf)));
    }
    printf("\n\n");
}

static int compare_int(const void *a, const void *b)
{
    return *(const int*)a - *(const int*)b;
}

static int show_profile_data(profile_data_t *data, options_t *opts)
{
    unsigned i, thread_list_len;
    int ret;
    int *t;

    if (data->header->num_threads > MAX_THREADS) {
        print_error("the profile contains %u threads, but only up to %d are "
                    "supported", data->header->num_threads, MAX_THREADS);
        return -EINVAL;
    }

    /* validate and count thread numbers */
    if (opts->thread_list[0] == -1) {
        for (i = 0; i < data->header->num_threads; ++i) {
            opts->thread_list[i] = i + 1;
        }
        opts->thread_list[i] = -1;
    } else {
        thread_list_len = 0;
        for (t = opts->thread_list; *t != -1; ++t) {
            if (*t > data->header->num_threads) {
                print_error("thread number %d is out of range (1..%u)",
                            *t, data->header->num_threads);
                return -EINVAL;
            }

            ++thread_list_len;
        }
        assert(thread_list_len > 0);

        /* sort thread numbers and check for duplicates */
        qsort(opts->thread_list, thread_list_len, sizeof(int), compare_int);
        for (t = opts->thread_list; *t != -1; ++t) {
            if (t[0] == t[1]) {
                print_error("duplicate thread number %d", t[0]);
                return -EINVAL;
            }
        }
    }

    /* redirect output if needed */
    if (!opts->raw) {
        ret = redirect_output(data, opts);
        if (ret < 0) {
            return ret;
        }
    }

    show_header(data, opts);

    if (data->header->mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) {
        show_profile_data_accum(data, opts);
        printf("\n");
    }

    if (data->header->mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) {
        for (t = opts->thread_list; *t != -1; ++t) {
            show_profile_data_log(data, opts, *t - 1);
        }
        printf("\n");
    }

    return 0;
}

static void usage()
{
    printf("Usage: ucx_read_profile [options] [profile-file]\n");
    printf("Options are:\n");
    printf("  -r              Show raw output\n");
    printf("  -T <threads>    Comma-separated list of threads to show, "
           "e.g. \"1,2,3\", or \"all\" to show all threads\n");
    printf("  -t <units>      Select time units to use:\n");
    printf("                     sec  - seconds\n");
    printf("                     msec - milliseconds\n");
    printf("                     usec - microseconds (default)\n");
    printf("                     nsec - nanoseconds\n");
    printf("  -h              Show this help message\n");
}

static int parse_args(int argc, char **argv, options_t *opts)
{
    int ret, c;

    opts->raw         = !isatty(fileno(stdout));
    opts->time_units  = TIME_UNITS_USEC;
    ret = parse_thread_list(opts->thread_list, "all");
    if (ret < 0) {
        return ret;
    }

    while ( (c = getopt(argc, argv, "rT:t:h")) != -1 ) {
        switch (c) {
        case 'r':
            opts->raw = 1;
            break;
        case 'T':
            ret = parse_thread_list(opts->thread_list, optarg);
            if (ret < 0) {
                return ret;
            }
            break;
        case 't':
            if (!strcasecmp(optarg, "sec")) {
                opts->time_units = TIME_UNITS_SEC;
            } else if (!strcasecmp(optarg, "msec")) {
                opts->time_units = TIME_UNITS_MSEC;
            } else if (!strcasecmp(optarg, "usec")) {
                opts->time_units = TIME_UNITS_USEC;
            } else if (!strcasecmp(optarg, "nsec")) {
                opts->time_units = TIME_UNITS_NSEC;
            } else {
                print_error("invalid time units '%s'\n", optarg);
                usage();
                return -1;
            }
            break;
        case 'h':
            usage();
            return -127;
        default:
            usage();
            return -1;
        }
    }

    if (optind >= argc) {
        print_error("missing profile file argument\n");
        usage();
        return -1;
    }

    opts->filename = argv[optind];
    return 0;
}

int main(int argc, char **argv)
{
    profile_data_t data = {0};
    options_t opts;
    int ret;

    ret = parse_args(argc, argv, &opts);
    if (ret < 0) {
        return (ret == -127) ? 0 : ret;
    }

    ret = read_profile_data(opts.filename, &data);
    if (ret < 0) {
        return ret;
    }

    ret = show_profile_data(&data, &opts);
    release_profile_data(&data);
    return ret;
}

