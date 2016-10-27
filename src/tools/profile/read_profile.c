/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/debug/profile.h>

#include <sys/signal.h>
#include <sys/fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdlib.h>
#include <getopt.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>


#define INDENT             4
#define LESS_COMMAND       "less"

#define TERM_COLOR_CLEAR   "\x1B[0m"
#define TERM_COLOR_RED     "\x1B[31m"
#define TERM_COLOR_GREEN   "\x1B[32m"
#define TERM_COLOR_YELLOW  "\x1B[33m"
#define TERM_COLOR_BLUE    "\x1B[34m"
#define TERM_COLOR_MAGENTA "\x1B[35m"
#define TERM_COLOR_CYAN    "\x1B[36m"
#define TERM_COLOR_WHITE   "\x1B[37m"
#define TERM_COLOR_GRAY    "\x1B[90m"


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
} options_t;


typedef struct {
    void                         *mem;
    size_t                       length;
    const ucs_profile_header_t   *header;
    const ucs_profile_location_t *locations;
    const ucs_profile_record_t   *records;
} profile_data_t;


/* Used to redirect output to a "less" command */
static int output_pipefds[2] = {-1, -1};


static const char* time_units_str[] = {
    [TIME_UNITS_NSEC] = "nsec",
    [TIME_UNITS_USEC] = "usec",
    [TIME_UNITS_MSEC] = "msec",
    [TIME_UNITS_SEC]  = "sec",
    [TIME_UNITS_LAST] = NULL
};


static int read_profile_data(const char *file_name, profile_data_t *data)
{
    struct stat stat;
    int ret, fd;

    fd = open(file_name, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Failed to open %s: %m\n", file_name);
        ret = fd;
        goto out;
    }

    ret = fstat(fd, &stat);
    if (ret < 0) {
        fprintf(stderr, "fstat(%s) failed: %m\n", file_name);
        goto out_close;
    }

    data->length = stat.st_size;
    data->mem    = mmap(NULL, stat.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data->mem == MAP_FAILED) {
        fprintf(stderr, "mmap(%s, length=%zd) failed: %m\n", file_name,
                data->length);
        goto out_close;
    }

    data->header    = data->mem;
    data->locations = (const void*)(data->header + 1);
    data->records   = (const void*)(data->locations + data->header->num_locations);

    ret = 0;

out_close:
    close(fd);
out:
    return ret;
}

static void release_profile_data(profile_data_t *data)
{
    munmap(data->mem, data->length);
}

static double time_to_usec(profile_data_t *data, options_t *opts, uint64_t time)
{
    static const double time_units_val[] = {
        [TIME_UNITS_NSEC] = 1e9,
        [TIME_UNITS_USEC] = 1e6,
        [TIME_UNITS_MSEC] = 1e3,
        [TIME_UNITS_SEC]  = 1e0
    };

    return time * time_units_val[opts->time_units] / data->header->one_second;
}

static void show_profile_data_accum(profile_data_t *data, options_t *opts)
{
    const ucs_profile_location_t *loc;

    printf("%25s %13s %13s %10s             FILE     FUNCTION\n",
           "NAME", "AVG", "TOTAL", "COUNT");

    for (loc = data->locations; loc < data->locations + data->header->num_locations;
                    ++loc)
    {
        switch (loc->type) {
        case UCS_PROFILE_TYPE_SAMPLE:
        case UCS_PROFILE_TYPE_SCOPE_END:
            printf("%25s %13.3f %13.0f %10ld %15s:%-4d %s()\n",
                   loc->name,
                   time_to_usec(data, opts, loc->total_time) / loc->count,
                   time_to_usec(data, opts, loc->total_time),
                   (long)loc->count,
                   loc->file, loc->line, loc->function);
        default:
            break;
        }
    }
}

static void show_profile_data_log(profile_data_t *data, options_t *opts)
{
    size_t num_recods               = data->header->num_records;
    const ucs_profile_record_t **stack[UCS_PROFILE_STACK_MAX * 2];
    const ucs_profile_record_t **scope_ends;
    const ucs_profile_location_t *loc;
    const ucs_profile_record_t *rec, *se, **sep;
    off_t min_nesting_idx;
    int nesting, min_nesting;
    uint64_t prev_time;
    char buf[256];

#define NAME_COLOR       (opts->raw ? "" : TERM_COLOR_CYAN)
#define TS_COLOR         (opts->raw ? "" : TERM_COLOR_WHITE)
#define LOC_COLOR        (opts->raw ? "" : TERM_COLOR_GRAY)
#define CLEAR_COLOR      (opts->raw ? "" : TERM_COLOR_CLEAR)
#define RECORD_FMT       "%s%10.3f%s%*s"
#define RECORD_ARG(_ts)  TS_COLOR, time_to_usec(data, opts, (_ts)), CLEAR_COLOR, \
                         INDENT * nesting, ""
#define PRINT_RECORD()   printf("%-*s %s%15s:%-4d %s()%s\n", \
                                (int)(50 + strlen(NAME_COLOR) + \
                                      2 * strlen(TS_COLOR) + \
                                      3 * strlen(CLEAR_COLOR)), \
                                buf, \
                                LOC_COLOR, \
                                basename(loc->file), loc->line, loc->function, \
                                CLEAR_COLOR)

    scope_ends = calloc(1, sizeof(*scope_ends) * num_recods);
    if (scope_ends == NULL) {
        fprintf(stderr, "Failed to allocate memory\n");
        return;
    }

    memset(stack, 0, sizeof(stack));

    /* Find the first record with minimal nesting level, which is the base of call stack */
    nesting         = 0;
    min_nesting     = 0;
    min_nesting_idx = -1;
    for (rec = data->records; rec < data->records + num_recods; ++rec) {
        loc = &data->locations[rec->location];
        switch (loc->type) {
        case UCS_PROFILE_TYPE_SCOPE_BEGIN:
            ++nesting;
            stack[nesting + UCS_PROFILE_STACK_MAX] = &scope_ends[rec - data->records];
            break;
        case UCS_PROFILE_TYPE_SCOPE_END:
            if (nesting < min_nesting) {
                min_nesting     = nesting;
                min_nesting_idx = rec - data->records;
            }
            sep = stack[nesting + UCS_PROFILE_STACK_MAX];
            if (sep != NULL) {
                *sep = rec;
            }
            --nesting;
            break;
        default:
            break;
        }
    }

    if (num_recods > 0) {
        prev_time = data->records[0].timestamp;
    } else {
        prev_time = 0;
    }

    /* Display records */
    nesting = 0;
    for (rec = data->records + min_nesting_idx + 1; rec < data->records + num_recods; ++rec) {
        loc = &data->locations[rec->location];
        switch (loc->type) {
        case UCS_PROFILE_TYPE_SAMPLE:
            snprintf(buf, sizeof(buf), RECORD_FMT"  %s%s%s",
                     RECORD_ARG(rec->timestamp - prev_time),
                     NAME_COLOR, loc->name, CLEAR_COLOR);
            PRINT_RECORD();
            break;
        case UCS_PROFILE_TYPE_SCOPE_BEGIN:
            se = scope_ends[rec - data->records];
            if (se != NULL) {
                snprintf(buf, sizeof(buf), RECORD_FMT"  %s%s%s %s%.3f%s {",
                         RECORD_ARG(rec->timestamp - prev_time),
                         NAME_COLOR, data->locations[se->location].name, CLEAR_COLOR,
                         TS_COLOR, time_to_usec(data, opts, se->timestamp - rec->timestamp),
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
        default:
            break;
        }
        prev_time = rec->timestamp;
    }

    free(scope_ends);
}

static void close_pipes()
{
    close(output_pipefds[0]);
    close(output_pipefds[1]);
}

static int redirect_output(const ucs_profile_header_t *hdr)
{
    char *less_argv[] = {LESS_COMMAND,
                         "-R" /* show colors */,
                         NULL};;
    struct winsize wsz;
    uint64_t num_lines;
    pid_t pid;
    int ret;

    ret = ioctl(STDOUT_FILENO, TIOCGWINSZ, &wsz);
    if (ret < 0) {
        fprintf(stderr, "ioctl(TIOCGWINSZ) failed: %m\n");
        return ret;
    }

    num_lines = 6 + /* header */
                ((hdr->mode & UCS_BIT(UCS_PROFILE_MODE_ACCUM)) ?
                                (hdr->num_locations + 2) : 0) +
                ((hdr->mode & UCS_BIT(UCS_PROFILE_MODE_LOG)) ?
                                (hdr->num_records    + 1) : 0) +
                1; /* footer */

    if (num_lines <= wsz.ws_row) {
        return 0; /* no need to use 'less' */
    }

    ret = pipe(output_pipefds);
    if (ret < 0) {
        fprintf(stderr, "pipe() failed: %m\n");
        return ret;
    }

    pid = fork();
    if (pid < 0) {
        fprintf(stderr, "fork() failed: %m\n");
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
            fprintf(stderr, "Failed to redirect stdout: %m\n");
            close_pipes();
            return ret;
        }

        close_pipes();
        return 0;
    } else {
        /* redirect input from pipe */
        ret = dup2(output_pipefds[0], fileno(stdin));
        if (ret < 0) {
            fprintf(stderr, "Failed to redirect stdin: %m\n");
            exit(ret);
        }

        close_pipes();
        return execvp(LESS_COMMAND, less_argv);
    }
}

static void show_header(profile_data_t *data, options_t *opts)
{
    printf("\n");
    printf("   command : %s\n", data->header->cmdline);
    printf("   host    : %s\n", data->header->hostname);
    printf("   pid     : %d\n", data->header->pid);
    printf("   units   : %s\n", time_units_str[opts->time_units]);
    printf("\n");
}

static int show_profile_data(profile_data_t *data, options_t *opts)
{
    int ret;

    if (!opts->raw) {
        ret = redirect_output(data->header);
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
        show_profile_data_log(data, opts);
        printf("\n");
    }

    return 0;
}

int parse_args(int argc, char **argv, options_t *opts)
{
    int c;

    opts->raw         = !isatty(fileno(stdout));
    opts->time_units  = TIME_UNITS_USEC;

    while ( (c = getopt(argc, argv, "hru:")) != -1 ) {
        switch (c) {
        case 'r':
            opts->raw = 1;
            break;
        case 'u':
            if (!strcasecmp(optarg, "sec")) {
                opts->time_units = TIME_UNITS_SEC;
            } else if (!strcasecmp(optarg, "msec")) {
                opts->time_units = TIME_UNITS_MSEC;
            } else if (!strcasecmp(optarg, "usec")) {
                opts->time_units = TIME_UNITS_USEC;
            } else if (!strcasecmp(optarg, "msec")) {
                opts->time_units = TIME_UNITS_NSEC;
            }
            break;
        case 'h':
        default:
            return -1;
        }
    }

    if (optind >= argc) {
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

    if (parse_args(argc, argv, &opts) < 0) {
        printf("Usage: %s [options] <file>\n", basename(argv[0]));
        printf("Options:\n");
        printf("      -r             raw output\n");
        printf("      -t UNITS       select time units (sec/msec/usec/nsec)\n");
        printf("\n");
        return -1;
    }

    if (read_profile_data(opts.filename, &data) < 0) {
        return -1;
    }

    ret = show_profile_data(&data, &opts);
    release_profile_data(&data);
    return ret;
}

