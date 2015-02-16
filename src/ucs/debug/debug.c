/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * $COPYRIGHT$
 * $HEADER$
 */

#define _GNU_SOURCE

#include "debug.h"
#include "log.h"

#include <ucs/sys/sys.h>
#include <sys/wait.h>
#include <execinfo.h>
#include <dlfcn.h>
#include <link.h>
#ifdef HAVE_DETAILED_BACKTRACE
#  if HAVE_LIBIBERTY_H
#    include <libiberty.h>
#elif HAVE_LIBIBERTY_LIBIBERTY_H
#    include <libiberty/libiberty.h>
#  endif
#  include <bfd.h>
#endif /* HAVE_DETAILED_BACKTRACE */


#define UCS_GDB_MAX_ARGS  32
#define BACKTRACE_MAX 64

struct dl_address_search {
    unsigned long            address;
    const char               *filename;
    unsigned long            base;
};

#ifdef HAVE_DETAILED_BACKTRACE

struct backtrace_line {
    unsigned long            address;
    char                     *file;
    char                     *function;
    unsigned                 lineno;
};

struct backtrace_file {
    struct dl_address_search dl;
    bfd                      *abfd;
    asymbol                  **syms;
};

typedef struct backtrace *backtrace_h;
struct backtrace {
    struct backtrace_line    lines[BACKTRACE_MAX];
    int                      size;
    int                      position;
};

struct backtrace_search {
    int                      count;
    struct backtrace_file    *file;
    int                      backoff; /* search the line where the function call
                                         took place, instead of return address */
    struct backtrace_line    *lines;
    int                      max_lines;
};

#endif /* HAVE_DETAILED_BACKTRACE */

#define UCS_SYS_SIGNAME(signame) [SIG ## signame] = #signame
const char *ucs_signal_names[] = {
    [0] = "SIGNAL0",
    UCS_SYS_SIGNAME(HUP),
    UCS_SYS_SIGNAME(INT),
    UCS_SYS_SIGNAME(QUIT),
    UCS_SYS_SIGNAME(ILL),
    UCS_SYS_SIGNAME(TRAP),
    UCS_SYS_SIGNAME(ABRT),
    UCS_SYS_SIGNAME(BUS),
    UCS_SYS_SIGNAME(FPE),
    UCS_SYS_SIGNAME(KILL),
    UCS_SYS_SIGNAME(USR1),
    UCS_SYS_SIGNAME(SEGV),
    UCS_SYS_SIGNAME(USR2),
    UCS_SYS_SIGNAME(PIPE),
    UCS_SYS_SIGNAME(ALRM),
    UCS_SYS_SIGNAME(TERM),
    UCS_SYS_SIGNAME(STKFLT),
    UCS_SYS_SIGNAME(CHLD),
    UCS_SYS_SIGNAME(CONT),
    UCS_SYS_SIGNAME(STOP),
    UCS_SYS_SIGNAME(TSTP),
    UCS_SYS_SIGNAME(TTIN),
    UCS_SYS_SIGNAME(TTOU),
    UCS_SYS_SIGNAME(URG),
    UCS_SYS_SIGNAME(XCPU),
    UCS_SYS_SIGNAME(XFSZ),
    UCS_SYS_SIGNAME(VTALRM),
    UCS_SYS_SIGNAME(PROF),
    UCS_SYS_SIGNAME(WINCH),
    UCS_SYS_SIGNAME(IO),
    UCS_SYS_SIGNAME(PWR),
    UCS_SYS_SIGNAME(SYS),
    [SIGSYS + 1] = NULL
};


static int dl_match_address(struct dl_phdr_info *info, size_t size, void *data)
{
    struct dl_address_search *dl = data;
    const ElfW(Phdr) *phdr;
    ElfW(Addr) load_base = info->dlpi_addr;
    long n;

    phdr = info->dlpi_phdr;
    for (n = info->dlpi_phnum; --n >= 0; phdr++) {
        if (phdr->p_type == PT_LOAD) {
            ElfW(Addr) vbaseaddr = phdr->p_vaddr + load_base;
            if (dl->address >= vbaseaddr && dl->address < vbaseaddr + phdr->p_memsz) {
                dl->filename = info->dlpi_name;
                dl->base     = info->dlpi_addr;
            }
        }
    }
    return 0;
}

static int dl_lookup_address(struct dl_address_search *dl)
{
    dl->filename = NULL;
    dl->base     = 0;

    dl_iterate_phdr(dl_match_address, dl);
    if (dl->filename == NULL) {
        return 0;
    }

    if (strlen(dl->filename) == 0) {
        dl->filename = ucs_get_exe();
    }
    return 1;
}

#ifdef HAVE_DETAILED_BACKTRACE

/*
 * The dl member in file should be initialized
 */
static int load_file(struct backtrace_file *file)
{
    long symcount;
    unsigned int size;
    char **matching;

    file->syms = NULL;
    file->abfd = bfd_openr(file->dl.filename, NULL);
    if (!file->abfd) {
        goto err;
    }

    if (bfd_check_format(file->abfd, bfd_archive)) {
        goto err_close;
    }

    if (!bfd_check_format_matches(file->abfd, bfd_object, &matching)) {
        goto err_close;
    }

    if ((bfd_get_file_flags(file->abfd) & HAS_SYMS) == 0) {
        goto err_close;
    }

    symcount = bfd_read_minisymbols(file->abfd, 0, (PTR)&file->syms, &size);
    if (symcount == 0) {
        free(file->syms);
        symcount = bfd_read_minisymbols(file->abfd, 1, (PTR)&file->syms, &size);
    }
    if (symcount < 0) {
        goto err_close;
    }

    return 1;

err_close:
    bfd_close(file->abfd);
err:
    return 0;
}

static void unload_file(struct backtrace_file *file)
{
    free(file->syms);
    bfd_close(file->abfd);
}

static void find_address_in_section(bfd *abfd, asection *section, void *data)
{
    struct backtrace_search *search = data;
    bfd_size_type size;
    bfd_vma vma;
    unsigned long address;
    const char *filename, *function;
    unsigned lineno;
    int found;

    if ((search->count > 0) || (search->max_lines == 0) ||
        ((bfd_get_section_flags(abfd, section) & SEC_ALLOC) == 0)) {
        return;
    }

    address = search->file->dl.address - search->file->dl.base;
    vma = bfd_get_section_vma(abfd, section);
    if (address < vma) {
        return;
    }

    size = bfd_section_size(abfd, section);
    if (address >= vma + size) {
        return;
    }

    /* Search in address-1 to get the calling line instead of return address */
    found = bfd_find_nearest_line(abfd, section, search->file->syms,
                                  address - vma - search->backoff,
                                  &filename, &function, &lineno);
    do {
        search->lines[search->count].address  = address;
        search->lines[search->count].file     = filename ? strdup(filename) : NULL;
        search->lines[search->count].function = function ? strdup(function) : NULL;
        search->lines[search->count].lineno   = lineno;
        if (search->count == 0) {
            /* To get the inliner info, search at the original address */
            bfd_find_nearest_line(abfd, section, search->file->syms, address - vma,
                                  &filename, &function, &lineno);
        }

        ++search->count;
        found = bfd_find_inliner_info(abfd, &filename, &function, &lineno);
    } while (found && (search->count < search->max_lines));
}

static int get_line_info(struct backtrace_file *file, int backoff,
                         struct backtrace_line *lines, int max)
{
    struct backtrace_search search;

    search.file      = file;
    search.backoff   = backoff;
    search.count     = 0;
    search.lines     = lines;
    search.max_lines = max;
    bfd_map_over_sections(file->abfd, find_address_in_section, &search);
    return search.count;
}

/**
 * Create a backtrace from the calling location.
 *
 * @return             Backtrace object.
 */
backtrace_h backtrace_create(void)
{
    struct backtrace_file file;
    void *addresses[BACKTRACE_MAX];
    int i, num_addresses;
    backtrace_h bckt;

    bckt = malloc(sizeof *bckt);
    if (!bckt) {
        return NULL;
    }

    num_addresses = backtrace(addresses, BACKTRACE_MAX);

    bckt->size = 0;
    for (i = 0; i < num_addresses; ++i) {
        file.dl.address = (unsigned long)addresses[i];
        if (dl_lookup_address(&file.dl) && load_file(&file)) {
            bckt->size += get_line_info(&file, 1, bckt->lines + bckt->size,
                                        BACKTRACE_MAX - bckt->size);
            unload_file(&file);
        }
    }

    bckt->position = 0;
    return bckt;
}

/**
 * Destroy a backtrace and free all memory.
 *
 * @param bckt          Backtrace object.
 */
void backtrace_destroy(backtrace_h bckt)
{
    int i;

    for (i = 0; i < bckt->size; ++i) {
        free(bckt->lines[i].function);
        free(bckt->lines[i].file);
    }
    free(bckt);
}

void ucs_debug_get_line_info(const char *filename, unsigned long base,
                             unsigned long address, ucs_debug_address_info_t *info)
{
    struct backtrace_file file;
    struct backtrace_line line;
    int count;

    file.dl.filename = filename;
    file.dl.base     = base;
    file.dl.address  = address;

    if (!load_file(&file)) {
        goto err;
    }

    count = get_line_info(&file, 0, &line, 1);
    if (count == 0) {
        goto err_unload;
    }

    if (line.function) {
        strncpy(info->function, line.function, sizeof(info->function));
    } else {
        strcpy(info->function, "???");
    }
    if (line.file) {
        strncpy(info->source_file, line.file, sizeof(info->source_file));
    } else {
        strcpy(info->function, "???");
    }
    info->line_number = line.lineno;

    free(line.function);
    free(line.file);
    unload_file(&file);
    return;

err_unload:
    unload_file(&file);
err:
    strcpy(info->function, "");
    strcpy(info->source_file, "");
    info->line_number = 0;
}

/**
 * Walk to the next backtrace line information.
 *
 * @param bckt          Backtrace object.
 * @param address       Filled with backtrace address.
 * @param file          Filled with a pointer to the source file name.
 * @param function      Filled with a pointer to function name.
 * @param lineno        Filled with source line number.
 *
 * NOTE: the file and function memory remains valid as long as the backtrace
 * object is not destroyed.
 */
int backtrace_next(backtrace_h bckt, unsigned long *address, char const ** file,
                   char const ** function, unsigned *lineno)
{
    struct backtrace_line *line;

    if (bckt->position >= bckt->size)
        return 0;

    line = &bckt->lines[bckt->position++];
    *address = line->address;
    *file = line->file;
    *function = line->function;
    *lineno = line->lineno;
    return 1;
}

void ucs_debug_print_backtrace(FILE *stream, int strip)
{
    backtrace_h bckt;
    unsigned long address;
    const char *file, *function;
    unsigned line;
    int i;

    bckt = backtrace_create();

    fprintf(stream, "==== backtrace ====\n");
    i = 0;
    while (backtrace_next(bckt, &address, &file, &function, &line)) {
        if (i >= strip) {
            fprintf(stream, "%2d 0x%016lx %s()  %s:%u\n", i, address,
                    function ? function : "??", file ? file : "??", line);
        }
        ++i;
    }
    fprintf(stream, "===================\n");

    backtrace_destroy(bckt);
}

const char *ucs_debug_get_symbol_name(void *address, char *buffer, size_t max)
{
    ucs_debug_address_info_t info;
    ucs_debug_lookup_address(address, &info);
    return strncpy(buffer, info.function, max);
}

#else /* HAVE_DETAILED_BACKTRACE */

void ucs_debug_get_line_info(const char *filename, unsigned long base, unsigned long address,
                             ucs_debug_address_info_t *info)
{
    strcpy(info->function, "");
    strcpy(info->source_file, "");
    info->line_number = 0;
}

void ucs_debug_print_backtrace(FILE *stream, int strip)
{
    char **symbols;
    void *addresses[BACKTRACE_MAX];
    int count, i;

    fprintf(stream, "==== backtrace ====\n");

    count = backtrace(addresses, BACKTRACE_MAX);
    symbols = backtrace_symbols(addresses, count);
    for (i = strip; i < count; ++i) {
            fprintf(stream, "   %2d  %s\n", i - strip, symbols[i]);
    }
    free(symbols);

    fprintf(stream, "===================\n");
}

const char *ucs_debug_get_symbol_name(void *address, char *buffer, size_t max)
{
    Dl_info info;
    int ret;

    ret = dladdr(address, &info);
    if (ret != 0) {
        return NULL;
    }

    return strncpy(buffer, info.dli_sname, max);
}

#endif /* HAVE_DETAILED_BACKTRACE */


static ucs_status_t ucs_debugger_attach()
{
    static const char *gdb_commands = "bt\n";
    const char *cmds;
    char *gdb_cmdline;
    char gdb_commands_file[256];
    char* argv[6 + UCS_GDB_MAX_ARGS];
    pid_t pid, debug_pid;
    int fd, ret, narg;
    int valgrind;
    char *self_exe;

    debug_pid = getpid();

    /* Fork a process which will execute gdb */
    pid = fork();
    if (pid < 0) {
        ucs_log_fatal_error("fork returned %d: %m", pid);
        return UCS_ERR_IO_ERROR;
    }

    valgrind = RUNNING_ON_VALGRIND;
    self_exe = strdup(ucs_get_exe());

    if (pid == 0) {
        gdb_cmdline = strdup(ucs_global_opts.gdb_command);
        narg = 0;
        argv[narg] = strtok(gdb_cmdline, " \t");
        while (argv[narg] != NULL) {
            ++narg;
            argv[narg] = strtok(NULL, " \t");
        }

        if (!valgrind) {
            argv[narg++] = "-p";
            if (asprintf(&argv[narg++], "%d", debug_pid)<0) {
                ucs_log_fatal_error("Failed to extract pid : %m");
                exit(-1);
            }
        }

        /* Generate a file name for gdb commands */
        memset(gdb_commands_file, 0, sizeof(gdb_commands_file));
        snprintf(gdb_commands_file, sizeof(gdb_commands_file) - 1,
                 "/tmp/.gdbcommands.%s", getlogin());

        /* Write gdb commands and add the file to argv is successful */
        fd = open(gdb_commands_file, O_WRONLY|O_TRUNC|O_CREAT, 0600);
        if (fd >= 0) {
            if (valgrind) {
                if (asprintf((char**)&cmds, "file %s\n"
                                            "target remote | vgdb\n"
                                            "%s",
                                            self_exe, gdb_commands) < 0) {
                    cmds = "";
                }
            } else {
                cmds = gdb_commands;
            }

            if (write(fd, cmds, strlen(cmds)) == strlen(cmds)) {
                argv[narg++] = "-x";
                argv[narg++] = gdb_commands_file;
            } else {
                ucs_log_fatal_error("Unable to write to command file: %m");
            }
            close(fd);
        } else {
            ucs_log_fatal_error("Unable to open '%s' for writing: %m",
                                gdb_commands_file);
        }

        argv[narg++] = NULL;

        /* Execute GDB */
        ret = execvp(argv[0], argv);
        if (ret < 0) {
            ucs_log_fatal_error("Failed to execute %s: %m", argv[0]);
            exit(-1);
        }
    }

    free(self_exe);
    waitpid(pid, &ret, 0);
    return UCS_OK;
}

static void UCS_F_NOINLINE ucs_debug_freeze()
{
    static volatile int freeze = 1;
    while (freeze) {
        pause();
    }
}

static int ucs_debug_stop_exclude_thread = -1;
static void ucs_debug_stop_handler(int signo)
{
    if (ucs_get_tid() == ucs_debug_stop_exclude_thread) {
        return;
    }

    ucs_debug_freeze();
}

static void ucs_debug_stop_other_threads()
{
    ucs_debug_stop_exclude_thread = ucs_get_tid();
    signal(SIGUSR1, ucs_debug_stop_handler);
    kill(0, SIGUSR1);
}

static ucs_status_t ucs_error_freeze()
{
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    char response;
    int ret;

    ucs_debug_stop_other_threads();

    if (pthread_mutex_trylock(&lock) == 0) {

        if (strlen(ucs_global_opts.gdb_command) && isatty(fileno(stdout)) &&
            isatty(fileno(stdin)))
        {
            ucs_log_fatal_error("Process frozen, press Enter to attach a debugger...");
            ret = read(fileno(stdin), &response, 1); /* Use low-level input to avoid deadlock */
            if ((ret == 1) && (response == '\n')) {
                ucs_debugger_attach();
            } else {
                ucs_debug_freeze();
            }
        } else {
            ucs_log_fatal_error("Process frozen...");
            ucs_debug_freeze();
        }

        pthread_mutex_unlock(&lock);
    } else {
        ucs_debug_freeze();
    }

    return UCS_OK;
}

static void ucs_error_signal_handler(int signo)
{
    ucs_debug_cleanup();
    ucs_log_flush();
    ucs_log_fatal_error("Caught signal %d (%s)", signo, strsignal(signo));
    if (signo != SIGINT && signo != SIGTERM) {
        ucs_handle_error();
    }
    raise(signo);
}

void ucs_handle_error()
{
    ucs_status_t status;

    switch (ucs_global_opts.handle_errors) {
    case UCS_HANDLE_ERROR_DEBUG:
        status = ucs_debugger_attach();
        if (status == UCS_OK) {
            break;
        }
        /* Fall thru */

    case UCS_HANDLE_ERROR_FREEZE:
        status = ucs_error_freeze();
        if (status == UCS_OK) {
            break;
        }
        /* Fall thru */

    case UCS_HANDLE_ERROR_BACKTRACE:
        ucs_debug_print_backtrace(stderr, 2);
        break;

    default:
        break;
    }
}

static void ucs_debug_signal_handler(int signo)
{
    ucs_log_flush();

    ucs_log_fatal_error("Got debug signal, raising log level",
                        ucs_get_host_name(), getpid());
    ucs_global_opts.log_level = UCS_LOG_LEVEL_TRACE_DATA;
}

static void ucs_set_signal_handler(__sighandler_t handler)
{
    int i;

    for (i = 0; i < ucs_global_opts.error_signals.count; ++i) {
        signal(ucs_global_opts.error_signals.signals[i], handler);
    }
}

ucs_status_t ucs_debug_lookup_address(void *address, ucs_debug_address_info_t *info)
{
    struct dl_address_search dl;

    dl.address = (unsigned long)address;
    if (!dl_lookup_address(&dl)) {
        return UCS_ERR_NO_ELEM;
    }

    memset(info, 0, sizeof(*info));
    info->file.base = dl.base;
    ucs_expand_path(dl.filename, info->file.path, sizeof(info->file.path));

    ucs_debug_get_line_info(dl.filename, dl.base, (unsigned long)address, info);
    return UCS_OK;
}

static struct dl_address_search *ucs_debug_get_lib_info()
{
    static struct dl_address_search dl = {0, NULL, 0};

    if (dl.address == 0) {
        dl.address = (unsigned long)&ucs_debug_get_lib_info;
        if (!dl_lookup_address(&dl)) {
            dl.filename = NULL;
            dl.base     = 0;
        }
    }

    /* If we failed to look up the address, return NULL */
    return (dl.filename == NULL || dl.base == 0) ? NULL : &dl;
}

const char *ucs_debug_get_lib_path()
{
    static char ucs_lib_path[256] = {0};
    struct dl_address_search *dl;

    if (!strlen(ucs_lib_path)) {
        dl = ucs_debug_get_lib_info();
        if (dl != NULL) {
            ucs_expand_path(dl->filename, ucs_lib_path, sizeof(ucs_lib_path));
        }
    }

    return ucs_lib_path;
}

unsigned long ucs_debug_get_lib_base_addr()
{
    struct dl_address_search *dl = ucs_debug_get_lib_info();
    return (dl == NULL) ? 0 : dl->base;
}

void ucs_debug_init()
{
    if (ucs_global_opts.handle_errors > UCS_HANDLE_ERROR_NONE) {
        ucs_set_signal_handler(ucs_error_signal_handler);
    }
    if (ucs_global_opts.debug_signo > 0) {
        signal(ucs_global_opts.debug_signo, ucs_debug_signal_handler);
    }

#ifdef HAVE_DETAILED_BACKTRACE
    bfd_init();
#endif
}

void ucs_debug_cleanup()
{
    if (ucs_global_opts.handle_errors > UCS_HANDLE_ERROR_NONE) {
        ucs_set_signal_handler(SIG_DFL);
    }
    if (ucs_global_opts.debug_signo > 0) {
        signal(ucs_global_opts.debug_signo, SIG_DFL);
    }
}
