/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "sys.h"
#include "checker.h"
#include "string.h"
#include "math.h"

#include <ucs/debug/log.h>
#include <ucs/time/time.h>
#include <ucm/util/sys.h>
#include <sys/ioctl.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <net/if.h>
#include <dirent.h>
#include <sched.h>


/* Default huge page size is 2 MBytes */
#define UCS_DEFAULT_HUGEPAGE_SIZE  (2 * UCS_MBYTE)
#define UCS_DEFAULT_MEM_FREE       640000
#define UCS_PROCESS_MAPS_FILE      "/proc/self/maps"


const char *ucs_get_host_name()
{
    static char hostname[256] = {0};

    if (*hostname == 0) {
        gethostname(hostname, sizeof(hostname));
        strtok(hostname, ".");
    }
    return hostname;
}

const char *ucs_get_user_name()
{
    static char username[256] = {0};

    if (*username == 0) {
        getlogin_r(username, sizeof(username));
    }
    return username;
}

void ucs_expand_path(const char *path, char *fullpath, size_t max)
{
    char cwd[1024] = {0};

    if (path[0] == '/') {
            strncpy(fullpath, path, max);
    } else if (getcwd(cwd, sizeof(cwd) - 1) != NULL) {
        snprintf(fullpath, max, "%s/%s", cwd, path);
    } else {
        ucs_warn("failed to expand path '%s' (%m), using original path", path);
        strncpy(fullpath, path, max);
    }
}

const char *ucs_get_exe()
{
    static char exe[1024];
    int ret;

    ret = readlink("/proc/self/exe", exe, sizeof(exe) - 1);
    if (ret < 0) {
        exe[0] = '\0';
    } else {
        exe[ret] = '\0';
    }

    return exe;
}

uint32_t ucs_file_checksum(const char *filename)
{
    char buffer[1024];
    ssize_t nread;
    int fd;
    uint32_t crc;

    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        return 0;
    }

    crc = 0;
    do {
        nread = read(fd, buffer, sizeof(buffer));
        if (nread > 0) {
            crc = ucs_calc_crc32(crc, buffer, nread);
        }
    } while (nread == sizeof(buffer));
    close(fd);

    return crc;
}

static uint64_t ucs_get_mac_address()
{
    static uint64_t mac_address = 0;
    struct ifreq ifr, *it, *end;
    struct ifconf ifc;
    char buf[1024];
    int sock;

    if (mac_address == 0) {
        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
        if (sock == -1) {
            ucs_error("failed to create socket: %m");
            return 0;
        }

        ifc.ifc_len = sizeof(buf);
        ifc.ifc_buf = buf;
        if (ioctl(sock, SIOCGIFCONF, &ifc) == -1) {
            ucs_error("ioctl(SIOCGIFCONF) failed: %m");
            close(sock);
            return 0;
        }

        it = ifc.ifc_req;
        end = it + (ifc.ifc_len / sizeof *it);
        for (it = ifc.ifc_req; it != end; ++it) {
            strcpy(ifr.ifr_name, it->ifr_name);
            if (ioctl(sock, SIOCGIFFLAGS, &ifr) != 0) {
                ucs_error("ioctl(SIOCGIFFLAGS) failed: %m");
                close(sock);
                return 0;
            }

            if (!(ifr.ifr_flags & IFF_LOOPBACK)) {
                if (ioctl(sock, SIOCGIFHWADDR, &ifr) != 0) {
                    ucs_error("ioctl(SIOCGIFHWADDR) failed: %m");
                    close(sock);
                    return 0;
                }

                memcpy(&mac_address, ifr.ifr_hwaddr.sa_data, 6);
                break;
            }
        }

        close(sock);
        ucs_trace("MAC address is 0x%012"PRIX64, mac_address);
    }

    return mac_address;
}

static uint64_t __sumup_host_name(unsigned prime_index)
{
    uint64_t sum, n;
    const char *p;
    unsigned i;

    sum = 0;
    i = prime_index;
    p = ucs_get_host_name();
    while (*p != '\0') {
        n = 0;
        strncpy((char*)&n, p, sizeof(n));
        sum += ucs_get_prime(i) * n;
        ++i;
        p += ucs_min(sizeof(n), strlen(p));
    }
    return sum;
}

uint64_t ucs_machine_guid()
{
    return ucs_get_prime(0) * ucs_get_mac_address() +
           __sumup_host_name(1);
}

int ucs_get_first_cpu()
{
    int first_cpu, total_cpus, ret;
    cpu_set_t mask;

    ret = sysconf(_SC_NPROCESSORS_CONF);
    if (ret < 0) {
        ucs_error("failed to get local cpu count: %m");
        return ret;
    }
    total_cpus = ret;

    CPU_ZERO(&mask);
    ret = sched_getaffinity(0, sizeof(mask), &mask);
    if (ret < 0) {
        ucs_error("failed to get process affinity: %m");
        return ret;
    }

    for (first_cpu = 0; first_cpu < total_cpus; ++first_cpu) {
        if (CPU_ISSET(first_cpu, &mask)) {
            return first_cpu;
        }
    }

    return total_cpus;
}

uint64_t ucs_generate_uuid(uint64_t seed)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);
    return seed +
           ucs_get_prime(0) * ucs_get_tid() +
           ucs_get_prime(1) * ucs_get_time() +
           ucs_get_prime(2) * ucs_get_mac_address() +
           ucs_get_prime(3) * tv.tv_sec +
           ucs_get_prime(4) * tv.tv_usec +
           __sumup_host_name(5);
}

ucs_status_t
ucs_open_output_stream(const char *config_str, FILE **p_fstream, int *p_need_close,
                       const char **p_next_token)
{
    FILE *output_stream;
    char filename[256];
    char *template;
    const char *p;
    size_t len;

    *p_need_close = 0;
    *p_fstream    = NULL;
    *p_next_token = config_str;

    len = strcspn(config_str, ":");
    if (!strncmp(config_str, "stdout", len)) {
        *p_fstream    = stdout;
        *p_next_token = config_str + len;
    } else if (!strncmp(config_str, "stderr", len)) {
        *p_fstream    = stderr;
        *p_next_token = config_str + len;
    } else {
        if (!strncmp(config_str, "file:", 5)) {
            p = config_str + 5;
        } else {
            p = config_str;
        }

        len = strcspn(p, ":");
        template = strndup(p, len);
        ucs_fill_filename_template(template, filename, sizeof(filename));
        free(template);

        output_stream = fopen(filename, "w");
        if (output_stream == NULL) {
            ucs_error("failed to open '%s' for writing: %m", filename);
            return UCS_ERR_IO_ERROR;
        }

        *p_fstream    = output_stream;
        *p_need_close = 1;
        *p_next_token = p + len;
    }

    return UCS_OK;
}

ssize_t ucs_read_file(char *buffer, size_t max, int silent,
                      const char *filename_fmt, ...)
{
    char filename[MAXPATHLEN];
    ssize_t read_bytes;
    va_list ap;
    int fd;

    va_start(ap, filename_fmt);
    vsnprintf(filename, MAXPATHLEN, filename_fmt, ap);
    va_end(ap);

    fd = open(filename, O_RDONLY);
    if (fd < 0) {
        if (!silent) {
            ucs_error("failed to open %s: %m", filename);
        }
        read_bytes = -1;
        goto out;
    }

    read_bytes = read(fd, buffer, max - 1);
    if (read_bytes < 0) {
        if (!silent) {
            ucs_error("failed to read from %s: %m", filename);
        }
        goto out_close;
    }

    if (read_bytes < max) {
        buffer[read_bytes] = '\0';
    }

out_close:
    close(fd);
out:
    return read_bytes;
}

size_t ucs_get_max_iov()
{
    static size_t max_iov = 1;

    if (1 == max_iov) {
        max_iov = ucs_max(sysconf(_SC_IOV_MAX), 1); /* max_iov shouldn't be zero */
    }
    return max_iov;
}

size_t ucs_get_page_size()
{
    static size_t page_size = 0;

    if (page_size == 0) {
        page_size = sysconf(_SC_PAGESIZE);
    }
    return page_size;
}

size_t ucs_get_meminfo_entry(const char* pattern)
{
    char buf[256];
    char final_pattern[80];
    int val = 0;
    size_t val_b = 0;
    FILE *f;

    f = fopen("/proc/meminfo", "r");
    if (f != NULL) {
        snprintf(final_pattern, sizeof(final_pattern), "%s: %s", pattern,
                 "%d kB");
        while (fgets(buf, sizeof(buf), f)) {
            if (sscanf(buf, final_pattern, &val) == 1) {
                val_b = val * 1024ull;
                break;
            }
        }
        fclose(f);
    }

    return val_b;
}

size_t ucs_get_memfree_size()
{
    size_t mem_free;

    mem_free = ucs_get_meminfo_entry("MemFree");
    if (mem_free == 0) {
        mem_free = UCS_DEFAULT_MEM_FREE;
        ucs_info("cannot determine free mem size, using default: %zu",
                  mem_free);
    }

    return mem_free;
}

size_t ucs_get_huge_page_size()
{
    static size_t huge_page_size = 0;

    /* Cache the huge page size value */
    if (huge_page_size == 0) {
        huge_page_size = ucs_get_meminfo_entry("Hugepagesize");
        if (huge_page_size == 0) {
            huge_page_size = UCS_DEFAULT_HUGEPAGE_SIZE;
            ucs_warn("cannot determine huge page size, using default: %zu",
                      huge_page_size);
        } else {
            ucs_trace("detected huge page size: %zu", huge_page_size);
        }
    }

    return huge_page_size;
}

size_t ucs_get_phys_mem_size()
{
    static size_t phys_pages = 0;

    if (phys_pages == 0) {
        phys_pages = sysconf(_SC_PHYS_PAGES);
    }
    return phys_pages * ucs_get_page_size();
}

#define UCS_SYS_THP_ENABLED_FILE "/sys/kernel/mm/transparent_hugepage/enabled"
int ucs_is_thp_enabled()
{
    char buf[256];
    int rc;

    rc = ucs_read_file(buf, sizeof(buf), 1, UCS_SYS_THP_ENABLED_FILE);
    if (rc < 0) {
        ucs_debug("failed to read %s:%m", UCS_SYS_THP_ENABLED_FILE);
        return 0;
    }

    return (strcmp(buf, "always madvise [never]") != 0);
}

#define UCS_PROC_SYS_SHMMAX_FILE "/proc/sys/kernel/shmmax"
size_t ucs_get_shmmax()
{
    char buf[256];
    size_t size = 0;
    int rc;

    rc = ucs_read_file(buf, sizeof(buf), 0, UCS_PROC_SYS_SHMMAX_FILE);
    if (rc < 0) {
        ucs_warn("failed to read %s:%m", UCS_PROC_SYS_SHMMAX_FILE);
        return 0;
    }

    rc = sscanf(buf, "%zu", &size);
    if (rc != 1) {
        ucs_warn("failed to parse: %m");
        return 0;
    }

    return size;
}

ucs_status_t ucs_sysv_alloc(size_t *size, void **address_p, int flags, int *shmid
                            UCS_MEMTRACK_ARG)
{
    struct shminfo shminfo, *shminfo_ptr;
    size_t alloc_size;
    void *ptr;
    int ret, err;

    alloc_size = ucs_memtrack_adjust_alloc_size(*size);

    if (flags & SHM_HUGETLB){
        alloc_size = ucs_align_up(alloc_size, ucs_get_huge_page_size());
    } else {
        alloc_size = ucs_align_up(alloc_size, ucs_get_page_size());
    }

    flags |= IPC_CREAT | SHM_R | SHM_W;
    *shmid = shmget(IPC_PRIVATE, alloc_size, flags);
    if (*shmid < 0) {
        switch (errno) {
        case ENFILE:
        case ENOMEM:
        case ENOSPC:
        case EPERM:
            if (!(flags & SHM_HUGETLB)) {
                err = errno;
                shminfo_ptr = &shminfo;
                if ((shmctl(0, IPC_INFO, (struct shmid_ds *) shminfo_ptr)) > -1) {
                    ucs_error("shmget failed: %s. (size=%zu). The max number of shared memory segments in the system is = %ld. "
                              "Please try to increase this value through /proc/sys/kernel/shmmni",
                              strerror(err), alloc_size, shminfo.shmmni);
                }
            }

            return UCS_ERR_NO_MEMORY;
        case EINVAL:
            ucs_error("A new segment was to be created and size < SHMMIN or size > SHMMAX, "
                      "or no new segment was to be created. A segment with given key existed, "
                      "but size is greater than the size of that segment. "
                      "Please check shared memory limits by 'ipcs -l'.");
            return UCS_ERR_NO_MEMORY;
        default:
            ucs_error("shmget(size=%zu, flags=0x%x) returned unexpected error: %m. "
                      "Please check shared memory limits by 'ipcs -l'.",
                      alloc_size, flags);
            return UCS_ERR_SHMEM_SEGMENT;
        }
    }

    /* Attach segment */
    if (*address_p) {
        ptr = shmat(*shmid, *address_p, SHM_REMAP);
    } else {
        ptr = shmat(*shmid, NULL, 0);
    }

    /* Remove segment, the attachment keeps a reference to the mapping */
    /* FIXME having additional attaches to a removed segment is not portable
    * behavior */
    ret = shmctl(*shmid, IPC_RMID, NULL);
    if (ret != 0) {
        ucs_warn("shmctl(IPC_RMID, shmid=%d) returned %d: %m", *shmid, ret);
    }

    /* Check if attachment was successful */
    if (ptr == (void*)-1) {
        if (errno == ENOMEM) {
            return UCS_ERR_NO_MEMORY;
        } else if (RUNNING_ON_VALGRIND && (errno == EINVAL)) {
            return UCS_ERR_NO_MEMORY;
        } else {
            ucs_error("shmat(shmid=%d) returned unexpected error: %m", *shmid);
            return UCS_ERR_SHMEM_SEGMENT;
        }
    }

    *address_p = ptr;
    *size      = alloc_size;

    ucs_memtrack_allocated(address_p, size UCS_MEMTRACK_VAL);
    return UCS_OK;
}

ucs_status_t ucs_sysv_free(void *address)
{
    int ret;

    ucs_memtrack_releasing(&address);
    ret = shmdt(address);
    if (ret) {
        ucs_warn("Unable to detach shared memory segment at %p: %m", address);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

typedef struct {
    unsigned long start;
    unsigned long end;
    int           prot;
    int           found;
} ucs_get_mem_prot_ctx_t;

static int ucs_get_mem_prot_cb(void *arg, void *addr, size_t length, int prot)
{
    ucs_get_mem_prot_ctx_t *ctx = arg;
    unsigned long seg_start = (uintptr_t)addr;
    unsigned long seg_end   = (uintptr_t)addr + length;

    if (ctx->start < seg_start) {
        ucs_trace("address 0x%lx is before next mapping 0x%lx..0x%lx", ctx->start,
                  seg_start, seg_end);
        return 1;
    } else if (ctx->start < seg_end) {
        ucs_trace("range 0x%lx..0x%lx overlaps with mapping 0x%lx..0x%lx prot 0x%x",
                  ctx->start, ctx->end, seg_start, seg_end, prot);

        if (!ctx->found) {
            /* first segment sets protection flags */
            ctx->prot  = prot;
            ctx->found = 1;
        } else {
            /* subsequent segments update protection flags */
            ctx->prot &= prot;
        }

        if (ctx->end <= seg_end) {
            /* finished going over entire memory region */
            return 1;
        }

        /* continue from the end of current segment */
        ctx->start = seg_end;
    }
    return 0;
}

int ucs_get_mem_prot(unsigned long start, unsigned long end)
{
    ucs_get_mem_prot_ctx_t ctx = { start, end, PROT_NONE, 0 };
    ucm_parse_proc_self_maps(ucs_get_mem_prot_cb, &ctx);
    return ctx.prot;
}

const char* ucs_get_process_cmdline()
{
    static char cmdline[1024] = {0};
    static int initialized = 0;
    ssize_t len;
    int i;

    if (!initialized) {
        len = ucs_read_file(cmdline, sizeof(cmdline), 1, "/proc/self/cmdline");
        for (i = 0; i < len; ++i) {
            if (cmdline[i] == '\0') {
                cmdline[i] = ' ';
            }
        }
        initialized = 1;
    }
    return cmdline;
}

ucs_status_t ucs_sys_fcntl_modfl(int fd, int add, int remove)
{
    int oldfl, ret;

    oldfl = fcntl(fd, F_GETFL);
    if (oldfl < 0) {
        ucs_error("fcntl(fd=%d, F_GETFL) returned %d: %m", fd, oldfl);
        return UCS_ERR_IO_ERROR;
    }

    ret = fcntl(fd, F_SETFL, (oldfl | add) & ~remove);
    if (ret < 0) {
        ucs_error("fcntl(fd=%d, F_SETFL) returned %d: %m", fd, ret);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

pid_t ucs_get_tid(void)
{
    return syscall(SYS_gettid);
}

int ucs_tgkill(int tgid, int tid, int sig)
{
    return syscall(SYS_tgkill, tgid, tid, sig);
}

double ucs_get_cpuinfo_clock_freq(const char *mhz_header)
{
    double mhz = 0.0;
    double m;
    int rc;
    FILE* f;
    char buf[256];
    char fmt[256];
    int warn;

    f = fopen("/proc/cpuinfo","r");
    if (!f) {
        return 0.0;
    }

    snprintf(fmt, sizeof(fmt), "%s : %%lf", mhz_header);

    warn = 0;
    while (fgets(buf, sizeof(buf), f)) {

        rc = sscanf(buf, fmt, &m);
        if (rc != 1) {
            continue;
        }

        if (mhz == 0.0) {
            mhz = m;
            continue;
        }

        if (mhz != m) {
            mhz = ucs_max(mhz,m);
            warn = 1;
        }
    }
    fclose(f);

    if (warn) {
        ucs_warn("Conflicting CPU frequencies detected, using: %.2f MHz", mhz);
    }
    return mhz * 1e6;
}

void ucs_empty_function()
{
}

ucs_status_t ucs_empty_function_return_success()
{
    return UCS_OK;
}

ucs_status_t ucs_empty_function_return_unsupported()
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t ucs_empty_function_return_inprogress()
{
    return UCS_INPROGRESS;
}

ucs_status_t ucs_empty_function_return_no_resource()
{
    return UCS_ERR_NO_RESOURCE;
}

ucs_status_ptr_t ucs_empty_function_return_ptr_no_resource()
{
    return UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
}

ucs_status_t ucs_empty_function_return_ep_timeout()
{
    return UCS_ERR_ENDPOINT_TIMEOUT;
}

ssize_t ucs_empty_function_return_bc_ep_timeout()
{
    return UCS_ERR_ENDPOINT_TIMEOUT;
}

ucs_status_t ucs_empty_function_return_busy()
{
    return UCS_ERR_BUSY;
}
