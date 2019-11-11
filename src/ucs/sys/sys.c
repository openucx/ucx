/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2014-2019. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif


#include <ucs/algorithm/crc.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
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
#include <ctype.h>
#ifdef HAVE_SYS_THR_H
#include <sys/thr.h>
#endif

#if HAVE_SYS_CAPABILITY_H
#  include <sys/capability.h>
#endif

/* Default huge page size is 2 MBytes */
#define UCS_DEFAULT_MEM_FREE       640000
#define UCS_PROCESS_SMAPS_FILE     "/proc/self/smaps"


const char *ucs_get_tmpdir()
{
    char *env_tmpdir;

    env_tmpdir = getenv("TMPDIR");
    if (env_tmpdir) {
        return env_tmpdir;
    } else {
        return "/tmp/";
    }
}

const char *ucs_get_host_name()
{
    static char hostname[HOST_NAME_MAX] = {0};

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
            crc = ucs_crc32(crc, buffer, nread);
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
        memcpy(&n, p, strnlen(p, sizeof(n)));
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

/*
 * If a certain system constant (name) is undefined on the underlying system the
 * sysconf routine returns -1.  ucs_sysconf return the negative value
 * a user and the user is responsible to define default value or abort.
 *
 * If an error occurs sysconf modified errno and ucs_sysconf aborts.
 *
 * Otherwise, a non-negative values is returned.
 */
static long ucs_sysconf(int name)
{
    long rc;
    errno = 0;

    rc = sysconf(name);
    ucs_assert_always(errno == 0);

    return rc;
}

int ucs_get_first_cpu()
{
    int first_cpu, total_cpus, ret;
    ucs_sys_cpuset_t mask;

    ret = ucs_sysconf(_SC_NPROCESSORS_CONF);
    if (ret < 0) {
        ucs_error("failed to get local cpu count: %m");
        return ret;
    }
    total_cpus = ret;

    CPU_ZERO(&mask);
    ret = ucs_sys_getaffinity(&mask);
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
ucs_open_output_stream(const char *config_str, ucs_log_level_t err_log_level,
                       FILE **p_fstream, int *p_need_close,
                       const char **p_next_token)
{
    FILE *output_stream;
    char filename[256];
    char *template;
    const char *p;
    size_t len;

    *p_next_token = config_str;

    len = strcspn(config_str, ":");
    if (!strncmp(config_str, "stdout", len)) {
        *p_fstream    = stdout;
        *p_need_close = 0;
        *p_next_token = config_str + len;
    } else if (!strncmp(config_str, "stderr", len)) {
        *p_fstream    = stderr;
        *p_need_close = 0;
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
            ucs_log(err_log_level, "failed to open '%s' for writing: %m",
                    filename);
            return UCS_ERR_IO_ERROR;
        }

        *p_fstream    = output_stream;
        *p_need_close = 1;
        *p_next_token = p + len;
    }

    return UCS_OK;
}

static ssize_t ucs_read_file_vararg(char *buffer, size_t max, int silent,
                                    const char *filename_fmt, va_list ap)
{
    char filename[MAXPATHLEN];
    ssize_t read_bytes;
    int fd;

    vsnprintf(filename, MAXPATHLEN, filename_fmt, ap);

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

ssize_t ucs_read_file(char *buffer, size_t max, int silent,
                      const char *filename_fmt, ...)
{
    ssize_t read_bytes;
    va_list ap;

    va_start(ap, filename_fmt);
    read_bytes = ucs_read_file_vararg(buffer, max, silent, filename_fmt, ap);
    va_end(ap);

    return read_bytes;
}

ucs_status_t ucs_read_file_number(long *value, int silent,
                                  const char *filename_fmt, ...)
{
    char buffer[64], *tail;
    ssize_t read_bytes;
    va_list ap;
    long n;

    va_start(ap, filename_fmt);
    read_bytes = ucs_read_file_vararg(buffer, sizeof(buffer) - 1, silent,
                                      filename_fmt, ap);
    va_end(ap);

    if (read_bytes < 0) {
        /* read error */
        return UCS_ERR_IO_ERROR;
    }

    n = strtol(buffer, &tail, 0);
    if ((*tail != '\0') && !isspace(*tail)) {
        /* parse error */
        return UCS_ERR_INVALID_PARAM;
    }

    *value = n;
    return UCS_OK;
}

ssize_t ucs_read_file_str(char *buffer, size_t max, int silent,
                          const char *filename_fmt, ...)
{
    size_t max_read = ucs_max(max, 1) - 1;
    ssize_t read_bytes;
    va_list ap;

    va_start(ap, filename_fmt);
    read_bytes = ucs_read_file_vararg(buffer, max_read, silent, filename_fmt, ap);
    va_end(ap);

    if ((read_bytes >= 0) && (max > 0)) {
        buffer[read_bytes] = '\0';
    }

    return read_bytes;
}

size_t ucs_get_page_size()
{
    static long page_size = 0;

    if (page_size == 0) {
        page_size = ucs_sysconf(_SC_PAGESIZE);
        if (page_size < 0) {
            page_size = 4096;
            ucs_debug("_SC_PAGESIZE is undefined, setting default value to %ld",
                      page_size);
        }
    }
    return page_size;
}

void ucs_get_mem_page_size(void *address, size_t size, size_t *min_page_size_p,
                           size_t *max_page_size_p)
{
    int found = 0;
    unsigned long start, end;
    unsigned long page_size_kb;
    size_t page_size;
    char buf[1024];
    FILE *file;
    int n;

    file = fopen(UCS_PROCESS_SMAPS_FILE, "r");
    if (!file) {
        goto out;
    }

    while (fgets(buf, sizeof(buf), file) != NULL) {
        n = sscanf(buf, "%lx-%lx", &start, &end);
        if (n != 2) {
            continue;
        }

        if (start > (uintptr_t)address + size) {
            /* the scanned range is after memory range of interest - stop */
            break;
        }
        if (end <= (uintptr_t)address) {
            /* the scanned range is still before the memory range of interest */
            continue;
        }

        while (fgets(buf, sizeof(buf), file) != NULL) {
            n = sscanf(buf, "KernelPageSize: %lu kB", &page_size_kb);
            if (n < 1) {
                continue;
            }

            page_size = page_size_kb * UCS_KBYTE;
            if (found) {
                *min_page_size_p = ucs_min(*min_page_size_p, page_size);
                *max_page_size_p = ucs_max(*max_page_size_p, page_size);
            } else {
                found            = 1;
                *min_page_size_p = page_size;
                *max_page_size_p = page_size;
            }
            break;
        }
    }

    fclose(file);

out:
    if (!found) {
        *min_page_size_p = *max_page_size_p = ucs_get_page_size();
    }
}

static ssize_t ucs_get_meminfo_entry(const char* pattern)
{
    char buf[256];
    char final_pattern[80];
    int val = 0;
    ssize_t val_b = -1;
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
    ssize_t mem_free;

    mem_free = ucs_get_meminfo_entry("MemFree");
    if (mem_free == -1) {
        mem_free = UCS_DEFAULT_MEM_FREE;
        ucs_info("cannot determine free mem size, using default: %zu",
                  mem_free);
    }

    return mem_free;
}

ssize_t ucs_get_huge_page_size()
{
    static ssize_t huge_page_size = 0;

    /* Cache the huge page size value */
    if (huge_page_size == 0) {
        huge_page_size = ucs_get_meminfo_entry("Hugepagesize");
        if (huge_page_size == -1) {
            ucs_debug("huge pages are not supported on the system");
        } else {
            ucs_trace("detected huge page size: %zu", huge_page_size);
        }
    }

    return huge_page_size;
}

size_t ucs_get_phys_mem_size()
{
    static size_t phys_mem_size = 0;
    long phys_pages;

    if (phys_mem_size == 0) {
        phys_pages = ucs_sysconf(_SC_PHYS_PAGES);
        if (phys_pages < 0) {
            ucs_debug("_SC_PHYS_PAGES is undefined, setting default value to %ld",
                      SIZE_MAX);
            phys_mem_size = SIZE_MAX;
        } else {
            phys_mem_size = phys_pages * ucs_get_page_size();
        }
    }
    return phys_mem_size;
}

#define UCS_SYS_THP_ENABLED_FILE "/sys/kernel/mm/transparent_hugepage/enabled"
int ucs_is_thp_enabled()
{
    char buf[256];
    int rc;

    rc = ucs_read_file(buf, sizeof(buf) - 1, 1, UCS_SYS_THP_ENABLED_FILE);
    if (rc < 0) {
        ucs_debug("failed to read %s:%m", UCS_SYS_THP_ENABLED_FILE);
        return 0;
    }

    buf[rc] = 0;
    return (strstr(buf, "[never]") == NULL);
}

#define UCS_PROC_SYS_SHMMAX_FILE "/proc/sys/kernel/shmmax"
size_t ucs_get_shmmax()
{
    ucs_status_t status;
    long size;

    status = ucs_read_file_number(&size, 0, UCS_PROC_SYS_SHMMAX_FILE);
    if (status != UCS_OK) {
        ucs_warn("failed to read %s:%m", UCS_PROC_SYS_SHMMAX_FILE);
        return 0;
    }

    return size;
}

static void ucs_sysv_shmget_error_check_ENOSPC(size_t alloc_size,
                                               const struct shminfo *ipc_info,
                                               char *buf, size_t max)
{
    unsigned long new_used_ids;
    unsigned long new_shm_tot;
    struct shm_info shm_info;
    char *p, *endp;
    int ret;

    p    = buf;
    endp = p + max;

    ret = shmctl(0, SHM_INFO, (struct shmid_ds *)&shm_info);
    if (ret >= 0) {
        return;
    }

    new_used_ids = shm_info.used_ids;
    if (new_used_ids > ipc_info->shmmni) {
        snprintf(p, endp - p,
                 ", total number of segments in the system (%lu) would exceed the"
                 " limit in /proc/sys/kernel/shmmni (=%lu)",
                 new_used_ids, ipc_info->shmmni);
        p += strlen(p);
    }

    new_shm_tot = shm_info.shm_tot +
                  (alloc_size + ucs_get_page_size() - 1) / ucs_get_page_size();
    if (new_shm_tot > ipc_info->shmall) {
        snprintf(p, endp - p,
                 ", total shared memory pages in the system (%lu) would exceed the"
                 " limit in /proc/sys/kernel/shmall (=%lu)",
                 new_shm_tot, ipc_info->shmall);
    }
}

ucs_status_t ucs_sys_get_proc_cap(uint32_t *effective)
{
#if HAVE_SYS_CAPABILITY_H
    cap_user_header_t hdr = ucs_alloca(sizeof(*hdr));
    cap_user_data_t data  = ucs_alloca(sizeof(*data) * _LINUX_CAPABILITY_U32S_3);
    int ret;

    hdr->pid     = 0; /* current thread */
    hdr->version = _LINUX_CAPABILITY_VERSION_3;
    ret = capget(hdr, data);
    if (ret) {
        ucs_debug("capget(pid=%d version=0x%x) failed: %m", hdr->pid,
                  hdr->version);
        return UCS_ERR_IO_ERROR;

    }

    *effective = data->effective;
    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static void ucs_sysv_shmget_error_check_EPERM(int flags, char *buf, size_t max)
{
#if HAVE_SYS_CAPABILITY_H
    ucs_status_t status;
    uint32_t ecap;

    UCS_STATIC_ASSERT(CAP_IPC_LOCK < 32);
    status = ucs_sys_get_proc_cap(&ecap);
    if ((status == UCS_OK) && !(ecap & UCS_BIT(CAP_IPC_LOCK))) {
        /* detected missing CAP_IPC_LOCK */
        snprintf(buf, max, ", CAP_IPC_LOCK privilege is needed for SHM_HUGETLB");
        return;
    }
#endif

    snprintf(buf, max,
             ", please check for CAP_IPC_LOCK privilege for using SHM_HUGETLB");
}

static void ucs_sysv_shmget_format_error(size_t alloc_size, int flags,
                                         const char *alloc_name, int sys_errno,
                                         char *buf, size_t max)
{
    struct shminfo ipc_info;
    char *p, *endp, *errp;
    int ret;

    buf[0] = '\0';
    p      = buf;
    endp   = p + max;

    snprintf(p, endp - p, "shmget(size=%zu flags=0x%x) for %s failed: %s",
             alloc_size, flags, alloc_name, strerror(sys_errno));
    p   += strlen(p);
    errp = p; /* save current string pointer to detect if anything was added */

    ret = shmctl(0, IPC_INFO, (struct shmid_ds *)&ipc_info);
    if (ret >= 0) {
        if ((sys_errno == EINVAL) && (alloc_size > ipc_info.shmmax)) {
            snprintf(p, endp - p,
                     ", allocation size exceeds /proc/sys/kernel/shmmax (=%zu)",
                     ipc_info.shmmax);
            p += strlen(p);
        }

        if (sys_errno == ENOSPC) {
            ucs_sysv_shmget_error_check_ENOSPC(alloc_size, &ipc_info, p, endp - p);
            p += strlen(p);
        }
    }

    if (sys_errno == EPERM) {
        ucs_sysv_shmget_error_check_EPERM(flags, p, endp - p);
        p += strlen(p);
    }

    /* default error message if no useful information was added to the string */
    if (p == errp) {
        snprintf(p, endp - p, ", please check shared memory limits by 'ipcs -l'");
    }
}

ucs_status_t ucs_sysv_alloc(size_t *size, size_t max_size, void **address_p,
                            int flags, const char *alloc_name, int *shmid)
{
    char error_string[256];
#ifdef SHM_HUGETLB
    ssize_t huge_page_size;
#endif
    size_t alloc_size;
    int sys_errno;
    void *ptr;
    int ret;

#ifdef SHM_HUGETLB
    if (flags & SHM_HUGETLB) {
        huge_page_size = ucs_get_huge_page_size();
        if (huge_page_size <= 0) {
            ucs_debug("huge pages are not supported on the system");
            return UCS_ERR_NO_MEMORY; /* Huge pages not supported */
        }
    }
    if (flags & SHM_HUGETLB) {
        alloc_size = ucs_align_up(*size, huge_page_size);
    } else
#endif
    {
        alloc_size = ucs_align_up(*size, ucs_get_page_size());
    }

    if (alloc_size >= max_size) {
        return UCS_ERR_EXCEEDS_LIMIT;
    }

    flags |= IPC_CREAT | SHM_R | SHM_W;
    *shmid = shmget(IPC_PRIVATE, alloc_size, flags);
    if (*shmid < 0) {
        sys_errno = errno;
        ucs_sysv_shmget_format_error(alloc_size, flags, alloc_name, sys_errno,
                                     error_string, sizeof(error_string));
        switch (sys_errno) {
        case ENOMEM:
        case EPERM:
#ifdef SHM_HUGETLB
            if (!(flags & SHM_HUGETLB))
#endif
	    {
                ucs_error("%s", error_string);
            }
            return UCS_ERR_NO_MEMORY;
        case ENOSPC:
        case EINVAL:
            ucs_error("%s", error_string);
            return UCS_ERR_NO_MEMORY;
        default:
            ucs_error("%s", error_string);
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

    ucs_memtrack_allocated(ptr, alloc_size UCS_MEMTRACK_VAL);
    *address_p = ptr;
    *size      = alloc_size;
    return UCS_OK;
}

ucs_status_t ucs_sysv_free(void *address)
{
    int ret;

    ucs_memtrack_releasing(address);
    ret = shmdt(address);
    if (ret) {
        ucs_warn("Unable to detach shared memory segment at %p: %m", address);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

ucs_status_t ucs_mmap_alloc(size_t *size, void **address_p,
                            int flags UCS_MEMTRACK_ARG)
{
    size_t alloc_length;
    void *addr;

    alloc_length = ucs_align_up_pow2(*size, ucs_get_page_size());

    addr = ucs_mmap(*address_p, alloc_length, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANON | flags, -1, 0 UCS_MEMTRACK_VAL);
    if (addr == MAP_FAILED) {
        return UCS_ERR_NO_MEMORY;
    }

    *size      = alloc_length;
    *address_p = addr;
    return UCS_OK;
}

ucs_status_t ucs_mmap_free(void *address, size_t length)
{
    int ret;
    size_t alloc_length;

    alloc_length = ucs_align_up_pow2(length, ucs_get_page_size());

    ret = ucs_munmap(address, alloc_length);
    if (ret != 0) {
        ucs_warn("munmap(address=%p, length=%zu) failed: %m", address, length);
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

static int ucs_get_mem_prot_cb(void *arg, void *addr, size_t length, int prot,
                               const char *path)
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

unsigned long ucs_sys_get_pfn(uintptr_t address)
{
    static const char *pagemap_file = "/proc/self/pagemap";
    static int initialized = 0;
    static int pagemap_fd;
    uint64_t data;
    off_t offset;
    ssize_t ret;

    if (!initialized) {
        pagemap_fd = open(pagemap_file, O_RDONLY);
        if (pagemap_fd < 0) {
            ucs_warn("failed to open %s: %m", pagemap_file);
        }
        initialized = 1;
    }

    if (pagemap_fd < 0) {
        return 0; /* could not open file */
    }

    offset = (address / ucs_get_page_size()) * sizeof(data);
    data   = 0;
    ret    = pread(pagemap_fd, &data, sizeof(data), offset);
    if (ret < 0) {
        ucs_warn("pread(file=%s offset=%zu) failed: %m", pagemap_file, offset);
        return 0;
    }

    if (!(data & UCS_BIT(63))) {
        ucs_trace("address 0x%lx not present", address);
        return 0;
    }

    return data & UCS_MASK(55);
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
#ifdef SYS_gettid
    return syscall(SYS_gettid);
#elif defined(HAVE_SYS_THR_H)
    long id;

    thr_self(&id);
    return (id);
#else
#error "Port me"
#endif
}

int ucs_tgkill(int tgid, int tid, int sig)
{
#ifdef SYS_tgkill
    return syscall(SYS_tgkill, tgid, tid, sig);
#elif defined(HAVE_SYS_THR_H)
    return (thr_kill2(tgid, tid, sig));
#else
#error "Port me"
#endif
}

double ucs_get_cpuinfo_clock_freq(const char *header, double scale)
{
    double value = 0.0;
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

    snprintf(fmt, sizeof(fmt), "%s : %%lf ", header);

    warn = 0;
    while (fgets(buf, sizeof(buf), f)) {

        rc = sscanf(buf, fmt, &m);
        if (rc != 1) {
            continue;
        }

        if (value == 0.0) {
            value = m;
            continue;
        }

        if (value != m) {
            value = ucs_max(value,m);
            warn = 1;
        }
    }
    fclose(f);

    if (warn) {
        ucs_debug("Conflicting CPU frequencies detected, using: %.2f", value);
    }

    return value * scale;
}

void *ucs_sys_realloc(void *old_ptr, size_t old_length, size_t new_length)
{
    void *ptr;

    new_length = ucs_align_up_pow2(new_length, ucs_get_page_size());
    if (old_ptr == NULL) {
        /* Note: Must pass the 0 offset as "long", otherwise it will be
         * partially undefined when converted to syscall arguments */
        ptr = (void*)syscall(__NR_mmap, NULL, new_length, PROT_READ|PROT_WRITE,
                             MAP_PRIVATE|MAP_ANONYMOUS, -1, 0ul);
        if (ptr == MAP_FAILED) {
            ucs_log_fatal_error("mmap(NULL, %zu, READ|WRITE, PRIVATE|ANON) failed: %m",
                                new_length);
            return NULL;
        }
    } else {
        old_length = ucs_align_up_pow2(old_length, ucs_get_page_size());
        ptr = (void*)syscall(__NR_mremap, old_ptr, old_length, new_length,
                             MREMAP_MAYMOVE);
        if (ptr == MAP_FAILED) {
            ucs_log_fatal_error("mremap(%p, %zu, %zu, MAYMOVE) failed: %m",
                                old_ptr, old_length, new_length);
            return NULL;
        }
    }

    return ptr;
}

void ucs_sys_free(void *ptr, size_t length)
{
    int ret;

    if (ptr != NULL) {
        length = ucs_align_up_pow2(length, ucs_get_page_size());
        ret = syscall(__NR_munmap, ptr, length);
        if (ret) {
            ucs_log_fatal_error("munmap(%p, %zu) failed: %m", ptr, length);
        }
    }
}

char* ucs_make_affinity_str(const ucs_sys_cpuset_t *cpuset, char *str, size_t len)
{
    int i = 0, prev = -1;
    char *p = str;

    for (i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, cpuset)) {
            if (prev < 0) {
                prev = i;
            }
        } else {
            if (prev > 0) {
                if (prev == i - 1) {
                    p += snprintf(p, str + len - p, "%d,", prev);
                } else {
                    p += snprintf(p, str + len - p, "%d-%d,", prev, i - 1);
                }
            }
            if (p > str + len) {
                p = str + len - 4;
                while (*p != ',') {
                    p--;
                }
                sprintf(p, "...");
                return str;
            }
            prev = -1;
        }
    }

    *(--p) = 0;
    return str;
}

int ucs_sys_setaffinity(ucs_sys_cpuset_t *cpuset)
{
    int ret;

#if defined(HAVE_SCHED_SETAFFINITY)
    ret = sched_setaffinity(0, sizeof(*cpuset), cpuset);
#elif defined(HAVE_CPUSET_SETAFFINITY)
    ret = cpuset_setaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, getpid(),
                             sizeof(*cpuset), cpuset);
#else
#error "Port me"
#endif
    return ret;
}

int ucs_sys_getaffinity(ucs_sys_cpuset_t *cpuset)
{
    int ret;

#if defined(HAVE_SCHED_GETAFFINITY)
    ret = sched_getaffinity(0, sizeof(*cpuset), cpuset);
#elif defined(HAVE_CPUSET_GETAFFINITY)
    ret = cpuset_getaffinity(CPU_LEVEL_WHICH, CPU_WHICH_PID, getpid(),
                             sizeof(*cpuset), cpuset);
#else
#error "Port me"
#endif
    return ret;
}

void ucs_sys_cpuset_copy(ucs_cpu_set_t *dst, const ucs_sys_cpuset_t *src)
{
    int c;

    UCS_CPU_ZERO(dst);
    for (c = 0; c < UCS_CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, src)) {
            UCS_CPU_SET(c, dst);
        }
    }
}
