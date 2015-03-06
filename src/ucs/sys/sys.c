/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#define _GNU_SOURCE
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucs/debug/log.h>
#include <ucs/time/time.h>

#include <sys/ioctl.h>
#include <sys/shm.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <net/if.h>
#include <dirent.h>
#include <sched.h>

/* Default huge page size is 2 MBytes */
#define UCS_DEFAULT_HUGEPAGE_SIZE  (2 * 1024 * 1024)
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
        ucs_warn("failed to expand path '%s' (%s), using original path", path,
                 strerror(errno));
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
           ucs_get_prime(0) * getpid() +
           ucs_get_prime(1) * ucs_get_time() +
           ucs_get_prime(2) * ucs_get_mac_address() +
           ucs_get_prime(3) * tv.tv_sec +
           ucs_get_prime(4) * tv.tv_usec +
           __sumup_host_name(5);
}

void ucs_fill_filename_template(const char *tmpl, char *buf, size_t max)
{
    char *p, *end;
    const char *pf, *pp;
    size_t length;
    time_t t;

    p = buf;
    end = buf + max - 1;
    *end = 0;
    pf = tmpl;
    while (*pf != 0 && p < end) {
        pp = strchr(pf, '%');
        if (pp == NULL) {
            strncpy(p, pf, end - p);
            p = end;
            break;
        }

        length = ucs_min(pp - pf, end - p);
        strncpy(p, pf, length);
        p += length;

        switch (*(pp + 1)) {
        case 'p':
            snprintf(p, end - p, "%d", getpid());
            pf = pp + 2;
            p += strlen(p);
            break;
        case 'h':
            snprintf(p, end - p, "%s", ucs_get_host_name());
            pf = pp + 2;
            p += strlen(p);
            break;
        case 'c':
            snprintf(p, end - p, "%02d", ucs_get_first_cpu());
            pf = pp + 2;
            p += strlen(p);
            break;
        case 't':
            t = time(NULL);
            strftime(p, end - p, "%Y-%m-%d-%H:%M:%S", localtime(&t));
            pf = pp + 2;
            p += strlen(p);
            break;
        case 'u':
            snprintf(p, end - p, "%s", basename(ucs_get_user_name()));
            pf = pp + 2;
            p += strlen(p);
            break;
        case 'e':
            snprintf(p, end - p, "%s", basename(ucs_get_exe()));
            pf = pp + 2;
            p += strlen(p);
            break;
        default:
            *(p++) = *pp;
            pf = pp + 1;
            break;
        }

        p += strlen(p);
    }
    *p = 0;
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

uint64_t ucs_string_to_id(const char* str)
{
    uint64_t id = 0;
    strncpy((char*)&id, str, sizeof(id) - 1); /* Last character will be \0 */
    return id;
}

void ucs_snprintf_zero(char *buf, size_t size, const char *fmt, ...)
{
    va_list ap;

    memset(buf, 0, size);
    va_start(ap, fmt);
    vsnprintf(buf, size - 1, fmt, ap);
    va_end(ap);
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

size_t ucs_get_page_size()
{
    static size_t page_size = 0;

    if (page_size == 0) {
        page_size = sysconf(_SC_PAGESIZE);
    }
    return page_size;
}

size_t ucs_get_huge_page_size()
{
    static size_t huge_page_size = 0;
    char buf[256];
    int size_kb;
    FILE *f;

    /* Cache the huge page size value */
    if (huge_page_size == 0) {
        f = fopen("/proc/meminfo", "r");
        if (f != NULL) {
            while (fgets(buf, sizeof(buf), f)) {
                if (sscanf(buf, "Hugepagesize:       %d kB", &size_kb) == 1) {
                    huge_page_size = size_kb * 1024;
                    break;
                }
            }
            fclose(f);
        }

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

ucs_status_t ucs_sysv_alloc(size_t *size, void **address_p, int flags, int *shmid)
{
    void *ptr;
    //int ret;
    struct shminfo shminfo, *shminfo_ptr;

    if (RUNNING_ON_VALGRIND) {
        flags &= ~SHM_HUGETLB;
    }

    if (flags & SHM_HUGETLB){
        *size = ucs_align_up(*size, ucs_get_huge_page_size());
    } else {
        *size = ucs_align_up(*size, ucs_get_page_size());
    }

    flags |= IPC_CREAT | SHM_R | SHM_W;
    *shmid = shmget(IPC_PRIVATE, *size, flags);
    if (*shmid < 0) {
        switch (errno) {
        case ENFILE:
        case ENOMEM:
        case ENOSPC:
        case EPERM:
            if (!(flags & SHM_HUGETLB)) {
                shminfo_ptr = &shminfo;
                if ((shmctl(0, IPC_INFO, (struct shmid_ds *) shminfo_ptr)) > -1) {
                    ucs_error("shmget failed (size=%zu): The max number of shared memory segments in the system is = %ld. "
                    "Please try to increase this value through /proc/sys/kernel/shmmni",
                    *size, shminfo.shmmni);
                }
            }

            return UCS_ERR_NO_MEMORY;
            break;
        case EINVAL:
            ucs_error("A new segment was to be created and size < SHMMIN or size > SHMMAX, "
            "or no new segment was to be created. A segment with given key existed, "
            "but size is greater than the size of that segment. "
            "Please check shared memory limits by 'ipcs -l'.");
            return UCS_ERR_NO_MEMORY;
            break;
        default:
            ucs_error("shmget(size=%zu, flags=0x%x) returned unexpected error: %m. "
            "Please check shared memory limits by 'ipcs -l'.",
                      *size, flags);
            return UCS_ERR_SHMEM_SEGMENT;
        }
    }

    /* Attach segment */
    ptr = shmat(*shmid, NULL, 0);
    /* debug
    printf("sysv_alloc shmid(d) = %d\n", *shmid);
    printf("sysv_alloc ptr(p) = %p\n", (void *)ptr);
    */
    if ((void *)ptr == (void*)-1) {
        if (errno == ENOMEM) {
            return UCS_ERR_NO_MEMORY;
        } else {
            ucs_error("shmat(shmid=%d) returned unexpected error: %m", *shmid);
        }
    }


    /* Remove segment, the attachment keeps a reference to the mapping */
    /* FIXME having additional attaches to a removed segment is not portable
    * behavior */
    //ret = shmctl(*shmid, IPC_RMID, NULL);
    //if (ret != 0) {
    //    ucs_warn("shmctl(IPC_RMID, shmid=%d) returned %d: %m", *shmid, ret);
    //}

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
    return UCS_OK;
}

void ucs_sysv_free(void *address)
{
    int ret;

    ret = shmdt(address);

    if (ret) {
        ucs_warn("Unable to detach shared memory segment at %p: %m", address);
    }
}

unsigned ucs_get_mem_prot(void *address, size_t length)
{
    static int maps_fd = -1;
    char buffer[1024];
    unsigned long start_addr, end_addr;
    unsigned prot_flags;
    char read_c, write_c, exec_c, priv_c;
    char *ptr, *newline;
    ssize_t read_size;
    size_t read_offset;
    int ret;

    if (maps_fd == -1) {
        maps_fd = open(UCS_PROCESS_MAPS_FILE, O_RDONLY);
        if (maps_fd < 0) {
            ucs_fatal("cannot open %s for reading: %m", UCS_PROCESS_MAPS_FILE);
        }
    }

    ret = lseek(maps_fd, 0, SEEK_SET);
    if (ret < 0) {
        ucs_fatal("failed to lseek(0): %m");
    }

    prot_flags = PROT_READ|PROT_WRITE|PROT_EXEC;

    read_offset = 0;
    while (1) {
        read_size = read(maps_fd, buffer + read_offset, sizeof(buffer) - 1 - read_offset);
        if (read_size < 0) {
            if (errno == EINTR) {
                continue;
            } else {
                ucs_fatal("failed to read from %s: %m", UCS_PROCESS_MAPS_FILE);
            }
        } else if (read_size == 0) {
            goto out;
        } else {
            buffer[read_size + read_offset] = 0;
        }

        ptr = buffer;
        while ( (newline = strchr(ptr, '\n')) != NULL ) {
            /* 00400000-0040b000 r-xp ... \n */
            ret = sscanf(ptr, "%lx-%lx %c%c%c%c", &start_addr, &end_addr, &read_c,
                         &write_c, &exec_c, &priv_c);
            if (ret != 6) {
                ucs_fatal("Parse error at %s", ptr);
            }

            /* Address will not appear on the list */
            if ((uintptr_t)address < start_addr) {
                goto out;
            }

            /* Start address falls within current VMA */
            if ((uintptr_t)address < end_addr) {
                ucs_trace_data("existing mapping: start=0x%08lx end=0x%08lx prot=%u",
                               start_addr, end_addr, prot_flags);

                if (read_c != 'r') {
                    prot_flags &= ~PROT_READ;
                }
                if (write_c != 'w') {
                    prot_flags &= ~PROT_WRITE;
                }
                if (exec_c != 'x') {
                    prot_flags &= ~PROT_EXEC;
                }

                /* Finished going over entire memory region */
                if ((uintptr_t)(address + length) <= end_addr) {
                    return prot_flags;
                }

                /* Start from the end of current VMA */
                address = (void*)end_addr;
            }

            ptr = newline + 1;
        }

        read_offset = strlen(ptr);
        memmove(buffer, ptr, read_offset);
    }
out:
    return PROT_NONE;
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

ucs_cpu_model_t ucs_get_cpu_model()
{
#ifdef __x86_64__
    unsigned _eax, _ebx, _ecx, _edx;
    unsigned model, family;
    unsigned ext_model, ext_family;

    /* Get CPU model/family */
    ucs_cpuid(0x1, &_eax, &_ebx, &_ecx, &_edx);

    model      = (_eax >> 4)  & UCS_MASK(8  - 4 );
    family     = (_eax >> 8)  & UCS_MASK(12 - 8 );
    ext_model  = (_eax >> 16) & UCS_MASK(20 - 16);
    ext_family = (_eax >> 20) & UCS_MASK(28 - 20);

    /* Adjust family/model */
    if (family == 0xf) {
        family += ext_family;
    }
    if (family == 0x6 || family == 0xf) {
        model = (ext_model << 4) | model;
    }

    /* Check known CPUs */
    if (family == 0x06) {
       switch (model) {
       case 0x3a:
       case 0x3e:
           return UCS_CPU_MODEL_INTEL_IVYBRIDGE;
       case 0x2a:
       case 0x2d:
           return UCS_CPU_MODEL_INTEL_SANDYBRIDGE;
       case 0x1a:
       case 0x1e:
       case 0x1f:
       case 0x2e:
           return UCS_CPU_MODEL_INTEL_NEHALEM;
       case 0x25:
       case 0x2c:
       case 0x2f:
           return UCS_CPU_MODEL_INTEL_WESTMERE;
       }
    }

    return UCS_CPU_MODEL_UNKNOWN;

#else
    return UCS_CPU_MODEL_UNKNOWN;
#endif
}

pid_t ucs_get_tid(void)
{
    return syscall(SYS_gettid);
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
        ucs_warn("Conflicting CPU frequencies detected, using: %.2f", mhz);
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
