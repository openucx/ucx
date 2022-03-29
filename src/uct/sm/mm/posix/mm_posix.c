/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/sm/mm/base/mm_md.h>
#include <uct/sm/mm/base/mm_iface.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/debug/log.h>
#include <ucs/sys/string.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/sys.h>
#include <sys/mman.h>
#include <sys/statvfs.h>


/* File open flags */
#define UCT_POSIX_SHM_CREATE_FLAGS      (O_CREAT | O_EXCL | O_RDWR) /* shm create flags */
#define UCT_POSIX_SHM_OPEN_MODE         0600           /* shm open/create mode */

/* Memory mapping parameters */
#define UCT_POSIX_MMAP_PROT             (PROT_READ | PROT_WRITE)

/* Shared memory segment flags */
#define UCT_POSIX_SEG_FLAG_PROCFS       UCS_BIT(63) /* use procfs mode: mmid encodes an
                                                       open fd symlink from procfs */
#define UCT_POSIX_SEG_FLAG_SHM_OPEN     UCS_BIT(62) /* use shm_open() rather than open() */
#define UCT_POSIX_SEG_FLAG_HUGETLB      UCS_BIT(61) /* use MAP_HUGETLB */
#define UCT_POSIX_SEG_FLAG_PID_NS       UCS_BIT(60) /* use PID NS in address */
#define UCT_POSIX_SEG_FLAGS_MASK        (UCT_POSIX_SEG_FLAG_PROCFS | \
                                         UCT_POSIX_SEG_FLAG_SHM_OPEN | \
                                         UCT_POSIX_SEG_FLAG_PID_NS | \
                                         UCT_POSIX_SEG_FLAG_HUGETLB)
#define UCT_POSIX_SEG_MMID_MASK         (~UCT_POSIX_SEG_FLAGS_MASK)

/* Packing mmid for procfs mode */
#define UCT_POSIX_PROCFS_MMID_FD_BITS   30  /* how many bits for file descriptor */
#define UCT_POSIX_PROCFS_MMID_PID_BITS  30  /* how many bits for pid */

/* Filesystem paths */
#define UCT_POSIX_SHM_OPEN_DIR          "/dev/shm"       /* directory path for shm_open() */
#define UCT_POSIX_FILE_FMT              "/ucx_shm_posix_%"PRIx64
#define UCT_POSIX_PROCFS_FILE_FMT       "/proc/%d/fd/%d" /* file pattern for procfs mode */


typedef struct uct_posix_md_config {
    uct_mm_md_config_t        super;
    char                      *dir;
    int                       use_proc_link;
} uct_posix_md_config_t;

typedef struct uct_posix_packed_rkey {
    uint64_t                  seg_id;     /* flags + mmid */
    uintptr_t                 address;
    size_t                    length;
} UCS_S_PACKED uct_posix_packed_rkey_t;


static ucs_config_field_t uct_posix_md_config_table[] = {
  {"MM_", "", NULL,
   ucs_offsetof(uct_posix_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_mm_md_config_table)},

  {"DIR", UCT_POSIX_SHM_OPEN_DIR,
   "The path to the backing file. If it's equal to " UCT_POSIX_SHM_OPEN_DIR " then \n"
   "shm_open() is used. Otherwise, open() is used.",
   ucs_offsetof(uct_posix_md_config_t, dir), UCS_CONFIG_TYPE_STRING},

  {"USE_PROC_LINK", "y", "Use /proc/<pid>/fd/<fd> to share posix file.\n"
   " y   - Use /proc/<pid>/fd/<fd> to share posix file.\n"
   " n   - Use original file path to share posix file.\n",
   ucs_offsetof(uct_posix_md_config_t, use_proc_link), UCS_CONFIG_TYPE_BOOL},

  {NULL}
};

static ucs_config_field_t uct_posix_iface_config_table[] = {
  {"MM_", "", NULL, 0, UCS_CONFIG_TYPE_TABLE(uct_mm_iface_config_table)},

  {NULL}
};

static int uct_posix_use_shm_open(const uct_posix_md_config_t *posix_config)
{
    return !strcmp(posix_config->dir, UCT_POSIX_SHM_OPEN_DIR);
}

static ucs_status_t uct_posix_query(int *attach_shm_file_p)
{
    *attach_shm_file_p = 1;
    return UCS_OK;
}

static size_t uct_posix_iface_addr_length(uct_mm_md_t *md)
{
    const uct_posix_md_config_t *posix_config =
                    ucs_derived_of(md->config, uct_posix_md_config_t);

    /* if shm_open is requested, the path to the backing file is /dev/shm
     * by default. however, if shm_open isn't used, the size of the path to the
     * requested backing file is needed so that the user would know how much
     * space to allocate for the rkey.
     */
    if (posix_config->use_proc_link) {
        return ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID) ? 0 : sizeof(ucs_sys_ns_t);
    }

    return uct_posix_use_shm_open(posix_config) ?
           0 : (strlen(posix_config->dir) + 1);
}

static ucs_status_t uct_posix_md_query(uct_md_h tl_md, uct_md_attr_t *md_attr)
{
    uct_mm_md_t *md                           = ucs_derived_of(tl_md, uct_mm_md_t);
    const uct_posix_md_config_t *posix_config =
                    ucs_derived_of(md->config, uct_posix_md_config_t);
    struct statvfs shm_statvfs;

    if (statvfs(posix_config->dir, &shm_statvfs) < 0) {
        ucs_error("could not stat shared memory device %s (%m)",
                  UCT_POSIX_SHM_OPEN_DIR);
        return UCS_ERR_NO_DEVICE;
    }

    uct_mm_md_query(&md->super, md_attr,
                    shm_statvfs.f_bsize * shm_statvfs.f_bavail);

    md_attr->rkey_packed_size = sizeof(uct_posix_packed_rkey_t) +
                                uct_posix_iface_addr_length(md);
    return UCS_OK;
}

static uint64_t uct_posix_mmid_procfs_pack(int fd)
{
    pid_t pid = getpid();

    UCS_STATIC_ASSERT(UCS_MASK(UCT_POSIX_PROCFS_MMID_PID_BITS +
                               UCT_POSIX_PROCFS_MMID_FD_BITS) ==
                      UCT_POSIX_SEG_MMID_MASK);

    ucs_assert(pid <= UCS_MASK(UCT_POSIX_PROCFS_MMID_PID_BITS));
    ucs_assert(fd  <= UCS_MASK(UCT_POSIX_PROCFS_MMID_FD_BITS));
    return pid | ((uint64_t)fd << UCT_POSIX_PROCFS_MMID_PID_BITS);
}

static void uct_posix_mmid_procfs_unpack(uint64_t mmid, int *pid_p, int *fd_p)
{
    *fd_p  = mmid >> UCT_POSIX_PROCFS_MMID_PID_BITS;
    *pid_p = mmid & UCS_MASK(UCT_POSIX_PROCFS_MMID_PID_BITS);
}

static ucs_status_t uct_posix_test_mem(int shm_fd, size_t length)
{
    const size_t chunk_size = 64 * UCS_KBYTE;
    size_t size_to_write, remaining;
    ssize_t single_write;
    ucs_status_t status;
    int *buf;

    buf = ucs_malloc(chunk_size, "write buffer");
    if (buf == NULL) {
        ucs_error("Failed to allocate memory for testing space for backing file.");
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    memset(buf, 0, chunk_size);
    if (lseek(shm_fd, 0, SEEK_SET) < 0) {
        ucs_error("lseek failed. %m");
        status = UCS_ERR_IO_ERROR;
        goto out_free_buf;
    }

    remaining = length;
    while (remaining > 0) {
        size_to_write = ucs_min(remaining, chunk_size);
        single_write = write(shm_fd, buf, size_to_write);

        if (single_write < 0) {
            switch(errno) {
            case ENOSPC:
                ucs_error("Not enough memory to write total of %zu bytes. "
                          "Please check that /dev/shm or the directory you specified has "
                          "more available memory.", length);
                status = UCS_ERR_NO_MEMORY;
                break;
            default:
                ucs_error("Failed to write %zu bytes. %m", size_to_write);
                status = UCS_ERR_IO_ERROR;
            }
            goto out_free_buf;
        }

        remaining -= single_write;
    }

    status = UCS_OK;

out_free_buf:
    ucs_free(buf);
out:
    return status;
}

ucs_status_t uct_posix_open_check_result(const char *func, const char *file_name,
                                         int open_flags, int ret, int *fd_p)
{
    if (ret >= 0) {
        *fd_p = ret;
        return UCS_OK;
    } else if (errno == EEXIST) {
        return UCS_ERR_ALREADY_EXISTS;
    } else {
        ucs_error("%s(file_name=%s flags=0x%x) failed: %m", func, file_name,
                  open_flags);
        return UCS_ERR_SHMEM_SEGMENT;
    }
}

static ucs_status_t uct_posix_shm_open(uint64_t mmid, int open_flags, int *fd_p)
{
    char file_name[NAME_MAX];
    int ret;

    ucs_snprintf_safe(file_name, sizeof(file_name), UCT_POSIX_FILE_FMT, mmid);
    ret = shm_open(file_name, open_flags | O_RDWR, UCT_POSIX_SHM_OPEN_MODE);
    return uct_posix_open_check_result("shm_open", file_name, open_flags, ret,
                                       fd_p);
}

static ucs_status_t uct_posix_file_open(const char *dir, uint64_t mmid,
                                        int open_flags, int* fd_p)
{
    char file_path[PATH_MAX];
    int ret;

    ucs_snprintf_safe(file_path, sizeof(file_path), "%s" UCT_POSIX_FILE_FMT,
                      dir, mmid);
    ret = open(file_path, open_flags | O_RDWR, UCT_POSIX_SHM_OPEN_MODE);
    return uct_posix_open_check_result("open", file_path, open_flags, ret, fd_p);
}

static ucs_status_t uct_posix_procfs_open(int pid, int peer_fd, int* fd_p)
{
    char file_path[PATH_MAX];
    int ret;

    ucs_snprintf_safe(file_path, sizeof(file_path), UCT_POSIX_PROCFS_FILE_FMT,
                      pid, peer_fd);
    ret = open(file_path, O_RDWR, UCT_POSIX_SHM_OPEN_MODE);
    return uct_posix_open_check_result("open", file_path, 0, ret, fd_p);
}

static ucs_status_t uct_posix_unlink(uct_mm_md_t *md, uint64_t seg_id)
{
    uct_posix_md_config_t *posix_config = ucs_derived_of(md->config,
                                                         uct_posix_md_config_t);
    char file_path[PATH_MAX];
    int ret;

    if (seg_id & UCT_POSIX_SEG_FLAG_SHM_OPEN) {
        ucs_snprintf_safe(file_path, sizeof(file_path), UCT_POSIX_FILE_FMT,
                          seg_id & UCT_POSIX_SEG_MMID_MASK);
        ret = shm_unlink(file_path);
        if (ret < 0) {
            ucs_error("shm_unlink(%s) failed: %m", file_path);
            return UCS_ERR_SHMEM_SEGMENT;
        }
    } else {
        ucs_snprintf_safe(file_path, sizeof(file_path), "%s" UCT_POSIX_FILE_FMT,
                          posix_config->dir, seg_id & UCT_POSIX_SEG_MMID_MASK);
        ret = unlink(file_path);
        if (ret < 0) {
            ucs_error("unlink(%s) failed: %m", file_path);
            return UCS_ERR_SHMEM_SEGMENT;
        }
    }

    return UCS_OK;
}

static ucs_status_t
uct_posix_mmap(void **address_p, size_t *length_p, int flags, int fd,
               const char *alloc_name, ucs_log_level_t err_level)
{
    size_t aligned_length;
    void *result;

    aligned_length = ucs_align_up_pow2(*length_p, ucs_get_page_size());

#ifdef MAP_HUGETLB
    if (flags & MAP_HUGETLB) {
        ssize_t huge_page_size = ucs_get_huge_page_size();
        size_t huge_aligned_length;

        if (huge_page_size <= 0) {
            ucs_debug("huge pages are not supported on the system");
            return UCS_ERR_NO_MEMORY; /* Huge pages not supported */
        }

        huge_aligned_length = ucs_align_up_pow2(aligned_length, huge_page_size);
        if (huge_aligned_length > (2 * aligned_length)) {
            return UCS_ERR_EXCEEDS_LIMIT; /* Do not align up by more than 2x */
        }

        aligned_length = huge_aligned_length;
    }
#endif

    result = ucs_mmap(*address_p, aligned_length, UCT_POSIX_MMAP_PROT,
                      MAP_SHARED | flags, fd, 0, alloc_name);
    if (result == MAP_FAILED) {
        ucs_log(err_level,
                "shared memory mmap(addr=%p, length=%zu, flags=%s%s, fd=%d) failed: %m",
                *address_p, aligned_length,
                (flags & MAP_FIXED)   ? " FIXED"   : "",
#ifdef MAP_HUGETLB
                (flags & MAP_HUGETLB) ? " HUGETLB" : "",
#else
                "",
#endif
                fd);
        return UCS_ERR_SHMEM_SEGMENT;
    }

    *address_p = result;
    *length_p  = aligned_length;

    return UCS_OK;
}

static ucs_status_t uct_posix_munmap(void *address, size_t length)
{
    int ret;

    ret = ucs_munmap(address, length);
    if (ret != 0) {
        ucs_warn("shared memory munmap(address=%p, length=%zu) failed: %m",
                 address, length);
        return UCS_ERR_SHMEM_SEGMENT;
    }

    return UCS_OK;
}

static ucs_status_t
uct_posix_mem_attach_common(uct_mm_seg_id_t seg_id, size_t length,
                            const char *dir, uct_mm_remote_seg_t *rseg)
{
    uint64_t mmid = seg_id & UCT_POSIX_SEG_MMID_MASK;
    int pid, peer_fd, fd;
    ucs_status_t status;
    int mmap_flags;

    ucs_assert(length > 0);
    rseg->cookie = (void*)length;

    if (seg_id & UCT_POSIX_SEG_FLAG_PROCFS) {
        uct_posix_mmid_procfs_unpack(mmid, &pid, &peer_fd);
        status = uct_posix_procfs_open(pid, peer_fd, &fd);
    } else if (seg_id & UCT_POSIX_SEG_FLAG_SHM_OPEN) {
        status = uct_posix_shm_open(mmid, 0, &fd);
    } else {
        ucs_assert(dir != NULL); /* for coverity */
        status = uct_posix_file_open(dir, mmid, 0, &fd);
    }
    if (status != UCS_OK) {
        return status;
    }

#ifdef MAP_HUGETLB
    mmap_flags = (seg_id & UCT_POSIX_SEG_FLAG_HUGETLB) ? MAP_HUGETLB : 0;
#else
    mmap_flags = 0;
#endif
    rseg->address = NULL;
    status = uct_posix_mmap(&rseg->address, &length, mmap_flags, fd,
                            "posix_attach", UCS_LOG_LEVEL_ERROR);
    close(fd);
    return status;
}

static int
uct_posix_is_reachable(uct_mm_md_t *md, uct_mm_seg_id_t seg_id,
                       const void *iface_addr)
{
    if (seg_id & UCT_POSIX_SEG_FLAG_PID_NS) {
        return ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID) == *(const ucs_sys_ns_t*)iface_addr;
    }

    return ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID);
}

static ucs_status_t uct_posix_mem_detach_common(const uct_mm_remote_seg_t *rseg)
{
    return uct_posix_munmap(rseg->address, (size_t)rseg->cookie);
}

static ucs_status_t
uct_posix_segment_open(uct_mm_md_t *md, uct_mm_seg_id_t *seg_id_p, int *fd_p)
{
    uct_posix_md_config_t *posix_config = ucs_derived_of(md->config,
                                                         uct_posix_md_config_t);
    uint64_t mmid, flags;
    ucs_status_t status;
    unsigned rand_seed;

    /* Generate random 32-bit shared memory id and make sure it's not used
     * already by opening the file with O_CREAT|O_EXCL */
    rand_seed = ucs_generate_uuid((uintptr_t)md);
    for (;;) {
        mmid = rand_r(&rand_seed);
        ucs_assert(mmid <= UCT_POSIX_SEG_MMID_MASK);
        if (uct_posix_use_shm_open(posix_config)) {
            flags  = UCT_POSIX_SEG_FLAG_SHM_OPEN;
            status = uct_posix_shm_open(mmid, UCT_POSIX_SHM_CREATE_FLAGS, fd_p);
        } else {
            flags  = 0;
            status = uct_posix_file_open(posix_config->dir, mmid,
                                         UCT_POSIX_SHM_CREATE_FLAGS, fd_p);
        }
        if (status == UCS_OK) {
            *seg_id_p = mmid | flags;
            return UCS_OK; /* found unique file name */
        } else if (status != UCS_ERR_ALREADY_EXISTS) {
            return status; /* unexpected error (e.g permission denied) */
        }
        /* file exists, retry */
    }
}

static ucs_status_t
uct_posix_mem_alloc(uct_md_h tl_md, size_t *length_p, void **address_p,
                    ucs_memory_type_t mem_type, unsigned flags,
                    const char *alloc_name, uct_mem_h *memh_p)
{
    uct_mm_md_t                     *md = ucs_derived_of(tl_md, uct_mm_md_t);
    uct_posix_md_config_t *posix_config = ucs_derived_of(md->config,
                                                         uct_posix_md_config_t);
    ucs_status_t status;
    uct_mm_seg_t *seg;
    int force_hugetlb;
    int mmap_flags;
    void *address;
    int fd;

    if (mem_type != UCS_MEMORY_TYPE_HOST) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_mm_seg_new(*address_p, *length_p, &seg);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_posix_segment_open(md, &seg->seg_id, &fd);
    if (status != UCS_OK) {
        goto err_free_seg;
    }

    /* Check if the location of the backing file has enough memory for the
     * needed size by trying to write there before calling mmap */
    status = uct_posix_test_mem(fd, seg->length);
    if (status != UCS_OK) {
        goto err_close;
    }

    /* If using procfs link instead of mmid, remove the original file and update
     * seg->seg_id */
    if (posix_config->use_proc_link) {
        status = uct_posix_unlink(md, seg->seg_id);
        if (status != UCS_OK) {
            goto err_close;
        }

        /* Replace mmid by pid+fd. Keep previous SHM_OPEN flag for mkey_pack() */
        seg->seg_id = uct_posix_mmid_procfs_pack(fd) |
                      (seg->seg_id & UCT_POSIX_SEG_FLAG_SHM_OPEN) |
                      UCT_POSIX_SEG_FLAG_PROCFS |
                      (ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID) ? 0 :
                       UCT_POSIX_SEG_FLAG_PID_NS);
    }

    /* mmap the shared memory segment that was created by shm_open */
    if (flags & UCT_MD_MEM_FLAG_FIXED) {
        mmap_flags   = MAP_FIXED;
    } else {
        seg->address = NULL;
        mmap_flags   = 0;
    }

    /* try HUGETLB mmap */
    address = MAP_FAILED;
    if (posix_config->super.hugetlb_mode != UCS_NO) {
        force_hugetlb = (posix_config->super.hugetlb_mode == UCS_YES);
#ifdef MAP_HUGETLB
        status = uct_posix_mmap(&seg->address, &seg->length,
                                mmap_flags | MAP_HUGETLB, fd, alloc_name,
                                force_hugetlb ? UCS_LOG_LEVEL_ERROR :
                                                UCS_LOG_LEVEL_DEBUG);
#else
        status = UCS_ERR_SHMEM_SEGMENT;
        if (force_hugetlb) {
            ucs_error("shared memory allocation failed: "
                      "MAP_HUGETLB is not supported on the system");
        }
#endif
        if ((status != UCS_OK) && force_hugetlb) {
            goto err_close;
        } else if (status == UCS_OK) {
            seg->seg_id |= UCT_POSIX_SEG_FLAG_HUGETLB;
       }
    }

    /* fallback to regular mmap */
    if (address == MAP_FAILED) {
        ucs_assert(posix_config->super.hugetlb_mode != UCS_YES);
        status = uct_posix_mmap(&seg->address, &seg->length, mmap_flags, fd,
                                alloc_name, UCS_LOG_LEVEL_ERROR);
        if (status != UCS_OK) {
            goto err_close;
        }
    }

    /* create new memory segment */
    ucs_debug("allocated posix shared memory at %p length %zu", seg->address,
              seg->length);

    if (!posix_config->use_proc_link) {
        /* closing the file here since the peers will open it by file system path */
        close(fd);
    }

    *address_p = seg->address;
    *length_p  = seg->length;
     *memh_p   = seg;
    return UCS_OK;

err_close:
    close(fd);
    if (!(seg->seg_id & UCT_POSIX_SEG_FLAG_PROCFS)) {
        uct_posix_unlink(md, seg->seg_id);
    }
err_free_seg:
    ucs_free(seg);
err:
    return status;
}

static ucs_status_t uct_posix_mem_free(uct_md_h tl_md, uct_mem_h memh)
{
    uct_mm_md_t   *md = ucs_derived_of(tl_md, uct_mm_md_t);
    uct_mm_seg_t *seg = memh;
    ucs_status_t status;
    int fd, dummy_pid;

    status = uct_posix_munmap(seg->address, seg->length);
    if (status != UCS_OK) {
        return status;
    }

    if (seg->seg_id & UCT_POSIX_SEG_FLAG_PROCFS) {
        uct_posix_mmid_procfs_unpack(seg->seg_id & UCT_POSIX_SEG_MMID_MASK,
                                     &dummy_pid, &fd);
        ucs_assert(dummy_pid == getpid());
        close(fd);
    } else {
        status = uct_posix_unlink(md, seg->seg_id);
        if (status != UCS_OK) {
            return status;
        }
    }

    ucs_free(seg);
    return UCS_OK;
}

static void uct_posix_copy_dir(uct_mm_md_t *md, void *buffer)
{
    const uct_posix_md_config_t *posix_config =
                     ucs_derived_of(md->config, uct_posix_md_config_t);

    memcpy(buffer, posix_config->dir, strlen(posix_config->dir) + 1);
}

static ucs_status_t uct_posix_iface_addr_pack(uct_mm_md_t *md, void *buffer)
{
    const uct_posix_md_config_t *posix_config =
                     ucs_derived_of(md->config, uct_posix_md_config_t);

    if (posix_config->use_proc_link) {
        if (!ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID)) {
            *(ucs_sys_ns_t*)buffer = ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID);
        }
        return UCS_OK;
    }

    if (!uct_posix_use_shm_open(posix_config)) {
        uct_posix_copy_dir(md, buffer);
    }

    return UCS_OK;
}

static ucs_status_t
uct_posix_md_mkey_pack(uct_md_h tl_md, uct_mem_h memh,
                       const uct_md_mkey_pack_params_t *params,
                       void *rkey_buffer)
{
    uct_mm_md_t *md                      = ucs_derived_of(tl_md, uct_mm_md_t);
    uct_mm_seg_t *seg                    = memh;
    uct_posix_packed_rkey_t *packed_rkey = rkey_buffer;

    packed_rkey->seg_id  = seg->seg_id;
    packed_rkey->address = (uintptr_t)seg->address;
    packed_rkey->length  = seg->length;
    if (!(seg->seg_id & UCT_POSIX_SEG_FLAG_SHM_OPEN) &&
        !(seg->seg_id & UCT_POSIX_SEG_FLAG_PROCFS)) {
        uct_posix_copy_dir(md, packed_rkey + 1);
    }

    return UCS_OK;
}

static ucs_status_t uct_posix_mem_attach(uct_mm_md_t *md, uct_mm_seg_id_t seg_id,
                                         size_t length, const void *iface_addr,
                                         uct_mm_remote_seg_t *remote_seg)
{
    return uct_posix_mem_attach_common(seg_id, length, iface_addr, remote_seg);
}

static void uct_posix_mem_detach(uct_mm_md_t *md, const uct_mm_remote_seg_t *rseg)
{
    uct_posix_mem_detach_common(rseg);
}

UCS_PROFILE_FUNC(ucs_status_t, uct_posix_rkey_unpack,
                 (component, rkey_buffer, rkey_p, handle_p),
                 uct_component_t *component, const void *rkey_buffer,
                 uct_rkey_t *rkey_p, void **handle_p)
{
    const uct_posix_packed_rkey_t *packed_rkey = rkey_buffer;
    uct_mm_remote_seg_t *rseg;
    ucs_status_t status;

    rseg = ucs_malloc(sizeof(*rseg), "posix_remote_seg");
    if (rseg == NULL) {
        ucs_error("failed to allocate posix remote segment descriptor");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_posix_mem_attach_common(packed_rkey->seg_id,
                                         packed_rkey->length,
                                         (const char*)(packed_rkey + 1), rseg);
    if (status != UCS_OK) {
        ucs_free(rseg);
        return status;
    }

    uct_mm_md_make_rkey(rseg->address, packed_rkey->address, rkey_p);
    *handle_p = rseg;
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_posix_rkey_release,(component, rkey, handle),
                 uct_component_t *component, uct_rkey_t rkey, void *handle)
{
    uct_mm_remote_seg_t *rseg = handle;
    ucs_status_t status;

    status = uct_posix_mem_detach_common(rseg);
    if (status != UCS_OK) {
        return status;
    }

    ucs_free(rseg);
    return UCS_OK;
}

static uct_mm_md_mapper_ops_t uct_posix_md_ops = {
    .super = {
        .close                  = uct_mm_md_close,
        .query                  = uct_posix_md_query,
        .mem_alloc              = uct_posix_mem_alloc,
        .mem_free               = uct_posix_mem_free,
        .mem_advise             = ucs_empty_function_return_unsupported,
        .mem_reg                = ucs_empty_function_return_unsupported,
        .mem_dereg              = ucs_empty_function_return_unsupported,
        .mkey_pack              = uct_posix_md_mkey_pack,
        .is_sockaddr_accessible = ucs_empty_function_return_zero_int,
        .detect_memory_type     = ucs_empty_function_return_unsupported
    },
    .query             = uct_posix_query,
    .iface_addr_length = uct_posix_iface_addr_length,
    .iface_addr_pack   = uct_posix_iface_addr_pack,
    .mem_attach        = uct_posix_mem_attach,
    .mem_detach        = uct_posix_mem_detach,
    .is_reachable      = uct_posix_is_reachable
};

UCT_MM_TL_DEFINE(posix, &uct_posix_md_ops, uct_posix_rkey_unpack,
                 uct_posix_rkey_release, "POSIX_",
                 uct_posix_iface_config_table);

UCT_SINGLE_TL_INIT(&uct_posix_component.super, posix,,,)

