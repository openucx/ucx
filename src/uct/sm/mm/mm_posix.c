/**
 * Copyright (c) UT-Battelle, LLC. 2014-2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "mm_pd.h"
#include "mm_iface.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <sys/mman.h>
#include <ucs/sys/sys.h>

#define UCT_MM_POSIX_SHM_OPEN_MODE  (0666)
#define UCT_MM_POSIX_MMAP_PROT      (PROT_READ | PROT_WRITE)
#define UCT_MM_POSIX_HUGETLB        UCS_BIT(0)
#define UCT_MM_POSIX_SHM_OPEN       UCS_BIT(1)
#define UCT_MM_POSIX_CTRL_BITS      2

typedef struct uct_posix_pd_config {
    uct_mm_pd_config_t      super;
    char                    *path;
    ucs_ternary_value_t     use_shm_open;
} uct_posix_pd_config_t;

static ucs_config_field_t uct_posix_pd_config_table[] = {
  {"MM_", "", NULL,
   ucs_offsetof(uct_posix_pd_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_mm_pd_config_table)},

  {"USE_SHM_OPEN", "try", "Use shm_open() for opening a file for memory mapping. "
   "Possible values are:\n"
   " y   - Use only shm_open() to open a backing file.\n"
   " n   - Use only open() to open a backing file.\n"
   " try - Try to use shm_open() and if it fails, use open().\n"
   "If shm_open() is used, the path to the file defaults to /dev/shm.\n"
   "If open() is used, the path to the file is specified in the parameter bellow (DIR).",
   ucs_offsetof(uct_posix_pd_config_t, use_shm_open), UCS_CONFIG_TYPE_TERNARY},

  {"DIR", "/tmp", "The path to the backing file in case open() is used.",
   ucs_offsetof(uct_posix_pd_config_t, path), UCS_CONFIG_TYPE_STRING},

  {NULL}
};

static ucs_status_t uct_posix_test_mem(size_t length, int shm_fd)
{
    int *buf;
    int chunk_size = 256 * 1024;
    ucs_status_t status = UCS_OK;
    size_t size_to_write, remaining;
    ssize_t single_write;

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
            ucs_error("Failed to write %zu bytes. %m", size_to_write);
            status = UCS_ERR_IO_ERROR;
            goto out_free_buf;
        }
        if (single_write != size_to_write) {
            ucs_error("Not enough memory to write total of %zu bytes. "
                      "Please check that /dev/shm or the directory you specified has "
                      "more available memory.", length);
            status = UCS_ERR_NO_MEMORY;
            goto out_free_buf;
        }
        remaining -= size_to_write;
    }

out_free_buf:
    ucs_free(buf);

out:
    return status;
}

static size_t uct_posix_get_path_size(uct_pd_h pd)
{
    uct_mm_pd_t *mm_pd = ucs_derived_of(pd, uct_mm_pd_t);
    uct_posix_pd_config_t *posix_config = ucs_derived_of(mm_pd->config,
                                                         uct_posix_pd_config_t);

    /* if shm_open is requested, the path to the backing file is /dev/shm
     * by default. however, if shm_open isn't used, in case UCS_NO was set for
     * use_shm_open or if UCS_TRY was set but using shm_open() was unsuccessful,
     * the size of the path to the requested backing file is needed so that the
     * user would know how much space to allocated for the rkey. */
    if (posix_config->use_shm_open == UCS_YES) {
        return 0;
    } else {
        return 1 + strlen(posix_config->path);
    }
}

static ucs_status_t uct_posix_set_path(char *file_name, int use_shm_open,
                                       const char *path, uint64_t uuid)
{
    ucs_status_t status;
    int ret, len;

    if (!use_shm_open) {
        strncpy(file_name, path, NAME_MAX);
    }

    len = strlen(file_name);
    ret = snprintf(file_name + len, NAME_MAX - len,
                   "/ucx_posix_mm_%s_%s_%016lx", ucs_get_user_name(),
                   ucs_get_host_name(), uuid);
    if ((ret >= (NAME_MAX - len)) || (ret < 1)) {
        status = UCS_ERR_INVALID_PARAM;
        return status;
    }

    return UCS_OK;
}

static ucs_status_t uct_posix_shm_open(const char *file_name, size_t length, int *shm_fd)
{
    ucs_status_t status;

    /* Create shared memory object and set its size */
    *shm_fd = shm_open(file_name, O_CREAT | O_RDWR | O_EXCL,
                       UCT_MM_POSIX_SHM_OPEN_MODE);
    if (*shm_fd == -1) {
        ucs_error("Error returned from shm_open %m. File name is: %s",
                  file_name);
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err;
    }
    if (ftruncate(*shm_fd, length) == -1) {
        ucs_error("Error returned from ftruncate %m");
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err_shm_unlink;
    }

    return UCS_OK;

err_shm_unlink:
    close(*shm_fd);
    if (shm_unlink(file_name) != 0) {
        ucs_warn("unable to shm_unlink the shared memory segment");
    }
err:
    return status;
}

static ucs_status_t uct_posix_open(const char *file_name, size_t length, int *shm_fd)
{
    ucs_status_t status;

    /* use open with the given path */
    *shm_fd = open(file_name, O_CREAT | O_RDWR | O_EXCL, UCT_MM_POSIX_SHM_OPEN_MODE);
    if (*shm_fd == -1) {
        ucs_error("Error returned from open %m . File name is: %s", file_name);
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err;
    }

    if (ftruncate(*shm_fd, length) == -1) {
        ucs_error("Error returned from ftruncate %m");
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err_close;
    }

    return UCS_OK;

err_close:
    close(*shm_fd);
    if (unlink(file_name) != 0) {
        ucs_warn("unable to unlink the shared memory segment");
    }
err:
    return status;
}

static ucs_status_t
uct_posix_open_backing_file(char *file_name, uint64_t *uuid, uct_posix_pd_config_t *config,
                            size_t length, int *shm_fd, const char **path_p)
{
    ucs_status_t status;

    if (config->use_shm_open != UCS_NO) {
        status = uct_posix_set_path(file_name, 1, NULL, *uuid >> UCT_MM_POSIX_CTRL_BITS);
        if (status != UCS_OK) {
            goto out;
        }

        status = uct_posix_shm_open(file_name, length, shm_fd);
        if ((config->use_shm_open == UCS_TRY) && (status != UCS_OK)) {
            goto use_open;
        } else {
            *uuid |= UCT_MM_POSIX_SHM_OPEN;
            goto out;
        }
    }

use_open:
    status = uct_posix_set_path(file_name, 0, config->path, *uuid >> UCT_MM_POSIX_CTRL_BITS);
    if (status != UCS_OK) {
        goto out;
    }

    status = uct_posix_open(file_name, length, shm_fd);
    if (status != UCS_OK) {
        return status;
    }

    *uuid &= ~UCT_MM_POSIX_SHM_OPEN;
    *path_p = config->path;
out:
    return status;
}

static ucs_status_t
uct_posix_alloc(uct_pd_h pd, size_t *length_p, ucs_ternary_value_t hugetlb,
                void **address_p, uct_mm_id_t *mmid_p, const char **path_p
                UCS_MEMTRACK_ARG)
{
    ucs_status_t status;
    int shm_fd = -1;
    uint64_t uuid;
    char *file_name;
    uct_mm_pd_t *mm_pd = ucs_derived_of(pd, uct_mm_pd_t);
    uct_posix_pd_config_t *posix_config = ucs_derived_of(mm_pd->config,
                                                         uct_posix_pd_config_t);

    if (0 == *length_p) {
        ucs_error("Unexpected length %zu", *length_p);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    file_name = ucs_calloc(1, NAME_MAX, "shared mr posix");
    if (file_name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        ucs_error("Failed to allocate memory for the shm_open file name. %m");
        goto err;
    }

    /* Generate a 64 bit uuid.
     * use 62 bits of it for creating the file_name of the backing file.
     * other 2 bits:
     * 1 bit is for indicating whether or not hugepages were used.
     * 1 bit is for indicating whether or not shm_open() was used. */
    uuid = ucs_generate_uuid(0);

    status = uct_posix_open_backing_file(file_name, &uuid, posix_config,
                                         *length_p, &shm_fd, path_p);
    if (status != UCS_OK) {
        goto err_free_file;
    }

    /* check is the location of the backing file has enough memory for the needed size
     * by trying to write there before calling mmap */
    status = uct_posix_test_mem(*length_p, shm_fd);
    if (status != UCS_OK) {
        goto err_shm_unlink;
    }

    status = UCS_ERR_NO_MEMORY;

    /* mmap the shared memory segment that was created by shm_open */
#ifdef MAP_HUGETLB
    if (hugetlb != UCS_NO) {
       (*address_p) = ucs_mmap(NULL, *length_p, UCT_MM_POSIX_MMAP_PROT,
                               MAP_SHARED | MAP_HUGETLB,
                               shm_fd, 0 UCS_MEMTRACK_VAL);
       if ((*address_p) !=  MAP_FAILED) {
           /* indicate that the memory was mapped with hugepages */
           uuid |= UCT_MM_POSIX_HUGETLB;
           goto out_ok;
       }

       ucs_debug("mm failed to allocate %zu bytes with hugetlb %m", *length_p);
    }

#else
    if (hugetlb == UCS_YES) {
        ucs_error("Hugepages were requested but they cannot be used with posix mmap.");
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err_shm_unlink;
    }
#endif

    if (hugetlb != UCS_YES) {
       (*address_p) = ucs_mmap(NULL, *length_p, UCT_MM_POSIX_MMAP_PROT,
                               MAP_SHARED, shm_fd, 0 UCS_MEMTRACK_VAL);
       if ((*address_p) != MAP_FAILED) {
           /* indicate that the memory was mapped without hugepages */
           uuid &= ~UCT_MM_POSIX_HUGETLB;
           goto out_ok;
       }

       ucs_debug("mm failed to allocate %zu bytes without hugetlb %m", *length_p);
    }

err_shm_unlink:
    close(shm_fd);
    if (shm_unlink(file_name) != 0) {
        ucs_warn("unable to unlink the shared memory segment");
    }
err_free_file:
    ucs_free(file_name);
err:
    return status;

out_ok:
    ucs_free(file_name);
    /* closing the shm_fd here won't unmap the mem region*/
    close(shm_fd);
    *mmid_p = uuid;
    return UCS_OK;
}

static ucs_status_t uct_posix_attach(uct_mm_id_t mmid, size_t length,
                                     void *remote_address,
                                     void **local_address,
                                     uint64_t *cookie, const char *path)
{
    void *ptr;
    char *file_name;
    int shm_fd;
    ucs_status_t status = UCS_OK;

    file_name = ucs_calloc(1, NAME_MAX, "shared mr posix");
    if (file_name == NULL) {
        ucs_error("Failed to allocate memory for file_name to attach. %m");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = uct_posix_set_path(file_name, mmid & UCT_MM_POSIX_SHM_OPEN, path,
                                mmid >> UCT_MM_POSIX_CTRL_BITS);
    if (status != UCS_OK) {
        goto err_free_file;
    }

    /* use the mmid (62 bits) to recreate the file_name for opening */
    if (mmid & UCT_MM_POSIX_SHM_OPEN) {
        shm_fd = shm_open(file_name, O_RDWR | O_EXCL, UCT_MM_POSIX_SHM_OPEN_MODE);
    } else {
        shm_fd = open(file_name, O_CREAT | O_RDWR, UCT_MM_POSIX_SHM_OPEN_MODE);
    }

    if (shm_fd == -1) {
        ucs_error("Error returned from shm_open/open in attach. %m. File name is: %s",
                  file_name);
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err_free_file;
    }

#ifdef MAP_HUGETLB
    if (mmid & UCT_MM_POSIX_HUGETLB) {
        ptr = ucs_mmap(NULL ,length, UCT_MM_POSIX_MMAP_PROT,
                       MAP_SHARED | MAP_HUGETLB,
                       shm_fd, 0 UCS_MEMTRACK_NAME("posix mmap attach"));
    } else
#endif
    {
        ptr = ucs_mmap(NULL ,length, UCT_MM_POSIX_MMAP_PROT, MAP_SHARED,
                       shm_fd, 0 UCS_MEMTRACK_NAME("posix mmap attach"));
    }
    if (ptr == MAP_FAILED) {
        ucs_error("ucs_mmap(shm_fd=%d) failed: %m", (int)shm_fd);
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err_close_fd;
    }

    *local_address = ptr;
    *cookie = 0xdeadbeef;

err_close_fd:
    /* closing the fd here won't unmap the mem region (if ucs_mmap was successful) */
    close(shm_fd);
err_free_file:
    ucs_free(file_name);
err:
    return status;
}

static ucs_status_t uct_posix_detach(uct_mm_remote_seg_t *mm_desc)
{
    int ret;

    ucs_memtrack_releasing(&mm_desc->address);
    ret = ucs_munmap(mm_desc->address, mm_desc->length);
    if (ret != 0) {
        ucs_warn("Unable to unmap shared memory segment at %p: %m", mm_desc->address);
        return UCS_ERR_SHMEM_SEGMENT;
    }

    return UCS_OK;
}

static ucs_status_t uct_posix_free(void *address, uct_mm_id_t mm_id, size_t length,
                                   const char *path)
{
    char *file_name;
    int ret;
    ucs_status_t status = UCS_OK;

    ucs_memtrack_releasing(&address);
    ret = ucs_munmap(address, length);
    if (ret != 0) {
        ucs_error("Unable to unmap shared memory segment at %p: %m", address);
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err;
    }

    file_name = ucs_calloc(1, NAME_MAX, "shared mr posix mmap");
    if (file_name == NULL) {
        ucs_error("Failed to allocate memory for the shm_unlink file name. %m");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = uct_posix_set_path(file_name, mm_id & UCT_MM_POSIX_SHM_OPEN, path,
                                mm_id >> UCT_MM_POSIX_CTRL_BITS);
    if (status != UCS_OK) {
        goto out_free_file;
    }

    /* use the mmid (62 bits uuid) to recreate the file_name for unlink */
    ret = (mm_id & UCT_MM_POSIX_SHM_OPEN) ? shm_unlink(file_name) : unlink(file_name);
    if (ret != 0) {
        ucs_warn("unable to unlink the shared memory segment. File name is: %s",
                 file_name);
        status = UCS_ERR_SHMEM_SEGMENT;
    }

out_free_file:
    ucs_free(file_name);

err:
    return status;
}

static uct_mm_mapper_ops_t uct_posix_mapper_ops = {
   .query   = ucs_empty_function_return_success,
   .get_path_size = uct_posix_get_path_size,
   .reg     = NULL,
   .dereg   = NULL,
   .alloc   = uct_posix_alloc,
   .attach  = uct_posix_attach,
   .detach  = uct_posix_detach,
   .free    = uct_posix_free
};

UCT_MM_COMPONENT_DEFINE(uct_posix_pd, "posix", &uct_posix_mapper_ops, uct_posix, "POSIX_")
UCT_PD_REGISTER_TL(&uct_posix_pd, &uct_mm_tl);
