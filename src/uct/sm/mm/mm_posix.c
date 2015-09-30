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

static ucs_config_field_t uct_posix_pd_config_table[] = {
  {"MM_", "", NULL,
   ucs_offsetof(uct_posix_pd_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_mm_pd_config_table)},

  {"DIR", "/dev/shm", "The path to the backing file",
   ucs_offsetof(uct_posix_pd_config_t, path), UCS_CONFIG_TYPE_STRING},

  {NULL}
};

static ucs_status_t
uct_posix_alloc(uct_pd_h pd, size_t *length_p, ucs_ternary_value_t hugetlb,
                void **address_p, uct_mm_id_t *mmid_p UCS_MEMTRACK_ARG)
{
    ucs_status_t status = UCS_ERR_NO_MEMORY;
    int shm_fd;
    uint64_t uuid;
    char *file_name;

    if (0 == *length_p) {
        ucs_error("Unexpected length %zu", *length_p);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    file_name = ucs_malloc(NAME_MAX, "shared mr posix");
    if (file_name == NULL) {
        status = UCS_ERR_NO_MEMORY;
        ucs_error("Failed to allocate memory for the shm_open file name. %m");
        goto err;
    }

    uuid = ucs_generate_uuid(0);
    /* use 63 bits of the uuid for creating the file_name.
     * 1 bit is for indicating whether or not hugepages were used */
    sprintf(file_name, "/ucx_shared_mr_uuid_%zu", uuid >> 1);

    /* Create shared memory object and set its size */
    shm_fd = shm_open(file_name, O_CREAT | O_RDWR | O_EXCL,
                      UCT_MM_POSIX_SHM_OPEN_MODE);
    if (shm_fd == -1) {
        ucs_error("Error returned from shm_open %m. File name is: %s",
                  file_name);
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err_free_file;
    }

    if (ftruncate(shm_fd, *length_p) == -1) {
        ucs_error("Error returned from ftruncate %m");
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err_shm_unlink;
    }

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
                                     uint64_t *cookie)
{
    void *ptr;
    char *file_name;
    int shm_fd;
    ucs_status_t status = UCS_OK;

    file_name = ucs_malloc(NAME_MAX, "shared mr posix");
    if (file_name == NULL) {
        ucs_error("Failed to allocate memory for file_name to attach. %m");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* use the mmid (63 bits) to recreate the file_name for opening */
    sprintf(file_name, "/ucx_shared_mr_uuid_%zu", mmid >> 1);
    shm_fd = shm_open(file_name, O_RDWR | O_EXCL,
                      UCT_MM_POSIX_SHM_OPEN_MODE);
    if (shm_fd == -1) {
        ucs_error("Error returned from shm_open in attach. %m. File name is: %s",
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

static ucs_status_t uct_posix_free(void *address, uct_mm_id_t mm_id, size_t length)
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

    file_name = ucs_malloc(NAME_MAX, "shared mr posix mmap");
    if (file_name == NULL) {
        ucs_error("Failed to allocate memory for the shm_unlink file name. %m");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* use the mmid (63 bits uuid) to recreate the file_name for shm_unlink */
    sprintf(file_name, "/ucx_shared_mr_uuid_%zu", mm_id >> 1);
    if (shm_unlink(file_name) != 0) {
        ucs_warn("unable to unlink the shared memory segment. File name is: %s",
                 file_name);
        status = UCS_ERR_SHMEM_SEGMENT;
        goto err_free_file;
    }

err_free_file:
    ucs_free(file_name);
err:
    return status;
}

static uct_mm_mapper_ops_t uct_posix_mapper_ops = {
   .query   = ucs_empty_function_return_success,
   .reg     = NULL,
   .dereg   = NULL,
   .alloc   = uct_posix_alloc,
   .attach  = uct_posix_attach,
   .detach  = uct_posix_detach,
   .free    = uct_posix_free
};

UCT_MM_COMPONENT_DEFINE(uct_posix_pd, "posix", &uct_posix_mapper_ops, uct_posix, "POSIX_")
UCT_PD_REGISTER_TL(&uct_posix_pd, &uct_mm_tl);
