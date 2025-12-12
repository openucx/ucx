/*
 * Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GAUDI_SCAL_H_
#define UCT_GAUDI_SCAL_H_

#include <stdint.h>

#define UCT_GAUID_SCAL_SUCCESS 0
#define DECLARE_HANDLE(name) \
    struct uct_gaudi_##name##__ { \
        int unused; \
    }; \
    typedef struct uct_gaudi_##name##__ *name

DECLARE_HANDLE(scal_handle_t);
DECLARE_HANDLE(scal_pool_handle_t);
DECLARE_HANDLE(scal_arc_fw_config_handle_t);

typedef struct _scal_memory_pool_infoV2 {
    scal_handle_t scal;
    const char    *name;
    unsigned      idx;
    uint64_t      device_base_address;
    void          *host_base_address;
    uint32_t core_base_address; // 0 when the pool is not mapped to the cores
    uint64_t totalSize;
    uint64_t freeSize;
    uint64_t device_base_allocated_address;
} scal_memory_pool_infoV2;

int scal_init(int fd, const char *config_file_path, scal_handle_t *scal,
              scal_arc_fw_config_handle_t fwCfg);
int scal_get_handle_from_fd(int fd, scal_handle_t *scal);
int scal_get_pool_handle_by_name(const scal_handle_t scal,
                                 const char *pool_name,
                                 scal_pool_handle_t *pool);
int scal_pool_get_infoV2(const scal_pool_handle_t pool,
                         scal_memory_pool_infoV2 *info);
#endif
