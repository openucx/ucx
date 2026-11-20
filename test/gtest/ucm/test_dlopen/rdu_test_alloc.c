/**
 * Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <stddef.h>
#include <stdint.h>

/*
 * Calls the allocation functions of the SambaNova rdu_devmem runtime through
 * the relocation table, as an application linked with the runtime does, so the
 * calls are intercepted by the UCM hooks.  The runtime library must be loaded
 * with RTLD_GLOBAL before this library.  The declarations match the runtime
 * header rdu_devmem.h, with the enum arguments and return values as int.
 */
int rdu_devmem_init(void);
void *rdu_malloc(uint32_t rdu_id, size_t size, int mem_type);
void rdu_free(uint32_t rdu_id, void *ptr);

int gtest_rdu_devmem_init(void)
{
    return rdu_devmem_init();
}

void *gtest_rdu_malloc(uint32_t rdu_id, size_t size, int mem_type)
{
    return rdu_malloc(rdu_id, size, mem_type);
}

void gtest_rdu_free(uint32_t rdu_id, void *ptr)
{
    rdu_free(rdu_id, ptr);
}
