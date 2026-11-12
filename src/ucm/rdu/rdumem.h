/**
 * Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCM_RDUMEM_H_
#define UCM_RDUMEM_H_

#include <stddef.h>
#include <stdint.h>


/*
 * Names of the runtime library, found through the dynamic loader search path:
 * the versioned SONAME first, then the unversioned name of older runtimes
 */
#define UCM_RDU_DEVMEM_LIBS \
    { \
        "librdu_devmem.so.1", "librdu_devmem.so" \
    }


/*
 * Allocation entry points of the SambaNova rdu_devmem runtime
 * (librdu_devmem.so).  The runtime header is not needed to build: the
 * declarations below match it, with the rdu_mem_type_t enum argument passed as
 * an int, and the library is looked up at run time.
 */
void *ucm_override_rdu_malloc(uint32_t rdu_id, size_t size, int mem_type);
void *ucm_rdu_malloc(uint32_t rdu_id, size_t size, int mem_type);

void ucm_override_rdu_free(uint32_t rdu_id, void *ptr);
void ucm_rdu_free(uint32_t rdu_id, void *ptr);

#endif
