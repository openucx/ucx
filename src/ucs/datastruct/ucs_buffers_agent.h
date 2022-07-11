/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_BUFFERS_AGENT_H_
#define UCS_BUFFERS_AGENT_H_

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/** @file ucs_buffers_agent.h */

typedef struct ucs_buffers_agent_buffer {
    uct_mem_h memh;
    void      *buf;
} ucs_buffers_agent_buffer_t;

typedef struct ucs_buffers_agent_ops {
    /**
     * Get buffer and uct memory handler from the memory allocation instance.
     *
     * @param [in]  agent memory allocator agent.
     * @param [in]  arg   uct obj arg. TODO - improve description.
     * @param [out] buf   agent buffer.
     *
     * @return            Error code as defined by @ref ucs_status_t
     */
    ucs_status_t (*get_buf)(void *agent, void *arg,
                            ucs_buffers_agent_buffer_t *buf);

    /**
     * Return buffer to the memory allocation instance.
     *
     * @param buff         to return.
     */
    void (*put_buf)(void *buff);
} ucs_buffers_agent_ops_t;

END_C_DECLS

#endif /* UCS_BUFFERS_AGENT_H_ */