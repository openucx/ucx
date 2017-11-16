/**
 * Copyright (C) Mellanox Technologies Ltd. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_DT_H_
#define UCP_DT_H_

#include "dt_contig.h"
#include "dt_iov.h"
#include "dt_generic.h"

#include <ucp/core/ucp_types.h>
#include <uct/api/uct.h>
#include <ucp/api/ucp.h>


/**
 * State of progressing sent/receive operation on a datatype.
 */
typedef struct ucp_dt_state {
    size_t                        offset;  /* Total offset in overall payload. */
    union {
        struct {
            uct_mem_h         memh;
        } contig[UCP_MAX_RNDV_LANES];
        struct {
            size_t                iov_offset;     /* Offset in the IOV item */
            size_t                iovcnt_offset;  /* The IOV item to start copy */
            size_t                iovcnt;         /* Number of IOV buffers */
            uct_mem_h             *memh;          /* Pointer to IOV memh[iovcnt] */
        } iov;
        struct {
            void                  *state;
        } generic;
    } dt;
} ucp_dt_state_t;


size_t ucp_dt_pack(ucp_datatype_t datatype, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length);

static UCS_F_ALWAYS_INLINE void
ucp_dt_clear_memh(ucp_dt_state_t *state)
{
    int i;
    for (i = 0; i < UCP_MAX_RNDV_LANES; i++) {
        state->dt.contig[i].memh = UCT_MEM_HANDLE_NULL;
    }
}

static UCS_F_ALWAYS_INLINE int
ucp_dt_is_empty_rndv_lane(ucp_dt_state_t *state, int idx)
{
    return state->dt.contig[idx].memh == UCT_MEM_HANDLE_NULL;
}

static UCS_F_ALWAYS_INLINE int
ucp_dt_have_rndv_lanes(ucp_dt_state_t *state)
{
    return !ucp_dt_is_empty_rndv_lane(state, 0);
}

#endif /* UCP_DT_H_ */

