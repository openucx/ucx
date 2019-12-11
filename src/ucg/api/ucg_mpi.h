/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCG_MPI_H_
#define UCG_MPI_H_

#include <ucg/api/ucg.h>

BEGIN_C_DECLS

/*
 * Below are the definitions targeted specifically for the MPI standard.
 * This includes a list of predefined collective operations, and the modifiers
 * that describe each. The macros generate functions with prototypes matching
 * the MPI requirement, including the same arguments the user would pass.
 */

enum ucg_predefined {
    UCG_PRIMITIVE_BARRIER,
    UCG_PRIMITIVE_REDUCE,
    UCG_PRIMITIVE_GATHER,
    UCG_PRIMITIVE_BCAST,
    UCG_PRIMITIVE_SCATTER,
    UCG_PRIMITIVE_ALLREDUCE,
    UCG_PRIMITIVE_ALLTOALL,
    UCG_PRIMITIVE_REDUCE_SCATTER,
    UCG_PRIMITIVE_ALLGATHER,
    UCG_PRIMITIVE_ALLGATHERV,
    UCG_PRIMITIVE_ALLTOALLW,
    UCG_PRIMITIVE_NEIGHBOR_ALLTOALLW
};

static enum ucg_collective_modifiers ucg_predefined_modifiers[] = {
    [UCG_PRIMITIVE_BARRIER]            = UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_BARRIER,
    [UCG_PRIMITIVE_REDUCE]             = UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_DESTINATION,
    [UCG_PRIMITIVE_GATHER]             = UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_DESTINATION,
    [UCG_PRIMITIVE_BCAST]              = UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE,
    [UCG_PRIMITIVE_SCATTER]            = UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE,
    [UCG_PRIMITIVE_ALLREDUCE]          = UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST,
    [UCG_PRIMITIVE_ALLTOALL]           = 0,
    [UCG_PRIMITIVE_REDUCE_SCATTER]     = UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE,
    [UCG_PRIMITIVE_ALLGATHER]          = UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST,
    [UCG_PRIMITIVE_ALLGATHERV]         = UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_VARIABLE_LENGTH,
    [UCG_PRIMITIVE_ALLTOALLW]          = UCG_GROUP_COLLECTIVE_MODIFIER_VARIABLE_LENGTH |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_VARIABLE_DATATYPE,
    [UCG_PRIMITIVE_NEIGHBOR_ALLTOALLW] = UCG_GROUP_COLLECTIVE_MODIFIER_NEIGHBOR |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_VARIABLE_LENGTH |
                                         UCG_GROUP_COLLECTIVE_MODIFIER_VARIABLE_DATATYPE,
};

#define UCG_COLL_PARAMS_BUF_R(_buf, _count, _dt_len, _dt_ext) \
    .buf    = _buf,                                           \
    .count  = _count,                                         \
    .dt_len = _dt_len,                                        \
    .dt_ext = _dt_ext,                                        \
    .op_ext = op

#define UCG_COLL_PARAMS_BUF_V(_buf, _counts, _dt_len, _dt_ext, _displs) \
    .buf    = _buf,                                                     \
    .counts = _counts,                                                  \
    .dt_len = _dt_len,                                                  \
    .dt_ext = _dt_ext,                                                  \
    .displs = _displs

#define UCG_COLL_PARAMS_BUF_W(_buf, _counts, _dts_len, _dts_ext, _displs) \
    .buf     = _buf,                                                      \
    .counts  = _counts,                                                   \
    .dts_len = _dts_len,                                                  \
    .dts_ext = _dts_ext,                                                  \
    .displs  = _displs

#define UCG_COLL_INIT_FUNC(_lname, _uname, _stype, _sargs, _rtype, _rargs, ...)\
static UCS_F_ALWAYS_INLINE ucs_status_t ucg_coll_##_lname##_init(__VA_ARGS__,  \
        ucg_group_h group, ucg_collective_callback_t cb, void *op,             \
        ucg_group_member_index_t root, unsigned modifiers, ucg_coll_h *coll_p) \
{                                                                              \
    enum ucg_collective_modifiers flags = modifiers |                          \
        ucg_predefined_modifiers[UCG_PRIMITIVE_##_uname];                      \
    ucg_collective_params_t params = {                                         \
            .type = {                                                          \
                    .modifiers = flags,                                        \
                    .root      = root,                                         \
            },                                                                 \
            .send = {                                                          \
                    UCG_COLL_PARAMS_BUF##_stype _sargs                         \
            },                                                                 \
            .recv = {                                                          \
                    UCG_COLL_PARAMS_BUF##_rtype _rargs                         \
            },                                                                 \
            .comp_cb        = cb                                               \
    };                                                                         \
    return ucg_collective_create(group, &params, coll_p);                      \
}

#define UCG_COLL_INIT_FUNC_SR1_RR1(_lname, _uname)                 \
UCG_COLL_INIT_FUNC(_lname, _uname,                                 \
                   _R, ((char*)sbuf, count, len_dtype, mpi_dtype), \
                   _R, (       rbuf, count, len_dtype, mpi_dtype), \
                   const void *sbuf, void *rbuf, int count,        \
                   size_t len_dtype, void *mpi_dtype)

#define UCG_COLL_INIT_FUNC_SR1_RRN(_lname, _uname)                    \
UCG_COLL_INIT_FUNC(_lname, _uname,                                    \
                   _R, ((char*)sbuf, scount, len_sdtype, mpi_sdtype), \
                   _R, (       rbuf, rcount, len_rdtype, mpi_rdtype), \
                   const void *sbuf, int scount, size_t len_sdtype, void *mpi_sdtype,\
                         void *rbuf, int rcount, size_t len_rdtype, void *mpi_rdtype)


#define UCG_COLL_INIT_FUNC_SR1_RVN(_lname, _uname)                              \
UCG_COLL_INIT_FUNC(_lname, _uname,                                              \
                   _R, ((char*)sbuf, scount,  len_sdtype,  mpi_sdtype),         \
                   _V, (       rbuf, rcounts, len_rdtype, mpi_rdtype, rdispls), \
                   const void *sbuf, void *rbuf,                                \
                   int  scount,  size_t len_sdtype, void *mpi_sdtype,           \
                   int *rcounts, size_t len_rdtype,                             \
                   void *mpi_rdtype, int *rdispls)

#define UCG_COLL_INIT_FUNC_SWN_RWN(_lname, _uname)                                \
UCG_COLL_INIT_FUNC(_lname, _uname,                                                \
                   _W, ((char*)sbuf, scounts, len_sdtypes, mpi_sdtypes, sdispls), \
                   _W, (       rbuf, rcounts, len_rdtypes, mpi_rdtypes, rdispls), \
                   const void *sbuf, void *rbuf, int *scounts,                    \
                   size_t *len_sdtypes, void **mpi_sdtypes, int *sdispls,         \
                   int *rcounts, size_t *len_rdtypes, void **mpi_rdtypes,         \
                   int *rdispls)


UCG_COLL_INIT_FUNC_SR1_RR1(allreduce,          ALLREDUCE)
UCG_COLL_INIT_FUNC_SR1_RR1(reduce,             REDUCE)
UCG_COLL_INIT_FUNC_SR1_RR1(bcast,              BCAST)
UCG_COLL_INIT_FUNC_SR1_RRN(gather,             GATHER)
UCG_COLL_INIT_FUNC_SR1_RRN(scatter,            SCATTER)
UCG_COLL_INIT_FUNC_SR1_RRN(allgather,          ALLGATHER)
UCG_COLL_INIT_FUNC_SR1_RVN(allgatherv,         ALLGATHERV)
UCG_COLL_INIT_FUNC_SR1_RRN(alltoall,           ALLTOALL)
UCG_COLL_INIT_FUNC_SWN_RWN(alltoallw,          ALLTOALLW)
UCG_COLL_INIT_FUNC_SWN_RWN(neighbor_alltoallw, NEIGHBOR_ALLTOALLW)
UCG_COLL_INIT_FUNC(barrier, BARRIER, _R, (0, 0, 0, 0), _R, (0, 0, 0, 0), int ign)

END_C_DECLS

#endif
