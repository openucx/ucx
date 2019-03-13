/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef _CRAY_GNI_PUB_H
#define _CRAY_GNI_PUB_H

#include <stdint.h>
#include <stddef.h>

#define GNI_RC_SUCCESS                           0
#define GNI_CDM_MODE_FORK_FULLCOPY               0
#define GNI_CDM_MODE_CACHED_AMO_ENABLED          0
#define GNI_CDM_MODE_ERR_NO_KILL                 0
#define GNI_CDM_MODE_FAST_DATAGRAM_POLL          0
#define GNI_DEVICE_GEMINI                        1
#define GNI_DEVICE_ARIES                         2
#define GNI_RC_NOT_DONE                          0
#define GNI_RC_TRANSACTION_ERROR                 1
#define GNI_RC_ERROR_NOMEM                       2
#define GNI_RC_ERROR_RESOURCE                    3
#define GNI_RC_ALIGNMENT_ERROR                   4
#define GNI_POST_RDMA_PUT                        0
#define GNI_POST_RDMA_GET                        0
#define GNI_POST_FMA_PUT                         0
#define GNI_POST_FMA_GET                         0
#define GNI_POST_FMA_AMO                         0
#define GNI_POST_AMO                             0
#define GNI_POST_TERMINATED                      0
#define GNI_POST_COMPLETED                       0
#define GNI_POST_PENDING                         0
#define GNI_POST_CQWRITE                         0
#define GNI_FMA_ATOMIC_ADD                       0
#define GNI_FMA_ATOMIC_FADD                      0
#define GNI_FMA_ATOMIC_CSWAP                     0
#define GNI_CDM_MODE_FMA_SHARED                  0
#define GNI_CQMODE_GLOBAL_EVENT                  0
#define GNI_DLVMODE_PERFORMANCE                  0
#define GNI_DLVMODE_NO_ADAPT                     0
#define GNI_DLVMODE_IN_ORDER                     0
#define GNI_FMA_ATOMIC2_FSWAP                    0
#define GNI_FMA_ATOMIC2_IADD_S                   0
#define GNI_FMA_ATOMIC2_FIADD_S                  0
#define GNI_FMA_ATOMIC2_FSWAP_S                  0
#define GNI_FMA_ATOMIC2_CSWAP_S                  0
#define GNI_FMA_ATOMIC2_FCSWAP_S                 0
#define GNI_FMA_ATOMIC2_AND_S                    0
#define GNI_FMA_ATOMIC2_OR_S                     0
#define GNI_FMA_ATOMIC2_XOR_S                    0
#define GNI_FMA_ATOMIC2_SWAP_S                   0
#define GNI_FMA_ATOMIC2_IMIN_S                   0
#define GNI_FMA_ATOMIC2_IMAX_S                   0
#define GNI_FMA_ATOMIC_AND                       0
#define GNI_FMA_ATOMIC_OR                        0
#define GNI_FMA_ATOMIC_XOR                       0
#define GNI_FMA_ATOMIC2_SWAP                     0
#define GNI_FMA_ATOMIC2_IMIN                     0
#define GNI_FMA_ATOMIC2_IMAX                     0
#define GNI_FMA_ATOMIC2_FPADD_S                  0
#define GNI_FMA_ATOMIC2_FPMIN_S                  0
#define GNI_FMA_ATOMIC2_FPMAX_S                  0
#define GNI_FMA_ATOMIC2_FPADD                    0
#define GNI_FMA_ATOMIC2_FPMIN                    0
#define GNI_FMA_ATOMIC2_FPMAX                    0
#define GNI_FMA_ATOMIC2_FAND_S                   0
#define GNI_FMA_ATOMIC2_FOR_S                    0
#define GNI_FMA_ATOMIC2_FXOR_S                   0
#define GNI_FMA_ATOMIC2_FIMIN_S                  0
#define GNI_FMA_ATOMIC2_FIMAX_S                  0
#define GNI_FMA_ATOMIC_FAND                      0
#define GNI_FMA_ATOMIC_FOR                       0
#define GNI_FMA_ATOMIC_FXOR                      0
#define GNI_FMA_ATOMIC2_FIMIN                    0
#define GNI_FMA_ATOMIC2_FIMAX                    0
#define GNI_FMA_ATOMIC2_FFPADD_S                 0
#define GNI_FMA_ATOMIC2_FFPMIN_S                 0
#define GNI_FMA_ATOMIC2_FFPMAX_S                 0
#define GNI_FMA_ATOMIC2_FFPADD                   0
#define GNI_FMA_ATOMIC2_FFPMIN                   0
#define GNI_FMA_ATOMIC2_FFPMAX                   0
#define GNI_RC_ILLEGAL_OP                        0
#define GNI_DATAGRAM_MAXSIZE                     128
#define GNI_RC_NO_MATCH                          1
#define GNI_MEM_READWRITE                        0
#define GNI_MEM_READ_ONLY                        0
#define GNI_MEM_RELAXED_PI_ORDERING              0
#define GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT       0
#define GNI_SMSG_TYPE_INVALID                    0
#define GNI_SMSG_ANY_TAG                         0
#define GNI_RC_INVALID_PARAM                     0

#define GNI_CQ_OVERRUN(...)                      0
#define GNI_GetCompleted(_cq, _data, _desc)      ({*(_desc)=0; 0;})
#define GNI_CdmGetNicAddress(...)                0
#define GNI_GetDeviceType(...)                   0
#define GNI_CdmCreate(d, t, c, m, cdm)           ({ (void)(m); 0; })
#define GNI_GetLocalDeviceIds(...)               0
#define GNI_GetNumLocalDevices(_count)           ({*(_count)=0; 0;})
#define GNI_CdmDestroy(...)                      0
#define GNI_CqGetEvent(_cq, _data)               ({ (void)(_cq); *(_data) = 0; 0; })
#define GNI_MemRegister(h, ...)                  ({ (void)h; 0; })
#define GNI_MemDeregister(h, ...)                ({ (void)h; 0; })
#define GNI_CqDestroy(...)                       ({0;})
#define GNI_CqErrorStr(...)                      0
#define GNI_CdmAttach(...)                       0
#define GNI_CqErrorRecoverable(...)              0
#define GNI_CqVectorMonitor(...)                 0
#define GNI_CqCreate(...)                        0
#define GNI_EpCreate(...)                        0
#define GNI_EpDestroy(...)                       ({0;})
#define GNI_EpBind(...)                          0
#define GNI_EpUnbind(...)                        0
#define GNI_PostRdma(...)                        0
#define GNI_PostFma(...)                         0
#define GNI_PostCqWrite(...)                     0
#define GNI_EpPostDataWId(e, sh, ml, rh, ...)    ({ (void)e; (void)sh; (void)ml; (void)rh; 0; })
#define GNI_PostDataProbeById(nh, id)            ({ *(id) = 0; 0; })
#define GNI_EpPostDataWaitById(e, i, x, s, a, j) ({ (void)e; *(s) = 0; *(a) = 0; *(j) = 0; 0; })
#define GNI_EpPostDataCancel(...)                0
#define GNI_CQ_GET_MSG_ID(...)                   0
#define GNI_SmsgGetNextWTag(_ep, _ptr, _tag)     ({ *(_ptr) = NULL; 0; })
#define GNI_SmsgRelease(...)                     0
#define GNI_SmsgInit(_ep, _local, _remote)       ({ (void)(_ep); (void)(_local); (void)(_remote); 0; })
#define GNI_EpSetEventData(_a, _b, _h)           ({ (void)(_h); 0; })
#define GNI_SmsgSendWTag(...)                    0
#define GNI_CQ_STATUS_OK(...)                    0
#define GNI_CQ_GET_INST_ID(...)                  0
#define GNI_SmsgBufferSizeNeeded(_a, _b)         ({ (void)*(_a); *(_b) = 0; 0; })
#define GNI_SmsgSetMaxRetrans(...)               0
#define GNI_EpPostDataCancelById(...)            0
#define GNI_PostdataProbeWaitById(n,j,i)         ({ *(i) = 0; 0; })
#define GNI_EpPostDataTestById(i, u, s, a, j)    ({ *(a) = 0; *(j) = 0; 0; })


typedef int   gni_nic_device_t;
typedef void* gni_nic_handle_t;
typedef void* gni_cq_handle_t;
typedef void* gni_cdm_handle_t;
typedef void* gni_ep_handle_t;
typedef int   gni_return_t;
typedef int   gni_cq_entry_t;
typedef int   gni_post_type_t;
typedef int   gni_fma_cmd_type_t;
typedef int   gni_amo_cmd_type_t;
typedef int   gni_post_state_t;

extern const char* gni_err_str[];

typedef struct {
    uint64_t         qword1;
    uint64_t         qword2;
} gni_mem_handle_t;

typedef struct {
    int                 type;
    int                 cq_mode;
    int                 dlvr_mode;
    uint64_t            local_addr;
    gni_mem_handle_t    local_mem_hndl;
    uint64_t            remote_addr;
    gni_mem_handle_t    remote_mem_hndl;
    size_t              length;
    gni_cq_handle_t     src_cq_hndl;
    gni_amo_cmd_type_t  amo_cmd;
    uint64_t            first_operand;
    uint64_t            second_operand;
    int                 rdma_mode;
    int                 cqwrite_value;
} gni_post_descriptor_t;


typedef struct {
    gni_mem_handle_t    mem_hndl;
    void                *msg_buffer;
    int                 msg_type;
    size_t              buff_size;
    size_t              mbox_offset;
    int                 mbox_maxcredit;
    size_t              msg_maxsize;
} gni_smsg_attr_t;

#endif

