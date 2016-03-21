/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_IB_INSTR_H
#define UCT_IB_INSTR_H

#include <ucs/debug/instrument.h>

#if HAVE_INSTRUMENTATION

#define UCT_IB_INSTRUMENT_CHECK_RECV_SIGNALED(_wr) (1)

#define UCT_IB_INSTRUMENT_CHECK_SEND_SIGNALED(_wr) \
    ((_wr->send_flags & IBV_SEND_SIGNALED) && (_wr->opcode == IBV_WR_SEND))

#define UCT_IB_INSTRUMENT_CHECK_SEND_EXP_SIGNALED(_wr) \
    ((_wr->exp_send_flags & IBV_EXP_SEND_SIGNALED) && \
     (_wr->exp_opcode == IBV_EXP_WR_SEND))

#define UCT_IB_INSTRUMENT_RECORD_WR_LEN(_type, _name, _wr, _check) { \
    typeof(_wr) iterator = (_wr); \
    while (iterator) { \
        if _check(iterator) { \
            size_t length = 0; \
            int sge_index; \
            for (sge_index = 0; sge_index < iterator->num_sge; sge_index++) { \
                length += iterator->sg_list[sge_index].length; \
            } \
            UCS_INSTRUMENT_RECORD(_type, _name, iterator->wr_id, length); \
        } \
        iterator = iterator->next; \
    } \
}

#define UCT_IB_INSTRUMENT_RECORD_RECV_WR_LEN(_name, _wr) \
        UCT_IB_INSTRUMENT_RECORD_WR_LEN(UCS_INSTRUMENT_TYPE_IB_RX, _name, _wr, \
                                        UCT_IB_INSTRUMENT_CHECK_RECV_SIGNALED)

#define UCT_IB_INSTRUMENT_RECORD_SEND_WR_LEN(_name, _wr) \
        UCT_IB_INSTRUMENT_RECORD_WR_LEN(UCS_INSTRUMENT_TYPE_IB_TX, _name, _wr, \
                                        UCT_IB_INSTRUMENT_CHECK_SEND_SIGNALED)

#define UCT_IB_INSTRUMENT_RECORD_SEND_EXP_WR_LEN(_name, _wr) \
        UCT_IB_INSTRUMENT_RECORD_WR_LEN(UCS_INSTRUMENT_TYPE_IB_TX, _name, _wr, \
                                        UCT_IB_INSTRUMENT_CHECK_SEND_EXP_SIGNALED)

#define UCT_IB_INSTRUMENT_RECORD_SEND_OP(op) \
        UCS_INSTRUMENT_RECORD(UCS_INSTRUMENT_TYPE_IB_TX, __FUNCTION__, op);

#else

#define UCT_IB_INSTRUMENT_RECORD_WR_LEN(...)
#define UCT_IB_INSTRUMENT_RECORD_RECV_WR_LEN(...)
#define UCT_IB_INSTRUMENT_RECORD_SEND_WR_LEN(...)
#define UCT_IB_INSTRUMENT_RECORD_SEND_EXP_WR_LEN(...)
#define UCT_IB_INSTRUMENT_RECORD_SEND_OP(...)

#endif

#endif
