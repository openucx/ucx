/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "commands.h"

#include <ucs/type/status.h>

void commands::stream_command_t::init() {
    set(-1, CommandType::JUCX_INVALID);
    length      = 0;
    comp_status = CompletionStatus::JUCX_ERR;
}

void commands::stream_command_t::set(uint64_t id, Type type) {
    request_id  = id;
    cmd_type    = type;
}
