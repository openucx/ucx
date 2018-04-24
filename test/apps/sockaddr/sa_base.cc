/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sa_base.h"
#include "sa_tcp.h"
#include "sa_util.h"

#include <cstring>


connection::~connection() {
}

void connection::set_id(uint64_t id) {
    m_id = id;
}

uint64_t connection::id() const {
    return m_id;
}

worker::~worker() {
}

std::shared_ptr<worker> worker::make(const std::string& mode,
                                     const struct sockaddr *listen_addr,
                                     socklen_t addrlen)
{
    if (mode == "tcp") {
        return std::make_shared<tcp_worker>(listen_addr, addrlen);
    } else {
        throw error("invalid mode: " + mode);
    }
}
