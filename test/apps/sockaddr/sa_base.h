/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SA_BASE_H_
#define SA_BASE_H_

#include "sa_util.h"

#include <sys/socket.h>
#include <functional>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <memory>


/* one data connection */
class connection {
public:
    virtual ~connection();

    uint64_t id() const;

protected:
    void set_id(uint64_t id);

private:
    uint64_t m_id;
};

typedef std::shared_ptr<connection> conn_ptr_t;


/* communication context */
class worker {
public:
    virtual ~worker();

    virtual conn_ptr_t connect(const struct sockaddr *addr, socklen_t addrlen) = 0;

    /* factory function to create workers of given type */
    static std::shared_ptr<worker> make(const std::string& mode,
                                        const struct sockaddr *listen_addr,
                                        socklen_t addrlen);
};

#endif
