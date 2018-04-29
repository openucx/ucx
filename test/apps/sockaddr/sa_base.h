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
#include <memory>


/* interface for classes which generate events */
class event_source {
public:
    virtual void add_to_evpoll(evpoll_set& evpoll) = 0;
};


/* one data connection */
class connection : public event_source {
public:
    virtual ~connection();

    virtual size_t send(const char *buffer, size_t size) = 0;

    virtual size_t recv(char *buffer, size_t size) = 0;

    virtual bool is_closed() const = 0;

    uint64_t id() const;

protected:
    void set_id(uint64_t id);

private:
    uint64_t m_id;
};

typedef std::shared_ptr<connection> conn_ptr_t;


/* communication context */
class worker : public event_source {
public:
    typedef std::function<void(conn_ptr_t)>         conn_handler_t;
    typedef std::function<void(uint64_t, uint32_t)> data_handler_t;

    virtual ~worker();

    virtual conn_ptr_t connect(const struct sockaddr *addr, socklen_t addrlen) = 0;

    virtual void wait(const evpoll_set& evpoll, conn_handler_t conn_handler,
                      data_handler_t data_handler, int timeout_ms) = 0;

    /* factory function to create workers of given type */
    static std::shared_ptr<worker> make(const std::string& mode,
                                        const struct sockaddr *listen_addr,
                                        socklen_t addrlen);
};

#endif
