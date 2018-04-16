/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sa_base.h"

#include <ucp/api/ucp.h>
#include <list>


/* noncopyable wrapper for ucx handles which calls custom destructor */
template <typename T>
class ucx_handle {
public:
    typedef void (*dtor_t)(T handle);

    ucx_handle();

    ucx_handle(const T& value, dtor_t dtor);

    ~ucx_handle();

    void reset();

    void reset(const T& value, dtor_t dtor);

    template <typename Ctor, typename... Args>
    void reset(dtor_t dtor, Ctor ctor, Args&&... args);

    operator T() const;

private:
    ucx_handle(const ucx_handle& other);

    const ucx_handle& operator=(const ucx_handle& other);

    T      m_value;
    dtor_t m_dtor;
};


/* ucx error exception */
class ucx_error : public error {
public:
    virtual ~ucx_error() throw();

    ucx_error(const std::string& message, ucs_status_t status);
};


class ucx_connection : public connection {
public:
    ucx_connection(const ucx_handle<ucp_worker_h>& worker,
                   const ucp_ep_params_t& params);

    ucx_connection(const ucx_handle<ucp_worker_h>& worker, ucp_ep_h ep);

    virtual void add_to_evpoll(evpoll_set& evpoll);

    virtual size_t send(const char *buffer, size_t size);

    virtual size_t recv(char *buffer, size_t size);

    virtual bool is_closed() const;

private:
    static void send_cb(void *req, ucs_status_t status);

    void set_id();

    void set_ep_params(ucp_ep_params_t& params);

    void wait(void *req);

    ucp_worker_h         m_worker; // TODO implement shared_handle??
    ucx_handle<ucp_ep_h> m_ep;
};


class ucx_worker : public worker {
public:
    ucx_worker(const struct sockaddr *listen_addr, socklen_t addrlen);

    virtual void add_to_evpoll(evpoll_set& evpoll);

    virtual conn_ptr_t connect(const struct sockaddr *addr, socklen_t addrlen);

    virtual void wait(const evpoll_set& evpoll, conn_handler_t conn_handler,
                      data_handler_t data_handler, int timeout_ms);

private:
    static void listener_accept_cb(ucp_ep_h ep, void *arg);

    ucx_handle<ucp_context_h>   m_context;
    ucx_handle<ucp_worker_h>    m_worker;
    ucx_handle<ucp_listener_h>  m_listener;
    std::list<conn_ptr_t>       m_conn_backlog;
};
