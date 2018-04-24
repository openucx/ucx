/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sa_ucx.h"

#include <sys/epoll.h>
#include <cstring>


template <typename T>
ucx_handle<T>::ucx_handle() : m_value(NULL), m_dtor(NULL) {
}

template <typename T>
ucx_handle<T>::ucx_handle(const T& value, dtor_t dtor) :
    m_value(value), m_dtor(dtor) {
}

template <typename T>
ucx_handle<T>::~ucx_handle() {
    reset();
}

template <typename T>
void ucx_handle<T>::reset() {
    if (m_value) {
        m_dtor(m_value);
        m_value = NULL;
    }
}

template <typename T>
void ucx_handle<T>::reset(const T& value, dtor_t dtor) {
    reset();
    m_value = value;
    m_dtor  = dtor;
}

template <typename T>
template <typename Ctor, typename... Args>
void ucx_handle<T>::reset(dtor_t dtor, Ctor ctor, Args&&... args)
{
    T value;
    ucs_status_t status = ctor(args..., &value);
    if (status != UCS_OK) {
        throw ucx_error("failed to create ucx handle", status);
    }
    reset(value, dtor);
}

template <typename T>
ucx_handle<T>::operator T() const {
    return m_value;
}

ucx_error::~ucx_error() throw() {
}

ucx_error::ucx_error(const std::string& message, ucs_status_t status) :
    error(message + ": " + ucs_status_string(status) +
          " (" + std::to_string(status) + ")") {
}

ucx_connection::ucx_connection(const ucx_handle<ucp_worker_h>& worker,
                               const ucp_ep_params_t& params) : m_worker(worker) {
    ucp_ep_params_t ep_params = params;
    set_ep_params(ep_params);
    m_ep.reset(ucp_ep_destroy, ucp_ep_create, worker, &ep_params);
    set_id();
}

ucx_connection::ucx_connection(const ucx_handle<ucp_worker_h>& worker, ucp_ep_h ep)
    : m_worker(worker), m_ep(ep, ucp_ep_destroy) {
    set_id();
    ucp_ep_params_t ep_params;
    ep_params.field_mask = 0;
    set_ep_params(ep_params);
    wait(ucp_ep_modify_nb(m_ep, &ep_params));
}

void ucx_connection::add_to_evpoll(evpoll_set& evpoll) {
}

size_t ucx_connection::send(const char *buffer, size_t size) {
    wait(ucp_stream_send_nb(m_ep, buffer, size, ucp_dt_make_contig(1), send_cb, 0));
    return size;
}

size_t ucx_connection::recv(char *buffer, size_t size) {
    size_t total_size = 0;
    for (;;) {
        size_t recv_length;
        ucs_status_ptr_t data = ucp_stream_recv_data_nb(m_ep, &recv_length);
        if (data == NULL) {
            return total_size;
        }

        if (UCS_PTR_IS_ERR(data)) {
            throw ucx_error("failed to receive ucx data", UCS_PTR_STATUS(data));
        }

        memcpy(buffer + total_size, data, recv_length);
        total_size += recv_length;
        ucp_stream_data_release(m_ep, data);
    }
}

bool ucx_connection::is_closed() const {
    return false;
}

void ucx_connection::send_cb(void *req, ucs_status_t status) {
}

void ucx_connection::set_id() {
    connection::set_id(reinterpret_cast<uint64_t>(this));
}

void ucx_connection::set_ep_params(ucp_ep_params_t& params) {
    // TODO set error handler
    params.field_mask |= UCP_EP_PARAM_FIELD_USER_DATA;
    params.user_data   = reinterpret_cast<void*>(this);
}


void ucx_connection::wait(void *req) {
    if (req == NULL) {
        return;
    }

    if (UCS_PTR_IS_ERR(req)) {
        throw ucx_error("ucx operation failed", UCS_PTR_STATUS(req));
    }

    ucs_status_t status;
    do {
        ucp_worker_progress(m_worker);
        status = ucp_request_check_status(req);
    } while (status == UCS_INPROGRESS);
    if (status != UCS_OK) {
        throw ucx_error("ucx request completed with error", status);
    }
}

ucx_worker::ucx_worker(const struct sockaddr *listen_addr, socklen_t addrlen) {
    ucp_params_t ctx_params;
    ctx_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ctx_params.features   = UCP_FEATURE_STREAM | UCP_FEATURE_WAKEUP;
    m_context.reset(ucp_cleanup, ucp_init, &ctx_params,
                    static_cast<const ucp_config*>(NULL));

    ucp_worker_params_t worker_params;
    worker_params.field_mask = 0;
    m_worker.reset(ucp_worker_destroy, ucp_worker_create, m_context,
                   &worker_params);

    ucp_listener_params_t listener_params;
    listener_params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                 UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
    listener_params.sockaddr.addr      = listen_addr;
    listener_params.sockaddr.addrlen   = addrlen;
    listener_params.accept_handler.cb  = listener_accept_cb;
    listener_params.accept_handler.arg = reinterpret_cast<void*>(this);
    m_listener.reset(ucp_listener_destroy, ucp_listener_create, m_worker,
                     &listener_params);
}

void ucx_worker::add_to_evpoll(evpoll_set& evpoll) {
}

conn_ptr_t ucx_worker::connect(const struct sockaddr *addr, socklen_t addrlen) {
    ucp_ep_params_t ep_params;
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = addr;
    ep_params.sockaddr.addrlen = addrlen;
    return std::make_shared<ucx_connection>(m_worker, ep_params);
}

void ucx_worker::wait(const evpoll_set& evpoll, conn_handler_t conn_handler,
                      data_handler_t data_handler, int timeout_ms) {

    while (!m_conn_backlog.empty()) {
        conn_handler(m_conn_backlog.front());
        m_conn_backlog.pop_front();
    }

    unsigned count = ucp_worker_progress(m_worker);
    if (count > 0) {
        static const size_t max_eps = 32;
        ucp_stream_poll_ep_t poll_eps[max_eps];
        ssize_t neps = ucp_stream_worker_poll(m_worker, poll_eps, max_eps, 0);
        if (neps < 0) {
            ucx_error("failed to arm worker", static_cast<ucs_status_t>(neps));
        }

        for (int i = 0; i < neps; ++i) {
            data_handler(reinterpret_cast<uint64_t>(poll_eps[i].user_data),
                         EPOLLIN);
        }
        return;
    }

    ucs_status_t status = ucp_worker_arm(m_worker);
    if (status == UCS_ERR_BUSY) {
        return;
    } else if (status != UCS_OK) {
        ucx_error("failed to arm worker", status);
    }

    // TODO epoll_wait
}

void ucx_worker::listener_accept_cb(ucp_ep_h ep, void *arg) {
    auto self = reinterpret_cast<ucx_worker*>(arg);
    self->m_conn_backlog.push_back(
                    std::make_shared<ucx_connection>(self->m_worker, ep));
}

