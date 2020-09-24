/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef IODEMO_UCX_WRAPPER_H_
#define IODEMO_UCX_WRAPPER_H_

#include <ucp/api/ucp.h>
#include <ucs/algorithm/crc.h>
#include <deque>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <ucs/datastruct/list.h>

#define MAX_LOG_PREFIX_SIZE   64

/* Forward declarations */
class UcxConnection;
struct ucx_request;

/*
 * UCX callback for send/receive completion
 */
class UcxCallback {
public:
    virtual ~UcxCallback();
    virtual void operator()(ucs_status_t status) = 0;
};


/*
 * Empty callback singleton
 */
class EmptyCallback : public UcxCallback {
public:
    /// @override
    virtual void operator()(ucs_status_t status);

    static EmptyCallback* get();
};


/*
 * Logger which can be enabled/disabled
 */
class UcxLog {
public:
    UcxLog(const char* prefix, bool enable = true);

    ~UcxLog();

    template<typename T>
    const UcxLog& operator<<(const T &t) const {
        if (_enable) {
            std::cout << t;
        }
        return *this;
    }

private:
    const bool               _enable;
};


/**
 * Holds UCX global context and worker
 */
class UcxContext {
public:
    UcxContext(size_t iomsg_size);

    virtual ~UcxContext();

    bool init();

    bool listen(const struct sockaddr* saddr, size_t addrlen);

    UcxConnection* connect(const struct sockaddr* saddr, size_t addrlen);

    void progress();

protected:

    // Called when new connection is created on server side
    virtual void dispatch_new_connection(UcxConnection *conn);

    // Called when new IO message is received
    virtual void dispatch_io_message(UcxConnection* conn, const void *buffer,
                                     size_t length);

    // Called when there is a fatal failure on the connection
    virtual void dispatch_connection_error(UcxConnection* conn);

private:
    typedef enum {
        WAIT_STATUS_OK,
        WAIT_STATUS_FAILED,
        WAIT_STATUS_TIMED_OUT
    } wait_status_t;

    friend class UcxConnection;

    static const ucp_tag_t IOMSG_TAG = 1ull << 63;

    static uint32_t get_next_conn_id();

    static void request_init(void *request);

    static void request_reset(ucx_request *r);

    static void request_release(void *request);

    static void connect_callback(ucp_conn_request_h conn_req, void *arg);

    static void iomsg_recv_callback(void *request, ucs_status_t status,
                                    ucp_tag_recv_info *info);

    static const std::string sockaddr_str(const struct sockaddr* saddr,
                                          size_t addrlen);

    ucp_worker_h worker() const;

    void progress_conn_requests();

    void progress_io_message();

    void progress_failed_connections();

    wait_status_t wait_completion(ucs_status_ptr_t status_ptr, const char *title,
                                  double timeout = 1e6);

    void recv_io_message();

    void add_connection(UcxConnection *conn);

    void remove_connection(UcxConnection *conn);

    void handle_connection_error(UcxConnection *conn);

    void destroy_connections();

    void destroy_listener();

    void destroy_worker();

    typedef std::map<uint32_t, UcxConnection*> conn_map_t;

    ucp_context_h                  _context;
    ucp_worker_h                   _worker;
    ucp_listener_h                 _listener;
    conn_map_t                     _conns;
    ucx_request*                   _iomsg_recv_request;
    std::string                    _iomsg_buffer;
    std::deque<ucp_conn_request_h> _conn_requests;
    std::deque<UcxConnection *>    _failed_conns;
};


class UcxConnection {
public:
    UcxConnection(UcxContext& context, uint32_t conn_id);

    ~UcxConnection();

    bool connect(const struct sockaddr* saddr, socklen_t addrlen);

    bool accept(ucp_conn_request_h conn_req);

    bool send_io_message(const void *buffer, size_t length,
                         UcxCallback* callback = EmptyCallback::get());

    bool send_data(const void *buffer, size_t length, uint32_t sn,
                   UcxCallback* callback = EmptyCallback::get());

    bool recv_data(void *buffer, size_t length, uint32_t sn,
                   UcxCallback* callback = EmptyCallback::get());

    void cancel_all();

    uint32_t id() const {
        return _conn_id;
    }

private:
    static ucp_tag_t make_data_tag(uint32_t conn_id, uint32_t sn);

    static ucp_tag_t make_iomsg_tag(uint32_t conn_id, uint32_t sn);

    static void stream_send_callback(void *request, ucs_status_t status);

    static void stream_recv_callback(void *request, ucs_status_t status,
                                     size_t recv_len);

    static void common_request_callback(void *request, ucs_status_t status);

    static void data_recv_callback(void *request, ucs_status_t status,
                                   ucp_tag_recv_info *info);

    static void error_callback(void *arg, ucp_ep_h ep, ucs_status_t status);

    void set_log_prefix(const struct sockaddr* saddr, socklen_t addrlen);

    bool connect_common(ucp_ep_params_t& ep_params);

    bool send_common(const void *buffer, size_t length, ucp_tag_t tag,
                     UcxCallback* callback);

    void request_started(ucx_request *r);

    void request_completed(ucx_request *r);

    void handle_connection_error(ucs_status_t status);

    void disconnect(enum ucp_ep_close_mode mode);

    void ep_close(enum ucp_ep_close_mode mode);

    bool process_request(const char *what, ucs_status_ptr_t ptr_status,
                         UcxCallback* callback);

    static unsigned    _num_instances;

    UcxContext&        _context;
    uint32_t           _conn_id;
    uint32_t           _remote_conn_id;
    char               _log_prefix[MAX_LOG_PREFIX_SIZE];
    ucp_ep_h           _ep;
    void*              _close_request;
    ucs_list_link_t    _all_requests;
};

#endif

