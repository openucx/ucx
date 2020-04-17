/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef IODEMO_UCX_WRAPPER_H_
#define IODEMO_UCX_WRAPPER_H_

#include <ucp/api/ucp.h>
#include <deque>
#include <exception>
#include <iostream>
#include <map>
#include <sstream>
#include <string>


class UcxConnection;
struct ucx_request;

class verbose_ostream {
public:
    template<typename T>
    verbose_ostream& operator<<(const T &t) {
        if (_enable) {
            if (_new_line) {
                std::cout << _prefix;
                _new_line = false;
            }

            std::cout << t;
        }

        return *this; 
    }

    typedef std::basic_ostream<char, std::char_traits<char> > EndLine;
    typedef EndLine& (*EndLineManipulator)(EndLine&);
    verbose_ostream& operator<<(EndLineManipulator manipulator) {
        if (_enable) {
            std::cout << manipulator;
            _new_line = true;
        }

        return *this;
    }

    verbose_ostream(bool enable) : _enable(enable), _new_line(true) {
    }

    ~verbose_ostream() {
        if (_enable) {
            std::cout << std::flush;
        }
    }

private:
    const bool               _enable;
    bool                     _new_line;
    static const std::string _prefix;
};

class UcxCallback {
public:
    virtual ~UcxCallback();
    virtual void operator()(ucs_status_t status);
};


class EmptyCallback : public UcxCallback {
public:
    virtual void operator()() {};

    static EmptyCallback* get() {
        // singleton
        static EmptyCallback instance;
        return &instance;
    }
};


class UcxContext {
public:
    UcxContext(size_t iomsg_size, bool verbose);
    virtual ~UcxContext();

    void listen(const struct sockaddr* saddr, size_t addrlen);

    UcxConnection* connect(const struct sockaddr* saddr, size_t addrlen);

    void disconnect(UcxConnection* conn);

    virtual void on_disconnect(UcxConnection* conn) _GLIBCXX_NOTHROW;

    void progress();

    void request_wait(const char *what, void *request);

    static void request_release(void *request);

    static ucp_tag_t make_tag(uint32_t conn_id, uint32_t sn);

    static ucp_tag_t make_io_msg_tag(uint32_t conn_id, uint32_t sn);

    virtual void dispatch_new_connection(UcxConnection *conn);

    virtual void dispatch_io_message(UcxConnection* conn, const void *buffer);

    ucp_worker_h worker() const;

    inline verbose_ostream& verbose_os();

protected:
    virtual void process_failed_connection(UcxConnection *conn);

private:
    static const ucp_tag_t IOMSG_TAG = 1ull << 63;

    void post_recv();

    static void iomsg_recv_callback(void *request, ucs_status_t status,
                                    ucp_tag_recv_info *info);

    void process_conn_request();

    void process_io_message();

    void add_connection(UcxConnection *conn);

    void remove_connection(UcxConnection *conn);

    static uint32_t get_next_conn_id();

    static void request_init(void *request);

    static void request_reset(ucx_request *r);

    static void connect_callback(ucp_conn_request_h conn_req, void *arg);

    typedef std::map<uint32_t, UcxConnection*> conn_map_t;

    void cleanup_conns();
    void cleanup_listener();
    void cleanup_worker();

    ucp_context_h                  _context;
    ucp_worker_h                   _worker;
    ucp_listener_h                 _listener;
    conn_map_t                     _conns;
    ucx_request*                   _iomsg_recv_request;
    std::string                    _iomsg_buffer;
    std::deque<ucp_conn_request_h> _conn_requests;
    std::deque<UcxConnection *>    _closing_conns;
    verbose_ostream                _verbose_os;
};

verbose_ostream& UcxContext::verbose_os() {
    return _verbose_os;
}

class UcxConnection {
public:
    UcxConnection(UcxContext& context, uint32_t conn_id,
                  const struct sockaddr* saddr, socklen_t addrlen);

    UcxConnection(UcxContext& context, uint32_t conn_id,
                  ucp_conn_request_h conn_req);

    ~UcxConnection();

    void establish();

    void disconnect();

    void on_disconnect() _GLIBCXX_NOTHROW;

    bool is_disconnected();

    void send_io_message(const void *buffer, size_t length,
                         UcxCallback* callback = EmptyCallback::get());

    void send_data(const void *buffer, size_t length, uint32_t sn,
                   UcxCallback* callback = EmptyCallback::get());

    void recv_data(void *buffer, size_t length, uint32_t sn,
                   UcxCallback* callback = EmptyCallback::get());

    uint32_t id() const {
        return _conn_id;
    }

private:
    void ep_create_common(ucp_ep_params_t& ep_params);

    void ep_close(enum ucp_ep_close_mode mode);

    void send_common(const void *buffer, size_t length, ucp_tag_t tag,
                     UcxCallback* callback);

    static void stream_send_callback(void *request, ucs_status_t status);

    static void stream_recv_callback(void *request, ucs_status_t status,
                                     size_t recv_len);

    static void common_request_callback(void *request, ucs_status_t status);

    static void data_recv_callback(void *request, ucs_status_t status,
                                   ucp_tag_recv_info *info);

    static void process_request(const char *what, ucs_status_ptr_t ptr_status,
                                UcxCallback* callback);

    static void error_handler(void *arg, ucp_ep_h ep,
                              ucs_status_t status) _GLIBCXX_NOTHROW;

    UcxContext&  _context;
    uint32_t     _conn_id;
    uint32_t     _remote_conn_id;
    ucp_ep_h     _ep;
    void*        _close_request;
};

#endif

