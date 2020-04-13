/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucx_wrapper.h"

#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <assert.h>

#include <limits>

const std::string verbose_ostream::_prefix("[VERBOSE] ");

struct ucx_request {
    UcxCallback  *callback;
    bool         completed;
    uint32_t     conn_id;
    ucs_status_t status;
};

static std::ostream& log() {
   return std::cout << "[UCX] ";
}

static const std::string
sockaddr_str(const struct sockaddr* saddr, size_t addrlen)
{
    char buf[128];

    if (saddr->sa_family != AF_INET) {
        return "<unknown address family>";
    }

    struct sockaddr_in in_addr = {0};
    memcpy(&in_addr, saddr, addrlen);
    inet_ntop(AF_INET, &in_addr.sin_addr, buf, sizeof(buf));
    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ":%d",
             ntohs(in_addr.sin_port));
    return buf;
}

UcxCallback::~UcxCallback()
{
}

struct UcxError : public std::exception {
    UcxError(const std::string& func_name, ucs_status_t status) :
        _err_str("UCX error: "), _status(status) {
        _err_str += func_name + "() failed: " + ucs_status_string(status) +
                    "\n";
    }

    ~UcxError() _GLIBCXX_NOTHROW {
    }

    const char* what() const _GLIBCXX_NOTHROW {
        return _err_str.c_str();
    }

    const char* status() const _GLIBCXX_NOTHROW {
        return ucs_status_string(_status);
    }
private:
    std::string  _err_str;
    ucs_status_t _status;
};

void UcxCallback::operator()()
{
}

UcxContext::UcxContext(size_t iomsg_size, bool verbose) :
    _listener(NULL), _iomsg_buffer(iomsg_size, '\0'), _verbose_os(verbose)
{
    /* Create context */
    ucp_params_t ucp_params;
    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                              UCP_PARAM_FIELD_REQUEST_INIT |
                              UCP_PARAM_FIELD_REQUEST_SIZE;
    ucp_params.features     = UCP_FEATURE_TAG |
                              UCP_FEATURE_STREAM;
    ucp_params.request_init = request_init;
    ucp_params.request_size = sizeof(ucx_request);
    ucs_status_t status = ucp_init(&ucp_params, NULL, &_context);
    if (status != UCS_OK) {
        throw UcxError("ucp_init", status);
    }

    log() << "created context " << _context << std::endl;

    /* Create worker */
    ucp_worker_params_t worker_params;
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    status = ucp_worker_create(_context, &worker_params, &_worker);
    if (status != UCS_OK) {
        ucp_cleanup(_context);
        throw UcxError("ucp_worker_create", status);
    }

    log() << "created worker " << _worker << std::endl;

    post_recv();
}

void UcxContext::cleanup_conns()
{
    while (!_conns.empty()) {
        disconnect(_conns.begin()->second);
    }

    while (!_closing_conns.empty()) {
        UcxConnection *conn = _closing_conns.front();
        _closing_conns.pop_front();
        delete conn;
    }
}

void UcxContext::cleanup_listener()
{
    if (_listener) {
        while (!_conn_requests.empty()) {
            assert(_listener);
            verbose_os() << "reject connection request " << _conn_requests.front()
                         << std::endl;
            ucp_listener_reject(_listener, _conn_requests.front());
            _conn_requests.pop_front();
        }

        ucp_listener_destroy(_listener);
    } else {
        assert(_conn_requests.empty());
    }
}

void UcxContext::cleanup_worker()
{
    ucp_request_cancel(_worker, _iomsg_recv_request);
    try {
        request_wait("iomsg_recv", _iomsg_recv_request);
    } catch (const UcxError& e) {
        log() << "io msg request is completed with status " << e.status()
              << std::endl;
    }

    ucp_worker_destroy(_worker);
}

UcxContext::~UcxContext()
{
    cleanup_conns();
    cleanup_listener();
    cleanup_worker();
    ucp_cleanup(_context);
}

void UcxContext::listen(const struct sockaddr* saddr, size_t addrlen)
{
    ucp_listener_params_t listener_params;

    listener_params.field_mask         = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                         UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    listener_params.sockaddr.addr      = saddr;
    listener_params.sockaddr.addrlen   = addrlen;
    listener_params.conn_handler.cb    = connect_callback;
    listener_params.conn_handler.arg   = reinterpret_cast<void*>(this);

    ucs_status_t status = ucp_listener_create(_worker, &listener_params,
                                              &_listener);
    if (status != UCS_OK) {
        throw UcxError("ucp_listener_create", status);
    }

    log() << "started listener " << _listener
          << " on " << sockaddr_str(saddr, addrlen) << std::endl;
}

UcxConnection* UcxContext::connect(const struct sockaddr* saddr, size_t addrlen)
{
    UcxConnection *conn = new UcxConnection(*this, get_next_conn_id(), saddr,
                                            addrlen);
    add_connection(conn);
    return conn;
}

void UcxContext::disconnect(UcxConnection* conn)
{
    verbose_os() << "closing connection " << conn << " with id = " << conn->id()
                 << std::endl;
    remove_connection(conn);
    conn->disconnect();
    if (!conn->is_disconnected()) {
        _closing_conns.push_back(conn);
    }
}

void UcxContext::on_disconnect(UcxConnection* conn) _GLIBCXX_NOTHROW
{
    verbose_os() << "closing connection " << conn << " with id = " << conn->id()
                 << std::endl;
    remove_connection(conn);
    conn->on_disconnect();
    if (!conn->is_disconnected()) {
        _closing_conns.push_back(conn);
    }
}

void UcxContext::dispatch_new_connection(UcxConnection *conn)
{
}

void UcxContext::dispatch_io_message(UcxConnection* conn, const void *buffer)
{
}

void UcxContext::progress()
{
    ucp_worker_progress(_worker);

    if (_iomsg_recv_request->completed) {
        process_io_message();
        request_release(_iomsg_recv_request);
        post_recv();
    }

    process_conn_request();

    if (!_closing_conns.empty()) {
        UcxConnection *conn = _closing_conns.front();
        if (conn->is_disconnected()) {
            _closing_conns.pop_front();
            delete conn;
            throw UcxError("connection", UCS_ERR_CONNECTION_RESET);
        }
    }
}

void UcxContext::request_wait(const char *what, void *request)
{
    if (UCS_PTR_IS_PTR(request)) {
        ucs_status_t status;
        do {
            ucp_worker_progress(_worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        request_release(request);
        if (UCS_STATUS_IS_ERR(status)) {
            throw UcxError(what, status);
        }
    } else if (UCS_PTR_IS_ERR(request)) {
        throw UcxError(what, UCS_PTR_STATUS(request));
    }
}

ucp_tag_t UcxContext::make_tag(uint32_t conn_id, uint32_t sn)
{
    return (static_cast<uint64_t>(conn_id) << 32) | sn;
}

ucp_tag_t UcxContext::make_io_msg_tag(uint32_t conn_id, uint32_t sn)
{
    return IOMSG_TAG | make_tag(conn_id, sn);
}

ucp_worker_h UcxContext::worker() const
{
    return _worker;
}

void UcxContext::post_recv()
{
    ucs_status_ptr_t ptr_status = ucp_tag_recv_nb(_worker, &_iomsg_buffer[0],
                                                  _iomsg_buffer.size(),
                                                  ucp_dt_make_contig(1),
                                                  IOMSG_TAG, IOMSG_TAG,
                                                  iomsg_recv_callback);
    assert(ptr_status != NULL);
    _iomsg_recv_request = reinterpret_cast<ucx_request*>(ptr_status);
}

void UcxContext::iomsg_recv_callback(void *request, ucs_status_t status,
                                     ucp_tag_recv_info *info)
{
    ucx_request *r = reinterpret_cast<ucx_request*>(request);
    r->completed = true;
    r->conn_id   = (info->sender_tag & ~IOMSG_TAG) >> 32;
}

void UcxContext::process_conn_request()
{
    if (_conn_requests.empty()) {
        return;
    }

    ucp_conn_request_h conn_req = _conn_requests.front();
    _conn_requests.pop_front();
    UcxConnection *conn = new UcxConnection(*this, get_next_conn_id(), conn_req);
    try {
        add_connection(conn);
    } catch (const UcxError& e) {
        log() << e.what();
    }
}

void UcxContext::process_failed_connection(UcxConnection *conn)
{
    log() << "closing failed connection " << conn << std::endl;
    remove_connection(conn);
    delete conn;
}

void UcxContext::process_io_message()
{
    uint32_t conn_id = _iomsg_recv_request->conn_id;
    conn_map_t::iterator iter = _conns.find(conn_id);
    if (iter == _conns.end()) {
        log() << "could not find connection with id " << conn_id << std::endl;
        return;
    }

    UcxConnection *conn = iter->second;
    dispatch_io_message(conn, &_iomsg_buffer[0]);
}

void UcxContext::add_connection(UcxConnection *conn)
{
    conn->establish();
    _conns[conn->id()] = conn;
}

void UcxContext::remove_connection(UcxConnection *conn)
{
    _conns.erase(conn->id());
}

uint32_t UcxContext::get_next_conn_id()
{
    static uint32_t conn_id = 1;
    return conn_id++;
}

void UcxContext::request_init(void *request)
{
    ucx_request *r = reinterpret_cast<ucx_request*>(request);
    request_reset(r);
}

void UcxContext::request_reset(ucx_request *r)
{
    r->completed = false;
    r->callback  = NULL;
}

void UcxContext::request_release(void *request)
{
    request_reset(reinterpret_cast<ucx_request*>(request));
    ucp_request_free(request);
}

void UcxContext::connect_callback(ucp_conn_request_h conn_req, void *arg)
{
    UcxContext *self = reinterpret_cast<UcxContext*>(arg);
    log() << "got new connection request " << conn_req << std::endl;
    self->_conn_requests.push_back(conn_req);
}

void UcxConnection::send_io_message(const void *buffer, size_t length,
                                    UcxCallback* callback)
{
    ucp_tag_t tag = UcxContext::make_io_msg_tag(_remote_conn_id, 0);
    send_common(buffer, length, tag, callback);
}

void UcxConnection::send_data(const void *buffer, size_t length, uint32_t sn,
                              UcxCallback* callback)
{
    ucp_tag_t tag = UcxContext::make_tag(_remote_conn_id, sn);
    send_common(buffer, length, tag, callback);
}

void UcxConnection::recv_data(void *buffer, size_t length, uint32_t sn,
                              UcxCallback* callback)
{
    ucp_tag_t tag      = UcxContext::make_tag(_conn_id, sn);
    ucp_tag_t tag_mask = std::numeric_limits<ucp_tag_t>::max();
    ucs_status_ptr_t ptr_status = ucp_tag_recv_nb(_context.worker(), buffer,
                                                  length, ucp_dt_make_contig(1),
                                                  tag, tag_mask,
                                                  data_recv_callback);
    process_request("ucp_tag_recv_nb", ptr_status, callback);
}

UcxConnection::UcxConnection(UcxContext &context, uint32_t conn_id,
                             const struct sockaddr *saddr, socklen_t addrlen) :
    _context(context), _conn_id(conn_id), _remote_conn_id(0),
    _close_request(NULL)
{
    ucp_ep_params_t ep_params;
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS       |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = saddr;
    ep_params.sockaddr.addrlen = addrlen;

    ep_create_common(ep_params);

    log() << "connected to " << sockaddr_str(saddr, addrlen)  << std::endl;
}

UcxConnection::UcxConnection(UcxContext &context, uint32_t conn_id,
                             ucp_conn_request_h conn_req) :
    _context(context), _conn_id(conn_id), _remote_conn_id(0),
    _close_request(NULL)
{
    ucp_ep_params_t ep_params;
    ep_params.field_mask   = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = conn_req;

    ep_create_common(ep_params);
}

UcxConnection::~UcxConnection()
{
    if (_ep != NULL) {
        disconnect();
    }

    try {
        _context.request_wait("ucp_ep_close_nb", _close_request);
    } catch (UcxError &e) {
        log() << e.what();
    }
    _close_request = NULL;
    assert(_ep == NULL);
}

void UcxConnection::establish()
{
    // send local connection id
    const ucp_datatype_t dt_int = ucp_dt_make_contig(sizeof(uint32_t));
    void *sreq = ucp_stream_send_nb(_ep, &_conn_id, 1, dt_int,
                                    stream_send_callback, 0);
    // receive remote connection id
    size_t recv_len;
    void *rreq = ucp_stream_recv_nb(_ep, &_remote_conn_id, 1, dt_int,
                                    stream_recv_callback, &recv_len,
                                    UCP_STREAM_RECV_FLAG_WAITALL);
    try {
        _context.request_wait("ucp_stream_recv_nb", rreq);
    } catch (const UcxError &e) {
        log() << e.what();
    }

    _context.request_wait("ucp_stream_send_nb", sreq);

    log() << "connection " << this << " local id " << _conn_id
          << " remote id " << _remote_conn_id << std::endl;
}

void UcxConnection::disconnect()
{
    log() << "closing ep " << _ep << std::endl;
    ep_close(UCP_EP_CLOSE_MODE_FLUSH);
}

void UcxConnection::on_disconnect() _GLIBCXX_NOTHROW
{
    log() << "handle disconnect on ep " << _ep << std::endl;
    ep_close(UCP_EP_CLOSE_MODE_FORCE);
}

bool UcxConnection::is_disconnected()
{
    // disconnect is in progress, check request completion
    if (UCS_PTR_IS_PTR(_close_request)) {
        return ucp_request_is_completed(_close_request);
    }

    return (_ep == NULL);
}

void UcxConnection::ep_create_common(ucp_ep_params_t& ep_params)
{
    // create endpoint
    ep_params.field_mask      |= UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = error_handler;
    ep_params.err_handler.arg  = reinterpret_cast<void*>(this);

    ucs_status_t status = ucp_ep_create(_context.worker(), &ep_params, &_ep);
    if (status != UCS_OK) {
        _ep = NULL;
        throw UcxError("ucp_ep_create", status);
    }

    log() << "connection " << this << " id " << _conn_id
          << " created ep " << _ep << std::endl;
}

void UcxConnection::ep_close(enum ucp_ep_close_mode mode)
{
    assert(_ep            != NULL);
    assert(_close_request == NULL);
    _close_request = ucp_ep_close_nb(_ep, mode);
    _ep            = NULL;
}

void UcxConnection::send_common(const void *buffer, size_t length, ucp_tag_t tag,
                                UcxCallback* callback)
{
    ucs_status_ptr_t ptr_status = ucp_tag_send_nb(_ep, buffer, length,
                                                  ucp_dt_make_contig(1), tag,
                                                  common_request_callback);
    process_request("ucp_tag_send_nb", ptr_status, callback);
}

void UcxConnection::stream_send_callback(void *request, ucs_status_t status)
{
}

void UcxConnection::stream_recv_callback(void *request, ucs_status_t status,
                                         size_t recv_len)
{
}

void UcxConnection::common_request_callback(void *request, ucs_status_t status)
{
    ucx_request *r = reinterpret_cast<ucx_request*>(request);
    if (r->callback) {
        // already processed by send function
        (*r->callback)();
        UcxContext::request_release(r);
    } else {
        // not yet processed by send function
        r->completed = true;
        r->status    = status;
    }
}

void UcxConnection::data_recv_callback(void *request, ucs_status_t status,
                                       ucp_tag_recv_info *info)
{
    common_request_callback(request, status);
}

void UcxConnection::process_request(const char *what,
                                    ucs_status_ptr_t ptr_status,
                                    UcxCallback* callback)
{
    if (ptr_status == NULL) {
        (*callback)();
    } else if (UCS_PTR_IS_ERR(ptr_status)) {
        log() << what << "failed with status"
              << ucs_status_string(UCS_PTR_STATUS(ptr_status)) << std::endl;
        throw UcxError(what, UCS_PTR_STATUS(ptr_status));
    } else {
        // pointer to request
        ucx_request *r = reinterpret_cast<ucx_request*>(ptr_status);
        if (r->completed) {
            // already completed by callback
            (*callback)();
            UcxContext::request_release(r);
        } else {
            // will be completed by callback
            r->callback = callback;
        }
    }
}

void UcxConnection::error_handler(void *arg, ucp_ep_h ep,
                                  ucs_status_t status) _GLIBCXX_NOTHROW
{
    UcxConnection *self = reinterpret_cast<UcxConnection*>(arg);

    assert(ep == self->_ep);
    log() << "detected error " << ucs_status_string(status)
          << " on connection " << self << std::endl;
    self->_context.on_disconnect(self);
}
