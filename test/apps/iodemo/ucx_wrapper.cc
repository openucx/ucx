/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucx_wrapper.h"

#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <string.h>
#include <assert.h>

#include <algorithm>
#include <limits>


#define AM_MSG_ID 0


struct ucx_request {
    UcxCallback                  *callback;
    UcxConnection                *conn;
    ucs_status_t                 status;
    bool                         completed;
    uint32_t                     conn_id;
    size_t                       recv_length;
    ucs_list_link_t              pos;
};

// Holds details of arrived AM message
struct UcxAmDesc {
    UcxAmDesc(void *data, const ucp_am_recv_param_t *param) :
        _data(data), _param(param) {
    }

    void                         *_data;
    const ucp_am_recv_param_t    *_param;
};

UcxCallback::~UcxCallback()
{
}

void EmptyCallback::operator()(ucs_status_t status)
{
}

EmptyCallback* EmptyCallback::get() {
    // singleton
    static EmptyCallback instance;
    return &instance;
}

bool UcxLog::use_human_time = false;

UcxLog::UcxLog(const char* prefix, bool enable)
{
    if (!enable) {
        _ss = NULL;
        return;
    }

    struct timeval tv;
    gettimeofday(&tv, NULL);

    struct tm tm;
    char str[32];
    if (use_human_time) {
        strftime(str, sizeof(str), "[%a %b %d %T] ", localtime_r(&tv.tv_sec, &tm));
    } else {
        snprintf(str, sizeof(str), "[%lu.%06lu] ", tv.tv_sec, tv.tv_usec);
    }

    _ss = new std::stringstream();
    (*_ss) << str << prefix << " ";
}

UcxLog::~UcxLog()
{
    if (_ss != NULL) {
        (*_ss) << std::endl;
        std::cout << (*_ss).str();
        delete _ss;
    }
}

#define UCX_LOG UcxLog("[UCX]", true)

UcxContext::UcxAcceptCallback::UcxAcceptCallback(UcxContext &context,
                                                 UcxConnection &connection) :
    _context(context), _connection(connection)
{
}

void UcxContext::UcxAcceptCallback::operator()(ucs_status_t status)
{
    if (status == UCS_OK) {
        _context.dispatch_connection_accepted(&_connection);
    }

    delete this;
}

UcxContext::UcxDisconnectCallback::UcxDisconnectCallback(UcxConnection &conn)
    : _conn(&conn)
{
}

UcxContext::UcxDisconnectCallback::~UcxDisconnectCallback()
{
    delete _conn;
}

void UcxContext::UcxDisconnectCallback::operator()(ucs_status_t status)
{
    delete this;
}

UcxContext::UcxContext(size_t iomsg_size, double connect_timeout, bool use_am) :
    _context(NULL), _worker(NULL), _listener(NULL), _iomsg_recv_request(NULL),
    _iomsg_buffer(iomsg_size, '\0'), _connect_timeout(connect_timeout),
    _use_am(use_am)
{
}

UcxContext::~UcxContext()
{
    destroy_connections();
    destroy_listener();
    destroy_worker();
    if (_context) {
        ucp_cleanup(_context);
    }
}

bool UcxContext::init()
{
    if (_context && _worker) {
        UCX_LOG << "context is already initialized";
        return true;
    }

    /* Create context */
    ucp_params_t ucp_params;
    ucp_params.field_mask   = UCP_PARAM_FIELD_FEATURES |
                              UCP_PARAM_FIELD_REQUEST_INIT |
                              UCP_PARAM_FIELD_REQUEST_SIZE;
    ucp_params.features     = _use_am ? UCP_FEATURE_AM :
                                        UCP_FEATURE_TAG | UCP_FEATURE_STREAM;
    ucp_params.request_init = request_init;
    ucp_params.request_size = sizeof(ucx_request);
    ucs_status_t status = ucp_init(&ucp_params, NULL, &_context);
    if (status != UCS_OK) {
        UCX_LOG << "ucp_init() failed: " << ucs_status_string(status);
        return false;
    }

    UCX_LOG << "created context " << _context << " with "
            << (_use_am ? "AM" : "TAG");

    /* Create worker */
    ucp_worker_params_t worker_params;
    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
    status = ucp_worker_create(_context, &worker_params, &_worker);
    if (status != UCS_OK) {
        ucp_cleanup(_context);
        _context = NULL;
        UCX_LOG << "ucp_worker_create() failed: " << ucs_status_string(status);
        return false;
    }

    UCX_LOG << "created worker " << _worker;

    if (_use_am) {
        set_am_handler(am_recv_callback, this);
    } else {
        recv_io_message();
    }

    return true;
}

bool UcxContext::listen(const struct sockaddr* saddr, size_t addrlen)
{
    ucp_listener_params_t listener_params;

    listener_params.field_mask       = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                       UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    listener_params.sockaddr.addr    = saddr;
    listener_params.sockaddr.addrlen = addrlen;
    listener_params.conn_handler.cb  = connect_callback;
    listener_params.conn_handler.arg = reinterpret_cast<void*>(this);

    ucs_status_t status = ucp_listener_create(_worker, &listener_params,
                                              &_listener);
    if (status != UCS_OK) {
        UCX_LOG << "ucp_listener_create() failed: " << ucs_status_string(status);
        return false;
    }

    UCX_LOG << "started listener " << _listener << " on "
            << sockaddr_str(saddr, addrlen);
    return true;
}

void UcxContext::progress()
{
    ucp_worker_progress(_worker);
    progress_io_message();
    progress_timed_out_conns();
    progress_conn_requests();
    progress_failed_connections();
    progress_disconnected_connections();
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
    r->completed   = false;
    r->callback    = NULL;
    r->conn        = NULL;
    r->status      = UCS_OK;
    r->recv_length = 0;
    r->pos.next    = NULL;
    r->pos.prev    = NULL;
}

void UcxContext::request_release(void *request)
{
    request_reset(reinterpret_cast<ucx_request*>(request));
    ucp_request_free(request);
}

void UcxContext::connect_callback(ucp_conn_request_h conn_req, void *arg)
{
    UcxContext *self = reinterpret_cast<UcxContext*>(arg);
    ucp_conn_request_attr_t conn_req_attr;
    conn_req_t conn_request;

    conn_req_attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;
    ucs_status_t status = ucp_conn_request_query(conn_req, &conn_req_attr);
    if (status == UCS_OK) {
        UCX_LOG << "got new connection request " << conn_req << " from client "
                << UcxContext::sockaddr_str((const struct sockaddr*)
                                            &conn_req_attr.client_address,
                                            sizeof(conn_req_attr.client_address));
    } else {
        UCX_LOG << "got new connection request " << conn_req
                << ", ucp_conn_request_query() failed ("
                << ucs_status_string(status) << ")";
    }

    conn_request.conn_request = conn_req;
    gettimeofday(&conn_request.arrival_time, NULL);

    self->_conn_requests.push_back(conn_request);
}

void UcxContext::iomsg_recv_callback(void *request, ucs_status_t status,
                                     ucp_tag_recv_info *info)
{
    ucx_request *r = reinterpret_cast<ucx_request*>(request);
    r->completed   = true;
    r->conn_id     = (info->sender_tag & ~IOMSG_TAG) >> 32;
    r->recv_length = info->length;
}

const std::string UcxContext::sockaddr_str(const struct sockaddr* saddr,
                                           size_t addrlen)
{
    char buf[128];
    int port;

    if (saddr->sa_family != AF_INET) {
        return "<unknown address family>";
    }

    struct sockaddr_storage addr = {0};
    memcpy(&addr, saddr, addrlen);
    switch (addr.ss_family) {
    case AF_INET:
        inet_ntop(AF_INET, &((struct sockaddr_in*)&addr)->sin_addr,
                  buf, sizeof(buf));
        port = ntohs(((struct sockaddr_in*)&addr)->sin_port);
        break;
    case AF_INET6:
        inet_ntop(AF_INET6, &((struct sockaddr_in6*)&addr)->sin6_addr,
                  buf, sizeof(buf));
        port = ntohs(((struct sockaddr_in6*)&addr)->sin6_port);
        break;
    default:
        return "<invalid address>";
    }

    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ":%d", port);
    return buf;
}

ucp_worker_h UcxContext::worker() const
{
    return _worker;
}

double UcxContext::connect_timeout() const
{
    return _connect_timeout;
}

int UcxContext::is_timeout_elapsed(struct timeval const *tv_prior, double timeout)
{
    struct timeval tv_current, elapsed;

    gettimeofday(&tv_current, NULL);
    timersub(&tv_current, tv_prior, &elapsed);
    return ((elapsed.tv_sec + (elapsed.tv_usec * 1e-6)) > timeout);
}

void UcxContext::progress_timed_out_conns()
{
    while (!_conns_in_progress.empty() &&
           (get_time() > _conns_in_progress.begin()->first)) {
        UcxConnection *conn = _conns_in_progress.begin()->second;
        _conns_in_progress.erase(_conns_in_progress.begin());
        conn->handle_connection_error(UCS_ERR_TIMED_OUT);
    }
}

void UcxContext::progress_conn_requests()
{
    while (!_conn_requests.empty()) {
        conn_req_t conn_request = _conn_requests.front();

        if (is_timeout_elapsed(&conn_request.arrival_time, _connect_timeout)) {
            UCX_LOG << "reject connection request " << conn_request.conn_request
                    << " since server's timeout (" << _connect_timeout
                    << " seconds) elapsed";
            ucp_listener_reject(_listener, conn_request.conn_request);
        } else {
            UcxConnection *conn = new UcxConnection(*this, _use_am);
            // Start accepting the connection request, and call
            // UcxAcceptCallback when connection is established
            conn->accept(conn_request.conn_request,
                         new UcxAcceptCallback(*this, *conn));
        }

        _conn_requests.pop_front();
    }
}

void UcxContext::progress_io_message()
{
    if (_use_am || !_iomsg_recv_request->completed) {
        return;
    }

    uint64_t conn_id = _iomsg_recv_request->conn_id;
    conn_map_t::iterator iter = _conns.find(conn_id);
    if (iter == _conns.end()) {
        UCX_LOG << "could not find connection with id " << conn_id;
    } else {
        UcxConnection *conn = iter->second;
        dispatch_io_message(conn, &_iomsg_buffer[0],
                            _iomsg_recv_request->recv_length);
    }
    request_release(_iomsg_recv_request);
    recv_io_message();
}

void UcxContext::progress_failed_connections()
{
    while (!_failed_conns.empty()) {
        UcxConnection *conn = _failed_conns.front();
        _failed_conns.pop_front();
        dispatch_connection_error(conn);
    }
}

void UcxContext::progress_disconnected_connections()
{
    std::list<UcxConnection *>::iterator it = _disconnecting_conns.begin();
    while (it != _disconnecting_conns.end()) {
        UcxConnection *conn = *it;
        if (conn->disconnect_progress()) {
            it = _disconnecting_conns.erase(it);
        } else {
            ++it;
        }
    }
}

UcxContext::wait_status_t
UcxContext::wait_completion(ucs_status_ptr_t status_ptr, const char *title,
                            double timeout)
{
    if (status_ptr == NULL) {
        return WAIT_STATUS_OK;
    } else if (UCS_PTR_IS_PTR(status_ptr)) {
        ucx_request *request = (ucx_request*)UCS_STATUS_PTR(status_ptr);
        ucs_status_t status;
        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        do {
            if (is_timeout_elapsed(&tv_start, timeout)) {
                UCX_LOG << title << " request " << status_ptr << " timed out";
                return WAIT_STATUS_TIMED_OUT;
            }

            ucp_worker_progress(_worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        request_release(request);

        if (status != UCS_OK) {
            UCX_LOG << title << " request " << status_ptr << " failed: " <<
                    ucs_status_string(status);
            return WAIT_STATUS_FAILED;
        } else {
            return WAIT_STATUS_OK;
        }
    } else {
        assert(UCS_PTR_IS_ERR(status_ptr));
        UCX_LOG << title << " operation failed: " <<
                ucs_status_string(UCS_PTR_STATUS(status_ptr));
        return WAIT_STATUS_FAILED;
    }
}

void UcxContext::recv_io_message()
{
    ucs_status_ptr_t status_ptr = ucp_tag_recv_nb(_worker, &_iomsg_buffer[0],
                                                  _iomsg_buffer.size(),
                                                  ucp_dt_make_contig(1),
                                                  IOMSG_TAG, IOMSG_TAG,
                                                  iomsg_recv_callback);
    assert(status_ptr != NULL);
    _iomsg_recv_request = reinterpret_cast<ucx_request*>(status_ptr);
}

void UcxContext::add_connection(UcxConnection *conn)
{
    assert(_conns.find(conn->id()) == _conns.end());
    _conns[conn->id()] = conn;
    UCX_LOG << "added " << conn->get_log_prefix() << " to connection map";
}

void UcxContext::remove_connection(UcxConnection *conn)
{
    conn_map_t::iterator i = _conns.find(conn->id());
    if (i != _conns.end()) {
        _conns.erase(i);
        UCX_LOG << "removed " << conn->get_log_prefix()
                << " from connection map";
    }
}

void UcxContext::remove_connection_inprogress(UcxConnection *conn)
{
    // we expect to remove connections from the list close to the same order
    // as created, so this linear search should be pretty fast
    timeout_conn_t::iterator i;
    for (i = _conns_in_progress.begin(); i != _conns_in_progress.end(); ++i) {
        if (i->second == conn) {
            _conns_in_progress.erase(i);
            return;
        }
    }
}

void UcxContext::move_connection_to_disconnecting(UcxConnection *conn)
{
    remove_connection(conn);
    assert(std::find(_disconnecting_conns.begin(), _disconnecting_conns.end(),
                     conn) == _disconnecting_conns.end());
    _disconnecting_conns.push_back(conn);
}

void UcxContext::dispatch_connection_accepted(UcxConnection* conn)
{
}

void UcxContext::handle_connection_error(UcxConnection *conn)
{
    remove_connection(conn);
    remove_connection_inprogress(conn);
    _failed_conns.push_back(conn);
}

void UcxContext::destroy_connections()
{
    while (!_conn_requests.empty()) {
        ucp_conn_request_h conn_req = _conn_requests.front().conn_request;
        UCX_LOG << "reject connection request " << conn_req;
        ucp_listener_reject(_listener, conn_req);
        _conn_requests.pop_front();
    }

    while (!_conns_in_progress.empty()) {
        UcxConnection &conn = *_conns_in_progress.begin()->second;
        _conns_in_progress.erase(_conns_in_progress.begin());
        conn.disconnect(new UcxDisconnectCallback(conn));
    }

    UCX_LOG << "destroy_connections";
    while (!_conns.empty()) {
        UcxConnection &conn = *_conns.begin()->second;
        _conns.erase(_conns.begin());
        conn.disconnect(new UcxDisconnectCallback(conn));
    }

    while (!_disconnecting_conns.empty()) {
        ucp_worker_progress(_worker);
        progress_disconnected_connections();
    }
}

double UcxContext::get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + (tv.tv_usec * 1e-6);
}

void UcxContext::destroy_listener()
{
    if (_listener) {
        ucp_listener_destroy(_listener);
    }
}

void UcxContext::destroy_worker()
{
    if (!_worker) {
        return;
    }

    if (_iomsg_recv_request != NULL) {
        ucp_request_cancel(_worker, _iomsg_recv_request);
        wait_completion(_iomsg_recv_request, "iomsg receive");
    }

    ucp_worker_destroy(_worker);
}

ucs_status_t UcxContext::am_recv_callback(void *arg, const void *header,
                                          size_t header_length,
                                          void *data, size_t length,
                                          const ucp_am_recv_param_t *param)
{
    UcxContext *self = reinterpret_cast<UcxContext*>(arg);

    assert(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP);
    assert(self->_use_am);

    uint64_t conn_id = reinterpret_cast<uint64_t>(param->reply_ep);
    conn_map_t::iterator iter = self->_conns.find(conn_id);
    if (iter == self->_conns.end()) {
        // TODO: change this to assert when data dropping is implemented in AM
        UCX_LOG << "could not find connection with ep " << param->reply_ep
                << "(" << conn_id << ")";
        return UCS_OK;
    }

    UcxConnection *conn = iter->second;

    UcxAmDesc data_desc(data, param);

    self->dispatch_am_message(conn, header, header_length, data_desc);

    return UCS_OK;
}

void UcxContext::set_am_handler(ucp_am_recv_callback_t cb, void *arg)
{
    ucp_am_handler_param_t param;

    param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                       UCP_AM_HANDLER_PARAM_FIELD_CB |
                       UCP_AM_HANDLER_PARAM_FIELD_ARG;
    param.id         = AM_MSG_ID;
    param.cb         = cb;
    param.arg        = arg;
    ucp_worker_set_am_recv_handler(_worker, &param);
}

#define UCX_CONN_LOG UcxLog(_log_prefix, true)

unsigned UcxConnection::_num_instances = 0;

UcxConnection::UcxConnection(UcxContext &context, bool use_am) :
    _context(context),
    _establish_cb(NULL),
    _disconnect_cb(NULL),
    _conn_id(context.get_next_conn_id()),
    _remote_conn_id(0),
    _ep(NULL),
    _close_request(NULL),
    _ucx_status(UCS_INPROGRESS),
    _use_am(use_am)
{
    ++_num_instances;
    struct sockaddr_in in_addr = {0};
    in_addr.sin_family         = AF_INET;
    set_log_prefix((const struct sockaddr*)&in_addr, sizeof(in_addr));
    ucs_list_head_init(&_all_requests);
    UCX_CONN_LOG << "created new connection " << this << " total: " << _num_instances;
}

UcxConnection::~UcxConnection()
{
    /* establish cb must be destroyed earlier since it accesses
     * the connection */
    assert(_establish_cb == NULL);
    assert(_disconnect_cb == NULL);
    assert(_ep == NULL);
    assert(ucs_list_is_empty(&_all_requests));
    assert(!UCS_PTR_IS_PTR(_close_request));

    UCX_CONN_LOG << "released";
    --_num_instances;
}

void UcxConnection::connect(const struct sockaddr *saddr, socklen_t addrlen,
                            UcxCallback *callback)
{
    set_log_prefix(saddr, addrlen);

    ucp_ep_params_t ep_params;
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = saddr;
    ep_params.sockaddr.addrlen = addrlen;

    char sockaddr_str[UCS_SOCKADDR_STRING_LEN];
    UCX_CONN_LOG << "Connecting to "
                 << ucs_sockaddr_str(saddr, sockaddr_str,
                                     UCS_SOCKADDR_STRING_LEN);
    connect_common(ep_params, callback);
}

void UcxConnection::accept(ucp_conn_request_h conn_req, UcxCallback *callback)
{
    ucp_conn_request_attr_t conn_req_attr;
    conn_req_attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;

    ucs_status_t status = ucp_conn_request_query(conn_req, &conn_req_attr);
    if (status == UCS_OK) {
        set_log_prefix((const struct sockaddr*)&conn_req_attr.client_address,
                       sizeof(conn_req_attr.client_address));
    } else {
        UCX_CONN_LOG << "ucp_conn_request_query() failed: " << ucs_status_string(status);
    }

    ucp_ep_params_t ep_params;
    ep_params.field_mask   = UCP_EP_PARAM_FIELD_CONN_REQUEST;
    ep_params.conn_request = conn_req;
    connect_common(ep_params, callback);
}

void UcxConnection::disconnect(UcxCallback *callback)
{
    /* establish cb must be destroyed earlier since it accesses
     * the connection */
    assert(_establish_cb == NULL);
    assert(_disconnect_cb == NULL);
    assert(_ep != NULL);

    UCX_CONN_LOG << "destroying, ep is " << _ep;
    ep_close(UCP_EP_CLOSE_MODE_FORCE);

    _disconnect_cb = callback;
    if (ucs_list_is_empty(&_all_requests)) {
        _context.move_connection_to_disconnecting(this);
    } else {
        cancel_all();
    }
}

bool UcxConnection::disconnect_progress()
{
    assert(_ep == NULL);
    assert(_disconnect_cb != NULL);

    if (UCS_PTR_IS_PTR(_close_request)) {
        if (ucp_request_check_status(_close_request) == UCS_INPROGRESS) {
            return false;
        } else {
            ucp_request_free(_close_request);
            _close_request = NULL;
        }
    }

    assert(ucs_list_is_empty(&_all_requests));
    UcxCallback *cb = _disconnect_cb;
    _disconnect_cb  = NULL;
    // invoke last since it can delete this object
    (*cb)(UCS_OK);
    return true;
}

bool UcxConnection::send_io_message(const void *buffer, size_t length,
                                    UcxCallback* callback)
{
    ucp_tag_t tag = make_iomsg_tag(_remote_conn_id, 0);
    return send_common(buffer, length, tag, callback);
}

bool UcxConnection::send_data(const void *buffer, size_t length, uint32_t sn,
                              UcxCallback* callback)
{
    ucp_tag_t tag = make_data_tag(_remote_conn_id, sn);
    return send_common(buffer, length, tag, callback);
}

bool UcxConnection::recv_data(void *buffer, size_t length, uint32_t sn,
                              UcxCallback* callback)
{
    if (_ep == NULL) {
        return false;
    }

    ucp_tag_t tag      = make_data_tag(_conn_id, sn);
    ucp_tag_t tag_mask = std::numeric_limits<ucp_tag_t>::max();
    ucs_status_ptr_t ptr_status = ucp_tag_recv_nb(_context.worker(), buffer,
                                                  length, ucp_dt_make_contig(1),
                                                  tag, tag_mask,
                                                  data_recv_callback);
    return process_request("ucp_tag_recv_nb", ptr_status, callback);
}

bool UcxConnection::send_am(const void *meta, size_t meta_length,
                            const void *buffer, size_t length,
                            UcxCallback* callback)
{
    if (_ep == NULL) {
        return false;
    }

    ucp_request_param_t param;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK  |
                         UCP_OP_ATTR_FIELD_FLAGS;
    param.cb.send      = common_request_callback_nbx;
    param.flags        = UCP_AM_SEND_REPLY;
    param.datatype     = 0; // make coverity happy

    ucs_status_ptr_t sptr = ucp_am_send_nbx(_ep, AM_MSG_ID, meta, meta_length,
                                            buffer, length, &param);
    return process_request("ucp_am_send_nbx", sptr, callback);
}

bool UcxConnection::recv_am_data(void *buffer, size_t length,
                                 const UcxAmDesc &data_desc,
                                 UcxCallback* callback)
{
    assert(_ep != NULL);

    if (!(data_desc._param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV)) {
        memcpy(buffer, data_desc._data, length);
        (*callback)(UCS_OK);
        return true;
    }

    ucp_request_param_t params;
    params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK  |
                          UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
    params.cb.recv_am   = am_data_recv_callback;
    ucs_status_ptr_t sp = ucp_am_recv_data_nbx(_context.worker(),
                                               data_desc._data,
                                               buffer, length, &params);
    return process_request("ucp_am_recv_data_nbx", sp, callback);
}

void UcxConnection::cancel_all()
{
    if (ucs_list_is_empty(&_all_requests)) {
        return;
    }

    ucx_request *request, *tmp;
    unsigned     count = 0;
    ucs_list_for_each_safe(request, tmp, &_all_requests, pos) {
        ++count;
        UCX_CONN_LOG << "canceling " << request << " request #" << count;
        ucp_request_cancel(_context.worker(), request);
    }
}

ucp_tag_t UcxConnection::make_data_tag(uint32_t conn_id, uint32_t sn)
{
    return (static_cast<uint64_t>(conn_id) << 32) | sn;
}

ucp_tag_t UcxConnection::make_iomsg_tag(uint32_t conn_id, uint32_t sn)
{
    return UcxContext::IOMSG_TAG | make_data_tag(conn_id, sn);
}

void UcxConnection::stream_send_callback(void *request, ucs_status_t status)
{
}

void UcxConnection::stream_recv_callback(void *request, ucs_status_t status,
                                         size_t recv_len)
{
    ucx_request *r      = reinterpret_cast<ucx_request*>(request);
    UcxConnection *conn = r->conn;

    if (!conn->is_established()) {
        assert(conn->_establish_cb == r->callback);
        conn->established(status);
    } else {
        assert(UCS_STATUS_IS_ERR(conn->ucx_status()));
    }

    conn->request_completed(r);
    UcxContext::request_release(r);
}

void UcxConnection::common_request_callback(void *request, ucs_status_t status)
{
    ucx_request *r = reinterpret_cast<ucx_request*>(request);

    assert(!r->completed);
    r->status = status;

    if (r->callback) {
        // already processed by send/recv function
        (*r->callback)(status);
        r->conn->request_completed(r);
        UcxContext::request_release(r);
    } else {
        // not yet processed by "process_request"
        r->completed = true;
    }
}

void UcxConnection::data_recv_callback(void *request, ucs_status_t status,
                                       ucp_tag_recv_info *info)
{
    common_request_callback(request, status);
}

void UcxConnection::common_request_callback_nbx(void *request,
                                                ucs_status_t status,
                                                void *user_data)
{
    common_request_callback(request, status);
}

void UcxConnection::am_data_recv_callback(void *request, ucs_status_t status,
                                          size_t length, void *user_data)
{
    common_request_callback(request, status);
}

void UcxConnection::error_callback(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    reinterpret_cast<UcxConnection*>(arg)->handle_connection_error(status);
}

void UcxConnection::set_log_prefix(const struct sockaddr* saddr,
                                   socklen_t addrlen)
{
    std::stringstream ss;
    ss << "[UCX-connection #" << _conn_id << " " <<
          UcxContext::sockaddr_str(saddr, addrlen) << "]";
    memset(_log_prefix, 0, MAX_LOG_PREFIX_SIZE);
    int length = ss.str().length();
    if (length >= MAX_LOG_PREFIX_SIZE) {
        length = MAX_LOG_PREFIX_SIZE - 1;
    }
    memcpy(_log_prefix, ss.str().c_str(), length);
}

void UcxConnection::connect_tag(UcxCallback *callback)
{
    const ucp_datatype_t dt_int = ucp_dt_make_contig(sizeof(uint32_t));
    size_t recv_len;

    // receive remote connection id
    void *rreq = ucp_stream_recv_nb(_ep, &_remote_conn_id, 1, dt_int,
                                    stream_recv_callback, &recv_len,
                                    UCP_STREAM_RECV_FLAG_WAITALL);
    if (UCS_PTR_IS_PTR(rreq)) {
        process_request("conn_id receive", rreq, callback);
        _context._conns_in_progress.push_back(std::make_pair(
                UcxContext::get_time() + _context._connect_timeout, this));
    } else {
        established(UCS_PTR_STATUS(rreq));
        if (rreq != NULL) {
            // failed to receive
            return;
        }
    }

    // send local connection id
    void *sreq = ucp_stream_send_nb(_ep, &_conn_id, 1, dt_int,
                                    stream_send_callback, 0);
    // we do not have to check the status here, in case if the endpoint is
    // failed we should handle it in ep_params.err_handler.cb set above
    if (UCS_PTR_IS_PTR(sreq)) {
        ucp_request_free(sreq);
    }
}

void UcxConnection::connect_am(UcxCallback *callback)
{
    // With AM use ep as a connection ID. AM receive callback provides
    // reply ep, which can be used for finding a proper connection.
    _conn_id = reinterpret_cast<uint64_t>(_ep);
    established(UCS_OK);
}

void UcxConnection::connect_common(ucp_ep_params_t &ep_params,
                                   UcxCallback *callback)
{
    _establish_cb = callback;

    // create endpoint
    ep_params.field_mask      |= UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = error_callback;
    ep_params.err_handler.arg  = reinterpret_cast<void*>(this);

    ucs_status_t status = ucp_ep_create(_context.worker(), &ep_params, &_ep);
    if (status != UCS_OK) {
        assert(_ep == NULL);
        UCX_LOG << "ucp_ep_create() failed: " << ucs_status_string(status);
        handle_connection_error(status);
        return;
    }

    UCX_CONN_LOG << "created endpoint " << _ep << ", connection id "
                 << _conn_id;

    if (_use_am) {
        connect_am(callback);
    } else {
        connect_tag(callback);
    }

    _context.add_connection(this);
}

void UcxConnection::established(ucs_status_t status)
{
    if (!_use_am && (status == UCS_OK)) {
        assert(_remote_conn_id != 0);
        UCX_CONN_LOG << "Remote id is " << _remote_conn_id;
    }

    _ucx_status = status;
    _context.remove_connection_inprogress(this);

    (*_establish_cb)(status);
    _establish_cb = NULL;
}

bool UcxConnection::send_common(const void *buffer, size_t length, ucp_tag_t tag,
                                UcxCallback* callback)
{
    if (_ep == NULL) {
        return false;
    }

    ucs_status_ptr_t ptr_status = ucp_tag_send_nb(_ep, buffer, length,
                                                  ucp_dt_make_contig(1), tag,
                                                  common_request_callback);
    return process_request("ucp_tag_send_nb", ptr_status, callback);
}

void UcxConnection::request_started(ucx_request *r)
{
    ucs_list_add_tail(&_all_requests, &r->pos);
}

void UcxConnection::request_completed(ucx_request *r)
{
    assert(r->conn == this);
    ucs_list_del(&r->pos);

    if (_disconnect_cb != NULL) {
        UCX_CONN_LOG << "completing request " << r << " with status \""
                     << ucs_status_string(r->status) << "\" (" << r->status
                     << ")" << " during disconnect";

        if (ucs_list_is_empty(&_all_requests)) {
            _context.move_connection_to_disconnecting(this);
        }
    }
}

void UcxConnection::handle_connection_error(ucs_status_t status)
{
    if (UCS_STATUS_IS_ERR(_ucx_status)) {
        return;
    }

    UCX_CONN_LOG << "detected error: " << ucs_status_string(status);
    _ucx_status = status;

    /* the upper layer should close the connection */
    if (is_established()) {
        _context.handle_connection_error(this);
    } else {
        (*_establish_cb)(status);
        _establish_cb = NULL;
    }
}

void UcxConnection::ep_close(enum ucp_ep_close_mode mode)
{
    static const char *mode_str[] = {"force", "flush"};
    if (_ep == NULL) {
        /* already closed */
        return;
    }

    assert(!_close_request);

    UCX_CONN_LOG << "closing ep " << _ep << " mode " << mode_str[mode];
    _close_request = ucp_ep_close_nb(_ep, mode);
    _ep            = NULL;
}

bool UcxConnection::process_request(const char *what,
                                    ucs_status_ptr_t ptr_status,
                                    UcxCallback* callback)
{
    ucs_status_t status;

    if (ptr_status == NULL) {
        (*callback)(UCS_OK);
        return true;
    } else if (UCS_PTR_IS_ERR(ptr_status)) {
        status = UCS_PTR_STATUS(ptr_status);
        UCX_CONN_LOG << what << " failed with status: "
                     << ucs_status_string(status);
        (*callback)(status);
        return false;
    } else {
        // pointer to request
        ucx_request *r = reinterpret_cast<ucx_request*>(ptr_status);
        if (r->completed) {
            // already completed by callback
            assert(ucp_request_is_completed(r));
            status = r->status;
            (*callback)(status);
            UcxContext::request_release(r);
            return status == UCS_OK;
        } else {
            // will be completed by callback
            r->callback = callback;
            r->conn     = this;
            request_started(r);
            return true;
        }
    }
}
