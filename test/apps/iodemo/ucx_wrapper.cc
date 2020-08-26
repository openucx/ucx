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

#include <limits>


struct ucx_request {
    UcxCallback                  *callback;
    UcxConnection                *conn;
    ucs_status_t                 status;
    bool                         completed;
    uint32_t                     conn_id;
    size_t                       recv_length;
    ucs_list_link_t              pos;
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

UcxLog::UcxLog(const char* prefix, bool enable) : _enable(enable)
{
    if (enable) {
        std::cout << prefix << " ";
    }
}

UcxLog::~UcxLog()
{
    if (_enable) {
        std::cout << std::endl;
    }
}

#define UCX_LOG UcxLog("[UCX]", true)

UcxContext::UcxContext(size_t iomsg_size) :
    _context(NULL), _worker(NULL), _listener(NULL), _iomsg_recv_request(NULL),
    _iomsg_buffer(iomsg_size, '\0')
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
    ucp_params.features     = UCP_FEATURE_TAG |
                              UCP_FEATURE_STREAM;
    ucp_params.request_init = request_init;
    ucp_params.request_size = sizeof(ucx_request);
    ucs_status_t status = ucp_init(&ucp_params, NULL, &_context);
    if (status != UCS_OK) {
        UCX_LOG << "ucp_init() failed: " << ucs_status_string(status);
        return false;
    }

    UCX_LOG << "created context " << _context;

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

    recv_io_message();
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

UcxConnection* UcxContext::connect(const struct sockaddr* saddr, size_t addrlen)
{
    UcxConnection *conn = new UcxConnection(*this, get_next_conn_id());
    if (!conn->connect(saddr, addrlen)) {
        delete conn;
        return NULL;
    }

    add_connection(conn);
    return conn;
}

void UcxContext::progress()
{
    ucp_worker_progress(_worker);
    progress_io_message();
    progress_conn_requests();
    progress_failed_connections();
}

void UcxContext::dispatch_new_connection(UcxConnection *conn)
{
    // To be implemented in a subclass
}

void UcxContext::dispatch_io_message(UcxConnection* conn, const void *buffer,
                                     size_t length)
{
    // To be implemented in a subclass
}

void UcxContext::dispatch_connection_error(UcxConnection* conn)
{
    // To be implemented in a subclass
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
    UCX_LOG << "got new connection request " << conn_req;
    self->_conn_requests.push_back(conn_req);
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

    inet_ntop(AF_INET, &((struct sockaddr_in*)&addr)->sin_addr,
              buf, sizeof(buf));
    port = ntohs(((struct sockaddr_in*)&addr)->sin_port);

    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ":%d", port);
    return buf;
}

ucp_worker_h UcxContext::worker() const
{
    return _worker;
}

void UcxContext::progress_conn_requests()
{
    while (!_conn_requests.empty()) {
        UcxConnection *conn = new UcxConnection(*this, get_next_conn_id());
        if (conn->accept(_conn_requests.front())) {
            add_connection(conn);
            dispatch_new_connection(conn);
        } else {
            delete conn;
        }
        _conn_requests.pop_front();
    }
}

void UcxContext::progress_io_message()
{
    if (!_iomsg_recv_request->completed) {
        return;
    }

    uint32_t conn_id = _iomsg_recv_request->conn_id;
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

UcxContext::wait_status_t
UcxContext::wait_completion(ucs_status_ptr_t status_ptr, double timeout)
{
    if (status_ptr == NULL) {
        return WAIT_STATUS_OK;
    } else if (UCS_PTR_IS_PTR(status_ptr)) {
        ucx_request *request = (ucx_request*)UCS_STATUS_PTR(status_ptr);
        ucs_status_t status;
        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        do {
            struct timeval tv_current, elapsed;
            gettimeofday(&tv_current, NULL);
            timersub(&tv_current, &tv_start, &elapsed);
            if (elapsed.tv_sec + (elapsed.tv_usec * 1e-6) > timeout) {
                return WAIT_STATUS_TIMED_OUT;
            }

            ucp_worker_progress(_worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        request_release(request);
        return (status == UCS_OK) ? WAIT_STATUS_OK : WAIT_STATUS_FAILED;
    } else {
        assert(UCS_PTR_IS_ERR(status_ptr));
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
    _conns[conn->id()] = conn;
}

void UcxContext::remove_connection(UcxConnection *conn)
{
    _conns.erase(conn->id());
}

void UcxContext::handle_connection_error(UcxConnection *conn)
{
    remove_connection(conn);
    _failed_conns.push_back(conn);
}

void UcxContext::destroy_connections()
{
    while (!_conn_requests.empty()) {
        UCX_LOG << "reject connection request " << _conn_requests.front();
        ucp_listener_reject(_listener, _conn_requests.front());
        _conn_requests.pop_front();
    }

    for (conn_map_t::iterator iter = _conns.begin(); iter != _conns.end(); ++iter) {
        delete iter->second;

    }
    _conns.clear();
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
        wait_completion(_iomsg_recv_request);
    }

    ucp_worker_destroy(_worker);
}


#define UCX_CONN_LOG UcxLog(_log_prefix, true)

unsigned UcxConnection::_num_instances = 0;

UcxConnection::UcxConnection(UcxContext &context, uint32_t conn_id) :
    _context(context), _conn_id(conn_id), _remote_conn_id(0),
    _ep(0), _close_request(NULL)
{
    ++_num_instances;
    struct sockaddr_in in_addr = {0};
    in_addr.sin_family = AF_INET;
    set_log_prefix((const struct sockaddr*)&in_addr, sizeof(in_addr));
    ucs_list_head_init(&_all_requests);
    UCX_CONN_LOG << "created new connection, total: " << _num_instances;
}

UcxConnection::~UcxConnection()
{
    // if _ep is NULL, connection was closed and removed by error handler
    if (_ep != NULL) {
        disconnect(UCP_EP_CLOSE_MODE_FORCE);
    }

    if (_close_request) {
        _context.wait_completion(_close_request);
    }

    // wait until all requests are completed
    if (!ucs_list_is_empty(&_all_requests)) {
        UCX_CONN_LOG << "waiting for " << ucs_list_length(&_all_requests) <<
                        " uncompleted requests";
    }
    while (!ucs_list_is_empty(&_all_requests)) {
        ucp_worker_progress(_context.worker());
    }

    UCX_CONN_LOG << "released";
    --_num_instances;
}

bool UcxConnection::connect(const struct sockaddr* saddr, socklen_t addrlen)
{
    set_log_prefix(saddr, addrlen);

    ucp_ep_params_t ep_params;
    ep_params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS       |
                                 UCP_EP_PARAM_FIELD_SOCK_ADDR;
    ep_params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    ep_params.sockaddr.addr    = saddr;
    ep_params.sockaddr.addrlen = addrlen;

    return connect_common(ep_params);
}

bool UcxConnection::accept(ucp_conn_request_h conn_req)
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

    return connect_common(ep_params);
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

void UcxConnection::cancel_all()
{
    if (ucs_list_is_empty(&_all_requests)) {
        return;
    }

    ucx_request *request, *tmp;
    unsigned     count = 0;
    ucs_list_for_each_safe(request, tmp, &_all_requests, pos) {
        ucp_request_cancel(_context.worker(), request);
        ++count;
    }

    UCX_CONN_LOG << "canceling " << count << " requests ";
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
}

void UcxConnection::common_request_callback(void *request, ucs_status_t status)
{
    ucx_request *r = reinterpret_cast<ucx_request*>(request);

    assert(!r->completed);
    if (r->callback) {
        // already processed by send function
        (*r->callback)(status);
        r->conn->request_completed(r);
        UcxContext::request_release(r);
    } else {
        // not yet processed by "process_request"
        r->completed = true;
        r->status    = status;
    }
}

void UcxConnection::data_recv_callback(void *request, ucs_status_t status,
                                       ucp_tag_recv_info *info)
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

bool UcxConnection::connect_common(ucp_ep_params_t& ep_params)
{
    UcxContext::wait_status_t wait_status;

    // create endpoint
    ep_params.field_mask      |= UCP_EP_PARAM_FIELD_ERR_HANDLER |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    ep_params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;
    ep_params.err_handler.cb   = error_callback;
    ep_params.err_handler.arg  = reinterpret_cast<void*>(this);

    ucs_status_t status = ucp_ep_create(_context.worker(), &ep_params, &_ep);
    if (status != UCS_OK) {
        UCX_LOG << "ucp_ep_create() failed: " << ucs_status_string(status);
        return false;
    }

    UCX_CONN_LOG << "created endpoint " << _ep << ", exchanging connection id";

    const ucp_datatype_t dt_int = ucp_dt_make_contig(sizeof(uint32_t));

    // receive remote connection id
    size_t recv_len;
    void *rreq = ucp_stream_recv_nb(_ep, &_remote_conn_id, 1, dt_int,
                                    stream_recv_callback, &recv_len,
                                    UCP_STREAM_RECV_FLAG_WAITALL);

    // send local connection id
    void *sreq = ucp_stream_send_nb(_ep, &_conn_id, 1, dt_int,
                                    stream_send_callback, 0);
    wait_status = _context.wait_completion(sreq, 5);
    if (wait_status != UcxContext::WAIT_STATUS_OK) {
        UCX_CONN_LOG << "failed to send remote connection id";
        ep_close(UCP_EP_CLOSE_MODE_FORCE);
        if (wait_status == UcxContext::WAIT_STATUS_TIMED_OUT) {
            _context.wait_completion(sreq);
        }
        // wait for receive request as well, which should be canceled by ep close
        _context.wait_completion(rreq);
        return false;
    }

    // wait to complete receiving remote connection id
    wait_status = _context.wait_completion(rreq, 5);
    if (wait_status != UcxContext::WAIT_STATUS_OK) {
        UCX_CONN_LOG << "failed to receive remote connection id";
        ep_close(UCP_EP_CLOSE_MODE_FORCE);
        if (wait_status == UcxContext::WAIT_STATUS_TIMED_OUT) {
            _context.wait_completion(rreq);
        }
        return false;
    }

    UCX_CONN_LOG << "remote id is " << _remote_conn_id;
    return true;
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
}

void UcxConnection::handle_connection_error(ucs_status_t status)
{
    UCX_CONN_LOG << "detected error: " << ucs_status_string(status);

    if (_remote_conn_id) {
        disconnect(UCP_EP_CLOSE_MODE_FORCE);
        _context.handle_connection_error(this);
    }
}

void UcxConnection::disconnect(enum ucp_ep_close_mode mode)
{
    _context.remove_connection(this);
    cancel_all();
    ep_close(mode);
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
        UCX_CONN_LOG << what << "failed with status: "
                     << ucs_status_string(status);
        (*callback)(status);
        return false;
    } else {
        // pointer to request
        ucx_request *r = reinterpret_cast<ucx_request*>(ptr_status);
        if (r->completed) {
            // already completed by callback
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


