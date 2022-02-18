/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef IODEMO_UCX_WRAPPER_H_
#define IODEMO_UCX_WRAPPER_H_

#include <ucp/api/ucp.h>
#include <ucs/algorithm/crc.h>
#include <ucs/datastruct/list.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sock.h>
#include <deque>
#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <sys/epoll.h>

#define MAX_LOG_PREFIX_SIZE   64

/* Forward declarations */
class UcxConnection;
struct ucx_request;

// Holds details of arrived AM message
struct UcxAmDesc {
    UcxAmDesc(void *data, const ucp_am_recv_param_t *param) :
        _data(data), _param(param) {
    }

    void                         *_data;
    const ucp_am_recv_param_t    *_param;
};

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
    static bool use_human_time;

    UcxLog(const char* prefix, bool enable = true,
           std::ostream *os = &std::cout, bool abort = false);
    ~UcxLog();

    template<typename T>
    UcxLog& operator<<(const T &t) {
        if (_ss != NULL) {
            (*_ss) << t;
        }
        return *this;
    }

private:
    std::stringstream        *_ss;
    std::ostream             *_os;
    bool                     _abort;
};


/**
 * Holds UCX global context and worker
 */
class UcxContext {
    class UcxAcceptCallback : public UcxCallback {
    public:
        UcxAcceptCallback(UcxContext &context, UcxConnection &connection);

        virtual void operator()(ucs_status_t status);

    private:
        UcxContext    &_context;
        UcxConnection &_connection;
    };

protected:
    class UcxDisconnectCallback : public UcxCallback {
    public:
        virtual void operator()(ucs_status_t status);
    };

public:
    typedef std::vector<uint8_t> iomsg_buffer_t;

    UcxContext(size_t iomsg_size, double connect_timeout, bool use_am,
               bool use_epoll = false);

    virtual ~UcxContext();

    bool init();

    bool listen(const struct sockaddr* saddr, size_t addrlen);

    void progress(unsigned count = 1);

    static const std::string sockaddr_str(const struct sockaddr* saddr,
                                          size_t addrlen);

    static double get_time();

    static void *malloc(size_t size, const char *name);

    static void *memalign(size_t alignment, size_t size, const char *name);

    static void free(void *ptr);

    bool map_buffer(size_t length, void *address, ucp_mem_h *memh);

    bool unmap_buffer(ucp_mem_h memh);

protected:

    // Called when new IO message is received
    virtual void dispatch_io_message(UcxConnection* conn, const void *buffer,
                                     size_t length) = 0;

    // Called when new AM message is received
    // (note IO message can be bundled with data)
    virtual void dispatch_am_message(UcxConnection* conn, const void *hdr,
                                     size_t hdr_length,
                                     const UcxAmDesc &data_desc) = 0;

    // Called when there is a fatal failure on the connection
    virtual void dispatch_connection_error(UcxConnection* conn) = 0;

    // Called when new server connection is accepted
    virtual void dispatch_connection_accepted(UcxConnection* conn);

    void destroy_connections();

    void wait_disconnected_connections();

    void destroy_listener();

    static inline void *ucx_am_get_data(const UcxAmDesc &desc)
    {
        return desc._data;
    }

    static inline bool ucx_am_is_rndv(const UcxAmDesc &desc)
    {
        return desc._param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV;
    }

private:
    typedef enum {
        WAIT_STATUS_OK,
        WAIT_STATUS_FAILED,
        WAIT_STATUS_TIMED_OUT
    } wait_status_t;

    typedef struct {
        ucp_conn_request_h conn_request;
        struct timeval     arrival_time;
    } conn_req_t;

    typedef std::map<uint64_t, UcxConnection*> conn_map_t;

    typedef std::vector<std::pair<double, UcxConnection*> > timeout_conn_t;

    friend class UcxConnection;

    static const ucp_tag_t IOMSG_TAG = 1ull << 63;

    static uint32_t get_next_conn_id();

    static void request_init(void *request);

    static void request_reset(ucx_request *r);

    static void request_release(void *request);

    static void connect_callback(ucp_conn_request_h conn_req, void *arg);

    static void iomsg_recv_callback(void *request, ucs_status_t status,
                                    ucp_tag_recv_info *info);

    static ucs_status_t am_recv_callback(void *arg, const void *header,
                                         size_t header_length,
                                         void *data, size_t length,
                                         const ucp_am_recv_param_t *param);

    ucp_worker_h worker() const;

    double connect_timeout() const;

    int is_timeout_elapsed(struct timeval const *tv_prior, double timeout);

    ucs_status_t epoll_init();

    bool progress_worker_event();

    void progress_timed_out_conns();

    void progress_conn_requests();

    void progress_io_message();

    void progress_failed_connections();

    void progress_disconnected_connections();

    wait_status_t wait_completion(ucs_status_ptr_t status_ptr, const char *title,
                                  double timeout = 1e6);

    void recv_io_message();

    void add_connection(UcxConnection *conn);

    void remove_connection(UcxConnection *conn);

    timeout_conn_t::iterator find_connection_inprogress(UcxConnection *conn);

    void remove_connection_inprogress(UcxConnection *conn);

    void move_connection_to_disconnecting(UcxConnection *conn);

    bool is_in_disconnecting_list(UcxConnection *conn)
    {
        return std::find(_disconnecting_conns.begin(),
                         _disconnecting_conns.end(), conn) !=
                _disconnecting_conns.end();
    }

    void handle_connection_error(UcxConnection *conn);

    void destroy_worker();

    void set_am_handler(ucp_am_recv_callback_t cb, void *arg);

    ucp_context_h               _context;
    ucp_worker_h                _worker;
    ucp_listener_h              _listener;
    conn_map_t                  _conns;
    std::deque<conn_req_t>      _conn_requests;
    timeout_conn_t              _conns_in_progress; // ordered in time
    std::deque<UcxConnection *> _failed_conns;
    std::list<UcxConnection *>  _disconnecting_conns;
    ucx_request                 *_iomsg_recv_request;
    iomsg_buffer_t              _iomsg_buffer;
    double                      _connect_timeout;
    bool                        _use_am;
    int                         _worker_fd;
    int                         _epoll_fd;
};


class UcxConnection {
public:
    UcxConnection(UcxContext &context, bool use_am);

    ~UcxConnection();

    void connect(const struct sockaddr *src_saddr,
                 const struct sockaddr *dst_saddr,
                 socklen_t addrlen,
                 UcxCallback *callback);

    void accept(ucp_conn_request_h conn_req, UcxCallback *callback);

    /**
     * The connection will be destroyed automatically after callback is called.
     */
    void disconnect(UcxCallback *callback);

    bool disconnect_progress();

    bool send_io_message(const void *buffer, size_t length,
                         UcxCallback* callback = EmptyCallback::get());

    bool send_data(const void *buffer, size_t length, ucp_mem_h memh,
                   uint32_t sn, UcxCallback *callback = EmptyCallback::get());

    bool recv_data(void *buffer, size_t length, ucp_mem_h memh, uint32_t sn,
                   UcxCallback *callback = EmptyCallback::get());

    bool send_am(const void *meta, size_t meta_length, const void *buffer,
                 size_t length, ucp_mem_h memh,
                 UcxCallback *callback = EmptyCallback::get());

    bool recv_am_data(void *buffer, size_t length, ucp_mem_h memh,
                      const UcxAmDesc &data_desc,
                      UcxCallback *callback = EmptyCallback::get());

    void iomsg_recv_defer(const UcxContext::iomsg_buffer_t &iomsg,
                          size_t iomsg_length);

    void cancel_all();

    uint64_t id() const {
        return _conn_id;
    }

    ucs_status_t ucx_status() const {
        return _ucx_status;
    }

    const char* get_log_prefix() const {
        return _log_prefix;
    }

    bool is_established() const {
        return _establish_cb == NULL;
    }

    const std::string& get_peer_name() const {
        return _remote_address;
    }

    bool is_disconnecting() const {
        return _disconnect_cb != NULL;
    }

    void handle_connection_error(ucs_status_t status);

    static size_t get_num_instances() {
        return _num_instances;
    }

private:
    static ucp_tag_t make_data_tag(uint32_t conn_id, uint32_t sn);

    static ucp_tag_t make_iomsg_tag(uint32_t conn_id, uint32_t sn);

    static void stream_send_callback(void *request, ucs_status_t status);

    static void stream_recv_callback(void *request, ucs_status_t status,
                                     size_t recv_len);

    static void common_request_callback(void *request, ucs_status_t status);

    static void common_request_callback_nbx(void *request, ucs_status_t status,
                                            void *user_data);

    static void am_data_recv_callback(void *request, ucs_status_t status,
                                      size_t length, void *user_data);

    static void data_recv_callback(void *request, ucs_status_t status,
                                   ucp_tag_recv_info *info);

    static void error_callback(void *arg, ucp_ep_h ep, ucs_status_t status);

    void set_log_prefix(const struct sockaddr* saddr, socklen_t addrlen);

    void print_addresses();

    void connect_common(ucp_ep_params_t &ep_params, UcxCallback *callback);

    void connect_tag(UcxCallback *callback);

    void connect_am(UcxCallback *callback);

    void established(ucs_status_t status);

    bool send_common(const void *buffer, size_t length, ucp_mem_h memh,
                     ucp_tag_t tag, UcxCallback *callback);

    void request_started(ucx_request *r);

    void request_completed(ucx_request *r);

    void ep_close(enum ucp_ep_close_mode mode);

    bool process_request(const char *what, ucs_status_ptr_t ptr_status,
                         UcxCallback* callback);

    static void invoke_callback(UcxCallback *&cb, ucs_status_t status);

    static unsigned _num_instances;

    UcxContext                             &_context;
    UcxCallback                            *_establish_cb;
    UcxCallback                            *_disconnect_cb;
    uint64_t                               _conn_id;
    uint64_t                               _remote_conn_id;
    char                                   _log_prefix[MAX_LOG_PREFIX_SIZE];
    ucp_ep_h                               _ep;
    std::string                            _remote_address;
    void                                   *_close_request;
    ucs_list_link_t                        _all_requests;
    ucs_status_t                           _ucx_status;
    bool                                   _use_am;
    std::queue<UcxContext::iomsg_buffer_t> _iomsg_recv_backlog;
};

#endif

