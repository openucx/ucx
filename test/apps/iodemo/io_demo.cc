/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucx_wrapper.h"

#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <iostream>
#include <string.h>
#include <getopt.h>
#include <assert.h>
#include <unistd.h>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <map>
#include <algorithm>
#include <limits>

#define ALIGNMENT       4096

/* IO operation type */
typedef enum {
    IO_READ,
    IO_WRITE,
    IO_COMP
} io_op_t;

static const char *io_op_names[] = {
    "read",
    "write",
    "completion"
};

/* test options */
typedef struct {
    std::vector<const char*> server_addrs;    
    int                      port_num;
    long                     client_retries;
    double                   client_timeout;
    size_t                   iomsg_size;
    size_t                   min_data_size;
    size_t                   max_data_size;
    size_t                   chunk_size;
    long                     iter_count;
    long                     window_size;
    std::vector<io_op_t>     operations;
    unsigned                 random_seed;
    size_t                   num_buffers;
    bool                     verbose;
} options_t;

#define LOG         UcxLog("[DEMO]", true)
#define VERBOSE_LOG UcxLog("[DEMO]", _test_opts.verbose)

template<class T>
class MemoryPool {
public:
    MemoryPool(size_t buffer_size = 0) :
        _num_allocated(0), _buffer_size(buffer_size) {
    }

    ~MemoryPool() {
        if (_num_allocated != _freelist.size()) {
            LOG << "Some items were not freed. Total:" << _num_allocated
                << ", current:" << _freelist.size() << ".";
        }
        
        for (size_t i = 0; i < _freelist.size(); i++) {
            delete _freelist[i];
        }
    }

    T * get() {
        T * item;
        
        if (_freelist.empty()) {
            item = new T(_buffer_size, this);
            _num_allocated++;
        } else {
            item = _freelist.back();
            _freelist.pop_back();
        }
        return item;
    }

    void put(T * item) {
        _freelist.push_back(item);
    }

private:
    std::vector<T*> _freelist;
    uint32_t        _num_allocated;
    size_t          _buffer_size;
};

/**
 * Linear congruential generator (LCG):
 * n[i + 1] = (n[i] * A + C) % M
 * where A, C, M used as in glibc
 */
class IoDemoRandom {
public:
    static void srand(unsigned seed) {
        _seed = seed & _M;
    }

    static inline int rand(int min = std::numeric_limits<int>::min(),
                           int max = std::numeric_limits<int>::max()) {
        _seed = (_seed * _A + _C) & _M;
        /* To resolve that LCG returns alternating even/odd values */
        if (max - min == 1) {
            return (_seed & 0x100) ? max : min;
        } else {
            return (int)_seed % (max - min + 1) + min;
        }
    }

private:
    static       unsigned     _seed;
    static const unsigned     _A;
    static const unsigned     _C;
    static const unsigned     _M;
};
unsigned IoDemoRandom::_seed    = 0;
const unsigned IoDemoRandom::_A = 1103515245U;
const unsigned IoDemoRandom::_C = 12345U;
const unsigned IoDemoRandom::_M = 0x7fffffffU;

class P2pDemoCommon : public UcxContext {
protected:

    /* IO request header */
    typedef struct {
        io_op_t     op;
        uint32_t    sn;
        size_t      data_size;
    } iomsg_hdr_t;

    typedef enum {
        XFER_TYPE_SEND,
        XFER_TYPE_RECV
    } xfer_type_t;
    
    /* Asynchronous IO message */
    class IoMessage : public UcxCallback {
    public:
        IoMessage(size_t buffer_size, MemoryPool<IoMessage>* pool) {
            _buffer      = malloc(buffer_size);
            _pool        = pool;
            _buffer_size = buffer_size;
        }
        
        void init(io_op_t op, uint32_t sn, size_t data_size) {
            iomsg_hdr_t *hdr = reinterpret_cast<iomsg_hdr_t*>(_buffer);
            assert(sizeof(*hdr) <= _buffer_size);
            hdr->op          = op;
            hdr->sn          = sn;
            hdr->data_size   = data_size;
        }

        ~IoMessage() {
            free(_buffer);
        }

        virtual void operator()(ucs_status_t status) {
            _pool->put(this);
        }

        void *buffer() {
            return _buffer;
        }

    private:
        void*                  _buffer;
        MemoryPool<IoMessage>* _pool;
        size_t                 _buffer_size;
    };

    P2pDemoCommon(const options_t& test_opts) :
        UcxContext(test_opts.iomsg_size), _test_opts(test_opts),
        _io_msg_pool(opts().iomsg_size), _cur_buffer_idx(0), _padding(0) {

        _data_buffers.resize(opts().num_buffers);
        for (size_t i = 0; i < _data_buffers.size(); ++i) {
            std::string &data_buffer = _data_buffers[i];
            data_buffer.resize(opts().max_data_size + ALIGNMENT);
            uintptr_t ptr = (uintptr_t)&data_buffer[0];
            _padding = ((ptr + ALIGNMENT - 1) & ~(ALIGNMENT - 1)) - ptr;
        }
    }

    const options_t& opts() const {
        return _test_opts;
    }

    inline void *buffer() {
        return &_data_buffers[_cur_buffer_idx][_padding];
    }

    inline void *buffer(size_t offset) {
        return &_data_buffers[_cur_buffer_idx][_padding + offset];
    }

    inline void next_buffer() {
        _cur_buffer_idx = (_cur_buffer_idx + 1) % _data_buffers.size();
        assert(_cur_buffer_idx < opts().num_buffers);
    }

    inline size_t get_data_size() {
        return IoDemoRandom::rand(opts().min_data_size,
                                  opts().max_data_size);
    }

    bool send_io_message(UcxConnection *conn, io_op_t op,
                         uint32_t sn, size_t data_size) {
        IoMessage *m = _io_msg_pool.get();
        m->init(op, sn, data_size);
        VERBOSE_LOG << "sending IO " << io_op_names[op] << ", sn " << sn
                    << " data size " << data_size;
        return conn->send_io_message(m->buffer(), opts().iomsg_size, m);
    }

    void send_recv_data_as_chunks(UcxConnection* conn, size_t data_size, uint32_t sn,
                                  xfer_type_t send_recv_data,
                                  UcxCallback* callback = EmptyCallback::get()) {
        size_t remaining = data_size;
        while (remaining > 0) {
            size_t xfer_size = std::min(opts().chunk_size, remaining);
            if (send_recv_data == XFER_TYPE_SEND) {
                conn->send_data(buffer(data_size - remaining), xfer_size, sn, callback);
            } else {
                conn->recv_data(buffer(data_size - remaining), xfer_size, sn, callback);
            }
            remaining -= xfer_size;
        }
    }

    void send_data_as_chunks(UcxConnection* conn, size_t data_size, uint32_t sn,
                             UcxCallback* callback = EmptyCallback::get()) {
        send_recv_data_as_chunks(conn, data_size, sn, XFER_TYPE_SEND, callback);
    }

    void recv_data_as_chunks(UcxConnection* conn, size_t data_size, uint32_t sn,
                             UcxCallback* callback = EmptyCallback::get()) {
        send_recv_data_as_chunks(conn, data_size, sn, XFER_TYPE_RECV, callback);
    }

    uint32_t get_chunk_cnt(size_t data_size) {
        return (data_size + opts().chunk_size - 1) / opts().chunk_size;
    }
    
    void send_data(UcxConnection* conn, size_t data_size, uint32_t sn,
                   UcxCallback* callback = EmptyCallback::get()) {
        send_data_as_chunks(conn, data_size, sn, callback);
    }
    
    void recv_data(UcxConnection* conn, size_t data_size, uint32_t sn,
                   UcxCallback* callback = EmptyCallback::get()) {
        recv_data_as_chunks(conn, data_size, sn, callback);
    }

protected:
    const options_t          _test_opts;
    MemoryPool<IoMessage>    _io_msg_pool;

private:
    std::vector<std::string> _data_buffers;
    size_t                   _cur_buffer_idx;
    size_t                   _padding;
};


class DemoServer : public P2pDemoCommon {
public:
    // sends an IO response when done
    class IoWriteResponseCallback : public UcxCallback {
    public:
        IoWriteResponseCallback(size_t buffer_size,
            MemoryPool<IoWriteResponseCallback>* pool) :
            _server(NULL), _conn(NULL), _sn(0), _data_size(0), _chunk_cnt(0) {
            _pool = pool;
        }

        void init(DemoServer *server, UcxConnection* conn, uint32_t sn,
                  size_t data_size, uint32_t chunk_cnt = 1) {
             _server    = server;
             _conn      = conn;
             _sn        = sn;
             _data_size = data_size;
             _chunk_cnt = chunk_cnt;
        }

        virtual void operator()(ucs_status_t status) {
            if (--_chunk_cnt > 0) {
                return;
            }
            if (status == UCS_OK) {
                _server->send_io_message(_conn, IO_COMP, _sn, _data_size);
            }
            _pool->put(this);
        }

    private:
        DemoServer*                          _server;
        UcxConnection*                       _conn;
        uint32_t                             _sn;
        size_t                               _data_size;
        uint32_t                             _chunk_cnt;
        MemoryPool<IoWriteResponseCallback>* _pool;
    };

    DemoServer(const options_t& test_opts) :
        P2pDemoCommon(test_opts), _callback_pool(0) {
    }

    void run() {
        struct sockaddr_in listen_addr;
        memset(&listen_addr, 0, sizeof(listen_addr));
        listen_addr.sin_family      = AF_INET;
        listen_addr.sin_addr.s_addr = INADDR_ANY;
        listen_addr.sin_port        = htons(opts().port_num);

        listen((const struct sockaddr*)&listen_addr, sizeof(listen_addr));
        for (;;) {
            try {
                progress();
            } catch (const std::exception &e) {
                std::cerr << e.what();
            }
        }
    }

    void handle_io_read_request(UcxConnection* conn, const iomsg_hdr_t *hdr) {
        // send data
        VERBOSE_LOG << "sending IO read data";
        assert(opts().max_data_size >= hdr->data_size);

        send_data(conn, hdr->data_size, hdr->sn);
        
        // send response as data
        VERBOSE_LOG << "sending IO read response";
        IoMessage *response = _io_msg_pool.get();
        response->init(IO_COMP, hdr->sn, 0);
        conn->send_data(response->buffer(), opts().iomsg_size, hdr->sn,
                        response);

        next_buffer();
    }

    void handle_io_write_request(UcxConnection* conn, const iomsg_hdr_t *hdr) {
        VERBOSE_LOG << "receiving IO write data";
        assert(opts().max_data_size >= hdr->data_size);
        assert(hdr->data_size != 0);

        IoWriteResponseCallback *w = _callback_pool.get();
        w->init(this, conn, hdr->sn, hdr->data_size, get_chunk_cnt(hdr->data_size));
        recv_data(conn, hdr->data_size, hdr->sn, w);

        next_buffer();
    }

    virtual void dispatch_connection_error(UcxConnection *conn) {
        LOG << "deleting connection " << conn;
        delete conn;
    }

    virtual void dispatch_io_message(UcxConnection* conn, const void *buffer,
                                     size_t length) {
        const iomsg_hdr_t *hdr = reinterpret_cast<const iomsg_hdr_t*>(buffer);

        VERBOSE_LOG << "got io message " << io_op_names[hdr->op] << " sn "
                    << hdr->sn << " data size " << hdr->data_size << " conn "
                    << conn;

        if (hdr->op == IO_READ) {
            handle_io_read_request(conn, hdr);
        } else if (hdr->op == IO_WRITE) {
            handle_io_write_request(conn, hdr);
        } else {
            LOG << "Invalid opcode: " << hdr->op;
        }
    }
protected:
    MemoryPool<IoWriteResponseCallback> _callback_pool;    
};


class DemoClient : public P2pDemoCommon {
public:
    class IoReadResponseCallback : public UcxCallback {
    public:
        IoReadResponseCallback(size_t buffer_size,
            MemoryPool<IoReadResponseCallback>* pool) :
            _counter(0), _io_counter(0), _chunk_cnt(0) {
            _buffer = malloc(buffer_size);
            _pool   = pool;
        }
        
        void init(long *counter, uint32_t chunk_cnt = 1) {
            _counter    = 0;
            _io_counter = counter;
            _chunk_cnt  = chunk_cnt;
        }

        ~IoReadResponseCallback() {
            free(_buffer);
        }

        virtual void operator()(ucs_status_t status) {
            /* wait data and response completion */
            if (++_counter < (1 + _chunk_cnt)) {
                return;
            }

            ++(*_io_counter);
            _pool->put(this);
        }

        void* buffer() {
            return _buffer;
        }

    private:
        long                                _counter;
        long*                               _io_counter;
        uint32_t                            _chunk_cnt;
        void*                               _buffer;
        MemoryPool<IoReadResponseCallback>* _pool;
    };

    DemoClient(const options_t& test_opts) :
        P2pDemoCommon(test_opts),
        _num_sent(0), _num_completed(0), _error_flag(true),
		_retry(0), _callback_pool(opts().iomsg_size)
    {
    }

    size_t do_io_read(UcxConnection *conn, uint32_t sn) {
        size_t data_size = get_data_size();

        if (!send_io_message(conn, IO_READ, sn, data_size)) {
            return data_size;
        }

        ++_num_sent;
        IoReadResponseCallback *r = _callback_pool.get();
        r->init(&_num_completed, get_chunk_cnt(data_size));
        recv_data(conn, data_size, sn, r);
        conn->recv_data(r->buffer(), opts().iomsg_size, sn, r);
        next_buffer();

        return data_size;
    }

    size_t do_io_write(UcxConnection *conn, uint32_t sn) {
        size_t data_size = get_data_size();

        if (!send_io_message(conn, IO_WRITE, sn, data_size)) {
            return data_size;
        }

        ++_num_sent;
        VERBOSE_LOG << "sending data " << buffer() << " size "
                    << data_size << " sn " << sn;
        send_data(conn, data_size, sn);
        next_buffer();

        return data_size;
    }

    virtual void dispatch_io_message(UcxConnection* conn, const void *buffer,
                                     size_t length) {
        const iomsg_hdr_t *hdr = reinterpret_cast<const iomsg_hdr_t*>(buffer);

        VERBOSE_LOG << "got io message " << io_op_names[hdr->op] << " sn "
                    << hdr->sn << " data size " << hdr->data_size
                    << " conn " << conn;

        if (hdr->op == IO_COMP) {
            ++_num_completed;
        }
    }

    virtual void dispatch_connection_error(UcxConnection *conn) {
        LOG << "setting error flag on connection " << conn;
        _error_flag = true;
    }

    bool wait_for_responses(long max_outstanding) {
        struct timeval tv_start, tv_curr, tv_diff;
        bool timer_started = false;
        long count;

        count = 0;
        while (((_num_sent - _num_completed) > max_outstanding) && !_error_flag) {
            if (count < 1000) {
                progress();
                ++count;
                continue;
            }

            count = 0;

            gettimeofday(&tv_curr, NULL);

            if (!timer_started) {
                tv_start      = tv_curr;
                timer_started = true;
                continue;
            }

            timersub(&tv_curr, &tv_start, &tv_diff);
            double elapsed = tv_diff.tv_sec + (tv_diff.tv_usec * 1e-6);
            if (elapsed > _test_opts.client_timeout * 10) {
                LOG << "timeout waiting for " << (_num_sent - _num_completed)
                    << " replies";
                _error_flag = true;
            }
        }

        return !_error_flag;
    }

    UcxConnection* connect(const char* server_addr) {
        struct sockaddr_in connect_addr;
        memset(&connect_addr, 0, sizeof(connect_addr));
        connect_addr.sin_family = AF_INET;
        connect_addr.sin_port   = htons(opts().port_num);
        inet_pton(AF_INET, server_addr, &connect_addr.sin_addr);

        return UcxContext::connect((const struct sockaddr*)&connect_addr,
                                   sizeof(connect_addr));
    }

    static double get_time() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return tv.tv_sec + (tv.tv_usec * 1e-6);
    }

    static std::string get_time_str() {
        char str[80];
        struct timeval tv;
        gettimeofday(&tv, NULL);
        snprintf(str, sizeof(str), "[%lu.%06lu]", tv.tv_sec, tv.tv_usec);
        return str;
    }

    bool run() {
        std::vector<UcxConnection*> conn;
        conn.resize(opts().server_addrs.size());
        for (size_t i = 0; i < conn.size(); i++) {
            conn[i] = connect(opts().server_addrs[i]);
            if (!conn[i]) {
                LOG << "Connect to server ["
                    << opts().server_addrs[i]
                    << "] Failed!";
                for (size_t j = 0; j < i; j++) {
                    delete conn[j];
                }
                return false;
            }
        }

        // reset number of retries after successful connection
        _retry = 0;

        _error_flag = false;

        // TODO reset these values by canceling requests
        _num_sent      = 0;
        _num_completed = 0;

        double prev_time     = get_time();
        long total_iter      = 0;
        long total_prev_iter = 0;
        std::vector<op_info_t> info;

        for (int i = 0; i < IO_COMP; ++i) {
            op_info_t op_info = {static_cast<io_op_t>(i), 0, 0};
            info.push_back(op_info);
        }

        while ((total_iter < opts().iter_count) && !_error_flag) {
            VERBOSE_LOG << " <<<< iteration " << total_iter << " >>>>";

            if (!wait_for_responses(opts().window_size - 1)) {
                break;
            }

            size_t conn_num = IoDemoRandom::rand(0, conn.size() - 1);
            io_op_t op      = get_op();
            size_t size;
            switch (op) {
            case IO_READ:
                size = do_io_read(conn[conn_num], total_iter);
                break;
            case IO_WRITE:
                size = do_io_write(conn[conn_num], total_iter);
                break;
            default:
                abort();
            }

            info[op].total_bytes += size;
            info[op].num_iters++;

            if (((total_iter % 10) == 0) && (total_iter > total_prev_iter)) {
                double curr_time = get_time();
                if (curr_time >= (prev_time + 1.0)) {
                    if (!wait_for_responses(0)) {
                        break;
                    }

                    report_performance(total_iter - total_prev_iter,
                                       curr_time - prev_time, info);
                    total_prev_iter = total_iter;
                    prev_time       = curr_time;
                }
            }

            ++total_iter;
        }

        if (wait_for_responses(0)) {
            report_performance(total_iter - total_prev_iter,
                               get_time() - prev_time, info);
        }

        for (size_t i = 0; i < conn.size(); i++) {
            delete conn[i];
        }
        return !_error_flag;
    }

    int update_retry() {
        if (++_retry >= opts().client_retries) {
            /* client failed all retries */
            return -1;
        }

        LOG << "retry " << _retry << "/" << opts().client_retries
            << " in " << opts().client_timeout << " seconds";
        usleep((int)(1e6 * opts().client_timeout));
        return 0;
    }

private:
    typedef struct {
        io_op_t   op;
        long      num_iters;
        size_t    total_bytes;
    } op_info_t;

    inline io_op_t get_op() {
        if (opts().operations.size() == 1) {
            return opts().operations[0];
        }

        return opts().operations[IoDemoRandom::rand(
                                 0, opts().operations.size() - 1)];
    }

    void report_performance(long num_iters, double elapsed,
                            std::vector<op_info_t> &info) {
        double latency_usec = (elapsed / num_iters) * 1e6;
        bool first_print    = true;

        for (unsigned i = 0; i < info.size(); ++i) {
            op_info_t *op_info = &info[i];

            if (!op_info->total_bytes) {
                continue;
            }

            if (first_print) {
                std::cout << get_time_str() << " ";
                first_print = false;
            } else {
                // print comma for non-first printouts
                std::cout << ", ";
            }

            double throughput_mbs = op_info->total_bytes /
                                    elapsed / (1024.0 * 1024.0);

            std::cout << op_info->num_iters << " "
                      << io_op_names[op_info->op] << "s at "
                      << throughput_mbs << " MB/s";

            // reset for the next round
            op_info->total_bytes = 0;
            op_info->num_iters   = 0;
        }

        if (opts().window_size == 1) {
            std::cout << ", average latency: " << latency_usec << " usec";
        }
        std::cout << std::endl;
    }

private:
    long         _num_sent;
    long         _num_completed;
    bool         _error_flag;
    unsigned     _retry;
protected:    
    MemoryPool<IoReadResponseCallback> _callback_pool;
};

static int set_data_size(char *str, options_t* test_opts)
{
    const static char token = ':';
    char *val1, *val2;

    if (strchr(str, token) == NULL) {
        test_opts->min_data_size =
            test_opts->max_data_size = strtol(str, NULL, 0);
        return 0;
    }

    val1 = strtok(str, ":");
    val2 = strtok(NULL, ":");

    if ((val1 != NULL) && (val2 != NULL)) {
        test_opts->min_data_size = strtol(val1, NULL, 0);
        test_opts->max_data_size = strtol(val2, NULL, 0);
    } else if (val1 != NULL) {
        if (str[0] == ':') {
            test_opts->min_data_size = 0;
            test_opts->max_data_size = strtol(val1, NULL, 0);
        } else {
            test_opts->min_data_size = strtol(val1, NULL, 0);
        }
    } else {
        return -1;
    }

    return 0;
}

static int parse_args(int argc, char **argv, options_t* test_opts)
{
    char *str;
    bool found;
    int c;

    test_opts->port_num       = 1337;
    test_opts->client_retries = 10;
    test_opts->client_timeout = 1.0;
    test_opts->min_data_size  = 4096;
    test_opts->max_data_size  = 4096;
    test_opts->chunk_size           = std::numeric_limits<unsigned>::max();
    test_opts->num_buffers    = 1;
    test_opts->iomsg_size     = 256;
    test_opts->iter_count     = 1000;
    test_opts->window_size    = 1;
    test_opts->random_seed    = std::time(NULL);
    test_opts->verbose        = false;

    while ((c = getopt(argc, argv, "p:c:r:d:b:i:w:k:o:t:s:v")) != -1) {
        switch (c) {
        case 'p':
            test_opts->port_num = atoi(optarg);
            break;
        case 'c':
            test_opts->client_retries = strtol(optarg, NULL, 0);
            break;
        case 'r':
            test_opts->iomsg_size = strtol(optarg, NULL, 0);
            break;
        case 'd':
            if (set_data_size(optarg, test_opts) == -1) {
                std::cout << "invalid data size range '" << optarg << "'" << std::endl;
                return -1;
            }
            break;
        case 'b':
            test_opts->num_buffers = strtol(optarg, NULL, 0);
            if (test_opts->num_buffers == 0) {
                std::cout << "number of buffers ('" << optarg << "')"
                          << " has to be > 0" << std::endl;
                return -1;
            }
            break;
        case 'i':
            test_opts->iter_count = strtol(optarg, NULL, 0);
            break;
        case 'w':
            test_opts->window_size = atoi(optarg);
            break;
        case 'k':
            test_opts->chunk_size = strtol(optarg, NULL, 0);
            break;
        case 'o':
            str = strtok(optarg, ",");
            while (str != NULL) {
                found = false;

                for (int op_it = 0; op_it < IO_COMP; ++op_it) {
                    if (!strcmp(io_op_names[op_it], str)) {
                        io_op_t op = static_cast<io_op_t>(op_it);
                        if (std::find(test_opts->operations.begin(),
                                      test_opts->operations.end(),
                                      op) == test_opts->operations.end()) {
                            test_opts->operations.push_back(op);
                        }
                        found = true;
                    }
                }

                if (!found) {
                    std::cout << "invalid operation name '" << str << "'" << std::endl;
                    return -1;
                }

                str = strtok(NULL, ",");
            }

            if (test_opts->operations.size() == 0) {
                std::cout << "no operation names were provided '" << optarg << "'" << std::endl;
                return -1;
            }
            break;
        case 't':
            test_opts->client_timeout = atof(optarg);
            break;
        case 's':
            test_opts->random_seed = strtoul(optarg, NULL, 0);
            break;
        case 'v':
            test_opts->verbose = true;
            break;
        case 'h':
        default:
            std::cout << "Usage: io_demo [options] [server_address]" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Supported options are:" << std::endl;
            std::cout << "  -p <port>                TCP port number to use" << std::endl;
            std::cout << "  -o <op1,op2,...,opN>     Comma-separated string of IO operations [read|write]" << std::endl;
            std::cout << "                             NOTE: if using several IO operations, performance" << std::endl;
            std::cout << "                                 measurments may be inaccurate" << std::endl;
            std::cout << "  -d <min>:<max>           Range that should be used to get data" << std::endl;
            std::cout << "                           size of IO payload" << std::endl;
            std::cout << "  -b <number of buffers>   Number of IO buffers to use for communications" << std::endl;
            std::cout << "  -i <iterations-count>    Number of iterations to run communication" << std::endl;
            std::cout << "  -w <window-size>         Number of outstanding requests" << std::endl;
            std::cout << "  -k <chunk-size>            Split the data transfer to chunks of this size" << std::endl;
            std::cout << "  -r <io-request-size>     Size of IO request packet" << std::endl;
            std::cout << "  -c <client retries>      Number of retries on client for failure" << std::endl;
            std::cout << "  -t <client timeout>      Client timeout, in seconds" << std::endl;
            std::cout << "  -s <random seed>         Random seed to use for randomizing" << std::endl;
            std::cout << "  -v                       Set verbose mode" << std::endl;
            std::cout << "" << std::endl;
            return -1;
        }
    }

    while (optind < argc) {
        test_opts->server_addrs.push_back(argv[optind++]);
    }

    if (test_opts->operations.size() == 0) {
        test_opts->operations.push_back(IO_WRITE);
    }

    return 0;
}

static int do_server(const options_t& test_opts)
{
    DemoServer server(test_opts);
    if (!server.init()) {
        return -1;
    }

    server.run();
    return 0;
}

static int do_client(const options_t& test_opts)
{
    IoDemoRandom::srand(test_opts.random_seed);
    LOG << "random seed: " << test_opts.random_seed;

    DemoClient client(test_opts);
    if (!client.init()) {
        return -1;
    }

    for (;;) {
        if (client.run()) {
            /* successful run */
            return 0;
        }

        if (client.update_retry() != 0) {
            break;
        }
    }

    return -1;
}

int main(int argc, char **argv)
{
    options_t test_opts;
    int ret;

    ret = parse_args(argc, argv, &test_opts);
    if (ret < 0) {
        return ret;
    }

    if (test_opts.server_addrs.empty()) {
        return do_server(test_opts);
    } else {
        return do_client(test_opts);
    }
}
