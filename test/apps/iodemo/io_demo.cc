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
#include <algorithm>

#define ALIGNMENT 4096


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
    const char           *server_addr;
    int                  port_num;
    long                 client_retries;
    double               client_timeout;
    size_t               iomsg_size;
    size_t               min_data_size;
    size_t               max_data_size;
    long                 iter_count;
    long                 window_size;
    std::vector<io_op_t> operations;
    unsigned             random_seed;
    size_t               num_buffers;
    bool                 verbose;
} options_t;

#define LOG         UcxLog("[DEMO]", true)
#define VERBOSE_LOG UcxLog("[DEMO]", _test_opts.verbose)

class P2pDemoCommon : public UcxContext {
protected:

    /* IO request header */
    typedef struct {
        io_op_t     op;
        uint32_t    sn;
        size_t      data_size;
    } iomsg_hdr_t;

    /* Asynchronous IO message */
    class IoMessage : public UcxCallback {
    public:
        IoMessage(size_t buffer_size, io_op_t op, uint32_t sn, size_t data_size) :
            _buffer(malloc(buffer_size)) {
            iomsg_hdr_t *hdr = reinterpret_cast<iomsg_hdr_t*>(_buffer);
            assert(sizeof(*hdr) <= buffer_size);
            hdr->op          = op;
            hdr->sn          = sn;
            hdr->data_size   = data_size;
        }

        ~IoMessage() {
            free(_buffer);
        }

        virtual void operator()(ucs_status_t status) {
            delete this;
        }

        void *buffer() {
            return _buffer;
        }

    private:
        void *_buffer;
    };

    P2pDemoCommon(const options_t& test_opts) :
        UcxContext(test_opts.iomsg_size), _test_opts(test_opts),
        _cur_buffer_idx(0) {

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

    inline void next_buffer() {
        _cur_buffer_idx = (_cur_buffer_idx + 1) % _data_buffers.size();
        assert(_cur_buffer_idx < opts().num_buffers);
    }

    inline size_t get_data_size() {
        return opts().min_data_size +
               (std::rand() % static_cast<size_t>(opts().max_data_size -
                                                  opts().min_data_size + 1));
    }

    bool send_io_message(UcxConnection *conn, io_op_t op,
                         uint32_t sn, size_t data_size) {
        IoMessage *m = new IoMessage(opts().iomsg_size, op, sn,
                                     data_size);
        VERBOSE_LOG << "sending IO " << io_op_names[op] << ", sn " << sn
                    << " data size " << data_size;
        return conn->send_io_message(m->buffer(), opts().iomsg_size, m);
    }

protected:
    const options_t          _test_opts;

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
        IoWriteResponseCallback(DemoServer *server, UcxConnection* conn,
                                uint32_t sn, size_t data_size) :
            _server(server), _conn(conn), _sn(sn), _data_size(data_size) {
        }

        virtual void operator()(ucs_status_t status) {
            if (status == UCS_OK) {
                _server->send_io_message(_conn, IO_COMP, _sn, _data_size);
            }
            delete this;
        }

    private:
        DemoServer*     _server;
        UcxConnection*  _conn;
        uint32_t        _sn;
        size_t          _data_size;
    };

    DemoServer(const options_t& test_opts) : P2pDemoCommon(test_opts) {
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
        conn->send_data(buffer(), hdr->data_size, hdr->sn);

        // send response as data
        VERBOSE_LOG << "sending IO read response";
        IoMessage *response = new IoMessage(opts().iomsg_size, IO_COMP, hdr->sn,
                                            0);
        conn->send_data(response->buffer(), opts().iomsg_size, hdr->sn,
                        response);

        next_buffer();
    }

    void handle_io_write_request(UcxConnection* conn, const iomsg_hdr_t *hdr) {
        VERBOSE_LOG << "receiving IO write data";
        assert(opts().max_data_size >= hdr->data_size);
        conn->recv_data(buffer(), hdr->data_size, hdr->sn,
                        new IoWriteResponseCallback(this, conn, hdr->sn,
                                                     hdr->data_size));

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
};


class DemoClient : public P2pDemoCommon {
public:
    class IoReadResponseCallback : public UcxCallback {
    public:
        IoReadResponseCallback(long *counter, size_t iomsg_size) :
            _counter(0), _io_counter(counter), _buffer(malloc(iomsg_size)) {
        }

        ~IoReadResponseCallback() {
            free(_buffer);
        }

        virtual void operator()(ucs_status_t status) {
            /* wait data and response completion */
            if (++_counter < 2) {
                return;
            }

            ++(*_io_counter);
            delete this;
        }

        void* buffer() {
            return _buffer;
        }

    private:
        long  _counter;
        long* _io_counter;
        void* _buffer;
    };

    DemoClient(const options_t& test_opts) :
        P2pDemoCommon(test_opts),
        _num_sent(0), _num_completed(0), _error_flag(true), _retry(0)
    {
    }

    size_t do_io_read(UcxConnection *conn, uint32_t sn) {
        size_t data_size = get_data_size();

        if (!send_io_message(conn, IO_READ, sn, data_size)) {
            return data_size;
        }

        ++_num_sent;
        IoReadResponseCallback *response =
                new IoReadResponseCallback(&_num_completed, opts().iomsg_size);
        conn->recv_data(buffer(), data_size, sn, response);
        conn->recv_data(response->buffer(), opts().iomsg_size, sn, response);

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
        conn->send_data(buffer(), data_size, sn);

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

    UcxConnection* connect() {
        struct sockaddr_in connect_addr;
        memset(&connect_addr, 0, sizeof(connect_addr));
        connect_addr.sin_family = AF_INET;
        connect_addr.sin_port   = htons(opts().port_num);
        inet_pton(AF_INET, opts().server_addr, &connect_addr.sin_addr);

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
        UcxConnection* conn = connect();
        if (!conn) {
            return false;
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

            io_op_t op = get_op();
            size_t size;
            switch (op) {
            case IO_READ:
                size = do_io_read(conn, total_iter);
                break;
            case IO_WRITE:
                size = do_io_write(conn, total_iter);
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

        delete conn;
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

        return opts().operations[std::rand() %
                                 opts().operations.size()];
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

    test_opts->server_addr    = NULL;
    test_opts->port_num       = 1337;
    test_opts->client_retries = 10;
    test_opts->client_timeout = 1.0;
    test_opts->min_data_size  = 4096;
    test_opts->max_data_size  = 4096;
    test_opts->num_buffers    = 1;
    test_opts->iomsg_size     = 256;
    test_opts->iter_count     = 1000;
    test_opts->window_size    = 1;
    test_opts->random_seed    = std::time(NULL);
    test_opts->verbose        = false;

    while ((c = getopt(argc, argv, "p:c:r:d:b:i:w:o:t:s:v")) != -1) {
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
            std::cout << "                           NOTE: if using \"random\", performance" << std::endl;
            std::cout << "                                 measurments may be inaccurate" << std::endl;
            std::cout << "  -d <min>:<max>           Range that should be used to get data" << std::endl;
            std::cout << "                           size of IO payload" << std::endl;
            std::cout << "  -b <number of buffers>   Number of IO buffers to use for communications" << std::endl;
            std::cout << "  -i <iterations-count>    Number of iterations to run communication" << std::endl;
            std::cout << "  -w <window-size>         Number of outstanding requests" << std::endl;
            std::cout << "  -r <io-request-size>     Size of IO request packet" << std::endl;
            std::cout << "  -c <client retries>      Number of retries on client for failure" << std::endl;
            std::cout << "  -t <client timeout>      Client timeout, in seconds" << std::endl;
            std::cout << "  -s <random seed>         Random seed to use for randomizing" << std::endl;
            std::cout << "  -v                       Set verbose mode" << std::endl;
            std::cout << "" << std::endl;
            return -1;
        }
    }

    if (optind < argc) {
        test_opts->server_addr = argv[optind];
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
    std::srand(test_opts.random_seed);
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

    if (test_opts.server_addr == NULL) {
        return do_server(test_opts);
    } else {
        return do_client(test_opts);
    }
}
