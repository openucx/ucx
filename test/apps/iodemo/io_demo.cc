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
    const char  *server_addr;
    int         port_num;
    long        client_retries;
    double      client_timeout;
    size_t      iomsg_size;
    size_t      data_size;
    long        iter_count;
    long        window_size;
    io_op_t     operation;
    bool        verbose;
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
        UcxContext(test_opts.iomsg_size), _test_opts(test_opts) {

        _data_buffer.resize(opts().data_size + ALIGNMENT);
        uintptr_t ptr = (uintptr_t)&_data_buffer[0];
        _padding = ((ptr + ALIGNMENT - 1) & ~(ALIGNMENT - 1)) - ptr;
    }

    const options_t& opts() const {
        return _test_opts;
    }

    void *buffer() {
        return &_data_buffer[_padding];
    }

    bool send_io_message(UcxConnection *conn, io_op_t op, uint32_t sn) {
        IoMessage *m = new IoMessage(opts().iomsg_size, op, sn,
                                     opts().data_size);
        VERBOSE_LOG << "sending IO " << io_op_names[op] << ", sn " << sn
                    << " data size " << opts().data_size;
        return conn->send_io_message(m->buffer(), opts().iomsg_size, m);
    }

protected:
    const options_t _test_opts;

private:
    std::string     _data_buffer;
    size_t          _padding;
};


class DemoServer : public P2pDemoCommon {
public:
    // sends an IO response when done
    class IoWriteResponseCallback : public UcxCallback {
    public:
        IoWriteResponseCallback(DemoServer *server, UcxConnection* conn,
                                uint32_t sn) :
            _server(server), _conn(conn), _sn(sn) {
        }

        virtual void operator()(ucs_status_t status) {
            if (status == UCS_OK) {
                _server->send_io_message(_conn, IO_COMP, _sn);
            }
            delete this;
        }

    private:
        DemoServer*     _server;
        UcxConnection*  _conn;
        uint32_t        _sn;
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
        assert(opts().data_size >= hdr->data_size);
        conn->send_data(buffer(), hdr->data_size, hdr->sn);

        // send response as data
        VERBOSE_LOG << "sending IO read response";
        IoMessage *response = new IoMessage(opts().iomsg_size, IO_COMP, hdr->sn,
                                            0);
        conn->send_data(response->buffer(), opts().iomsg_size, hdr->sn,
                        response);
    }

    void handle_io_write_request(UcxConnection* conn, const iomsg_hdr_t *hdr) {
        VERBOSE_LOG << "receiving IO write data";
        assert(opts().data_size >= hdr->data_size);
        conn->recv_data(buffer(), hdr->data_size, hdr->sn,
                        new IoWriteResponseCallback(this, conn, hdr->sn));
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
        _num_sent(0), _num_completed(0), _error_flag(true)
    {
    }

    void do_io_read(UcxConnection *conn, uint32_t sn) {
        if (!send_io_message(conn, IO_READ, sn)) {
            return;
        }

        ++_num_sent;
        IoReadResponseCallback *response =
                new IoReadResponseCallback(&_num_completed, opts().iomsg_size);
        conn->recv_data(buffer(), opts().data_size, sn, response);
        conn->recv_data(response->buffer(), opts().iomsg_size, sn, response);
    }

    void do_io_write(UcxConnection *conn, uint32_t sn) {
        if (!send_io_message(conn, IO_WRITE, sn)) {
            return;
        }

        ++_num_sent;
        VERBOSE_LOG << "sending data " << buffer() << " size "
                    << opts().data_size << " sn " << sn;
        conn->send_data(buffer(), opts().data_size, sn);
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
        while ((_num_sent - _num_completed > max_outstanding) && !_error_flag) {
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
            if (elapsed > _test_opts.client_timeout) {
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

    void report_performance(long num_iter, double elapsed) {
        double latency_usec   = (elapsed / num_iter) * 1e6;
        double throughput_mbs = (num_iter * opts().data_size) / elapsed /
                                (1024.0 * 1024.0);

        std::cout << num_iter << " iterations of " << io_op_names[opts().operation]
                  << ", throughput: " << throughput_mbs << " MB/s";
        if (opts().window_size == 1) {
            std::cout << " latency: " << latency_usec << " usec";
        }
        std::cout << std::endl;
    }

    bool run() {
        UcxConnection* conn = connect();
        if (!conn) {
            return false;
        }

        _error_flag = false;

        // TODO reset these values by canceling reuqests
        _num_sent      = 0;
        _num_completed = 0;

        double prev_time = get_time();
        long   prev_iter = 0;

        long iteration = 0;
        while ((iteration < opts().iter_count) && !_error_flag) {
            VERBOSE_LOG << " <<<< iteration " << iteration << " >>>>";

            if (!wait_for_responses(opts().window_size - 1)) {
                break;
            }

            switch (opts().operation) {
            case IO_READ:
                do_io_read(conn, iteration);
                break;
            case IO_WRITE:
                do_io_write(conn, iteration);
                break;
            default:
                break;
            }

            if (((iteration % 10) == 0) && (iteration > prev_iter)) {
                double curr_time = get_time();
                if (curr_time >= (prev_time + 1.0)) {
                    if (!wait_for_responses(0)) {
                        break;
                    }

                    report_performance(iteration - prev_iter,
                                       curr_time - prev_time);
                    prev_iter = iteration;
                    prev_time = curr_time;
                }
            }

            ++iteration;
        }

        if (wait_for_responses(0)) {
            report_performance(opts().iter_count - prev_iter,
                               get_time() - prev_time);
        }

        delete conn;
        return !_error_flag;
    }

private:
    long         _num_sent;
    long         _num_completed;
    bool         _error_flag;
};

static int parse_args(int argc, char **argv, options_t* test_opts)
{
    bool found;
    int c;

    test_opts->server_addr    = NULL;
    test_opts->port_num       = 1337;
    test_opts->client_retries = 10;
    test_opts->client_timeout = 1.0;
    test_opts->iomsg_size     = 256;
    test_opts->data_size      = 4096;
    test_opts->iter_count     = 1000;
    test_opts->window_size    = 1;
    test_opts->operation      = IO_WRITE;
    test_opts->verbose        = false;

    while ((c = getopt(argc, argv, "p:c:r:d:i:w:o:t:v")) != -1) {
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
            test_opts->data_size = strtol(optarg, NULL, 0);
            break;
        case 'i':
            test_opts->iter_count = strtol(optarg, NULL, 0);
            break;
        case 'w':
            test_opts->window_size = atoi(optarg);
            break;
        case 'o':
            found = false;
            for (int op = 0; op < IO_COMP; ++op) {
                if (!strcmp(io_op_names[op], optarg)) {
                    test_opts->operation = (io_op_t)op;
                    found = true;
                }
            }
            if (!found) {
                std::cout << "invalid operation name '" << optarg << "'" << std::endl;
                return -1;
            }
            break;
        case 't':
            test_opts->client_timeout = atof(optarg);
            break;
        case 'v':
            test_opts->verbose = true;
            break;
        case 'h':
        default:
            std::cout << "Usage: io_demo [ options] [ server_address]" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Supported options are:" << std::endl;
            std::cout << "  -p <port>                TCP port number to use" << std::endl;
            std::cout << "  -o <io-operation>        IO operation [read|write]" << std::endl;
            std::cout << "  -d <data-size>           Size of IO payload" << std::endl;
            std::cout << "  -i <iterations-count>    Number of iterations to run communication" << std::endl;
            std::cout << "  -w <window-size>         Number of outstanding requests" << std::endl;
            std::cout << "  -r <io-request-size>     Size of IO request packet" << std::endl;
            std::cout << "  -c <client retries>      Number of retries on client for failure" << std::endl;
            std::cout << "  -t <client timeout>      Client timeout, in seconds" << std::endl;
            std::cout << "  -v                       Set verbose mode" << std::endl;
            std::cout << "" << std::endl;
            return -1;
        }
    }

    if (optind < argc) {
        test_opts->server_addr = argv[optind];
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
    DemoClient client(test_opts);
    if (!client.init()) {
        return -1;
    }

    unsigned retry = 0;
    for (;;) {
        if (client.run()) {
            /* successful run */
            return 0;
        }

        ++retry;
        if (retry >= test_opts.client_retries) {
            /* client failed all retries */
            return -1;
        }

        LOG << "retry " << retry << "/" << test_opts.client_retries
            << " in " << test_opts.client_timeout << " seconds";
        usleep((int)(1e6 * test_opts.client_timeout));
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
