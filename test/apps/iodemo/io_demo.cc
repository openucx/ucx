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
    long        client_iter;
    size_t      iomsg_size;
    size_t      data_size;
    long        iter_count;
    long        window_size;
    io_op_t     operation;
    bool        verbose;
} options_t;


class P2pDemoCommon : protected UcxContext {
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

        virtual void operator()() {
            delete this;
        }

        void *buffer() {
            return _buffer;
        }

    private:
        void *_buffer;
    };

    P2pDemoCommon(const options_t& test_opts) :
        UcxContext(test_opts.iomsg_size, test_opts.verbose),
        _test_opts(test_opts) {

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

    void send_io_message(UcxConnection *conn, io_op_t op, uint32_t sn) {
        IoMessage *m = new IoMessage(opts().iomsg_size, op, sn,
                                     opts().data_size);
        verbose_os() << "sending IO " << io_op_names[op] << ", sn " << sn
                     << " data size " << opts().data_size << std::endl;
        conn->send_io_message(m->buffer(), opts().iomsg_size, m);
    }

private:
    const options_t _test_opts;
    std::string     _data_buffer;
    size_t          _padding;
};


class DemoServer : private P2pDemoCommon {
public:
    // sends an IO response when done
    class IoWriteResponseCallback : public UcxCallback {
    public:
        IoWriteResponseCallback(DemoServer *server, UcxConnection* conn,
                                uint32_t sn) :
            _server(server), _conn(conn), _sn(sn) {
        }

        virtual void operator()() {
            _server->send_io_message(_conn, IO_COMP, _sn);
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
        assert(opts().data_size == hdr->data_size);
        conn->send_data(buffer(), hdr->data_size, hdr->sn);

        // send response as data
        IoMessage *response = new IoMessage(opts().iomsg_size, IO_COMP, hdr->sn,
                                            0);
        conn->send_data(response->buffer(), opts().iomsg_size, hdr->sn,
                        response);
    }

    void handle_io_write_request(UcxConnection* conn, const iomsg_hdr_t *hdr) {
        assert(opts().data_size == hdr->data_size);
        conn->recv_data(buffer(), hdr->data_size, hdr->sn,
                        new IoWriteResponseCallback(this, conn, hdr->sn));
    }

    virtual void dispatch_io_message(UcxConnection* conn, const void *buffer) {
        const iomsg_hdr_t *hdr = reinterpret_cast<const iomsg_hdr_t*>(buffer);

        verbose_os() << "got io message " << io_op_names[hdr->op] << " sn "
                     << hdr->sn << " data size " << hdr->data_size << " conn "
                     << conn << std::endl;

        if (hdr->op == IO_READ) {
            handle_io_read_request(conn, hdr);
        } else if (hdr->op == IO_WRITE) {
            handle_io_write_request(conn, hdr);
        }
    }
};


class DemoClient : private P2pDemoCommon {
public:
    class IoReadResponseCallback : public UcxCallback {
    public:
        IoReadResponseCallback(long *counter, size_t iomsg_size) :
            _counter(0), _io_counter(counter), _buffer(malloc(iomsg_size)) {
        }

        ~IoReadResponseCallback() {
            free(_buffer);
        }

        virtual void operator()() {
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
        _num_sent(0), _num_completed(0) {
    }

    void do_io_read(UcxConnection *conn, uint32_t sn) {
        ++_num_sent;
        send_io_message(conn, IO_READ, sn);
        IoReadResponseCallback *response =
                new IoReadResponseCallback(&_num_completed, opts().iomsg_size);
        conn->recv_data(buffer(), opts().data_size, sn, response);
        conn->recv_data(response->buffer(), opts().iomsg_size, sn, response);
    }

    void do_io_write(UcxConnection *conn, uint32_t sn) {
        ++_num_sent;
        send_io_message(conn, IO_WRITE, sn);
        verbose_os() << "sending data " << buffer() << " size "
                     << opts().data_size << " sn " << sn << std::endl;
        conn->send_data(buffer(), opts().data_size, sn);
    }

    virtual void dispatch_io_message(UcxConnection* conn, const void *buffer) {
        const iomsg_hdr_t *hdr = reinterpret_cast<const iomsg_hdr_t*>(buffer);

        verbose_os() << "got io message " << io_op_names[hdr->op] << " sn "
                     << hdr->sn << " data size " << hdr->data_size
                     << " conn " << conn << std::endl;

        if (hdr->op == IO_COMP) {
            ++_num_completed;
        }
    }

    virtual void on_disconnect(UcxConnection *conn) _GLIBCXX_NOTHROW {
        if (_num_sent > _num_completed) {
            std::cout << _num_sent - _num_completed
                      << " io messages are without response on conn " << conn
                      << std::endl;
            _num_completed = _num_sent;
        }

        P2pDemoCommon::on_disconnect(conn);
    }

    void wait_for_responses(long max_outstanding) {
        while (_num_sent - _num_completed > max_outstanding) {
            progress();
        }
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
        std::cout<< std::endl;
    }

    void run() {
        UcxConnection* conn = connect();

        double prev_time = get_time();
        long   prev_iter = 0;

        for (long i = 0; i < opts().iter_count; ++i) {
            verbose_os() << " <<<< iteration " << i << " >>>>" << std::endl;

            wait_for_responses(opts().window_size - 1);
            switch (opts().operation) {
            case IO_READ:
                do_io_read(conn, i);
                break;
            case IO_WRITE:
                do_io_write(conn, i);
                break;
            default:
                break;
            }

            if (((i % 10) == 0) && (i > prev_iter)) {
                double curr_time = get_time();
                if (curr_time >= (prev_time + 1.0)) {
                    wait_for_responses(0);
                    report_performance(i - prev_iter, curr_time - prev_time);
                    prev_iter = i;
                    prev_time = curr_time;
                }
            }
        }

        wait_for_responses(0);
        report_performance(opts().iter_count - prev_iter,
                           get_time() - prev_time);

        disconnect(conn);
    }

private:
    long         _num_sent;
    long         _num_completed;
};

static int parse_args(int argc, char **argv, options_t* test_opts)
{
    bool found;
    int c;

    test_opts->server_addr = NULL;
    test_opts->port_num    = 1337;
    test_opts->client_iter = 1;
    test_opts->iomsg_size  = 256;
    test_opts->data_size   = 4096;
    test_opts->iter_count  = 1000;
    test_opts->window_size = 1;
    test_opts->operation   = IO_WRITE;
    test_opts->verbose     = false;

    while ((c = getopt(argc, argv, "p:c:r:d:i:w:o:v")) != -1) {
        switch (c) {
        case 'p':
            test_opts->port_num = atoi(optarg);
            break;
        case 'c':
            test_opts->client_iter = strtol(optarg, NULL, 0);
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
        case 'v':
            test_opts->verbose = true;
            break;
        case 'h':
        default:
            std::cout << "Usage: io_demo [ options] [ server_address]" << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Supported options are:" << std::endl;
            std::cout << "  -p <port>                TCP port number to use" << std::endl;
            std::cout << "  -c <client iterations>   Number of iterations to run client" << std::endl;
            std::cout << "  -r <io-request-size>     Size of IO request packet" << std::endl;
            std::cout << "  -d <data-size>           Size of IO payload" << std::endl;
            std::cout << "  -i <iterations-count>    Number of iterations to run communication" << std::endl;
            std::cout << "  -w <window-size>         Number of outstanding requests" << std::endl;
            std::cout << "  -o <io-operation>        IO operation [read|write]" << std::endl;
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

int main(int argc, char **argv)
{
    options_t test_opts;
    int ret;

    ret = parse_args(argc, argv, &test_opts);
    if (ret < 0) {
        return ret;
    }

    if (test_opts.server_addr == NULL) {
        try {
            DemoServer server(test_opts);
            server.run();
            ret = 0;
        } catch (const std::exception &e) {
            std::cerr << e.what();
            ret = -1;
        }
    } else {
        for (unsigned i = 0; i < test_opts.client_iter; ++i) {
            try {
                DemoClient client(test_opts);
                client.run();
                ret = 0;
            } catch (const std::exception &e) {
                std::cerr << e.what();
                ret = -1;
            }
        }
    }

    return ret;
}
