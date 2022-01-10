/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include <arpa/inet.h>
#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <float.h>
#include <getopt.h>
#include <iostream>
#include <netinet/in.h>
#include <set>
#include <signal.h>
#include <sstream>
#include <stdlib.h>
#include <stdexcept>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <ucp/api/ucp.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>
#include <ucs/time/time.h>


/* program options */
typedef struct {
    int           port_num;
    std::string   file_name;
    size_t        block_size;
    unsigned      num_blocks;
    std::string   server_address;
    unsigned      iter_count;
    bool          random_access;
    int           mmap_protocol;
    unsigned      num_outstanding;
} options_t;

enum register_type {
    READ     = 0, // Read block data into bounce buffer
    MMAP     = 1, // Memory map whole file
    REGISTER = 2  // Memory map and pre-register whole file
};

/* To break on Ctrl-C */
static volatile bool keep_running = true;

/**
 * @brief Class to handle file and blocks
 */
class FileData
{
    int fd;
    options_t *test_opts;
public:
    void* base_address;
    FileData(options_t *test_opts);
    ~FileData();

    /**
     * @brief Get the block address. Used with mmap protocol.
     *
     * @return void* - address of mmaped block
     */
    void* get_block(unsigned block_id);

    /**
     * @brief Read block into pre-registered bounce buffer.
     *
     * @param block_id
     * @param bounce_buffer
     */
    void get_block(unsigned block_id, void *bounce_buffer);
};

FileData::FileData(options_t *test_opts)
{
    this->test_opts = test_opts;
    this->fd = open(test_opts->file_name.c_str(), O_RDWR);
    if (test_opts->mmap_protocol >= MMAP) {
        this->base_address = mmap(NULL, test_opts->block_size * test_opts->num_blocks, PROT_WRITE | PROT_READ, MAP_SHARED, this->fd, 0);
    }
}

void FileData::get_block(unsigned block_id, void *bounce_buffer)
{
    assert(test_opts->mmap_protocol == READ);
    double start_time = ucs_get_accurate_time();
    pread(this->fd, bounce_buffer, test_opts->block_size, block_id * test_opts->block_size);
    ucs_trace("Read block of size: %lu in %.0f ns\n", test_opts->block_size, (ucs_get_accurate_time() - start_time)* 1e9);
}

void* FileData::get_block(unsigned block_id)
{
    assert((test_opts->mmap_protocol == MMAP) || (test_opts->mmap_protocol == REGISTER));
    return UCS_PTR_BYTE_OFFSET(this->base_address, block_id * test_opts->block_size);
}

FileData::~FileData()
{
    close(this->fd);
    if (test_opts->mmap_protocol >= MMAP) {
        munmap(this->base_address, test_opts->block_size * test_opts->num_blocks);
    }
}

/**
 * @brief UCX implementation benchmark
 */
class Ucx
{
private:
    options_t                    *test_opts;
    ucp_context_h                context;
    ucp_worker_h                 worker;
    ucp_ep_h                     endpoint;
    std::set<ucp_ep_h>           reverse_ep;
    ucp_listener_h               listener;
    FileData                     *file_data;
    volatile unsigned            received_blocks = 0;
    ucp_mem_h                    memory;
    void                         *memory_address;
    double                       start_time;
    volatile int                 completed_requests;
    size_t                       request_size;
    void                         *requests;
    struct {
        double min_latency;
        double max_latency;
        double total_time;
    }                            benchmark_result;

    void init_context();
    void init_memory();
    void init_worker();
    void init_listener();
    void connect();
    void prepare_client();
    void prepare_server();
    bool is_client();
    void end_iteration();
    void fetch_blocks(unsigned *block_ids);
    void close();

    static void error_handler(void *arg, ucp_ep_h ep, ucs_status_t status);
    static void connect_callback(ucp_conn_request_h conn_req, void *arg);
    static ucs_status_t server_am_handle(void *arg, const void *header,
                                         size_t header_length,
                                         void *data, size_t length,
                                         const ucp_am_recv_param_t *param);
    static ucs_status_t client_am_handle(void *arg, const void *header,
                                         size_t header_length,
                                         void *data, size_t length,
                                         const ucp_am_recv_param_t *param);
    static void client_am_data_recv_handle(void *request,
                                           ucs_status_t status,
                                           size_t length, void *user_data);

public:
    Ucx(options_t *test_opts);
    void run_benchmark();
    void init();
    ~Ucx();
};

void Ucx::init() {
    init_context();
    init_worker();

    if (is_client()) {
        connect();
        prepare_client();
    } else {
        init_listener();
        prepare_server();
    }
}

bool Ucx::is_client() {
    return !this->test_opts->server_address.empty();
}

void Ucx::init_context() {
    ucp_params_t params = {0};
    ucp_context_attr_t attr = {0};

    params.field_mask = UCP_PARAM_FIELD_FEATURES;
    params.features   = UCP_FEATURE_AM;

    ucs_status_t status = ucp_init(&params, NULL, &this->context);

    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to init context: ") + ucs_status_string(status));
    }

    attr.field_mask = UCP_ATTR_FIELD_REQUEST_SIZE;
    status = ucp_context_query(this->context, &attr);

    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to query context: ") + ucs_status_string(status));
    }

    request_size = attr.request_size;

    if (is_client()) {
        requests = ucs_malloc(attr.request_size, "External request");
    } else {
        requests = ucs_malloc(attr.request_size * test_opts->num_outstanding, "External requests");
    }
}

void Ucx::init_memory() {
    ucp_mem_map_params params = {0};
    ucp_mem_attr attr = {0};

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS  ;
    params.length     = this->test_opts->block_size * this->test_opts->num_outstanding;
    params.flags      = UCP_MEM_MAP_ALLOCATE;

    ucs_status_t status = ucp_mem_map(this->context, &params, &this->memory);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to init memory: ") + ucs_status_string(status));
    }

    attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
    status = ucp_mem_query(this->memory, &attr);
    if (status != UCS_OK) {
        throw std::runtime_error(ucs_status_string(status));
    }

    this->memory_address = attr.address;
}

void Ucx::init_worker() {
    ucp_worker_params_t params = {0};

    ucs_status_t status = ucp_worker_create(this->context, &params, &this->worker);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to init worker: ") + ucs_status_string(status));
    }
}

void Ucx::error_handler(void *arg, ucp_ep_h ep, ucs_status_t status)
{
    Ucx* ucx = reinterpret_cast<Ucx*>(arg);

    if (ucx->is_client()) {
        keep_running = false;
        throw std::runtime_error(std::string("Endpoint error: ") + ucs_status_string(status));
    } else {
        ucx->reverse_ep.erase(ep);
    }
}

void Ucx::connect_callback(ucp_conn_request_h conn_req, void *arg)
{
    ucp_ep_params params;
    ucp_ep_h ep;

    params.field_mask      = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                             UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                             UCP_EP_PARAM_FIELD_ERR_HANDLER;
    params.conn_request    = conn_req;
    params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
    params.err_handler.cb  = error_handler;
    params.err_handler.arg = arg;

    Ucx* ucx = reinterpret_cast<Ucx*>(arg);

    ucs_status_t status = ucp_ep_create(ucx->worker, &params, &ep);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to init endpoint: ") + ucs_status_string(status));
    }

    ucx->reverse_ep.insert(ep);
}

void Ucx::init_listener() {
    struct sockaddr_in listen_addr = {0};
    ucp_listener_params_t listener_params = {0};
    ucp_listener_attr_t attrs = {0};

    listen_addr.sin_family      = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port        = htons(this->test_opts->port_num);


    listener_params.field_mask       = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                       UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
    listener_params.sockaddr.addr    = (const struct sockaddr*)&listen_addr;
    listener_params.sockaddr.addrlen = sizeof(listen_addr);
    listener_params.conn_handler.cb  = connect_callback;
    listener_params.conn_handler.arg = this;

    ucs_status_t status = ucp_listener_create(this->worker, &listener_params, &this->listener);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to init listener: ") + ucs_status_string(status));
    }

    attrs.field_mask = UCP_LISTENER_ATTR_FIELD_SOCKADDR;
    ucp_listener_query(this->listener, &attrs);

    struct sockaddr_in* in = (struct sockaddr_in*)&attrs.sockaddr;
    printf("Started listener on %s:%d\n", inet_ntoa(in->sin_addr), htons(in->sin_port));
}

ucs_status_t Ucx::server_am_handle(void *arg, const void *header,
                                   size_t header_length,
                                   void *data, size_t length,
                                   const ucp_am_recv_param_t *param)
{
    ucp_request_param_t request_param = {0};
    request_param.op_attr_mask = UCP_OP_ATTR_FIELD_REQUEST;
    Ucx* ucx = reinterpret_cast<Ucx*>(arg);

    for (int o = 0; o < ucx->test_opts->num_outstanding; o++)
    {
        unsigned block_id = ((const unsigned*)header)[o];
        void *block_address;
        if (ucx->test_opts->mmap_protocol == READ) {
            ucx->file_data->get_block(block_id, UCS_PTR_BYTE_OFFSET(ucx->memory_address, o * ucx->test_opts->block_size));
            block_address = ucx->memory_address;
        } else {
            block_address = ucx->file_data->get_block(block_id);
        }

        request_param.request = UCS_PTR_BYTE_OFFSET(ucx->requests, (o + 1) * ucx->request_size);

        ucp_am_send_nbx(param->reply_ep, 1, NULL, 0, block_address, ucx->test_opts->block_size, &request_param);
    }

    return UCS_OK;
}

void Ucx::client_am_data_recv_handle(void *request,
                                     ucs_status_t status,
                                     size_t length, void *user_data)
{
    ucp_request_free(request);
    Ucx* ucx = reinterpret_cast<Ucx*>(user_data);

    ucx->end_iteration();
}

ucs_status_t Ucx::client_am_handle(void *arg, const void *header,
                                   size_t header_length,
                                   void *data, size_t length,
                                   const ucp_am_recv_param_t *param)
{
    Ucx* ucx = reinterpret_cast<Ucx*>(arg);
    if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
        ucx->end_iteration();
    } else {
        ucp_request_param_t param;
        param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
        param.cb.recv_am   = client_am_data_recv_handle;
        param.user_data    = arg;
        ucp_am_recv_data_nbx(ucx->worker, data,
            UCS_PTR_BYTE_OFFSET(ucx->memory_address, ucx->completed_requests * ucx->test_opts->block_size),
            ucx->test_opts->block_size, &param);
    }
    return UCS_OK;
}

void Ucx::prepare_server() {
    this->file_data = new FileData(this->test_opts);

    if (test_opts->mmap_protocol == REGISTER) {
        // Register whole file
        ucp_mem_map_params params = {0};
        params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                             UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                             UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        params.address     = file_data->base_address;
        params.length      = test_opts->block_size * test_opts->num_blocks;
        params.memory_type = UCS_MEMORY_TYPE_HOST;

        ucs_status_t status = ucp_mem_map(this->context, &params, &this->memory);
        if (status != UCS_OK) {
            throw std::runtime_error(std::string("Failed to register file: ") + ucs_status_string(status));
        }
    } else {
        // Prepare bounce buffers.
        init_memory();
    }

    ucp_am_handler_param_t params = {0};

    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                        UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                        UCP_AM_HANDLER_PARAM_FIELD_CB |
                        UCP_AM_HANDLER_PARAM_FIELD_ARG;
    params.id         = 1;
    params.cb         = server_am_handle;
    params.arg        = this;
    params.flags      = UCP_AM_FLAG_WHOLE_MSG;

    ucp_worker_set_am_recv_handler(this->worker, &params);
}

void Ucx::prepare_client() {
    init_memory();

    ucp_am_handler_param_t params = {0};

    params.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                        UCP_AM_HANDLER_PARAM_FIELD_FLAGS |
                        UCP_AM_HANDLER_PARAM_FIELD_CB |
                        UCP_AM_HANDLER_PARAM_FIELD_ARG;
    params.id         = 1;
    params.cb         = client_am_handle;
    params.arg        = this;
    params.flags      git = UCP_AM_FLAG_WHOLE_MSG;

    ucp_worker_set_am_recv_handler(this->worker, &params);
}

void Ucx::connect() {
    struct sockaddr_in remote_addr = {0};
    ucp_ep_params params = {0};

    remote_addr.sin_family      = AF_INET;
    remote_addr.sin_port        = htons(this->test_opts->port_num);
    inet_pton(AF_INET, this->test_opts->server_address.c_str(), &remote_addr.sin_addr);

    params.field_mask       = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                              UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | UCP_EP_PARAM_FIELD_ERR_HANDLER;
    params.sockaddr.addr    = (const struct sockaddr*)&remote_addr;
    params.sockaddr.addrlen = sizeof(remote_addr);
    params.flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    params.err_handler.cb   = error_handler;
    params.err_handler.arg  = this;
    params.err_mode         = UCP_ERR_HANDLING_MODE_PEER;

    ucs_status_t status = ucp_ep_create(this->worker, &params, &this->endpoint);
    if (status != UCS_OK) {
        throw std::runtime_error(std::string("Failed to init endpoint: ") + ucs_status_string(status));
    }
}

void Ucx::fetch_blocks(unsigned *block_ids)
{
    ucp_request_param_t param = {0};

    param.op_attr_mask  = UCP_OP_ATTR_FIELD_FLAGS | UCP_OP_ATTR_FIELD_REQUEST;
    param.flags         = UCP_AM_SEND_FLAG_REPLY | UCP_AM_SEND_FLAG_EAGER;
    param.request       = UCS_PTR_BYTE_OFFSET(this->requests, this->request_size);
    ucp_am_send_nbx(this->endpoint, 1, block_ids, sizeof(unsigned) * test_opts->num_outstanding, NULL, 0, &param);
}

void Ucx::end_iteration()
{
    completed_requests++;
    received_blocks++;
    if (completed_requests == test_opts->num_outstanding) {
        double duration = ucs_get_accurate_time() - this->start_time;
        if (duration < benchmark_result.min_latency) {
            benchmark_result.min_latency = duration;
        } else if (duration > benchmark_result.max_latency) {
            benchmark_result.max_latency = duration;
        }
        benchmark_result.total_time += duration;
        printf("[%d/%d] Received %d %lu bytes blocks of total size: %lu in %.0f usec. Bandwidth: %.0f Mb/s\n",
                    received_blocks, test_opts->num_blocks * test_opts->iter_count,
                    test_opts->num_outstanding, test_opts->block_size, test_opts->block_size*test_opts->num_outstanding,
                    duration * 1e6, (test_opts->num_outstanding * test_opts->block_size) / (1024.0 * 1024.0) / duration);
    }
}

void Ucx::run_benchmark()
{
    if (is_client()) {
        unsigned block_ids[test_opts->num_outstanding];
        for (unsigned i = 0; i < test_opts->iter_count; i++) {
            benchmark_result = {0};
            benchmark_result.min_latency = DBL_MAX;
            for (unsigned b = 0; b < test_opts->num_blocks; b += test_opts->num_outstanding) {
                // Submit n outstanding requests
                completed_requests = 0;
                start_time = ucs_get_accurate_time();
                for (int o = 0; o < test_opts->num_outstanding; o++) {
                    if (test_opts->random_access) {
                        int block_id;
                        ucs_rand_range(0, test_opts->num_blocks - 1, &block_id);
                        block_ids[o] = block_id;
                    } else {
                        block_ids[o] = (b + o) % test_opts->num_blocks;
                    }
                }
                fetch_blocks(block_ids);
                while ((completed_requests < test_opts -> num_outstanding) && keep_running) {
                        ucp_worker_progress(this->worker);
                }
            }
            size_t total_size = test_opts->block_size * received_blocks;
            printf("Fetched file of size: %lu in %.3f seconds. Per block statistics: \n", total_size, benchmark_result.total_time);
            printf("Min lat: %.0f usec, avg lat: %.0f usec, max lat: %.0f usec. Avg bw: %.0f Mb/s \n", benchmark_result.min_latency * 1e6,
                                                                                        benchmark_result.total_time * 1e6 / received_blocks,
                                                                                        benchmark_result.max_latency * 1e6,
                                                                                        total_size / (1024.0 * 1024.0) / benchmark_result.total_time);
        }
    } else {
        while(keep_running) {
            ucp_worker_progress(this->worker);
        }
    }
}

Ucx::Ucx(options_t *test_opts)
{
    this->test_opts  = test_opts;
}

void Ucx::close() {
    ucp_request_param_t param;
    param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
    param.flags        = UCP_EP_CLOSE_FLAG_FORCE;

    if (is_client()) {
        ucp_ep_close_nbx(this->endpoint, &param);
    } else {
        delete this->file_data;

        for (ucp_ep_h ep : this->reverse_ep) {
            ucp_ep_close_nbx(this->endpoint, &param);
        }
        this->reverse_ep.clear();
        ucp_listener_destroy(this->listener);
    }

    ucp_mem_unmap(this->context, this->memory);
    ucs_free(this->requests);
    ucp_worker_destroy(this->worker);
    ucp_cleanup(this->context);
}

Ucx::~Ucx()
{
    close();
}


static void print_info(int argc, char **argv)
{
    // Process ID and hostname
    char host[64];
    gethostname(host, sizeof(host));
    std::cout << "Starting file_transfer_demo pid " << getpid() << " on " << host << std::endl;

    // Command line
    std::stringstream cmdline;
    for (int i = 0; i < argc; ++i) {
        cmdline << argv[i] << " ";
    }
    std::cout << "Command line: " << cmdline.str() << std::endl;

    // UCX library path
    Dl_info info;
    int ret = dladdr((void*)ucp_init_version, &info);
    if (ret) {
        std::cout << "UCX library path: " << info.dli_fname << std::endl;
    }
}

static int parse_args(int argc, char **argv, options_t *test_opts)
{
    int c;

    test_opts->port_num        = 47654;
    test_opts->iter_count      = 1;
    test_opts->random_access   = false;
    test_opts->mmap_protocol   = READ;
    test_opts->num_blocks      = 1;
    test_opts->num_outstanding = 1;

    while ((c = getopt(argc, argv, "p:f:s:n:a:i:o:rum:h")) != -1) {
        switch (c) {
        case 'p':
            test_opts->port_num = atoi(optarg);
            break;
        case 'f':
            test_opts->file_name = std::string(optarg);
            break;
        case 's':
            test_opts->block_size = atol(optarg);
            break;
        case 'n':
            test_opts->num_blocks = atoi(optarg);
            break;
        case 'a':
            test_opts->server_address = std::string(optarg);
            break;
        case 'i':
            test_opts->iter_count = atoi(optarg);
            break;
        case 'o':
            test_opts->num_outstanding = atoi(optarg);
            break;
        case 'r':
            test_opts->random_access = true;
            break;
        case 'm':
            if (std::string(optarg).compare("mmap") == 0) {
                test_opts->mmap_protocol = MMAP;
            } else  if (std::string(optarg).compare("register") == 0) {
                test_opts->mmap_protocol = REGISTER;
            }
            break;
        case 'h':
        default:
            std::cout << "Usage: file_transfer_demo [options]" << std::endl;
            std::cout << "Supported options are:" << std::endl;
            std::cout << "  -a <address>                Address of a remote server" << std::endl;
            std::cout << "  -p <port>                   TCP port number to use (default: 47654)" << std::endl;
            std::cout << "  -f <file_name>              File name to mmap" << std::endl;
            std::cout << "  -s <block_size>             Size of an individual block (default: 1m)" << std::endl;
            std::cout << "  -n <num_blocks>             Number of blocks in file (default: 1)" << std::endl;
            std::cout << "  -i <iterations-count>       Number of iterations to run communication (default: 1)" << std::endl;
            std::cout << "  -m <map_protocol>           What map protocol to use (read (default), mmap, register)." << std::endl;
            std::cout << "  -o <num_outstanding>        Number of outstanding operations. (default: 1)" << std::endl;
            std::cout << "  -r                          Access blocks in random order" << std::endl;
            return -1;
        }
    }

    return 0;
}

int main(int argc, char **argv)
{
    options_t test_opts;

    int ret;
    signal(SIGINT, [](int) { keep_running = false; });

    print_info(argc, argv);
    ret = parse_args(argc, argv, &test_opts);
    if (ret < 0) {
        return ret;
    }

    Ucx* benchmark = new Ucx(&test_opts);

    try {
        benchmark->init();
        benchmark->run_benchmark();
    } catch( const std::exception & ex ) {
        std::cerr << ex.what() << std::endl;
    }

    delete benchmark;

    return 0;
}
