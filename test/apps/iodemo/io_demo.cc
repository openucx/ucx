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
#include <queue>
#include <algorithm>
#include <limits>
#include <malloc.h>
#include <dlfcn.h>
#include <set>

#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#define ALIGNMENT           4096
#define BUSY_PROGRESS_COUNT 1000

/* IO operation type */
typedef enum {
    IO_READ,
    IO_WRITE,
    IO_OP_MAX,
    IO_COMP_MIN  = IO_OP_MAX,
    IO_READ_COMP = IO_COMP_MIN,
    IO_WRITE_COMP
} io_op_t;

static const char *io_op_names[] = {
    "read",
    "write",
    "read completion",
    "write completion"
};

/* test options */
typedef struct {
    std::vector<const char*> servers;
    int                      port_num;
    double                   connect_timeout;
    double                   client_timeout;
    long                     retries;
    double                   retry_interval;
    double                   client_runtime_limit;
    double                   print_interval;
    size_t                   iomsg_size;
    size_t                   min_data_size;
    size_t                   max_data_size;
    size_t                   chunk_size;
    long                     iter_count;
    long                     window_size;
    long                     conn_window_size;
    std::vector<io_op_t>     operations;
    unsigned                 random_seed;
    size_t                   num_offcache_buffers;
    bool                     verbose;
    bool                     validate;
    bool                     use_am;
    ucs_memory_type_t        memory_type;
} options_t;

#define LOG_PREFIX  "[DEMO]"
#define LOG         UcxLog(LOG_PREFIX)
#define VERBOSE_LOG UcxLog(LOG_PREFIX, _test_opts.verbose)

template<class BufferType, bool use_offcache = false> class ObjectPool {
public:
    ObjectPool(size_t buffer_size, const std::string &name,
               size_t offcache = 0) :
        _buffer_size(buffer_size), _num_allocated(0), _name(name)
    {
        for (size_t i = 0; i < offcache; ++i) {
            _offcache_queue.push(get_free());
        }
    }

    ~ObjectPool()
    {
        while (!_offcache_queue.empty()) {
            _free_stack.push_back(_offcache_queue.front());
            _offcache_queue.pop();
        }

        if (_num_allocated != _free_stack.size()) {
            LOG << (_num_allocated - _free_stack.size())
                << " buffers were not released from " << _name;
        }

        for (size_t i = 0; i < _free_stack.size(); i++) {
            delete _free_stack[i];
        }
    }

    inline BufferType *get()
    {
        BufferType *item = get_free();

        if (use_offcache && !_offcache_queue.empty()) {
            _offcache_queue.push(item);
            item = _offcache_queue.front();
            _offcache_queue.pop();
        }

        return item;
    }

    inline void put(BufferType *item)
    {
        _free_stack.push_back(item);
    }

    inline size_t allocated() const {
        return _num_allocated;
    }

    virtual ucs_memory_type_t memory_type() const
    {
        return UCS_MEMORY_TYPE_HOST;
    }

protected:
    size_t buffer_size() const
    {
        return _buffer_size;
    }

    virtual BufferType *construct() = 0;

private:
    inline BufferType *get_free()
    {
        BufferType *item;

        if (_free_stack.empty()) {
            item = construct();
            _num_allocated++;
        } else {
            item = _free_stack.back();
            _free_stack.pop_back();
        }
        return item;
    }

private:
    size_t                   _buffer_size;
    std::vector<BufferType*> _free_stack;
    std::queue<BufferType*>  _offcache_queue;
    uint32_t                 _num_allocated;
    std::string              _name;
};

template<class BufferType, bool use_offcache = false>
class MemoryPool : public ObjectPool<BufferType, use_offcache> {
public:
    MemoryPool(size_t buffer_size, const std::string &name,
               size_t offcache = 0) :
        ObjectPool<BufferType, use_offcache>::ObjectPool(buffer_size, name,
                                                         offcache)
    {
    }

public:
    virtual BufferType *construct()
    {
        return new BufferType(this->buffer_size(), *this);
    }
};

template<typename BufferType>
class BufferMemoryPool : public ObjectPool<BufferType, true> {
public:
    BufferMemoryPool(size_t buffer_size, const std::string &name,
                     ucs_memory_type_t memory_type, size_t offcache = 0) :
        ObjectPool<BufferType, true>(buffer_size, name, offcache),
        _memory_type(memory_type)
    {
    }

    virtual BufferType *construct()
    {
        return BufferType::allocate(this->buffer_size(), *this, _memory_type);
    }

    virtual ucs_memory_type_t memory_type() const
    {
        return _memory_type;
    }

private:
    ucs_memory_type_t _memory_type;
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

    template <typename T>
    static inline T rand(T min = std::numeric_limits<T>::min(),
                         T max = std::numeric_limits<T>::max() - 1) {
        return rand(_seed, min, max);
    }

    template <typename T>
    static inline T rand(unsigned &seed, T min = std::numeric_limits<T>::min(),
                         T max = std::numeric_limits<T>::max() - 1) {
        seed = (seed * _A + _C) & _M;
        /* To resolve that LCG returns alternating even/odd values */
        if (max - min == 1) {
            return (seed & 0x100) ? max : min;
        } else {
            return T(seed) % (max - min + 1) + min;
        }
    }

    template <typename unsigned_type>
    static inline unsigned_type urand(unsigned_type max)
    {
        assert(max < std::numeric_limits<unsigned_type>::max());
        assert(unsigned_type(0) == std::numeric_limits<unsigned_type>::min());

        return rand(_seed, unsigned_type(0), max - 1);
    }

    static void *get_host_fill_buffer(void *buffer, size_t size,
                                      ucs_memory_type_t memory_type)
    {
        static std::vector<uint8_t> _buffer;

        if (memory_type == UCS_MEMORY_TYPE_CUDA) {
            _buffer.resize(size);
            return _buffer.data();
        }

        return buffer;
    }

    static void fill_commit(void *buffer, void *fill_buffer, size_t size,
                            ucs_memory_type_t memory_type)
    {
#ifdef HAVE_CUDA
        if (memory_type == UCS_MEMORY_TYPE_CUDA) {
            cudaMemcpy(buffer, fill_buffer, size, cudaMemcpyDefault);
        }
#endif
    }

    static inline void fill(unsigned &seed, void *buffer, size_t size,
                            ucs_memory_type_t memory_type)
    {
        void *fill_buffer = get_host_fill_buffer(buffer, size, memory_type);
        size_t body_count = size / sizeof(uint64_t);
        size_t tail_count = size & (sizeof(uint64_t) - 1);
        uint64_t *body    = reinterpret_cast<uint64_t*>(fill_buffer);
        uint8_t *tail     = reinterpret_cast<uint8_t*>(body + body_count);

        fill(seed, body, body_count);
        fill(seed, tail, tail_count);

        fill_commit(buffer, fill_buffer, size, memory_type);
    }

    static const void *get_host_validate_buffer(const void *buffer, size_t size,
                                                ucs_memory_type_t memory_type)
    {
#ifdef HAVE_CUDA
        static std::vector<uint8_t> _buffer;

        if (memory_type == UCS_MEMORY_TYPE_CUDA) {
            _buffer.resize(size);
            cudaMemcpy(_buffer.data(), buffer, size, cudaMemcpyDefault);
            return _buffer.data();
        }
#endif
        return buffer;
    }

    static inline size_t validate(unsigned &seed, const void *buffer,
                                  size_t size, ucs_memory_type_t memory_type)
    {
        size_t body_count    = size / sizeof(uint64_t);
        size_t tail_count    = size & (sizeof(uint64_t) - 1);
        const uint64_t *body = reinterpret_cast<const uint64_t*>(
                get_host_validate_buffer(buffer, size, memory_type));
        const uint8_t *tail  = reinterpret_cast<const uint8_t*>(body + body_count);

        size_t err_pos = validate(seed, body, body_count);
        if (err_pos < body_count) {
            return err_pos * sizeof(body[0]);
        }

        err_pos = validate(seed, tail, tail_count);
        if (err_pos < tail_count) {
            return (body_count * sizeof(body[0])) + (err_pos * sizeof(tail[0]));
        }

        return size;
    }

private:
    template<typename T>
    static inline void fill(unsigned &seed, T *buffer, size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            buffer[i] = rand<T>(seed);
        }
    }

    template<typename T>
    static inline size_t validate(unsigned &seed, const T *buffer, size_t count)
    {
        for (size_t i = 0; i < count; ++i) {
            if (buffer[i] != rand<T>(seed)) {
                return i;
            }
        }

        return count;
    }

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
public:
    /* IO header */
    typedef struct {
        uint32_t    sn;
        uint8_t     op;
        uint64_t    data_size;
    } iomsg_t;

protected:
    typedef enum {
        XFER_TYPE_SEND,
        XFER_TYPE_RECV
    } xfer_type_t;

    class Buffer {
    public:
        Buffer(void *buffer, size_t size, BufferMemoryPool<Buffer> &pool,
               ucs_memory_type_t memory_type = UCS_MEMORY_TYPE_HOST) :
            _capacity(size),
            _buffer(buffer),
            _size(0),
            _pool(pool),
            _memory_type(memory_type)
        {
        }

        static Buffer *allocate(size_t size, BufferMemoryPool<Buffer> &pool,
                                ucs_memory_type_t memory_type)
        {
#ifdef HAVE_CUDA
            cudaError_t cerr;
#endif
            void *buffer;

            switch (memory_type) {
#ifdef HAVE_CUDA
            case UCS_MEMORY_TYPE_CUDA:
                cerr = cudaMalloc(&buffer, size);
                if (cerr != cudaSuccess) {
                    buffer = NULL;
                }
                break;
            case UCS_MEMORY_TYPE_CUDA_MANAGED:
                cerr = cudaMallocManaged(&buffer, size, cudaMemAttachGlobal);
                if (cerr != cudaSuccess) {
                    buffer = NULL;
                }
                break;
#endif
            case UCS_MEMORY_TYPE_HOST:
                buffer = memalign(ALIGNMENT, size);
                break;
            default:
                LOG << "ERROR: Unsupported memory type requested: "
                    << ucs_memory_type_names[memory_type];
                abort();
            }
            if (buffer == NULL) {
                throw std::bad_alloc();
            }

            return new Buffer(buffer, size, pool, memory_type);
        }

        ~Buffer()
        {
            switch (_memory_type) {
#ifdef HAVE_CUDA
            case UCS_MEMORY_TYPE_CUDA:
            case UCS_MEMORY_TYPE_CUDA_MANAGED:
                cudaFree(_buffer);
                break;
#endif
            case UCS_MEMORY_TYPE_HOST:
                free(_buffer);
                break;
            default:
                /* Unreachable - would fail in ctor */
                abort();
            }
        }

        inline size_t capacity() const
        {
            return _capacity;
        }

        void release()
        {
            _pool.put(this);
        }

        inline void *buffer(size_t offset = 0) const
        {
            return (uint8_t*)_buffer + offset;
        }

        inline void resize(size_t size)
        {
            assert(size <= _capacity);
            _size = size;
        }

        inline size_t size() const
        {
            return _size;
        }

    public:
        const size_t             _capacity;

    private:
        void                     *_buffer;
        size_t                   _size;
        BufferMemoryPool<Buffer> &_pool;
        ucs_memory_type_t        _memory_type;
    };

    class BufferIov {
    public:
        BufferIov(size_t size, MemoryPool<BufferIov> &pool) :
            _memory_type(UCS_MEMORY_TYPE_UNKNOWN), _pool(pool)
        {
            _iov.reserve(size);
        }

        size_t size() const {
            return _iov.size();
        }

        void init(size_t data_size, BufferMemoryPool<Buffer> &chunk_pool,
                  uint32_t sn, bool validate)
        {
            assert(_iov.empty());

            _memory_type  = chunk_pool.memory_type();
            Buffer *chunk = chunk_pool.get();
            _iov.resize(get_chunk_cnt(data_size, chunk->capacity()));

            size_t remaining = init_chunk(0, chunk, data_size);
            for (size_t i = 1; i < _iov.size(); ++i) {
                remaining = init_chunk(i, chunk_pool.get(), remaining);
            }

            assert(remaining == 0);

            if (validate) {
                fill_data(sn, _memory_type);
            }
        }

        inline Buffer &operator[](size_t i) const
        {
            return *_iov[i];
        }

        void release() {
            while (!_iov.empty()) {
                _iov.back()->release();
                _iov.pop_back();
            }

            _pool.put(this);
        }

        inline size_t validate(unsigned seed) const {
            assert(!_iov.empty());

            for (size_t iov_err_pos = 0, i = 0; i < _iov.size(); ++i) {
                size_t buf_err_pos = IoDemoRandom::validate(seed,
                                                            _iov[i]->buffer(),
                                                            _iov[i]->size(),
                                                            _memory_type);
                iov_err_pos       += buf_err_pos;
                if (buf_err_pos < _iov[i]->size()) {
                    return iov_err_pos;
                }
            }

            return _npos;
        }

        inline size_t npos() const {
            return _npos;
        }

    private:
        size_t init_chunk(size_t i, Buffer *chunk, size_t remaining) {
            _iov[i] = chunk;
            _iov[i]->resize(std::min(_iov[i]->capacity(), remaining));
            return remaining - _iov[i]->size();
        }

        void fill_data(unsigned seed, ucs_memory_type_t memory_type)
        {
            for (size_t i = 0; i < _iov.size(); ++i) {
                IoDemoRandom::fill(seed, _iov[i]->buffer(), _iov[i]->size(),
                                   memory_type);
            }
        }

        static const size_t    _npos = static_cast<size_t>(-1);
        ucs_memory_type_t _memory_type;
        std::vector<Buffer*>   _iov;
        MemoryPool<BufferIov>& _pool;
    };

    /* Asynchronous IO message */
    class IoMessage : public UcxCallback {
    public:
        IoMessage(size_t io_msg_size, MemoryPool<IoMessage>& pool) :
            _buffer(malloc(io_msg_size)), _pool(pool),
            _io_msg_size(io_msg_size) {

            if (_buffer == NULL) {
                throw std::bad_alloc();
            }
        }

        void init(io_op_t op, uint32_t sn, size_t data_size, bool validate) {
            iomsg_t *m = reinterpret_cast<iomsg_t *>(_buffer);

            m->sn        = sn;
            m->op        = op;
            m->data_size = data_size;
            if (validate) {
                void *tail       = reinterpret_cast<void*>(m + 1);
                size_t tail_size = _io_msg_size - sizeof(*m);
                IoDemoRandom::fill(sn, tail, tail_size, UCS_MEMORY_TYPE_HOST);
            }
        }

        ~IoMessage() {
            free(_buffer);
        }

        virtual void operator()(ucs_status_t status) {
            _pool.put(this);
        }

        void *buffer() const {
            return _buffer;
        }

        const iomsg_t* msg() const{
            return reinterpret_cast<iomsg_t*>(buffer());
        }

    private:
        void*                  _buffer;
        MemoryPool<IoMessage>& _pool;
        size_t                 _io_msg_size;
    };

    class SendCompleteCallback : public UcxCallback {
    public:
        SendCompleteCallback(size_t buffer_size,
                             MemoryPool<SendCompleteCallback>& pool) :
            _op_counter(NULL), _counter(0), _iov(NULL), _io_msg(NULL),
            _pool(pool) {
        }

        void init(BufferIov* iov, long* op_counter, IoMessage *io_msg = NULL) {
            _op_counter = op_counter;
            _counter    = iov->size();
            _iov        = iov;
            _io_msg     = io_msg;
            assert(_counter > 0);
        }

        virtual void operator()(ucs_status_t status) {
            if (--_counter > 0) {
                return;
            }

            if (_op_counter != NULL) {
                ++(*_op_counter);
            }

            if (_io_msg != NULL) {
                (*_io_msg)(status);
            }

            _iov->release();
            _pool.put(this);
        }

    private:
        long*                             _op_counter;
        size_t                            _counter;
        BufferIov*                        _iov;
        IoMessage*                        _io_msg;
        MemoryPool<SendCompleteCallback>& _pool;
    };

    P2pDemoCommon(const options_t &test_opts) :
        UcxContext(test_opts.iomsg_size, test_opts.connect_timeout,
                   test_opts.use_am),
        _test_opts(test_opts),
        _io_msg_pool(test_opts.iomsg_size, "io messages"),
        _send_callback_pool(0, "send callbacks"),
        _data_buffers_pool(get_chunk_cnt(test_opts.max_data_size,
                                         test_opts.chunk_size),
                           "data iovs"),
        _data_chunks_pool(test_opts.chunk_size, "data chunks",
                          test_opts.memory_type)
    {
    }

    const options_t& opts() const {
        return _test_opts;
    }

    inline size_t get_data_size() {
        return IoDemoRandom::rand(opts().min_data_size,
                                  opts().max_data_size);
    }

    bool send_io_message(UcxConnection *conn, io_op_t op, uint32_t sn,
                         size_t data_size, bool validate) {
        IoMessage *m = _io_msg_pool.get();
        m->init(op, sn, data_size, validate);
        return send_io_message(conn, m);
    }

    void send_recv_data(UcxConnection* conn, const BufferIov &iov, uint32_t sn,
                        xfer_type_t send_recv_data,
                        UcxCallback* callback = EmptyCallback::get()) {
        for (size_t i = 0; i < iov.size(); ++i) {
            if (send_recv_data == XFER_TYPE_SEND) {
                conn->send_data(iov[i].buffer(), iov[i].size(), sn, callback);
            } else {
                conn->recv_data(iov[i].buffer(), iov[i].size(), sn, callback);
            }
        }
    }

    void send_data(UcxConnection* conn, const BufferIov &iov, uint32_t sn,
                   UcxCallback* callback = EmptyCallback::get()) {
        send_recv_data(conn, iov, sn, XFER_TYPE_SEND, callback);
    }

    void recv_data(UcxConnection* conn, const BufferIov &iov, uint32_t sn,
                   UcxCallback* callback = EmptyCallback::get()) {
        send_recv_data(conn, iov, sn, XFER_TYPE_RECV, callback);
    }

    static uint32_t get_chunk_cnt(size_t data_size, size_t chunk_size) {
        return (data_size + chunk_size - 1) / chunk_size;
    }

    static void validate(const BufferIov& iov, unsigned seed) {
        assert(iov.size() != 0);

        size_t err_pos = iov.validate(seed);
        if (err_pos != iov.npos()) {
            LOG << "ERROR: iov data corruption at " << err_pos << " position";
            abort();
        }
    }

    static void validate(const iomsg_t *msg, size_t iomsg_size) {
        unsigned seed   = msg->sn;
        const void *buf = msg + 1;
        size_t buf_size = iomsg_size - sizeof(*msg);

        size_t err_pos = IoDemoRandom::validate(seed, buf, buf_size,
                                                UCS_MEMORY_TYPE_HOST);
        if (err_pos < buf_size) {
            LOG << "ERROR: io msg data corruption at " << err_pos << " position";
            abort();
        }
    }

    static void validate(const iomsg_t *msg, uint32_t sn, size_t iomsg_size) {
        if (sn != msg->sn) {
            LOG << "ERROR: io msg sn mismatch " << sn << " != " << msg->sn;
            abort();
        }

        validate(msg, iomsg_size);
    }

private:
    bool send_io_message(UcxConnection *conn, IoMessage *msg) {
        VERBOSE_LOG << "sending IO " << io_op_names[msg->msg()->op] << ", sn "
                    << msg->msg()->sn << " size " << sizeof(iomsg_t);

        /* send IO_READ_COMP as a data since the transaction must be matched
         * by sn on receiver side */
        if (msg->msg()->op == IO_READ_COMP) {
            return conn->send_data(msg->buffer(), opts().iomsg_size,
                                   msg->msg()->sn, msg);
        } else {
            return conn->send_io_message(msg->buffer(), opts().iomsg_size, msg);
        }
    }

protected:
    const options_t                  _test_opts;
    MemoryPool<IoMessage>            _io_msg_pool;
    MemoryPool<SendCompleteCallback> _send_callback_pool;
    MemoryPool<BufferIov>            _data_buffers_pool;
    BufferMemoryPool<Buffer> _data_chunks_pool;
};

class DemoServer : public P2pDemoCommon {
public:
    // sends an IO response when done
    class IoWriteResponseCallback : public UcxCallback {
    public:
        IoWriteResponseCallback(size_t buffer_size,
            MemoryPool<IoWriteResponseCallback>& pool) :
            _server(NULL), _conn(NULL), _op_cnt(NULL), _chunk_cnt(0), _sn(0),
            _iov(NULL), _pool(pool) {
        }

        void init(DemoServer *server, UcxConnection* conn, uint32_t sn,
                  BufferIov *iov, long* op_cnt) {
            _server    = server;
            _conn      = conn;
            _op_cnt    = op_cnt;
            _sn        = sn;
            _iov       = iov;
            _chunk_cnt = iov->size();
        }

        virtual void operator()(ucs_status_t status) {
            if (--_chunk_cnt > 0) {
                return;
            }

            if (status == UCS_OK) {
                if (_server->opts().use_am) {
                    IoMessage *m = _server->_io_msg_pool.get();
                    m->init(IO_WRITE_COMP, _sn, 0, _server->opts().validate);
                    _conn->send_am(m->buffer(), _server->opts().iomsg_size,
                                   NULL, 0ul, m);
                } else {
                    _server->send_io_message(_conn, IO_WRITE_COMP, _sn, 0,
                                             _server->opts().validate);
                }
                if (_server->opts().validate) {
                    validate(*_iov, _sn);
                }
            }

            assert(_op_cnt != NULL);
            ++(*_op_cnt);

            _iov->release();
            _pool.put(this);
        }

    private:
        DemoServer*                          _server;
        UcxConnection*                       _conn;
        long*                                _op_cnt;
        uint32_t                             _chunk_cnt;
        uint32_t                             _sn;
        BufferIov*                           _iov;
        MemoryPool<IoWriteResponseCallback>& _pool;
    };

    typedef struct {
        long    read_count;
        long    write_count;
        long    active_conns;
    } state_t;

    DemoServer(const options_t& test_opts) :
        P2pDemoCommon(test_opts), _callback_pool(0, "callbacks") {
        _curr_state.read_count   = 0;
        _curr_state.write_count  = 0;
        _curr_state.active_conns = 0;
        save_prev_state();
    }

    void run() {
        struct sockaddr_in listen_addr;
        memset(&listen_addr, 0, sizeof(listen_addr));
        listen_addr.sin_family      = AF_INET;
        listen_addr.sin_addr.s_addr = INADDR_ANY;
        listen_addr.sin_port        = htons(opts().port_num);

        for (long retry = 1;; ++retry) {
            if (listen((const struct sockaddr*)&listen_addr,
                       sizeof(listen_addr))) {
                break;
            }

            if (retry > opts().retries) {
                return;
            }

            {
                UcxLog log(LOG_PREFIX);
                log << "restarting listener on "
                    << UcxContext::sockaddr_str((struct sockaddr*)&listen_addr,
                                                sizeof(listen_addr))
                    << " in " << opts().retry_interval << " seconds (retry "
                    << retry;

                if (opts().retries < std::numeric_limits<long>::max()) {
                    log << "/" << opts().retries;
                }

                log << ")";
            }

            sleep(opts().retry_interval);
        }

        for (double prev_time = 0.0; ;) {
            try {
                for (size_t i = 0; i < BUSY_PROGRESS_COUNT; ++i) {
                    progress();
                }

                double curr_time = get_time();
                if (curr_time >= (prev_time + opts().print_interval)) {
                    prev_time = curr_time;
                    report_state();
                }
            } catch (const std::exception &e) {
                std::cerr << e.what();
            }
        }
    }

    void handle_io_read_request(UcxConnection* conn, const iomsg_t *msg) {
        // send data
        VERBOSE_LOG << "sending IO read data";
        assert(opts().max_data_size >= msg->data_size);

        BufferIov *iov           = _data_buffers_pool.get();
        SendCompleteCallback *cb = _send_callback_pool.get();

        iov->init(msg->data_size, _data_chunks_pool, msg->sn, opts().validate);
        cb->init(iov, &_curr_state.read_count);

        send_data(conn, *iov, msg->sn, cb);

        // send response as data
        VERBOSE_LOG << "sending IO read response";
        send_io_message(conn, IO_READ_COMP, msg->sn, 0, opts().validate);
    }

    void handle_io_am_read_request(UcxConnection* conn, const iomsg_t *msg) {
        VERBOSE_LOG << "sending AM IO read data";
        assert(opts().max_data_size >= msg->data_size);

        IoMessage *m = _io_msg_pool.get();
        m->init(IO_READ_COMP, msg->sn, msg->data_size, opts().validate);

        BufferIov *iov = _data_buffers_pool.get();
        iov->init(msg->data_size, _data_chunks_pool, msg->sn, opts().validate);

        SendCompleteCallback *cb = _send_callback_pool.get();
        cb->init(iov, &_curr_state.read_count, m);

        assert(iov->size() == 1);

        // Send IO_READ_COMP as AM header and first iov element as payload
        // (note that multi-iov send is not supported for IODEMO with AM yet)
        conn->send_am(m->buffer(), opts().iomsg_size, (*iov)[0].buffer(),
                      (*iov)[0].size(), cb);
    }

    void handle_io_write_request(UcxConnection* conn, const iomsg_t *msg) {
        VERBOSE_LOG << "receiving IO write data";
        assert(msg->data_size != 0);

        BufferIov *iov             = _data_buffers_pool.get();
        IoWriteResponseCallback *w = _callback_pool.get();

        iov->init(msg->data_size, _data_chunks_pool, msg->sn, opts().validate);
        w->init(this, conn, msg->sn, iov, &_curr_state.write_count);

        recv_data(conn, *iov, msg->sn, w);
    }

    void handle_io_am_write_request(UcxConnection* conn, const iomsg_t *msg,
                                    const UcxAmDesc &data_desc) {
        VERBOSE_LOG << "receiving AM IO write data";
        assert(msg->data_size != 0);

        BufferIov *iov             = _data_buffers_pool.get();
        IoWriteResponseCallback *w = _callback_pool.get();

        iov->init(msg->data_size, _data_chunks_pool, msg->sn, opts().validate);
        w->init(this, conn, msg->sn, iov, &_curr_state.write_count);

        assert(iov->size() == 1);

        conn->recv_am_data((*iov)[0].buffer(), (*iov)[0].size(), data_desc, w);
    }

    virtual void dispatch_connection_accepted(UcxConnection* conn) {
        ++_curr_state.active_conns;
    }

    virtual void dispatch_connection_error(UcxConnection *conn) {
        LOG << "disconnecting connection with status "
            << ucs_status_string(conn->ucx_status());
        --_curr_state.active_conns;
        conn->disconnect(new UcxDisconnectCallback(*conn));
    }

    virtual void dispatch_io_message(UcxConnection* conn, const void *buffer,
                                     size_t length) {
        iomsg_t const *msg = reinterpret_cast<const iomsg_t*>(buffer);

        VERBOSE_LOG << "got io message " << io_op_names[msg->op] << " sn "
                    << msg->sn << " data size " << msg->data_size
                    << " conn " << conn;

        if (opts().validate) {
            assert(length == opts().iomsg_size);
            validate(msg, length);
        }

        if (msg->op == IO_READ) {
            handle_io_read_request(conn, msg);
        } else if (msg->op == IO_WRITE) {
            handle_io_write_request(conn, msg);
        } else {
            LOG << "Invalid opcode: " << msg->op;
        }
    }

    virtual void dispatch_am_message(UcxConnection* conn, const void *buffer,
                                     size_t length,
                                     const UcxAmDesc &data_desc) {
        iomsg_t const *msg = reinterpret_cast<const iomsg_t*>(buffer);

        VERBOSE_LOG << "got io (AM) message " << io_op_names[msg->op] << " sn "
                    << msg->sn << " data size " << msg->data_size
                    << " conn " << conn;

        if (opts().validate) {
            assert(length == opts().iomsg_size);
            validate(msg, length);
        }

        if (msg->op == IO_READ) {
            handle_io_am_read_request(conn, msg);
        } else if (msg->op == IO_WRITE) {
            handle_io_am_write_request(conn, msg, data_desc);
        } else {
            LOG << "Invalid opcode: " << msg->op;
        }
    }

private:
    void save_prev_state() {
        _prev_state = _curr_state;
    }

    void report_state() {
        LOG << "read:" << _curr_state.read_count -
                          _prev_state.read_count << " ops, "
            << "write:" << _curr_state.write_count -
                           _prev_state.write_count << " ops, "
            << "active connections:" << _curr_state.active_conns
            << ", buffers:" << _data_buffers_pool.allocated();
        save_prev_state();
    }

private:
    MemoryPool<IoWriteResponseCallback> _callback_pool;
    state_t                             _prev_state;
    state_t                             _curr_state;
};


class DemoClient : public P2pDemoCommon {
private:
    class DisconnectCallback : public UcxCallback {
    public:
        DisconnectCallback(DemoClient &client, UcxConnection &conn) :
            _client(client), _conn(&conn) {
        }

        virtual ~DisconnectCallback() {
            delete _conn;
        }

        virtual void operator()(ucs_status_t status) {
            server_info_t &server_info = _client.get_server_info(_conn);

            _client._num_sent -= get_num_uncompleted(server_info);

            // Remove connection pointer
            _client._server_index_lookup.erase(_conn);

            // Remove active servers entry
            _client.active_servers_remove(server_info.active_index);

            reset_server_info(server_info);
            delete this;
        }

    private:
        DemoClient    &_client;
        UcxConnection *_conn;
    };

public:
    typedef struct {
        UcxConnection* conn;
        long           retry_count;               /* Connect retry counter */
        double         prev_connect_time;         /* timestamp of last connect attempt */
        size_t         active_index;              /* Index in active vector */
        long           num_sent[IO_OP_MAX];       /* Number of sent operations */
        long           num_completed[IO_OP_MAX];  /* Number of completed operations */
        long           prev_completed[IO_OP_MAX]; /* Completed in last report */
    } server_info_t;

    class ConnectCallback : public UcxCallback {
    public:
        ConnectCallback(DemoClient &client, size_t server_idx) :
            _client(client), _server_idx(server_idx)
        {
        }

        virtual void operator()(ucs_status_t status)
        {
            if (status == UCS_OK) {
                _client.connect_succeed(_server_idx);
            } else {
                _client.connect_failed(_server_idx);
            }

            _client._connecting_servers.erase(_server_idx);
            delete this;
        }

    private:
        DemoClient   &_client;
        const size_t _server_idx;
    };

    class IoReadResponseCallback : public UcxCallback {
    public:
        IoReadResponseCallback(size_t buffer_size,
            MemoryPool<IoReadResponseCallback>& pool) :
            _comp_counter(0), _client(NULL),
            _server_index(std::numeric_limits<size_t>::max()),
            _sn(0), _validate(false), _iov(NULL), _buffer(malloc(buffer_size)),
            _buffer_size(buffer_size), _meta_comp_counter(0), _pool(pool) {

            if (_buffer == NULL) {
                throw std::bad_alloc();
            }
        }

        void init(DemoClient *client, size_t server_index,
                  uint32_t sn, bool validate, BufferIov *iov,
                  int meta_comp_counter = 1) {
            /* wait for all data chunks and the read response completion */
            _comp_counter      = iov->size() + meta_comp_counter;
            _client            = client;
            _server_index      = server_index;
            _sn                = sn;
            _validate          = validate;
            _iov               = iov;
            _meta_comp_counter = meta_comp_counter;
        }

        ~IoReadResponseCallback() {
            free(_buffer);
        }

        virtual void operator()(ucs_status_t status) {
            if (--_comp_counter > 0) {
                return;
            }

            assert(_server_index != std::numeric_limits<size_t>::max());
            _client->handle_operation_completion(_server_index, IO_READ);

            if (_validate && (status == UCS_OK)) {
                validate(*_iov, _sn);

                if (_meta_comp_counter != 0) {
                    // With tag API, we also wait for READ_COMP arrival, so need
                    // to validate it. With AM API, READ_COMP arrives as AM
                    // header together with data descriptor, we validate it in
                    // place to avoid unneeded memory copy to this
                    // IoReadResponseCallback _buffer.
                    iomsg_t *msg = reinterpret_cast<iomsg_t*>(_buffer);
                    validate(msg, _sn, _buffer_size);
                }
            }

            _iov->release();
            _pool.put(this);
        }

        void* buffer() {
            return _buffer;
        }

    private:
        long                                _comp_counter;
        DemoClient*                         _client;
        size_t                              _server_index;
        uint32_t                            _sn;
        bool                                _validate;
        BufferIov*                          _iov;
        void*                               _buffer;
        const size_t                        _buffer_size;
        int                                 _meta_comp_counter;
        MemoryPool<IoReadResponseCallback>& _pool;
    };

    DemoClient(const options_t &test_opts) :
        P2pDemoCommon(test_opts),
        _num_active_servers_to_use(0),
        _num_sent(0),
        _num_completed(0),
        _status(OK),
        _start_time(get_time()),
        _read_callback_pool(opts().iomsg_size, "read callbacks")
    {
    }

    typedef enum {
        OK,
        CONN_RETRIES_EXCEEDED,
        RUNTIME_EXCEEDED
    } status_t;

    size_t get_server_index(const UcxConnection *conn) {
        assert(_server_index_lookup.size() == _active_servers.size());

        std::map<const UcxConnection*, size_t>::const_iterator i =
                                                _server_index_lookup.find(conn);
        return (i == _server_index_lookup.end()) ? _server_info.size() :
               i->second;
    }

    server_info_t &get_server_info(const UcxConnection *conn) {
        const size_t server_index = get_server_index(conn);

        assert(server_index < _server_info.size());
        return _server_info[server_index];
    }

    void commit_operation(size_t server_index, io_op_t op) {
        server_info_t& server_info = _server_info[server_index];

        assert(get_num_uncompleted(server_info) < opts().conn_window_size);

        ++server_info.num_sent[op];
        ++_num_sent;
        if (get_num_uncompleted(server_info) == opts().conn_window_size) {
            active_servers_make_unused(server_info.active_index);
        }
    }

    void handle_operation_completion(size_t server_index, io_op_t op) {
        assert(server_index < _server_info.size());
        server_info_t& server_info = _server_info[server_index];

        assert(get_num_uncompleted(server_info) <= opts().conn_window_size);
        assert(_server_index_lookup.find(server_info.conn) !=
               _server_index_lookup.end());
        assert(_num_completed < _num_sent);

        if (get_num_uncompleted(server_info) == opts().conn_window_size) {
            active_servers_make_used(server_info.active_index);
        }

        ++_num_completed;
        ++server_info.num_completed[op];
    }

    size_t do_io_read(size_t server_index, uint32_t sn) {
        server_info_t& server_info = _server_info[server_index];
        size_t data_size           = get_data_size();
        bool validate              = opts().validate;

        if (!send_io_message(server_info.conn, IO_READ, sn, data_size,
                             validate)) {
            return 0;
        }

        commit_operation(server_index, IO_READ);

        BufferIov *iov            = _data_buffers_pool.get();
        IoReadResponseCallback *r = _read_callback_pool.get();

        iov->init(data_size, _data_chunks_pool, sn, validate);
        r->init(this, server_index, sn, validate, iov);

        recv_data(server_info.conn, *iov, sn, r);
        server_info.conn->recv_data(r->buffer(), opts().iomsg_size, sn, r);

        return data_size;
    }

    size_t do_io_read_am(size_t server_index, uint32_t sn) {
        server_info_t& server_info = _server_info[server_index];
        size_t data_size           = get_data_size();

        commit_operation(server_index, IO_READ);

        IoMessage *m = _io_msg_pool.get();
        m->init(IO_READ, sn, data_size, opts().validate);

        server_info.conn->send_am(m->buffer(), opts().iomsg_size, NULL, 0, m);

        return data_size;
    }

    size_t do_io_write(size_t server_index, uint32_t sn) {
        server_info_t& server_info = _server_info[server_index];
        size_t data_size           = get_data_size();
        bool validate              = opts().validate;

        if (!send_io_message(server_info.conn, IO_WRITE, sn, data_size,
                             validate)) {
            return 0;
        }

        commit_operation(server_index, IO_WRITE);

        BufferIov *iov           = _data_buffers_pool.get();
        SendCompleteCallback *cb = _send_callback_pool.get();

        iov->init(data_size, _data_chunks_pool, sn, validate);
        cb->init(iov, NULL);

        VERBOSE_LOG << "sending data " << iov << " size "
                    << data_size << " sn " << sn;
        send_data(server_info.conn, *iov, sn, cb);

        return data_size;
    }

    size_t do_io_write_am(size_t server_index, uint32_t sn) {
        server_info_t& server_info = _server_info[server_index];
        size_t data_size           = get_data_size();
        bool validate              = opts().validate;

        commit_operation(server_index, IO_WRITE);

        IoMessage *m = _io_msg_pool.get();
        m->init(IO_WRITE, sn, data_size, validate);

        BufferIov *iov = _data_buffers_pool.get();
        iov->init(data_size, _data_chunks_pool, sn, validate);

        SendCompleteCallback *cb = _send_callback_pool.get();
        cb->init(iov, NULL, m);

        VERBOSE_LOG << "sending IO_WRITE (AM) data " << iov << " size "
                    << data_size << " sn " << sn;

        assert(iov->size() == 1);

        // Send IO_WRITE as AM header and first iov element as payload
        // (note that multi-iov send is not supported for IODEMO with AM yet)
        server_info.conn->send_am(m->buffer(), opts().iomsg_size,
                                  (*iov)[0].buffer(), (*iov)[0].size(), cb);

        return data_size;
    }

    void disconnect_uncompleted_servers(const char *reason) {
        std::vector<size_t> server_idxs;
        server_idxs.reserve(_active_servers.size());

        for (size_t i = 0; i < _active_servers.size(); ++i) {
            if (get_num_uncompleted(_active_servers[i]) > 0) {
                server_idxs.push_back(_active_servers[i]);
            }
        }

        while (!server_idxs.empty()) {
            disconnect_server(server_idxs.back(), reason);
            server_idxs.pop_back();
        }
    }

    virtual void dispatch_io_message(UcxConnection* conn, const void *buffer,
                                     size_t length) {
        iomsg_t const *msg = reinterpret_cast<const iomsg_t*>(buffer);

        VERBOSE_LOG << "got io message " << io_op_names[msg->op] << " sn "
                    << msg->sn << " data size " << msg->data_size
                    << " conn " << conn;

        if (msg->op >= IO_COMP_MIN) {
            assert(msg->op == IO_WRITE_COMP);

            size_t server_index = get_server_index(conn);
            if (server_index < _server_info.size()) {
                handle_operation_completion(server_index, IO_WRITE);
            } else {
                /* do not increment _num_completed here since we decremented
                 * _num_sent on connection termination */
                LOG << "got WRITE completion on failed connection";
            }
        }
    }

    virtual void dispatch_am_message(UcxConnection* conn, const void *buffer,
                                     size_t length,
                                     const UcxAmDesc &data_desc) {
        iomsg_t const *msg = reinterpret_cast<const iomsg_t*>(buffer);

        VERBOSE_LOG << "got AM io message " << io_op_names[msg->op] << " sn "
                    << msg->sn << " data size " << msg->data_size
                    << " conn " << conn;

        assert(msg->op >= IO_COMP_MIN);

        if (opts().validate) {
            assert(length == opts().iomsg_size);
            validate(msg, opts().iomsg_size);
        }

        // Client can receive IO_WRITE_COMP or IO_READ_COMP only
        size_t server_index = get_server_index(conn);
        if (msg->op == IO_WRITE_COMP) {
            assert(msg->op == IO_WRITE_COMP);
            handle_operation_completion(server_index, IO_WRITE);
        } else if (msg->op == IO_READ_COMP) {
            BufferIov *iov = _data_buffers_pool.get();
            iov->init(msg->data_size, _data_chunks_pool, msg->sn, opts().validate);

            IoReadResponseCallback *r = _read_callback_pool.get();
            r->init(this, server_index, msg->sn, opts().validate, iov, 0);

            assert(iov->size() == 1);

            conn->recv_am_data((*iov)[0].buffer(), msg->data_size, data_desc, r);
        }
    }

    static long get_num_uncompleted(const server_info_t& server_info) {
        long num_uncompleted;

        num_uncompleted = (server_info.num_sent[IO_READ] +
                           server_info.num_sent[IO_WRITE]) -
                          (server_info.num_completed[IO_READ] +
                           server_info.num_completed[IO_WRITE]);

        assert(num_uncompleted >= 0);

        return num_uncompleted;
    }

    long get_num_uncompleted(size_t server_index) const {
        assert(server_index < _server_info.size());
        return get_num_uncompleted(_server_info[server_index]);
    }

    static void reset_server_info(server_info_t& server_info) {
        server_info.conn                   = NULL;
        for (int op = 0; op < IO_OP_MAX; ++op) {
            server_info.num_sent[op]       = 0;
            server_info.num_completed[op]  = 0;
            server_info.prev_completed[op] = 0;
        }
    }

    virtual void dispatch_connection_error(UcxConnection *conn) {
        size_t server_index = get_server_index(conn);
        if (server_index < _server_info.size()) {
            disconnect_server(server_index,
                              ucs_status_string(conn->ucx_status()));
        }
    }

    void disconnect_server(size_t server_index, const char *reason) {
        server_info_t& server_info = _server_info[server_index];

        if (server_info.conn->is_disconnecting()) {
            LOG << "not disconnecting " << server_info.conn << " with "
                << get_num_uncompleted(server_info) << " uncompleted operations"
                " (read: " << server_info.num_completed[IO_READ] << "/"
                << server_info.num_sent[IO_READ] << "; write: "
                << server_info.num_completed[IO_WRITE] << "/"
                << server_info.num_sent[IO_WRITE] << ") due to \"" << reason
                << "\" because disconnection is already in progress";
            return;
        }

        LOG << "disconnecting connection " << server_info.conn << " with "
            << get_num_uncompleted(server_info) << " uncompleted operations"
            " (read: " << server_info.num_completed[IO_READ] << "/"
            << server_info.num_sent[IO_READ] << "; write: "
            << server_info.num_completed[IO_WRITE] << "/"
            << server_info.num_sent[IO_WRITE] << ") due to \"" << reason
            << "\"";

        // Destroying the connection will complete its outstanding operations
        server_info.conn->disconnect(new DisconnectCallback(*this,
                                                            *server_info.conn));
    }

    void wait_for_responses(long max_outstanding) {
        bool timer_started  = false;
        bool timer_finished = false;
        double start_time   = 0.; // avoid compile error
        double curr_time, elapsed_time;
        long count = 0;

        while (((_num_sent - _num_completed) > max_outstanding) &&
               (_status == OK)) {
            if ((count++ < BUSY_PROGRESS_COUNT) || timer_finished) {
                progress();
                continue;
            }

            count     = 0;
            curr_time = get_time();

            if (!timer_started) {
                start_time    = curr_time;
                timer_started = true;
                continue;
            }

            elapsed_time = curr_time - start_time;
            if (elapsed_time > _test_opts.client_timeout) {
                LOG << "timeout waiting for " << (_num_sent - _num_completed)
                    << " replies";
                disconnect_uncompleted_servers("timeout for replies");
                timer_finished = true;
            }
            check_time_limit(curr_time);
        }
    }

    void connect(size_t server_index)
    {
        const char *server = opts().servers[server_index];
        struct sockaddr_in connect_addr;
        std::string server_addr;
        int port_num;

        memset(&connect_addr, 0, sizeof(connect_addr));
        connect_addr.sin_family = AF_INET;

        const char *port_separator = strchr(server, ':');
        if (port_separator == NULL) {
            /* take port number from -p argument */
            port_num    = opts().port_num;
            server_addr = server;
        } else {
            /* take port number from the server parameter */
            server_addr = std::string(server)
                            .substr(0, port_separator - server);
            port_num    = atoi(port_separator + 1);
        }

        connect_addr.sin_port = htons(port_num);
        int ret = inet_pton(AF_INET, server_addr.c_str(), &connect_addr.sin_addr);
        if (ret != 1) {
            LOG << "invalid address " << server_addr;
            abort();
        }

        if (!_connecting_servers.insert(server_index).second) {
            LOG << server_name(server_index) << " is already connecting";
            abort();
        }

        UcxConnection *conn = new UcxConnection(*this, opts().use_am);
        _server_info[server_index].conn = conn;
        conn->connect((const struct sockaddr*)&connect_addr,
                      sizeof(connect_addr),
                      new ConnectCallback(*this, server_index));
    }

    const std::string server_name(size_t server_index) {
        std::stringstream ss;
        ss << "server [" << server_index << "] " << opts().servers[server_index];
        return ss.str();
    }

    void connect_succeed(size_t server_index)
    {
        server_info_t &server_info = _server_info[server_index];
        long attempts              = server_info.retry_count + 1;

        server_info.retry_count                = 0;
        server_info.prev_connect_time          = 0.;
        _server_index_lookup[server_info.conn] = server_index;
        active_servers_add(server_index);
        LOG << "Connected to " << server_name(server_index) << " after "
            << attempts << " attempts";
    }

    void connect_failed(size_t server_index) {
        server_info_t& server_info = _server_info[server_index];

        // The connection should close itself calling error handler
        server_info.conn = NULL;

        ++server_info.retry_count;

        UcxLog log(LOG_PREFIX);
        log << "Connect to " << server_name(server_index) << " failed"
            << " (retry " << server_info.retry_count;
        if (opts().retries < std::numeric_limits<long>::max()) {
            log << "/" << opts().retries;
        }
        log << ")";

        if (server_info.retry_count >= opts().retries) {
            /* If at least one server exceeded its retries, bail */
            _status = CONN_RETRIES_EXCEEDED;
        }
    }

    void connect_all(bool force) {
        if (_active_servers.size() == _server_info.size()) {
            assert(_status == OK);
            // All servers are connected
            return;
        }

        if (!force && !_active_servers.empty()) {
            // The active list is not empty, and we don't have to check the
            // connect retry timeout
            return;
        }

        double curr_time = get_time();
        for (size_t server_index = 0; server_index < _server_info.size();
             ++server_index) {
            server_info_t& server_info = _server_info[server_index];
            if (server_info.conn != NULL) {
                // Already connecting to this server
                continue;
            }

            // If retry count exceeded for at least one server, we should have
            // exited already
            assert(_status == OK);
            assert(server_info.retry_count < opts().retries);

            if (curr_time < (server_info.prev_connect_time +
                             opts().retry_interval)) {
                // Not enough time elapsed since previous connection attempt
                continue;
            }

            connect(server_index);
            server_info.prev_connect_time = curr_time;
            assert(server_info.conn != NULL);
            assert(_status == OK);
        }
    }

    size_t pick_server_index() const {
        assert(_num_active_servers_to_use != 0);

        /* Pick a random connected server to which the client has credits
         * to send (its conn's window is not full) */
        size_t active_index = IoDemoRandom::rand(size_t(0),
                                                 _num_active_servers_to_use - 1);
        size_t server_index = _active_servers[active_index];
        assert(get_num_uncompleted(server_index) < opts().conn_window_size);
        assert(_server_info[server_index].conn != NULL);

        return server_index;
    }

    static inline bool is_control_iter(long iter) {
        return (iter % 10) == 0;
    }

    status_t run() {
        _server_info.resize(opts().servers.size());
        std::for_each(_server_info.begin(), _server_info.end(),
                      reset_server_info);

        _status = OK;

        // TODO reset these values by canceling requests
        _num_sent      = 0;
        _num_completed = 0;

        uint32_t sn                  = IoDemoRandom::rand<uint32_t>();
        double prev_time             = get_time();
        long total_iter              = 0;
        long total_prev_iter         = 0;
        op_info_t op_info[IO_OP_MAX] = {{0,0}};

        while ((total_iter < opts().iter_count) && (_status == OK)) {
            connect_all(is_control_iter(total_iter));
            if (_status != OK) {
                break;
            }

            if (_active_servers.empty()) {
                if (_connecting_servers.empty()) {
                    LOG << "All remote servers are down, reconnecting in "
                        << opts().retry_interval << " seconds";
                    sleep(opts().retry_interval);
                    check_time_limit(get_time());
                } else {
                    progress();
                }
                continue;
            }

            VERBOSE_LOG << " <<<< iteration " << total_iter << " >>>>";
            long conns_window_size = opts().conn_window_size *
                                     _active_servers.size();
            long max_outstanding   = std::min(opts().window_size,
                                              conns_window_size) - 1;
            wait_for_responses(max_outstanding);
            if (_status != OK) {
                break;
            }

            if (_num_active_servers_to_use == 0) {
                // It is possible that the number of active servers to use is 0
                // after wait_for_responses(), if some clients were closed in
                // UCP Worker progress during handling of remote disconnection
                // from servers
                continue;
            }

            size_t server_index = pick_server_index();
            io_op_t op          = get_op();
            size_t size;
            switch (op) {
            case IO_READ:
                if (opts().use_am) {
                    size = do_io_read_am(server_index, sn);
                } else {
                    size = do_io_read(server_index, sn);
                }
                break;
            case IO_WRITE:
                if (opts().use_am) {
                    size = do_io_write_am(server_index, sn);
                } else {
                    size = do_io_write(server_index, sn);
                }
                break;
            default:
                abort();
            }

            op_info[op].total_bytes += size;
            op_info[op].num_iters++;

            if (is_control_iter(total_iter) && (total_iter > total_prev_iter)) {
                /* Print performance every 1 second */
                double curr_time = get_time();
                if (curr_time >= (prev_time + opts().print_interval)) {
                    wait_for_responses(0);
                    if (_status != OK) {
                        break;
                    }

                    report_performance(total_iter - total_prev_iter,
                                       curr_time - prev_time, op_info);
                    total_prev_iter = total_iter;
                    prev_time       = curr_time;

                    check_time_limit(curr_time);
                }
            }

            ++total_iter;
            ++sn;
        }

        wait_for_responses(0);
        if (_status == OK) {
            double curr_time = get_time();
            report_performance(total_iter - total_prev_iter,
                               curr_time - prev_time, op_info);
        }

        for (size_t server_index = 0; server_index < _server_info.size();
             ++server_index) {
            LOG << "Disconnecting from " << server_name(server_index);
            UcxConnection& conn = *_server_info[server_index].conn;
            conn.disconnect(new DisconnectCallback(*this, conn));
        }

        if (!_active_servers.empty()) {
            LOG << "Waiting for " << _active_servers.size()
                << " disconnects to complete";
            do {
                progress();
            } while (!_active_servers.empty());
        }

        assert(_server_index_lookup.empty());

        return _status;
    }

    status_t get_status() const {
        return _status;
    }

    static const char* get_status_str(status_t status) {
        switch (status) {
        case OK:
            return "OK";
        case CONN_RETRIES_EXCEEDED:
            return "connection retries exceeded";
        case RUNTIME_EXCEEDED:
            return "run-time exceeded";
        default:
            return "invalid status";
        }
    }

private:
    typedef struct {
        long      num_iters;
        size_t    total_bytes;
    } op_info_t;

    inline io_op_t get_op() {
        if (opts().operations.size() == 1) {
            return opts().operations[0];
        }

        return opts().operations[IoDemoRandom::rand(
                                 size_t(0), opts().operations.size() - 1)];
    }

    void report_performance(long num_iters, double elapsed, op_info_t *op_info) {
        if (num_iters == 0) {
            return;
        }

        double latency_usec = (elapsed / num_iters) * 1e6;
        bool first_print    = true;

        UcxLog log(LOG_PREFIX);

        for (unsigned op_id = 0; op_id < IO_OP_MAX; ++op_id) {
            if (!op_info[op_id].total_bytes) {
                continue;
            }

            if (!first_print) {
                log << ", "; // print comma for non-first operation
            }
            first_print = false;

            // Report bandwidth
            double throughput_mbs = op_info[op_id].total_bytes /
                                    elapsed / (1024.0 * 1024.0);
            log << io_op_names[op_id] << " " << throughput_mbs << " MB/s";
            op_info[op_id].total_bytes = 0;

            // Collect min/max among all connections
            long delta_min = std::numeric_limits<long>::max(), delta_max = 0;
            size_t min_index = 0;
            for (size_t server_index = 0; server_index < _server_info.size();
                 ++server_index) {
                server_info_t& server_info = _server_info[server_index];
                long delta_completed       = server_info.num_completed[op_id] -
                                             server_info.prev_completed[op_id];
                if ((delta_completed < delta_min) ||
                    ((delta_completed == delta_min) &&
                     (server_info.retry_count >
                      _server_info[min_index].retry_count))) {
                    min_index = server_index;
                }

                delta_min = std::min(delta_completed, delta_min);
                delta_max = std::max(delta_completed, delta_max);

                server_info.prev_completed[op_id] =
                        server_info.num_completed[op_id];
            }

            // Report delta of min/max/total operations for every connection
            log << " min:" << delta_min << " (" << opts().servers[min_index]
               << ") max:" << delta_max << " total:"
                << op_info[op_id].num_iters << " ops";
            op_info[op_id].num_iters = 0;
        }

        log << ", active:" << _active_servers.size();

        if (opts().window_size == 1) {
            log << ", latency:" << latency_usec << " usec";
        }

        log << ", buffers:" << _data_buffers_pool.allocated();
    }

    inline void check_time_limit(double current_time) {
        if ((_status == OK) &&
            ((current_time - _start_time) >= opts().client_runtime_limit)) {
            _status = RUNTIME_EXCEEDED;
        }
    }

    void active_servers_swap(size_t index1, size_t index2) {
        size_t& active_server1 = _active_servers[index1];
        size_t& active_server2 = _active_servers[index2];

        std::swap(_server_info[active_server1].active_index,
                  _server_info[active_server2].active_index);
        std::swap(active_server1, active_server2);
    }

    void active_servers_add(size_t server_index) {
        assert(_num_active_servers_to_use <= _active_servers.size());

        _active_servers.push_back(server_index);
        _server_info[server_index].active_index = _active_servers.size() - 1;
        active_servers_make_used(_server_info[server_index].active_index);
        assert(_num_active_servers_to_use <= _active_servers.size());
    }

    void active_servers_remove(size_t active_index) {
        assert(active_index < _active_servers.size());

        if (active_index < _num_active_servers_to_use) {
            active_servers_make_unused(active_index);
            active_index = _num_active_servers_to_use;
        }

        assert(active_index >= _num_active_servers_to_use);
        active_servers_swap(active_index, _active_servers.size() - 1);
        _active_servers.pop_back();
    }

    void active_servers_make_unused(size_t active_index) {
        assert(active_index < _num_active_servers_to_use);
        --_num_active_servers_to_use;
        active_servers_swap(active_index, _num_active_servers_to_use);
    }

    void active_servers_make_used(size_t active_index) {
        assert(active_index >= _num_active_servers_to_use);
        active_servers_swap(active_index, _num_active_servers_to_use);
        ++_num_active_servers_to_use;
    }

private:
    std::vector<server_info_t>              _server_info;
    // Connection establishment is in progress
    std::set<size_t>                        _connecting_servers;
    // Active servers is the list of communicating servers
    std::vector<size_t>                     _active_servers;
    // Num active servers to use handles window size, server becomes "unused" if
    // its window is full
    size_t                                  _num_active_servers_to_use;
    std::map<const UcxConnection*, size_t>  _server_index_lookup;
    long                                    _num_sent;
    long                                    _num_completed;
    status_t                                _status;
    double                                  _start_time;
    MemoryPool<IoReadResponseCallback>      _read_callback_pool;
};

static int set_data_size(char *str, options_t *test_opts)
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

static int set_time(char *str, double *dest_p)
{
    char units[3] = "";
    int num_fields;
    double value;
    double per_sec;

    if (!strcmp(str, "inf")) {
        *dest_p = std::numeric_limits<double>::max();
        return 0;
    }

    num_fields = sscanf(str, "%lf%c%c", &value, &units[0], &units[1]);
    if (num_fields == 1) {
        per_sec = 1;
    } else if ((num_fields == 2) || (num_fields == 3)) {
        if (!strcmp(units, "h")) {
            per_sec = 1.0 / 3600.0;
        } else if (!strcmp(units, "m")) {
            per_sec = 1.0 / 60.0;
        } else if (!strcmp(units, "s")) {
            per_sec = 1;
        } else if (!strcmp(units, "ms")) {
            per_sec = 1e3;
        } else if (!strcmp(units, "us")) {
            per_sec = 1e6;
        } else if (!strcmp(units, "ns")) {
            per_sec = 1e9;
        } else {
            return -1;
        }
    } else {
        return -1;
    }

    *(double*)dest_p = value / per_sec;
    return 0;
}

static void adjust_opts(options_t *test_opts) {
    if (test_opts->operations.size() == 0) {
        test_opts->operations.push_back(IO_WRITE);
    }

    if (test_opts->use_am &&
        (test_opts->chunk_size < test_opts->max_data_size)) {
        std::cout << "ignoring chunk size parameter, because it is not supported"
                     " with AM API" << std::endl;
        test_opts->chunk_size = test_opts->max_data_size;
        return;
    }

    test_opts->chunk_size = std::min(test_opts->chunk_size,
                                     test_opts->max_data_size);

    // randomize servers to optimize startup
    std::random_shuffle(test_opts->servers.begin(), test_opts->servers.end(),
                        IoDemoRandom::urand<size_t>);

    UcxLog vlog(LOG_PREFIX, test_opts->verbose);
    vlog << "List of servers:";
    for (size_t i = 0; i < test_opts->servers.size(); ++i) {
        vlog << " " << test_opts->servers[i];
    }
}

static int parse_window_size(const char *optarg, long &window_size,
                             const std::string &window_size_str) {
    window_size = strtol(optarg, NULL, 0);
    if ((window_size <= 0) ||
        // If the converted value falls out of range of corresponding
        // return type, LONG_MAX is returned
        (window_size == std::numeric_limits<long>::max())) {
        std::cout << "invalid " << window_size_str << " size '" << optarg
                  << "'" << std::endl;
        return -1;
    }

    return 0;
}

static int parse_args(int argc, char **argv, options_t *test_opts)
{
    char *str;
    bool found;
    int c;

    test_opts->port_num              = 1337;
    test_opts->connect_timeout       = 20.0;
    test_opts->client_timeout        = 50.0;
    test_opts->retries               = std::numeric_limits<long>::max();
    test_opts->retry_interval        = 5.0;
    test_opts->client_runtime_limit  = std::numeric_limits<double>::max();
    test_opts->print_interval        = 1.0;
    test_opts->min_data_size         = 4096;
    test_opts->max_data_size         = 4096;
    test_opts->chunk_size            = std::numeric_limits<unsigned>::max();
    test_opts->num_offcache_buffers  = 0;
    test_opts->iomsg_size            = 256;
    test_opts->iter_count            = 1000;
    test_opts->window_size           = 16;
    test_opts->conn_window_size      = 16;
    test_opts->random_seed           = std::time(NULL) ^ getpid();
    test_opts->verbose               = false;
    test_opts->validate              = false;
    test_opts->use_am                = false;
    test_opts->memory_type           = UCS_MEMORY_TYPE_HOST;

    while ((c = getopt(argc, argv, "p:c:r:d:b:i:w:a:k:o:t:n:l:s:y:vqAHP:m:")) !=
           -1) {
        switch (c) {
        case 'p':
            test_opts->port_num = atoi(optarg);
            break;
        case 'c':
            if (strcmp(optarg, "inf")) {
                test_opts->retries = strtol(optarg, NULL, 0);
            }
            break;
        case 'y':
            if (set_time(optarg, &test_opts->retry_interval) != 0) {
                std::cout << "invalid '" << optarg
                          << "' value for retry interval" << std::endl;
                return -1;
            }
            break;
        case 'r':
            test_opts->iomsg_size = strtol(optarg, NULL, 0);
            if (test_opts->iomsg_size < sizeof(P2pDemoCommon::iomsg_t)) {
                std::cout << "io message size must be >= "
                          << sizeof(P2pDemoCommon::iomsg_t) << std::endl;
                return -1;
            }
            break;
        case 'd':
            if (set_data_size(optarg, test_opts) == -1) {
                std::cout << "invalid data size range '" << optarg << "'" << std::endl;
                return -1;
            }
            break;
        case 'b':
            test_opts->num_offcache_buffers = strtol(optarg, NULL, 0);
            break;
        case 'i':
            test_opts->iter_count = strtol(optarg, NULL, 0);
            if (test_opts->iter_count == 0) {
                test_opts->iter_count = std::numeric_limits<long int>::max();
            }
            break;
        case 'w':
            if (parse_window_size(optarg, test_opts->window_size,
                                  "window") != 0) {
                return -1;
            }
            break;
        case 'a':
            if (parse_window_size(optarg, test_opts->conn_window_size,
                                  "per connection window") != 0) {
                return -1;
            }
            break;
        case 'k':
            test_opts->chunk_size = strtol(optarg, NULL, 0);
            break;
        case 'o':
            str = strtok(optarg, ",");
            while (str != NULL) {
                found = false;

                for (int op_it = 0; op_it < IO_OP_MAX; ++op_it) {
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
        case 'n':
            if (set_time(optarg, &test_opts->connect_timeout) != 0) {
                std::cout << "invalid '" << optarg << "' value for connect timeout" << std::endl;
                return -1;
            }
            break;
        case 't':
            if (set_time(optarg, &test_opts->client_timeout) != 0) {
                std::cout << "invalid '" << optarg << "' value for client timeout" << std::endl;
                return -1;
            }
            break;
        case 'l':
            if (set_time(optarg, &test_opts->client_runtime_limit) != 0) {
                std::cout << "invalid '" << optarg << "' value for client run-time limit" << std::endl;
                return -1;
            }
            break;
        case 's':
            test_opts->random_seed = strtoul(optarg, NULL, 0);
            break;
        case 'v':
            test_opts->verbose = true;
            break;
        case 'q':
            test_opts->validate = true;
            break;
        case 'A':
            test_opts->use_am = true;
            break;
        case 'H':
            UcxLog::use_human_time = true;
            break;
        case 'P':
            test_opts->print_interval = atof(optarg);
            break;
        case 'm':
            if (!strcmp(optarg, "host")) {
                test_opts->memory_type = UCS_MEMORY_TYPE_HOST;
#ifdef HAVE_CUDA
            } else if (!strcmp(optarg, "cuda")) {
                test_opts->memory_type = UCS_MEMORY_TYPE_CUDA;
            } else if (!strcmp(optarg, "cuda-managed")) {
                test_opts->memory_type = UCS_MEMORY_TYPE_CUDA_MANAGED;
#endif
            } else {
                std::cout << "Invalid '" << optarg << "' value for memory type"
                          << std::endl;
                return -1;
            }
            break;
        case 'h':
        default:
            std::cout << "Usage: io_demo [options] [server_address]" << std::endl;
            std::cout << "       or io_demo [options] [server_address0:port0] [server_address1:port1]..." << std::endl;
            std::cout << "" << std::endl;
            std::cout << "Supported options are:" << std::endl;
            std::cout << "  -p <port>                   TCP port number to use" << std::endl;
            std::cout << "  -n <connect timeout>        Timeout for connecting to the peer (or \"inf\")" << std::endl;
            std::cout << "  -o <op1,op2,...,opN>        Comma-separated string of IO operations [read|write]" << std::endl;
            std::cout << "                              NOTE: if using several IO operations, performance" << std::endl;
            std::cout << "                                    measurements may be inaccurate" << std::endl;
            std::cout << "  -d <min>:<max>              Range that should be used to get data" << std::endl;
            std::cout << "                              size of IO payload" << std::endl;
            std::cout << "  -b <number of buffers>      Number of offcache IO buffers" << std::endl;
            std::cout << "  -i <iterations-count>       Number of iterations to run communication" << std::endl;
            std::cout << "  -w <window-size>            Number of outstanding requests" << std::endl;
            std::cout << "  -a <conn-window-size>       Number of outstanding requests per connection" << std::endl;
            std::cout << "  -k <chunk-size>             Split the data transfer to chunks of this size" << std::endl;
            std::cout << "  -r <io-request-size>        Size of IO request packet" << std::endl;
            std::cout << "  -t <client timeout>         Client timeout (or \"inf\")" << std::endl;
            std::cout << "  -c <retries>                Number of connection retries on client or " << std::endl;
            std::cout << "                              listen retries on server" << std::endl;
            std::cout << "                              (or \"inf\") for failure" << std::endl;
            std::cout << "  -y <retry interval>         Retry interval" << std::endl;
            std::cout << "  -l <client run-time limit>  Time limit to run the IO client (or \"inf\")" << std::endl;
            std::cout << "                              Examples: -l 17.5s; -l 10m; 15.5h" << std::endl;
            std::cout << "  -s <random seed>            Random seed to use for randomizing" << std::endl;
            std::cout << "  -v                          Set verbose mode" << std::endl;
            std::cout << "  -q                          Enable data integrity and transaction check" << std::endl;
            std::cout << "  -A                          Use UCP Active Messages API (use TAG API otherwise)" << std::endl;
            std::cout << "  -H                          Use human-readable timestamps" << std::endl;
            std::cout << "  -P <interval>               Set report printing interval"  << std::endl;
            std::cout << "" << std::endl;
            std::cout << "  -m <memory_type>            Memory type to use. Possible values: host"
#ifdef HAVE_CUDA
                      << ", cuda, cuda-managed"
#endif
                      << std::endl;
            return -1;
        }
    }

    while (optind < argc) {
        test_opts->servers.push_back(argv[optind++]);
    }

    adjust_opts(test_opts);

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

    DemoClient::status_t status = client.run();
    LOG << "Client exit with status '" << DemoClient::get_status_str(status) << "'";
    return ((status == DemoClient::OK) ||
            (status == DemoClient::RUNTIME_EXCEEDED)) ? 0 : -1;
}

static void print_info(int argc, char **argv)
{
    // Process ID and hostname
    char host[64];
    gethostname(host, sizeof(host));
    LOG << "Starting io_demo pid " << getpid() << " on " << host;

    // Command line
    std::stringstream cmdline;
    for (int i = 0; i < argc; ++i) {
        cmdline << argv[i] << " ";
    }
    LOG << "Command line: " << cmdline.str();

    // UCX library path
    Dl_info info;
    int ret = dladdr((void*)ucp_init_version, &info);
    if (ret) {
        LOG << "UCX library path: " << info.dli_fname;
    }
}

int main(int argc, char **argv)
{
    options_t test_opts;
    int ret;

    print_info(argc, argv);

    ret = parse_args(argc, argv, &test_opts);
    if (ret < 0) {
        return ret;
    }

    if (test_opts.servers.empty()) {
        return do_server(test_opts);
    } else {
        return do_client(test_opts);
    }
}
