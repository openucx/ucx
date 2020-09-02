/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TEST_HELPERS_H
#define UCS_TEST_HELPERS_H

#include "gtest.h"

#include <common/mem_buffer.h>

#include <ucs/config/types.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sock.h>
#include <ucs/time/time.h>

#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <sys/socket.h>
#include <dirent.h>
#include <stdint.h>


#ifndef UINT16_MAX
#define UINT16_MAX (65535)
#endif /* UINT16_MAX */


/* Test output */
#define UCS_TEST_MESSAGE \
    ucs::detail::message_stream("INFO")


/* Skip test */
#define UCS_TEST_SKIP \
    do { \
        throw ucs::test_skip_exception(); \
    } while(0)


#define UCS_TEST_SKIP_R(_reason) \
    do { \
        throw ucs::test_skip_exception(_reason); \
    } while(0)


/* Abort test */
#define UCS_TEST_ABORT(_message) \
    do { \
        std::stringstream ss; \
        ss << _message; \
        GTEST_MESSAGE_(ss.str().c_str(), ::testing::TestPartResult::kFatalFailure); \
        throw ucs::test_abort_exception(); \
    } while(0)


/* UCS error check */
#define EXPECT_UCS_OK(_expr) \
    do { \
        ucs_status_t _status = (_expr); \
        EXPECT_EQ(UCS_OK, _status) << "Error: " << ucs_status_string(_status); \
    } while (0)


#define ASSERT_UCS_OK(_expr, ...) \
    do { \
        ucs_status_t _status = (_expr); \
        if ((_status) != UCS_OK) { \
            UCS_TEST_ABORT("Error: " << ucs_status_string(_status)  __VA_ARGS__); \
        } \
    } while (0)


#define ASSERT_UCS_OK_OR_INPROGRESS(_expr) \
    do { \
        ucs_status_t _status = (_expr); \
        if (((_status) != UCS_OK) && ((_status) != UCS_INPROGRESS)) { \
            UCS_TEST_ABORT("Error: " << ucs_status_string(_status)); \
        } \
    } while (0)


#define ASSERT_UCS_OK_OR_BUSY(_expr) \
    do { \
        ucs_status_t _status = (_expr); \
        if (((_status) != UCS_OK) && ((_status) != UCS_ERR_BUSY)) { \
            UCS_TEST_ABORT("Error: " << ucs_status_string(_status)); \
        } \
    } while (0)


#define ASSERT_UCS_PTR_OK(_expr) \
    do { \
        ucs_status_ptr_t _status = (_expr); \
        if (UCS_PTR_IS_ERR(_status)) { \
            UCS_TEST_ABORT("Error: " << ucs_status_string(UCS_PTR_STATUS(_status))); \
        } \
    } while (0)


#define EXPECT_UD_CHECK(_val1, _val2, _exp_ud, _exp_non_ud) \
    do { \
        if (has_ud()) { \
            EXPECT_##_exp_ud(_val1, _val2); \
        } else { \
            EXPECT_##_exp_non_ud(_val1, _val2); \
        } \
    } while (0)


/* Run code block with given time limit */
#define UCS_TEST_TIME_LIMIT(_seconds) \
    for (ucs_time_t _start_time = ucs_get_time(), _elapsed = 0; \
         _start_time != 0; \
         ((ucs_time_to_sec(_elapsed = ucs_get_time() - _start_time) >= \
                         (_seconds) * ucs::test_time_multiplier()) && \
          (ucs::perf_retry_count > 0)) \
                         ? (GTEST_NONFATAL_FAILURE_("Time limit exceeded:") << \
                                         "Expected time: " << ((_seconds) * ucs::test_time_multiplier()) << " seconds\n" << \
                                         "Actual time: " << ucs_time_to_sec(_elapsed) << " seconds", 0) \
                         : 0, \
              _start_time = 0)


/**
 * Scoped exit for C++. Usage:
 *
 * UCS_TEST_SCOPE_EXIT() { <code> } UCS_TEST_SCOPE_EXIT_END
 */
#define _UCS_TEST_SCOPE_EXIT(_classname, ...) \
    class _classname { \
    public: \
        _classname() {} \
        ~_classname()
#define UCS_TEST_SCOPE_EXIT(...) \
    _UCS_TEST_SCOPE_EXIT(UCS_PP_APPEND_UNIQUE_ID(onexit), ## __VA_ARGS__)


#define UCS_TEST_SCOPE_EXIT_END \
    } UCS_PP_APPEND_UNIQUE_ID(onexit_var);


/**
 * Make uct_iov_t iov[iovcnt] array with pointer elements to original buffer
 */
#define UCS_TEST_GET_BUFFER_IOV(_name_iov, _name_iovcnt, _buffer_ptr, _buffer_length, _memh, _iovcnt) \
        uct_iov_t _name_iov[_iovcnt]; \
        const size_t _name_iovcnt = _iovcnt; \
        const size_t _buffer_iov_length = _buffer_length / _name_iovcnt; \
        size_t _buffer_iov_length_it = 0; \
        for (size_t iov_it = 0; iov_it < _name_iovcnt; ++iov_it) { \
            _name_iov[iov_it].buffer = (char *)(_buffer_ptr) + _buffer_iov_length_it; \
            _name_iov[iov_it].count  = 1; \
            _name_iov[iov_it].stride = 0; \
            _name_iov[iov_it].memh   = _memh; \
            if (iov_it == (_name_iovcnt - 1)) { /* Last iteration */ \
                _name_iov[iov_it].length = _buffer_length - _buffer_iov_length_it; \
            } else { \
                _name_iov[iov_it].length = _buffer_iov_length; \
                _buffer_iov_length_it += _buffer_iov_length; \
            } \
        }



namespace ucs {

extern const double test_timeout_in_sec;
extern const double watchdog_timeout_default;

extern std::set< const ::testing::TestInfo*> skipped_tests;

typedef enum {
    WATCHDOG_STOP,
    WATCHDOG_RUN,
    WATCHDOG_TIMEOUT_SET,
    WATCHDOG_DEFAULT_SET,
    WATCHDOG_TEST
} test_watchdog_state_t;

typedef struct {
    pthread_t             thread;
    pthread_mutex_t       mutex;
    pthread_cond_t        cv;
    double                timeout;
    pthread_t             watched_thread;
    pthread_barrier_t     barrier;
    test_watchdog_state_t state;
    int                   kill_signal;
} test_watchdog_t;

void *watchdog_func(void *arg);
void watchdog_signal(bool barrier = 1);
void watchdog_set(test_watchdog_state_t new_state, double new_timeout);
void watchdog_set(test_watchdog_state_t new_state);
void watchdog_set(double new_timeout);
test_watchdog_state_t watchdog_get_state();
double watchdog_get_timeout();
int watchdog_get_kill_signal();
int watchdog_start();
void watchdog_stop();

void analyze_test_results();

class test_abort_exception : public std::exception {
};


class exit_exception : public std::exception {
public:
    exit_exception(bool failed) : m_failed(failed) {
    }

    virtual ~exit_exception() throw() {
    }

    bool failed() const {
        return m_failed;
    }

private:
    const bool m_failed;
};


class test_skip_exception : public std::exception {
public:
    test_skip_exception(const std::string& reason = "") : m_reason(reason) {
    }
    virtual ~test_skip_exception() throw() {
    }

    virtual const char* what() const throw() {
        return m_reason.c_str();
    }

private:
    const std::string m_reason;
};


/**
 * @return Time multiplier for performance tests.
 */
int test_time_multiplier();


/**
 * Get current time + @a timeout_in_sec.
 */
ucs_time_t get_deadline(double timeout_in_sec = test_timeout_in_sec);


/**
 * @return System limit on number of TCP connections.
 */
int max_tcp_connections();

 
/**
 * Signal-safe sleep.
 */
void safe_sleep(double sec);
void safe_usleep(double usec);


/**
 * Check if the given interface has an IPv4 or an IPv6 address.
 */
bool is_inet_addr(const struct sockaddr* ifa_addr);


/**
 * Check if the given network device is supported by rdmacm.
 */
bool is_rdmacm_netdev(const char *ifa_name);


/**
 * Get an available port on the host.
 */
uint16_t get_port();


/**
 * Address to use for mmap(FIXED)
 */
void *mmap_fixed_address();


/*
 * Returns a compacted string with just head and tail, e.g "xxx...yyy"
 */
std::string compact_string(const std::string &str, size_t length);


/*
 * Converts exit status from waitpid()/system() to a status string
 */
std::string exit_status_info(int exit_status);


/**
 * Return the IP address of the given interface address.
 */
template <typename S>
std::string sockaddr_to_str(const S *saddr) {
    static char buffer[UCS_SOCKADDR_STRING_LEN];
    return ::ucs_sockaddr_str(reinterpret_cast<const struct sockaddr*>(saddr),
                              buffer, UCS_SOCKADDR_STRING_LEN);
}

/**
 * Wrapper for struct sockaddr_storage to unify work flow for IPv4 and IPv6
 */
class sock_addr_storage {
public:
    sock_addr_storage();

    sock_addr_storage(const ucs_sock_addr_t &ucs_sock_addr);

    void set_sock_addr(const struct sockaddr &addr, const size_t size);

    void reset_to_any();

    bool operator==(const struct sockaddr_storage &sockaddr) const;

    void set_port(uint16_t port);

    uint16_t get_port() const;

    size_t get_addr_size() const;

    ucs_sock_addr_t to_ucs_sock_addr() const;

    std::string to_str() const;

    const struct sockaddr* get_sock_addr_ptr() const;

private:
    struct sockaddr_storage m_storage;
    size_t                  m_size;
    bool                    m_is_valid;
};


std::ostream& operator<<(std::ostream& os, const sock_addr_storage& sa_storage);


/*
 * For gtest's EXPECT_EQ
 */
template <typename T>
static std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    static const size_t LIMIT = 2000;
    size_t i = 0;
    for (std::vector<char>::const_iterator iter = vec.begin();
         iter != vec.end(); ++iter) {
        if (i >= LIMIT) {
            os << "...";
            break;
        }
        os << "[" << i << "]=" << *iter << " ";
        ++i;
    }
    return os << std::endl;
}

std::ostream& operator<<(std::ostream& os, const std::vector<char>& vec);

static inline int rand() {
    /* coverity[dont_call] */
    return ::rand();
}

static inline void srand(unsigned seed) {
    /* coverity[dont_call] */
    return ::srand(seed);
}

void fill_random(void *data, size_t size);

/* C can be vector or string */
template <typename C>
static void fill_random(C& c) {
    fill_random(&c[0], sizeof(c[0]) * c.size());
}

/* C can be vector or string */
template <typename C>
static void fill_random(C& c, size_t size) {
    fill_random(&c[0], sizeof(c[0]) * size);
}

template <typename T>
static inline T random_upper() {
  return static_cast<T>((rand() / static_cast<double>(RAND_MAX)) *
                        static_cast<double>(std::numeric_limits<T>::max()));
}

template <typename T>
class hex_num {
public:
    hex_num(const T num) : m_num(num) {
    }

    operator T() const {
        return m_num;
    }

    template<typename N>
    friend std::ostream& operator<<(std::ostream& os, const hex_num<N>& h);
private:
    const T m_num;
};

template <typename T>
hex_num<T> make_hex(const T num) {
    return hex_num<T>(num);
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const hex_num<T>& h) {
    return os << std::hex << h.m_num << std::dec;
}
class scoped_setenv {
public:
    scoped_setenv(const char *name, const char *value);
    ~scoped_setenv();
private:
    scoped_setenv(const scoped_setenv&);
    const std::string m_name;
    std::string       m_old_value;
};

class ucx_env_cleanup {
public:
    ucx_env_cleanup();
    ~ucx_env_cleanup();
private:
    std::vector<std::string> ucx_env_storage;
};

template <typename T>
std::string to_string(const T& value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template <typename T>
std::string to_hex_string(const T& value) {
    std::stringstream ss;
    ss << std::hex << value;
    return ss.str();
}

template <typename T>
T from_string(const std::string& str) {
    T value;
    return (std::stringstream(str) >> value).fail() ? 0 : value;
}

template <typename T>
class ptr_vector_base {
public:
    typedef std::vector<T*> vec_type;
    typedef typename vec_type::const_iterator const_iterator;

    ptr_vector_base() {
    }

    virtual ~ptr_vector_base() {
        clear();
    }

    /** Add and take ownership */
    void push_back(T* ptr) {
        m_vec.push_back(ptr);
    }

    void push_front(T* ptr) {
        m_vec.insert(m_vec.begin(), ptr);
    }

    virtual void clear() {
        while (!m_vec.empty()) {
            T* ptr = m_vec.back();
            m_vec.pop_back();
            release(ptr);
        }
    }

    const_iterator begin() const {
        return m_vec.begin();
    }

    const_iterator end() const {
        return m_vec.end();
    }

    T* front() {
        return m_vec.front();
    }

    T* back() {
        return m_vec.back();
    }

    size_t size() const {
        return m_vec.size();
    }

protected:
    ptr_vector_base(const ptr_vector_base&);
    vec_type m_vec;

    void release(T *ptr) {
        delete ptr;
    }
};

template<> inline void ptr_vector_base<void>::release(void *ptr) {
    free(ptr);
}


template <typename T>
class ptr_vector : public ptr_vector_base<T> {
public:
    T& at(size_t index) const {
        return *ptr_vector_base<T>::m_vec.at(index);
    }

    size_t remove(T *value) {
        const size_t removed = std::distance(std::remove(this->m_vec.begin(),
                                                         this->m_vec.end(),
                                                         value),
                                             this->m_vec.end());
        if (removed) {
            this->m_vec.resize(this->m_vec.size() - removed);
            this->release(value);
        }
        return removed;
    }
};

template <>
class ptr_vector<void> : public ptr_vector_base<void> {
};


/**
 * Safely wraps C handles
 */
template <typename T, typename ArgT = void *>
class handle {
public:
    typedef T handle_type;
    typedef void (*dtor_t)(T handle);
    typedef void (*dtor2_t)(T handle, ArgT arg);

    handle() : m_initialized(false), m_value(NULL), m_dtor(NULL),
               m_dtor_with_arg(NULL), m_dtor_arg(NULL) {
    }

    handle(const T& value, dtor_t dtor) : m_initialized(true), m_value(value),
                                          m_dtor(dtor), m_dtor_with_arg(NULL),
                                          m_dtor_arg(NULL) {
        EXPECT_TRUE(value != NULL);
    }

    handle(const T& value, dtor2_t dtor, ArgT arg) :
        m_initialized(true), m_value(value), m_dtor(NULL),
        m_dtor_with_arg(dtor), m_dtor_arg(arg)
    {
        EXPECT_TRUE(value != NULL);
    }

    handle(const handle& other) : m_initialized(false), m_value(NULL),
                                  m_dtor(NULL), m_dtor_with_arg(NULL),
                                  m_dtor_arg(NULL) {
        *this = other;
    }

    ~handle() {
        reset();
    }

    void reset() {
        if (m_initialized) {
            release();
        }
    }

    void revoke() const {
        m_initialized = false;
    }

    void reset(const T& value, dtor_t dtor) {
        reset();
        if (value == NULL) {
            throw std::invalid_argument("value cannot be NULL");
        }
        m_value = value;
        m_dtor  = dtor;
        m_dtor_with_arg = NULL;
        m_dtor_arg      = NULL;
        m_initialized   = true;
    }

    void reset(const T& value, dtor2_t dtor, ArgT arg) {
        reset();
        if (value == NULL) {
            throw std::invalid_argument("value cannot be NULL");
        }
        m_value = value;
        m_dtor  = NULL;
        m_dtor_with_arg = dtor;
        m_dtor_arg      = arg;
        m_initialized   = true;
    }

    const handle& operator=(const handle& other) {
        reset();
        if (other.m_initialized) {
            if (other.m_dtor) {
                reset(other.m_value, other.m_dtor);
            } else {
                reset(other.m_value, other.m_dtor_with_arg, other.m_dtor_arg);
            }
            other.revoke();
        }
        return *this;
    }

    operator T() const {
        return get();
    }

    operator bool() const {
        return m_initialized;
    }

    T get() const {
        return m_initialized ? m_value : NULL;
    }

    T operator->() const {
        return get();
    }

private:

    void release() {
        if (m_dtor) {
            m_dtor(m_value);
        } else {
            m_dtor_with_arg(m_value, m_dtor_arg);
        }

        m_initialized = false;
    }

    mutable bool   m_initialized;
    T              m_value;
    dtor_t         m_dtor;
    dtor2_t        m_dtor_with_arg;
    ArgT           m_dtor_arg;
};

/* simplified version of std::auto_ptr which was deprecated in newer stdc++
 * versions in favor of unique_ptr */
template <typename T>
class auto_ptr {
public:
    auto_ptr() : m_ptr(NULL) {
    }

    auto_ptr(T* ptr) : m_ptr(NULL) {
        reset(ptr);
    }

    ~auto_ptr() {
        reset();
    }

    void reset(T* ptr = NULL) {
        if (m_ptr) {
            delete m_ptr;
        }
        m_ptr = ptr;
    }

    operator T*() const {
        return m_ptr;
    }

    T* operator->() const {
        return m_ptr;
    }

private:
    auto_ptr(const auto_ptr&); /* disable copy */
    auto_ptr operator=(const auto_ptr&); /* disable assign */

    T* m_ptr;
};

#define UCS_TEST_TRY_CREATE_HANDLE(_t, _handle, _dtor, _ctor, ...) \
    ({ \
        _t h; \
        ucs_status_t status = _ctor(__VA_ARGS__, &h); \
        ASSERT_UCS_OK_OR_BUSY(status); \
        if (status == UCS_OK) { \
            _handle.reset(h, _dtor); \
        } \
        status; \
    })

#define UCS_TEST_CREATE_HANDLE(_t, _handle, _dtor, _ctor, ...) \
    { \
        _t h; \
        ucs_status_t status = _ctor(__VA_ARGS__, &h); \
        ASSERT_UCS_OK(status); \
        _handle.reset(h, _dtor); \
    }

#define UCS_TEST_CREATE_HANDLE_IF_SUPPORTED(_t, _handle, _dtor, _ctor, ...) \
    { \
        _t h; \
        ucs_status_t status = _ctor(__VA_ARGS__, &h); \
        if (status == UCS_ERR_UNSUPPORTED) { \
            UCS_TEST_SKIP_R(std::string("Unsupported operation: ") + \
                            UCS_PP_MAKE_STRING(_ctor)); \
        } \
        ASSERT_UCS_OK(status); \
        _handle.reset(h, _dtor); \
    }

class size_value {
public:
    explicit size_value(size_t value) : m_value(value) {}

    size_t value() const {
        return m_value;
    }
private:
    size_t m_value;
};


template <typename O>
static inline O& operator<<(O& os, const size_value& sz)
{
    size_t v = sz.value();

    std::iostream::fmtflags f(os.flags());

    /* coverity[format_changed] */
    os << std::fixed << std::setprecision(1);
    if (v < 1024) {
        os << v;
    } else if (v < 1024 * 1024) {
        os << (v / 1024.0) << "k";
    } else if (v < 1024 * 1024 * 1024) {
        os << (v / 1024.0 / 1024.0) << "m";
    } else {
        os << (v / 1024.0 / 1024.0 / 1024.0) << "g";
    }

    os.flags(f);
    return os;
}


class auto_buffer {
public:
    auto_buffer(size_t size);
    ~auto_buffer();
    void* operator*() const;
private:
    void *m_ptr;
};


template <typename T>
static void deleter(T *ptr) {
    delete ptr;
}

extern int    perf_retry_count;
extern double perf_retry_interval;

namespace detail {

class message_stream {
public:
    message_stream(const std::string& title);
    ~message_stream();

    template <typename T>
    message_stream& operator<<(const T& value) {
        msg << value;
        return *this;
    }

    message_stream& operator<< (std::ostream&(*f)(std::ostream&)) {
        if (f == (std::basic_ostream<char>& (*)(std::basic_ostream<char>&)) &std::flush) {
            std::string s = msg.str();
            if (!s.empty()) {
                std::cout << s << std::flush;
                msg.str("");
            }
            msg.clear();
        } else {
            msg << f;
        }
        return *this;
    }

    message_stream& operator<< (const size_value& value) {
            msg << value.value();
            return *this;
    }

    std::iostream::fmtflags flags() {
        return msg.flags();
    }

    void flags(std::iostream::fmtflags f) {
        msg.flags(f);
    }
private:
    std::ostringstream msg;
};

} // detail

/**
 * N-ary Cartesian product over the N vectors provided in the input vector
 * The cardinality of the result vector:
 * output.size = input[0].size * input[1].size * ... * input[input.size].size
 */
template<typename T>
void cartesian_product(std::vector<std::vector<T> > &final_output,
                       std::vector<T> &cur_output,
                       typename std::vector<std::vector<T> >
                       ::const_iterator cur_input,
                       typename std::vector<std::vector<T> >
                       ::const_iterator end_input) {
    if (cur_input == end_input) {
        final_output.push_back(cur_output);
        return;
    }

    const std::vector<T> &cur_vector = *cur_input;

    cur_input++;

    for (typename std::vector<T>::const_iterator iter =
            cur_vector.begin(); iter != cur_vector.end(); ++iter) {
        cur_output.push_back(*iter);
        ucs::cartesian_product(final_output, cur_output,
                               cur_input, end_input);
        cur_output.pop_back();
    }
}

template<typename T>
void cartesian_product(std::vector<std::vector<T> > &output,
                       const std::vector<std::vector<T> > &input)
{
    std::vector<T> cur_output;
    cartesian_product(output, cur_output, input.begin(), input.end());
}

template<typename T>
std::vector<std::vector<T> > make_pairs(const std::vector<T> &input_vec) {
    std::vector<std::vector<T> > result;
    std::vector<std::vector<T> > input;

    input.push_back(input_vec);
    input.push_back(input_vec);

    ucs::cartesian_product(result, input);

    return result;
}

std::vector<std::vector<ucs_memory_type_t> > supported_mem_type_pairs();


/**
 * Skip test if address sanitizer is enabled
 */
void skip_on_address_sanitizer();

} // ucs

#endif /* UCS_TEST_HELPERS_H */
