/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2012.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_TEST_HELPERS_H
#define UCS_TEST_HELPERS_H

#include <gtest/gtest.h>
#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>
#include <string>

namespace ucs {

class test_abort_exception : public std::exception {
};


class test_skip_exception : public std::exception {
};

/**
 * @return Time multiplier for performance tests.
 */
int test_time_multiplier();


/**
 * Signal-safe sleep.
 */
void safe_usleep(double usec);


/*
 * For gtest's EXPECT_EQ
 */
template <typename T>
static std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    static const size_t LIMIT = 2000;
    size_t i = 0;
    for (std::vector<char>::const_iterator iter = vec.begin(); iter != vec.end(); ++iter) {
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

template <typename OutputIterator>
static void fill_random(OutputIterator begin, OutputIterator end) {
    for (OutputIterator iter = begin; iter != end; ++iter) {
        *iter = rand();
    }
}

template <typename T>
static inline T random_upper() {
  return static_cast<T>((rand() / static_cast<double>(RAND_MAX)) * std::numeric_limits<T>::max());
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

template <typename T>
std::string to_string(const T& value) {
    std::stringstream ss;
    ss << value;
    return ss.str();
}

template <typename T>
class ptr_vector {
public:
    typedef std::vector<T*> vec_type;
    typedef typename vec_type::const_iterator const_iterator;

    ptr_vector() {
    }

    ~ptr_vector() {
        clear();
    }

    /** Add and take ownership */
    void push_back(T* ptr) {
        m_vec.push_back(ptr);
    }

    const T& operator[](size_t index) const {
        return *m_vec[index];
    }

    void clear() {
        while (!m_vec.empty()) {
            T* ptr = m_vec.back();
            m_vec.pop_back();
            delete ptr;
        }
    }

    const_iterator begin() const {
        return m_vec.begin();
    }

    const_iterator end() const {
        return m_vec.end();
    }
private:
    ptr_vector(const ptr_vector&);
    vec_type m_vec;
};

namespace detail {

class message_stream {
public:
    message_stream(const std::string& title);
    ~message_stream();

    template <typename T>
    std::ostream& operator<<(const T& value) const {
        return std::cout << value;
    }
};

} // detail

} // ucs


/* Test output */
#define UCS_TEST_MESSAGE \
    ucs::detail::message_stream("INFO")


/* Skip test */
#define UCS_TEST_SKIP \
    do { \
        throw ucs::test_skip_exception(); \
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
#define EXPECT_UCS_OK(_error)  EXPECT_EQ(UCS_OK, _error) << "Error: " << ucs_error_string(_error)
#define ASSERT_UCS_OK(_error) \
    do { \
        if ((_error) != UCS_OK) { \
            UCS_TEST_ABORT("Error: " << ucs_status_string(_error)); \
        } \
    } while (0)


/* Run code block with given time limit */
#define UCS_TEST_TIME_LIMIT(_seconds) \
    for (ucs_time_t _start_time = ucs_get_time(), _elapsed = 0; \
         _start_time != 0; \
         (ucs_time_to_sec(_elapsed = ucs_get_time() - _start_time) >= \
                         (_seconds) * ucs::test_time_multiplier()) \
                         ? (GTEST_NONFATAL_FAILURE_("Time limit exceeded:") << \
                                         "Expected time: " << ((_seconds) * ucs::test_time_multiplier()) << " seconds\n" << \
                                         "Actual time: " << ucs_time_to_sec(_elapsed) << " seconds", 0) \
                         : 0, \
              _start_time = 0)


#endif
