/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SA_UTIL_H_
#define SA_UTIL_H_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>


/* runtime error exception */
class error : public std::exception {
public:
    error(const std::string& message);

    virtual ~error() throw();

    virtual const char* what() const throw();

private:
    std::string m_message;
};


/* system error exception */
class sys_error : public error {
public:
    virtual ~sys_error() throw();

    sys_error(const std::string& message, int errn);
};


/* file descriptor wrapper which closes the file automatically */
class file_desc {
public:
    file_desc(int fd);

    virtual ~file_desc();

    operator int() const;

private:
    file_desc(const file_desc&);

    const file_desc& operator=(const file_desc&);

    int m_fd;
};


/* event poll set */
class evpoll_set : public file_desc {
public:
    struct event {
        int      fd;
        uint32_t ev_flags;
    };

    evpoll_set();

    void add(int fd, uint32_t ev_flags);

    void wait(std::vector<event>& events, int timeout_ms = -1) const;

private:
    static int create_epfd();
};

#define LOG_INFO \
    log(log::INFO, __FILE__, __LINE__)
#define LOG_DEBUG \
    log(log::DEBUG, __FILE__, __LINE__)

/* logger */
class log {
public:
    typedef enum {
        INFO,
        DEBUG
    } level_t;

    log(level_t level, const std::string& file, int line);
    ~log();

    template <typename T>
    log& operator<<(const T& value) {
        m_msg << value;
        return *this;
    }

    static void more_verbose();

private:
    static std::string level_str(level_t level);

    static level_t     m_log_level;
    const bool         m_enabled;
    std::ostringstream m_msg;
};

#endif
