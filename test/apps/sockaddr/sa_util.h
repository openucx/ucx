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


#endif
