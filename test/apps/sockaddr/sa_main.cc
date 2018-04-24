/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sa_base.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>
#include <getopt.h>
#include <netdb.h>


class application {
public:
    class usage_exception : public error {
    public:
        usage_exception(const std::string& message = "");
    };

    application(int argc, char **argv);

    int run();

    static void usage(const std::string& error);

private:
    typedef std::pair<std::string, int> dest_t;
    typedef std::vector<dest_t>         dest_vec_t;

    void parse_hostfile(const std::string& filename);

    void create_worker();

    std::string             m_mode;
    int                     m_port;
    dest_vec_t              m_dests;
    std::shared_ptr<worker> m_worker;
};

application::usage_exception::usage_exception(const std::string& message) :
                error(message) {
};

application::application(int argc, char **argv) :
                m_port(0) {
    int c;

    while ( (c = getopt(argc, argv, "p:f:m:h")) != -1 ) {
        switch (c) {
        case 'p':
            m_port = atoi(optarg);
            break;
        case 'f':
            parse_hostfile(optarg);
            break;
        case 'm':
            m_mode = optarg;
            break;
        default:
            throw usage_exception();
        }
    }

    if (m_mode.empty()) {
        throw usage_exception("missing mode argument");
    }

    if (m_dests.empty()) {
        throw usage_exception("no remote destinations specified");
    }

    if (!m_port) {
        throw usage_exception("local port not specified");
    }
}

int application::run() {
    create_worker();

    // TODO create connections and exchange data

    m_worker.reset();
    return 0;
}

void application::create_worker() {
    struct sockaddr_in inaddr_any;
    memset(&inaddr_any, 0, sizeof(inaddr_any));
    inaddr_any.sin_family      = AF_INET;
    inaddr_any.sin_port        = htons(m_port);
    inaddr_any.sin_addr.s_addr = INADDR_ANY;

    m_worker = worker::make(m_mode, reinterpret_cast<struct sockaddr *>(&inaddr_any),
                            sizeof(inaddr_any));
}

void application::usage(const std::string& error) {
    if (!error.empty()) {
        std::cout << "Error: " << error << std::endl;
        std::cout << std::endl;
    }
    std::cout << "Usage: ./sa [ options ]" << std::endl;
    std::cout << "Options:"                                                           << std::endl;
    std::cout << "    -m <mode>    Application mode (tcp)"                            << std::endl;
    std::cout << "    -p <port>    Local port number to listen on"                    << std::endl;
    std::cout << "    -f <file>    File with list of hosts and ports to connect to"   << std::endl;
    std::cout << "                 Each line in the file is formatter as follows:"    << std::endl;
    std::cout << "                    <address> <port>"                               << std::endl;
}

void application::parse_hostfile(const std::string& filename) {
    std::ifstream f(filename.c_str());
    if (!f) {
        throw error("failed to open '" + filename + "'");
    }

    std::string line;
    int lineno = 1;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        if (line.empty()) {
            continue;
        }

        dest_t dest;
        if ((ss >> dest.first) && (ss >> dest.second)) {
            m_dests.push_back(dest);
        } else {
            std::stringstream errss;
            errss << "syntax error in file '" << filename << "' line " << lineno <<
                     " near `" << line << "'";
            throw error(errss.str());
        }
        ++lineno;
    }
}

int main(int argc, char **argv)
{
    try {
        application app(argc, argv);
        return app.run();
    } catch (application::usage_exception& e) {
        application::usage(e.what());
        return -127;
    } catch (error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}
