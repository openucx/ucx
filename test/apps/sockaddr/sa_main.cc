/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "sa_base.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>
#include <map>
#include <sys/epoll.h>
#include <getopt.h>
#include <netdb.h>
#include <unistd.h>


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
    typedef struct {
        std::string        hostname;
        int                port;
    } dest_t;

    typedef std::vector<dest_t> dest_vec_t;

    enum connection_type {
        CONNECTION_CLIENT,
        CONNECTION_SERVER
    };

    struct params {
        params() : port(0),
                   total_conns(1000),
                   conn_ratio(1.5),
                   request_size(32),
                   response_size(1024) {
        }

        std::string         mode;
        int                 port;
        int                 total_conns;
        double              conn_ratio;
        size_t              request_size;
        size_t              response_size;
        dest_vec_t          dests;
   };

    struct connection_state {
        conn_ptr_t          conn_ptr;
        connection_type     conn_type;
        size_t              bytes_sent;
        size_t              bytes_recvd;
        std::string         send_data;
        std::string         recv_data;
    };

    typedef std::shared_ptr<connection_state>    conn_state_ptr_t;
    typedef std::map<uint64_t, conn_state_ptr_t> conn_map_t;

    void parse_hostfile(const std::string& filename);

    void initiate_connections();

    int max_conns_inflight() const;

    void create_worker();

    void add_connection(conn_ptr_t conn_ptr, connection_type conn_type);

    conn_ptr_t connect(const dest_t& dst);

    void advance_connection(conn_state_ptr_t s, uint32_t events);

    void connection_completed(conn_state_ptr_t s);

    static void pton(const dest_t& dst, struct sockaddr_storage& saddr,
                     socklen_t &addrlen);

    template <typename O>
    friend typename O::__basic_ostream& operator<<(O& os, connection_type conn_type);

    params                  m_params;
    std::shared_ptr<worker> m_worker;
    evpoll_set              m_evpoll;
    conn_map_t              m_connections;
    int                     m_num_conns_inflight;
    int                     m_num_conns_started;
};


application::usage_exception::usage_exception(const std::string& message) :
                error(message) {
};

application::application(int argc, char **argv) : m_num_conns_inflight(0),
                m_num_conns_started(0) {
    int c;

    while ( (c = getopt(argc, argv, "p:f:m:r:n:S:s:vh")) != -1 ) {
        switch (c) {
        case 'p':
            m_params.port = atoi(optarg);
            break;
        case 'f':
            parse_hostfile(optarg);
            break;
        case 'm':
            m_params.mode = optarg;
            break;
        case 'r':
            m_params.conn_ratio = atof(optarg);
            break;
        case 'n':
            m_params.total_conns = atoi(optarg);
            break;
        case 'S':
            m_params.request_size = atoi(optarg);
            break;
        case 's':
            m_params.response_size = atoi(optarg);
            break;
        case 'v':
            log::more_verbose();
            break;
        default:
            throw usage_exception();
        }
    }

    if (m_params.mode.empty()) {
        throw usage_exception("missing mode argument");
    }

    if (m_params.dests.empty()) {
        throw usage_exception("no remote destinations specified");
    }

    if (m_params.port == 0) {
        throw usage_exception("local port not specified");
    }
}

int application::run() {
    LOG_INFO << "starting application with "
             << max_conns_inflight() << " simultaneous connections, "
             << m_params.total_conns << " total";

    create_worker();

    while ((m_num_conns_started > m_params.total_conns) || !m_connections.empty()) {
        initiate_connections();
        m_worker->wait(m_evpoll,
                       [this](conn_ptr_t conn) {
                           LOG_DEBUG << "accepted new connection";
                           add_connection(conn, CONNECTION_SERVER);
                       },
                       [this](uint64_t conn_id, uint32_t events) {
                           LOG_DEBUG << "new event on connection id "
                                     << conn_id << " events "
                                     << ((events & EPOLLIN ) ? "i" : "")
                                     << ((events & EPOLLOUT) ? "o" : "")
                                     << ((events & EPOLLERR) ? "e" : "")
                                     ;
                           advance_connection(m_connections.at(conn_id), events);
                       },
                       -1);
    }

    LOG_INFO << "all connections completed";

    m_worker.reset();
    return 0;
}

void application::create_worker() {
    struct sockaddr_in inaddr_any;
    memset(&inaddr_any, 0, sizeof(inaddr_any));
    inaddr_any.sin_family      = AF_INET;
    inaddr_any.sin_port        = htons(m_params.port);
    inaddr_any.sin_addr.s_addr = INADDR_ANY;

    m_worker = worker::make(m_params.mode, reinterpret_cast<struct sockaddr *>(&inaddr_any),
                            sizeof(inaddr_any));
    m_worker->add_to_evpoll(m_evpoll);
}

std::shared_ptr<connection> application::connect(const dest_t& dst) {
    struct sockaddr_storage saddr;
    socklen_t addrlen;
    pton(dst, saddr, addrlen);
    return m_worker->connect(reinterpret_cast<const struct sockaddr*>(&saddr),
                             addrlen);
}

template <typename O>
typename O::__basic_ostream& operator<<(O& os, application::connection_type conn_type) {
    switch (conn_type) {
    case application::CONNECTION_CLIENT:
        return os << "client";
    case application::CONNECTION_SERVER:
        return os << "server";
    default:
        return os;
    }
}

void application::add_connection(conn_ptr_t conn_ptr, connection_type conn_type) {
    auto s = std::make_shared<connection_state>();
    s->conn_type   = conn_type;
    s->conn_ptr    = conn_ptr;
    s->bytes_sent  = 0;
    s->bytes_recvd = 0;

    switch (s->conn_type) {
    case CONNECTION_CLIENT:
        s->send_data.assign(m_params.request_size, 'r');
        s->recv_data.resize(m_params.response_size);
        break;
    case CONNECTION_SERVER:
        s->send_data.resize(m_params.response_size);
        s->recv_data.resize(m_params.request_size);
        break;
    }

    LOG_DEBUG << "add " << conn_type << " connection with id " << conn_ptr->id();
    conn_ptr->add_to_evpoll(m_evpoll);
    m_connections[conn_ptr->id()] = s;
    advance_connection(s, 0);
}

void application::initiate_connections() {
    int max = max_conns_inflight();
    while ((m_num_conns_started < m_params.total_conns) && (m_num_conns_inflight < max)) {
        /* coverity[dont_call] */
        const dest_t& dest = m_params.dests[::rand() % m_params.dests.size()];
        ++m_num_conns_started;
        ++m_num_conns_inflight;
        LOG_DEBUG << "connecting to " << dest.hostname << ":" << dest.port;
        add_connection(connect(dest), CONNECTION_CLIENT);
    }
}

int application::max_conns_inflight() const {
    return m_params.conn_ratio * m_params.dests.size() + 0.5;
}

void application::advance_connection(conn_state_ptr_t s, uint32_t events) {
    LOG_DEBUG << "advance " << s->conn_type << " connection id " << s->conn_ptr->id()
              << " total sent " << s->bytes_sent << ", received " << s->bytes_recvd;
    switch (s->conn_type) {
    case CONNECTION_CLIENT:
        if (s->bytes_sent < m_params.request_size) {
            /* more data should be sent */
            size_t nsent = s->conn_ptr->send(&s->send_data[s->bytes_sent],
                                             m_params.request_size - s->bytes_sent);
            LOG_DEBUG << "sent " << nsent << " bytes on connection id "
                      << s->conn_ptr->id();
            s->bytes_sent += nsent;
        }
        if (events & EPOLLIN) {
            size_t nrecv = s->conn_ptr->recv(&s->recv_data[s->bytes_recvd],
                                             m_params.response_size - s->bytes_recvd);
            LOG_DEBUG << "received " << nrecv << " bytes on connection id "
                       << s->conn_ptr->id();
            s->bytes_recvd += nrecv;
        }
        if (s->bytes_recvd == m_params.response_size) {
            connection_completed(s);
        }
        break;
    case CONNECTION_SERVER:
        if (events & EPOLLIN) {
            size_t nrecv = s->conn_ptr->recv(&s->recv_data[s->bytes_recvd],
                                             m_params.request_size - s->bytes_recvd);
            LOG_DEBUG << "received " << nrecv << " bytes on connection id "
                      << s->conn_ptr->id();
            s->bytes_recvd += nrecv;
        }
        if ((s->bytes_recvd == m_params.request_size) &&
            (s->bytes_sent < m_params.response_size)) {
            /* more data should be sent */
            size_t nsent = s->conn_ptr->send(&s->send_data[s->bytes_sent],
                                             m_params.response_size - s->bytes_sent);
            LOG_DEBUG << "sent " << nsent << " bytes on connection id "
                      << s->conn_ptr->id();
            s->bytes_sent += nsent;
        }
        if (s->conn_ptr->is_closed()) {
            connection_completed(s);
        }
        break;
    }
}

void application::connection_completed(conn_state_ptr_t s) {
    LOG_DEBUG << "completed " << s->conn_type << " connection id " << s->conn_ptr->id();
    m_connections.erase(s->conn_ptr->id());
    --m_num_conns_inflight;
}

void application::pton(const dest_t& dst, struct sockaddr_storage& saddr,
                       socklen_t &addrlen) {

    struct hostent *he = gethostbyname(dst.hostname.c_str());
    if (he == NULL || he->h_addr_list == NULL) {
        throw error("host " + dst.hostname + " not found: "+ hstrerror(h_errno));
    }

    memset(&saddr, 0, sizeof(saddr));
    saddr.ss_family = he->h_addrtype;

    void *addr;
    int addr_datalen = 0;
    switch (saddr.ss_family) {
    case AF_INET:
        reinterpret_cast<struct sockaddr_in*>(&saddr)->sin_port =
                        htons(dst.port);
        /* cppcheck-suppress internalAstError */
        addr         = &reinterpret_cast<struct sockaddr_in*>(&saddr)->sin_addr;
        addrlen      = sizeof(struct sockaddr_in);
        addr_datalen = sizeof(struct in_addr);
        break;
    case AF_INET6:
        reinterpret_cast<struct sockaddr_in6*>(&saddr)->sin6_port =
                        htons(dst.port);
        addr         = &reinterpret_cast<struct sockaddr_in6*>(&saddr)->sin6_addr;
        addrlen      = sizeof(struct sockaddr_in6);
        addr_datalen = sizeof(struct in6_addr);
        break;
    default:
        throw error("unsupported address family");
    }

    if (he->h_length != addr_datalen) {
        throw error("mismatching address length");
    }

    memcpy(addr, he->h_addr_list[0], addr_datalen);
}

void application::usage(const std::string& error) {
    if (!error.empty()) {
        std::cout << "Error: " << error << std::endl;
        std::cout << std::endl;
    }

    params defaults;
    std::cout << "Usage: ./sa [ options ]" << std::endl;
    std::cout << "Options:"                                                           << std::endl;
    std::cout << "    -m <mode>    Application mode (tcp)"                            << std::endl;
    std::cout << "    -p <port>    Local port number to listen on"                    << std::endl;
    std::cout << "    -f <file>    File with list of hosts and ports to connect to"   << std::endl;
    std::cout << "                 Each line in the file is formatter as follows:"    << std::endl;
    std::cout << "                    <address> <port>"                               << std::endl;
    std::cout << "    -r <ratio>   How many in-flight connection to hold as multiple" << std::endl;
    std::cout << "                 of number of possible destinations (" << defaults.conn_ratio << ")" << std::endl;
    std::cout << "    -n <count>   How many total exchanges to perform (" << defaults.total_conns << ")" << std::endl;
    std::cout << "    -S <size>    Request message size, in bytes (" << defaults.request_size << ")" << std::endl;
    std::cout << "    -s <size>    Response message size, in bytes (" << defaults.response_size << ")" << std::endl;
    std::cout << "    -v           Increase verbosity level (may be specified several times)" << std::endl;
}

void application::parse_hostfile(const std::string& filename) {
    std::ifstream f(filename.c_str());
    if (!f) {
        throw error("failed to open '" + filename + "'");
    }

    /*
     * Each line in the file contains 2 whitespace-separated tokens: host-name
     * and port number.
     */
    std::string line;
    int lineno = 1;
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        if (line.empty()) {
            continue;
        }

        dest_t dest;
        if ((ss >> dest.hostname) && (ss >> dest.port)) {
            m_params.dests.push_back(dest);
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
