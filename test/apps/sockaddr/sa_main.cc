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
#include <cstdlib>
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
    typedef std::pair<std::string, int> dest_t;
    typedef std::vector<dest_t>         dest_vec_t;

    enum connection_type {
        CONNECTION_CLIENT,
        CONNECTION_SERVER
    };

    struct defaults {
        static const int    CONN_COUNT;;
        static const double CONN_RATIO;
        static const size_t REQUEST_SIZE;
        static const size_t RESPONSE_SIZE;
        static const int    WAIT_TIME;
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

    std::string             m_mode;
    int                     m_port;
    dest_vec_t              m_dests;
    int                     m_conn_backlog;
    double                  m_conn_ratio;
    size_t                  m_request_size;
    size_t                  m_response_size;
    int                     m_wait_time;
    std::shared_ptr<worker> m_worker;
    evpoll_set              m_evpoll;
    conn_map_t              m_connections;
    int                     m_num_conns_inflight;
};

const int    application::defaults::CONN_COUNT    = 1000;
const double application::defaults::CONN_RATIO    = 1.5;
const size_t application::defaults::REQUEST_SIZE  = 32;
const size_t application::defaults::RESPONSE_SIZE = 1024;
const int    application::defaults::WAIT_TIME     = 0;


application::usage_exception::usage_exception(const std::string& message) :
                error(message) {
};

application::application(int argc, char **argv) :
                m_port(0),
                m_conn_backlog(defaults::CONN_COUNT),
                m_conn_ratio(defaults::CONN_RATIO),
                m_request_size(defaults::REQUEST_SIZE),
                m_response_size(defaults::RESPONSE_SIZE),
                m_wait_time(defaults::WAIT_TIME),
                m_num_conns_inflight(0) {
    int c;

    while ( (c = getopt(argc, argv, "p:f:m:r:n:S:s:w:vh")) != -1 ) {
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
        case 'r':
            m_conn_ratio = atof(optarg);
            break;
        case 'n':
            m_conn_backlog = atoi(optarg);
            break;
        case 'S':
            m_request_size = atoi(optarg);
            break;
        case 's':
            m_response_size = atoi(optarg);
            break;
        case 'w':
            m_wait_time = atoi(optarg);
            break;
        case 'v':
            log::more_verbose();
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
    LOG_INFO << "starting application with "
             << max_conns_inflight() << " simultaneous connections, "
             << m_conn_backlog << " total";

    create_worker();

    if (m_wait_time > 0) {
        LOG_DEBUG << "waiting " << m_wait_time << " seconds";
        ::sleep(m_wait_time);
    }

    while ((m_conn_backlog > 0) || !m_connections.empty()) {
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
    inaddr_any.sin_port        = htons(m_port);
    inaddr_any.sin_addr.s_addr = INADDR_ANY;

    m_worker = worker::make(m_mode, reinterpret_cast<struct sockaddr *>(&inaddr_any),
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
        s->send_data.assign(m_request_size, 'r');
        s->recv_data.resize(m_response_size);
        break;
    case CONNECTION_SERVER:
        s->send_data.resize(m_response_size);
        s->recv_data.resize(m_request_size);
        break;
    }

    LOG_DEBUG << "add " << conn_type << " connection with id " << conn_ptr->id();
    conn_ptr->add_to_evpoll(m_evpoll);
    m_connections[conn_ptr->id()] = s;
    advance_connection(s, 0);
}

void application::initiate_connections() {
    int max = max_conns_inflight();
    while ((m_conn_backlog > 0) && (m_num_conns_inflight < max)) {
        /* coverity[dont_call] */
        const dest_t& dest = m_dests[::rand() % m_dests.size()];
        --m_conn_backlog;
        ++m_num_conns_inflight;
        LOG_DEBUG << "connecting to " << dest.first << ":" << dest.second;
        add_connection(connect(dest), CONNECTION_CLIENT);
    }
}

int application::max_conns_inflight() const {
    return m_conn_ratio * m_dests.size() + 0.5;
}

void application::advance_connection(conn_state_ptr_t s, uint32_t events) {
    LOG_DEBUG << "advance " << s->conn_type << " connection id " << s->conn_ptr->id()
              << " total sent " << s->bytes_sent << ", received " << s->bytes_recvd;
    switch (s->conn_type) {
    case CONNECTION_CLIENT:
        if (s->bytes_sent < m_request_size) {
            /* more data should be sent */
            size_t nsent = s->conn_ptr->send(&s->send_data[s->bytes_sent],
                                             m_request_size - s->bytes_sent);
            LOG_DEBUG << "sent " << nsent << " bytes on connection id "
                      << s->conn_ptr->id();
            s->bytes_sent += nsent;
        }
        if (events & EPOLLIN) {
            size_t nrecv = s->conn_ptr->recv(&s->recv_data[s->bytes_recvd],
                                             m_response_size - s->bytes_recvd);
            LOG_DEBUG << "received " << nrecv << " bytes on connection id "
                       << s->conn_ptr->id();
            s->bytes_recvd += nrecv;
        }
        if (s->bytes_recvd == m_response_size) {
            connection_completed(s);
        }
        break;
    case CONNECTION_SERVER:
        if (events & EPOLLIN) {
            size_t nrecv = s->conn_ptr->recv(&s->recv_data[s->bytes_recvd],
                                             m_request_size - s->bytes_recvd);
            LOG_DEBUG << "received " << nrecv << " bytes on connection id "
                      << s->conn_ptr->id();
            s->bytes_recvd += nrecv;
        }
        if ((s->bytes_recvd == m_request_size) && (s->bytes_sent < m_response_size)) {
            /* more data should be sent */
            size_t nsent = s->conn_ptr->send(&s->send_data[s->bytes_sent],
                                             m_response_size - s->bytes_sent);
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

    struct hostent *he = gethostbyname(dst.first.c_str());
    if (he == NULL || he->h_addr_list == NULL) {
        throw error("host " + dst.first + " not found: "+ hstrerror(h_errno));
    }

    void *addr;
    memset(&saddr, 0, sizeof(saddr));
    saddr.ss_family = he->h_addrtype;

    int addr_datalen = 0;
    switch (saddr.ss_family) {
    case AF_INET:
        reinterpret_cast<struct sockaddr_in*>(&saddr)->sin_port =
                        htons(dst.second);
        addr         = &reinterpret_cast<struct sockaddr_in*>(&saddr)->sin_addr;
        addrlen      = sizeof(struct sockaddr_in);
        addr_datalen = sizeof(struct in_addr);
        break;
    case AF_INET6:
        reinterpret_cast<struct sockaddr_in6*>(&saddr)->sin6_port =
                        htons(dst.second);
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
    std::cout << "Usage: ./sa [ options ]" << std::endl;
    std::cout << "Options:"                                                           << std::endl;
    std::cout << "    -m <mode>    Application mode (tcp|ucx)"                        << std::endl;
    std::cout << "    -p <port>    Local port number to listen on"                    << std::endl;
    std::cout << "    -f <file>    File with list of hosts and ports to connect to"   << std::endl;
    std::cout << "                 Each line in the file is formatter as follows:"    << std::endl;
    std::cout << "                    <address> <port>"                               << std::endl;
    std::cout << "    -r <ratio>   How many in-flight connection to hold as multiple" << std::endl;
    std::cout << "                 of number of possible destinations (" << defaults::CONN_RATIO << ")" << std::endl;
    std::cout << "    -n <count>   How many total exchanges to perform (" << defaults::CONN_COUNT << ")" << std::endl;
    std::cout << "    -S <size>    Request message size, in bytes (" << defaults::REQUEST_SIZE << ")" << std::endl;
    std::cout << "    -s <size>    Response message size, in bytes (" << defaults::RESPONSE_SIZE << ")" << std::endl;
    std::cout << "    -w <secs>    Time to wait, in seconds, before connecting (" << defaults::WAIT_TIME << ")" << std::endl;
    std::cout << "    -v           Increase verbosity level (may be specified several times)" << std::endl;

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
