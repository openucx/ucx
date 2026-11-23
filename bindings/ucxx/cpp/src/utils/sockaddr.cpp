/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <arpa/inet.h>
#include <cstdio>
#include <cstring>
#include <memory>
#include <netdb.h>
#include <string>
#include <sys/socket.h>

#include <ucxx/exception.h>
#include <ucxx/utils/sockaddr.h>

namespace ucxx {

namespace utils {

std::unique_ptr<struct addrinfo, void (*)(struct addrinfo*)> get_addrinfo(const char* ip_address,
                                                                          uint16_t port)
{
  std::unique_ptr<struct addrinfo, void (*)(struct addrinfo*)> info(nullptr, ::freeaddrinfo);
  {
    char ports[6];
    struct addrinfo* result = nullptr;
    struct addrinfo hints;
    // Don't restrict lookups
    ::memset(&hints, 0, sizeof(hints));
    // Except, port is numeric, address may be NULL meaning the
    // returned address is the wildcard.
    hints.ai_flags = AI_NUMERICSERV | AI_PASSIVE;

    int portsLength = ::snprintf(ports, sizeof(ports), "%u", port);
    if (portsLength < 0 || static_cast<size_t>(portsLength) > sizeof(ports))
      throw ucxx::Error(std::string("Invalid port"));
    if (::getaddrinfo(ip_address, ports, &hints, &result))
      throw ucxx::Error(std::string("Invalid IP address or hostname"));
    info.reset(result);
  }
  return info;
}

void sockaddr_get_ip_port_str(const struct sockaddr_storage* sockaddr,
                              char* ip_str,
                              char* port_str,
                              size_t max_str_size)
{
  const struct sockaddr_in* addr_in   = nullptr;
  const struct sockaddr_in6* addr_in6 = nullptr;

  switch (sockaddr->ss_family) {
    case AF_INET:
      addr_in = reinterpret_cast<decltype(addr_in)>(sockaddr);
      inet_ntop(AF_INET, &addr_in->sin_addr, ip_str, max_str_size);
      snprintf(port_str, max_str_size, "%u", ntohs(addr_in->sin_port));
      break;
    case AF_INET6:
      addr_in6 = reinterpret_cast<decltype(addr_in6)>(sockaddr);
      inet_ntop(AF_INET6, &addr_in6->sin6_addr, ip_str, max_str_size);
      snprintf(port_str, max_str_size, "%u", ntohs(addr_in6->sin6_port));
      break;
    default:
      snprintf(ip_str, max_str_size, "Invalid address family");
      snprintf(port_str, max_str_size, "Invalid address family");
      break;
  }
}

}  // namespace utils

}  // namespace ucxx
