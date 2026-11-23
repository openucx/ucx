/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <netdb.h>

namespace ucxx {

namespace utils {

/**
 * @brief Get an addrinfo struct corresponding to an address and port.
 *
 * This information can later be used to bind a UCP listener or endpoint.
 *
 * @param[in] ip_address  valid socket address (e.g., IP address or hostname) or NULL as a
 *                        wildcard for "all" to set the socket address storage to.
 * @param[in] port        port to set the socket address storage to.
 *
 * @returns unique pointer wrapping a `struct addrinfo` (frees the addrinfo when out of scope)
 */
[[nodiscard]] std::unique_ptr<struct addrinfo, void (*)(struct addrinfo*)> get_addrinfo(
  const char* ip_address, uint16_t port);

/**
 * @brief Get socket address and port of a socket address storage.
 *
 * Get the socket address (usually the IP address) and port from a socket address storage
 * pointer.
 *
 * @param[in] sock_addr     pointer to the socket address storage.
 * @param[in] ip_str        socket address (or IP) contained the socket address storage.
 * @param[in] port_str      port contained the socket address storage.
 * @param[in] max_str_size  size of the `ip_str` and `port_str` strings.
 */
void sockaddr_get_ip_port_str(const struct sockaddr_storage* sock_addr,
                              char* ip_str,
                              char* port_str,
                              size_t max_str_size);

}  // namespace utils

}  // namespace ucxx
