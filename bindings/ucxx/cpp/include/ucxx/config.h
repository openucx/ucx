/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ucp/api/ucp.h>

#include <ucxx/typedefs.h>

namespace ucxx {

/**
 * @brief Component encapsulating the UCP configuration.
 *
 * The UCP layer provides a handle to its configuration in form of `ucp_config_t*` object,
 * this class encapsulates that object and provides methods to simplify its handling.
 */
class Config {
 private:
  ucp_config_t* _handle{nullptr};  ///< Handle to the UCP config
  ConfigMap _configMap;            ///< Map containing all visible UCP configurations

  /**
   * @brief Read UCX configuration and apply user options.
   *
   * Read UCX configuration defaults and environment variable modifiers and apply user
   * configurations overriding previously set configurations.
   *
   * @param[in] userOptions user-defined options overriding defaults and environment
   *                        variable modifiers.
   */
  void readUCXConfig(ConfigMap userOptions);

  /**
   * @brief Parse UCP configurations and convert them to a map.
   *
   * Parse UCP configurations obtained from `ucp_config_print()` and convert them to a map
   * for easy access.
   *
   * @returns The map to the UCP configurations defined for the process.
   */
  [[nodiscard]] ConfigMap ucxConfigToMap();

 public:
  Config()                         = delete;
  Config(const Config&)            = delete;
  Config& operator=(Config const&) = delete;
  Config(Config&& o)               = delete;
  Config& operator=(Config&& o)    = delete;

  /**
   * @brief Constructor that reads the UCX configuration and apply user options.
   *
   * Read UCX configuration defaults and environment variable modifiers and apply user
   * configurations overriding previously set configurations.
   *
   * @param[in] userOptions user-defined options overriding defaults and environment
   *                        variable modifiers.
   */
  explicit Config(ConfigMap userOptions);

  ~Config();

  /**
   * @brief Get the configuration map.
   *
   * Get the configuration map with all visible UCP configurations that are in effect for
   * the current process.
   *
   * @returns The map to the UCP configurations defined for the process.
   */
  [[nodiscard]] ConfigMap get();

  /**
   * @brief Get the underlying `ucp_config_t*` handle
   *
   * Lifetime of the `ucp_config_t*` handle is managed by the `ucxx::Config` object and
   * its ownership is non-transferrable. Once the `ucxx::Config` is destroyed the handle
   * is not valid anymore, it is the user's responsibility to ensure the owner's lifetime
   * while using the handle.
   *
   * @code{.cpp}
   * // config is `ucxx::Config`
   * ucp_config_t* configHandle = config.getHandle();
   * @endcode
   *
   * @return The underlying `ucp_config_t*` handle.
   */
  [[nodiscard]] ucp_config_t* getHandle();
};

}  // namespace ucxx
