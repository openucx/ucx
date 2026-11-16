/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <cstdlib>
#include <string>

#include <ucxx/log.h>
#include <ucxx/typedefs.h>

namespace ucxx {

ucs_log_component_config_t ucxx_log_component_config = {logLevelDefault, "UCXX"};

// Functions
void parseLogLevel()
{
  std::string logLevelName{};

  if (const char* env = std::getenv("UCXX_LOG_LEVEL")) {
    logLevelName = std::string(env);
    std::transform(
      logLevelName.begin(), logLevelName.end(), logLevelName.begin(), [](unsigned char c) {
        return std::toupper(c);
      });

    auto level = logLevelNames.find(logLevelName);
    if (!logLevelName.empty() && level != logLevelNames.end())
      ucxx_log_component_config.log_level = (ucs_log_level_t)level->second;
    else
      ucxx_warn("UCXX_LOG_LEVEL %s unknown, defaulting to UCXX_LOG_LEVEL=%s",
                logLevelName.c_str(),
                logLevelNameDefault);

    ucxx_info("UCXX_LOG_LEVEL: %s", logLevelName.c_str());
  }
}

}  // namespace ucxx
