/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <string>
#include <unordered_map>

#include <ucs/debug/log_def.h>

#include <ucxx/typedefs.h>

namespace ucxx {

/**
 * @brief The UCXX log level component configuration.
 *
 * The type with the UCXX log level component configuration.
 */
extern ucs_log_component_config_t ucxx_log_component_config;

// Macros
#ifndef UCXX_MAX_LOG_LEVEL
#define UCXX_MAX_LOG_LEVEL ucxx::UCXX_LOG_LEVEL_LAST
#endif

#define ucxx_log_component_is_enabled(_level, _comp_log_config)                               \
  ucs_unlikely(                                                                               \
    ((_level) <= UCXX_MAX_LOG_LEVEL) &&                                                       \
    ((_level) <=                                                                              \
     (ucxx::ucxx_log_level_t)(reinterpret_cast<ucs_log_component_config_t*>(_comp_log_config) \
                                ->log_level)))

#define ucxx_log_is_enabled(_level) \
  ucxx_log_component_is_enabled(_level, &ucxx::ucxx_log_component_config)

#define ucxx_log_component(_level, _comp_log_config, _fmt, ...)    \
  do {                                                             \
    if (ucxx_log_component_is_enabled(_level, _comp_log_config)) { \
      ucs_log_dispatch(__FILE__,                                   \
                       __LINE__,                                   \
                       __func__,                                   \
                       (ucs_log_level_t)(_level),                  \
                       _comp_log_config,                           \
                       _fmt,                                       \
                       ##__VA_ARGS__);                             \
    }                                                              \
  } while (0)

#define ucxx_log(_level, _fmt, ...)                                                    \
  do {                                                                                 \
    ucxx_log_component(_level, &ucxx::ucxx_log_component_config, _fmt, ##__VA_ARGS__); \
  } while (0)

#define ucxx_error(_fmt, ...)       ucxx_log(ucxx::UCXX_LOG_LEVEL_ERROR, _fmt, ##__VA_ARGS__)
#define ucxx_warn(_fmt, ...)        ucxx_log(ucxx::UCXX_LOG_LEVEL_WARN, _fmt, ##__VA_ARGS__)
#define ucxx_diag(_fmt, ...)        ucxx_log(ucxx::UCXX_LOG_LEVEL_DIAG, _fmt, ##__VA_ARGS__)
#define ucxx_info(_fmt, ...)        ucxx_log(ucxx::UCXX_LOG_LEVEL_INFO, _fmt, ##__VA_ARGS__)
#define ucxx_debug(_fmt, ...)       ucxx_log(ucxx::UCXX_LOG_LEVEL_DEBUG, _fmt, ##__VA_ARGS__)
#define ucxx_trace(_fmt, ...)       ucxx_log(ucxx::UCXX_LOG_LEVEL_TRACE, _fmt, ##__VA_ARGS__)
#define ucxx_trace_req(_fmt, ...)   ucxx_log(ucxx::UCXX_LOG_LEVEL_TRACE_REQ, _fmt, ##__VA_ARGS__)
#define ucxx_trace_data(_fmt, ...)  ucxx_log(ucxx::UCXX_LOG_LEVEL_TRACE_DATA, _fmt, ##__VA_ARGS__)
#define ucxx_trace_async(_fmt, ...) ucxx_log(ucxx::UCXX_LOG_LEVEL_TRACE_ASYNC, _fmt, ##__VA_ARGS__)
#define ucxx_trace_func(_fmt, ...) \
  ucxx_log(ucxx::UCXX_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ##__VA_ARGS__)
#define ucxx_trace_poll(_fmt, ...) ucxx_log(ucxx::UCXX_LOG_LEVEL_TRACE_POLL, _fmt, ##__VA_ARGS__)

/**
 * @brief Map of log level names to their respective types.
 *
 * Map of log level names, used by the user to specify levels to enable, to their respective
 * internal types.
 */
const std::unordered_map<std::string, ucxx_log_level_t> logLevelNames = {
  {"FATAL", UCXX_LOG_LEVEL_FATAL},
  {"ERROR", UCXX_LOG_LEVEL_ERROR},
  {"WARN", UCXX_LOG_LEVEL_WARN},
  {"DIAG", UCXX_LOG_LEVEL_DIAG},
  {"INFO", UCXX_LOG_LEVEL_INFO},
  {"DEBUG", UCXX_LOG_LEVEL_DEBUG},
  {"TRACE", UCXX_LOG_LEVEL_TRACE},
  {"REQ", UCXX_LOG_LEVEL_TRACE_REQ},
  {"DATA", UCXX_LOG_LEVEL_TRACE_DATA},
  {"ASYNC", UCXX_LOG_LEVEL_TRACE_ASYNC},
  {"FUNC", UCXX_LOG_LEVEL_TRACE_FUNC},
  {"POLL", UCXX_LOG_LEVEL_TRACE_POLL},
  {"", UCXX_LOG_LEVEL_LAST},
  {"PRINT", UCXX_LOG_LEVEL_PRINT}};

/**
 * @brief The name of the default log level.
 *
 * The name of default log level, must be one of the keys in `logLevelNames`.
 */
const char logLevelNameDefault[] = "WARN";

/**
 * @brief The type of the default log level.
 *
 * The type of default log level, automatically set based on the value of
 * `logLevelNameDefault`.
 */
const ucs_log_level_t logLevelDefault = (ucs_log_level_t)logLevelNames.at(logLevelNameDefault);

/**
 * @brief Parse the active log level.
 *
 * Parse the active log level and set appropriate internal values to match the specified
 * level.
 */
void parseLogLevel();

}  // namespace ucxx
