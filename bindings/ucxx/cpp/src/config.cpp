/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <sstream>
#include <string>

#include <ucxx/config.h>
#include <ucxx/exception.h>
#include <ucxx/utils/file_descriptor.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

void Config::readUCXConfig(ConfigMap userOptions)
{
  ucs_status_t status;

  status = ucp_config_read(NULL, NULL, &_handle);
  utils::ucsErrorThrow(status);

  // Modify the UCX configuration options based on `userOptions`
  for (const auto& kv : userOptions) {
    status = ucp_config_modify(_handle, kv.first.c_str(), kv.second.c_str());
    if (status != UCS_OK) {
      ucp_config_release(_handle);

      if (status == UCS_ERR_NO_ELEM)
        utils::ucsErrorThrow(status,
                             std::string("Option ") + kv.first + std::string("doesn't exist"));
      else
        utils::ucsErrorThrow(status);
    }
  }
}

ConfigMap Config::ucxConfigToMap()
{
  if (_configMap.empty()) {
    FILE* textFileDescriptor = utils::createTextFileDescriptor();
    ucp_config_print(_handle, textFileDescriptor, NULL, UCS_CONFIG_PRINT_CONFIG);
    std::istringstream text{utils::decodeTextFileDescriptor(textFileDescriptor)};

    std::string delim = "=";
    std::string line;
    while (std::getline(text, line)) {
      size_t split  = line.find(delim);
      std::string k = line.substr(4, split - 4);  // 4 to strip "UCX_" prefix
      std::string v = line.substr(split + delim.length(), std::string::npos);
      _configMap[k] = v;
    }
  }

  return _configMap;
}

Config::Config(ConfigMap userOptions) { readUCXConfig(userOptions); }

Config::~Config()
{
  if (_handle != nullptr) ucp_config_release(_handle);
}

ConfigMap Config::get() { return ucxConfigToMap(); }

ucp_config_t* Config::getHandle() { return _handle; }

}  // namespace ucxx
