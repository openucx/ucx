/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <internal/constructors.h>
#include <ucxx/context.h>

#include <ucxx/log.h>
#include <ucxx/utils/file_descriptor.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

Context::Context(const ConfigMap ucxConfig, const uint64_t featureFlags)
  : _config{ucxConfig}, _featureFlags{featureFlags}
{
  parseLogLevel();

  // UCP
  ucp_params_t params = {.field_mask = UCP_PARAM_FIELD_FEATURES, .features = featureFlags};

  utils::ucsErrorThrow(ucp_init(&params, _config.getHandle(), &_handle));
  ucxx_trace("ucxx::Context created: %p, UCP handle: %p", this, _handle);

  ucp_context_attr_t attr = {.field_mask = UCP_ATTR_FIELD_MEMORY_TYPES};
  ucp_context_query(_handle, &attr);
  _cudaSupport = (attr.memory_types & UCS_MEMORY_TYPE_CUDA) == UCS_MEMORY_TYPE_CUDA;

  // UCX supports CUDA if TLS is "all", or one of {"cuda",
  // "cuda_copy", "cuda_ipc"} is in the active transports.
  // If the transport list is negated ("^" at start), then it is to be
  // interpreted as all \ given
  auto configMap = _config.get();
  auto tls       = configMap.find("TLS");
  if (_cudaSupport) {
    if (tls != configMap.end()) {
      auto tls_value = tls->second;
      if (!tls_value.empty() && tls_value[0] == '^') {
        std::size_t current = 1;  // Skip the ^
        do {
          // UCX_TLS lists disabled transports, if this contains either
          // "cuda" or "cuda_copy", then there is no cuda support (just
          // disabling "cuda_ipc" is fine)
          auto next  = tls_value.find_first_of(',', current);
          auto field = tls_value.substr(current, next - current);
          current    = next + 1;
          if (field == "cuda" || field == "cuda_copy") {
            _cudaSupport = false;
            break;
          }
        } while (current != std::string::npos + 1);
      } else {
        // UCX_TLS lists enabled transports, all, or anything with cuda
        // enables cuda support
        _cudaSupport = tls_value == "all" || tls_value.find("cuda") != std::string::npos;
      }
    }
  }

  ucxx_info("UCP initialized using config: ");
  for (const auto& kv : configMap)
    ucxx_info("  %s: %s", kv.first.c_str(), kv.second.c_str());
}

std::shared_ptr<Context> detail::ConstructorFactory::createContext(const ConfigMap ucxConfig,
                                                                   const uint64_t featureFlags)
{
  return std::shared_ptr<Context>(new Context(ucxConfig, featureFlags));
}

Context::~Context()
{
  if (_handle != nullptr) ucp_cleanup(_handle);
  ucxx_trace("ucxx::Context destroyed: %p, UCP handle: %p", this, _handle);
}

ConfigMap Context::getConfig() { return _config.get(); }

ucp_context_h Context::getHandle() { return _handle; }

std::string Context::getInfo()
{
  FILE* TextFileDescriptor = utils::createTextFileDescriptor();
  ucp_context_print_info(_handle, TextFileDescriptor);
  return utils::decodeTextFileDescriptor(TextFileDescriptor);
}

uint64_t Context::getFeatureFlags() const { return _featureFlags; }

bool Context::hasCudaSupport() const { return _cudaSupport; }

WorkerBuilder Context::workerBuilder()
{
  return WorkerBuilder(std::static_pointer_cast<Context>(shared_from_this()));
}

MemoryHandleBuilder Context::memoryHandleBuilder(size_t size)
{
  return MemoryHandleBuilder(std::static_pointer_cast<Context>(shared_from_this()), size);
}

}  // namespace ucxx
