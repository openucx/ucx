/**
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <atomic>
#include <features.h>

#ifdef __GLIBC__
#include <gnu/libc-version.h>
#include <string>
#endif

#include <ucxx/log.h>
#include <ucxx/utils/callback_notifier.h>

namespace ucxx {
namespace utils {

#ifdef __GLIBC__
static const bool _useSpinlock = []() {
  auto const libcVersion = std::string{gnu_get_libc_version()};
  auto const dot         = libcVersion.find(".");
  if (dot == std::string::npos) {
    return false;
  } else {
    // See https://sourceware.org/bugzilla/show_bug.cgi?id=13165
    auto const glibcMajor = std::stoi(libcVersion.substr(0, dot).data());
    auto const glibcMinor = std::stoi(libcVersion.substr(dot + 1).data());
    auto const use        = glibcMajor < 2 || (glibcMajor == 2 && glibcMinor < 25);
    ucxx_debug("glibc version %s detected, spinlock use is %d", libcVersion.c_str(), use);
    return use;
  }
}();
#else
static constexpr bool _useSpinlock = false;
#endif

void CallbackNotifier::set()
{
  if (_useSpinlock) {
    _flag.store(true, std::memory_order_release);
  } else {
    {
      std::lock_guard lock(_mutex);
      // This can be relaxed because the mutex is providing
      // ordering.
      _flag.store(true, std::memory_order_relaxed);
    }
    _conditionVariable.notify_one();
  }
}

bool CallbackNotifier::wait(uint64_t period,
                            std::function<void(void)> signalWorkerFunction,
                            uint64_t signalInterval)
{
  if (_useSpinlock) {
    while (!_flag.load(std::memory_order_acquire)) {}
  } else {
    std::unique_lock lock(_mutex);
    // Likewise here, the mutex provides ordering.
    if (period > 0) {
      if (signalInterval > 0) {
        auto attempts = (period + signalInterval - 1) / signalInterval;
        bool ret      = false;
        for (size_t i = 0; i < attempts; ++i) {
          ret = _conditionVariable.wait_for(
            lock, std::chrono::duration<uint64_t, std::nano>(signalInterval), [this]() {
              return _flag.load(std::memory_order_relaxed) == true;
            });
          if (signalWorkerFunction) signalWorkerFunction();
        }
        return ret;
      } else {
        return _conditionVariable.wait_for(
          lock, std::chrono::duration<uint64_t, std::nano>(period), [this]() {
            return _flag.load(std::memory_order_relaxed);
          });
      }
    } else {
      _conditionVariable.wait(lock,
                              [this]() { return _flag.load(std::memory_order_relaxed) == true; });
    }
  }
  return true;
}

}  // namespace utils
}  // namespace ucxx
