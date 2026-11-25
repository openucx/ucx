/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <exception>
#include <string>
#include <utility>

namespace ucxx {

/**
 * @brief The base class for all UCX exceptions.
 *
 * The base class for all UCX errors that may occur made into C++ exceptions.
 */
class Error : public std::exception {
 private:
  std::string _msg{};

 public:
  /**
   * @brief The base class constructor.
   *
   * The base class constructor taking the explanatory string of the error.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit Error(const std::string& msg) : _msg{std::move(msg)} {}

  /**
   * @brief Returns an explanatory string.
   *
   * Returns an explanatory string of the UCX error that has occurred.
   *
   * @returns the explanatory string.
   */
  const char* what() const noexcept override { return this->_msg.c_str(); }
};

/**
 * @brief The exception for `UCS_ERR_NO_MESSAGE`.
 *
 * The exception raised when `UCS_ERR_NO_MESSAGE` occurs in UCX.
 */
class NoMessageError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_NO_MESSAGE` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_NO_MESSAGE` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit NoMessageError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_NO_RESOURCE`.
 *
 * The exception raised when `UCS_ERR_NO_RESOURCE` occurs in UCX.
 */
class NoResourceError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_NO_RESOURCE` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_NO_RESOURCE` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit NoResourceError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_IO_ERROR`.
 *
 * The exception raised when `UCS_ERR_IO_ERROR` occurs in UCX.
 */
class IOError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_IO_ERROR` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_IO_ERROR` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit IOError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_NO_MEMORY`.
 *
 * The exception raised when `UCS_ERR_NO_MEMORY` occurs in UCX.
 */
class NoMemoryError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_NO_MEMORY` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_NO_MEMORY` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit NoMemoryError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_INVALID_PARAM`.
 *
 * The exception raised when `UCS_ERR_INVALID_PARAM` occurs in UCX.
 */
class InvalidParamError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_INVALID_PARAM` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_INVALID_PARAM` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit InvalidParamError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_UNREACHABLE`.
 *
 * The exception raised when `UCS_ERR_UNREACHABLE` occurs in UCX.
 */
class UnreachableError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_UNREACHABLE` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_UNREACHABLE` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit UnreachableError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_INVALID_ADDR`.
 *
 * The exception raised when `UCS_ERR_INVALID_ADDR` occurs in UCX.
 */
class InvalidAddrError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_INVALID_ADDR` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_INVALID_ADDR` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit InvalidAddrError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_NOT_IMPLEMENTED`.
 *
 * The exception raised when `UCS_ERR_NOT_IMPLEMENTED` occurs in UCX.
 */
class NotImplementedError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_NOT_IMPLEMENTED` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_NOT_IMPLEMENTED` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit NotImplementedError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_MESSAGE_TRUNCATED`.
 *
 * The exception raised when `UCS_ERR_MESSAGE_TRUNCATED` occurs in UCX.
 */
class MessageTruncatedError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_MESSAGE_TRUNCATED` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_MESSAGE_TRUNCATED` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit MessageTruncatedError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_NO_PROGRESS`.
 *
 * The exception raised when `UCS_ERR_NO_PROGRESS` occurs in UCX.
 */
class NoProgressError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_NO_PROGRESS` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_NO_PROGRESS` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit NoProgressError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_BUFFER_TOO_SMALL`.
 *
 * The exception raised when `UCS_ERR_BUFFER_TOO_SMALL` occurs in UCX.
 */
class BufferTooSmallError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_BUFFER_TOO_SMALL` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_BUFFER_TOO_SMALL` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit BufferTooSmallError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_NO_ELEM`.
 *
 * The exception raised when `UCS_ERR_NO_ELEM` occurs in UCX.
 */
class NoElemError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_NO_ELEM` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_NO_ELEM` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit NoElemError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_SOME_CONNECTS_FAILED`.
 *
 * The exception raised when `UCS_ERR_SOME_CONNECTS_FAILED` occurs in UCX.
 */
class SomeConnectsFailedError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_SOME_CONNECTS_FAILED` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_SOME_CONNECTS_FAILED` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit SomeConnectsFailedError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_NO_DEVICE`.
 *
 * The exception raised when `UCS_ERR_NO_DEVICE` occurs in UCX.
 */
class NoDeviceError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_NO_DEVICE` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_NO_DEVICE` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit NoDeviceError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_BUSY`.
 *
 * The exception raised when `UCS_ERR_BUSY` occurs in UCX.
 */
class BusyError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_BUSY` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_BUSY` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit BusyError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_CANCELED`.
 *
 * The exception raised when `UCS_ERR_CANCELED` occurs in UCX.
 */
class CanceledError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_CANCELED` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_CANCELED` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit CanceledError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_SHMEM_SEGMENT`.
 *
 * The exception raised when `UCS_ERR_SHMEM_SEGMENT` occurs in UCX.
 */
class ShmemSegmentError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_SHMEM_SEGMENT` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_SHMEM_SEGMENT` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit ShmemSegmentError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_ALREADY_EXISTS`.
 *
 * The exception raised when `UCS_ERR_ALREADY_EXISTS` occurs in UCX.
 */
class AlreadyExistsError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_ALREADY_EXISTS` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_ALREADY_EXISTS` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit AlreadyExistsError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_OUT_OF_RANGE`.
 *
 * The exception raised when `UCS_ERR_OUT_OF_RANGE` occurs in UCX.
 */
class OutOfRangeError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_OUT_OF_RANGE` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_OUT_OF_RANGE` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit OutOfRangeError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_TIMED_OUT`.
 *
 * The exception raised when `UCS_ERR_TIMED_OUT` occurs in UCX.
 */
class TimedOutError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_TIMED_OUT` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_TIMED_OUT` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit TimedOutError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_EXCEEDS_LIMIT`.
 *
 * The exception raised when `UCS_ERR_EXCEEDS_LIMIT` occurs in UCX.
 */
class ExceedsLimitError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_EXCEEDS_LIMIT` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_EXCEEDS_LIMIT` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit ExceedsLimitError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_UNSUPPORTED`.
 *
 * The exception raised when `UCS_ERR_UNSUPPORTED` occurs in UCX.
 */
class UnsupportedError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_UNSUPPORTED` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_UNSUPPORTED` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit UnsupportedError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_REJECTED`.
 *
 * The exception raised when `UCS_ERR_REJECTED` occurs in UCX.
 */
class RejectedError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_REJECTED` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_REJECTED` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit RejectedError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_NOT_CONNECTED`.
 *
 * The exception raised when `UCS_ERR_NOT_CONNECTED` occurs in UCX.
 */
class NotConnectedError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_NOT_CONNECTED` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_NOT_CONNECTED` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit NotConnectedError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_CONNECTION_RESET`.
 *
 * The exception raised when `UCS_ERR_CONNECTION_RESET` occurs in UCX.
 */
class ConnectionResetError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_CONNECTION_RESET` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_CONNECTION_RESET` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit ConnectionResetError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_FIRST_LINK_FAILURE`.
 *
 * The exception raised when `UCS_ERR_FIRST_LINK_FAILURE` occurs in UCX.
 */
class FirstLinkFailureError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_FIRST_LINK_FAILURE` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_FIRST_LINK_FAILURE` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit FirstLinkFailureError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_LAST_LINK_FAILURE`.
 *
 * The exception raised when `UCS_ERR_LAST_LINK_FAILURE` occurs in UCX.
 */
class LastLinkFailureError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_LAST_LINK_FAILURE` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_LAST_LINK_FAILURE` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit LastLinkFailureError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_FIRST_ENDPOINT_FAILURE`.
 *
 * The exception raised when `UCS_ERR_FIRST_ENDPOINT_FAILURE` occurs in UCX.
 */
class FirstEndpointFailureError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_FIRST_ENDPOINT_FAILURE` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_FIRST_ENDPOINT_FAILURE` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit FirstEndpointFailureError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_ENDPOINT_TIMEOUT`.
 *
 * The exception raised when `UCS_ERR_ENDPOINT_TIMEOUT` occurs in UCX.
 */
class EndpointTimeoutError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_ENDPOINT_TIMEOUT` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_ENDPOINT_TIMEOUT` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit EndpointTimeoutError(const std::string& msg) : Error(msg) {}
};

/**
 * @brief The exception for `UCS_ERR_LAST_ENDPOINT_FAILURE`.
 *
 * The exception raised when `UCS_ERR_LAST_ENDPOINT_FAILURE` occurs in UCX.
 */
class LastEndpointFailureError : public Error {
 public:
  /**
   * @brief The `UCS_ERR_LAST_ENDPOINT_FAILURE` constructor.
   *
   * The constructor for an exception raised with `UCS_ERR_LAST_ENDPOINT_FAILURE` occurs in UCX.
   *
   * @param[in] msg the explanatory string of the error.
   */
  explicit LastEndpointFailureError(const std::string& msg) : Error(msg) {}
};

}  // namespace ucxx
