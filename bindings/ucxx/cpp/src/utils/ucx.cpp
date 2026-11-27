/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <string>
#include <utility>

#include <ucs/type/status.h>
#include <ucxx/exception.h>
#include <ucxx/utils/ucx.h>

namespace ucxx {

namespace utils {

void ucsErrorThrow(const ucs_status_t status, std::string userMessage)
{
  std::string message = userMessage.empty() ? ucs_status_string(status) : std::move(userMessage);

  switch (status) {
    case UCS_OK: return;
    case UCS_INPROGRESS: return;
    case UCS_ERR_NO_MESSAGE: throw ucxx::NoMessageError(std::move(message)); return;
    case UCS_ERR_NO_RESOURCE: throw ucxx::NoResourceError(std::move(message)); return;
    case UCS_ERR_IO_ERROR: throw ucxx::IOError(std::move(message)); return;
    case UCS_ERR_NO_MEMORY: throw ucxx::NoMemoryError(std::move(message)); return;
    case UCS_ERR_INVALID_PARAM: throw ucxx::InvalidParamError(std::move(message)); return;
    case UCS_ERR_UNREACHABLE: throw ucxx::UnreachableError(std::move(message)); return;
    case UCS_ERR_INVALID_ADDR: throw ucxx::InvalidAddrError(std::move(message)); return;
    case UCS_ERR_NOT_IMPLEMENTED: throw ucxx::NotImplementedError(std::move(message)); return;
    case UCS_ERR_MESSAGE_TRUNCATED: throw ucxx::MessageTruncatedError(std::move(message)); return;
    case UCS_ERR_NO_PROGRESS: throw ucxx::NoProgressError(std::move(message)); return;
    case UCS_ERR_BUFFER_TOO_SMALL: throw ucxx::BufferTooSmallError(std::move(message)); return;
    case UCS_ERR_NO_ELEM: throw ucxx::NoElemError(std::move(message)); return;
    case UCS_ERR_SOME_CONNECTS_FAILED:
      throw ucxx::SomeConnectsFailedError(std::move(message));
      return;
    case UCS_ERR_NO_DEVICE: throw ucxx::NoDeviceError(std::move(message)); return;
    case UCS_ERR_BUSY: throw ucxx::BusyError(std::move(message)); return;
    case UCS_ERR_CANCELED: throw ucxx::CanceledError(std::move(message)); return;
    case UCS_ERR_SHMEM_SEGMENT: throw ucxx::ShmemSegmentError(std::move(message)); return;
    case UCS_ERR_ALREADY_EXISTS: throw ucxx::AlreadyExistsError(std::move(message)); return;
    case UCS_ERR_OUT_OF_RANGE: throw ucxx::OutOfRangeError(std::move(message)); return;
    case UCS_ERR_TIMED_OUT: throw ucxx::TimedOutError(std::move(message)); return;
    case UCS_ERR_EXCEEDS_LIMIT: throw ucxx::ExceedsLimitError(std::move(message)); return;
    case UCS_ERR_UNSUPPORTED: throw ucxx::UnsupportedError(std::move(message)); return;
    case UCS_ERR_REJECTED: throw ucxx::RejectedError(std::move(message)); return;
    case UCS_ERR_NOT_CONNECTED: throw ucxx::NotConnectedError(std::move(message)); return;
    case UCS_ERR_CONNECTION_RESET: throw ucxx::ConnectionResetError(std::move(message)); return;
    case UCS_ERR_FIRST_LINK_FAILURE: throw ucxx::FirstLinkFailureError(std::move(message)); return;
    case UCS_ERR_LAST_LINK_FAILURE: throw ucxx::LastLinkFailureError(std::move(message)); return;
    case UCS_ERR_FIRST_ENDPOINT_FAILURE:
      throw ucxx::FirstEndpointFailureError(std::move(message));
      return;
    case UCS_ERR_ENDPOINT_TIMEOUT: throw ucxx::EndpointTimeoutError(std::move(message)); return;
    case UCS_ERR_LAST_ENDPOINT_FAILURE:
      throw ucxx::LastEndpointFailureError(std::move(message));
      return;
    default: throw ucxx::Error(std::move(message)); return;
  }
}

}  // namespace utils

}  // namespace ucxx
