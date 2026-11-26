/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <string>

#include <ucp/api/ucp.h>

namespace ucxx {

namespace utils {

/**
 * @brief Throw appropriate exception on UCS error.
 *
 * Throw an exception appropriately mapped from a UCS error. For status `UCS_OK` and
 * `UCS_INPROGRESS` or unknown values, calling this function acts as a no-op. Additionally
 * set the `msg_context` string to a human-readable error message.
 *
 * @param[in]  status       UCS status for which to check state.
 * @param[out] userMessage  user-defined, human-readable error message.
 *
 * @throw NoMessageError if `status == UCS_ERR_NO_MESSAGE`
 * @throw NoResourceError if `status == UCS_ERR_NO_RESOURCE`
 * @throw IOError if `status == UCS_ERR_IO_ERROR`
 * @throw NoMemoryError if `status == UCS_ERR_NO_MEMORY`
 * @throw InvalidParamError if `status == UCS_ERR_INVALID_PARAM`
 * @throw UnreachableError if `status == UCS_ERR_UNREACHABLE`
 * @throw InvalidAddrError if `status == UCS_ERR_INVALID_ADDR`
 * @throw NotImplementedError if `status == UCS_ERR_NOT_IMPLEMENTED`
 * @throw MessageTruncatedError if `status == UCS_ERR_MESSAGE_TRUNCATED`
 * @throw NoProgressError if `status == UCS_ERR_NO_PROGRESS`
 * @throw BufferTooSmallError if `status == UCS_ERR_BUFFER_TOO_SMALL`
 * @throw NoElemError if `status == UCS_ERR_NO_ELEM`
 * @throw SomeConnectsFailedError if `status == UCS_ERR_SOME_CONNECTS_FAILED`
 * @throw NoDeviceError if `status == UCS_ERR_NO_DEVICE`
 * @throw BusyError if `status == UCS_ERR_BUSY`
 * @throw CanceledError if `status == UCS_ERR_CANCELED`
 * @throw ShmemSegmentError if `status == UCS_ERR_SHMEM_SEGMENT`
 * @throw AlreadyExistsError if `status == UCS_ERR_ALREADY_EXISTS`
 * @throw OutOfRangeError if `status == UCS_ERR_OUT_OF_RANGE`
 * @throw TimedOutError if `status == UCS_ERR_TIMED_OUT`
 * @throw ExceedsLimitError if `status == UCS_ERR_EXCEEDS_LIMIT`
 * @throw UnsupportedError if `status == UCS_ERR_UNSUPPORTED`
 * @throw RejectedError if `status == UCS_ERR_REJECTED`
 * @throw NotConnectedError if `status == UCS_ERR_NOT_CONNECTED`
 * @throw ConnectionResetError if `status == UCS_ERR_CONNECTION_RESET`
 * @throw FirstLinkFailureError if `status == UCS_ERR_FIRST_LINK_FAILURE`
 * @throw LastLinkFailureError if `status == UCS_ERR_LAST_LINK_FAILURE`
 * @throw FirstEndpointFailureError if `status == UCS_ERR_FIRST_ENDPOINT_FAILURE`
 * @throw EndpointTimeoutError if `status == UCS_ERR_ENDPOINT_TIMEOUT`
 * @throw LastEndpointFailureError if `status == UCS_ERR_LAST_ENDPOINT_FAILURE`
 */
void ucsErrorThrow(const ucs_status_t status, std::string userMessage = "");

}  // namespace utils

}  // namespace ucxx
