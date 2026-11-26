/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <ucxx/buffer.h>
#include <ucxx/delayed_submission.h>
#include <ucxx/internal/request_am.h>
#include <ucxx/request_am.h>
#include <ucxx/typedefs.h>

#include <memory>
#include <utility>
#include <vector>

namespace ucxx {

namespace internal {

RecvAmMessage::RecvAmMessage(internal::AmData* amData,
                             ucp_ep_h ep,
                             std::shared_ptr<RequestAm> request,
                             std::shared_ptr<Buffer> buffer,
                             AmReceiverCallbackType receiverCallback,
                             std::vector<std::byte> userHeader)
  : _amData(amData), _ep(ep), _request(request)
{
  std::visit(data::dispatch{
               [this, buffer, &userHeader](data::AmReceive& amReceive) {
                 amReceive._buffer     = buffer;
                 amReceive._userHeader = std::move(userHeader);
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _request->_requestData);

  if (receiverCallback) {
    _request->_callback = [this, receiverCallback](ucs_status_t, std::shared_ptr<void>) {
      receiverCallback(_request, _ep);
    };
  }
}

void RecvAmMessage::callback(void* request, ucs_status_t status)
{
  std::visit(data::dispatch{
               [this, request, status](data::AmReceive) {
                 _request->callback(request, status);
                 {
                   std::lock_guard<std::mutex> lock(_amData->_mutex);
                   _amData->_recvAmMessageMap.erase(_request);
                 }
               },
               [](auto) { throw std::runtime_error("Unreachable"); },
             },
             _request->_requestData);
}

}  // namespace internal

}  // namespace ucxx
