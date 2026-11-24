/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <ucp/api/ucp_compat.h>
#include <ucs/type/status.h>
#include <utility>
#include <vector>

#include <ucp/api/ucp.h>

#include <internal/constructors.h>
#include <ucxx/component.h>
#include <ucxx/endpoint.h>

#include <ucxx/exception.h>
#include <ucxx/listener.h>
#include <ucxx/remote_key.h>
#include <ucxx/request_am.h>
#include <ucxx/request_am_builder.h>
#include <ucxx/request_data.h>
#include <ucxx/request_endpoint_close.h>
#include <ucxx/request_endpoint_close_builder.h>
#include <ucxx/request_flush.h>
#include <ucxx/request_flush_builder.h>
#include <ucxx/request_mem.h>
#include <ucxx/request_mem_builder.h>
#include <ucxx/request_stream.h>
#include <ucxx/request_stream_builder.h>
#include <ucxx/request_tag.h>
#include <ucxx/request_tag_builder.h>
#include <ucxx/request_tag_multi.h>
#include <ucxx/request_tag_multi_builder.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/callback_notifier.h>
#include <ucxx/utils/sockaddr.h>
#include <ucxx/utils/ucx.h>
#include <ucxx/worker.h>

namespace ucxx {

/**
 * Context passed to UCP endpoint error callback. Holds a weak_ptr to the endpoint
 * so the callback can safely check if the endpoint still exists (lock() before use).
 * If the endpoint was already destroyed, lock() returns null and we skip the callback body.
 *
 * The pointer we pass to UCP (err_handler.arg) is a heap-allocated EndpointErrorCallbackContext.
 * We only free it inside the callback when it runs. UCP is expected to invoke the error callback
 * when the endpoint is closed (e.g. ucp_ep_close_nbx) or when it errors; in those cases the
 * callback runs and we do not leak. If the callback is never invoked (e.g. process exit without
 * closing endpoints, or worker destroyed without closing), that allocation is intentionally not
 * freed because the callback may still run asynchronously after ~Endpoint() and would then
 * dereference a freed pointer, however, this likely indicates an implementation bug, since
 * ~Endpoint() ensures the underlying UCP endpoint is closed.
 */
struct EndpointErrorCallbackContext {
  std::weak_ptr<Endpoint> endpoint;

  explicit EndpointErrorCallbackContext(std::shared_ptr<Endpoint> ep) : endpoint(std::move(ep)) {}
};

static std::shared_ptr<Worker> getWorker(std::shared_ptr<Component> workerOrListener)
{
  auto worker = std::dynamic_pointer_cast<Worker>(workerOrListener);
  if (worker == nullptr) {
    auto listener = std::dynamic_pointer_cast<Listener>(workerOrListener);
    if (listener == nullptr)
      throw std::invalid_argument(
        "Invalid object, it's not a shared_ptr to either ucxx::Worker nor ucxx::Listener");

    worker = std::static_pointer_cast<Worker>(listener->getParent());
  }
  return worker;
}

void endpointErrorCallback(void* arg, ucp_ep_h ep, ucs_status_t status)
{
  if (arg == nullptr) return;

  std::unique_ptr<EndpointErrorCallbackContext> ctx(
    static_cast<EndpointErrorCallbackContext*>(arg));
  auto endpoint = ctx->endpoint.lock();
  // Endpoint has been destroyed.
  if (!endpoint) return;

  // Endpoint is already closing.
  if (endpoint->_closing.exchange(true)) return;

  endpoint->_status = status;
  auto worker       = ::ucxx::getWorker(endpoint->_parent);
  worker->scheduleRequestCancel(endpoint->_inflightRequests->release());
  {
    std::lock_guard<std::mutex> lock(endpoint->_mutex);
    if (endpoint->_closeCallback) {
      ucxx_debug("ucxx::Endpoint::%s: %p, UCP handle: %p, calling user close callback",
                 __func__,
                 endpoint.get(),
                 ep);
      endpoint->_closeCallback(status, endpoint->_closeCallbackArg);
      endpoint->_closeCallback    = nullptr;
      endpoint->_closeCallbackArg = nullptr;
    }
  }

  // Connection reset and timeout often represent just a normal remote
  // endpoint disconnect, log only in debug mode.
  if (status == UCS_ERR_CONNECTION_RESET || status == UCS_ERR_ENDPOINT_TIMEOUT)
    ucxx_debug("ucxx::Endpoint::%s: %p, UCP handle: %p, error callback called with status %d: %s",
               __func__,
               endpoint.get(),
               ep,
               status,
               ucs_status_string(status));
  else
    ucxx_error("ucxx::Endpoint::%s: %p, UCP handle: %p, error callback called with status %d: %s",
               __func__,
               endpoint.get(),
               ep,
               status,
               ucs_status_string(status));
}

Endpoint::Endpoint(std::shared_ptr<Component> workerOrListener, bool endpointErrorHandling)
  : _endpointErrorHandling{endpointErrorHandling}
{
  auto worker = ::ucxx::getWorker(workerOrListener);

  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  setParent(workerOrListener);
}

void Endpoint::create(ucp_ep_params_t* params)
{
  auto worker = ::ucxx::getWorker(_parent);

  if (_endpointErrorHandling) {
    params->err_mode       = UCP_ERR_HANDLING_MODE_PEER;
    params->err_handler.cb = endpointErrorCallback;
    params->err_handler.arg =
      new EndpointErrorCallbackContext(std::static_pointer_cast<Endpoint>(shared_from_this()));
  } else {
    params->err_mode        = UCP_ERR_HANDLING_MODE_NONE;
    params->err_handler.cb  = nullptr;
    params->err_handler.arg = nullptr;
  }

  if (worker->isProgressThreadRunning()) {
    ucs_status_t status = UCS_INPROGRESS;

    size_t maxAttempts = 3;
    for (uint64_t i = 0; i < maxAttempts; ++i) {
      if (worker->registerGenericPre(
            [this, &worker, &params, &status]() {
              status = ucp_ep_create(worker->getHandle(), params, &_handle);
            },
            3000000000 /* 3s */))
        break;

      if (i == maxAttempts - 1) {
        status = UCS_ERR_TIMED_OUT;
        ucxx_error("Timeout waiting for ucp_ep_create, all attempts failed");
      } else {
        ucxx_warn("Timeout waiting for ucp_ep_create, retrying");
      }
    }
    utils::ucsErrorThrow(status);
  } else {
    utils::ucsErrorThrow(ucp_ep_create(worker->getHandle(), params, &_handle));
  }

  ucxx_trace("ucxx::Endpoint created: %p, UCP handle: %p, parent: %p, endpointErrorHandling: %d",
             this,
             _handle,
             _parent.get(),
             _endpointErrorHandling);
}

std::shared_ptr<Endpoint> detail::ConstructorFactory::createEndpointFromHostname(
  std::shared_ptr<Worker> worker, std::string ipAddress, uint16_t port, bool endpointErrorHandling)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  ucp_ep_params_t params = {.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR |
                                          UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                          UCP_EP_PARAM_FIELD_ERR_HANDLER,
                            .flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER};
  auto info              = ucxx::utils::get_addrinfo(ipAddress.c_str(), port);

  params.sockaddr.addrlen = info->ai_addrlen;
  params.sockaddr.addr    = info->ai_addr;

  auto ep = std::shared_ptr<Endpoint>(new Endpoint(worker, endpointErrorHandling));
  ep->create(&params);
  return ep;
}

std::shared_ptr<Endpoint> detail::ConstructorFactory::createEndpointFromConnRequest(
  std::shared_ptr<Listener> listener, ucp_conn_request_h connRequest, bool endpointErrorHandling)
{
  if (listener == nullptr || listener->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");

  ucp_ep_params_t params = {
    .field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_CONN_REQUEST |
                  UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE | UCP_EP_PARAM_FIELD_ERR_HANDLER,
    .flags        = UCP_EP_PARAMS_FLAGS_NO_LOOPBACK,
    .conn_request = connRequest};

  auto ep = std::shared_ptr<Endpoint>(new Endpoint(listener, endpointErrorHandling));
  ep->create(&params);
  return ep;
}

std::shared_ptr<Endpoint> detail::ConstructorFactory::createEndpointFromWorkerAddress(
  std::shared_ptr<Worker> worker, std::shared_ptr<Address> address, bool endpointErrorHandling)
{
  if (worker == nullptr || worker->getHandle() == nullptr)
    throw ucxx::Error("Worker not initialized");
  if (address == nullptr || address->getHandle() == nullptr || address->getLength() == 0)
    throw ucxx::Error("Address not initialized");

  ucp_ep_params_t params = {.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS |
                                          UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                          UCP_EP_PARAM_FIELD_ERR_HANDLER,
                            .address = address->getHandle()};

  auto ep = std::shared_ptr<Endpoint>(new Endpoint(worker, endpointErrorHandling));
  ep->create(&params);
  return ep;
}

Endpoint::~Endpoint()
{
  closeBlocking(10000000000 /* 10s */);
  ucxx_trace("ucxx::Endpoint destroyed: %p, UCP handle: %p", this, _originalHandle);
}

RequestEndpointCloseBuilder Endpoint::closeBuilder()
{
  return requestEndpointCloseBuilder(std::static_pointer_cast<Endpoint>(shared_from_this()),
                                     data::EndpointClose(_endpointErrorHandling));
}

std::shared_ptr<RequestEndpointClose> Endpoint::closeRequest(
  const bool force,
  const bool enablePythonFuture,
  EndpointCloseCallbackUserFunction callbackFunction,
  EndpointCloseCallbackUserData callbackData)
{
  if (_closing.exchange(true) || _handle == nullptr) return nullptr;

  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());

  auto combineCallbacksFunction = [this, callbackFunction, callbackData](
                                    ucs_status_t status,
                                    EndpointCloseCallbackUserData /* callbackData */) {
    _status = status;
    if (callbackFunction) callbackFunction(status, callbackData);
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (_closeCallback) {
        _closeCallback(status, _closeCallbackArg);
        _closeCallback    = nullptr;
        _closeCallbackArg = nullptr;
      }
    }
  };

  auto req = detail::ConstructorFactory::createRequestEndpointClose(
    endpoint, data::EndpointClose(force), enablePythonFuture, combineCallbacksFunction, nullptr);
  std::ignore = registerInflightRequest(req);
  return req;
}

void Endpoint::closeBlocking(uint64_t period, uint64_t maxAttempts)
{
  if (_closing.exchange(true) || _handle == nullptr) return;

  size_t canceled = cancelInflightRequestsBlocking(3000000000 /* 3s */, 3);
  ucxx_debug("ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, canceled %lu requests",
             __func__,
             this,
             _handle,
             canceled);

  ucp_request_param_t param{};
  if (_endpointErrorHandling)
    param = {.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS, .flags = UCP_EP_CLOSE_FLAG_FORCE};

  auto worker             = ::ucxx::getWorker(_parent);
  ucs_status_ptr_t status = nullptr;
  bool closeComplete      = false;

  auto updateCloseStatus = [this, &closeComplete, &status]() {
    if (UCS_PTR_IS_PTR(status)) {
      ucs_status_t s;
      if ((s = ucp_request_check_status(status)) != UCS_INPROGRESS) {
        _status       = s;
        closeComplete = true;
      }
    } else {
      _status       = UCS_PTR_STATUS(status);
      closeComplete = true;
      if (_status != UCS_OK) {
        ucxx_error(
          "ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, Error while closing endpoint: %s",
          __func__,
          this,
          _handle,
          ucs_status_string(_status));
      }
    }
  };

  if (worker->isProgressThreadRunning()) {
    bool submitted = false;
    for (uint64_t i = 0; i < maxAttempts && !closeComplete; ++i) {
      if (!submitted) {
        if (!worker->registerGenericPre(
              [this, &status, &param, &updateCloseStatus]() {
                status = ucp_ep_close_nbx(_handle, &param);
                updateCloseStatus();
              },
              period))
          continue;
        submitted = true;
      }

      if (!closeComplete && !worker->registerGenericPost(updateCloseStatus, period)) continue;
    }

    if (!closeComplete) {
      _status = UCS_ERR_ENDPOINT_TIMEOUT;
      ucxx_debug(
        "ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, all attempts to close timed out",
        __func__,
        this,
        _handle);
    }
  } else {
    status = ucp_ep_close_nbx(_handle, &param);
    updateCloseStatus();
    if (UCS_PTR_IS_PTR(status)) {
      while (!closeComplete) {
        worker->progress();
        updateCloseStatus();
      }
    }
  }
  ucxx_trace("ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, closed", __func__, this, _handle);

  if (UCS_PTR_IS_PTR(status)) ucp_request_free(status);

  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_closeCallback) {
      ucxx_debug("ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, calling user close callback",
                 __func__,
                 this,
                 _handle);
      _closeCallback(_status, _closeCallbackArg);
      _closeCallback    = nullptr;
      _closeCallbackArg = nullptr;
    }
  }

  std::swap(_handle, _originalHandle);
}

ucp_ep_h Endpoint::getHandle() { return _handle; }

bool Endpoint::isAlive() const
{
  if (!_endpointErrorHandling) return true;

  return _status == UCS_INPROGRESS;
}

void Endpoint::raiseOnError()
{
  ucs_status_t status = _status;

  if (status == UCS_OK || status == UCS_INPROGRESS || !_endpointErrorHandling) return;

  std::string statusString{ucs_status_string(status)};
  std::stringstream errorMsgStream;
  errorMsgStream << "Endpoint " << std::hex << _handle << " error: " << statusString;

  utils::ucsErrorThrow(status, errorMsgStream.str());
}

void Endpoint::setCloseCallback(EndpointCloseCallbackUserFunction closeCallback,
                                EndpointCloseCallbackUserData closeCallbackArg)
{
  std::lock_guard<std::mutex> lock(_mutex);

  if (_closing.load() && closeCallback != nullptr && closeCallbackArg != nullptr)
    throw std::runtime_error("Endpoint is closing or has already closed.");

  _closeCallback    = closeCallback;
  _closeCallbackArg = closeCallbackArg;
}

std::shared_ptr<Request> Endpoint::registerInflightRequest(std::shared_ptr<Request> request)
{
  if (!request->isCompleted()) _inflightRequests->insert(request);

  /**
   * If the endpoint closed or errored while the request was being submitted, the error
   * handler may have been called already and we need to register any new requests for
   * cancelation, including the present one.
   */
  if (_status != UCS_INPROGRESS) {
    auto worker = ::ucxx::getWorker(_parent);
    worker->scheduleRequestCancel(_inflightRequests->release());
  }

  return request;
}

void Endpoint::removeInflightRequest(std::shared_ptr<Request> request)
{
  _inflightRequests->remove(request);
}

size_t Endpoint::cancelInflightRequests() { return _inflightRequests->cancelAll(); }

size_t Endpoint::cancelInflightRequestsBlocking(uint64_t period, uint64_t maxAttempts)
{
  auto worker     = ::ucxx::getWorker(_parent);
  size_t canceled = 0;

  if (std::this_thread::get_id() == worker->getProgressThreadId()) {
    canceled = _inflightRequests->cancelAll();
    for (uint64_t i = 0; i < maxAttempts && _inflightRequests->getCancelingSize() > 0; ++i)
      worker->progress();
  } else if (worker->isProgressThreadRunning()) {
    bool cancelSuccess = false;
    for (uint64_t i = 0; i < maxAttempts && !cancelSuccess; ++i) {
      if (!worker->registerGenericPre(
            [this, &canceled]() { canceled += _inflightRequests->cancelAll(); }, period))
        continue;

      if (!worker->registerGenericPost(
            [this, &cancelSuccess]() {
              cancelSuccess = _inflightRequests->getCancelingSize() == 0;
            },
            period))
        continue;
    }
    if (!cancelSuccess)
      ucxx_debug(
        "ucxx::Endpoint::%s, Endpoint: %p, UCP handle: %p, all attempts to "
        "cancel inflight requests failed",
        __func__,
        this,
        _handle);
  } else {
    canceled = _inflightRequests->cancelAll();
  }

  return canceled;
}

size_t Endpoint::getCancelingSize() const { return _inflightRequests->getCancelingSize(); }

RequestAmBuilder Endpoint::amSendBuilder(const void* const buffer,
                                         const size_t length,
                                         const ucs_memory_type_t memoryType)
{
  auto params       = AmSendParams{};
  params.memoryType = memoryType;

  return amSendBuilder(buffer, length, params);
}

RequestAmBuilder Endpoint::amSendBuilder(const void* const buffer,
                                         const size_t length,
                                         const AmSendParams& params)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestAmBuilder(std::move(endpoint), data::AmSend(buffer, length, params));
}

RequestAmBuilder Endpoint::amSendBuilder(std::vector<ucp_dt_iov_t> iov, const AmSendParams& params)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestAmBuilder(std::move(endpoint), data::AmSend(std::move(iov), params));
}

RequestAmBuilder Endpoint::amRecvBuilder()
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestAmBuilder(std::move(endpoint), data::AmReceive());
}

RequestMemBuilder Endpoint::memGetBuilder(void* buffer,
                                          size_t length,
                                          uint64_t remoteAddr,
                                          ucp_rkey_h rkey)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestMemBuilder(std::move(endpoint), data::MemGet(buffer, length, remoteAddr, rkey));
}

RequestMemBuilder Endpoint::memGetBuilder(void* buffer,
                                          size_t length,
                                          std::shared_ptr<RemoteKey> remoteKey)
{
  return memGetBuilder(buffer, length, remoteKey->getBaseAddress(), remoteKey->getHandle());
}

RequestMemBuilder Endpoint::memPutBuilder(const void* const buffer,
                                          size_t length,
                                          uint64_t remoteAddr,
                                          ucp_rkey_h rkey)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestMemBuilder(std::move(endpoint), data::MemPut(buffer, length, remoteAddr, rkey));
}

RequestMemBuilder Endpoint::memPutBuilder(const void* const buffer,
                                          size_t length,
                                          std::shared_ptr<RemoteKey> remoteKey)
{
  return memPutBuilder(buffer, length, remoteKey->getBaseAddress(), remoteKey->getHandle());
}

RequestStreamBuilder Endpoint::streamSendBuilder(const void* const buffer, size_t length)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestStreamBuilder(std::move(endpoint), data::StreamSend(buffer, length));
}

RequestStreamBuilder Endpoint::streamRecvBuilder(void* buffer, size_t length)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestStreamBuilder(std::move(endpoint), data::StreamReceive(buffer, length));
}

RequestTagBuilder Endpoint::tagSendBuilder(const void* const buffer, size_t length, Tag tag)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestTagBuilder(std::move(endpoint), data::TagSend(buffer, length, tag));
}

RequestTagBuilder Endpoint::tagRecvBuilder(void* buffer, size_t length, Tag tag, TagMask tagMask)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestTagBuilder(std::move(endpoint), data::TagReceive(buffer, length, tag, tagMask));
}

RequestTagMultiBuilder Endpoint::tagMultiSendBuilder(const std::vector<const void*>& buffer,
                                                     const std::vector<size_t>& size,
                                                     const std::vector<int>& isCUDA,
                                                     const Tag tag)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestTagMultiBuilder(std::move(endpoint), data::TagMultiSend(buffer, size, isCUDA, tag));
}

RequestTagMultiBuilder Endpoint::tagMultiRecvBuilder(const Tag tag, const TagMask tagMask)
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestTagMultiBuilder(std::move(endpoint), data::TagMultiReceive(tag, tagMask));
}

RequestFlushBuilder Endpoint::flushBuilder()
{
  auto endpoint = std::static_pointer_cast<Endpoint>(shared_from_this());
  return RequestFlushBuilder(std::move(endpoint), data::Flush());
}

std::shared_ptr<Worker> Endpoint::getWorker() { return ::ucxx::getWorker(_parent); }

RemoteKeyBuilder Endpoint::remoteKeyBuilder(SerializedRemoteKey serializedRemoteKey)
{
  return RemoteKeyBuilder(std::static_pointer_cast<Endpoint>(shared_from_this()),
                          std::move(serializedRemoteKey));
}

}  // namespace ucxx
