/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdio>
#include <cstring>
#include <functional>
#include <ios>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>

#include <internal/constructors.h>
#include <ucxx/buffer.h>
#include <ucxx/internal/request_am.h>
#include <ucxx/request_am.h>
#include <ucxx/request_flush.h>
#include <ucxx/request_flush_builder.h>
#include <ucxx/request_tag.h>
#include <ucxx/request_tag_builder.h>
#include <ucxx/tag_probe.h>
#include <ucxx/typedefs.h>
#include <ucxx/utils/callback_notifier.h>
#include <ucxx/utils/file_descriptor.h>
#include <ucxx/utils/ucx.h>
#include <ucxx/worker.h>

namespace ucxx {

Worker::Worker(std::shared_ptr<Context> context,
               const bool enableDelayedSubmission,
               const bool enableFuture)
  : _enableFuture(enableFuture)
{
  if (context == nullptr || context->getHandle() == nullptr)
    throw std::runtime_error("Context not initialized");
#if UCXX_ENABLE_CCCL
  _cudaBufferType = BufferType::CCCL;
#endif

  ucp_worker_params_t params = {.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
                                .thread_mode = UCS_THREAD_MODE_MULTI};
  utils::ucsErrorThrow(ucp_worker_create(context->getHandle(), &params, &_handle));

  _delayedSubmissionCollection =
    std::make_shared<DelayedSubmissionCollection>(enableDelayedSubmission);

  if (context->getFeatureFlags() & UCP_FEATURE_AM) {
    unsigned int AM_MSG_ID            = 0;
    _amData                           = std::make_shared<internal::AmData>();
    _amData->_registerInflightRequest = [this](std::shared_ptr<Request> req) {
      return registerInflightRequest(req);
    };
    registerAmAllocator(UCS_MEMORY_TYPE_HOST,
                        [](size_t length) { return std::make_shared<HostBuffer>(length); });

    ucp_am_handler_param_t am_handler_param = {.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                                                             UCP_AM_HANDLER_PARAM_FIELD_CB |
                                                             UCP_AM_HANDLER_PARAM_FIELD_ARG,
                                               .id  = AM_MSG_ID,
                                               .cb  = RequestAm::recvCallback,
                                               .arg = _amData.get()};
    utils::ucsErrorThrow(ucp_worker_set_am_recv_handler(_handle, &am_handler_param));
  }

  ucxx_trace(
    "ucxx::Worker created: %p, UCP handle: %p, enableDelayedSubmission: %d, enableFuture: %d",
    this,
    _handle,
    enableDelayedSubmission,
    _enableFuture);

  setParent(context);
}

void Worker::drainWorkerTagRecv()
{
  auto context = std::static_pointer_cast<Context>(_parent);
  if (!(context->getFeatureFlags() & UCP_FEATURE_TAG)) return;

  ucp_tag_message_h message;
  ucp_tag_recv_info_t info;

  while ((message = ucp_tag_probe_nb(_handle, 0, 0, 1, &info)) != NULL) {
    ucxx_debug(
      "ucxx::Worker::%s, Worker: %p, UCP handle: %p, tag: 0x%lx, length: %lu, "
      "draining tag receive messages",
      __func__,
      this,
      _handle,
      info.sender_tag,
      info.length);

    std::vector<char> buf(info.length);

    ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE,
                                 .datatype     = ucp_dt_make_contig(1)};

    ucs_status_ptr_t status =
      ucp_tag_msg_recv_nbx(_handle, buf.data(), info.length, message, &param);

    if (UCS_PTR_IS_PTR(status)) {
      while (ucp_request_check_status(status) == UCS_INPROGRESS)
        progress();

      ucp_request_free(status);
    } else if (UCS_PTR_IS_ERR(status)) {
      ucxx_debug("ucxx::Worker::%s failed to drain tag receive: %s",
                 __func__,
                 ucs_status_string(UCS_PTR_STATUS(status)));
    }
  }
}

std::shared_ptr<RequestAm> Worker::getAmRecv(
  ucp_ep_h ep, std::function<std::shared_ptr<RequestAm>()> createAmRecvRequestFunction)
{
  std::lock_guard<std::mutex> lock(_amData->_mutex);

  auto& recvPool = _amData->_recvPool;
  auto& recvWait = _amData->_recvWait;

  auto reqs = recvPool.find(ep);
  if (reqs != recvPool.end() && !reqs->second.empty()) {
    auto req = reqs->second.front();
    reqs->second.pop();
    if (reqs->second.empty()) recvPool.erase(reqs);
    return req;
  } else {
    auto req        = createAmRecvRequestFunction();
    auto [queue, _] = recvWait.try_emplace(ep, std::queue<std::shared_ptr<RequestAm>>());
    queue->second.push(req);
    return req;
  }
}

std::shared_ptr<Worker> detail::ConstructorFactory::createWorker(std::shared_ptr<Context> context,
                                                                 const bool enableDelayedSubmission,
                                                                 const bool enableFuture)
{
  auto worker = std::shared_ptr<Worker>(new Worker(context, enableDelayedSubmission, enableFuture));
  // We can only get a `shared_ptr<Worker>` for the Active Messages callback after it's
  // been created, thus this cannot be in the constructor.
  if (worker->_amData != nullptr) {
    worker->_amData->_worker = worker;

    std::stringstream ownerStream;
    ownerStream << "worker " << worker->getHandle();
    worker->_amData->_ownerString = ownerStream.str();
  }

  return worker;
}

Worker::~Worker()
{
  size_t canceled = cancelInflightRequests(3000000000 /* 3s */, 3);
  ucxx_debug("ucxx::Worker::%s, Worker: %p, UCP handle: %p, canceled %lu requests",
             __func__,
             this,
             _handle,
             canceled);

  if (isProgressThreadRunning()) {
    ucxx_warn(
      "The progress thread should be explicitly stopped with `stopProgressThread()` to prevent "
      "unintended effects, such as destructors being called from that thread.");
    stopProgressThreadNoWarn();
  }
  if (_notifier && _notifier->isRunning()) {
    ucxx_warn(
      "The notifier thread should be explicitly stopped with `stopNotifierThread()` to prevent "
      "unintended effects, such as destructors being called from that thread.");
    _notifier->stopRequestNotifierThread();
  }

  drainWorkerTagRecv();

  ucp_worker_destroy(_handle);
  ucxx_trace("Worker destroyed: %p, UCP handle: %p", this, _handle);

  if (_epollFileDescriptor >= 0) close(_epollFileDescriptor);
}

ucp_worker_h Worker::getHandle() { return _handle; }

std::string Worker::getInfo()
{
  FILE* TextFileDescriptor = utils::createTextFileDescriptor();
  ucp_worker_print_info(this->_handle, TextFileDescriptor);
  return utils::decodeTextFileDescriptor(TextFileDescriptor);
}

Worker::Attributes Worker::queryAttributes() const
{
  ucp_worker_attr_t attr = {
    .field_mask = UCP_WORKER_ATTR_FIELD_THREAD_MODE | UCP_WORKER_ATTR_FIELD_MAX_AM_HEADER |
                  UCP_WORKER_ATTR_FIELD_NAME | UCP_WORKER_ATTR_FIELD_MAX_INFO_STRING};

  utils::ucsErrorThrow(ucp_worker_query(_handle, &attr));

  return Attributes{
    .threadMode     = attr.thread_mode,
    .maxAmHeader    = attr.max_am_header,
    .name           = std::string(attr.name, ::strnlen(attr.name, sizeof(attr.name))),
    .maxDebugString = attr.max_debug_string,
  };
}

bool Worker::isDelayedRequestSubmissionEnabled() const
{
  return _delayedSubmissionCollection->isDelayedRequestSubmissionEnabled();
}

bool Worker::isFutureEnabled() const { return _enableFuture; }

bool Worker::isRequestAttributesEnabled() const noexcept { return _enableRequestAttributes; }

BufferType Worker::getCudaBufferType() const { return _cudaBufferType; }

void Worker::setCudaBufferType(BufferType bufferType)
{
  if (bufferType != BufferType::CCCL)
    throw std::invalid_argument("cudaBufferType must be BufferType::CCCL");
  _cudaBufferType = bufferType;
}

void Worker::initBlockingProgressMode()
{
  // In blocking progress mode, we create an epoll file
  // descriptor that we can wait on later.
  // We also introduce an additional eventfd to allow
  // canceling the wait.
  int err;

  // Return if blocking progress mode was already initialized
  if (_epollFileDescriptor >= 0) return;

  utils::ucsErrorThrow(ucp_worker_get_efd(_handle, &_workerFileDescriptor));

  arm();

  _epollFileDescriptor = epoll_create(1);
  if (_epollFileDescriptor == -1) throw std::ios_base::failure("epoll_create(1) returned -1");

  epoll_event workerEvent = {.events = EPOLLIN,
                             .data   = {
                                 .fd = _workerFileDescriptor,
                             }};

  err = epoll_ctl(_epollFileDescriptor, EPOLL_CTL_ADD, _workerFileDescriptor, &workerEvent);
  if (err != 0) {
    throw std::ios_base::failure(std::string("epoll_ctl() returned ") + std::to_string(err));
  }
}

int Worker::getEpollFileDescriptor()
{
  if (_epollFileDescriptor == 0)
    throw std::runtime_error("Worker not running in blocking progress mode");

  return _epollFileDescriptor;
}

bool Worker::arm()
{
  ucs_status_t status = ucp_worker_arm(_handle);
  if (status == UCS_ERR_BUSY) return false;
  utils::ucsErrorThrow(status);
  return true;
}

bool Worker::progressWorkerEvent(const int epollTimeout)
{
  int ret;
  epoll_event ev;

  if (progress()) return true;

  if ((_epollFileDescriptor == -1) || !arm()) return false;

  do {
    ret = epoll_wait(_epollFileDescriptor, &ev, 1, epollTimeout);
  } while ((ret == -1) && (errno == EINTR || errno == EAGAIN));

  return false;
}

void Worker::signal() { utils::ucsErrorThrow(ucp_worker_signal(_handle)); }

bool Worker::waitProgress()
{
  utils::ucsErrorThrow(ucp_worker_wait(_handle));
  return progress();
}

bool Worker::progressOnce() { return ucp_worker_progress(_handle) != 0; }

bool Worker::progressPending()
{
  bool ret = false, prog = false;
  do {
    prog = progressOnce();
    ret |= prog;
  } while (prog);
  return ret;
}

bool Worker::progress()
{
  bool ret                     = progressPending();
  bool progressScheduledCancel = false;

  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);

    // Before canceling requests scheduled for cancelation, attempt to let them complete.
    progressScheduledCancel =
      _inflightRequestsToCancel != nullptr && _inflightRequestsToCancel->size() > 0;
  }
  if (progressScheduledCancel) ret |= progressPending();

  // Requests that were not completed now must be canceled.
  if (cancelInflightRequests(3000000000 /* 3s */, 3) > 0) ret |= progressPending();

  return ret;
}

void Worker::registerDelayedSubmission(std::shared_ptr<Request> request,
                                       DelayedSubmissionCallbackType callback)
{
  if (_delayedSubmissionCollection->isDelayedRequestSubmissionEnabled()) {
    _delayedSubmissionCollection->registerRequest(request, callback);

    /* Waking the progress event is needed here because the UCX request is
     * not dispatched immediately. Thus we must signal the progress task so
     * it will ensure the request is dispatched.
     */
    signal();
  } else {
    callback();
  }
}

bool Worker::registerGenericPre(DelayedSubmissionCallbackType callback, uint64_t period)
{
  if (std::this_thread::get_id() == getProgressThreadId()) {
    /**
     * If the method is called from within the progress thread (e.g., from the
     * listener callback), execute it immediately.
     */
    callback();

    return true;
  } else {
    utils::CallbackNotifier callbackNotifier{};
    auto notifiableCallback = [&callback, &callbackNotifier]() {
      callback();
      callbackNotifier.set();
    };

    auto id = _delayedSubmissionCollection->registerGenericPre(notifiableCallback);

    /* Waking the progress event is needed here because the UCX request is
     * not dispatched immediately. Thus we must signal the progress task so
     * it will ensure the request is dispatched.
     */
    std::function<void()> signalWorkerFunction = []() {};
    if (isProgressThreadRunning() && !_progressThreadPollingMode.load(std::memory_order_acquire)) {
      signalWorkerFunction = [this]() { return this->signal(); };
    }
    signalWorkerFunction();

    size_t retryCount = 0;
    while (true) {
      auto ret = callbackNotifier.wait(period, signalWorkerFunction);

      try {
        if (!ret) _delayedSubmissionCollection->cancelGenericPre(id);
        return ret;
      } catch (const std::runtime_error& e) {
        if (++retryCount % 10 == 0)
          ucxx_warn(
            "Could not cancel after %lu attempts, the callback has not returned and the process "
            "may stop responding.",
            retryCount);
      }
    }
  }
}

bool Worker::registerGenericPost(DelayedSubmissionCallbackType callback, uint64_t period)
{
  if (std::this_thread::get_id() == getProgressThreadId()) {
    /**
     * If the method is called from within the progress thread (e.g., from the
     * listener callback), execute it immediately.
     */
    callback();

    return true;
  } else {
    utils::CallbackNotifier callbackNotifier{};
    auto notifiableCallback = [&callback, &callbackNotifier]() {
      callback();
      callbackNotifier.set();
    };

    auto id = _delayedSubmissionCollection->registerGenericPost(notifiableCallback);

    /* Waking the progress event is needed here because the UCX request is
     * not dispatched immediately. Thus we must signal the progress task so
     * it will ensure the request is dispatched.
     */
    std::function<void()> signalWorkerFunction = []() {};
    if (isProgressThreadRunning() && !_progressThreadPollingMode.load(std::memory_order_acquire)) {
      signalWorkerFunction = [this]() { return this->signal(); };
    }
    signalWorkerFunction();

    size_t retryCount = 0;
    while (true) {
      auto ret = callbackNotifier.wait(period, signalWorkerFunction);

      try {
        if (!ret) _delayedSubmissionCollection->cancelGenericPost(id);
        return ret;
      } catch (const std::runtime_error& e) {
        if (++retryCount % 10 == 0)
          ucxx_warn(
            "Could not cancel after %lu attempts, the callback has not returned and the process "
            "may stop responding.",
            retryCount);
      }
    }
  }
}

#define THROW_FUTURE_NOT_IMPLEMENTED()                                                      \
  do {                                                                                      \
    throw std::runtime_error(                                                               \
      "ucxx::Worker's future support not implemented, please ensure you use an "            \
      "implementation with future support and that enableFuture=true is set when creating " \
      "the Worker to use this method.");                                                    \
  } while (0)

void Worker::populateFuturesPool() { THROW_FUTURE_NOT_IMPLEMENTED(); }

void Worker::clearFuturesPool() { THROW_FUTURE_NOT_IMPLEMENTED(); }

std::shared_ptr<Future> Worker::getFuture() { THROW_FUTURE_NOT_IMPLEMENTED(); }

RequestNotifierWaitState Worker::waitRequestNotifier(uint64_t) { THROW_FUTURE_NOT_IMPLEMENTED(); }

void Worker::runRequestNotifier() { THROW_FUTURE_NOT_IMPLEMENTED(); }

void Worker::stopRequestNotifierThread() { THROW_FUTURE_NOT_IMPLEMENTED(); }

void Worker::setProgressThreadStartCallback(std::function<void(void*)> callback, void* callbackArg)
{
  _progressThreadStartCallback    = callback;
  _progressThreadStartCallbackArg = callbackArg;
}

void Worker::startProgressThread(const bool pollingMode, const int epollTimeout)
{
  if (isProgressThreadRunning()) {
    ucxx_debug(
      "ucxx::Worker::%s, Worker: %p, UCP handle: %p, worker progress thread "
      "already running",
      __func__,
      this,
      _handle);
    return;
  }

  std::function<bool()> progressFunction;
  std::function<void()> signalWorkerFunction;
  if (pollingMode) {
    progressFunction     = [this]() { return this->progress(); };
    signalWorkerFunction = []() {};
  } else {
    initBlockingProgressMode();
    progressFunction = [this, epollTimeout]() { return this->progressWorkerEvent(epollTimeout); };
    signalWorkerFunction = [this]() { return this->signal(); };
  }

  _progressThreadId.store(std::thread::id(), std::memory_order_release);
  _progressThreadPollingMode.store(pollingMode, std::memory_order_release);

  auto setThreadId = [this]() {
    _progressThreadId.store(std::this_thread::get_id(), std::memory_order_release);
  };

  _progressThread = WorkerProgressThread(pollingMode,
                                         progressFunction,
                                         signalWorkerFunction,
                                         setThreadId,
                                         _progressThreadStartCallback,
                                         _progressThreadStartCallbackArg,
                                         _delayedSubmissionCollection);

  _progressThreadRunning.store(true, std::memory_order_release);
}

void Worker::stopProgressThreadNoWarn()
{
  _progressThread.stop();
  _progressThreadRunning.store(false, std::memory_order_release);
  _progressThreadId.store(std::thread::id(), std::memory_order_release);
}

void Worker::stopProgressThread()
{
  if (!isProgressThreadRunning())
    ucxx_debug(
      "ucxx::Worker::%s, Worker: %p, UCP handle: %p, worker progress thread not "
      "running or already stopped",
      __func__,
      this,
      _handle);
  else
    stopProgressThreadNoWarn();
}

bool Worker::isProgressThreadRunning()
{
  return _progressThreadRunning.load(std::memory_order_acquire);
}

std::thread::id Worker::getProgressThreadId()
{
  return _progressThreadId.load(std::memory_order_acquire);
}

size_t Worker::cancelInflightRequests(uint64_t period, uint64_t maxAttempts)
{
  size_t canceled = 0;

  auto inflightRequestsToCancel = std::make_unique<InflightRequests>();
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    std::swap(_inflightRequestsToCancel, inflightRequestsToCancel);
  }

  if (std::this_thread::get_id() == getProgressThreadId()) {
    canceled = inflightRequestsToCancel->cancelAll();
    for (uint64_t i = 0; i < maxAttempts && inflightRequestsToCancel->getCancelingSize() > 0; ++i)
      progressPending();
  } else if (isProgressThreadRunning()) {
    bool cancelSuccess = false;
    for (uint64_t i = 0; i < maxAttempts && !cancelSuccess; ++i) {
      if (!registerGenericPre(
            [&canceled, &inflightRequestsToCancel]() {
              canceled += inflightRequestsToCancel->cancelAll();
            },
            period))
        continue;

      if (!registerGenericPost(
            [this, &inflightRequestsToCancel, &cancelSuccess]() {
              cancelSuccess = inflightRequestsToCancel->getCancelingSize() == 0;
            },
            period))
        continue;
    }

    if (!cancelSuccess)
      ucxx_debug(
        "ucxx::Worker::%s, Worker: %p, UCP handle: %p, all attempts to cancel "
        "inflight requests failed",
        __func__,
        this,
        _handle);
  } else {
    canceled = inflightRequestsToCancel->cancelAll();
  }

  if (inflightRequestsToCancel->getCancelingSize() > 0) {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    _inflightRequestsToCancel->merge(inflightRequestsToCancel->release());
  }

  return canceled;
}

void Worker::scheduleRequestCancel(TrackedRequests trackedRequests)
{
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    ucxx_debug(
      "ucxx::Worker::%s, Worker: %p, UCP handle: %p, scheduling cancelation of "
      "%lu requests",
      __func__,
      this,
      _handle,
      trackedRequests.inflight.size() + trackedRequests.canceling.size());
    _inflightRequestsToCancel->merge(std::move(trackedRequests));
  }
}

std::shared_ptr<Request> Worker::registerInflightRequest(std::shared_ptr<Request> request)
{
  if (!request->isCompleted()) {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    _inflightRequests->insert(request);
  }

  return request;
}

void Worker::removeInflightRequest(std::shared_ptr<Request> request)
{
  {
    std::lock_guard<std::mutex> lock(_inflightRequestsMutex);
    _inflightRequests->remove(request);
  }
}

std::shared_ptr<TagProbeInfo> Worker::tagProbe(const Tag tag,
                                               const TagMask tagMask,
                                               const bool remove) const
{
  ucp_tag_recv_info_t info;
  ucp_tag_message_h tag_message = ucp_tag_probe_nb(_handle, tag, tagMask, remove ? 1 : 0, &info);

  if (tag_message != NULL) {
    return detail::ConstructorFactory::createTagProbeInfo(info, remove ? tag_message : nullptr);
  } else {
    return detail::ConstructorFactory::createTagProbeInfo();
  }
}

RequestTagBuilder Worker::tagRecvBuilder(void* buffer, size_t length, Tag tag, TagMask tagMask)
{
  auto worker = std::static_pointer_cast<Worker>(shared_from_this());
  return RequestTagBuilder(std::move(worker), data::TagReceive(buffer, length, tag, tagMask));
}

RequestTagBuilder Worker::tagRecvWithHandleBuilder(void* buffer,
                                                   std::shared_ptr<TagProbeInfo> probeInfo)
{
  if (!probeInfo->isMatched()) { throw std::invalid_argument("TagProbeInfo must be matched"); }
  const auto length = probeInfo->getInfo().length;
  return tagRecvWithHandleBuilder(buffer, length, std::move(probeInfo));
}

RequestTagBuilder Worker::tagRecvWithHandleBuilder(void* buffer,
                                                   size_t length,
                                                   std::shared_ptr<TagProbeInfo> probeInfo)
{
  if (!probeInfo->isMatched()) { throw std::invalid_argument("TagProbeInfo must be matched"); }

  // getHandle() will throw runtime_error if handle is nullptr or consumed
  try {
    probeInfo->getHandle();
  } catch (const std::runtime_error& e) {
    throw std::logic_error(std::string("TagProbeInfo handle validation failed: ") + e.what());
  }

  auto worker = std::static_pointer_cast<Worker>(shared_from_this());
  return RequestTagBuilder(std::move(worker),
                           data::TagReceiveWithHandle(buffer, length, probeInfo));
}

AddressBuilder Worker::addressBuilder()
{
  return AddressBuilder(std::static_pointer_cast<Worker>(shared_from_this()));
}

EndpointBuilder Worker::endpointBuilder(std::string ipAddress, uint16_t port)
{
  return EndpointBuilder(
    std::static_pointer_cast<Worker>(shared_from_this()), std::move(ipAddress), port);
}

EndpointBuilder Worker::endpointBuilder(std::shared_ptr<Address> address)
{
  return EndpointBuilder(std::static_pointer_cast<Worker>(shared_from_this()), std::move(address));
}

ListenerBuilder Worker::listenerBuilder(uint16_t port,
                                        ucp_listener_conn_callback_t callback,
                                        void* callbackArgs)
{
  return ListenerBuilder(
    std::static_pointer_cast<Worker>(shared_from_this()), port, callback, callbackArgs);
}

void Worker::registerAmAllocator(ucs_memory_type_t memoryType, AmAllocatorType allocator)
{
  if (_amData == nullptr)
    throw std::runtime_error("Active Messages was not enabled during context creation");
  _amData->_allocators.insert_or_assign(memoryType, allocator);
}

void Worker::registerAmReceiverCallback(AmReceiverCallbackInfo info,
                                        AmReceiverCallbackType callback)
{
  if (info.owner == "ucxx") throw std::runtime_error("The owner name 'ucxx' is reserved.");
  if (_amData->_receiverCallbacks.find(info.owner) == _amData->_receiverCallbacks.end())
    _amData->_receiverCallbacks[info.owner] = {};
  if (_amData->_receiverCallbacks[info.owner].find(info.id) !=
      _amData->_receiverCallbacks[info.owner].end())
    throw std::runtime_error("Callback with given owner and identifier is already registered");

  _amData->_receiverCallbacks[info.owner][info.id] = callback;
}

bool Worker::amProbe(const ucp_ep_h endpointHandle) const
{
  std::lock_guard<std::mutex> lock(_amData->_mutex);
  return _amData->_recvPool.find(endpointHandle) != _amData->_recvPool.end();
}

RequestFlushBuilder Worker::flushBuilder()
{
  auto worker = std::static_pointer_cast<Worker>(shared_from_this());
  return RequestFlushBuilder(std::move(worker), data::Flush());
}

}  // namespace ucxx
