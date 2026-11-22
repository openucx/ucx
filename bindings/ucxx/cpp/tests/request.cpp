/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <condition_variable>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <ucp/api/ucp.h>
#include <ucs/type/status.h>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ucxx/api.h>

#include "include/utils.h"
#include "ucxx/buffer.h"
#include "ucxx/utils/ucx.h"

#ifndef UCXX_TESTS_ENABLE_RMM
#define UCXX_TESTS_ENABLE_RMM 0
#endif

#if UCXX_TESTS_ENABLE_RMM
#include <cuda/stream>
#include <rmm/device_buffer.hpp>
#endif

#if UCXX_ENABLE_CCCL
#include <cuda_runtime.h>
#endif

namespace {

using ::testing::Combine;
using ::testing::ContainerEq;
using ::testing::Values;

typedef std::vector<int> DataContainerType;

struct ProgressThreadStartBarrier {
  std::mutex mutex{};
  std::condition_variable condition{};
  bool started{false};
  bool released{false};
};

void waitForProgressThreadRelease(void* arg)
{
  auto& barrier = *static_cast<ProgressThreadStartBarrier*>(arg);
  std::unique_lock<std::mutex> lock(barrier.mutex);
  barrier.started = true;
  barrier.condition.notify_one();
  barrier.condition.wait(lock, [&barrier]() { return barrier.released; });
}

enum class TestBufferType {
  Host,
  RMM,
  CCCL,
};

bool isCudaBufferType(TestBufferType bufferType) { return bufferType != TestBufferType::Host; }

#if UCXX_TESTS_ENABLE_RMM
class RMMTestBuffer : public ucxx::Buffer {
 private:
  std::unique_ptr<rmm::device_buffer> _buffer;

 public:
  explicit RMMTestBuffer(size_t size)
    : ucxx::Buffer(ucxx::BufferType::Invalid, size),
      _buffer{std::make_unique<rmm::device_buffer>(
        size, cuda::stream_ref{cudaStream_t{cudaStreamDefault}})}
  {
  }

  void* data() override
  {
    if (!_buffer) throw std::runtime_error("Invalid object");
    return _buffer->data();
  }
};
#endif

class RequestTest
  : public ::testing::TestWithParam<std::tuple<TestBufferType, bool, bool, ProgressMode, size_t>> {
 protected:
  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};
  std::function<void()> _progressWorker;

  TestBufferType _bufferType;
  ucs_memory_type_t _memoryType;
  bool _registerCustomAmAllocator;
  bool _enableDelayedSubmission;
  ProgressMode _progressMode;
  size_t _messageLength;
  size_t _messageSize;
  size_t _rndvThresh{8192};

  size_t _numBuffers{0};
  std::vector<DataContainerType> _send;
  std::vector<DataContainerType> _recv;
  std::vector<std::unique_ptr<ucxx::Buffer>> _sendBuffer;
  std::vector<std::unique_ptr<ucxx::Buffer>> _recvBuffer;
  std::vector<const void*> _sendPtr{nullptr};
  std::vector<void*> _recvPtr{nullptr};

  void buildWorker(bool enableRequestAttributes)
  {
    auto builder = ucxx::workerBuilder(_context)
                     .delayedSubmission(_enableDelayedSubmission)
                     .requestAttributes(enableRequestAttributes);

#if UCXX_ENABLE_CCCL
    if (isCudaBufferType(_bufferType)) builder.cudaBufferType(ucxx::BufferType::CCCL);
#endif

    _worker = builder.build();

    if (_progressMode == ProgressMode::Blocking) {
      _worker->initBlockingProgressMode();
    } else if (_progressMode == ProgressMode::ThreadPolling ||
               _progressMode == ProgressMode::ThreadBlocking) {
      _worker->setProgressThreadStartCallback(::createCudaContextCallback, nullptr);

      if (_progressMode == ProgressMode::ThreadPolling) _worker->startProgressThread(true);
      if (_progressMode == ProgressMode::ThreadBlocking) _worker->startProgressThread(false);
    }

    _progressWorker = getProgressFunction(_worker, _progressMode);

    _ep = _worker->endpointBuilder(_worker->addressBuilder().build()).build();
  }

  void rebuildWorker(bool enableRequestAttributes)
  {
    if (_worker && _worker->isProgressThreadRunning()) _worker->stopProgressThread();
    _ep.reset();
    _worker.reset();
    buildWorker(enableRequestAttributes);
  }

  void SetUp()
  {
    std::tie(_bufferType,
             _registerCustomAmAllocator,
             _enableDelayedSubmission,
             _progressMode,
             _messageLength) = GetParam();

    if (_bufferType == TestBufferType::RMM) {
#if !UCXX_TESTS_ENABLE_RMM
      GTEST_SKIP() << "UCXX tests were not built with RMM support";
#endif
    }

    if (_bufferType == TestBufferType::CCCL) {
#if !UCXX_ENABLE_CCCL
      GTEST_SKIP() << "UCXX was not built with CCCL support";
#endif
    }

    _memoryType  = isCudaBufferType(_bufferType) ? UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
    _messageSize = _messageLength * sizeof(int);

    _context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags)
                 .configMap({{"RNDV_THRESH", std::to_string(_rndvThresh)}})
                 .build();
    buildWorker(false);
  }

  void TearDown()
  {
    if (_progressMode == ProgressMode::ThreadPolling ||
        _progressMode == ProgressMode::ThreadBlocking)
      _worker->stopProgressThread();
  }

  void allocate(const size_t numBuffers = 1, const bool allocateRecvBuffer = true)
  {
    _numBuffers = numBuffers;

    _send.resize(_numBuffers);
    _recv.resize(_numBuffers);
    _sendBuffer.resize(_numBuffers);
    _sendPtr.resize(_numBuffers);
    if (allocateRecvBuffer) {
      _recvBuffer.resize(_numBuffers);
      _recvPtr.resize(_numBuffers);
    }

    for (size_t i = 0; i < _numBuffers; ++i) {
      _send[i].resize(_messageLength);
      _recv[i].resize(_messageLength);

      std::iota(_send[i].begin(), _send[i].end(), i);

      if (_bufferType == TestBufferType::Host) {
        _sendBuffer[i] = std::make_unique<ucxx::HostBuffer>(_messageSize);
        if (allocateRecvBuffer) _recvBuffer[i] = std::make_unique<ucxx::HostBuffer>(_messageSize);
#if UCXX_TESTS_ENABLE_RMM
      } else if (_bufferType == TestBufferType::RMM) {
        _sendBuffer[i] = std::make_unique<RMMTestBuffer>(_messageSize);
        if (allocateRecvBuffer) _recvBuffer[i] = std::make_unique<RMMTestBuffer>(_messageSize);
#endif
#if UCXX_ENABLE_CCCL
      } else if (_bufferType == TestBufferType::CCCL) {
        _sendBuffer[i] = std::make_unique<ucxx::CCCLBuffer>(_messageSize);
        if (allocateRecvBuffer) _recvBuffer[i] = std::make_unique<ucxx::CCCLBuffer>(_messageSize);
#endif
      }

      copyMemoryTypeAware(_sendBuffer[i]->data(), _send[i].data(), _messageSize, false);

      _sendPtr[i] = _sendBuffer[i]->data();
      if (allocateRecvBuffer) _recvPtr[i] = _recvBuffer[i]->data();
    }
#if UCXX_TESTS_ENABLE_RMM
    if (_bufferType == TestBufferType::RMM) {
      auto stream = cuda::stream_ref{cudaStream_t { cudaStreamDefault }};
      stream.sync();
    }
#endif
    if (_bufferType == TestBufferType::CCCL) { cudaStreamSynchronize(nullptr); }
  }

  void copyResults()
  {
    for (size_t i = 0; i < _numBuffers; ++i)
      copyMemoryTypeAware(_recv[i].data(), _recvPtr[i], _messageSize, false);
#if UCXX_TESTS_ENABLE_RMM
    if (_bufferType == TestBufferType::RMM) {
      auto stream = cuda::stream_ref{cudaStream_t { cudaStreamDefault }};
      stream.sync();
    }
#endif
    if (_bufferType == TestBufferType::CCCL) { cudaStreamSynchronize(nullptr); }
  }

  void copyMemoryTypeAware(void* dst, const void* src, size_t size, bool synchronize = true)
  {
    if (_memoryType == UCS_MEMORY_TYPE_HOST) {
      memcpy(dst, src, size);
#if UCXX_TESTS_ENABLE_RMM
    } else if (_memoryType == UCS_MEMORY_TYPE_CUDA && _bufferType == TestBufferType::RMM) {
      auto stream = cuda::stream_ref{cudaStream_t { cudaStreamDefault }};
      RMM_CUDA_TRY(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream.get()));
      if (synchronize) { stream.sync(); }
#endif
    } else if (_memoryType == UCS_MEMORY_TYPE_CUDA) {
      cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, nullptr);
      if (synchronize) cudaStreamSynchronize(nullptr);
    }
  }
};

TEST(RequestCancellationTest, CancelBeforeDelayedSubmission)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto worker  = ucxx::workerBuilder(context).delayedSubmission(true).build();
  auto ep      = worker->endpointBuilder(worker->addressBuilder().build()).build();

  ProgressThreadStartBarrier barrier{};
  worker->setProgressThreadStartCallback(waitForProgressThreadRelease, &barrier);
  worker->startProgressThread(true);

  {
    std::unique_lock<std::mutex> lock(barrier.mutex);
    barrier.condition.wait(lock, [&barrier]() { return barrier.started; });
  }

  int recvBuffer{};
  auto request =
    ep->tagRecvBuilder(&recvBuffer, sizeof(recvBuffer), ucxx::Tag{0}, ucxx::TagMaskFull).build();
  request->cancel();

  const bool completedBeforeSubmission = request->isCompleted();
  const auto statusBeforeSubmission    = request->getStatus();

  {
    std::lock_guard<std::mutex> lock(barrier.mutex);
    barrier.released = true;
  }
  barrier.condition.notify_one();

  // Wait until the delayed-submission iteration has completed before cleaning up.
  EXPECT_TRUE(worker->registerGenericPost([]() {}));
  if (!request->isCompleted()) request->cancel();
  while (!request->isCompleted())
    std::this_thread::yield();
  worker->stopProgressThread();

  EXPECT_TRUE(completedBeforeSubmission);
  EXPECT_EQ(statusBeforeSubmission, UCS_ERR_CANCELED);
}

TEST_P(RequestTest, ProgressAm)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_registerCustomAmAllocator && _memoryType == UCS_MEMORY_TYPE_CUDA) {
#if UCXX_ENABLE_CCCL
    _worker->registerAmAllocator(UCS_MEMORY_TYPE_CUDA, [](size_t length) {
      return std::make_shared<ucxx::CCCLBuffer>(length);
    });
#else
    GTEST_SKIP() << "CCCL support is required for CUDA receive allocations";
#endif
  }

  allocate(1, false);

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSendBuilder(_sendPtr[0], _messageSize, _memoryType).build());
  requests.push_back(_ep->amRecvBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq = requests[1];
  _recvPtr[0]  = recvReq->getRecvBuffer()->data();

  // Messages of size `_rndvThresh` or larger are rendezvous and will use the custom
  // allocator, smaller messages are eager and will always be host-allocated.
  const auto expectedRecvBufferType = (_registerCustomAmAllocator && _messageSize >= _rndvThresh)
                                        ? ucxx::BufferType::CCCL
                                        : ucxx::BufferType::Host;
  ASSERT_THAT(recvReq->getRecvBuffer()->getType(), expectedRecvBufferType);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressAmIovHost)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_memoryType != UCS_MEMORY_TYPE_HOST) {
    GTEST_SKIP() << "IOV test uses host buffers for deterministic validation";
  }

  const size_t messageLength = std::max<size_t>(4, _messageLength);
  std::vector<int> send(messageLength);
  std::iota(send.begin(), send.end(), 0);

  const size_t firstSegmentLength  = messageLength / 2;
  const size_t secondSegmentLength = messageLength - firstSegmentLength;
  std::vector<ucp_dt_iov_t> iov(2);
  iov[0].buffer = send.data();
  iov[0].length = firstSegmentLength * sizeof(int);
  iov[1].buffer = send.data() + firstSegmentLength;
  iov[1].length = secondSegmentLength * sizeof(int);

  auto amSendParams       = ucxx::AmSendParams{};
  amSendParams.datatype   = UCP_DATATYPE_IOV;
  amSendParams.memoryType = UCS_MEMORY_TYPE_HOST;

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSendBuilder(iov, amSendParams).build());
  requests.push_back(_ep->amRecvBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq    = requests[1];
  auto recvBuffer = recvReq->getRecvBuffer();
  ASSERT_EQ(recvBuffer->getType(), ucxx::BufferType::Host);
  ASSERT_EQ(recvBuffer->getSize(), messageLength * sizeof(int));

  std::vector<int> recv(reinterpret_cast<int*>(recvBuffer->data()),
                        reinterpret_cast<int*>(recvBuffer->data()) + messageLength);
  ASSERT_THAT(recv, ContainerEq(send));
}

TEST_P(RequestTest, ProgressAmIovValidation)
{
  auto amSendParams       = ucxx::AmSendParams{};
  amSendParams.datatype   = UCP_DATATYPE_IOV;
  amSendParams.memoryType = UCS_MEMORY_TYPE_HOST;

  EXPECT_THROW(std::ignore = _ep->amSendBuilder(std::vector<ucp_dt_iov_t>{}, amSendParams).build(),
               std::runtime_error);

  std::vector<ucp_dt_iov_t> iovWithNullBuffer(1);
  iovWithNullBuffer[0].buffer = nullptr;
  iovWithNullBuffer[0].length = 16;
  EXPECT_THROW(std::ignore = _ep->amSendBuilder(iovWithNullBuffer, amSendParams).build(),
               std::runtime_error);

  std::vector<int> send{1, 2, 3, 4};
  std::vector<ucp_dt_iov_t> validIov(1);
  validIov[0].buffer = send.data();
  validIov[0].length = send.size() * sizeof(send[0]);

  auto wrongDatatypeParams     = amSendParams;
  wrongDatatypeParams.datatype = ucp_dt_make_contig(1);
  EXPECT_THROW(std::ignore = _ep->amSendBuilder(validIov, wrongDatatypeParams).build(),
               std::runtime_error);
}

TEST_P(RequestTest, ProgressAmMemoryTypePolicyStrict)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  const size_t bytes = std::max(_rndvThresh + 128, sizeof(int));
  std::vector<uint8_t> send(bytes, 42);

  auto amSendParams             = ucxx::AmSendParams{};
  amSendParams.memoryType       = UCS_MEMORY_TYPE_CUDA;
  amSendParams.memoryTypePolicy = ucxx::AmSendMemoryTypePolicy::ErrorOnUnsupported;

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSendBuilder(send.data(), send.size(), amSendParams).build());
  requests.push_back(_ep->amRecvBuilder().build());

  // Wait for completion without calling checkError(), since the receive request
  // is expected to complete with UCS_ERR_UNSUPPORTED.
  while (!requests[0]->isCompleted() || !requests[1]->isCompleted())
    _progressWorker();

  // When the receiver rejects a rendezvous transfer, UCX propagates the error to
  // both sides, so the send may also complete with UCS_ERR_UNSUPPORTED.
  ASSERT_EQ(requests[1]->getStatus(), UCS_ERR_UNSUPPORTED);
}

TEST_P(RequestTest, ProgressAmReceiverCallback)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_registerCustomAmAllocator && _memoryType == UCS_MEMORY_TYPE_CUDA) {
#if UCXX_ENABLE_CCCL
    _worker->registerAmAllocator(UCS_MEMORY_TYPE_CUDA, [](size_t length) {
      return std::make_shared<ucxx::CCCLBuffer>(length);
    });
#else
    GTEST_SKIP() << "CCCL support is required for CUDA receive allocations";
#endif
  }

  // Define AM receiver callback's owner and id for callback
  ucxx::AmReceiverCallbackInfo receiverCallbackInfo("TestApp", 0);

  // Mutex required for blocking progress mode, otherwise `receivedRequests` may be
  // accessed before `push_back()` completed.
  std::mutex mutex;

  // Define AM receiver callback and register with worker
  std::vector<std::shared_ptr<ucxx::Request>> receivedRequests;
  auto callback = ucxx::AmReceiverCallbackType(
    [this, &receivedRequests, &mutex](std::shared_ptr<ucxx::Request> req, ucp_ep_h) {
      {
        std::lock_guard<std::mutex> lock(mutex);
        receivedRequests.push_back(req);
      }
    });
  _worker->registerAmReceiverCallback(receiverCallbackInfo, callback);

  allocate(1, false);

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSendBuilder(_sendPtr[0], _messageSize, _memoryType)
                       .receiverCallbackInfo(receiverCallbackInfo)
                       .build());
  waitRequests(_worker, requests, _progressWorker);

  while (receivedRequests.size() < 1)
    _progressWorker();

  {
    std::lock_guard<std::mutex> lock(mutex);
    _recvPtr[0] = receivedRequests[0]->getRecvBuffer()->data();

    // Messages larger than `_rndvThresh` are rendezvous and will use custom allocator,
    // smaller messages are eager and will always be host-allocated.
    const auto expectedRecvBufferType = (_registerCustomAmAllocator && _messageSize >= _rndvThresh)
                                          ? ucxx::BufferType::CCCL
                                          : ucxx::BufferType::Host;
    ASSERT_THAT(receivedRequests[0]->getRecvBuffer()->getType(), expectedRecvBufferType);
  }

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressAmUserHeader)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_memoryType != UCS_MEMORY_TYPE_HOST) {
    GTEST_SKIP() << "User header test uses host buffers only";
  }

  allocate(1, false);

  const std::string sentHeader = "test-header-payload-\x00\x01\x02\xff";

  auto amSendParams = ucxx::AmSendParams{};
  amSendParams.setUserHeader(sentHeader);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSendBuilder(_sendPtr[0], _messageSize, amSendParams).build());
  requests.push_back(_ep->amRecvBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq = requests[1];
  ASSERT_EQ(recvReq->getRecvHeader(), sentHeader);

  _recvPtr[0] = recvReq->getRecvBuffer()->data();
  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressAmIovUserHeader)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_memoryType != UCS_MEMORY_TYPE_HOST) {
    GTEST_SKIP() << "IOV user header test uses host buffers only";
  }

  const size_t messageLength = std::max<size_t>(4, _messageLength);
  std::vector<int> send(messageLength);
  std::iota(send.begin(), send.end(), 0);

  const size_t firstSegmentLength  = messageLength / 2;
  const size_t secondSegmentLength = messageLength - firstSegmentLength;
  std::vector<ucp_dt_iov_t> iov(2);
  iov[0].buffer = send.data();
  iov[0].length = firstSegmentLength * sizeof(int);
  iov[1].buffer = send.data() + firstSegmentLength;
  iov[1].length = secondSegmentLength * sizeof(int);

  const std::string sentHeader = "iov-user-header-data";

  auto amSendParams       = ucxx::AmSendParams{};
  amSendParams.datatype   = UCP_DATATYPE_IOV;
  amSendParams.memoryType = UCS_MEMORY_TYPE_HOST;
  amSendParams.setUserHeader(sentHeader);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSendBuilder(iov, amSendParams).build());
  requests.push_back(_ep->amRecvBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq    = requests[1];
  auto recvBuffer = recvReq->getRecvBuffer();
  ASSERT_EQ(recvBuffer->getType(), ucxx::BufferType::Host);
  ASSERT_EQ(recvBuffer->getSize(), messageLength * sizeof(int));
  ASSERT_EQ(recvReq->getRecvHeader(), sentHeader);

  std::vector<int> recv(reinterpret_cast<int*>(recvBuffer->data()),
                        reinterpret_cast<int*>(recvBuffer->data()) + messageLength);
  ASSERT_THAT(recv, ContainerEq(send));
}

TEST_P(RequestTest, ProgressAmEmptyUserHeader)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  if (_memoryType != UCS_MEMORY_TYPE_HOST) {
    GTEST_SKIP() << "User header test uses host buffers only";
  }

  allocate(1, false);

  // Send without user header (default empty)
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSendBuilder(_sendPtr[0], _messageSize, _memoryType).build());
  requests.push_back(_ep->amRecvBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  auto recvReq = requests[1];
  ASSERT_EQ(recvReq->getRecvHeader(), std::string{});

  _recvPtr[0] = recvReq->getRecvBuffer()->data();
  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressStream)
{
  allocate();

  // Submit and wait for transfers to complete
  if (_messageSize == 0) {
    EXPECT_THROW(
      std::ignore = _ep->streamSendBuilder(_sendPtr[0], _messageSize).pythonFuture(0).build(),
      std::runtime_error);
    EXPECT_THROW(
      std::ignore = _ep->streamRecvBuilder(_recvPtr[0], _messageSize).pythonFuture(0).build(),
      std::runtime_error);
  } else {
    std::vector<std::shared_ptr<ucxx::Request>> requests;
    requests.push_back(_ep->streamSendBuilder(_sendPtr[0], _messageSize).pythonFuture(0).build());
    requests.push_back(_ep->streamRecvBuilder(_recvPtr[0], _messageSize).pythonFuture(0).build());
    waitRequests(_worker, requests, _progressWorker);

    copyResults();

    // Assert data correctness
    ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
  }
}

TEST_P(RequestTest, ProgressTag)
{
  allocate();

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->tagSendBuilder(_sendPtr[0], _messageSize, ucxx::Tag{0}).build());
  requests.push_back(
    _ep->tagRecvBuilder(_recvPtr[0], _messageSize, ucxx::Tag{0}, ucxx::TagMaskFull).build());
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressTagRequestAttributes)
{
  if (_messageSize < _rndvThresh)
    GTEST_SKIP() << "Eager messages do not create a ucp_request and thus no debug info";

  rebuildWorker(true);

  allocate();

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->tagSendBuilder(_sendPtr[0], _messageSize, ucxx::Tag{0}).build());
  requests.push_back(
    _ep->tagRecvBuilder(_recvPtr[0], _messageSize, ucxx::Tag{0}, ucxx::TagMaskFull).build());
  waitRequests(_worker, requests, _progressWorker);

  for (const auto& request : requests) {
    auto debugString = request->queryAttributes().debugString;
    ASSERT_THAT(debugString, ::testing::HasSubstr("length " + std::to_string(_messageSize)));
    ASSERT_THAT(debugString,
                ::testing::HasSubstr(_memoryType == UCS_MEMORY_TYPE_HOST ? "host" : "cuda"));
  }

  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

class RequestAttributesDisabledTest : public ::testing::Test {
 protected:
  static constexpr size_t kMessageLength = 1024;
  static constexpr size_t kMessageSize   = kMessageLength * sizeof(int);

  std::shared_ptr<ucxx::Context> _context;
  std::shared_ptr<ucxx::Worker> _worker;
  std::shared_ptr<ucxx::Endpoint> _ep;
  std::function<void()> _progressWorker;
  std::vector<int> _sendBuf;
  std::vector<int> _recvBuf;

  void SetUp() override
  {
    _context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
    _worker  = ucxx::workerBuilder(_context).build();
    ASSERT_FALSE(_worker->isRequestAttributesEnabled());

    _ep             = _worker->endpointBuilder(_worker->addressBuilder().build()).build();
    _progressWorker = getProgressFunction(_worker, ProgressMode::Polling);

    _sendBuf.resize(kMessageLength);
    _recvBuf.resize(kMessageLength);
    std::iota(_sendBuf.begin(), _sendBuf.end(), 0);
  }

  void expectAllThrow(const std::vector<std::shared_ptr<ucxx::Request>>& requests) const
  {
    for (const auto& request : requests) {
      EXPECT_THROW(std::ignore = request->queryAttributes(), ucxx::UnsupportedError);
    }
  }
};

TEST_F(RequestAttributesDisabledTest, Tag)
{
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->tagSendBuilder(_sendBuf.data(), kMessageSize, ucxx::Tag{0}).build());
  requests.push_back(
    _ep->tagRecvBuilder(_recvBuf.data(), kMessageSize, ucxx::Tag{0}, ucxx::TagMaskFull).build());
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow(requests);
  ASSERT_THAT(_recvBuf, ::testing::ContainerEq(_sendBuf));
}

TEST_F(RequestAttributesDisabledTest, Stream)
{
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->streamSendBuilder(_sendBuf.data(), kMessageSize).pythonFuture(0).build());
  requests.push_back(_ep->streamRecvBuilder(_recvBuf.data(), kMessageSize).pythonFuture(0).build());
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow(requests);
  ASSERT_THAT(_recvBuf, ::testing::ContainerEq(_sendBuf));
}

TEST_F(RequestAttributesDisabledTest, Am)
{
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(
    _ep->amSendBuilder(_sendBuf.data(), kMessageSize, UCS_MEMORY_TYPE_HOST).build());
  requests.push_back(_ep->amRecvBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow(requests);

  auto recvBuffer = requests[1]->getRecvBuffer();
  ASSERT_EQ(recvBuffer->getSize(), kMessageSize);
  std::vector<int> received(reinterpret_cast<int*>(recvBuffer->data()),
                            reinterpret_cast<int*>(recvBuffer->data()) + kMessageLength);
  ASSERT_THAT(received, ::testing::ContainerEq(_sendBuf));
}

TEST_F(RequestAttributesDisabledTest, MemoryGet)
{
  auto memoryHandle = _context->memoryHandleBuilder(kMessageSize)
                        .buffer(nullptr)
                        .memoryType(UCS_MEMORY_TYPE_HOST)
                        .build();
  std::memcpy(
    reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _sendBuf.data(), kMessageSize);

  auto serializedRemoteKey = memoryHandle->remoteKeyBuilder().build()->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::shared_ptr<ucxx::Request> request =
    _ep->memGetBuilder(_recvBuf.data(), kMessageSize, remoteKey).build();
  std::vector<std::shared_ptr<ucxx::Request>> requests{request, _ep->flushBuilder().build()};
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow({request});
  ASSERT_THAT(_recvBuf, ::testing::ContainerEq(_sendBuf));
}

TEST_F(RequestAttributesDisabledTest, MemoryPut)
{
  auto memoryHandle = _context->memoryHandleBuilder(kMessageSize)
                        .buffer(nullptr)
                        .memoryType(UCS_MEMORY_TYPE_HOST)
                        .build();

  auto serializedRemoteKey = memoryHandle->remoteKeyBuilder().build()->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::shared_ptr<ucxx::Request> request =
    _ep->memPutBuilder(_sendBuf.data(), kMessageSize, remoteKey).build();
  std::vector<std::shared_ptr<ucxx::Request>> requests{request, _ep->flushBuilder().build()};
  waitRequests(_worker, requests, _progressWorker);

  expectAllThrow({request});

  std::memcpy(
    _recvBuf.data(), reinterpret_cast<void*>(memoryHandle->getBaseAddress()), kMessageSize);
  ASSERT_THAT(_recvBuf, ::testing::ContainerEq(_sendBuf));
}

TEST_P(RequestTest, ProgressStreamRequestAttributes)
{
  if (_messageSize == 0) GTEST_SKIP() << "Stream rejects zero-length transfers";

  rebuildWorker(true);

  allocate();

  std::shared_ptr<ucxx::Request> sendRequest =
    _ep->streamSendBuilder(_sendPtr[0], _messageSize).pythonFuture(0).build();
  std::shared_ptr<ucxx::Request> recvRequest =
    _ep->streamRecvBuilder(_recvPtr[0], _messageSize).pythonFuture(0).build();
  std::vector<std::shared_ptr<ucxx::Request>> requests{sendRequest, recvRequest};
  waitRequests(_worker, requests, _progressWorker);

  try {
    auto sendDebug = sendRequest->queryAttributes().debugString;
    EXPECT_FALSE(sendDebug.empty());
    EXPECT_THAT(sendDebug, ::testing::HasSubstr("length " + std::to_string(_messageSize)));
  } catch (const ucxx::NoElemError&) {
    // Send completed inline; no UCP request handle to query.
  }

  try {
    auto recvDebug = recvRequest->queryAttributes().debugString;
    EXPECT_THAT(recvDebug, ::testing::HasSubstr("no debug info"));
  } catch (const ucxx::NoElemError&) {
    // Recv completed inline; no UCP request handle to query.
  }

  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressAmRequestAttributes)
{
  if (_messageSize < _rndvThresh)
    GTEST_SKIP() << "Eager messages complete inline without a UCP request to query";
  if (_progressMode == ProgressMode::Wait)
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";

  rebuildWorker(true);

  allocate(1, false);

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->amSendBuilder(_sendPtr[0], _messageSize, _memoryType).build());
  requests.push_back(_ep->amRecvBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  for (const auto& request : requests) {
    auto debugString = request->queryAttributes().debugString;
    ASSERT_THAT(debugString, ::testing::HasSubstr("length " + std::to_string(_messageSize)));
  }

  auto recvReq = requests[1];
  _recvPtr[0]  = recvReq->getRecvBuffer()->data();
  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryGetRequestAttributes)
{
  if (_messageSize == 0) GTEST_SKIP() << "Zero-length memGet completes without a UCP request";

  rebuildWorker(true);

  allocate();

  auto memoryHandle =
    _context->memoryHandleBuilder(_messageSize).buffer(nullptr).memoryType(_memoryType).build();
  copyMemoryTypeAware(
    reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _sendPtr[0], _messageSize);

  auto localRemoteKey      = memoryHandle->remoteKeyBuilder().build();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::shared_ptr<ucxx::Request> request =
    _ep->memGetBuilder(_recvPtr[0], _messageSize, remoteKey).build();
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(request);
  requests.push_back(_ep->flushBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  auto debugString = request->queryAttributes().debugString;
  ASSERT_THAT(debugString, ::testing::HasSubstr("length " + std::to_string(_messageSize)));

  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryPutRequestAttributes)
{
  if (_messageSize == 0) GTEST_SKIP() << "Zero-length memPut completes without a UCP request";

  rebuildWorker(true);

  allocate();

  auto memoryHandle =
    _context->memoryHandleBuilder(_messageSize).buffer(nullptr).memoryType(_memoryType).build();

  auto localRemoteKey      = memoryHandle->remoteKeyBuilder().build();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::shared_ptr<ucxx::Request> request =
    _ep->memPutBuilder(_sendPtr[0], _messageSize, remoteKey).build();
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(request);
  requests.push_back(_ep->flushBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  try {
    auto debugString = request->queryAttributes().debugString;
    EXPECT_FALSE(debugString.empty());
    EXPECT_THAT(debugString, ::testing::HasSubstr("length " + std::to_string(_messageSize)));
  } catch (const ucxx::NoElemError&) {
    // Request completed inline; no UCP request handle to query.
  }

  copyMemoryTypeAware(
    _recvPtr[0], reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _messageSize);

  copyResults();
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, ProgressTagMulti)
{
  if (_progressMode == ProgressMode::Wait) {
    GTEST_SKIP() << "Interrupting UCP worker progress operation in wait mode is not possible";
  }

  const size_t numMulti         = 8;
  const bool allocateRecvBuffer = false;

  if (isCudaBufferType(_bufferType) && _worker->getCudaBufferType() == ucxx::BufferType::Invalid) {
    GTEST_SKIP() << "CUDA buffer allocation support not enabled";
  }

  allocate(numMulti, allocateRecvBuffer);

  // Allocate buffers for request sizes/types
  std::vector<size_t> multiSize(numMulti, _messageSize);
  std::vector<int> multiIsCUDA(numMulti, isCudaBufferType(_bufferType));

  // Submit and wait for transfers to complete
  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->tagMultiSendBuilder(_sendPtr, multiSize, multiIsCUDA, ucxx::Tag{0})
                       .pythonFuture(false)
                       .build());
  requests.push_back(
    _ep->tagMultiRecvBuilder(ucxx::Tag{0}, ucxx::TagMaskFull).pythonFuture(false).build());
  waitRequests(_worker, requests, _progressWorker);

  auto recvRequest = requests[1];

  _recvPtr.resize(_numBuffers);
  size_t transferIdx = 0;

  // Populate recv pointers
  for (const auto& br :
       std::dynamic_pointer_cast<ucxx::RequestTagMulti>(requests[1])->_bufferRequests) {
    // br->buffer == nullptr are headers
    if (br->buffer) {
      auto expectedBufferType =
        isCudaBufferType(_bufferType) ? _worker->getCudaBufferType() : ucxx::BufferType::Host;
      ASSERT_EQ(br->buffer->getType(), expectedBufferType);
      ASSERT_EQ(br->buffer->getSize(), _messageSize);

      _recvPtr[transferIdx] = br->buffer->data();

      ++transferIdx;
    }
  }

  copyResults();

  // Assert data correctness
  for (size_t i = 0; i < numMulti; ++i)
    ASSERT_THAT(_recv[i], ContainerEq(_send[i]));
}

TEST_P(RequestTest, TagUserCallback)
{
  allocate();

  std::vector<std::shared_ptr<ucxx::Request>> requests(2);
  std::vector<ucs_status_t> requestStatus(2, UCS_INPROGRESS);

  auto checkStatus = [&requests, &requestStatus](ucs_status_t status,
                                                 ::ucxx::RequestCallbackUserData data) {
    auto idx           = *std::static_pointer_cast<size_t>(data);
    requestStatus[idx] = status;
  };

  auto sendIndex = std::make_shared<size_t>(0u);
  auto recvIndex = std::make_shared<size_t>(1u);

  // Submit and wait for transfers to complete
  requests[0] = _ep->tagSendBuilder(_sendPtr[0], _messageSize, ucxx::Tag{0})
                  .pythonFuture(false)
                  .callbackFunction(checkStatus)
                  .callbackData(sendIndex)
                  .build();
  requests[1] = _ep->tagRecvBuilder(_recvPtr[0], _messageSize, ucxx::Tag{0}, ucxx::TagMaskFull)
                  .pythonFuture(false)
                  .callbackFunction(checkStatus)
                  .callbackData(recvIndex)
                  .build();
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  for (const auto request : requests)
    ASSERT_THAT(request->getStatus(), UCS_OK);
  for (const auto status : requestStatus)
    ASSERT_THAT(status, UCS_OK);

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, TagUserCallbackDiscardReturn)
{
  allocate();

  std::vector<ucs_status_t> requestStatus(2, UCS_INPROGRESS);

  auto checkStatus = [&requestStatus](ucs_status_t status, ::ucxx::RequestCallbackUserData data) {
    auto idx           = *std::static_pointer_cast<size_t>(data);
    requestStatus[idx] = status;
  };

  auto checkCompletion = [&requestStatus, this]() {
    std::vector<size_t> completed(2, 0);
    while (std::accumulate(completed.begin(), completed.end(), 0) != 2) {
      _progressWorker();
      std::transform(
        requestStatus.begin(), requestStatus.end(), completed.begin(), [](ucs_status_t status) {
          return status == UCS_INPROGRESS ? 0 : 1;
        });
    }
  };

  auto sendIndex = std::make_shared<size_t>(0u);
  auto recvIndex = std::make_shared<size_t>(1u);

  // Submit and wait for transfers to complete via callbacks; the shared_ptr is discarded
  // but the request is kept alive by the endpoint's inflight-request registry.
  std::shared_ptr<ucxx::Request> sendReq =
    _ep->tagSendBuilder(_sendPtr[0], _messageSize, ucxx::Tag{0})
      .pythonFuture(false)
      .callbackFunction(checkStatus)
      .callbackData(sendIndex)
      .build();
  std::shared_ptr<ucxx::Request> recvReq =
    _ep->tagRecvBuilder(_recvPtr[0], _messageSize, ucxx::Tag{0}, ucxx::TagMaskFull)
      .pythonFuture(false)
      .callbackFunction(checkStatus)
      .callbackData(recvIndex)
      .build();
  checkCompletion();

  copyResults();

  for (const auto status : requestStatus)
    ASSERT_THAT(status, UCS_OK);

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryGet)
{
  allocate();

  auto memoryHandle =
    _context->memoryHandleBuilder(_messageSize).buffer(nullptr).memoryType(_memoryType).build();
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  // Fill memory handle with send data
  copyMemoryTypeAware(
    reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _sendPtr[0], _messageSize);

  auto localRemoteKey      = memoryHandle->remoteKeyBuilder().build();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memGetBuilder(_recvPtr[0], _messageSize, remoteKey).build());
  requests.push_back(_ep->flushBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryGetPreallocated)
{
  allocate();

  // Memory handles are always non-const
  auto memoryHandle = _context->memoryHandleBuilder(_messageSize)
                        .buffer(const_cast<void*>(_sendPtr[0]))
                        .memoryType(_memoryType)
                        .build();
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  auto localRemoteKey      = memoryHandle->remoteKeyBuilder().build();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memGetBuilder(_recvPtr[0], _messageSize, remoteKey).build());
  requests.push_back(_ep->flushBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryGetWithOffset)
{
  if (_messageLength < 2) GTEST_SKIP() << "Message too small to perform operations with offsets";
  allocate();

  size_t offset      = 1;
  size_t offsetBytes = offset * sizeof(_send[0][0]);

  auto memoryHandle =
    _context->memoryHandleBuilder(_messageSize).buffer(nullptr).memoryType(_memoryType).build();
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  // Fill memory handle with send data
  copyMemoryTypeAware(
    reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _sendPtr[0], _messageSize);

  auto localRemoteKey      = memoryHandle->remoteKeyBuilder().build();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep
                       ->memGetBuilder(reinterpret_cast<char*>(_recvPtr[0]) + offsetBytes,
                                       _messageSize - offsetBytes,
                                       remoteKey)
                       .remoteAddressOffset(offsetBytes)
                       .build());
  requests.push_back(_ep->flushBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert offset data correctness
  auto recvOffset = DataContainerType(_recv[0].begin() + offset, _recv[0].end());
  auto sendOffset = DataContainerType(_send[0].begin() + offset, _send[0].end());
  ASSERT_THAT(recvOffset, sendOffset);
}

TEST_P(RequestTest, MemoryPut)
{
  allocate();

  auto memoryHandle =
    _context->memoryHandleBuilder(_messageSize).buffer(nullptr).memoryType(_memoryType).build();
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  auto localRemoteKey      = memoryHandle->remoteKeyBuilder().build();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memPutBuilder(_sendPtr[0], _messageSize, remoteKey).build());
  requests.push_back(_ep->flushBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  // Copy memory handle data to receive buffer
  copyMemoryTypeAware(
    _recvPtr[0], reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _messageSize);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryPutPreallocated)
{
  allocate();

  auto memoryHandle =
    _context->memoryHandleBuilder(_messageSize).buffer(_recvPtr[0]).memoryType(_memoryType).build();
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  auto localRemoteKey      = memoryHandle->remoteKeyBuilder().build();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep->memPutBuilder(_sendPtr[0], _messageSize, remoteKey).build());
  requests.push_back(_ep->flushBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  copyResults();

  // Assert data correctness
  ASSERT_THAT(_recv[0], ContainerEq(_send[0]));
}

TEST_P(RequestTest, MemoryPutWithOffset)
{
  if (_messageLength < 2) GTEST_SKIP() << "Message too small to perform operations with offsets";
  allocate();

  size_t offset      = 1;
  size_t offsetBytes = offset * sizeof(_send[0][0]);

  auto memoryHandle =
    _context->memoryHandleBuilder(_messageSize).buffer(nullptr).memoryType(_memoryType).build();
  // If message size is 0, there's no allocation and memory type is then "host" by default.
  if (_messageSize > 0) ASSERT_EQ(memoryHandle->getMemoryType(), _memoryType);

  auto localRemoteKey      = memoryHandle->remoteKeyBuilder().build();
  auto serializedRemoteKey = localRemoteKey->serialize();
  auto remoteKey           = _ep->remoteKeyBuilder(serializedRemoteKey).build();

  std::vector<std::shared_ptr<ucxx::Request>> requests;
  requests.push_back(_ep
                       ->memPutBuilder(reinterpret_cast<const char*>(_sendPtr[0]) + offsetBytes,
                                       _messageSize - offsetBytes,
                                       remoteKey)
                       .remoteAddressOffset(offsetBytes)
                       .build());
  requests.push_back(_ep->flushBuilder().build());
  waitRequests(_worker, requests, _progressWorker);

  // Copy memory handle data to receive buffer
  copyMemoryTypeAware(
    _recvPtr[0], reinterpret_cast<void*>(memoryHandle->getBaseAddress()), _messageSize);

  copyResults();

  // Assert offset data correctness
  auto recvOffset = DataContainerType(_recv[0].begin() + offset, _recv[0].end());
  auto sendOffset = DataContainerType(_send[0].begin() + offset, _send[0].end());
  ASSERT_THAT(recvOffset, sendOffset);
}

INSTANTIATE_TEST_SUITE_P(ProgressModes,
                         RequestTest,
                         Combine(Values(TestBufferType::Host),
                                 Values(false),
                                 Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        // ProgressMode::Wait,  // Hangs on Stream
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));

INSTANTIATE_TEST_SUITE_P(DelayedSubmission,
                         RequestTest,
                         Combine(Values(TestBufferType::Host),
                                 Values(false),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));

#if UCXX_TESTS_ENABLE_RMM
INSTANTIATE_TEST_SUITE_P(RMMProgressModes,
                         RequestTest,
                         Combine(Values(TestBufferType::RMM),
                                 Values(false, true),
                                 Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        // ProgressMode::Wait,  // Hangs on Stream
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));

INSTANTIATE_TEST_SUITE_P(RMMDelayedSubmission,
                         RequestTest,
                         Combine(Values(TestBufferType::RMM),
                                 Values(false, true),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));
#endif

#if UCXX_ENABLE_CCCL
INSTANTIATE_TEST_SUITE_P(CCCLProgressModes,
                         RequestTest,
                         Combine(Values(TestBufferType::CCCL),
                                 Values(false, true),
                                 Values(false),
                                 Values(ProgressMode::Polling,
                                        ProgressMode::Blocking,
                                        // ProgressMode::Wait,  // Hangs on Stream
                                        ProgressMode::ThreadPolling,
                                        ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));

INSTANTIATE_TEST_SUITE_P(CCCLDelayedSubmission,
                         RequestTest,
                         Combine(Values(TestBufferType::CCCL),
                                 Values(false, true),
                                 Values(true),
                                 Values(ProgressMode::ThreadPolling, ProgressMode::ThreadBlocking),
                                 Values(0, 1, 1024, 2048, 1048576)));
#endif

}  // namespace
