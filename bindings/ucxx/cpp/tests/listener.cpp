/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <tuple>
#include <ucs/config/types.h>
#include <ucs/type/status.h>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#if UCXX_ENABLE_CCCL
#include <cuda_runtime_api.h>
#endif

#include <ucxx/api.h>

#include "include/utils.h"

namespace {

struct ListenerContainer {
  ucs_status_t status{UCS_OK};
  std::shared_ptr<ucxx::Worker> worker{nullptr};
  std::shared_ptr<ucxx::Listener> listener{nullptr};
  std::shared_ptr<ucxx::Endpoint> endpoint{nullptr};
  bool transferCompleted{false};
  bool endpointErrorHandling{false};
};

typedef std::shared_ptr<ListenerContainer> ListenerContainerPtr;

static void listenerCallback(ucp_conn_request_h connRequest, void* arg)
{
  ListenerContainer* listenerContainer = reinterpret_cast<ListenerContainer*>(arg);
  ucp_conn_request_attr_t attr{};
  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;

  listenerContainer->status = ucp_conn_request_query(connRequest, &attr);
  if (listenerContainer->status != UCS_OK) return;

  listenerContainer->endpoint = listenerContainer->listener->endpointBuilder(connRequest)
                                  .endpointErrorHandling(listenerContainer->endpointErrorHandling)
                                  .build();
}

class ListenerTestBase {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build()};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  bool _endpointErrorHandling{true};

  ListenerContainerPtr createListenerContainer()
  {
    auto listenerContainer                   = std::make_shared<ListenerContainer>();
    listenerContainer->worker                = _worker;
    listenerContainer->endpointErrorHandling = _endpointErrorHandling;
    return listenerContainer;
  }
};

class ListenerTest : public ListenerTestBase,
                     public ::testing::Test,
                     public ::testing::WithParamInterface<bool> {
 protected:
  virtual void SetUp()
  {
    _endpointErrorHandling = GetParam();
    _worker                = _context->workerBuilder().build();
  }

  virtual std::shared_ptr<ucxx::Listener> buildListener(ListenerContainerPtr listenerContainer)
  {
    auto listener = _worker->listenerBuilder(0, listenerCallback, listenerContainer.get()).build();
    listenerContainer->listener = listener;
    return listener;
  }
};

class ListenerPortTest : public ListenerTestBase,
                         public ::testing::Test,
                         public ::testing::WithParamInterface<uint16_t> {
 protected:
  uint16_t _port;
  virtual void SetUp()
  {
    _port   = GetParam();
    _worker = _context->workerBuilder().build();
  }
  virtual std::shared_ptr<ucxx::Listener> buildListener(ListenerContainerPtr listenerContainer)
  {
    auto listener =
      _worker->listenerBuilder(_port, listenerCallback, listenerContainer.get()).build();
    listenerContainer->listener = listener;
    return listener;
  }
};

TEST_P(ListenerTest, HandleIsValid)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  ASSERT_TRUE(listener->getHandle() != nullptr);
}

TEST_P(ListenerTest, EndpointSendRecv)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  auto progress          = getProgressFunction(_worker, ProgressMode::Polling);

  progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();
  while (listenerContainer->endpoint == nullptr)
    progress();

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  std::vector<int> client_buf{123};
  std::vector<int> server_buf{0};
  requests.push_back(
    ep->tagSendBuilder(client_buf.data(), client_buf.size() * sizeof(int), ucxx::Tag{0}).build());
  requests.push_back(
    listenerContainer->endpoint
      ->tagRecvBuilder(
        &server_buf.front(), server_buf.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull)
      .build());
  ::waitRequests(_worker, requests, progress);

  ASSERT_EQ(server_buf[0], client_buf[0]);

  requests.push_back(
    listenerContainer->endpoint
      ->tagSendBuilder(&server_buf.front(), server_buf.size() * sizeof(int), ucxx::Tag{1})
      .build());
  requests.push_back(
    ep->tagRecvBuilder(
        client_buf.data(), client_buf.size() * sizeof(int), ucxx::Tag{1}, ucxx::TagMaskFull)
      .build());
  ::waitRequests(_worker, requests, progress);
  ASSERT_EQ(client_buf[0], server_buf[0]);

  std::vector<int> buf{0};
}

TEST_P(ListenerTest, IsAlive)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();
  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  ASSERT_TRUE(ep->isAlive());

  std::vector<int> buf{123};
  std::shared_ptr<ucxx::Request> send_req =
    ep->tagSendBuilder(buf.data(), buf.size() * sizeof(int), ucxx::Tag{0}).build();
  while (!send_req->isCompleted())
    _worker->progress();

  listenerContainer->endpoint = nullptr;

  loopWithTimeout(std::chrono::milliseconds(5000), [this, ep]() {
    _worker->progress();
    return !ep->isAlive();
  });

  if (_endpointErrorHandling)
    ASSERT_FALSE(ep->isAlive());
  else
    ASSERT_TRUE(ep->isAlive());
}

TEST_P(ListenerTest, RaiseOnError)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();
  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  listenerContainer->endpoint = nullptr;

  loopWithTimeout(std::chrono::milliseconds(5000), [this, ep]() {
    try {
      _worker->progress();
      ep->raiseOnError();
    } catch (ucxx::Error) {
      return true;
    }
    return false;
  });

  if (_endpointErrorHandling) EXPECT_THROW(ep->raiseOnError(), ucxx::Error);
}

TEST_P(ListenerTest, EndpointCloseCallback)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();

  struct CallbackData {
    ucs_status_t status{UCS_INPROGRESS};
    bool closed{false};
  };

  auto callbackData = std::make_shared<CallbackData>();
  ep->setCloseCallback(
    [](ucs_status_t status, ucxx::EndpointCloseCallbackUserData callbackData) {
      auto cbData    = std::static_pointer_cast<CallbackData>(callbackData);
      cbData->status = status;
      cbData->closed = true;
    },
    callbackData);

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  ASSERT_FALSE(callbackData->closed);
  ASSERT_EQ(callbackData->status, UCS_INPROGRESS);

  listenerContainer->endpoint = nullptr;

  loopWithTimeout(std::chrono::milliseconds(5000), [this, &callbackData]() {
    _worker->progress();
    return callbackData->closed;
  });

  ASSERT_TRUE(callbackData->closed);
  EXPECT_NE(callbackData->status, UCS_INPROGRESS);
}

TEST_P(ListenerTest, EndpointNonBlockingClose)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  auto closeRequest = ep->closeBuilder().build();

  auto f = [this, &closeRequest]() {
    _worker->progress();
    return closeRequest->isCompleted();
  };
  loopWithTimeout(std::chrono::milliseconds(5000), f);

  if (_endpointErrorHandling)
    ASSERT_FALSE(ep->isAlive());
  else
    ASSERT_TRUE(ep->isAlive());
  EXPECT_NE(closeRequest->getStatus(), UCS_INPROGRESS);
}

TEST_P(ListenerTest, EndpointNonBlockingCloseWithCallbacks)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  auto closeCallback = [](ucs_status_t status, ucxx::EndpointCloseCallbackUserData data) {
    auto dataStatus = std::static_pointer_cast<bool>(data);
    *dataStatus     = status;
  };
  auto closeCallbackEndpoint = std::make_shared<ucs_status_t>(UCS_INPROGRESS);
  auto closeCallbackRequest  = std::make_shared<ucs_status_t>(UCS_INPROGRESS);

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort())
              .endpointErrorHandling(_endpointErrorHandling)
              .build();
  ep->setCloseCallback(closeCallback, closeCallbackEndpoint);

  while (listenerContainer->endpoint == nullptr)
    _worker->progress();

  auto closeRequest = ep->closeBuilder()
                        .pythonFuture(false)
                        .callbackFunction(closeCallback)
                        .callbackData(closeCallbackRequest)
                        .build();

  auto f = [this, &closeRequest]() {
    _worker->progress();
    return closeRequest->isCompleted();
  };
  loopWithTimeout(std::chrono::milliseconds(5000), f);

  if (_endpointErrorHandling)
    ASSERT_FALSE(ep->isAlive());
  else
    ASSERT_TRUE(ep->isAlive());
  EXPECT_NE(closeRequest->getStatus(), UCS_INPROGRESS);
  ASSERT_NE(*closeCallbackEndpoint, UCS_INPROGRESS);
  ASSERT_NE(*closeCallbackRequest, UCS_INPROGRESS);
}

INSTANTIATE_TEST_SUITE_P(EndpointErrorHandling, ListenerTest, ::testing::Values(true));

TEST_P(ListenerPortTest, Port)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = buildListener(listenerContainer);
  _worker->progress();

  if (GetParam() == 0)
    ASSERT_GE(listener->getPort(), 1024);
  else
    ASSERT_EQ(listener->getPort(), 12345);
}

INSTANTIATE_TEST_SUITE_P(PortAssignment, ListenerPortTest, ::testing::Values(0, 12345));

class ListenerHostTest : public ListenerTestBase, public ::testing::Test {
 protected:
  void SetUp() override { _worker = _context->workerBuilder().build(); }
};

#if UCXX_ENABLE_CCCL
class ListenerCudaCloseAfterSendTest : public ::testing::Test {
 protected:
  static constexpr size_t BufferSize{1 << 20};
  static constexpr size_t NumBuffers{3};
  static constexpr ucxx::Tag Tag{0};
  static constexpr ucxx::Tag ReceiptTag{1};

  std::shared_ptr<ucxx::Context> _context{nullptr};
  std::shared_ptr<ucxx::Worker> _clientWorker{nullptr};
  std::shared_ptr<ucxx::Worker> _serverWorker{nullptr};
  ListenerContainerPtr _listenerContainer{nullptr};
  std::shared_ptr<ucxx::Listener> _listener{nullptr};
  std::shared_ptr<ucxx::Endpoint> _client{nullptr};
  std::shared_ptr<ucxx::Endpoint> _server{nullptr};

  void SetUp() override
  {
    ASSERT_EQ(cudaFree(nullptr), cudaSuccess);

    _context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags)
                 .configMap({{"RNDV_THRESH", "8192"}})
                 .build();
    _clientWorker = _context->workerBuilder().cudaBufferType(ucxx::BufferType::CCCL).build();
    _serverWorker = _context->workerBuilder().cudaBufferType(ucxx::BufferType::CCCL).build();

    _listenerContainer                        = std::make_shared<ListenerContainer>();
    _listenerContainer->worker                = _serverWorker;
    _listenerContainer->endpointErrorHandling = true;
    _listener =
      _serverWorker->listenerBuilder(0, listenerCallback, _listenerContainer.get()).build();
    _listenerContainer->listener = _listener;

    _client = _clientWorker->endpointBuilder("127.0.0.1", _listener->getPort())
                .endpointErrorHandling(true)
                .build();
    ASSERT_TRUE(loopWithTimeout(std::chrono::milliseconds(5000), [this]() {
      _clientWorker->progress();
      _serverWorker->progress();
      return _listenerContainer->endpoint != nullptr;
    }));
    ASSERT_EQ(_listenerContainer->status, UCS_OK);
    _server = _listenerContainer->endpoint;
  }

  bool waitForRequest(const std::shared_ptr<ucxx::Request>& request)
  {
    return loopWithTimeout(std::chrono::milliseconds(5000), [this, request]() {
      _clientWorker->progress();
      _serverWorker->progress();
      return request->isCompleted();
    });
  }

  void closeServer() { _server->closeBlocking(10000000000 /* 10s */); }

  std::vector<std::shared_ptr<ucxx::Buffer>> allocateCudaBuffers()
  {
    std::vector<std::shared_ptr<ucxx::Buffer>> buffers;
    for (size_t i = 0; i < NumBuffers; ++i) {
      auto buffer = ucxx::allocateBuffer(ucxx::BufferType::CCCL, BufferSize);
      EXPECT_EQ(cudaMemset(buffer->data(), static_cast<int>(i + 1), BufferSize), cudaSuccess);
      buffers.push_back(std::move(buffer));
    }
    return buffers;
  }

  std::shared_ptr<ucxx::RequestTagMulti> postMultiReceive(
    const std::shared_ptr<ucxx::Endpoint>& endpoint)
  {
    return endpoint->tagMultiRecvBuilder(Tag, ucxx::TagMaskFull).pythonFuture(false).build();
  }

  std::shared_ptr<ucxx::RequestTagMulti> postMultiSend(
    const std::shared_ptr<ucxx::Endpoint>& endpoint,
    const std::vector<std::shared_ptr<ucxx::Buffer>>& buffers)
  {
    std::vector<const void*> pointers;
    std::vector<size_t> sizes(NumBuffers, BufferSize);
    std::vector<int> isCuda(NumBuffers, true);

    for (const auto& buffer : buffers)
      pointers.push_back(buffer->data());

    return endpoint->tagMultiSendBuilder(pointers, sizes, isCuda, Tag).pythonFuture(false).build();
  }

  std::shared_ptr<ucxx::Request> postReceiptReceive(const std::shared_ptr<ucxx::Endpoint>& endpoint,
                                                    uint8_t* receipt)
  {
    return endpoint->tagRecvBuilder(receipt, sizeof(*receipt), ReceiptTag, ucxx::TagMaskFull)
      .pythonFuture(false)
      .build();
  }

  std::shared_ptr<ucxx::Request> postReceiptSend(const std::shared_ptr<ucxx::Endpoint>& endpoint)
  {
    static const uint8_t receipt{1};
    return endpoint->tagSendBuilder(&receipt, sizeof(receipt), ReceiptTag)
      .pythonFuture(false)
      .build();
  }

  static std::vector<std::shared_ptr<ucxx::Buffer>> getReceivedBuffers(
    const std::shared_ptr<ucxx::RequestTagMulti>& request)
  {
    std::vector<std::shared_ptr<ucxx::Buffer>> buffers;
    for (const auto& child : request->_bufferRequests)
      if (child->buffer) buffers.push_back(child->buffer);
    return buffers;
  }

  bool waitForUnexpectedTag()
  {
    return loopWithTimeout(std::chrono::milliseconds(5000), [this]() {
      _serverWorker->progressOnce();
      _clientWorker->progressOnce();
      return _clientWorker->tagProbe(Tag)->isMatched();
    });
  }

  void expectChildrenCompletedSuccessfully(const std::shared_ptr<ucxx::RequestTagMulti>& request)
  {
    for (size_t i = 0; i < request->_bufferRequests.size(); ++i) {
      const auto& child = request->_bufferRequests[i]->request;
      ASSERT_NE(child, nullptr) << "child " << i;
      EXPECT_TRUE(child->isCompleted()) << "child " << i;
      EXPECT_EQ(child->getStatus(), UCS_OK) << "child " << i;
    }
  }
};

TEST_F(ListenerCudaCloseAfterSendTest, TagMultiExpectedMessageConfirmsBeforeSenderClose)
{
  auto sendBuffers = allocateCudaBuffers();
  auto receive     = postMultiReceive(_client);
  uint8_t receipt{0};
  auto receiptReceive = postReceiptReceive(_server, &receipt);
  auto send           = postMultiSend(_server, sendBuffers);

  ASSERT_TRUE(waitForRequest(send));
  ASSERT_EQ(send->getStatus(), UCS_OK);
  expectChildrenCompletedSuccessfully(send);

  ASSERT_TRUE(waitForRequest(receive));
  ASSERT_EQ(receive->getStatus(), UCS_OK);
  expectChildrenCompletedSuccessfully(receive);

  auto receiptSend = postReceiptSend(_client);
  ASSERT_TRUE(waitForRequest(receiptSend));
  ASSERT_EQ(receiptSend->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(receiptReceive));
  ASSERT_EQ(receiptReceive->getStatus(), UCS_OK);
  ASSERT_EQ(receipt, 1);

  closeServer();
}

TEST_F(ListenerCudaCloseAfterSendTest, TagMultiExpectedMessageCompletesBeforeSenderClose)
{
  auto sendBuffers = allocateCudaBuffers();
  auto receive     = postMultiReceive(_client);
  auto send        = postMultiSend(_server, sendBuffers);

  ASSERT_TRUE(waitForRequest(send));
  ASSERT_EQ(send->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(receive));
  ASSERT_EQ(receive->getStatus(), UCS_OK);

  closeServer();

  expectChildrenCompletedSuccessfully(send);
  expectChildrenCompletedSuccessfully(receive);
}

TEST_F(ListenerCudaCloseAfterSendTest, TagMultiUnexpectedMessageConfirmsBeforeSenderClose)
{
  // Complete the client-to-server half of the echo before submitting the response.
  auto clientBuffers = allocateCudaBuffers();
  auto serverReceive = postMultiReceive(_server);
  auto clientSend    = postMultiSend(_client, clientBuffers);
  ASSERT_TRUE(waitForRequest(clientSend));
  ASSERT_EQ(clientSend->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(serverReceive));
  ASSERT_EQ(serverReceive->getStatus(), UCS_OK);

  auto serverBuffers = getReceivedBuffers(serverReceive);
  ASSERT_EQ(serverBuffers.size(), NumBuffers);
  clientSend.reset();
  serverReceive.reset();

  // Mirror the coroutine ordering where the server begins its echo before the
  // client posts recv_multi(). The non-removing probe proves the header entered
  // UCX's unexpected-message queue and leaves it there for tagMultiRecv.
  uint8_t receipt{0};
  auto receiptReceive = postReceiptReceive(_server, &receipt);
  auto serverSend     = postMultiSend(_server, serverBuffers);
  ASSERT_TRUE(waitForUnexpectedTag());
  auto clientReceive = postMultiReceive(_client);

  ASSERT_TRUE(waitForRequest(serverSend));
  ASSERT_EQ(serverSend->getStatus(), UCS_OK);
  expectChildrenCompletedSuccessfully(serverSend);

  ASSERT_TRUE(waitForRequest(clientReceive));
  ASSERT_EQ(clientReceive->getStatus(), UCS_OK);
  expectChildrenCompletedSuccessfully(clientReceive);

  auto receiptSend = postReceiptSend(_client);
  ASSERT_TRUE(waitForRequest(receiptSend));
  ASSERT_EQ(receiptSend->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(receiptReceive));
  ASSERT_EQ(receiptReceive->getStatus(), UCS_OK);
  ASSERT_EQ(receipt, 1);

  closeServer();
}

TEST_F(ListenerCudaCloseAfterSendTest, TagUnexpectedMessageConfirmsBeforeSenderClose)
{
  auto sendBuffer = ucxx::allocateBuffer(ucxx::BufferType::CCCL, BufferSize);
  auto recvBuffer = ucxx::allocateBuffer(ucxx::BufferType::CCCL, BufferSize);
  ASSERT_EQ(cudaMemset(sendBuffer->data(), 1, BufferSize), cudaSuccess);

  uint8_t receipt{0};
  auto receiptReceive = postReceiptReceive(_server, &receipt);
  auto send =
    _server->tagSendBuilder(sendBuffer->data(), BufferSize, Tag).pythonFuture(false).build();
  ASSERT_TRUE(waitForUnexpectedTag());
  auto receive = _client->tagRecvBuilder(recvBuffer->data(), BufferSize, Tag, ucxx::TagMaskFull)
                   .pythonFuture(false)
                   .build();

  ASSERT_TRUE(waitForRequest(send));
  ASSERT_EQ(send->getStatus(), UCS_OK);

  ASSERT_TRUE(waitForRequest(receive));
  ASSERT_EQ(receive->getStatus(), UCS_OK);

  auto receiptSend = postReceiptSend(_client);
  ASSERT_TRUE(waitForRequest(receiptSend));
  ASSERT_EQ(receiptSend->getStatus(), UCS_OK);
  ASSERT_TRUE(waitForRequest(receiptReceive));
  ASSERT_EQ(receiptReceive->getStatus(), UCS_OK);
  ASSERT_EQ(receipt, 1);

  closeServer();
}
#endif

TEST_F(ListenerHostTest, BindToHost)
{
  auto listenerContainer = createListenerContainer();
  auto listener          = _worker->listenerBuilder(0, listenerCallback, listenerContainer.get())
                    .host("127.0.0.1")
                    .build();
  listenerContainer->listener = listener;
  auto progress               = getProgressFunction(_worker, ProgressMode::Polling);

  progress();

  ASSERT_EQ(listener->getIp(), "127.0.0.1");
  ASSERT_GE(listener->getPort(), 1024);

  auto ep = _worker->endpointBuilder("127.0.0.1", listener->getPort()).build();
  while (listenerContainer->endpoint == nullptr)
    progress();

  std::vector<std::shared_ptr<ucxx::Request>> requests;

  std::vector<int> client_buf{123};
  std::vector<int> server_buf{0};
  requests.push_back(
    ep->tagSendBuilder(client_buf.data(), client_buf.size() * sizeof(int), ucxx::Tag{0}).build());
  requests.push_back(
    listenerContainer->endpoint
      ->tagRecvBuilder(
        &server_buf.front(), server_buf.size() * sizeof(int), ucxx::Tag{0}, ucxx::TagMaskFull)
      .build());
  ::waitRequests(_worker, requests, progress);

  ASSERT_EQ(server_buf[0], client_buf[0]);
}

TEST_F(ListenerHostTest, BindToInvalidHost)
{
  auto listenerContainer = createListenerContainer();
  EXPECT_THROW(std::ignore = _worker->listenerBuilder(0, listenerCallback, listenerContainer.get())
                               .host("invalid-address!")
                               .build(),
               ucxx::Error);
}

}  // namespace
