/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/api.h>
#include <ucxx/request_am.h>
#include <ucxx/request_endpoint_close.h>
#include <ucxx/request_flush.h>
#include <ucxx/request_mem.h>
#include <ucxx/request_stream.h>
#include <ucxx/request_tag.h>

#include "include/utils.h"

namespace {

class RequestBuilderTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build()};
  std::shared_ptr<ucxx::Worker> _worker{nullptr};
  std::shared_ptr<ucxx::Endpoint> _ep{nullptr};

  void SetUp() override
  {
    _worker = _context->workerBuilder().build();
    _ep     = _worker->endpointBuilder(_worker->addressBuilder().build()).build();
  }

  void progressUntilCompleted(std::shared_ptr<ucxx::Request> req)
  {
    while (!req->isCompleted())
      _worker->progress();
    req->checkError();
  }
};

template <typename BuilderType, typename TargetType>
void assertBuildRequiresNonConstBuilder()
{
  static_assert(std::is_invocable_r<std::shared_ptr<TargetType>,
                                    decltype(&BuilderType::build),
                                    BuilderType&>::value,
                "Non-const builder can call build() to produce target shared_ptr");
  static_assert(!std::is_invocable_r<std::shared_ptr<TargetType>,
                                     decltype(&BuilderType::build),
                                     const BuilderType&>::value,
                "Const builder cannot call build() to produce target shared_ptr");
}

template <typename BuilderType, typename TargetType>
void assertConversionRequiresNonConstBuilder()
{
  static_assert(std::is_convertible<BuilderType&, std::shared_ptr<TargetType>>::value,
                "Non-const builder can implicitly convert to target shared_ptr");
  static_assert(!std::is_convertible<const BuilderType&, std::shared_ptr<TargetType>>::value,
                "Const builder cannot implicitly convert to target shared_ptr");
}

template <typename T, typename = void>
struct HasEndpointAmSendBuilderReceiverCallbackArg : std::false_type {};

template <typename T>
struct HasEndpointAmSendBuilderReceiverCallbackArg<
  T,
  std::void_t<decltype(std::declval<T&>().amSendBuilder(
    static_cast<const void*>(nullptr),
    size_t{},
    UCS_MEMORY_TYPE_HOST,
    std::optional<ucxx::AmReceiverCallbackInfo>{}))>> : std::true_type {};

template <typename T, typename = void>
struct HasEndpointAmRecvBuilderFutureArg : std::false_type {};

template <typename T>
struct HasEndpointAmRecvBuilderFutureArg<
  T,
  std::void_t<decltype(std::declval<T&>().amRecvBuilder(false))>> : std::true_type {};

template <typename T, typename = void>
struct HasEndpointMemPutBuilderRemoteKeyOffsetArg : std::false_type {};

template <typename T>
struct HasEndpointMemPutBuilderRemoteKeyOffsetArg<
  T,
  std::void_t<decltype(std::declval<T&>().memPutBuilder(
    static_cast<const void*>(nullptr), size_t{}, std::shared_ptr<ucxx::RemoteKey>{}, uint64_t{}))>>
  : std::true_type {};

template <typename T, typename = void>
struct HasEndpointMemGetBuilderRemoteKeyOffsetArg : std::false_type {};

template <typename T>
struct HasEndpointMemGetBuilderRemoteKeyOffsetArg<
  T,
  std::void_t<decltype(std::declval<T&>().memGetBuilder(
    static_cast<void*>(nullptr), size_t{}, std::shared_ptr<ucxx::RemoteKey>{}, uint64_t{}))>>
  : std::true_type {};

template <typename T, typename = void>
struct HasEndpointStreamSendBuilderFutureArg : std::false_type {};

template <typename T>
struct HasEndpointStreamSendBuilderFutureArg<
  T,
  std::void_t<decltype(std::declval<T&>().streamSendBuilder(
    static_cast<const void*>(nullptr), size_t{}, false))>> : std::true_type {};

template <typename T, typename = void>
struct HasEndpointStreamRecvBuilderFutureArg : std::false_type {};

template <typename T>
struct HasEndpointStreamRecvBuilderFutureArg<
  T,
  std::void_t<decltype(std::declval<T&>().streamRecvBuilder(
    static_cast<void*>(nullptr), size_t{}, false))>> : std::true_type {};

template <typename T, typename = void>
struct HasEndpointTagSendBuilderCallbackArgs : std::false_type {};

template <typename T>
struct HasEndpointTagSendBuilderCallbackArgs<
  T,
  std::void_t<decltype(std::declval<T&>().tagSendBuilder(static_cast<const void*>(nullptr),
                                                         size_t{},
                                                         ucxx::Tag{},
                                                         false,
                                                         ucxx::RequestCallbackUserFunction{},
                                                         ucxx::RequestCallbackUserData{}))>>
  : std::true_type {};

template <typename T, typename = void>
struct HasEndpointTagRecvBuilderCallbackArgs : std::false_type {};

template <typename T>
struct HasEndpointTagRecvBuilderCallbackArgs<
  T,
  std::void_t<decltype(std::declval<T&>().tagRecvBuilder(static_cast<void*>(nullptr),
                                                         size_t{},
                                                         ucxx::Tag{},
                                                         ucxx::TagMask{},
                                                         false,
                                                         ucxx::RequestCallbackUserFunction{},
                                                         ucxx::RequestCallbackUserData{}))>>
  : std::true_type {};

template <typename T, typename = void>
struct HasEndpointTagMultiSendBuilderFutureArg : std::false_type {};

template <typename T>
struct HasEndpointTagMultiSendBuilderFutureArg<
  T,
  std::void_t<decltype(std::declval<T&>().tagMultiSendBuilder(
    std::declval<const std::vector<const void*>&>(),
    std::declval<const std::vector<size_t>&>(),
    std::declval<const std::vector<int>&>(),
    ucxx::Tag{},
    false))>> : std::true_type {};

template <typename T, typename = void>
struct HasEndpointTagMultiRecvBuilderFutureArg : std::false_type {};

template <typename T>
struct HasEndpointTagMultiRecvBuilderFutureArg<
  T,
  std::void_t<decltype(std::declval<T&>().tagMultiRecvBuilder(
    ucxx::Tag{}, ucxx::TagMask{}, false))>> : std::true_type {};

template <typename T, typename = void>
struct HasEndpointFlushBuilderFutureArg : std::false_type {};

template <typename T>
struct HasEndpointFlushBuilderFutureArg<
  T,
  std::void_t<decltype(std::declval<T&>().flushBuilder(false))>> : std::true_type {};

template <typename T, typename = void>
struct HasWorkerTagRecvBuilderCallbackArgs : std::false_type {};

template <typename T>
struct HasWorkerTagRecvBuilderCallbackArgs<
  T,
  std::void_t<decltype(std::declval<T&>().tagRecvBuilder(static_cast<void*>(nullptr),
                                                         size_t{},
                                                         ucxx::Tag{},
                                                         ucxx::TagMask{},
                                                         false,
                                                         ucxx::RequestCallbackUserFunction{},
                                                         ucxx::RequestCallbackUserData{}))>>
  : std::true_type {};

template <typename T, typename = void>
struct HasWorkerTagRecvWithHandleBuilderCallbackArgs : std::false_type {};

template <typename T>
struct HasWorkerTagRecvWithHandleBuilderCallbackArgs<
  T,
  std::void_t<decltype(std::declval<T&>().tagRecvWithHandleBuilder(
    static_cast<void*>(nullptr),
    std::shared_ptr<ucxx::TagProbeInfo>{},
    false,
    ucxx::RequestCallbackUserFunction{},
    ucxx::RequestCallbackUserData{}))>> : std::true_type {};

template <typename T, typename = void>
struct HasWorkerFlushBuilderFutureArg : std::false_type {};

template <typename T>
struct HasWorkerFlushBuilderFutureArg<T,
                                      std::void_t<decltype(std::declval<T&>().flushBuilder(false))>>
  : std::true_type {};

TEST_F(RequestBuilderTest, FlushBuilderBasicWorker)
{
  auto req = ucxx::requestFlushBuilder(_worker, ucxx::data::Flush{}).build();
  static_assert(std::is_same<decltype(req), std::shared_ptr<ucxx::RequestFlush>>::value,
                ".build() returns shared_ptr<RequestFlush>");

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, FlushBuilderBasicEndpoint)
{
  auto req = ucxx::requestFlushBuilder(_ep, ucxx::data::Flush{}).build();

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, FlushBuilderWithPythonFuture)
{
  auto req = ucxx::requestFlushBuilder(_worker, ucxx::data::Flush{}).pythonFuture(false).build();

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, FlushBuilderMethodChaining)
{
  ucxx::RequestCallbackUserFunction callbackFn{nullptr};
  ucxx::RequestCallbackUserData callbackData{nullptr};
  auto req = ucxx::requestFlushBuilder(_worker, ucxx::data::Flush{})
               .pythonFuture(false)
               .callbackFunction(callbackFn)
               .callbackData(callbackData)
               .build();

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, FlushBuilderImplicitConversion)
{
  std::shared_ptr<ucxx::RequestFlush> req = ucxx::requestFlushBuilder(_worker, ucxx::data::Flush{});

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, TagBuilderSendReceivePair)
{
  std::vector<int> sendBuf{42, 43, 44};
  std::vector<int> recvBuf(3);
  auto tag = ucxx::Tag{1};

  auto sendData = ucxx::data::TagSend(sendBuf.data(), sendBuf.size() * sizeof(int), tag);
  auto recvData =
    ucxx::data::TagReceive(recvBuf.data(), recvBuf.size() * sizeof(int), tag, ucxx::TagMaskFull);

  auto sendReq = ucxx::requestTagBuilder(_ep, sendData).build();
  auto recvReq = ucxx::requestTagBuilder(_worker, recvData).build();
  static_assert(std::is_same<decltype(sendReq), std::shared_ptr<ucxx::RequestTag>>::value,
                ".build() returns shared_ptr<RequestTag>");

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();

  EXPECT_EQ(sendBuf, recvBuf);
}

TEST_F(RequestBuilderTest, TagBuilderWithPythonFuture)
{
  std::vector<int> sendBuf{1};
  std::vector<int> recvBuf(1);
  auto tag = ucxx::Tag{2};

  auto sendData = ucxx::data::TagSend(sendBuf.data(), sendBuf.size() * sizeof(int), tag);
  auto recvData =
    ucxx::data::TagReceive(recvBuf.data(), recvBuf.size() * sizeof(int), tag, ucxx::TagMaskFull);

  auto sendReq = ucxx::requestTagBuilder(_ep, sendData).pythonFuture(false).build();
  auto recvReq = ucxx::requestTagBuilder(_worker, recvData).pythonFuture(false).build();

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();
}

TEST_F(RequestBuilderTest, TagBuilderImplicitConversion)
{
  std::vector<int> sendBuf{7};
  std::vector<int> recvBuf(1);
  auto tag = ucxx::Tag{3};

  auto sendData = ucxx::data::TagSend(sendBuf.data(), sendBuf.size() * sizeof(int), tag);
  auto recvData =
    ucxx::data::TagReceive(recvBuf.data(), recvBuf.size() * sizeof(int), tag, ucxx::TagMaskFull);

  std::shared_ptr<ucxx::RequestTag> sendReq = ucxx::requestTagBuilder(_ep, sendData);
  std::shared_ptr<ucxx::RequestTag> recvReq = ucxx::requestTagBuilder(_worker, recvData);

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();
}

TEST_F(RequestBuilderTest, StreamBuilderSendReceivePair)
{
  std::vector<int> sendBuf{10, 20};
  std::vector<int> recvBuf(2);

  auto sendData = ucxx::data::StreamSend(sendBuf.data(), sendBuf.size() * sizeof(int));
  auto recvData = ucxx::data::StreamReceive(recvBuf.data(), recvBuf.size() * sizeof(int));
  auto sendReq  = ucxx::requestStreamBuilder(_ep, sendData).build();
  auto recvReq  = ucxx::requestStreamBuilder(_ep, recvData).build();
  static_assert(std::is_same<decltype(sendReq), std::shared_ptr<ucxx::RequestStream>>::value,
                ".build() returns shared_ptr<RequestStream>");

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();
}

TEST_F(RequestBuilderTest, StreamBuilderImplicitConversion)
{
  std::vector<int> sendBuf{99};
  std::vector<int> recvBuf(1);

  auto sendData = ucxx::data::StreamSend(sendBuf.data(), sendBuf.size() * sizeof(int));
  auto recvData = ucxx::data::StreamReceive(recvBuf.data(), recvBuf.size() * sizeof(int));

  std::shared_ptr<ucxx::RequestStream> sendReq = ucxx::requestStreamBuilder(_ep, sendData);
  std::shared_ptr<ucxx::RequestStream> recvReq = ucxx::requestStreamBuilder(_ep, recvData);

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();
}

TEST_F(RequestBuilderTest, EndpointCloseBuilderAutoType)
{
  auto ep      = _worker->endpointBuilder(_worker->addressBuilder().build()).build();
  auto builder = ucxx::requestEndpointCloseBuilder(ep, ucxx::data::EndpointClose{false});
  static_assert(std::is_same<decltype(builder), ucxx::RequestEndpointCloseBuilder>::value,
                "auto without .build() is RequestEndpointCloseBuilder");
}

TEST_F(RequestBuilderTest, EndpointCloseBuilderMethodChaining)
{
  auto ep = _worker->endpointBuilder(_worker->addressBuilder().build()).build();

  ucxx::RequestCallbackUserFunction callbackFn{nullptr};
  ucxx::RequestCallbackUserData callbackData{nullptr};
  auto builder = ucxx::requestEndpointCloseBuilder(ep, ucxx::data::EndpointClose{false})
                   .pythonFuture(false)
                   .callbackFunction(callbackFn)
                   .callbackData(callbackData);

  static_assert(std::is_same<decltype(builder), ucxx::RequestEndpointCloseBuilder>::value,
                "chained builder without .build() is still RequestEndpointCloseBuilder");
}

TEST_F(RequestBuilderTest, EndpointCloseBuilderBuild)
{
  auto ep  = _worker->endpointBuilder(_worker->addressBuilder().build()).build();
  auto req = ucxx::requestEndpointCloseBuilder(ep, ucxx::data::EndpointClose{true}).build();

  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, AllBuilderAutoTypes)
{
  std::vector<int> buf{0};
  auto tag = ucxx::Tag{0};

  auto tagBuilder = ucxx::requestTagBuilder(_ep, ucxx::data::TagSend(buf.data(), sizeof(int), tag));
  static_assert(std::is_same<decltype(tagBuilder), ucxx::RequestTagBuilder>::value,
                "auto without .build() is RequestTagBuilder");

  auto amBuilder = ucxx::requestAmBuilder(_ep, ucxx::data::AmSend(buf.data(), sizeof(int)));
  static_assert(std::is_same<decltype(amBuilder), ucxx::RequestAmBuilder>::value,
                "auto without .build() is RequestAmBuilder");

  // RequestMemBuilder - don't submit a request here; that needs a real remote key.
  auto memBuilder =
    ucxx::requestMemBuilder(_ep, ucxx::data::MemPut(buf.data(), sizeof(int), 0, nullptr));
  static_assert(std::is_same<decltype(memBuilder), ucxx::RequestMemBuilder>::value,
                "auto without .build() is RequestMemBuilder");
  static_assert(
    std::is_same<decltype(memBuilder.build()), std::shared_ptr<ucxx::RequestMem>>::value,
    "calling .build() on RequestMemBuilder returns shared_ptr<RequestMem>");

  auto streamBuilder =
    ucxx::requestStreamBuilder(_ep, ucxx::data::StreamSend(buf.data(), sizeof(int)));
  static_assert(std::is_same<decltype(streamBuilder), ucxx::RequestStreamBuilder>::value,
                "auto without .build() is RequestStreamBuilder");

  auto flushBuilder = ucxx::requestFlushBuilder(_worker, ucxx::data::Flush{});
  static_assert(std::is_same<decltype(flushBuilder), ucxx::RequestFlushBuilder>::value,
                "auto without .build() is RequestFlushBuilder");

  {
    auto closeEp = _worker->endpointBuilder(_worker->addressBuilder().build()).build();
    auto closeBuilder =
      ucxx::requestEndpointCloseBuilder(closeEp, ucxx::data::EndpointClose{false});
    static_assert(std::is_same<decltype(closeBuilder), ucxx::RequestEndpointCloseBuilder>::value,
                  "auto without .build() is RequestEndpointCloseBuilder");
  }

  std::vector<const void*> multiSendBuf{buf.data()};
  std::vector<size_t> multiSendLen{sizeof(int)};
  std::vector<int> isCUDA{0};
  auto tagMultiBuilder = ucxx::requestTagMultiBuilder(
    _ep, ucxx::data::TagMultiSend(multiSendBuf, multiSendLen, isCUDA, tag));
  static_assert(std::is_same<decltype(tagMultiBuilder), ucxx::RequestTagMultiBuilder>::value,
                "auto without .build() is RequestTagMultiBuilder");

  auto flushReq = flushBuilder.build();
  static_assert(std::is_same<decltype(flushReq), std::shared_ptr<ucxx::RequestFlush>>::value,
                "calling .build() on RequestFlushBuilder returns shared_ptr<RequestFlush>");

  ASSERT_TRUE(flushReq != nullptr);
  progressUntilCompleted(flushReq);
}

TEST(RequestBuilderTraitsTest, AllBuildersAreMoveOnly)
{
  static_assert(!std::is_copy_constructible<ucxx::RequestAmBuilder>::value,
                "RequestAmBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::RequestAmBuilder>::value,
                "RequestAmBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::RequestAmBuilder>::value,
                "RequestAmBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::RequestEndpointCloseBuilder>::value,
                "RequestEndpointCloseBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::RequestEndpointCloseBuilder>::value,
                "RequestEndpointCloseBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::RequestEndpointCloseBuilder>::value,
                "RequestEndpointCloseBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::RequestFlushBuilder>::value,
                "RequestFlushBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::RequestFlushBuilder>::value,
                "RequestFlushBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::RequestFlushBuilder>::value,
                "RequestFlushBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::RequestMemBuilder>::value,
                "RequestMemBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::RequestMemBuilder>::value,
                "RequestMemBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::RequestMemBuilder>::value,
                "RequestMemBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::RequestStreamBuilder>::value,
                "RequestStreamBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::RequestStreamBuilder>::value,
                "RequestStreamBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::RequestStreamBuilder>::value,
                "RequestStreamBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::RequestTagBuilder>::value,
                "RequestTagBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::RequestTagBuilder>::value,
                "RequestTagBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::RequestTagBuilder>::value,
                "RequestTagBuilder must remain move constructible");

  static_assert(!std::is_copy_constructible<ucxx::RequestTagMultiBuilder>::value,
                "RequestTagMultiBuilder must not be copy constructible");
  static_assert(!std::is_copy_assignable<ucxx::RequestTagMultiBuilder>::value,
                "RequestTagMultiBuilder must not be copy assignable");
  static_assert(std::is_move_constructible<ucxx::RequestTagMultiBuilder>::value,
                "RequestTagMultiBuilder must remain move constructible");
}

TEST(RequestBuilderTraitsTest, BuildAndConversionRequireNonConstBuilders)
{
  assertBuildRequiresNonConstBuilder<ucxx::RequestAmBuilder, ucxx::RequestAm>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestAmBuilder, ucxx::RequestAm>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestAmBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::RequestEndpointCloseBuilder,
                                     ucxx::RequestEndpointClose>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestEndpointCloseBuilder,
                                          ucxx::RequestEndpointClose>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestEndpointCloseBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::RequestFlushBuilder, ucxx::RequestFlush>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestFlushBuilder, ucxx::RequestFlush>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestFlushBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::RequestMemBuilder, ucxx::RequestMem>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestMemBuilder, ucxx::RequestMem>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestMemBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::RequestStreamBuilder, ucxx::RequestStream>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestStreamBuilder, ucxx::RequestStream>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestStreamBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::RequestTagBuilder, ucxx::RequestTag>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestTagBuilder, ucxx::RequestTag>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestTagBuilder, ucxx::Request>();

  assertBuildRequiresNonConstBuilder<ucxx::RequestTagMultiBuilder, ucxx::RequestTagMulti>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestTagMultiBuilder, ucxx::RequestTagMulti>();
  assertConversionRequiresNonConstBuilder<ucxx::RequestTagMultiBuilder, ucxx::Request>();
}

TEST(RequestBuilderTraitsTest, EndpointWorkerBuilderMethodsReturnBuilders)
{
  using Endpoint = ucxx::Endpoint;
  using Worker   = ucxx::Worker;

  static_assert(std::is_same<decltype(std::declval<Endpoint&>().amSendBuilder(
                               static_cast<const void*>(nullptr), size_t{}, UCS_MEMORY_TYPE_HOST)),
                             ucxx::RequestAmBuilder>::value,
                "Endpoint::amSendBuilder returns RequestAmBuilder");
  static_assert(std::is_same<decltype(std::declval<Endpoint&>().amRecvBuilder()),
                             ucxx::RequestAmBuilder>::value,
                "Endpoint::amRecvBuilder returns RequestAmBuilder");
  static_assert(std::is_same<decltype(std::declval<ucxx::RequestAmBuilder&>().receiverCallbackInfo(
                               std::optional<ucxx::AmReceiverCallbackInfo>{})),
                             ucxx::RequestAmBuilder&>::value,
                "RequestAmBuilder::receiverCallbackInfo returns RequestAmBuilder&");
  static_assert(std::is_same<decltype(std::declval<Endpoint&>().memPutBuilder(
                               static_cast<const void*>(nullptr), size_t{}, uint64_t{}, nullptr)),
                             ucxx::RequestMemBuilder>::value,
                "Endpoint::memPutBuilder returns RequestMemBuilder");
  static_assert(std::is_same<decltype(std::declval<Endpoint&>().memGetBuilder(
                               static_cast<void*>(nullptr), size_t{}, uint64_t{}, nullptr)),
                             ucxx::RequestMemBuilder>::value,
                "Endpoint::memGetBuilder returns RequestMemBuilder");
  static_assert(std::is_same<decltype(std::declval<Endpoint&>().streamSendBuilder(
                               static_cast<const void*>(nullptr), size_t{})),
                             ucxx::RequestStreamBuilder>::value,
                "Endpoint::streamSendBuilder returns RequestStreamBuilder");
  static_assert(std::is_same<decltype(std::declval<Endpoint&>().streamRecvBuilder(
                               static_cast<void*>(nullptr), size_t{})),
                             ucxx::RequestStreamBuilder>::value,
                "Endpoint::streamRecvBuilder returns RequestStreamBuilder");
  static_assert(std::is_same<decltype(std::declval<Endpoint&>().tagSendBuilder(
                               static_cast<const void*>(nullptr), size_t{}, ucxx::Tag{})),
                             ucxx::RequestTagBuilder>::value,
                "Endpoint::tagSendBuilder returns RequestTagBuilder");
  static_assert(
    std::is_same<decltype(std::declval<Endpoint&>().tagRecvBuilder(
                   static_cast<void*>(nullptr), size_t{}, ucxx::Tag{}, ucxx::TagMask{})),
                 ucxx::RequestTagBuilder>::value,
    "Endpoint::tagRecvBuilder returns RequestTagBuilder");
  static_assert(std::is_same<decltype(std::declval<Endpoint&>().tagMultiSendBuilder(
                               std::declval<const std::vector<const void*>&>(),
                               std::declval<const std::vector<size_t>&>(),
                               std::declval<const std::vector<int>&>(),
                               ucxx::Tag{})),
                             ucxx::RequestTagMultiBuilder>::value,
                "Endpoint::tagMultiSendBuilder returns RequestTagMultiBuilder");
  static_assert(std::is_same<decltype(std::declval<Endpoint&>().tagMultiRecvBuilder(
                               ucxx::Tag{}, ucxx::TagMask{})),
                             ucxx::RequestTagMultiBuilder>::value,
                "Endpoint::tagMultiRecvBuilder returns RequestTagMultiBuilder");
  static_assert(std::is_same<decltype(std::declval<Endpoint&>().flushBuilder()),
                             ucxx::RequestFlushBuilder>::value,
                "Endpoint::flushBuilder returns RequestFlushBuilder");
  static_assert(
    std::is_same<decltype(std::declval<Worker&>().tagRecvBuilder(
                   static_cast<void*>(nullptr), size_t{}, ucxx::Tag{}, ucxx::TagMask{})),
                 ucxx::RequestTagBuilder>::value,
    "Worker::tagRecvBuilder returns RequestTagBuilder");
  static_assert(std::is_same<decltype(std::declval<Worker&>().tagRecvWithHandleBuilder(
                               static_cast<void*>(nullptr), std::shared_ptr<ucxx::TagProbeInfo>{})),
                             ucxx::RequestTagBuilder>::value,
                "Worker::tagRecvWithHandleBuilder returns RequestTagBuilder");
  static_assert(std::is_same<decltype(std::declval<Worker&>().flushBuilder()),
                             ucxx::RequestFlushBuilder>::value,
                "Worker::flushBuilder returns RequestFlushBuilder");
}

TEST(RequestBuilderTraitsTest, EndpointWorkerBuilderMethodsRejectOptionalArguments)
{
  static_assert(!HasEndpointAmSendBuilderReceiverCallbackArg<ucxx::Endpoint>::value,
                "Endpoint::amSendBuilder should configure receiver callbacks via builders");
  static_assert(!HasEndpointAmRecvBuilderFutureArg<ucxx::Endpoint>::value,
                "Endpoint::amRecvBuilder should configure Python futures via builders");
  static_assert(!HasEndpointMemPutBuilderRemoteKeyOffsetArg<ucxx::Endpoint>::value,
                "Endpoint::memPutBuilder should keep optional offsets out of the builder API");
  static_assert(!HasEndpointMemGetBuilderRemoteKeyOffsetArg<ucxx::Endpoint>::value,
                "Endpoint::memGetBuilder should keep optional offsets out of the builder API");
  static_assert(!HasEndpointStreamSendBuilderFutureArg<ucxx::Endpoint>::value,
                "Endpoint::streamSendBuilder should configure Python futures via builders");
  static_assert(!HasEndpointStreamRecvBuilderFutureArg<ucxx::Endpoint>::value,
                "Endpoint::streamRecvBuilder should configure Python futures via builders");
  static_assert(!HasEndpointTagSendBuilderCallbackArgs<ucxx::Endpoint>::value,
                "Endpoint::tagSendBuilder should configure callbacks via builders");
  static_assert(!HasEndpointTagRecvBuilderCallbackArgs<ucxx::Endpoint>::value,
                "Endpoint::tagRecvBuilder should configure callbacks via builders");
  static_assert(!HasEndpointTagMultiSendBuilderFutureArg<ucxx::Endpoint>::value,
                "Endpoint::tagMultiSendBuilder should configure Python futures via builders");
  static_assert(!HasEndpointTagMultiRecvBuilderFutureArg<ucxx::Endpoint>::value,
                "Endpoint::tagMultiRecvBuilder should configure Python futures via builders");
  static_assert(!HasEndpointFlushBuilderFutureArg<ucxx::Endpoint>::value,
                "Endpoint::flushBuilder should configure Python futures via builders");
  static_assert(!HasWorkerTagRecvBuilderCallbackArgs<ucxx::Worker>::value,
                "Worker::tagRecvBuilder should configure callbacks via builders");
  static_assert(!HasWorkerTagRecvWithHandleBuilderCallbackArgs<ucxx::Worker>::value,
                "Worker::tagRecvWithHandleBuilder should configure callbacks via builders");
  static_assert(!HasWorkerFlushBuilderFutureArg<ucxx::Worker>::value,
                "Worker::flushBuilder should configure Python futures via builders");
}

TEST(RequestBuilderSingleUseTest, BuildAttemptMarksBuilderBuilt)
{
  auto builder =
    ucxx::requestFlushBuilder(std::shared_ptr<ucxx::Component>{nullptr}, ucxx::data::Flush{});

  EXPECT_THROW(std::ignore = builder.build(), ucxx::Error);
  EXPECT_THROW(std::ignore = builder.build(), std::logic_error);
}

TEST(RequestBuilderSingleUseTest, ImplicitConversionAttemptMarksBuilderBuilt)
{
  auto builder =
    ucxx::requestFlushBuilder(std::shared_ptr<ucxx::Component>{nullptr}, ucxx::data::Flush{});

  EXPECT_THROW(std::ignore = static_cast<std::shared_ptr<ucxx::Request>>(builder), ucxx::Error);
  EXPECT_THROW(std::ignore = static_cast<std::shared_ptr<ucxx::Request>>(builder),
               std::logic_error);
}

TEST_F(RequestBuilderTest, EndpointFlushBuilderMethod)
{
  auto builder = _ep->flushBuilder();
  static_assert(std::is_same<decltype(builder), ucxx::RequestFlushBuilder>::value,
                "ep->flushBuilder() returns RequestFlushBuilder");

  auto req = builder.build();
  static_assert(std::is_same<decltype(req), std::shared_ptr<ucxx::RequestFlush>>::value,
                "ep->flushBuilder().build() returns shared_ptr<RequestFlush>");
  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, WorkerFlushBuilderMethod)
{
  auto builder = _worker->flushBuilder();
  static_assert(std::is_same<decltype(builder), ucxx::RequestFlushBuilder>::value,
                "worker->flushBuilder() returns RequestFlushBuilder");

  auto req = builder.build();
  ASSERT_TRUE(req != nullptr);
  progressUntilCompleted(req);
}

TEST_F(RequestBuilderTest, EndpointWorkerTagBuilderMethods)
{
  std::vector<int> sendBuf{10, 20, 30};
  std::vector<int> recvBuf(3);
  auto tag     = ucxx::Tag{142};
  auto tagMask = ucxx::TagMaskFull;

  auto sendBuilder = _ep->tagSendBuilder(sendBuf.data(), sendBuf.size() * sizeof(int), tag);
  auto recvBuilder =
    _worker->tagRecvBuilder(recvBuf.data(), recvBuf.size() * sizeof(int), tag, tagMask);

  static_assert(std::is_same<decltype(sendBuilder), ucxx::RequestTagBuilder>::value,
                "ep->tagSendBuilder() returns RequestTagBuilder");
  static_assert(std::is_same<decltype(recvBuilder), ucxx::RequestTagBuilder>::value,
                "worker->tagRecvBuilder() returns RequestTagBuilder");

  auto sendReq = sendBuilder.build();
  auto recvReq = recvBuilder.build();
  static_assert(std::is_same<decltype(sendReq), std::shared_ptr<ucxx::RequestTag>>::value,
                "ep->tagSendBuilder().build() returns shared_ptr<RequestTag>");

  ASSERT_TRUE(sendReq != nullptr);
  ASSERT_TRUE(recvReq != nullptr);

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();

  EXPECT_EQ(sendBuf, recvBuf);
}

TEST_F(RequestBuilderTest, EndpointTagSendBuilderMethodWithCallbackChain)
{
  std::vector<int> sendBuf{1, 2};
  std::vector<int> recvBuf(2);
  auto tag     = ucxx::Tag{199};
  auto tagMask = ucxx::TagMaskFull;

  bool callbackCalled                  = false;
  ucxx::RequestCallbackUserFunction cb = [&callbackCalled](ucs_status_t, std::shared_ptr<void>) {
    callbackCalled = true;
  };

  auto sendReq = _ep->tagSendBuilder(sendBuf.data(), sendBuf.size() * sizeof(int), tag)
                   .callbackFunction(cb)
                   .build();
  auto recvReq =
    _worker->tagRecvBuilder(recvBuf.data(), recvBuf.size() * sizeof(int), tag, tagMask).build();

  while (!sendReq->isCompleted() || !recvReq->isCompleted())
    _worker->progress();
  sendReq->checkError();
  recvReq->checkError();

  EXPECT_EQ(sendBuf, recvBuf);
  EXPECT_TRUE(callbackCalled);
}

TEST_F(RequestBuilderTest, EndpointBuilderMethodAutoTypes)
{
  std::vector<int> buf{0};
  auto tag = ucxx::Tag{0};

  auto tagBuilder = _ep->tagSendBuilder(buf.data(), sizeof(int), tag);
  static_assert(std::is_same<decltype(tagBuilder), ucxx::RequestTagBuilder>::value,
                "ep->tagSendBuilder() returns RequestTagBuilder");

  auto amBuilder = _ep->amSendBuilder(buf.data(), sizeof(int), UCS_MEMORY_TYPE_HOST);
  static_assert(std::is_same<decltype(amBuilder), ucxx::RequestAmBuilder>::value,
                "ep->amSendBuilder() returns RequestAmBuilder");

  auto memBuilder = _ep->memPutBuilder(buf.data(), sizeof(int), 0, nullptr);
  static_assert(std::is_same<decltype(memBuilder), ucxx::RequestMemBuilder>::value,
                "ep->memPutBuilder() returns RequestMemBuilder");

  auto streamBuilder = _ep->streamSendBuilder(buf.data(), sizeof(int));
  static_assert(std::is_same<decltype(streamBuilder), ucxx::RequestStreamBuilder>::value,
                "ep->streamSendBuilder() returns RequestStreamBuilder");

  std::vector<const void*> multiSendBuf{buf.data()};
  std::vector<size_t> multiSendLen{sizeof(int)};
  std::vector<int> isCUDA{0};
  auto tagMultiBuilder = _ep->tagMultiSendBuilder(multiSendBuf, multiSendLen, isCUDA, tag);
  static_assert(std::is_same<decltype(tagMultiBuilder), ucxx::RequestTagMultiBuilder>::value,
                "ep->tagMultiSendBuilder() returns RequestTagMultiBuilder");
}

}  // namespace
