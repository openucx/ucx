/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include <ucxx/address_builder.h>
#include <ucxx/api.h>
#include <ucxx/endpoint_builder.h>
#include <ucxx/listener_builder.h>
#include <ucxx/memory_handle_builder.h>
#include <ucxx/remote_key_builder.h>
#include <ucxx/tag_probe_builder.h>

namespace {

static void listenerCallback(ucp_conn_request_h, void*) {}

static std::vector<std::string> TlsConfig{"^tcp", "^tcp,sm", "tcp", "tcp,sm", "all"};

class ContextBuilderCustomConfigTest : public testing::TestWithParam<std::string> {};

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

class ComponentBuilderTest : public ::testing::Test {
 protected:
  std::shared_ptr<ucxx::Context> _context{
    ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build()};
  std::shared_ptr<ucxx::Worker> _worker{ucxx::workerBuilder(_context).build()};
};

TEST(ContextBuilderTest, BasicBuilderWithAuto)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), ucxx::Context::defaultFeatureFlags);
}

TEST(ContextBuilderTest, BuilderWithFeatureFlags)
{
  uint64_t featureFlags = UCP_FEATURE_TAG | UCP_FEATURE_WAKEUP;
  auto context          = ucxx::contextBuilder(featureFlags).build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
}

TEST(ContextBuilderTest, BuilderWithConfigMap)
{
  auto context =
    ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).configMap({{"TLS", "tcp"}}).build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  auto config = context->getConfig();
  ASSERT_EQ(config["TLS"], "tcp");
}

TEST(ContextBuilderTest, BuilderMethodChainingConfigFirst)
{
  uint64_t featureFlags = UCP_FEATURE_RMA | UCP_FEATURE_STREAM;
  auto context          = ucxx::contextBuilder(featureFlags).configMap({{"TLS", "all"}}).build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
  auto config = context->getConfig();
  ASSERT_EQ(config["TLS"], "all");
}

TEST(ContextBuilderTest, BuilderMethodChainingFlagsFirst)
{
  uint64_t featureFlags = UCP_FEATURE_AM | UCP_FEATURE_RMA;
  auto context          = ucxx::contextBuilder(featureFlags).configMap({{"TLS", "tcp"}}).build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
  auto config = context->getConfig();
  ASSERT_EQ(config["TLS"], "tcp");
}

TEST(ContextBuilderTest, BuilderExplicitTypeSpecification)
{
  uint64_t featureFlags                  = UCP_FEATURE_TAG;
  std::shared_ptr<ucxx::Context> context = ucxx::contextBuilder(featureFlags);

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
}

TEST(ContextBuilderTest, BuilderAutoTypes)
{
  auto builder1 = ucxx::contextBuilder(UCP_FEATURE_TAG);
  static_assert(std::is_same<decltype(builder1), ucxx::ContextBuilder>::value,
                "auto without .build() is ContextBuilder");

  auto builder2 = ucxx::contextBuilder(UCP_FEATURE_TAG).configMap({{"TLS", "tcp"}});
  static_assert(std::is_same<decltype(builder2), ucxx::ContextBuilder>::value,
                "auto with .configMap() but without .build() is ContextBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::ContextBuilder, ucxx::Context>();
  assertConversionRequiresNonConstBuilder<ucxx::ContextBuilder, ucxx::Context>();

  auto context1 = builder1.build();
  static_assert(std::is_same<decltype(context1), std::shared_ptr<ucxx::Context>>::value,
                "Calling .build() on builder returns shared_ptr<Context>");

  std::shared_ptr<ucxx::Context> context2 = builder2;
  static_assert(std::is_same<decltype(context2), std::shared_ptr<ucxx::Context>>::value,
                "Implicit conversion with explicit type works");

  auto context3 = ucxx::contextBuilder(UCP_FEATURE_RMA).build();
  static_assert(std::is_same<decltype(context3), std::shared_ptr<ucxx::Context>>::value,
                "auto with .build() must be shared_ptr<Context>");

  auto context4 = ucxx::contextBuilder(UCP_FEATURE_TAG).build();
  static_assert(std::is_same<decltype(context4), std::shared_ptr<ucxx::Context>>::value,
                "auto with contextBuilder(flags).build() must be shared_ptr<Context>");

  auto context5 = ucxx::contextBuilder(UCP_FEATURE_RMA).configMap({{"TLS", "tcp"}}).build();
  static_assert(std::is_same<decltype(context5), std::shared_ptr<ucxx::Context>>::value,
                "auto with .configMap().build() must be shared_ptr<Context>");

  auto context6 = ucxx::contextBuilder(UCP_FEATURE_RMA).configMap({{"TLS", "all"}}).build();
  static_assert(std::is_same<decltype(context6), std::shared_ptr<ucxx::Context>>::value,
                "auto with full chain and .build() must be shared_ptr<Context>");

  ASSERT_TRUE(context1 != nullptr);
  ASSERT_TRUE(context2 != nullptr);
  ASSERT_TRUE(context3 != nullptr);
  ASSERT_TRUE(context4 != nullptr);
  ASSERT_TRUE(context5 != nullptr);
  ASSERT_TRUE(context6 != nullptr);
}

TEST(ContextBuilderTest, BuilderImplicitConversion)
{
  std::shared_ptr<ucxx::Context> context = ucxx::contextBuilder(UCP_FEATURE_RMA);

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), UCP_FEATURE_RMA);
}

TEST(ContextBuilderTest, BuilderSingleConstructionPerSet)
{
  auto builder = ucxx::contextBuilder(UCP_FEATURE_TAG);
  auto context = builder.build();

  ASSERT_TRUE(context != nullptr);
  ASSERT_TRUE(context->getHandle() != nullptr);
  ASSERT_EQ(context->getFeatureFlags(), UCP_FEATURE_TAG);

  // Same builder can be used multiple times, but creates separate contexts
  auto context2 = builder.build();
  ASSERT_TRUE(context2 != nullptr);
  ASSERT_NE(context->getHandle(), context2->getHandle());  // Different contexts
}

TEST(ContextBuilderTest, BuilderDifferentInstances)
{
  auto context1 = ucxx::contextBuilder(UCP_FEATURE_TAG).build();
  auto context2 = ucxx::contextBuilder(UCP_FEATURE_TAG).build();

  // Different builder chains should create different contexts
  ASSERT_TRUE(context1 != nullptr);
  ASSERT_TRUE(context2 != nullptr);
  ASSERT_NE(context1->getHandle(), context2->getHandle());
}

TEST(ContextBuilderTest, BuilderContextIsValid)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();

  auto worker = context->workerBuilder().build();
  ASSERT_TRUE(worker != nullptr);
}

TEST_P(ContextBuilderCustomConfigTest, BuilderTLS)
{
  auto tls                           = GetParam();
  static constexpr auto featureFlags = ucxx::Context::defaultFeatureFlags;
  auto context = ucxx::contextBuilder(featureFlags).configMap({{"TLS", tls}}).build();

  auto configMapOut = context->getConfig();
  ASSERT_GT(configMapOut.size(), 1u);
  ASSERT_NE(configMapOut.find("TLS"), configMapOut.end());
  ASSERT_EQ(configMapOut["TLS"], tls);

  ASSERT_EQ(context->getFeatureFlags(), featureFlags);
}

TEST(WorkerBuilderTest, BasicBuilderWithAuto)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto worker  = ucxx::workerBuilder(context).build();

  ASSERT_TRUE(worker != nullptr);
  ASSERT_TRUE(worker->getHandle() != nullptr);
  ASSERT_FALSE(worker->isDelayedRequestSubmissionEnabled());
  ASSERT_FALSE(worker->isFutureEnabled());
}

TEST(WorkerBuilderTest, BuilderWithOptions)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto worker  = ucxx::workerBuilder(context).delayedSubmission(true).pythonFuture(true).build();

  ASSERT_TRUE(worker != nullptr);
  ASSERT_TRUE(worker->getHandle() != nullptr);
  ASSERT_TRUE(worker->isDelayedRequestSubmissionEnabled());
  ASSERT_TRUE(worker->isFutureEnabled());
}

TEST(WorkerBuilderTest, BuilderMethodChainingOrder1)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto worker  = ucxx::workerBuilder(context).delayedSubmission(true).pythonFuture(false).build();

  ASSERT_TRUE(worker != nullptr);
  ASSERT_TRUE(worker->getHandle() != nullptr);
  ASSERT_TRUE(worker->isDelayedRequestSubmissionEnabled());
  ASSERT_FALSE(worker->isFutureEnabled());
}

TEST(WorkerBuilderTest, BuilderMethodChainingOrder2)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto worker  = ucxx::workerBuilder(context).pythonFuture(true).delayedSubmission(false).build();

  ASSERT_TRUE(worker != nullptr);
  ASSERT_TRUE(worker->getHandle() != nullptr);
  ASSERT_FALSE(worker->isDelayedRequestSubmissionEnabled());
  ASSERT_TRUE(worker->isFutureEnabled());
}

TEST(WorkerBuilderTest, BuilderExplicitTypeSpecification)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  std::shared_ptr<ucxx::Worker> worker = ucxx::workerBuilder(context);

  ASSERT_TRUE(worker != nullptr);
  ASSERT_TRUE(worker->getHandle() != nullptr);
  ASSERT_FALSE(worker->isDelayedRequestSubmissionEnabled());
  ASSERT_FALSE(worker->isFutureEnabled());
}

TEST(WorkerBuilderTest, BuilderAutoTypes)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();

  auto builder1 = ucxx::workerBuilder(context);
  static_assert(std::is_same<decltype(builder1), ucxx::WorkerBuilder>::value,
                "auto without .build() is WorkerBuilder");

  auto builder2 = ucxx::workerBuilder(context).delayedSubmission(true).pythonFuture(true);
  static_assert(std::is_same<decltype(builder2), ucxx::WorkerBuilder>::value,
                "auto with config methods but without .build() is WorkerBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::WorkerBuilder, ucxx::Worker>();
  assertConversionRequiresNonConstBuilder<ucxx::WorkerBuilder, ucxx::Worker>();

  auto worker1 = builder1.build();
  static_assert(std::is_same<decltype(worker1), std::shared_ptr<ucxx::Worker>>::value,
                "Calling .build() on builder returns shared_ptr<Worker>");

  std::shared_ptr<ucxx::Worker> worker2 = builder2;
  static_assert(std::is_same<decltype(worker2), std::shared_ptr<ucxx::Worker>>::value,
                "Implicit conversion with explicit type works");

  auto worker3 = ucxx::workerBuilder(context).build();
  static_assert(std::is_same<decltype(worker3), std::shared_ptr<ucxx::Worker>>::value,
                "auto with .build() must be shared_ptr<Worker>");

  auto worker4 = ucxx::workerBuilder(context).delayedSubmission(true).pythonFuture(true).build();
  static_assert(std::is_same<decltype(worker4), std::shared_ptr<ucxx::Worker>>::value,
                "auto with config methods and .build() must be shared_ptr<Worker>");

  ASSERT_TRUE(worker1 != nullptr);
  ASSERT_TRUE(worker2 != nullptr);
  ASSERT_TRUE(worker3 != nullptr);
  ASSERT_TRUE(worker4 != nullptr);
}

TEST(WorkerBuilderTest, BuilderImplicitConversion)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  std::shared_ptr<ucxx::Worker> worker = ucxx::workerBuilder(context);

  ASSERT_TRUE(worker != nullptr);
  ASSERT_TRUE(worker->getHandle() != nullptr);
}

TEST(WorkerBuilderTest, BuilderSingleConstructionPerSet)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto builder = ucxx::workerBuilder(context);
  auto worker  = builder.build();

  ASSERT_TRUE(worker != nullptr);
  ASSERT_TRUE(worker->getHandle() != nullptr);

  // Same builder can be used multiple times, but creates separate workers
  auto worker2 = builder.build();
  ASSERT_TRUE(worker2 != nullptr);
  ASSERT_NE(worker->getHandle(), worker2->getHandle());
}

TEST(WorkerBuilderTest, BuilderDifferentInstances)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto worker1 = ucxx::workerBuilder(context).build();
  auto worker2 = ucxx::workerBuilder(context).build();
  ASSERT_TRUE(worker1 != nullptr);
  ASSERT_TRUE(worker2 != nullptr);
  ASSERT_NE(worker1->getHandle(), worker2->getHandle());
}

TEST(WorkerBuilderTest, RequestAttributesDefaultDisabled)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto worker  = ucxx::workerBuilder(context).build();

  ASSERT_TRUE(worker != nullptr);
  ASSERT_FALSE(worker->isRequestAttributesEnabled());
}

TEST(WorkerBuilderTest, RequestAttributesEnabled)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto worker  = ucxx::workerBuilder(context).requestAttributes(true).build();

  ASSERT_TRUE(worker != nullptr);
  ASSERT_TRUE(worker->isRequestAttributesEnabled());
  ASSERT_FALSE(worker->isDelayedRequestSubmissionEnabled());
  ASSERT_FALSE(worker->isFutureEnabled());
}

TEST(WorkerBuilderTest, RequestAttributesExplicitDisable)
{
  auto context = ucxx::contextBuilder(ucxx::Context::defaultFeatureFlags).build();
  auto worker  = ucxx::workerBuilder(context).requestAttributes(false).build();

  ASSERT_TRUE(worker != nullptr);
  ASSERT_FALSE(worker->isRequestAttributesEnabled());
}

TEST_F(ComponentBuilderTest, ContextChildBuilders)
{
  auto workerBuilder = _context->workerBuilder().delayedSubmission(true).pythonFuture(true);
  static_assert(std::is_same<decltype(workerBuilder), ucxx::WorkerBuilder>::value,
                "context->workerBuilder() returns WorkerBuilder");
  auto worker = workerBuilder.build();
  ASSERT_TRUE(worker != nullptr);
  ASSERT_TRUE(worker->isDelayedRequestSubmissionEnabled());
  ASSERT_TRUE(worker->isFutureEnabled());

  auto memoryHandleBuilder = _context->memoryHandleBuilder(128).memoryType(UCS_MEMORY_TYPE_HOST);
  static_assert(std::is_same<decltype(memoryHandleBuilder), ucxx::MemoryHandleBuilder>::value,
                "context->memoryHandleBuilder() returns MemoryHandleBuilder");
  auto memoryHandle = memoryHandleBuilder.build();
  ASSERT_TRUE(memoryHandle != nullptr);
  ASSERT_EQ(memoryHandle->getMemoryType(), UCS_MEMORY_TYPE_HOST);
}

TEST_F(ComponentBuilderTest, WorkerChildBuilders)
{
  auto addressBuilder = _worker->addressBuilder();
  static_assert(std::is_same<decltype(addressBuilder), ucxx::AddressBuilder>::value,
                "worker->addressBuilder() returns AddressBuilder");
  auto address = addressBuilder.build();
  ASSERT_TRUE(address != nullptr);
  ASSERT_TRUE(address->getHandle() != nullptr);

  auto endpointBuilder = _worker->endpointBuilder(address).endpointErrorHandling(false);
  static_assert(std::is_same<decltype(endpointBuilder), ucxx::EndpointBuilder>::value,
                "worker->endpointBuilder(address) returns EndpointBuilder");
  auto endpoint = endpointBuilder.build();
  ASSERT_TRUE(endpoint != nullptr);

  auto hostnameEndpointBuilder = _worker->endpointBuilder("127.0.0.1", 12345);
  static_assert(std::is_same<decltype(hostnameEndpointBuilder), ucxx::EndpointBuilder>::value,
                "worker->endpointBuilder(hostname, port) returns EndpointBuilder");

  auto globalAddressEndpointBuilder = ucxx::endpointBuilder(_worker, address);
  static_assert(std::is_same<decltype(globalAddressEndpointBuilder), ucxx::EndpointBuilder>::value,
                "endpointBuilder(worker, address) returns EndpointBuilder");

  auto globalHostnameEndpointBuilder = ucxx::endpointBuilder(_worker, "127.0.0.1", 12345);
  static_assert(std::is_same<decltype(globalHostnameEndpointBuilder), ucxx::EndpointBuilder>::value,
                "endpointBuilder(worker, hostname, port) returns EndpointBuilder");

  auto listenerBuilder = _worker->listenerBuilder(0, listenerCallback, nullptr);
  static_assert(std::is_same<decltype(listenerBuilder), ucxx::ListenerBuilder>::value,
                "worker->listenerBuilder() returns ListenerBuilder");
  auto listener = listenerBuilder.build();
  ASSERT_TRUE(listener != nullptr);

  auto connRequestEndpointBuilder = listener->endpointBuilder(nullptr);
  static_assert(std::is_same<decltype(connRequestEndpointBuilder), ucxx::EndpointBuilder>::value,
                "listener->endpointBuilder(connRequest) returns EndpointBuilder");

  auto globalConnRequestEndpointBuilder = ucxx::endpointBuilder(listener, nullptr);
  static_assert(
    std::is_same<decltype(globalConnRequestEndpointBuilder), ucxx::EndpointBuilder>::value,
    "endpointBuilder(listener, connRequest) returns EndpointBuilder");
}

TEST_F(ComponentBuilderTest, AddressBuilderFromWorkerAndString)
{
  auto builder = ucxx::AddressBuilder(_worker);
  static_assert(std::is_same<decltype(builder), ucxx::AddressBuilder>::value,
                "AddressBuilder constructor returns AddressBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::AddressBuilder, ucxx::Address>();
  assertConversionRequiresNonConstBuilder<ucxx::AddressBuilder, ucxx::Address>();

  auto address = builder.build();
  static_assert(std::is_same<decltype(address), std::shared_ptr<ucxx::Address>>::value,
                "AddressBuilder::build returns shared_ptr<Address>");
  ASSERT_TRUE(address != nullptr);
  ASSERT_TRUE(address->getHandle() != nullptr);
  ASSERT_GT(address->getLength(), 0u);

  std::string addressString{address->getStringView()};
  std::shared_ptr<ucxx::Address> copiedAddress = ucxx::AddressBuilder(addressString);
  ASSERT_TRUE(copiedAddress != nullptr);
  ASSERT_TRUE(copiedAddress->getHandle() != nullptr);
  ASSERT_EQ(copiedAddress->getStringView(), addressString);
}

TEST_F(ComponentBuilderTest, EndpointBuilderFromWorkerAddress)
{
  auto builder =
    ucxx::endpointBuilder(_worker, _worker->addressBuilder().build()).endpointErrorHandling(false);
  static_assert(std::is_same<decltype(builder), ucxx::EndpointBuilder>::value,
                "endpointBuilder returns EndpointBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::EndpointBuilder, ucxx::Endpoint>();
  assertConversionRequiresNonConstBuilder<ucxx::EndpointBuilder, ucxx::Endpoint>();

  auto endpoint = builder.build();
  static_assert(std::is_same<decltype(endpoint), std::shared_ptr<ucxx::Endpoint>>::value,
                "EndpointBuilder::build returns shared_ptr<Endpoint>");
  ASSERT_TRUE(endpoint != nullptr);
  ASSERT_TRUE(endpoint->getHandle() != nullptr);
  ASSERT_TRUE(endpoint->isAlive());
}

TEST_F(ComponentBuilderTest, ListenerBuilder)
{
  auto builder = ucxx::ListenerBuilder(_worker, 0, listenerCallback, nullptr);
  static_assert(std::is_same<decltype(builder), ucxx::ListenerBuilder>::value,
                "ListenerBuilder constructor returns ListenerBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::ListenerBuilder, ucxx::Listener>();
  assertConversionRequiresNonConstBuilder<ucxx::ListenerBuilder, ucxx::Listener>();

  auto listener = builder.build();
  static_assert(std::is_same<decltype(listener), std::shared_ptr<ucxx::Listener>>::value,
                "ListenerBuilder::build returns shared_ptr<Listener>");
  ASSERT_TRUE(listener != nullptr);
  ASSERT_TRUE(listener->getHandle() != nullptr);
  ASSERT_GT(listener->getPort(), 0u);
}

TEST_F(ComponentBuilderTest, ListenerBuilderHost)
{
  auto builder = ucxx::ListenerBuilder(_worker, 0, listenerCallback, nullptr);

  auto listener = builder.host("127.0.0.1").build();
  ASSERT_TRUE(listener != nullptr);
  ASSERT_TRUE(listener->getHandle() != nullptr);
  ASSERT_EQ(listener->getIp(), "127.0.0.1");
  ASSERT_GT(listener->getPort(), 0u);
}

TEST_F(ComponentBuilderTest, MemoryHandleBuilder)
{
  std::vector<char> buffer(128);

  auto builder = ucxx::MemoryHandleBuilder(_context, buffer.size())
                   .buffer(buffer.data())
                   .memoryType(UCS_MEMORY_TYPE_HOST);
  static_assert(std::is_same<decltype(builder), ucxx::MemoryHandleBuilder>::value,
                "MemoryHandleBuilder constructor returns MemoryHandleBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::MemoryHandleBuilder, ucxx::MemoryHandle>();
  assertConversionRequiresNonConstBuilder<ucxx::MemoryHandleBuilder, ucxx::MemoryHandle>();

  auto memoryHandle = builder.build();
  static_assert(std::is_same<decltype(memoryHandle), std::shared_ptr<ucxx::MemoryHandle>>::value,
                "MemoryHandleBuilder::build returns shared_ptr<MemoryHandle>");
  ASSERT_TRUE(memoryHandle != nullptr);
  ASSERT_TRUE(memoryHandle->getHandle() != nullptr);
  ASSERT_EQ(memoryHandle->getSize(), buffer.size());
  ASSERT_EQ(memoryHandle->getMemoryType(), UCS_MEMORY_TYPE_HOST);
}

TEST_F(ComponentBuilderTest, RemoteKeyBuilderFromMemoryHandleAndSerialized)
{
  auto memoryHandle = ucxx::MemoryHandleBuilder(_context, 128).build();

  auto builder = ucxx::RemoteKeyBuilder(memoryHandle);
  static_assert(std::is_same<decltype(builder), ucxx::RemoteKeyBuilder>::value,
                "RemoteKeyBuilder constructor returns RemoteKeyBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::RemoteKeyBuilder, ucxx::RemoteKey>();
  assertConversionRequiresNonConstBuilder<ucxx::RemoteKeyBuilder, ucxx::RemoteKey>();

  auto localRemoteKey = builder.build();
  static_assert(std::is_same<decltype(localRemoteKey), std::shared_ptr<ucxx::RemoteKey>>::value,
                "RemoteKeyBuilder::build returns shared_ptr<RemoteKey>");
  ASSERT_TRUE(localRemoteKey != nullptr);
  ASSERT_EQ(localRemoteKey->getSize(), memoryHandle->getSize());

  auto endpoint = ucxx::endpointBuilder(_worker, _worker->addressBuilder().build()).build();
  std::shared_ptr<ucxx::RemoteKey> unpackedRemoteKey =
    ucxx::RemoteKeyBuilder(endpoint, localRemoteKey->serialize());
  ASSERT_TRUE(unpackedRemoteKey != nullptr);
  ASSERT_TRUE(unpackedRemoteKey->getHandle() != nullptr);
  ASSERT_EQ(unpackedRemoteKey->getSize(), localRemoteKey->getSize());
}

TEST_F(ComponentBuilderTest, RemoteKeyChildBuilders)
{
  auto memoryHandle = _context->memoryHandleBuilder(128).build();
  auto builder      = memoryHandle->remoteKeyBuilder();
  static_assert(std::is_same<decltype(builder), ucxx::RemoteKeyBuilder>::value,
                "memoryHandle->remoteKeyBuilder() returns RemoteKeyBuilder");

  auto localRemoteKey = builder.build();
  ASSERT_TRUE(localRemoteKey != nullptr);

  auto endpoint      = _worker->endpointBuilder(_worker->addressBuilder().build()).build();
  auto unpackBuilder = endpoint->remoteKeyBuilder(localRemoteKey->serialize());
  static_assert(std::is_same<decltype(unpackBuilder), ucxx::RemoteKeyBuilder>::value,
                "endpoint->remoteKeyBuilder() returns RemoteKeyBuilder");

  auto unpackedRemoteKey = unpackBuilder.build();
  ASSERT_TRUE(unpackedRemoteKey != nullptr);
  ASSERT_TRUE(unpackedRemoteKey->getHandle() != nullptr);
}

TEST(ComponentBuilderStandaloneTest, TagProbeInfoBuilder)
{
  auto builder = ucxx::TagProbeInfoBuilder();
  static_assert(std::is_same<decltype(builder), ucxx::TagProbeInfoBuilder>::value,
                "TagProbeInfoBuilder constructor returns TagProbeInfoBuilder");
  assertBuildRequiresNonConstBuilder<ucxx::TagProbeInfoBuilder, ucxx::TagProbeInfo>();
  assertConversionRequiresNonConstBuilder<ucxx::TagProbeInfoBuilder, ucxx::TagProbeInfo>();

  auto unmatched = builder.build();
  static_assert(std::is_same<decltype(unmatched), std::shared_ptr<ucxx::TagProbeInfo>>::value,
                "TagProbeInfoBuilder::build returns shared_ptr<TagProbeInfo>");
  ASSERT_TRUE(unmatched != nullptr);
  ASSERT_FALSE(unmatched->isMatched());

  ucp_tag_recv_info_t info{};
  info.sender_tag = 7;
  info.length     = 64;

  std::shared_ptr<ucxx::TagProbeInfo> matched = ucxx::TagProbeInfoBuilder(info, nullptr);
  ASSERT_TRUE(matched != nullptr);
  ASSERT_TRUE(matched->isMatched());
  ASSERT_EQ(matched->getInfo().senderTag, ucxx::Tag{7});
  ASSERT_EQ(matched->getInfo().length, 64u);
}

INSTANTIATE_TEST_SUITE_P(TLS, ContextBuilderCustomConfigTest, testing::ValuesIn(TlsConfig));

}  // namespace
