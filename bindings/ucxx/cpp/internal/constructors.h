/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include <ucxx/buffer.h>
#include <ucxx/component.h>
#include <ucxx/request_data.h>
#include <ucxx/typedefs.h>

namespace ucxx {

class Address;
class Context;
class Endpoint;
class Listener;
class MemoryHandle;
class RemoteKey;
class RequestAm;
class RequestEndpointClose;
class RequestFlush;
class RequestMem;
class RequestStream;
class RequestTag;
class RequestTagMulti;
class TagProbeInfo;
class Worker;

namespace detail {

/**
 * @brief Internal factory for constructing UCXX objects with non-public constructors.
 *
 * This factory centralizes privileged construction used by builders and internal operations.
 * It is an implementation detail and is not installed as part of the UCXX public API.
 */
class ConstructorFactory {
 public:
  [[nodiscard]] static std::shared_ptr<Address> createAddressFromWorker(
    std::shared_ptr<Worker> worker);
  [[nodiscard]] static std::shared_ptr<Address> createAddressFromString(
    std::string_view addressString);
  [[nodiscard]] static std::shared_ptr<Listener> createListener(
    std::shared_ptr<Worker> worker,
    std::string host,
    uint16_t port,
    ucp_listener_conn_callback_t callback,
    void* callbackArgs);
  [[nodiscard]] static std::shared_ptr<RemoteKey> createRemoteKeyFromMemoryHandle(
    std::shared_ptr<MemoryHandle> memoryHandle);
  [[nodiscard]] static std::shared_ptr<RemoteKey> createRemoteKeyFromSerialized(
    std::shared_ptr<Endpoint> endpoint, SerializedRemoteKey serializedRemoteKey);
  [[nodiscard]] static std::shared_ptr<TagProbeInfo> createTagProbeInfo();
  [[nodiscard]] static std::shared_ptr<TagProbeInfo> createTagProbeInfo(
    const ucp_tag_recv_info_t& info, ucp_tag_message_h handle);
  [[nodiscard]] static std::shared_ptr<Context> createContext(ConfigMap ucxConfig,
                                                              uint64_t featureFlags);
  [[nodiscard]] static std::shared_ptr<Endpoint> createEndpointFromHostname(
    std::shared_ptr<Worker> worker,
    std::string ipAddress,
    uint16_t port,
    bool endpointErrorHandling);
  [[nodiscard]] static std::shared_ptr<Endpoint> createEndpointFromConnRequest(
    std::shared_ptr<Listener> listener, ucp_conn_request_h connRequest, bool endpointErrorHandling);
  [[nodiscard]] static std::shared_ptr<Endpoint> createEndpointFromWorkerAddress(
    std::shared_ptr<Worker> worker, std::shared_ptr<Address> address, bool endpointErrorHandling);
  [[nodiscard]] static std::shared_ptr<Worker> createWorker(std::shared_ptr<Context> context,
                                                            bool enableDelayedSubmission,
                                                            bool enableFuture);
  [[nodiscard]] static std::shared_ptr<MemoryHandle> createMemoryHandle(
    std::shared_ptr<Context> context, size_t size, void* buffer, ucs_memory_type_t memoryType);
  [[nodiscard]] static std::shared_ptr<RequestAm> createRequestAm(
    std::shared_ptr<Endpoint> endpoint,
    std::variant<data::AmSend, data::AmReceive> requestData,
    bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);
  [[nodiscard]] static std::shared_ptr<RequestEndpointClose> createRequestEndpointClose(
    std::shared_ptr<Endpoint> endpoint,
    data::EndpointClose requestData,
    bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);
  [[nodiscard]] static std::shared_ptr<RequestFlush> createRequestFlush(
    std::shared_ptr<Component> endpointOrWorker,
    data::Flush requestData,
    bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);
  [[nodiscard]] static std::shared_ptr<RequestStream> createRequestStream(
    std::shared_ptr<Endpoint> endpoint,
    std::variant<data::StreamSend, data::StreamReceive> requestData,
    bool enablePythonFuture);
  [[nodiscard]] static std::shared_ptr<RequestTag> createRequestTag(
    std::shared_ptr<Component> endpointOrWorker,
    std::variant<data::TagSend, data::TagReceive, data::TagReceiveWithHandle> requestData,
    bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);
  [[nodiscard]] static std::shared_ptr<RequestMem> createRequestMem(
    std::shared_ptr<Endpoint> endpoint,
    std::variant<data::MemPut, data::MemGet> requestData,
    bool enablePythonFuture,
    RequestCallbackUserFunction callbackFunction,
    RequestCallbackUserData callbackData);
  [[nodiscard]] static std::shared_ptr<RequestTagMulti> createRequestTagMulti(
    std::shared_ptr<Endpoint> endpoint,
    std::variant<data::TagMultiSend, data::TagMultiReceive> requestData,
    bool enablePythonFuture);
};

}  // namespace detail

}  // namespace ucxx
