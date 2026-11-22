/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <stdexcept>

#include <internal/constructors.h>
#include <ucxx/log.h>
#include <ucxx/tag_probe.h>

namespace ucxx {

TagRecvInfo::TagRecvInfo(const ucp_tag_recv_info_t& info)
  : senderTag(Tag(info.sender_tag)), length(info.length)
{
}

TagProbeInfo::TagProbeInfo(const ucp_tag_recv_info_t& info, ucp_tag_message_h handle)
  : _matched(true), _info(TagRecvInfo(info)), _handle(handle), _consumed(false)
{
}

namespace detail {

std::shared_ptr<TagProbeInfo> ConstructorFactory::createTagProbeInfo()
{
  return std::shared_ptr<TagProbeInfo>(new TagProbeInfo());
}

std::shared_ptr<TagProbeInfo> ConstructorFactory::createTagProbeInfo(
  const ucp_tag_recv_info_t& info, ucp_tag_message_h handle)
{
  return std::shared_ptr<TagProbeInfo>(new TagProbeInfo(info, handle));
}

}  // namespace detail

TagProbeInfo::~TagProbeInfo()
{
  // Check if handle is populated and has not been consumed
  if (_matched && _handle != nullptr && !_consumed) {
    ucxx_warn(
      "ucxx::TagProbeInfo::%s, destroying %p unconsumed message handle %p detected from tag 0x%lx "
      "with length %lu. ucxx::Worker::tagRecvWithHandle() must be called to consume the handle.",
      __func__,
      this,
      _handle,
      _info.value().senderTag,
      _info.value().length);
  }
}

bool TagProbeInfo::isMatched() const { return _matched; }

const TagRecvInfo& TagProbeInfo::getInfo() const
{
  if (!_matched) {
    throw std::runtime_error("TagProbeInfo::getInfo() called on unmatched message");
  }
  return _info.value();
}

ucp_tag_message_h TagProbeInfo::getHandle() const
{
  if (_handle == nullptr) {
    throw std::runtime_error("TagProbeInfo::getHandle() called with null handle");
  }
  if (_consumed) {
    throw std::runtime_error("TagProbeInfo::getHandle() called on consumed handle");
  }
  return _handle;
}

ucp_tag_message_h TagProbeInfo::releaseHandle() const
{
  if (_handle == nullptr) {
    throw std::runtime_error("TagProbeInfo::releaseHandle() called with null handle");
  }
  if (_consumed) {
    throw std::runtime_error("TagProbeInfo::releaseHandle() called on consumed handle");
  }
  consume();
  return _handle;
}

void TagProbeInfo::consume() const { _consumed = true; }

}  // namespace ucxx
