/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include <ucp/api/ucp.h>

#include <ucxx/delayed_submission.h>
#include <ucxx/log.h>

namespace ucxx {

RequestDelayedSubmissionCollection::RequestDelayedSubmissionCollection(std::string name,
                                                                       const bool enabled)
  : BaseDelayedSubmissionCollection<
      std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType>>{std::move(name), enabled}
{
}

void RequestDelayedSubmissionCollection::scheduleLog(
  ItemIdType id, std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item)
{
  ucxx_trace_req("Registered %s [%lu]: %p", _name.c_str(), id, item.first.get());
}

void RequestDelayedSubmissionCollection::processItem(
  ItemIdType id, std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item)
{
  auto& req      = item.first;
  auto& callback = item.second;

  ucxx_trace_req("Submitting %s [%lu] callback: %p", _name.c_str(), id, req.get());

  if (callback) callback();
}

GenericDelayedSubmissionCollection::GenericDelayedSubmissionCollection(std::string name)
  : BaseDelayedSubmissionCollection<DelayedSubmissionCallbackType>{std::move(name), true}
{
}

void GenericDelayedSubmissionCollection::scheduleLog(ItemIdType id, DelayedSubmissionCallbackType)
{
  ucxx_trace_req("Registered %s [%lu]", _name.c_str(), id);
}

void GenericDelayedSubmissionCollection::processItem(ItemIdType id,
                                                     DelayedSubmissionCallbackType callback)
{
  ucxx_trace_req("Submitting %s [%lu] callback", _name.c_str(), id);

  if (callback) callback();
}

DelayedSubmissionCollection::DelayedSubmissionCollection(bool enableDelayedRequestSubmission)
  : _requests(RequestDelayedSubmissionCollection{"request", enableDelayedRequestSubmission}),
    _enableDelayedRequestSubmission(enableDelayedRequestSubmission)
{
}

bool DelayedSubmissionCollection::isDelayedRequestSubmissionEnabled() const
{
  return _enableDelayedRequestSubmission;
}

void DelayedSubmissionCollection::processPre()
{
  _requests.process();

  _genericPre.process();
}

void DelayedSubmissionCollection::processPost() { _genericPost.process(); }

void DelayedSubmissionCollection::registerRequest(std::shared_ptr<Request> request,
                                                  DelayedSubmissionCallbackType callback)
{
  std::ignore = _requests.schedule({request, callback});
}

ItemIdType DelayedSubmissionCollection::registerGenericPre(DelayedSubmissionCallbackType callback)
{
  return _genericPre.schedule(callback);
}

ItemIdType DelayedSubmissionCollection::registerGenericPost(DelayedSubmissionCallbackType callback)
{
  return _genericPost.schedule(callback);
}

void DelayedSubmissionCollection::cancelGenericPre(ItemIdType id) { _genericPre.cancel(id); }

void DelayedSubmissionCollection::cancelGenericPost(ItemIdType id) { _genericPost.cancel(id); }

}  // namespace ucxx
