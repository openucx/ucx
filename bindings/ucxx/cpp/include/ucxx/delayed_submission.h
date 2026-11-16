/**
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

#include <ucp/api/ucp.h>
#include <ucs/memory/memory_type.h>

#include <ucxx/log.h>
#include <ucxx/request_data.h>

namespace ucxx {

/**
 * @brief A user-defined function to execute as part of delayed submission callback.
 *
 * A user-defined function to execute in the scope of a `ucxx::DelayedSubmission`, allowing
 * execution of custom code upon the completion of the delayed submission.
 */
typedef std::function<void()> DelayedSubmissionCallbackType;

/**
 * @brief Type for identifying items in `DelayedSubmissionCollection`.
 *
 * A 64-bit unsigned integer used to uniquely identify items in `DelayedSubmissionCollection`,
 * allowing for tracking and cancellation of specific items.
 */
typedef uint64_t ItemIdType;

/**
 * @brief Base type for a collection of delayed submissions.
 *
 * Base type for a collection of delayed submission. Delayed submissions may have different
 * purposes and this class encapsulates generic data for all derived types.
 */
template <typename T>
class BaseDelayedSubmissionCollection {
 protected:
  std::string _name{"undefined"};  ///< The human-readable name of the collection, used for logging
  bool _enabled{true};    ///< Whether the resource required to process the collection is enabled.
  ItemIdType _itemId{0};  ///< The item ID counter, used to allow cancelation.
  std::optional<ItemIdType> _processing{
    std::nullopt};  ///< The ID of the item being processed, if any.
  std::deque<std::pair<ItemIdType, T>> _collection{};  ///< The collection.
  std::set<ItemIdType> _canceled{};                    ///< IDs of canceled items.
  std::mutex _mutex{};  ///< Mutex to provide access to `_collection`.

  /**
   * @brief Log message during `schedule()`.
   *
   * Log a specialized message while `schedule()` is being executed.
   *
   * @param[in] id        the ID of the scheduled item, as returned by `schedule()`.
   * @param[in] item      the callback that was passed as argument to `schedule()`.
   */
  virtual void scheduleLog(ItemIdType id, T item) = 0;

  /**
   * @brief Process a single item during `process()`.
   *
   * Method called by `process()` to process a single item of the collection.
   *
   * @param[in] id        the ID of the scheduled item, as returned by `schedule()`.
   * @param[in] item      the callback that was passed as argument to `schedule()` when
   *                      the first registered.
   */
  virtual void processItem(ItemIdType id, T item) = 0;

 public:
  /**
   * @brief Constructor for a thread-safe delayed submission collection.
   *
   * Construct a thread-safe delayed submission collection. A delayed submission collection
   * provides two operations: schedule and process. The `schedule()` method will push an
   * operation into the collection, whereas the `process()` will invoke all callbacks that
   * were previously pushed into the collection and clear the collection.
   *
   * @param[in] name    human-readable name of the collection, used for logging.
   * @param[in] enabled whether the resource is enabled, if `false` an exception is raised
   *                    when attempting to schedule a callable. Disabled instances of this
   *                    class should only be used to provide a consistent interface among
   *                    implementations.
   */
  explicit BaseDelayedSubmissionCollection(std::string name, const bool enabled)
    : _name{std::move(name)}, _enabled{enabled}
  {
  }

  BaseDelayedSubmissionCollection()                                                  = delete;
  BaseDelayedSubmissionCollection(const BaseDelayedSubmissionCollection&)            = delete;
  BaseDelayedSubmissionCollection& operator=(BaseDelayedSubmissionCollection const&) = delete;
  BaseDelayedSubmissionCollection(BaseDelayedSubmissionCollection&& o)               = delete;
  BaseDelayedSubmissionCollection& operator=(BaseDelayedSubmissionCollection&& o)    = delete;

  /**
   * @brief Register a callable or complex-type for delayed submission.
   *
   * Register a simple callback, or complex-type with a callback (requires specialization),
   * for delayed submission that will be executed when the request is in fact submitted when
   * `process()` is called.
   *
   * Raise an exception if `false` was specified as the `enabled` argument to the constructor.
   *
   * @throws std::runtime_error if `_enabled` is `false`.
   *
   * @param[in] item            the callback that will be executed by `process()` when the
   *                            operation is submitted.
   *
   * @returns the ID of the scheduled item which can be used cancelation requests.
   */
  [[nodiscard]] virtual ItemIdType schedule(T item)
  {
    if (!_enabled) throw std::runtime_error("Resource is disabled.");

    ItemIdType id;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      id = _itemId++;
      _collection.emplace_back(id, item);
    }
    scheduleLog(id, item);

    return id;
  }

  /**
   * @brief Process all pending callbacks.
   *
   * Process all pending generic. Generic callbacks are deemed completed when their
   * execution completes.
   */
  void process()
  {
    // Process only those that were already inserted to prevent from never
    // returning if `_collection` grows indefinitely.
    size_t toProcess = 0;
    {
      std::lock_guard<std::mutex> lock(_mutex);
      toProcess = _collection.size();
    }

    for (size_t i = 0; i < toProcess; ++i) {
      std::pair<ItemIdType, T> item;
      {
        std::lock_guard<std::mutex> lock(_mutex);
        item = std::move(_collection.front());
        _collection.pop_front();
        if (_canceled.erase(item.first)) continue;
        _processing = std::optional<ItemIdType>{item.first};
      }

      processItem(item.first, item.second);
    }

    {
      // Clear the value of `_processing` as no more requests will be processed.
      std::lock_guard<std::mutex> lock(_mutex);
      _processing = std::nullopt;
    }
  }

  /**
   * @brief Cancel a pending callback.
   *
   * Cancel a pending callback and thus do not execute it, unless the execution has
   * already begun, in which case cancelation cannot be done.
   *
   * @throws std::runtime_error if the item is being processed and canceling is not
   *                            possible anymore.
   *
   * @param[in] id        the ID of the scheduled item, as returned by `schedule()`.
   */
  void cancel(ItemIdType id)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_processing.has_value() && _processing.value() == id)
      throw std::runtime_error("Cannot cancel, item is being processed.");

    _canceled.insert(id);
    ucxx_trace_req("Canceled item: %lu", id);
  }
};

/**
 * @brief A collection of delayed request submissions.
 *
 * A collection of delayed submissions used specifically for message transfer
 * `ucxx::Request` submissions.
 */
class RequestDelayedSubmissionCollection
  : public BaseDelayedSubmissionCollection<
      std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType>> {
 protected:
  void scheduleLog(
    ItemIdType id,
    std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item) override;

  void processItem(
    ItemIdType id,
    std::pair<std::shared_ptr<Request>, DelayedSubmissionCallbackType> item) override;

 public:
  /**
   * @brief Constructor of a collection of delayed request submissions.
   *
   * Construct a collection of delayed submissions used specifically for message transfer
   * `ucxx::Request` submissions.
   *
   * @param[in] name    the human-readable name of the type of delayed submission for
   *                    debugging purposes.
   * @param[in] enabled whether delayed request submissions should be enabled.
   */
  explicit RequestDelayedSubmissionCollection(std::string name, const bool enabled);
};

/**
 * @brief A collection of delayed submissions of generic callbacks.
 *
 * A collection of delayed submissions used specifically for execution of generic callbacks
 * at pre-defined stages of the progress loop.
 */
class GenericDelayedSubmissionCollection
  : public BaseDelayedSubmissionCollection<DelayedSubmissionCallbackType> {
 protected:
  void scheduleLog(ItemIdType id, DelayedSubmissionCallbackType item) override;

  void processItem(ItemIdType id, DelayedSubmissionCallbackType callback) override;

 public:
  /**
   * @brief Constructor of a collection of delayed submissions of generic callbacks.
   *
   * Construct a collection of delayed submissions used specifically for execution of
   * generic callbacks at pre-defined stages of the progress loop.
   *
   * @param[in] name    the human-readable name of the type of delayed submission for
   *                    debugging purposes.
   */
  explicit GenericDelayedSubmissionCollection(std::string name);
};

/**
 * @brief A collection of delayed submissions of multiple types.
 *
 * A collection of delayed submissions of multiple types used by the owner to manage each
 * of the delayed submission types via specialized methods.
 */
class DelayedSubmissionCollection {
 private:
  GenericDelayedSubmissionCollection _genericPre{
    "generic pre"};  ///< The collection of all known generic pre-progress operations.
  GenericDelayedSubmissionCollection _genericPost{
    "generic post"};  ///< The collection of all known generic post-progress operations.
  RequestDelayedSubmissionCollection _requests{
    "request", false};  ///< The collection of all known delayed request submission operations.
  bool _enableDelayedRequestSubmission{false};

 public:
  /**
   * @brief Default delayed submission collection constructor.
   *
   * Construct an empty collection of delayed submissions. Despite its name, a delayed
   * submission registration may be processed right after registration, thus effectively
   * making it an immediate submission.
   *
   * @param[in] enableDelayedRequestSubmission  whether request submission should be
   *                                            enabled, if `false`, only generic
   *                                            callbacks are enabled.
   */
  explicit DelayedSubmissionCollection(bool enableDelayedRequestSubmission = false);

  DelayedSubmissionCollection()                                              = delete;
  DelayedSubmissionCollection(const DelayedSubmissionCollection&)            = delete;
  DelayedSubmissionCollection& operator=(DelayedSubmissionCollection const&) = delete;
  DelayedSubmissionCollection(DelayedSubmissionCollection&& o)               = delete;
  DelayedSubmissionCollection& operator=(DelayedSubmissionCollection&& o)    = delete;

  /**
   * @brief Process pending delayed request submission and generic-pre callback operations.
   *
   * Process all pending delayed request submissions and generic callbacks. Generic
   * callbacks are deemed completed when their execution completes. On the other hand, the
   * execution of the delayed request submission callbacks does not imply completion of the
   * operation, only that it has been submitted. The completion of each delayed request
   * submission is handled externally by the implementation of the object being processed,
   * for example by checking the result of `ucxx::Request::isCompleted()`.
   *
   * Generic callbacks may be used to to pass information between threads on the subject
   * that requests have been in fact processed, therefore, requests are processed first,
   * then generic callbacks are.
   */
  void processPre();

  /**
   * @brief Process all pending generic-post callback operations.
   *
   * Process all pending generic-post callbacks. Generic callbacks are deemed completed when
   * their execution completes.
   */
  void processPost();

  /**
   * @brief Register a request for delayed submission.
   *
   * Register a request for delayed submission with a callback that will be executed when
   * the request is in fact submitted when `processPre()` is called.
   *
   * @throws std::runtime_error if delayed request submission was disabled at construction.
   *
   * @param[in] request   the request to which the callback belongs, ensuring it remains
   *                      alive until the callback is invoked.
   * @param[in] callback  the callback that will be executed by `processPre()` when the
   *                      operation is submitted.
   */
  void registerRequest(std::shared_ptr<Request> request, DelayedSubmissionCallbackType callback);

  /**
   * @brief Register a generic callback to execute during `processPre()`.
   *
   * Register a generic callback that will be executed when `processPre()` is called.
   * Lifetime of the callback must be ensured by the caller.
   *
   * @param[in] callback  the callback that will be executed by `processPre()`.
   *
   * @returns the ID of the scheduled item which can be used cancelation requests.
   */
  [[nodiscard]] ItemIdType registerGenericPre(DelayedSubmissionCallbackType callback);

  /**
   * @brief Register a generic callback to execute during `processPost()`.
   *
   * Register a generic callback that will be executed when `processPost()` is called.
   * Lifetime of the callback must be ensured by the caller.
   *
   * @param[in] callback  the callback that will be executed by `processPre()`.
   *
   * @returns the ID of the scheduled item which can be used cancelation requests.
   */
  [[nodiscard]] ItemIdType registerGenericPost(DelayedSubmissionCallbackType callback);

  /**
   * @brief Cancel a generic callback scheduled for `processPre()` execution.
   *
   * Cancel the execution of a generic callback that has been previously scheduled for
   * execution during `processPre()`. This can be useful if the caller of
   * `registerGenericPre()` has given up and will not anymore be able to guarantee the
   * lifetime of the callback.
   *
   * @param[in] id        the ID of the scheduled item, as returned
   *                      by `registerGenericPre()`.
   */
  void cancelGenericPre(ItemIdType id);

  /**
   * @brief Cancel a generic callback scheduled for `processPost()` execution.
   *
   * Cancel the execution of a generic callback that has been previously scheduled for
   * execution during `processPos()`. This can be useful if the caller of
   * `registerGenericPre()` has given up and will not anymore be able to guarantee the
   * lifetime of the callback.
   *
   * @param[in] id        the ID of the scheduled item, as returned
   *                      by `registerGenericPos()`.
   */
  void cancelGenericPost(ItemIdType id);

  /**
   * @brief Inquire if delayed request submission is enabled.
   *
   * Check whether delayed submission request is enabled, in which case `registerRequest()`
   * may be used to register requests that will be executed during `processPre()`.
   *
   * @returns `true` if a delayed request submission is enabled, `false` otherwise.
   */
  [[nodiscard]] bool isDelayedRequestSubmissionEnabled() const;
};

}  // namespace ucxx
