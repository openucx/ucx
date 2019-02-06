/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucp;

import static org.ucx.jucx.UcxTools.UCS_BIT;
import static org.ucx.jucx.UcxTools.UcsTreadMode;

import java.nio.ByteBuffer;
import java.util.BitSet;

/**
 * Tuning parameters for the UCP worker.
 *
 * <p>The structure defines the parameters that are used for the
 * UCP worker tuning during the UCP worker "creation".</p>
 */
public class UcpWorkerParams {

  @Override
  public String toString() {
    return "UcpWorkerParams{"
      + "fieldMask=" + fieldMask
      + ", threadMode=" + threadMode
      + ", cpuMask=" + cpuMask
      + ", events=" + events
      + ", userData=" + userData.asCharBuffer().toString()
      + ", eventFD=" + eventFD
      + '}';
  }

  /**
   * UCP worker parameters field mask.
   *
   * <p>The enumeration allows specifying which fields in @ref ucp_worker_params_t are
   * present. It is used for the enablement of backward compatibility support.
   */
  public enum UcpWorkerParamField {
    UCP_WORKER_PARAM_FIELD_THREAD_MODE(UCS_BIT(0L)),  // UCP thread mode
    UCP_WORKER_PARAM_FIELD_CPU_MASK(UCS_BIT(1L)),     // Worker's CPU bitmap
    UCP_WORKER_PARAM_FIELD_EVENTS(UCS_BIT(2)),        // Worker's events bitmap
    UCP_WORKER_PARAM_FIELD_USER_DATA(UCS_BIT(3)),     // User data
    UCP_WORKER_PARAM_FIELD_EVENT_FD(UCS_BIT(4));      // External event file descriptor

    private long value;

    UcpWorkerParamField(long value) {
      this.value = value;
    }

    public long getValue() {
      return value;
    }
  }

  public enum UcpWakeupEventTypes {
    UCP_WAKEUP_RMA(UCS_BIT(0L)),      // Remote memory access send completion
    UCP_WAKEUP_AMO(UCS_BIT(1L)),      // Atomic operation send completion
    UCP_WAKEUP_TAG_SEND(UCS_BIT(2L)), // Tag send completion
    UCP_WAKEUP_TAG_RECV(UCS_BIT(3L)), // Tag receive completion
    UCP_WAKEUP_TX(UCS_BIT(10L)),      // This event type will generate an event on completion of any
    // outgoing operation (complete or partial, according to the underlying protocol) for any type
    // of transfer (send, atomic, or RMA).
    UCP_WAKEUP_RX(UCS_BIT(11L)),      // This event type will generate an event on completion
    // of any receive operation (complete or partial, according to the underlying protocol).
    UCP_WAKEUP_EDGE(UCS_BIT(16L));
    /**
     * < Use edge-triggered wakeup. The event
     * file descriptor will be signaled only
     * for new events, rather than existing
     * ones.
     */

    private long value;

    UcpWakeupEventTypes(long value) {
      this.value = value;
    }

    public long getValue() {
      return value;
    }
  }

  /**
   * Mask of valid fields in this structure, using bits from @ref ucp_worker_params_field.
   * Fields not specified in this mask would be ignored.
   * Provides ABI compatibility with respect to adding new fields.
   */
  public long fieldMask;

  /**
   * The parameter threadMode suggests the thread safety mode which worker
   * and the associated resources should be created with. This is an
   * optional parameter. The default value is UCS_THREAD_MODE_SINGLE and
   * it is used when the value of the parameter is not set. When this
   * parameter along with its corresponding bit in the
   * field_mask - UCP_WORKER_PARAM_FIELD_THREAD_MODE is set, the
   * ucp_worker_create attempts to create worker with this thread mode.
   * The thread mode with which worker is created can differ from the
   * suggested mode. The actual thread mode of the worker should be obtained
   * using the query interface ucp_worker_query.
   */
  public UcsTreadMode threadMode;

  /**
   * Mask of which CPUs worker resources should preferably be allocated on.
   * This value is optional.
   * If it's not set (along with its corresponding bit in the field_mask -
   * UCP_WORKER_PARAM_FIELD_CPU_MASK), resources are allocated according to
   * system's default policy.
   */
  public BitSet cpuMask;

  /**
   * Mask of events {@code UcpWakeupEventTypes} which are expected on wakeup.
   * This value is optional.
   * If it's not set (along with its corresponding bit in the field_mask -
   * UCP_WORKER_PARAM_FIELD_EVENTS), all types of events will trigger on
   * wakeup.
   */
  public long events;

  /**
   * User data associated with the current worker.
   * This value is optional.
   * If it's not set (along with its corresponding bit in the field_mask -
   * UCP_WORKER_PARAM_FIELD_USER_DATA), it will default to NULL.
   */
  public ByteBuffer userData;

  /**
   * External event file descriptor.
   * This value is optional.
   * If {@code UcpWorkerParamField.UCP_WORKER_PARAM_FIELD_EVENT_FD} is set in the field_mask, events
   * on the worker will be reported on the provided event file descriptor. In
   * this case, calling @ref ucp_worker_get_efd will result in an error.
   * The provided file descriptor must be capable of aggregating notifications
   * for arbitrary events, for example @c epoll(7) on Linux systems.
   *
   * <p>{@code userData} will be used as the event user-data on systems which
   * support it. For example, on Linux, it will be placed in
   * epoll_data_t::ptr, when returned from epoll_wait(2).</p>
   *
   * <p>Otherwise, events would be reported to the event file descriptor returned
   * from ucp_worker_get_efd().</p>
   */
  public int eventFD;
}
