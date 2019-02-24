/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.ucx.jucx.ucp;

import java.nio.ByteBuffer;
import java.util.BitSet;

import org.ucx.jucx.ucs.UcsConstants;

public class UcpWorkerParams {

    /**
     * Mask of valid fields in this structure.
     * Fields not specified in this mask would be ignored.
     * Provides ABI compatibility with respect to adding new fields.
     */
    private long fieldMask;


    private int threadMode;

    private BitSet cpuMask = new BitSet();

    private long events;

    private ByteBuffer userData;

    private int eventFD;

    /**
     * Suggests the thread safety mode which worker and the associated resources
     * should be created with. The default value is UCS_THREAD_MODE_SINGLE and
     * it is used when the value of the parameter is not set. When this
     * parameter is set, the {@link UcpWorker#UcpWorker(UcpContext)}
     * attempts to create worker with this thread mode.
     * The thread mode with which worker is created can differ from the
     * suggested mode.
     */
    public UcpWorkerParams setThreadMode(UcsConstants.UcsThreadMode threadMode) {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        this.threadMode = threadMode.ordinal();
        return this;
    }

    /**
     * Mask of which CPUs worker resources should preferably be allocated on.
     * This value is optional.
     * If it's not set, resources are allocated according to system's default policy.
     */
    public UcpWorkerParams setCpu(int cpuNum) {
        this.fieldMask |=  UcpConstants.UCP_WORKER_PARAM_FIELD_CPU_MASK;
        this.cpuMask.set(cpuNum);
        return this;
    }

    /**
     * Remote memory access send completion.
     */
    public UcpWorkerParams requestWakeupRMA() {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_EVENTS;
        this.events |= UcpConstants.UCP_WAKEUP_RMA;
        return this;
    }

    /**
     * Atomic operation send completion.
     */
    public UcpWorkerParams requestWakeupAMO() {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_EVENTS;
        this.events |= UcpConstants.UCP_WAKEUP_AMO;
        return this;
    }

    /**
     * Tag send completion.
     */
    public UcpWorkerParams requestWakeupTagSend() {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_EVENTS;
        this.events |= UcpConstants.UCP_WAKEUP_TAG_SEND;
        return this;
    }

    /**
     * Tag receive completion.
     */
    public UcpWorkerParams requestWakeupTagRecv() {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_EVENTS;
        this.events |= UcpConstants.UCP_WAKEUP_TAG_RECV;
        return this;
    }

    /**
     * This event type will generate an event on completion of any
     * outgoing operation (complete or  partial, according to the
     * underlying protocol) for any type of transfer (send, atomic, or RMA).
     */
    public UcpWorkerParams requestWakeupTX() {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_EVENTS;
        this.events |= UcpConstants.UCP_WAKEUP_TX;
        return this;
    }

    /**
     * This event type will generate an event on completion of any receive
     * operation (complete or partial, according to the underlying protocol).
     */
    public UcpWorkerParams requestWakeupRX() {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_EVENTS;
        this.events |= UcpConstants.UCP_WAKEUP_RX;
        return this;
    }

    /**
     * Use edge-triggered wakeup. The event file descriptor will be signaled only
     * for new events, rather than existing ones.
     */
    public UcpWorkerParams requestWakeupEdge() {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_EVENTS;
        this.events |= UcpConstants.UCP_WAKEUP_EDGE;
        return this;
    }

    /**
     * User data associated with the current worker.
     */
    public UcpWorkerParams setUserData(ByteBuffer userData) {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_USER_DATA;
        this.userData = userData;
        return this;
    }

    /**
     * External event file descriptor.
     *
     * <p>Events on the worker will be reported on the provided event file descriptor.
     * The provided file descriptor must be capable of aggregating notifications
     * for arbitrary events, for example @c epoll(7) on Linux systems.
     *
     * <p>{@link userData} will be used as the event user-data on systems which
     * support it. For example, on Linux, it will be placed in
     * epoll_data_t::ptr, when returned from epoll_wait(2).</p>
     *
     * <p>Otherwise, events would be reported to the event file descriptor returned
     * from ucp_worker_get_efd().</p>
     */
    public UcpWorkerParams setEventFD(int eventFD) {
        this.fieldMask |= UcpConstants.UCP_WORKER_PARAM_FIELD_EVENT_FD;
        this.eventFD = eventFD;
        return this;
    }
}
