/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucp;

import org.openucx.jucx.NativeLibs;
import org.openucx.jucx.UcxCallback;

public class UcpConstants {
    static {
        NativeLibs.load();
        loadConstants();
    }

    /**
     * UCP context parameters field mask.
     *
     * <p>The enumeration allows specifying which fields in {@link UcpParams} are
     * present. It is used for the enablement of backward compatibility support.
     */
    static long UCP_PARAM_FIELD_FEATURES;
    static long UCP_PARAM_FIELD_TAG_SENDER_MASK;
    static long UCP_PARAM_FIELD_MT_WORKERS_SHARED;
    static long UCP_PARAM_FIELD_ESTIMATED_NUM_EPS;

    /**
     * UCP configuration features
     *
     * <p>The enumeration list describes the features supported by UCP.
     * An application can request the features using "UCP parameters"
     * during "UCP initialization" process.
     */
    static long UCP_FEATURE_TAG;
    static long UCP_FEATURE_RMA;
    static long UCP_FEATURE_AMO32;
    static long UCP_FEATURE_AMO64;
    static long UCP_FEATURE_WAKEUP;
    static long UCP_FEATURE_STREAM;
    static long UCP_FEATURE_AM;

    /**
     * UCP worker parameters field mask.
     *
     * <p>The enumeration allows specifying which fields in {@link UcpWorker} are
     * present. It is used for the enablement of backward compatibility support.
     */
    static long UCP_WORKER_PARAM_FIELD_THREAD_MODE;
    static long UCP_WORKER_PARAM_FIELD_CPU_MASK;
    static long UCP_WORKER_PARAM_FIELD_EVENTS;
    static long UCP_WORKER_PARAM_FIELD_USER_DATA;
    static long UCP_WORKER_PARAM_FIELD_EVENT_FD;

    /**
     * Mask of events which are expected on wakeup.
     * If it's not set all types of events will trigger on
     * wakeup.
     */
    static long UCP_WAKEUP_RMA;
    static long UCP_WAKEUP_AMO;
    static long UCP_WAKEUP_TAG_SEND;
    static long UCP_WAKEUP_TAG_RECV;
    static long UCP_WAKEUP_TX;
    static long UCP_WAKEUP_RX;
    static long UCP_WAKEUP_EDGE;

    /**
     * UCP listener parameters field mask.
     */
    static long UCP_LISTENER_PARAM_FIELD_SOCK_ADDR;
    static long UCP_LISTENER_PARAM_FIELD_ACCEPT_HANDLER;
    static long UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;

    /**
     * UCP endpoint parameters field mask.
     */
    static long UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
    static long UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
    static long UCP_EP_PARAM_FIELD_ERR_HANDLER;
    static long UCP_EP_PARAM_FIELD_USER_DATA;
    static long UCP_EP_PARAM_FIELD_SOCK_ADDR;
    static long UCP_EP_PARAM_FIELD_FLAGS;
    static long UCP_EP_PARAM_FIELD_CONN_REQUEST;

    /**
     * UCP error handling mode.
     */
    static int UCP_ERR_HANDLING_MODE_PEER;

    /**
     * The enumeration list describes the endpoint's parameters flags.
     */
    static long UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    static long UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;

    /**
     *  The enumeration is used to specify the behavior of UcpEndpoint closeNonBlocking.
     */
    static int UCP_EP_CLOSE_MODE_FORCE;
    static int UCP_EP_CLOSE_MODE_FLUSH;

    /**
     * UCP memory mapping parameters field mask.
     */
    static long UCP_MEM_MAP_PARAM_FIELD_ADDRESS;
    static long UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    static long UCP_MEM_MAP_PARAM_FIELD_FLAGS;

    /**
     *  The enumeration list describes the memory mapping flags.
     */
    static long UCP_MEM_MAP_NONBLOCK;
    static long UCP_MEM_MAP_ALLOCATE;
    static long UCP_MEM_MAP_FIXED;

    /**
     * The enumeration defines behavior of
     * {@link UcpEndpoint#recvStreamNonBlocking(long, long, long, UcxCallback)}  function.
     */
    public static long UCP_STREAM_RECV_FLAG_WAITALL;

    private static native void loadConstants();
}
