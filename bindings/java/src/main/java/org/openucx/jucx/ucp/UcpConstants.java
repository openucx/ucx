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
    static long UCP_WORKER_PARAM_FIELD_CLIENT_ID;

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
    static long UCP_EP_PARAM_FIELD_NAME;

    /**
     * UCP error handling mode.
     */
    static int UCP_ERR_HANDLING_MODE_PEER;

    /**
     * The enumeration list describes the endpoint's parameters flags.
     */
    static long UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
    static long UCP_EP_PARAMS_FLAGS_NO_LOOPBACK;
    static long UCP_EP_PARAMS_FLAGS_SEND_CLIENT_ID;

    /**
     *  The enumeration is used to specify the behavior of UcpEndpoint closeNonBlocking.
     */
    static int UCP_EP_CLOSE_FLAG_FORCE;

    /**
     * UCP memory mapping parameters field mask.
     */
    static long UCP_MEM_MAP_PARAM_FIELD_ADDRESS;
    static long UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    static long UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    static long UCP_MEM_MAP_PARAM_FIELD_PROT;
    static long UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;

    /**
     *  The enumeration list describes the memory mapping flags.
     */
    static long UCP_MEM_MAP_NONBLOCK;
    static long UCP_MEM_MAP_ALLOCATE;
    static long UCP_MEM_MAP_FIXED;

    /**
     * The enumeration list describes the memory mapping protections supported by
     * {@link UcpContext#memoryMap(UcpMemMapParams)}
     */
    public static long UCP_MEM_MAP_PROT_LOCAL_READ;
    public static long UCP_MEM_MAP_PROT_LOCAL_WRITE;
    public static long UCP_MEM_MAP_PROT_REMOTE_READ;
    public static long UCP_MEM_MAP_PROT_REMOTE_WRITE;

    /**
     * The enumeration defines behavior of
     * {@link UcpEndpoint#recvStreamNonBlocking(long, long, long, UcxCallback)}  function.
     */
    public static long UCP_STREAM_RECV_FLAG_WAITALL;

    /**
     * Indicates that the data provided in {@link UcpAmRecvCallback} callback
     * can be held by the user. If {@link org.openucx.jucx.ucs.UcsConstants.STATUS#UCS_INPROGRESS}
     * is returned from the callback, the data parameter will persist and the user has to call
     * {@link UcpWorker#amDataRelease } when data is no longer needed. This flag is
     * mutually exclusive with {@link UcpConstants#UCP_AM_RECV_ATTR_FLAG_RNDV}.
     */
    public static long UCP_AM_RECV_ATTR_FLAG_DATA;

    /**
     * Indicates that the arriving data was sent using rendezvous protocol.
     * In this case dataAddress parameter of the {@link UcpAmRecvCallback#onReceive} points
     * to the internal UCP descriptor, which can be used for obtaining the actual
     * data by calling {@link UcpWorker#recvAmDataNonBlocking} routine. This flag is mutually
     * exclusive with {@link UcpConstants#UCP_AM_RECV_ATTR_FLAG_DATA}.
     */
    public static long UCP_AM_RECV_ATTR_FLAG_RNDV;

    /**
     * Flags dictate the behavior of {@link UcpEndpoint#sendAmNonBlocking} routine.
     */
    public static long UCP_AM_SEND_FLAG_REPLY;
    public static long UCP_AM_SEND_FLAG_EAGER;
    public static long UCP_AM_SEND_FLAG_RNDV;

    /**
     * Flags for a UCP Active Message callback.
     */

    /**
     * Indicates that the entire message will be handled in one callback. With this
     * option, message ordering is not guaranteed (i.e. receive callbacks may be
     * invoked in a different order than messages were sent).
     * If this flag is not set, the data callback may be invoked several times for
     * the same message (if, for example, it was split into several fragments by
     * the transport layer). It is guaranteed that the first data callback for a
     * particular message is invoked for the first fragment. The ordering of first
     * message fragments is guaranteed (i.e. receive callbacks will be called
     * in the order the messages were sent). The order of other fragments is not
     * guaranteed. User header is passed with the first fragment only.
     */
    public static long UCP_AM_FLAG_WHOLE_MSG;

    /**
     * Guarantees that the specified {@link UcpAmRecvCallback#onReceive} callback,
     * will always be called with {@link UcpConstants#UCP_AM_RECV_ATTR_FLAG_DATA} flag set,
     * and {@link UcpAmData#canPersist()} will return true, so the data will be accessible outside
     * the callback, until {@link UcpWorker#amDataRelease} is called.
     */
    public static long UCP_AM_FLAG_PERSISTENT_DATA;

    private static native void loadConstants();
}
