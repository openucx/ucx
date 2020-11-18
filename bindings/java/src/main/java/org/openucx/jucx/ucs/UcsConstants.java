/*
 * Copyright (C) Mellanox Technologies Ltd. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

package org.openucx.jucx.ucs;

import org.openucx.jucx.NativeLibs;

public class UcsConstants {
    static {
        load();
    }

    public static class ThreadMode {
        static {
            load();
        }
        /**
         * Multiple threads can access concurrently
         */
        public static int UCS_THREAD_MODE_MULTI;
    }

    /**
     * Status codes
     */
    public static class STATUS {
        static {
            load();
        }

        /* Operation completed successfully */
        public static int UCS_OK;

        /* Operation is queued and still in progress */
        public static int UCS_INPROGRESS;

        /* Failure codes */
        public static int UCS_ERR_NO_MESSAGE;
        public static int UCS_ERR_NO_RESOURCE;
        public static int UCS_ERR_IO_ERROR;
        public static int UCS_ERR_NO_MEMORY;
        public static int UCS_ERR_INVALID_PARAM;
        public static int UCS_ERR_UNREACHABLE;
        public static int UCS_ERR_INVALID_ADDR;
        public static int UCS_ERR_NOT_IMPLEMENTED;
        public static int UCS_ERR_MESSAGE_TRUNCATED;
        public static int UCS_ERR_NO_PROGRESS;
        public static int UCS_ERR_BUFFER_TOO_SMALL;
        public static int UCS_ERR_NO_ELEM;
        public static int UCS_ERR_SOME_CONNECTS_FAILED;
        public static int UCS_ERR_NO_DEVICE;
        public static int UCS_ERR_BUSY;
        public static int UCS_ERR_CANCELED;
        public static int UCS_ERR_SHMEM_SEGMENT;
        public static int UCS_ERR_ALREADY_EXISTS;
        public static int UCS_ERR_OUT_OF_RANGE;
        public static int UCS_ERR_TIMED_OUT;
        public static int UCS_ERR_EXCEEDS_LIMIT;
        public static int UCS_ERR_UNSUPPORTED;
        public static int UCS_ERR_REJECTED;
        public static int UCS_ERR_NOT_CONNECTED;
        public static int UCS_ERR_CONNECTION_RESET;

        public static int UCS_ERR_FIRST_LINK_FAILURE;
        public static int UCS_ERR_LAST_LINK_FAILURE;
        public static int UCS_ERR_FIRST_ENDPOINT_FAILURE;
        public static int UCS_ERR_ENDPOINT_TIMEOUT;
        public static int UCS_ERR_LAST_ENDPOINT_FAILURE;

        public static int UCS_ERR_LAST;
    }

    private static void load() {
        NativeLibs.load();
        loadConstants();
    }

    private static native void loadConstants();
}
