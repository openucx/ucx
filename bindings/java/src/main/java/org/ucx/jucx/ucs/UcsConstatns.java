package org.ucx.jucx.ucs;

public class UcsConstatns {

    /**
     * Thread sharing mode
     *
     * <p>Specifies thread sharing mode of an object.
     */
    public enum UcsThreadMode {
        UCS_THREAD_MODE_SINGLE,     /**< Only the master thread can access
                                         (i.e. the thread that initialized the context;
                                         multiple threads may exist and never access) */
        UCS_THREAD_MODE_SERIALIZED, /**< Multiple threads can access, but only one at a time */
        UCS_THREAD_MODE_MULTI,      /**< Multiple threads can access concurrently */
        UCS_THREAD_MODE_LAST
    }
}
