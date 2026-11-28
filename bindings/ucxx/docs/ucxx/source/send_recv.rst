Send/Recv Internals
===================

Generally UCX creates connections between endpoints with the following steps:

1. Create a ``Listener`` with defined IP address and port
    a. ``Listener`` defines a callback function to process communications from endpoints
2. Connect an ``Endpoint`` to the ``Listener``
3. ``Endpoint`` communicates with the ``Listener``
4. When finished, close ``Endpoint`` and ``Listener``


Below we go into more detail as we create an echo server in UCX and compare with `Python Sockets <https://docs.python.org/3/library/socket.html#example>`_

Server
------
First, we create the server -- in UCXX, we create a server with ``create_listener`` and build a blocking call to keep the listener alive. The listener invokes a callback function when an incoming connection is accepted. This callback should take in an ``Endpoint`` as an argument for ``send``/``recv``.

For Python sockets, the server is similarly constructed. ``bind`` opens a connection on a given port and ``accept`` is Python Sockets' blocking call for incoming connections. In both UCXX and Sockets, once a connection has been made, both receive data and echo the same data back to the client

+------------------------------------------------------+----------------------------------------------------------+
| UCX                                                  | Python Sockets                                           |
+------------------------------------------------------+----------------------------------------------------------+
| .. code-block:: python                               | .. code-block:: python                                   |
|                                                      |                                                          |
|    async def echo_server(ep):                        |     s = socket.socket(...)                               |
|        obj = await ep.recv_obj()                     |     s.bind((HOST, PORT))                                 |
|        await ep.send_obj(obj)                        |     s.listen(1)                                          |
|                                                      |                                                          |
|    lf = ucxx.create_listener(echo_server, port)      |     while True:                                          |
|                                                      |         conn, addr = s.accept()                          |
|    while not lf.closed():                            |         data = conn.recv(1024)                           |
|        await asyncio.sleep(0.1)                      |         if not data: break                               |
|                                                      |         conn.sendall(data)                               |
|                                                      |         conn.close()                                     |
+------------------------------------------------------+----------------------------------------------------------+

.. note::

    In this example we create servers which listen forever. In production applications developers should also call appropriate closing functions

Client
------

For Sockets, on the client-side we connect to the established host/port combination and send data to the socket. The client-side is a bit more interesting in UCXX: ``create_endpoint``, also uses a host/port combination to establish a connection, and after an ``Endpoint`` is created, ``hello, world`` is passed back and forth between the client an server.

+------------------------------------------------------+----------------------------------------------------------+
| UCX                                                  | Python Sockets                                           |
+------------------------------------------------------+----------------------------------------------------------+
| .. code-block:: python                               | .. code-block:: python                                   |
|                                                      |                                                          |
|    client = await ucxx.create_endpoint(addr, port)   |    s = socket.socket(...)                                |
|                                                      |    s.connect((HOST, PORT))                               |
|    msg = bytearray(b"hello, world")                  |    s.sendall(b'hello, world')                            |
|    await client.send_obj(msg)                        |    echo_msg = s.recv(1024)                               |
|    echo_msg = await client.recv_obj()                |                                                          |
|    await client.close()                              |    s.close()                                             |
|                                                      |                                                          |
+------------------------------------------------------+----------------------------------------------------------+

So what happens with ``create_endpoint`` ? Unlike Sockets, UCX employs a tag-matching strategy where endpoints are created with a unique id and send/receive operations also use unique ids (these are called ``tags``). With standard TCP connections, when a incoming requests is made, a socket is created with a unique 4-tuple: client address, client port, server address, and server port. With this uniqueness, threads and processes alike are now free to communicate with one another. Again, UCX, uses tags for uniqueness so when an incoming request is made, the receiver matches the ``Endpoint`` ID and a unique tag -- for more details on tag-matching please see the `this page <https://www.kernel.org/doc/html/latest/infiniband/tag_matching.html>`_.

``create_endpoint``, will create an ``Endpoint`` with three steps:

#. Generate unique IDs to use as tags
#. Exchange endpoint info such as tags
#. Use the info to create an endpoint

Again, an ``Endpoint`` sends and receives with `unique tags <http://openucx.github.io/ucx/api/v1.8/html/group___u_c_t___t_a_g.html>`_.

.. code-block:: python

    ep = Endpoint(
            endpoint=ucx_ep,
            ctx=self,
            msg_tag_send=peer_info["msg_tag"],
            msg_tag_recv=msg_tag,
            guarantee_msg_order=guarantee_msg_order,
        )

Most users will not care about these details but developers and interested network enthusiasts may. Looking at the DEBUG (``UCXPY_LOG_LEVEL=DEBUG``) output of the client can help clarify what UCXX/UCX is doing under the hood::


    # client = await ucxx.create_endpoint(addr, port)
    [1757536870.872005] [dgx13:1377244] UCXPY  DEBUG create_endpoint() client: 0x7f5a161a8080, error handling: True, msg-tag-send: 0xf3c3e246293e6852, msg-tag-recv: 0xdf227087928e03f6, ctrl-tag-send: 0x9a69cc1d6f54a0c6, ctrl-tag-recv: 0xe1cc0be4bc1d722a

    # await client.send_obj(msg)
    [1757536870.872140] [dgx13:1377244] UCXPY  DEBUG [Send #000] ep: 0x7f5a161a8080, tag: 0xf3c3e246293e6852, nbytes: 8, type: <class 'array.array'>
    [1757536870.872498] [dgx13:1377244] UCXPY  DEBUG [Send #001] ep: 0x7f5a161a8080, tag: 0xf3c3e246293e6852, nbytes: 12, type: <class 'bytearray'>

    # echo_msg = await client.recv_obj()
    [1757536870.872404] [dgx13:1377244] UCXPY  DEBUG [Recv #000] ep: 0x7f5a161a8080, tag: 0xdf227087928e03f6, nbytes: 8, type: <class 'array.array'>
    [1757536870.872600] [dgx13:1377244] UCXPY  DEBUG [Recv #001] ep: 0x7f5a161a8080, tag: 0xdf227087928e03f6, nbytes: 12, type: <class 'bytearray'>


We can see from the above that when the ``Endpoint`` is created, 4 tags are generated: ``msg-tag-send``, ``msg-tag-recv``, ``ctrl-tag-send``, and ``ctrl-tag-recv``. This data is transmitted to the server via a `stream <https://openucx.github.io/ucx/api/latest/html/group___u_c_p___c_o_m_m.html#gae9fe6efe6b05e4e78f58bee68c68b252>`_ communication in an `exchange peer info <https://github.com/rapidsai/ucxx/blob/main/python/ucxx/ucxx/_lib_async/exchange_peer_info.py>`_ convenience function.

Next, the client sends data on the ``msg-tag-send`` tag. Two messages are sent, the size of the data ``8 bytes`` and data itself. The server receives the data and immediately echos the data back. The client then receives two messages the size of the data and the data itself.
